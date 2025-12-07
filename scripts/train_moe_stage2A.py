#!/usr/bin/env python3
"""
Stage 2A: Task-Grouped MoE Training for Expert Specialization

Key differences vs Stage 2:
  * Uses Stage 2A dataset config (math/code/summarization/translation)
  * Loads from Stage 1.5 checkpoint `outputs/llama-3b-moe-mixed-sft/checkpoint-4500`
  * Optional freezing of non-MoE LoRA adapters (defaults to frozen)
  * Handles per-task train/validation splits defined in YAML
"""

import os
import sys
import yaml
import logging
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configuration_llama_moe import LlamaMoEConfig
from src.modeling_llama_moe import LlamaMoEForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

AutoConfig.register("llama_moe", LlamaMoEConfig)
AutoModelForCausalLM.register(LlamaMoEConfig, LlamaMoEForCausalLM)
AutoTokenizer.register(LlamaMoEConfig, fast_tokenizer_class=PreTrainedTokenizerFast)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}"""

DATASET_OVERRIDES: Dict[str, Dict[str, str]] = {
    "gsm8k": {"config": "main", "format": "gsm8k", "task_type": "math"},
    "iamtarun/python_code_instructions_18k_alpaca": {
        "format": "alpaca",
        "task_type": "code",
    },
    "abisee/cnn_dailymail": {
        "format": "cnn_dailymail",
        "task_type": "summarization",
    },
    "wmt/wmt19": {
        "format": "translation",
        "task_type": "translation",
        "config": "zh-en",
        "source_lang": "zh",
        "target_lang": "en",
    },
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="outputs/llama-3b-moe-mixed-sft/checkpoint-4500"
    )
    use_flash_attn: bool = field(default=True)
    use_qlora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    aux_loss_alpha: float = field(default=0.01)
    freeze_non_moe_lora: bool = field(
        default=True, metadata={"help": "Freeze attention/non-expert LoRA adapters"}
    )


@dataclass
class DataArguments:
    data_config: str = field(default="configs/data_task_stage2A.yaml")
    max_seq_length: int = field(default=1024)
    steps_per_task: int = field(default=300, metadata={"help": "Steps per task before switching (for task cycling)"})
    eval_loss_only: bool = field(
        default=True,
        metadata={"help": "Only compute eval loss (skip predictions/metrics) to save memory. Set False to compute accuracy metrics."}
    )


def load_data_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def slugify_task(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def format_alpaca(example: Dict) -> Dict:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "") or example.get("response", "")

    if input_text:
        instruction = f"{instruction}\n\nInput: {input_text}"

    return {"instruction": instruction, "response": output}


def format_gsm8k(example: Dict) -> Dict:
    return {"instruction": example.get("question", ""), "response": example.get("answer", "")}


def format_cnn_dailymail(example: Dict) -> Dict:
    article = example.get("article", "")
    highlights = example.get("highlights", "")
    instruction = "Summarize the following article:\n\n" + article
    return {"instruction": instruction, "response": highlights}


def format_translation(example: Dict, source_lang: str, target_lang: str) -> Dict:
    translation = example.get("translation", {})
    src = translation.get(source_lang, "")
    tgt = translation.get(target_lang, "")
    instruction = f"Translate the following {source_lang.upper()} text to {target_lang.upper()}:\n\n{src}"
    return {"instruction": instruction, "response": tgt}


FORMATTER_REGISTRY: Dict[str, Callable] = {
    "alpaca": format_alpaca,
    "gsm8k": format_gsm8k,
    "cnn_dailymail": format_cnn_dailymail,
}


def apply_dataset_defaults(dataset_cfg: Dict) -> Dict:
    merged = {}
    overrides = DATASET_OVERRIDES.get(dataset_cfg["name"], {})
    merged.update(overrides)
    merged.update(dataset_cfg)
    if "task_type" not in merged:
        merged["task_type"] = slugify_task(dataset_cfg["name"])
    if "format" not in merged:
        merged["format"] = "alpaca"
    return merged


def resolve_formatter(dataset_cfg: Dict) -> Callable:
    fmt = dataset_cfg.get("format", "alpaca")
    if fmt == "translation":
        source_lang = dataset_cfg.get("source_lang", "zh")
        target_lang = dataset_cfg.get("target_lang", "en")
        return partial(format_translation, source_lang=source_lang, target_lang=target_lang)
    if fmt not in FORMATTER_REGISTRY:
        logger.warning(f"No formatter registered for '{fmt}', defaulting to Alpaca-style.")
    return FORMATTER_REGISTRY.get(fmt, format_alpaca)


def build_split_string(split: str, samples: Optional[int]) -> str:
    if samples is None or samples <= 0:
        return split
    return f"{split}[:{samples}]"


def load_split_dataset(
    dataset_cfg: Dict,
    split_key: str,
    sample_key: str,
    formatter: Callable,
) -> Optional[Dataset]:
    split = dataset_cfg.get(split_key)
    if not split:
        return None

    samples = dataset_cfg.get(sample_key)
    name = dataset_cfg["name"]
    config_name = dataset_cfg.get("config")
    use_streaming = dataset_cfg.get("streaming", False) and samples is not None

    def _take_stream_samples(raw_dataset):
        limited = list(islice(raw_dataset, samples))
        if not limited:
            raise ValueError(f"Streaming yielded 0 samples for {name}:{split}")
        return Dataset.from_list(limited)

    try:
        if use_streaming:
            logger.info(
                f"    ↪ Streaming first {samples} samples from {name}:{split}"
            )
            if config_name:
                ds_iter = load_dataset(
                    name, config_name, split=split, streaming=True
                )
            else:
                ds_iter = load_dataset(name, split=split, streaming=True)
            ds = _take_stream_samples(ds_iter)
        else:
            split_str = build_split_string(split, samples)
            if config_name:
                ds = load_dataset(name, config_name, split=split_str)
            else:
                ds = load_dataset(name, split=split_str)
    except Exception as exc:
        logger.error(f"Failed to load {name} ({split_str}): {exc}")
        raise

    ds = ds.map(
        formatter,
        remove_columns=ds.column_names,
        desc=f"Formatting {name}:{split}",
    )
    ds = ds.filter(lambda x: len(x["instruction"]) > 0 and len(x["response"]) > 0)
    task_type = dataset_cfg["task_type"]
    ds = ds.map(lambda x: {**x, "task_type": task_type})
    return ds


def load_stage2a_tasks(config: Dict) -> List[Dict]:
    tasks = []
    datasets_cfg = config.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("No datasets listed in Stage 2A config.")

    processing_cfg = config.get("processing", {})
    seed = processing_cfg.get("seed", 42)

    logger.info("=" * 80)
    logger.info("Loading Stage 2A task-grouped datasets")
    logger.info("=" * 80)

    for dataset_cfg in datasets_cfg:
        if not dataset_cfg.get("enabled", False):
            continue

        merged_cfg = apply_dataset_defaults(dataset_cfg)
        formatter = resolve_formatter(merged_cfg)

        train_split = merged_cfg.get("train_split")
        vali_split = merged_cfg.get("vali_split")
        train_samples = merged_cfg.get("train_samples")
        validation_samples = merged_cfg.get("validation_samples")

        # Check if train and validation come from the same split
        # If so, we need to ensure no overlap
        if train_split and vali_split and train_split == vali_split:
            logger.info(
                f"  ⚠️  {merged_cfg['task_type']}: train and validation both use '{train_split}' split"
            )
            logger.info(
                f"     Loading full split and splitting with no overlap (seed={seed})"
            )

            # Load enough samples for both train and validation
            total_samples = (train_samples or 0) + (validation_samples or 0)
            if total_samples == 0:
                raise ValueError(
                    f"Dataset {merged_cfg['name']} needs train_samples + validation_samples > 0"
                )

            # Load the full split (enough for both train and validation)
            full_ds = load_split_dataset(merged_cfg, "train_split", total_samples, formatter)

            if full_ds is None or len(full_ds) < total_samples:
                raise ValueError(
                    f"Dataset {merged_cfg['name']} doesn't have enough samples. "
                    f"Requested {total_samples}, got {len(full_ds) if full_ds else 0}"
                )

            # Shuffle with seed for reproducibility, then split
            full_ds = full_ds.shuffle(seed=seed)
            validation_samples_actual = validation_samples or 0

            # Split: validation gets first N samples, train gets the rest
            if validation_samples_actual > 0:
                eval_ds = full_ds.select(range(validation_samples_actual))
                train_ds = full_ds.select(range(validation_samples_actual, len(full_ds)))
            else:
                eval_ds = None
                train_ds = full_ds

            # Limit train to requested size if specified
            if train_samples and len(train_ds) > train_samples:
                train_ds = train_ds.select(range(train_samples))

            logger.info(
                f"     Split: {len(train_ds)} train, {len(eval_ds) if eval_ds else 0} validation (no overlap)"
            )
        else:
            # Different splits or no validation - load normally
            train_ds = load_split_dataset(merged_cfg, "train_split", "train_samples", formatter)
            if train_ds is None or len(train_ds) == 0:
                raise ValueError(f"Dataset {merged_cfg['name']} returned no training samples.")

            eval_ds = load_split_dataset(merged_cfg, "vali_split", "validation_samples", formatter)

        tasks.append(
            {
                "task_type": merged_cfg["task_type"],
                "name": merged_cfg["name"],
                "train_dataset": train_ds,
                "eval_dataset": eval_ds,
                "steps_per_cycle": merged_cfg.get("steps_per_cycle", 300),
                "description": merged_cfg.get("description", merged_cfg["task_type"]),
            }
        )

        logger.info(
            f"  • {merged_cfg['task_type']:>12}: "
            f"{len(train_ds)} train samples"
            + (f", {len(eval_ds)} eval samples" if eval_ds else ", no eval split")
        )

    logger.info("=" * 80 + "\n")
    return tasks


# Task-specific response prompts to help experts learn output format
TASK_RESPONSE_PROMPTS = {
    "math": "Answer: ",
    "summarization": "TL;DR: ",
    "code": "",  # Code tasks typically don't need a prefix
    "translation": "",  # Translation tasks typically don't need a prefix
}


def tokenize_function(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_length: int,
    prompt_template: str,
) -> Dict[str, List[int]]:
    instructions = examples["instruction"]
    responses = examples["response"]
    task_types = examples.get("task_type", [None] * len(instructions))

    input_ids_list = []
    attention_masks = []
    labels_list = []

    for instruction, response, task_type in zip(instructions, responses, task_types):
        # Task label at the beginning (for router/expert selection)
        task_prefix = f"<|task_{task_type}|> " if task_type else ""
        
        # Task-specific prompt at the end of instruction (for output format learning)
        # This will be part of instruction tokens (masked), teaching the model the format
        task_suffix = TASK_RESPONSE_PROMPTS.get(task_type, "") if task_type else ""
        
        # Format instruction with task prefix
        full_instruction = f"{task_prefix}{instruction}"
        
        # Format prompt template with task suffix after "### Response:\n"
        # The suffix is part of the instruction context (visible but masked in labels)
        prompt_with_suffix = prompt_template.format(
            instruction=full_instruction, 
            response=task_suffix
        )

        instruction_tokens = tokenizer(
            prompt_with_suffix,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        # Response tokens (without the suffix, as it's already in instruction)
        response_tokens = tokenizer(
            response + tokenizer.eos_token,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        max_instruction_length = int(max_length * 0.75)
        if len(instruction_tokens) > max_instruction_length:
            instruction_tokens = instruction_tokens[:max_instruction_length]

        remaining = max_length - len(instruction_tokens)
        if len(response_tokens) > remaining:
            response_tokens = response_tokens[:remaining]

        input_ids = instruction_tokens + response_tokens
        labels = [-100] * len(instruction_tokens) + response_tokens[:]

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
        attention_mask = [1] * (len(instruction_tokens) + len(response_tokens)) + [0] * pad_len

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_masks,
        "labels": labels_list,
    }


class TaskCyclingTrainer(Trainer):
    def __init__(self, task_datasets: Dict[str, Dataset], steps_per_task: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_datasets = task_datasets
        self.steps_per_task = steps_per_task
        self.task_order = list(task_datasets.keys())
        self.current_task_idx = 0
        self.current_task = self.task_order[0]
        self.steps_in_current_task = 0

        logger.info("\n🔄 Task Cycling Configuration:")
        logger.info(f"  Task order: {' → '.join(self.task_order)}")
        logger.info(f"  Steps per task: {steps_per_task}\n")

    def get_train_dataloader(self) -> DataLoader:
        current_dataset = self.task_datasets[self.current_task]
        from torch.utils.data import RandomSampler

        sampler = RandomSampler(current_dataset)
        return DataLoader(
            current_dataset,
            sampler=sampler,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Override eval dataloader to use minimal memory settings."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # Use fewer workers and no pin_memory for eval to reduce memory usage
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=min(2, self.args.dataloader_num_workers),  # Max 2 workers for eval
            pin_memory=False,  # Disable pin_memory for eval to save memory
            shuffle=False,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        self.steps_in_current_task += 1
        if self.steps_in_current_task >= self.steps_per_task:
            self.current_task_idx = (self.current_task_idx + 1) % len(self.task_order)
            self.current_task = self.task_order[self.current_task_idx]
            self.steps_in_current_task = 0
            logger.info("\n" + "=" * 60)
            logger.info(f"🔄 Switching to task: {self.current_task.upper()}")
            logger.info("=" * 60 + "\n")
            self._train_dataloader = None
        return loss

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.state.global_step > 0 and len(self.state.log_history) > 0:
            self.state.log_history[-1]["current_task"] = self.current_task
        return super()._maybe_log_save_evaluate(*args, **kwargs)


class TaskLoggingCallback(TrainerCallback):
    def __init__(self, trainer: TaskCyclingTrainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if hasattr(self.trainer, "current_task"):
            logs["task"] = self.trainer.current_task
            logs["task_step"] = self.trainer.steps_in_current_task
        if "loss" in logs:
            logs["train_perplexity"] = float(np.exp(logs["loss"]))
        if "eval_loss" in logs:
            logs["eval_perplexity"] = float(np.exp(logs["eval_loss"]))


class ExpertUtilizationCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or model is None:
            return
        try:
            expert_usage = []
            for _, module in model.named_modules():
                if hasattr(module, "gate") and hasattr(module.gate, "expert_counts"):
                    expert_usage.append(module.gate.expert_counts)

            if not expert_usage:
                return

            avg_usage = np.mean(expert_usage, axis=0)
            if avg_usage.sum() == 0:
                return
            usage_probs = avg_usage / avg_usage.sum()
            eps = 1e-10
            entropy = -np.sum(usage_probs * np.log(usage_probs + eps))
            max_entropy = np.log(len(usage_probs))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            mean_usage = np.mean(avg_usage)
            std_usage = np.std(avg_usage)
            cv = std_usage / mean_usage if mean_usage > 0 else 0

            logs["expert_entropy"] = float(normalized_entropy)
            logs["expert_balance_cv"] = float(cv)
            logs["expert_max_usage"] = float(np.max(usage_probs))
            logs["expert_min_usage"] = float(np.min(usage_probs))

            for idx, usage in enumerate(usage_probs[:8]):
                logs[f"expert_{idx}_usage"] = float(usage)
        except Exception as exc:
            logger.debug(f"Expert stats collection failed: {exc}")


def freeze_non_moe_adapters(model):
    logger.info("\n🔒 Freezing non-MoE (attention) LoRA adapters...")
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        if any(attn in name for attn in attention_modules) and "lora" in name.lower():
            param.requires_grad = False
            frozen_params += param.numel()
        elif param.requires_grad:
            trainable_params += param.numel()

    total = frozen_params + trainable_params
    logger.info(f"  Frozen parameters: {frozen_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    if total > 0:
        logger.info(f"  Trainable %: {100 * trainable_params / total:.2f}%\n")
    return model


def update_model_aux_loss_alpha(model, aux_loss_alpha: float):
    logger.info(f"🔧 Setting aux_loss_alpha = {aux_loss_alpha}")
    for name, module in model.named_modules():
        if hasattr(module, "gate") and hasattr(module.gate, "alpha"):
            module.gate.alpha = aux_loss_alpha
            logger.info(f"  Updated {name}.gate.alpha")
    logger.info("")


def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    predictions, labels = eval_preds
    predictions = predictions.flatten()
    labels = labels.flatten()
    mask = labels != -100
    if mask.sum() == 0:
        return {"accuracy": 0.0}
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    accuracy = float(correct) / float(total)
    return {"accuracy": accuracy, "correct_tokens": int(correct), "total_tokens": int(total)}


# Removed apply_training_overrides and related helper functions
# All training configs now come from command line (shell script)
# YAML config is only used for data/processing configuration


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank <= 0 else logging.WARN,
    )

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2A: Task-Grouped MoE Training for Expert Specialization")
    logger.info("=" * 80 + "\n")

    data_config = load_data_config(data_args.data_config)
    processing_cfg = data_config.get("processing", {})

    if "max_seq_length" in processing_cfg:
        data_args.max_seq_length = processing_cfg["max_seq_length"]

    # Training configs come from command line (shell script), not YAML
    # YAML is only used for data/processing configuration
    tasks = load_stage2a_tasks(data_config)
    steps_per_task = data_args.steps_per_task

    model_kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16}
    if model_args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if model_args.use_qlora:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    logger.info(f"Loading Stage 1.5 checkpoint from {model_args.model_name_or_path}")
    model = LlamaMoEForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    
    # Ensure config has correct model_type (fix if loaded from checkpoint with wrong type)
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        if model.config.model_type != "llama_moe":
            logger.warning(f"Fixing model_type in loaded config: {model.config.model_type} -> llama_moe")
            model.config.model_type = "llama_moe"

    if model_args.use_qlora:
        from peft import PeftModel

        logger.info("📦 Loading LoRA adapters...")
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            is_trainable=True,
        )
        model.print_trainable_parameters()

        if model_args.freeze_non_moe_lora:
            model = freeze_non_moe_adapters(model)

    update_model_aux_loss_alpha(model, model_args.aux_loss_alpha)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    task_tokens = sorted({f"<|task_{task['task_type']}>"
                          for task in tasks})
    if task_tokens:
        logger.info("\n🏷️  Adding task-specific special tokens...")
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": task_tokens})
        logger.info(f"  Added {num_added} task tokens: {', '.join(task_tokens)}")
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"  Resized embeddings to {len(tokenizer)} tokens\n")

    prompt_template = processing_cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    tokenize_fn = partial(
        tokenize_function,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        prompt_template=prompt_template,
    )

    logger.info("Tokenizing task datasets...")
    tokenized_task_datasets: Dict[str, Dataset] = {}
    eval_datasets: List[Dataset] = []

    for task in tasks:
        task_type = task["task_type"]
        train_ds = task["train_dataset"].map(
            tokenize_fn,
            batched=True,
            remove_columns=task["train_dataset"].column_names,
            num_proc=4,
            desc=f"Tokenizing train split ({task_type})",
        )
        tokenized_task_datasets[task_type] = train_ds
        logger.info(f"  ✅ {task_type}: {len(train_ds)} tokenized samples")

        if task["eval_dataset"] is not None:
            eval_ds = task["eval_dataset"].map(
                tokenize_fn,
                batched=True,
                remove_columns=task["eval_dataset"].column_names,
                num_proc=2,
                desc=f"Tokenizing eval split ({task_type})",
            )
            eval_datasets.append(eval_ds)

    eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
    if eval_dataset:
        # Limit eval dataset size to avoid OOM during evaluation
        # Evaluate on a subset if dataset is too large
        # Reduced to 50 samples to avoid OOM from prediction accumulation
        max_eval_samples = 200  # Limit to 50 samples total across all tasks
        if len(eval_dataset) > max_eval_samples:
            logger.warning(
                f"Eval dataset has {len(eval_dataset)} samples, limiting to {max_eval_samples} to avoid OOM"
            )
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"\nEval dataset size: {len(eval_dataset)} samples")
    else:
        logger.info("\nNo evaluation split available; skipping evaluation.")

    # Use prediction_loss_only=True to avoid accumulating full predictions in memory
    # Computing metrics (accuracy) requires accumulating ALL predictions (logits) in memory,
    # which can be 200 samples × 1024 tokens × vocab_size = massive memory usage
    # Only computing loss avoids this accumulation entirely
    use_metrics = not data_args.eval_loss_only
    if eval_dataset:
        if use_metrics:
            logger.warning(
                "⚠️  Computing full metrics (accuracy) - this requires accumulating predictions in memory."
                " If you get OOM, use --eval_loss_only True to only compute loss."
            )
        else:
            logger.info("  Using eval_loss_only=True (only compute loss, skip predictions to save memory)")

    trainer = TaskCyclingTrainer(
        task_datasets=tokenized_task_datasets,
        steps_per_task=steps_per_task,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=list(tokenized_task_datasets.values())[0],
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if (eval_dataset and use_metrics) else None,
    )
    
    # Override prediction_loss_only to skip predictions if not using metrics
    if eval_dataset and not use_metrics:
        trainer.args.prediction_loss_only = True

    trainer.add_callback(TaskLoggingCallback(trainer))
    trainer.add_callback(ExpertUtilizationCallback())

    logger.info("\n🚀 Starting Stage 2A training...")
    logger.info(f"  Task order: {' → '.join(tokenized_task_datasets.keys())}")
    logger.info(f"  Steps per task: {steps_per_task}")
    if training_args.max_steps and training_args.max_steps > 0:
        logger.info(f"  Max steps: {training_args.max_steps}")
    logger.info("")

    resume_checkpoint = training_args.resume_from_checkpoint
    if resume_checkpoint:
        logger.info(f"Resume: {resume_checkpoint} | LR={training_args.learning_rate} | Warmup={training_args.warmup_ratio}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info("\n💾 Saving Stage 2A checkpoint...")
    
    # Ensure config has correct model_type before saving
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        if model.config.model_type != "llama_moe":
            logger.warning(f"Fixing model_type in config: {model.config.model_type} -> llama_moe")
            model.config.model_type = "llama_moe"
    
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Double-check that saved config has correct model_type
    import json
    import os
    config_path = os.path.join(training_args.output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        if config_dict.get("model_type") != "llama_moe":
            logger.warning(f"Fixing model_type in saved config.json: {config_dict.get('model_type')} -> llama_moe")
            config_dict["model_type"] = "llama_moe"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info("✅ Fixed config.json model_type")

    logger.info("\n✅ Stage 2A training complete.")


if __name__ == "__main__":
    train()