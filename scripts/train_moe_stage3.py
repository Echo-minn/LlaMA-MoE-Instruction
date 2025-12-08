#!/usr/bin/env python3
"""Stage 3: update all adapters and router weights for overall performance"""

import os
import sys
import yaml
import logging
import json
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configuration_llama_moe import LlamaMoEConfig
from src.modeling_llama_moe import LlamaMoEForCausalLM

AutoConfig.register("llama_moe", LlamaMoEConfig)
AutoModelForCausalLM.register(LlamaMoEConfig, LlamaMoEForCausalLM)
AutoTokenizer.register(LlamaMoEConfig, fast_tokenizer_class=PreTrainedTokenizerFast)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}"""

TASK_PROMPT_TEMPLATES = {
    "math": """### Question:
{instruction}

### Answer:
{response}""",
    "code": """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}

### Output:
{response}""",
    "summarization": """### Article:
{instruction}

### Highlights Summary:
{response}""",
    "translation": """### en:
{instruction}

### zh:
{response}""",
}

DATASET_OVERRIDES: Dict[str, Dict[str, str]] = {
    "gsm8k": {"config": "main", "format": "gsm8k", "task_type": "math"},
    "iamtarun/python_code_instructions_18k_alpaca": {"format": "alpaca", "task_type": "code"},
    "abisee/cnn_dailymail": {"format": "cnn_dailymail", "task_type": "summarization"},
    "wmt/wmt19": {"format": "translation", "task_type": "translation", "config": "zh-en", "source_lang": "en", "target_lang": "zh"},
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="outputs/llama-3b-moe-stage1/checkpoint-1800")
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Optional path to load PEFT adapters from"})
    use_flash_attn: bool = field(default=True)
    use_qlora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    aux_loss_alpha: float = field(default=0.05)
    aux_loss_initial: Optional[float] = field(default=None)
    aux_loss_warmup_steps: int = field(default=0)
    aux_loss_decay_steps: Optional[int] = field(default=None)
    freeze_non_moe_lora: bool = field(default=False, metadata={"help": "For Stage 3, default False to train both MoE and attention adapters"})
    pretraining_tp: int = field(default=1, metadata={"help": "Tensor parallelism degree (1=disabled)"})


@dataclass
class DataArguments:
    data_config: str = field(default="configs/data_task_stage2A.yaml")
    max_seq_length: int = field(default=1024)
    steps_per_task: int = field(default=300)
    intra_group_shuffle: bool = field(default=True, metadata={"help": "Shuffle samples within each task group"})
    shuffle_tasks: bool = field(
        default=True,
        metadata={"help": "If True, shuffle all tasks together instead of cycling. Default True for Stage 3."}
    )
    eval_loss_only: bool = field(default=True)


def load_data_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def slugify_task(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")


def format_alpaca(example: Dict, task_type: str, task_templates: Dict[str, str]) -> Dict:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "") or example.get("response", "")
    if input_text:
        instruction = f"{instruction}\n\nInput: {input_text}"
    template = task_templates.get(task_type, DEFAULT_PROMPT_TEMPLATE)
    return {"text": template.format(instruction=instruction, response=""), "response": output}


def format_gsm8k(example: Dict, task_type: str, task_templates: Dict[str, str]) -> Dict:
    question = example.get("question", "")
    answer = example.get("answer", "")
    template = task_templates.get(task_type, DEFAULT_PROMPT_TEMPLATE)
    return {"text": template.format(instruction=question, response=""), "response": answer}


def format_cnn_dailymail(example: Dict, task_type: str, task_templates: Dict[str, str]) -> Dict:
    article = example.get("article", "")
    highlights = example.get("highlights", "")
    template = task_templates.get(task_type, DEFAULT_PROMPT_TEMPLATE)
    return {"text": template.format(instruction=article, response=""), "response": highlights}


def format_translation(example: Dict, source_lang: str, target_lang: str, task_type: str, task_templates: Dict[str, str]) -> Dict:
    translation = example.get("translation", {})
    src = translation.get(source_lang, "")
    tgt = translation.get(target_lang, "")
    template = task_templates.get(task_type, DEFAULT_PROMPT_TEMPLATE)
    return {"text": template.format(instruction=src, response=""), "response": tgt}


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


def resolve_formatter(dataset_cfg: Dict, task_templates: Dict[str, str]) -> Callable:
    fmt = dataset_cfg.get("format", "alpaca")
    task_type = dataset_cfg.get("task_type", "alpaca")
    if fmt == "translation":
        source_lang = dataset_cfg.get("source_lang", "en")
        target_lang = dataset_cfg.get("target_lang", "zh")
        return partial(format_translation, source_lang=source_lang, target_lang=target_lang, task_type=task_type, task_templates=task_templates)
    formatter = FORMATTER_REGISTRY.get(fmt, format_alpaca)
    if fmt not in FORMATTER_REGISTRY:
        logger.warning(f"No formatter registered for '{fmt}', defaulting to Alpaca-style.")
    return partial(formatter, task_type=task_type, task_templates=task_templates)


def build_split_string(split: str, samples: Optional[int]) -> str:
    if samples is None or samples <= 0:
        return split
    return f"{split}[:{samples}]"


def load_split_dataset(dataset_cfg: Dict, split_key: str, sample_key: str, formatter: Callable, offset: Optional[int] = None) -> Optional[Dataset]:
    """
    Load a dataset split with optional offset.
    
    Args:
        dataset_cfg: Dataset configuration dict
        split_key: Key for split name (e.g., "train_split")
        sample_key: Key for sample count (e.g., "train_samples") or direct int value
        formatter: Function to format examples
        offset: Optional offset to start loading from (for stage 3: use samples [offset, offset+samples])
    """
    split = dataset_cfg.get(split_key)
    if not split:
        return None
    # Support both string key and direct int value
    if isinstance(sample_key, int):
        samples = sample_key
    else:
        samples = dataset_cfg.get(sample_key)
    name = dataset_cfg["name"]
    config_name = dataset_cfg.get("config")
    use_streaming = dataset_cfg.get("streaming", False) and samples is not None

    def _take_stream_samples(raw_dataset, num_samples, start_offset=0):
        # Skip first start_offset samples, then take num_samples
        if start_offset > 0:
            # Skip offset samples
            _ = list(islice(raw_dataset, start_offset))
        limited = list(islice(raw_dataset, num_samples))
        if not limited:
            raise ValueError(f"Streaming yielded 0 samples for {name}:{split} (offset={start_offset}, samples={num_samples})")
        return Dataset.from_list(limited)

    split_str = split  # Default for error messages
    try:
        if use_streaming:
            if offset is not None and offset > 0:
                logger.info(f"    ↪ Streaming samples [{offset}, {offset+samples}] from {name}:{split}")
            else:
                logger.info(f"    ↪ Streaming first {samples} samples from {name}:{split}")
            if config_name:
                ds_iter = load_dataset(name, config_name, split=split, streaming=True)
            else:
                ds_iter = load_dataset(name, split=split, streaming=True)
            ds = _take_stream_samples(ds_iter, samples, offset or 0)
        else:
            # For non-streaming, load more samples if offset is specified
            if offset is not None and offset > 0:
                # Load samples from offset to offset+samples
                total_needed = offset + samples
                split_str = build_split_string(split, total_needed)
                logger.info(f"    ↪ Loading samples [{offset}, {offset+samples}] from {name}:{split}")
            else:
                split_str = build_split_string(split, samples)
            if config_name:
                ds = load_dataset(name, config_name, split=split_str)
            else:
                ds = load_dataset(name, split=split_str)
            
            # Apply offset if specified
            if offset is not None and offset > 0:
                if len(ds) < offset + samples:
                    logger.warning(f"    ⚠️  Dataset {name} has only {len(ds)} samples, but need {offset+samples}. Using samples [0, {samples}] instead.")
                    ds = ds.select(range(min(samples, len(ds))))
                else:
                    ds = ds.select(range(offset, offset + samples))
    except Exception as exc:
        logger.error(f"Failed to load {name} ({split_str}): {exc}")
        raise

    ds = ds.map(formatter, remove_columns=ds.column_names, desc=f"Formatting {name}:{split}")
    ds = ds.filter(lambda x: len(x.get("text", "")) > 0 and len(x.get("response", "")) > 0)
    task_type = dataset_cfg["task_type"]
    ds = ds.map(lambda x: {**x, "task_type": task_type})
    return ds


def load_stage2a_tasks(config: Dict, task_templates: Dict[str, str], intra_group_shuffle: bool = True) -> List[Dict]:
    """
    Load tasks for Stage 3 training.
    For training data, uses samples [train_samples, 2*train_samples] if available,
    otherwise falls back to [0, train_samples] (same as Stage 1).
    """
    tasks = []
    datasets_cfg = config.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("No datasets listed in Stage 3 config.")

    processing_cfg = config.get("processing", {})
    seed = processing_cfg.get("seed", 42)

    logger.info("=" * 80)
    logger.info("Loading Stage 3 task-grouped datasets")
    logger.info("Stage 3 uses samples [train_samples, 2*train_samples] when available")
    logger.info("=" * 80)

    for dataset_cfg in datasets_cfg:
        if not dataset_cfg.get("enabled", False):
            continue

        merged_cfg = apply_dataset_defaults(dataset_cfg)
        formatter = resolve_formatter(merged_cfg, task_templates)
        train_split = merged_cfg.get("train_split")
        vali_split = merged_cfg.get("vali_split")
        train_samples = merged_cfg.get("train_samples")
        validation_samples = merged_cfg.get("validation_samples")

        if train_split and vali_split and train_split == vali_split:
            logger.info(f"  ⚠️  {merged_cfg['task_type']}: train and validation both use '{train_split}' split")
            logger.info(f"     Loading full split and splitting with no overlap (seed={seed})")
            total_samples = (train_samples or 0) + (validation_samples or 0)
            if total_samples == 0:
                raise ValueError(f"Dataset {merged_cfg['name']} needs train_samples + validation_samples > 0")
            
            # For Stage 3, try to use samples [train_samples, 2*train_samples] for training
            # First check if we have enough samples by trying to load 2*train_samples
            train_offset = None
            try:
                # Try to load 2*train_samples to check availability
                test_ds = load_split_dataset(merged_cfg, "train_split", 2 * train_samples, formatter, offset=None)
                if test_ds is not None and len(test_ds) >= 2 * train_samples:
                    logger.info(f"     ✓ Dataset has enough samples, using [{train_samples}, {2*train_samples}] for training")
                    train_offset = train_samples
                else:
                    logger.info(f"     ⚠️  Dataset has only {len(test_ds) if test_ds else 0} samples, using [0, {train_samples}] (same as Stage 1)")
            except Exception as e:
                logger.warning(f"     ⚠️  Could not check extended range, using [0, {train_samples}]: {e}")
            
            # Load training data with offset if available
            full_ds = load_split_dataset(merged_cfg, "train_split", train_samples, formatter, offset=train_offset)
            if full_ds is None or len(full_ds) < train_samples:
                raise ValueError(f"Dataset {merged_cfg['name']} doesn't have enough samples. Requested {train_samples}, got {len(full_ds) if full_ds else 0}")
            
            # For validation, still use the first samples
            validation_samples_actual = validation_samples or 0
            if validation_samples_actual > 0:
                # Load validation from beginning
                eval_ds = load_split_dataset(merged_cfg, "train_split", validation_samples_actual, formatter, offset=0)
            else:
                eval_ds = None
            
            if len(full_ds) > train_samples:
                train_ds = full_ds.select(range(train_samples))
            else:
                train_ds = full_ds
            logger.info(f"     Split: {len(train_ds)} train, {len(eval_ds) if eval_ds else 0} validation (no overlap)")
        else:
            # Different splits for train and validation
            # For training, try to use samples [train_samples, 2*train_samples]
            train_offset = None
            
            # Check if we can use extended range
            try:
                # Try to load 2*train_samples to check availability
                test_ds = load_split_dataset(merged_cfg, "train_split", 2 * train_samples, formatter, offset=None)
                if test_ds is not None and len(test_ds) >= 2 * train_samples:
                    logger.info(f"  ✓ {merged_cfg['task_type']}: Using samples [{train_samples}, {2*train_samples}] (Stage 3 extended range)")
                    train_offset = train_samples
                else:
                    logger.info(f"  ⚠️  {merged_cfg['task_type']}: Only {len(test_ds) if test_ds else 0} samples available, using [0, {train_samples}] (same as Stage 1)")
            except Exception as e:
                logger.warning(f"  ⚠️  {merged_cfg['task_type']}: Could not check extended range, using [0, {train_samples}]: {e}")
            
            train_ds = load_split_dataset(merged_cfg, "train_split", "train_samples", formatter, offset=train_offset)
            if train_ds is None or len(train_ds) == 0:
                raise ValueError(f"Dataset {merged_cfg['name']} returned no training samples.")
            eval_ds = load_split_dataset(merged_cfg, "vali_split", "validation_samples", formatter)

        if intra_group_shuffle:
            train_ds = train_ds.shuffle(seed=seed)

        tasks.append({
            "task_type": merged_cfg["task_type"],
            "name": merged_cfg["name"],
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
        })

        logger.info(f"  • {merged_cfg['task_type']:>12}: {len(train_ds)} train samples" + (f", {len(eval_ds)} eval samples" if eval_ds else ", no eval split"))

    logger.info("=" * 80 + "\n")
    return tasks


def tokenize_function(examples: Dict, tokenizer: AutoTokenizer, max_length: int) -> Dict[str, List[int]]:
    texts = examples["text"]
    responses = examples["response"]
    input_ids_list = []
    attention_masks = []
    labels_list = []

    for text, response in zip(texts, responses):
        instruction_tokens = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        response_tokens = tokenizer(response + tokenizer.eos_token, add_special_tokens=False, truncation=False)["input_ids"]
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

    return {"input_ids": input_ids_list, "attention_mask": attention_masks, "labels": labels_list}


class TaskCyclingTrainer(Trainer):
    def __init__(self, task_datasets: Dict[str, Dataset], steps_per_task: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_datasets = task_datasets
        self.steps_per_task = steps_per_task
        self.task_order = list(task_datasets.keys())
        self.current_task_idx = 0
        self.current_task = self.task_order[0]
        self.global_step_when_switched = 0
        self._task_state_restored = False

        logger.info("\n🔄 Task Cycling Configuration:")
        logger.info(f"  Task order: {' → '.join(self.task_order)}")
        logger.info(f"  Steps per task: {steps_per_task}\n")
    
    def _restore_task_state_from_checkpoint(self):
        if self._task_state_restored:
            return
        if hasattr(self.state, 'global_step') and self.state.global_step is not None and self.state.global_step > 0:
            total_steps_in_full_cycle = len(self.task_order) * self.steps_per_task
            steps_in_current_cycle = self.state.global_step % total_steps_in_full_cycle
            self.current_task_idx = steps_in_current_cycle // self.steps_per_task
            self.current_task = self.task_order[self.current_task_idx]
            cycles_completed = self.state.global_step // total_steps_in_full_cycle
            self.global_step_when_switched = (cycles_completed * total_steps_in_full_cycle) + (self.current_task_idx * self.steps_per_task)
            logger.info("\n" + "=" * 60)
            logger.info(f"🔄 Restored task cycling state from checkpoint:")
            logger.info(f"   Global step: {self.state.global_step}")
            logger.info(f"   Current task: {self.current_task.upper()} (index {self.current_task_idx})")
            logger.info(f"   Switched at step: {self.global_step_when_switched}")
            logger.info("=" * 60 + "\n")
        self._task_state_restored = True

    def get_train_dataloader(self) -> DataLoader:
        current_dataset = self.task_datasets[self.current_task]
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
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=min(2, self.args.dataloader_num_workers),
            pin_memory=False,
            shuffle=False,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self._task_state_restored:
            self._restore_task_state_from_checkpoint()
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        if hasattr(self.state, 'global_step') and self.state.global_step is not None:
            steps_since_switch = self.state.global_step - self.global_step_when_switched
            if steps_since_switch >= self.steps_per_task:
                self.current_task_idx = (self.current_task_idx + 1) % len(self.task_order)
                self.current_task = self.task_order[self.current_task_idx]
                self.global_step_when_switched = self.state.global_step
                logger.info("\n" + "=" * 60)
                logger.info(f"🔄 Switching to task: {self.current_task.upper()} (at global step {self.state.global_step})")
                logger.info("=" * 60 + "\n")
                self._train_dataloader = None
        return loss

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if hasattr(self.state, 'global_step') and self.state.global_step is not None:
            steps_since_switch = self.state.global_step - self.global_step_when_switched
            if steps_since_switch >= self.steps_per_task:
                self.current_task_idx = (self.current_task_idx + 1) % len(self.task_order)
                self.current_task = self.task_order[self.current_task_idx]
                self.global_step_when_switched = self.state.global_step
                logger.info("\n" + "=" * 60)
                logger.info(f"🔄 Switching to task: {self.current_task.upper()} (at global step {self.state.global_step})")
                logger.info("=" * 60 + "\n")
                self._train_dataloader = None
        if self.state.global_step > 0 and len(self.state.log_history) > 0:
            self.state.log_history[-1]["current_task"] = self.current_task
        return super()._maybe_log_save_evaluate(*args, **kwargs)


class TaskLoggingCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if hasattr(self.trainer, "current_task"):
            logs["task"] = self.trainer.current_task
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
                    counts = module.gate.expert_counts.cpu().numpy()
                    expert_usage.append(counts)
                    module.gate.expert_counts.zero_()
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


class AuxLossSchedulerCallback(TrainerCallback):
    def __init__(self, initial_alpha: float, final_alpha: float, warmup_steps: int = 0, decay_steps: int = None):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.last_logged_step = -1
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step is None:
            return
        if self.decay_steps is None:
            total_steps = args.max_steps if args.max_steps > 0 else 10000
            progress = min(1.0, max(0.0, (state.global_step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)))
            current_alpha = self.initial_alpha * (1 - progress) + self.final_alpha * progress
        else:
            progress = min(1.0, max(0.0, (state.global_step - self.warmup_steps) / max(1, self.decay_steps)))
            current_alpha = self.initial_alpha * (1 - progress) + self.final_alpha * progress
        if state.global_step < self.warmup_steps:
            current_alpha = self.initial_alpha
        for name, module in model.named_modules():
            if hasattr(module, "gate") and hasattr(module.gate, "alpha"):
                module.gate.alpha = current_alpha
        if state.global_step % args.logging_steps == 0 and state.global_step != self.last_logged_step:
            logger.info(f"🔧 Aux loss alpha: {current_alpha:.4f} (step {state.global_step})")
            self.last_logged_step = state.global_step


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


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank <= 0 else logging.WARN,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Stage 3: Update all adapters and router weights for overall performance")
    logger.info("=" * 80 + "\n")

    if model_args.pretraining_tp > 1:
        world_size = training_args.world_size if hasattr(training_args, 'world_size') else 1
        if world_size > 1 and world_size % model_args.pretraining_tp != 0:
            raise ValueError(f"Tensor parallelism degree ({model_args.pretraining_tp}) must divide total number of processes ({world_size})")
        data_parallel_size = world_size // model_args.pretraining_tp
        logger.info(f"🔀 Parallelism: TP={model_args.pretraining_tp}, DP={data_parallel_size}, Total={world_size}")

    data_config = load_data_config(data_args.data_config)
    processing_cfg = data_config.get("processing", {})
    if "max_seq_length" in processing_cfg:
        data_args.max_seq_length = processing_cfg["max_seq_length"]

    model_kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16}
    if model_args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if model_args.use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    adapter_config_path = os.path.join(model_args.model_name_or_path, "adapter_config.json")
    has_adapter_in_model_path = os.path.exists(adapter_config_path)
    resume_checkpoint = getattr(training_args, 'resume_from_checkpoint', None)
    has_adapter_in_resume = False
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        resume_adapter_config = os.path.join(resume_checkpoint, "adapter_config.json")
        has_adapter_in_resume = os.path.exists(resume_adapter_config)
        if has_adapter_in_resume:
            logger.info(f"📦 Detected PEFT adapter in resume checkpoint: {resume_checkpoint}")
    
    if has_adapter_in_resume:
        adapter_source = resume_checkpoint
        adapter_source_has_adapter = True
    elif model_args.adapter_path:
        adapter_source = model_args.adapter_path
        adapter_config_path_check = os.path.join(adapter_source, "adapter_config.json")
        adapter_source_has_adapter = os.path.exists(adapter_config_path_check)
    else:
        adapter_source = model_args.model_name_or_path
        adapter_source_has_adapter = has_adapter_in_model_path
    
    if adapter_source_has_adapter:
        logger.info(f"📦 Detected PEFT adapter at: {adapter_source}")
        peft_config = PeftConfig.from_pretrained(adapter_source)
        base_model_path = peft_config.base_model_name_or_path
        logger.info(f"   Loading base model from: {base_model_path}")
        config = LlamaMoEConfig.from_pretrained(base_model_path, trust_remote_code=True)
        if model_args.pretraining_tp > 1:
            config.pretraining_tp = model_args.pretraining_tp
            logger.info(f"🔀 Tensor Parallelism enabled: TP={model_args.pretraining_tp}")
        model = LlamaMoEForCausalLM.from_pretrained(base_model_path, config=config, **model_kwargs)
    else:
        base_model_path = model_args.model_name_or_path
        logger.info(f"Loading base model from {base_model_path}")
        config = LlamaMoEConfig.from_pretrained(base_model_path, trust_remote_code=True)
        if model_args.pretraining_tp > 1:
            config.pretraining_tp = model_args.pretraining_tp
            logger.info(f"🔀 Tensor Parallelism enabled: TP={model_args.pretraining_tp}")
        model = LlamaMoEForCausalLM.from_pretrained(base_model_path, config=config, **model_kwargs)
    
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        if model.config.model_type != "llama_moe":
            logger.warning(f"Fixing model_type in loaded config: {model.config.model_type} -> llama_moe")
            model.config.model_type = "llama_moe"

    if model_args.use_qlora:
        logger.info("🔧 Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        if adapter_source_has_adapter:
            logger.info(f"📦 Loading existing LoRA adapters from {adapter_source}...")
            model = PeftModel.from_pretrained(model, adapter_source, is_trainable=True)
            model.print_trainable_parameters()
            
            # For Stage 3, check if attention adapters exist and add them if missing
            logger.info("\n🔍 Checking for attention (non-MoE) adapters...")
            attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            has_attention_adapters = False
            frozen_attention_adapters = 0
            
            # Check if any attention modules have LoRA adapters by checking parameter names
            for name, param in model.named_parameters():
                if any(attn in name for attn in attention_modules) and "lora" in name.lower():
                    has_attention_adapters = True
                    if not param.requires_grad:
                        frozen_attention_adapters += 1
            
            if not has_attention_adapters:
                logger.info("  ⚠️  No attention adapters found. Adding attention LoRA adapters for Stage 3...")
                # Get unique attention module names from the model structure
                attention_target_modules = []
                for name, module in model.named_modules():
                    if any(attn in name for attn in attention_modules):
                        module_name = name.split(".")[-1]
                        if module_name not in attention_target_modules:
                            attention_target_modules.append(module_name)
                
                if attention_target_modules:
                    attention_lora_config = LoraConfig(
                        r=model_args.lora_r,
                        lora_alpha=model_args.lora_alpha,
                        target_modules=attention_target_modules,
                        lora_dropout=model_args.lora_dropout,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                    )
                    # Add attention adapters to existing model
                    model.add_adapter("attention", attention_lora_config)
                    # For training, we want both adapters active - PEFT will handle this during forward pass
                    logger.info(f"  ✅ Added attention adapters targeting: {attention_target_modules}")
                    logger.info(f"  ℹ️  Both 'default' (MoE) and 'attention' adapters will be used during training")
                else:
                    logger.warning("  ⚠️  Could not find attention modules to add adapters to")
            else:
                if frozen_attention_adapters > 0:
                    logger.info(f"  ✅ Attention adapters exist but {frozen_attention_adapters} parameters are frozen")
                else:
                    logger.info("  ✅ Attention adapters already exist and are trainable")
            
            # Unfreeze all adapters for Stage 3 (both MoE and attention)
            logger.info("\n🔓 Unfreezing all adapters for Stage 3 training...")
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if "lora" in name.lower() and not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_count += 1
            
            if unfrozen_count > 0:
                logger.info(f"  ✅ Unfrozen {unfrozen_count} adapter parameters")
            else:
                logger.info("  ✅ All adapters are already trainable")
            
            model.print_trainable_parameters()
        else:
            logger.info("📦 Initializing new LoRA adapters...")
            target_modules = []
            for name, module in model.named_modules():
                if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
                    target_modules.append(name.split(".")[-1])
                elif "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                    if "mlp" in name or "experts" in name:
                        target_modules.append(name.split(".")[-1])
            target_modules = list(dict.fromkeys(target_modules))
            peft_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        # For Stage 3, we don't freeze adapters - train everything
        # Only freeze if explicitly requested (for backward compatibility)
        if model_args.freeze_non_moe_lora:
            logger.warning("⚠️  freeze_non_moe_lora=True: Freezing attention adapters (not recommended for Stage 3)")
            model = freeze_non_moe_adapters(model)
        else:
            logger.info("✅ All adapters (MoE + attention) are trainable for Stage 3")

    initial_aux_loss = model_args.aux_loss_initial if model_args.aux_loss_initial is not None else model_args.aux_loss_alpha
    update_model_aux_loss_alpha(model, initial_aux_loss)

    task_templates = processing_cfg.get("task_prompt_templates", {})
    task_templates = {**TASK_PROMPT_TEMPLATES, **task_templates}
    tasks = load_stage2a_tasks(data_config, task_templates, intra_group_shuffle=data_args.intra_group_shuffle)
    steps_per_task = data_args.steps_per_task

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenize_fn = partial(tokenize_function, tokenizer=tokenizer, max_length=data_args.max_seq_length)

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
    
    # If shuffle_tasks is True, combine all tasks into one shuffled dataset
    if data_args.shuffle_tasks:
        logger.info("\n🔀 Shuffling all tasks together for better routing diversity...")
        all_train_datasets = list(tokenized_task_datasets.values())
        shuffled_dataset = concatenate_datasets(all_train_datasets)
        processing_cfg = data_config.get("processing", {})
        seed = processing_cfg.get("seed", 42)
        shuffled_dataset = shuffled_dataset.shuffle(seed=seed)
        logger.info(f"  Combined {len(tokenized_task_datasets)} tasks into {len(shuffled_dataset)} samples")
        logger.info(f"  This helps the router learn to distinguish tasks in each batch\n")
        # Replace task_datasets with a single "mixed" task
        tokenized_task_datasets = {"mixed": shuffled_dataset}

    eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
    if eval_dataset:
        max_eval_samples = 200
        if len(eval_dataset) > max_eval_samples:
            logger.warning(f"Eval dataset has {len(eval_dataset)} samples, limiting to {max_eval_samples} to avoid OOM")
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"\nEval dataset size: {len(eval_dataset)} samples")
    else:
        logger.info("\nNo evaluation split available; skipping evaluation.")

    use_metrics = not data_args.eval_loss_only
    if eval_dataset:
        if use_metrics:
            logger.warning("⚠️  Computing full metrics (accuracy) - this requires accumulating predictions in memory. If you get OOM, use --eval_loss_only True to only compute loss.")
        else:
            logger.info("  Using eval_loss_only=True (only compute loss, skip predictions to save memory)")

    # Use regular Trainer if shuffling, otherwise use TaskCyclingTrainer
    if data_args.shuffle_tasks:
        logger.info("  Using shuffled training (all tasks mixed together)")
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=list(tokenized_task_datasets.values())[0],
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if (eval_dataset and use_metrics) else None,
        )
    else:
        logger.info("  Using task cycling training")
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
    
    if eval_dataset and not use_metrics:
        trainer.args.prediction_loss_only = True

    trainer.add_callback(TaskLoggingCallback(trainer))
    trainer.add_callback(ExpertUtilizationCallback())
    
    if model_args.aux_loss_initial is not None and model_args.aux_loss_initial != model_args.aux_loss_alpha:
        logger.info(f"\n📈 Aux loss scheduling: {model_args.aux_loss_initial:.4f} → {model_args.aux_loss_alpha:.4f}")
        logger.info(f"   Warmup steps: {model_args.aux_loss_warmup_steps}, Decay steps: {model_args.aux_loss_decay_steps or 'all'}")
        trainer.add_callback(AuxLossSchedulerCallback(
            initial_alpha=model_args.aux_loss_initial,
            final_alpha=model_args.aux_loss_alpha,
            warmup_steps=model_args.aux_loss_warmup_steps,
            decay_steps=model_args.aux_loss_decay_steps,
        ))

    logger.info("\n🚀 Starting Stage 3 training...")
    if data_args.shuffle_tasks:
        logger.info(f"  Training mode: Shuffled (all tasks mixed)")
    else:
        logger.info(f"  Task order: {' → '.join(tokenized_task_datasets.keys())}")
        logger.info(f"  Steps per task: {steps_per_task}")
    if training_args.max_steps and training_args.max_steps > 0:
        logger.info(f"  Max steps: {training_args.max_steps}")
    logger.info("")

    resume_checkpoint = training_args.resume_from_checkpoint
    if resume_checkpoint:
        logger.info(f"Resume: {resume_checkpoint} | LR={training_args.learning_rate} | Warmup={training_args.warmup_ratio}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info("\n💾 Saving Stage 3 checkpoint...")
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        if model.config.model_type != "llama_moe":
            logger.warning(f"Fixing model_type in config: {model.config.model_type} -> llama_moe")
            model.config.model_type = "llama_moe"
    
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
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

    logger.info("\n✅ Stage 3 training complete.")


if __name__ == "__main__":
    train()
