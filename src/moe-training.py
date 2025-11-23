import copy
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import numpy as np
import torch
import torch.distributed
import datasets
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, 
    PreTrainedTokenizer, Trainer, TrainingArguments, TrainerCallback, utils
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from modeling_llama_moe import LlamaMoEModel, LlamaMoEForCausalLM

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"
logger = logging.getLogger(__name__)

def build_instruction_prompt(instruction: str):
    return '''
You are an AI assistant, developed in CMU 11667 25 Fall Course. For politically sensitive questions, security and privacy issues, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

@dataclass
class ModelArguments:
    trainable : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default="embed_tokens,lm_head")
    use_lora : Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="models/Llama-3.1-8B")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the evaluation data."})

@dataclass
class TrainingArguments(TrainingArguments):
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project name. If None, uses report_to setting."},
    )

class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

class MoEExpertStatsCallback(TrainerCallback):
    """Callback to log MoE expert usage statistics during training."""
    
    def __init__(self):
        self.expert_stats = {}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log expert statistics if available in model outputs."""
        if logs is None:
            return
        
        # Try to extract MoE statistics from model if available
        model = kwargs.get("model")
        if model is not None:
            try:
                # Look for MoE layers and collect statistics
                expert_usage = self._collect_expert_stats(model)
                if expert_usage:
                    logs.update(expert_usage)
            except Exception as e:
                # Silently fail if we can't collect stats
                pass
    
    def _collect_expert_stats(self, model):
        """Collect expert usage statistics from MoE layers."""
        stats = {}
        total_experts = 0
        expert_usage_counts = {}
        
        # Traverse model to find MoE layers
        for name, module in model.named_modules():
            # Check if this is an MoE layer (has gate and experts)
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                n_experts = len(module.experts)
                total_experts += n_experts
                
                # Try to get recent gate statistics if available
                if hasattr(module.gate, '_last_expert_indices'):
                    expert_indices = module.gate._last_expert_indices
                    if expert_indices is not None:
                        # Count expert usage
                        unique, counts = torch.unique(expert_indices, return_counts=True)
                        for expert_id, count in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                            expert_usage_counts[int(expert_id)] = expert_usage_counts.get(int(expert_id), 0) + count
        
        if total_experts > 0 and expert_usage_counts:
            # Calculate load balancing metrics
            usage_values = list(expert_usage_counts.values())
            if usage_values:
                mean_usage = np.mean(usage_values)
                std_usage = np.std(usage_values)
                cv = std_usage / mean_usage if mean_usage > 0 else 0.0  # Coefficient of variation
                
                stats["moe/num_experts"] = total_experts
                stats["moe/active_experts"] = len(expert_usage_counts)
                stats["moe/mean_expert_usage"] = mean_usage
                stats["moe/std_expert_usage"] = std_usage
                stats["moe/expert_usage_cv"] = cv  # Lower is better (more balanced)
        
        return stats

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            # return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        len(tokenized.input_ids) for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        # Ensure pad_token_id is not None
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )

def _format_messages(msg):
    """Format messages to text when chat_template is not available."""
    text_parts = []
    for m in msg:
        role = m.get('role', 'user')
        content = m.get('content', '')
        if role == 'user':
            text_parts.append(f"User: {content}")
        elif role == 'assistant':
            text_parts.append(f"Assistant: {content}")
        else:
            text_parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(text_parts)

def train_tokenize_function(examples, tokenizer):
    """Tokenize function supporting both instruction/output and messages formats."""
    if 'messages' in examples:
        # Handle ultrachat_200k format with messages
        texts = []
        for msg in examples['messages']:
            if tokenizer.chat_template:
                try:
                    text = tokenizer.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=False
                    )
                    texts.append(text)
                except:
                    texts.append(_format_messages(msg))
            else:
                texts.append(_format_messages(msg))
        sources = [""] * len(texts)
        targets = texts
    else:
        # Original instruction/output format
        sources = [build_instruction_prompt(inst) for inst in examples['instruction']]
        targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    return preprocess(sources, targets, tokenizer)

def eval_tokenize_function(examples, tokenizer):
    """Same as train_tokenize_function but for evaluation."""
    return train_tokenize_function(examples, tokenizer)

def compute_metrics(eval_pred):
    """Compute perplexity and other metrics for evaluation.
    
    Args:
        eval_pred: A tuple of (predictions, labels) where:
            - predictions: numpy array of shape (batch_size, seq_len, vocab_size) with logits
            - labels: numpy array of shape (batch_size, seq_len) with label token ids
    """
    predictions, labels = eval_pred
    
    # Ensure predictions and labels are numpy arrays
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Shift predictions and labels for next-token prediction
    # predictions: (batch, seq_len, vocab_size) -> (batch, seq_len-1, vocab_size)
    # labels: (batch, seq_len) -> (batch, seq_len-1)
    shift_logits = predictions[..., :-1, :].reshape(-1, predictions.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    
    # Mask out ignored labels (where label == IGNORE_INDEX)
    mask = shift_labels != IGNORE_INDEX
    if mask.sum() == 0:
        # No valid labels to evaluate
        return {
            "perplexity": float('inf'),
            "eval_loss": float('inf'),
            "accuracy": 0.0,
        }
    
    shift_logits = shift_logits[mask]
    shift_labels = shift_labels[mask]
    
    # Convert to torch for computation
    shift_logits = torch.from_numpy(shift_logits).float()
    shift_labels = torch.from_numpy(shift_labels).long()
    
    # Compute cross entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(shift_logits, shift_labels)
    perplexity = torch.exp(loss).item()
    
    # Compute accuracy
    pred_ids = torch.argmax(shift_logits, dim=-1)
    accuracy = (pred_ids == shift_labels).float().mean().item()
    
    return {
        "perplexity": perplexity,
        "eval_loss": loss.item(),
        "accuracy": accuracy,
    }

def build_model(model_args, training_args, checkpoint_dir):
    """Build model with quantization and LoRA if specified."""
    if not model_args.use_lora:
        assert model_args.bits in [16, 32], "Full precision training requires bits=16 or 32"
    
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }
    
    # Add quantization config for LoRA with quantization
    if model_args.use_lora and model_args.bits < 16:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=model_args.bits == 4,
            load_in_8bit=model_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.double_quant,
            bnb_4bit_quant_type=model_args.quant_type,
        )
    
    # Add attention implementation if specified
    if hasattr(model_args, 'attn_implementation') and model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    
    model = LlamaMoEForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if compute_dtype == torch.float16 and model_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('='*80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('='*80)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32
    # Tokenizer
    
    if model_args.use_lora and model_args.bits < 16:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if model_args.use_lora:
        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        else:
            logger.info(f'Init LoRA modules...')
            target_modules = model_args.trainable.split(',')
            modules_to_save = model_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            lora_rank = model_args.lora_rank
            lora_dropout = model_args.lora_dropout
            lora_alpha = model_args.lora_alpha
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

def _load_config_from_json(config_file):
    """Load config from JSON file and convert to command line args format."""
    import json
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    config_args = []
    for key, value in config_dict.items():
        if isinstance(value, bool):
            if value:  # Only add flag if True
                config_args.append(f'--{key}')
        elif isinstance(value, list):
            for item in value:
                config_args.extend([f'--{key}', str(item)])
        else:
            config_args.extend([f'--{key}', str(value)])
    return config_args

def train():
    """Main training function."""
    import argparse
    
    # Parse config_file argument first
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument('--config_file', type=str, default=None)
    base_args, remaining_args = base_parser.parse_known_args()
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # Load config from JSON if provided, merge with command line args
    if base_args.config_file:
        config_args = _load_config_from_json(base_args.config_file)
        all_args = config_args + remaining_args
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=all_args)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    utils.logging.set_verbosity(log_level)
    utils.logging.enable_default_handler()
    utils.logging.enable_explicit_format()
    
    # Configure wandb project if specified
    if training_args.wandb_project and training_args.report_to and 'wandb' in training_args.report_to:
        import os
        os.environ['WANDB_PROJECT'] = training_args.wandb_project
    
    if training_args.local_rank == 0:
        logger.info('='*100)
        logger.info(training_args)
        # Log DeepSpeed configuration if used
        if hasattr(training_args, 'deepspeed') and training_args.deepspeed:
            logger.info(f'Using DeepSpeed configuration: {training_args.deepspeed}')
            logger.info('DeepSpeed ZeRO will handle distributed training (no DDP needed)')
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    logger.info("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    logger.info("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(model_args.model_name_or_path))
    
    resume_from_checkpoint_dir = get_last_checkpoint(training_args.output_dir)
    model = build_model(model_args, training_args, resume_from_checkpoint_dir)
        
    # Load training dataset
    if data_args.data_path.endswith('.parquet') or (os.path.exists(data_args.data_path) and not '/' in data_args.data_path.split('/')[-1]):
        raw_train_datasets = load_dataset('parquet', data_files=data_args.data_path, split="train", cache_dir=training_args.cache_dir)
    else:
        raw_dataset = load_dataset(data_args.data_path, cache_dir=training_args.cache_dir)
        raw_train_datasets = raw_dataset.get('train_sft') or raw_dataset.get('train')
        if raw_train_datasets is None:
            raise ValueError(f"Dataset {data_args.data_path} must have 'train_sft' or 'train' split")
    
    if training_args.local_rank > 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    if training_args.local_rank == 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        logger.info("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    # Load evaluation dataset if provided
    eval_dataset = None
    if data_args.eval_data_path is not None:
        if training_args.local_rank == 0:
            logger.info("Loading evaluation dataset from: {}".format(data_args.eval_data_path))
        
        # Load evaluation dataset
        if data_args.eval_data_path.endswith('.parquet') or (os.path.exists(data_args.eval_data_path) and not '/' in data_args.eval_data_path.split('/')[-1]):
            raw_eval_datasets = load_dataset('parquet', data_files=data_args.eval_data_path, split="train", cache_dir=training_args.cache_dir)
        else:
            raw_eval_dataset = load_dataset(data_args.eval_data_path, cache_dir=training_args.cache_dir)
            raw_eval_datasets = (raw_eval_dataset.get('test_sft') or 
                                  raw_eval_dataset.get('test') or 
                                  raw_eval_dataset.get('validation'))
            if raw_eval_datasets is None:
                raise ValueError(f"Dataset {data_args.eval_data_path} must have 'test_sft', 'test', or 'validation' split")
        
        if training_args.local_rank > 0 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        eval_dataset = raw_eval_datasets.map(
            eval_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_eval_datasets.column_names,
            load_from_cache_file=True,
            desc="Running Encoding on Eval",
            fn_kwargs={"tokenizer": tokenizer}
        )
        
        if training_args.local_rank == 0 and torch.distributed.is_initialized():
            torch.distributed.barrier()
            logger.info("Evaluation dataset samples:", len(eval_dataset))
            for index in random.sample(range(min(len(eval_dataset), 100)), 2):
                logger.info(f"Eval sample {index}: {tokenizer.decode(list(eval_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        data_collator=data_collator,
    )

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        **data_module
    )
    
    # Note: When using DeepSpeed ZeRO, DDP is not used, so no need to set static graph
    # DeepSpeed handles distributed training and gradient checkpointing compatibility automatically
    
    # Add callbacks
    if model_args.use_lora:
        trainer.add_callback(SavePeftModelCallback)
    trainer.add_callback(MoEExpertStatsCallback())
    
    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint_dir)
    trainer.save_state()
    
    # Final evaluation if eval dataset is provided
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        logger.info("Final evaluation metrics: {}".format(eval_metrics))
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    if not model_args.use_lora:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()