#!/usr/bin/env python3
"""
Stage 2: Task-Grouped MoE Training for Expert Specialization
Sequential task training with frozen attention LoRA
"""

import os
import sys
import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    TrainerCallback,
)
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configuration_llama_moe import LlamaMoEConfig
from src.modeling_llama_moe import LlamaMoEForCausalLM

# Register model
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
AutoConfig.register("llama_moe", LlamaMoEConfig)
AutoModelForCausalLM.register(LlamaMoEConfig, LlamaMoEForCausalLM)
AutoTokenizer.register(LlamaMoEConfig, fast_tokenizer_class=PreTrainedTokenizerFast)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/Llama-3.2-3B-Instruct-MoE-8x")
    use_flash_attn: bool = field(default=True)
    use_qlora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    aux_loss_alpha: float = field(default=0.01, metadata={"help": "Load balancing loss weight"})


@dataclass
class DataArguments:
    data_config: str = field(default="configs/data_task_grouped.yaml")
    max_seq_length: int = field(default=1024)


def load_data_config(config_path):
    """Load dataset configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def format_alpaca(example):
    """Format Alpaca-style data"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '') or example.get('response', '')
    
    if input_text:
        prompt = f"{instruction}\n\nInput: {input_text}"
    else:
        prompt = instruction
    
    return {'instruction': prompt, 'response': output}


def format_ultrachat(example):
    """Format UltraChat conversation data"""
    messages = example.get('messages', [])
    
    if not messages or len(messages) < 2:
        return {'instruction': '', 'response': ''}
    
    # Take first user message as instruction and first assistant message as response
    instruction = ""
    response = ""
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'user' and not instruction:
            instruction = content
        elif role == 'assistant' and instruction and not response:
            response = content
            break
    
    return {'instruction': instruction, 'response': response}


def format_gsm8k(example):
    """Format GSM8K math problems"""
    question = example.get('question', '')
    answer = example.get('answer', '')
    
    return {'instruction': question, 'response': answer}


def load_single_dataset(dataset_config):
    """Load a single dataset based on configuration"""
    name = dataset_config['name']
    format_type = dataset_config.get('format', 'alpaca')
    num_samples = dataset_config.get('samples', 10000)
    
    logger.info(f"  Loading {name} ({format_type}, {num_samples} samples)...")
    
    try:
        split = dataset_config.get('split', 'train')
        config_name = dataset_config.get('config', None)
        
        if config_name:
            dataset = load_dataset(name, config_name, split=f"{split}[:{num_samples}]")
        else:
            dataset = load_dataset(name, split=f"{split}[:{num_samples}]")
        
        # Format based on type
        if format_type == 'alpaca':
            dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
        elif format_type == 'ultrachat':
            dataset = dataset.map(format_ultrachat, remove_columns=dataset.column_names)
        elif format_type == 'gsm8k':
            dataset = dataset.map(format_gsm8k, remove_columns=dataset.column_names)
        
        # Filter out empty responses
        dataset = dataset.filter(lambda x: len(x['instruction']) > 0 and len(x['response']) > 0)
        
        # Add task type label
        task_type = dataset_config.get('task_type', 'general')
        dataset = dataset.map(lambda x: {**x, 'task_type': task_type})
        
        logger.info(f"    ✅ {len(dataset)} valid samples")
        return dataset, task_type
        
    except Exception as e:
        logger.error(f"    ❌ Failed to load {name}: {e}")
        return None, None


def load_task_grouped_datasets(config_path):
    """Load datasets keeping tasks separate (not mixed)"""
    config = load_data_config(config_path)
    
    logger.info("=" * 80)
    logger.info("Loading Task-Grouped Datasets (Stage 2)")
    logger.info("=" * 80)
    
    task_datasets = {}  # {task_type: dataset}
    task_configs = {}   # {task_type: config}
    
    for dataset_config in config['datasets']:
        if not dataset_config.get('enabled', False):
            continue
        
        dataset, task_type = load_single_dataset(dataset_config)
        if dataset is not None:
            task_datasets[task_type] = dataset
            task_configs[task_type] = dataset_config
    
    if not task_datasets:
        raise ValueError("No datasets loaded! Check your configuration.")
    
    logger.info(f"\n📊 Loaded {len(task_datasets)} separate task datasets:")
    for task_type, dataset in task_datasets.items():
        steps_per_cycle = task_configs[task_type].get('steps_per_cycle', 300)
        logger.info(f"  - {task_type}: {len(dataset)} samples, {steps_per_cycle} steps/cycle")
    
    logger.info("=" * 80 + "\n")
    
    return task_datasets, task_configs, config


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples with instruction masking and task-specific tokens"""
    instructions = examples['instruction']
    responses = examples['response']
    task_types = examples.get('task_type', [None] * len(instructions))
    
    all_input_ids = []
    all_labels = []
    all_attention_mask = []
    
    for instruction, response, task_type in zip(instructions, responses, task_types):
        # Prepend task-specific token if available
        task_prefix = ""
        if task_type:
            task_prefix = f"<|task_{task_type}|> "
        
        instruction_text = f"{task_prefix}### Instruction:\n{instruction}\n\n### Response:\n"
        
        instruction_tokens = tokenizer(
            instruction_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        
        response_tokens = tokenizer(
            response + tokenizer.eos_token,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        
        max_instruction_length = int(max_length * 0.75)
        
        if len(instruction_tokens) > max_instruction_length:
            instruction_tokens = instruction_tokens[:max_instruction_length]
        
        remaining_space = max_length - len(instruction_tokens)
        
        if len(response_tokens) > remaining_space:
            response_tokens = response_tokens[:remaining_space]
        
        input_ids = instruction_tokens + response_tokens
        labels = [-100] * len(instruction_tokens) + response_tokens[:]
        
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length
        attention_mask = [1] * (len(instruction_tokens) + len(response_tokens)) + [0] * padding_length
        
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_mask.append(attention_mask)
    
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


class TaskCyclingTrainer(Trainer):
    """Custom trainer that cycles through tasks sequentially"""
    
    def __init__(self, task_datasets: Dict[str, Dataset], steps_per_task: int = 300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.task_datasets = task_datasets
        self.steps_per_task = steps_per_task
        self.task_order = list(task_datasets.keys())
        self.current_task_idx = 0
        self.current_task = self.task_order[0]
        self.steps_in_current_task = 0
        
        logger.info(f"\n🔄 Task Cycling Configuration:")
        logger.info(f"  Task order: {' → '.join(self.task_order)}")
        logger.info(f"  Steps per task: {steps_per_task}")
        logger.info(f"  Starting task: {self.current_task}\n")
    
    def get_train_dataloader(self) -> DataLoader:
        """Return dataloader for current task"""
        # Get current task dataset
        current_dataset = self.task_datasets[self.current_task]
        
        # Create dataloader
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
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to track steps and switch tasks"""
        # Perform normal training step (pass all args to support both old and new transformers)
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        
        # Track steps in current task
        self.steps_in_current_task += 1
        
        # Check if we need to switch tasks
        if self.steps_in_current_task >= self.steps_per_task:
            # Switch to next task
            self.current_task_idx = (self.current_task_idx + 1) % len(self.task_order)
            self.current_task = self.task_order[self.current_task_idx]
            self.steps_in_current_task = 0
            
            # Log task switch
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Switching to task: {self.current_task.upper()}")
            logger.info(f"{'='*60}\n")
            
            # Force dataloader refresh
            self._train_dataloader = None
        
        return loss
    
    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """Override to log current task"""
        # Add current task to logs
        if self.state.global_step > 0 and len(self.state.log_history) > 0:
            self.state.log_history[-1]['current_task'] = self.current_task
        
        return super()._maybe_log_save_evaluate(*args, **kwargs)


class TaskLoggingCallback(TrainerCallback):
    """Callback to log current task and compute perplexity from loss"""
    
    def __init__(self, trainer):
        self.trainer = trainer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Add task info
            if hasattr(self.trainer, 'current_task'):
                logs['task'] = self.trainer.current_task
                logs['task_step'] = self.trainer.steps_in_current_task
            
            # Compute perplexity from loss
            if 'loss' in logs:
                logs['train_perplexity'] = np.exp(logs['loss'])
            if 'eval_loss' in logs:
                logs['eval_perplexity'] = np.exp(logs['eval_loss'])


class ExpertUtilizationCallback(TrainerCallback):
    """Callback to track and log MoE expert utilization statistics"""
    
    def __init__(self):
        self.expert_counts = {}
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Log expert utilization statistics"""
        if logs is None or model is None:
            return
        
        # Collect routing statistics from the model
        try:
            expert_usage = []
            total_tokens = 0
            
            # Iterate through MoE layers to get expert selection counts
            for name, module in model.named_modules():
                if hasattr(module, 'gate') and hasattr(module.gate, 'expert_counts'):
                    # Some gate implementations track expert selection
                    counts = module.gate.expert_counts
                    expert_usage.append(counts)
                    total_tokens += counts.sum()
            
            if expert_usage and total_tokens > 0:
                # Compute statistics
                avg_usage = np.mean(expert_usage, axis=0)  # Average across layers
                
                # Normalize to get probabilities
                usage_probs = avg_usage / avg_usage.sum() if avg_usage.sum() > 0 else avg_usage
                
                # Compute entropy (higher = more balanced usage)
                eps = 1e-10
                entropy = -np.sum(usage_probs * np.log(usage_probs + eps))
                max_entropy = np.log(len(usage_probs))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                # Compute coefficient of variation (lower = more balanced)
                mean_usage = np.mean(avg_usage)
                std_usage = np.std(avg_usage)
                cv = std_usage / mean_usage if mean_usage > 0 else 0
                
                # Log metrics
                logs['expert_entropy'] = float(normalized_entropy)
                logs['expert_balance_cv'] = float(cv)
                logs['expert_max_usage'] = float(np.max(usage_probs))
                logs['expert_min_usage'] = float(np.min(usage_probs))
                
                # Log individual expert usage (top 8 experts for 8-expert model)
                for i, usage in enumerate(usage_probs[:8]):
                    logs[f'expert_{i}_usage'] = float(usage)
                    
        except Exception as e:
            # Silently skip if we can't get expert stats
            logger.debug(f"Could not collect expert stats: {e}")
            pass


def freeze_attention_lora(model):
    """Freeze attention LoRA parameters, keep expert FFN LoRA and gates trainable"""
    logger.info("\n🔒 Freezing attention LoRA parameters...")
    
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    expert_modules = ["gate_proj", "up_proj", "down_proj"]
    gate_modules = ["gate.weight"]
    
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        # Freeze attention LoRA
        if any(attn_mod in name for attn_mod in attention_modules) and "lora" in name.lower():
            param.requires_grad = False
            frozen_params += param.numel()
        # Keep expert FFN LoRA trainable
        elif any(exp_mod in name for exp_mod in expert_modules) and "lora" in name.lower():
            param.requires_grad = True
            trainable_params += param.numel()
        # Keep gates trainable
        elif any(gate_mod in name for gate_mod in gate_modules):
            param.requires_grad = True
            trainable_params += param.numel()
        # Everything else depends on original setting
        elif param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"  Frozen parameters: {frozen_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable %: {100 * trainable_params / (trainable_params + frozen_params):.2f}%\n")
    
    return model


def update_model_aux_loss_alpha(model, aux_loss_alpha):
    """Update aux_loss_alpha in model's gate modules"""
    logger.info(f"🔧 Setting aux_loss_alpha = {aux_loss_alpha}")
    
    for name, module in model.named_modules():
        if hasattr(module, 'gate') and hasattr(module.gate, 'alpha'):
            module.gate.alpha = aux_loss_alpha
            logger.info(f"  Updated {name}.gate.alpha = {aux_loss_alpha}")
    
    logger.info("")


def compute_metrics(eval_preds):
    """Compute accuracy for evaluation (memory-efficient version)"""
    predictions, labels = eval_preds
    
    # predictions shape: (batch_size, seq_len) - argmax predictions only
    # labels shape: (batch_size, seq_len)
    
    # Note: We removed perplexity calculation to avoid OOM from storing full logits
    # Perplexity can be approximated from eval_loss: perplexity ≈ exp(loss)
    
    # Flatten arrays
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Only compute accuracy on non-masked tokens (labels != -100)
    mask = labels != -100
    
    if mask.sum() == 0:
        return {"accuracy": 0.0}
    
    # Compute accuracy
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    accuracy = float(correct) / float(total)
    
    return {
        "accuracy": accuracy,
        "correct_tokens": int(correct),
        "total_tokens": int(total),
    }


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank <= 0 else logging.WARN,
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("STAGE 2: Task-Grouped MoE Training for Expert Specialization")
    logger.info(f"{'='*80}\n")
    
    # Load task-grouped datasets
    task_datasets, task_configs, data_config = load_task_grouped_datasets(data_args.data_config)
    
    # Get steps per task from config
    steps_per_task = data_config['training'].get('steps_per_task', 300)
    
    # Load model WITH existing LoRA adapters from Stage 1
    logger.info(f"Loading model from {model_args.model_name_or_path}...")
    logger.info("  Note: Loading existing LoRA adapters from Stage 1 checkpoint\n")
    
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    
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
    
    # Load base model
    model = LlamaMoEForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # Load existing LoRA adapters from Stage 1 checkpoint
    if model_args.use_qlora:
        from peft import PeftModel
        
        logger.info("📦 Loading trained LoRA adapters from Stage 1...")
        model = PeftModel.from_pretrained(
            model, 
            model_args.model_name_or_path,
            is_trainable=True
        )
        
        logger.info("\n📊 Initial trainable parameters (with loaded LoRA):")
        model.print_trainable_parameters()
        
        # Freeze attention LoRA (Stage 2 specific)
        model = freeze_attention_lora(model)
    
    # Update aux_loss_alpha for load balancing
    update_model_aux_loss_alpha(model, model_args.aux_loss_alpha)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Add task-specific special tokens
    task_tokens = [
        "<|task_math|>",
        "<|task_code|>",
        "<|task_general|>",
        "<|task_conversation|>",
        "<|task_other|>",
        "<|task_general_instruction|>",
        "<|task_reasoning|>",
        "<|task_creative|>",
    ]
    
    logger.info("\n🏷️  Adding task-specific special tokens...")
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": task_tokens})
    logger.info(f"  Added {num_added} task tokens: {', '.join(task_tokens)}")
    
    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"  Resized embeddings to {len(tokenizer)} tokens\n")
    
    # Tokenize all task datasets
    logger.info("Tokenizing task datasets...")
    tokenized_task_datasets = {}
    
    for task_type, dataset in task_datasets.items():
        logger.info(f"  Tokenizing {task_type}...")
        
        # Split into train/eval
        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
        train_ds = split_dataset["train"]
        
        # Tokenize
        train_ds = train_ds.map(
            lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            remove_columns=train_ds.column_names,
            num_proc=4,
            desc=f"Tokenizing {task_type}"
        )
        
        tokenized_task_datasets[task_type] = train_ds
        logger.info(f"    ✅ {len(train_ds)} samples ready")
    
    # Create eval dataset (mix a small amount from each task)
    logger.info("\nCreating mixed evaluation dataset...")
    eval_datasets = []
    for task_type, dataset in task_datasets.items():
        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
        eval_ds = split_dataset["test"]
        eval_ds = eval_ds.map(
            lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            remove_columns=eval_ds.column_names,
            num_proc=2,
        )
        # Take subset for eval (100 samples per task for final evaluation)
        eval_ds = eval_ds.select(range(min(100, len(eval_ds))))
        eval_datasets.append(eval_ds)
    
    from datasets import concatenate_datasets
    eval_dataset = concatenate_datasets(eval_datasets)
    logger.info(f"  Eval dataset: {len(eval_dataset)} samples\n")
    
    # Create cycling trainer
    trainer = TaskCyclingTrainer(
        task_datasets=tokenized_task_datasets,
        steps_per_task=steps_per_task,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=list(tokenized_task_datasets.values())[0],  # Placeholder, will be cycled
        eval_dataset=eval_dataset,
    )
    
    # Add task logging callback
    trainer.add_callback(TaskLoggingCallback(trainer))
    
    # Add expert utilization tracking
    trainer.add_callback(ExpertUtilizationCallback())
    
    logger.info("🚀 Starting Stage 2 training...")
    logger.info(f"  Task order: {' → '.join(tokenized_task_datasets.keys())}")
    logger.info(f"  Steps per task: {steps_per_task}")
    logger.info(f"  Total steps: {training_args.max_steps}\n")
    
    # Check if we should resume from checkpoint
    resume_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        resume_checkpoint = training_args.resume_from_checkpoint
        logger.info(f"   Resuming from checkpoint: {resume_checkpoint}\n")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    logger.info("\n💾 Saving model...")
    
    # Ensure config has correct model_type before saving
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        if model.config.model_type != "llama_moe":
            logger.warning(f"Fixing model_type in config: {model.config.model_type} -> llama_moe")
            model.config.model_type = "llama_moe"
    
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Verify saved config has correct model_type
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
    
    logger.info("\n✅ Stage 2 training complete!")
    logger.info(f"\n📊 Next step: Check routing with:")
    logger.info(f"  bash scripts/run_routing_check.sh {training_args.output_dir}/checkpoint-{training_args.max_steps} 4\n")


if __name__ == "__main__":
    train()

