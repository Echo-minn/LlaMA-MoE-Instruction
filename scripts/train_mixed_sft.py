#!/usr/bin/env python3
"""
Mixed Dataset SFT Training for MoE Expert Specialization
Supports validation mode (single dataset) and full mode (all datasets)
"""

import os
import sys
import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional, List
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configuration_llama_moe import LlamaMoEConfig
from src.modeling_llama_moe import LlamaMoEForCausalLM

# Register model
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
AutoConfig.register("llama_moe", LlamaMoEConfig)
AutoModelForCausalLM.register(LlamaMoEConfig, LlamaMoEForCausalLM)
# Register tokenizer with fast tokenizer class (like train_sft_reasoning.py)
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
    num_experts_to_train: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    data_config: str = field(default="configs/data_mix_instruction.yaml")
    max_seq_length: int = field(default=1024)

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

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

def format_oasst(example):
    """Format OASST conversation data"""
    # OASST has message tree structure
    # For simplicity, take first user-assistant pair
    messages = example.get('messages', [])
    
    instruction = ""
    response = ""
    
    for msg in messages:
        if msg.get('role') == 'user' and not instruction:
            instruction = msg.get('content', '')
        elif msg.get('role') == 'assistant' and not response:
            response = msg.get('content', '')
            break
    
    return {'instruction': instruction, 'response': response}

def format_gsm8k(example):
    """Format GSM8K math problems"""
    question = example.get('question', '')
    answer = example.get('answer', '')
    
    return {'instruction': question, 'response': answer}

def load_single_dataset(dataset_config, mode='validation'):
    """Load a single dataset based on configuration"""
    name = dataset_config['name']
    format_type = dataset_config.get('format', 'alpaca')
    
    # Determine number of samples
    if mode == 'validation':
        num_samples = dataset_config.get('samples_validation', 5000)
    else:
        num_samples = dataset_config.get('samples', 10000)
    
    logger.info(f"Loading {name} ({format_type} format, {num_samples} samples)...")
    
    try:
        # Load dataset
        split = dataset_config.get('split', 'train')
        config_name = dataset_config.get('config', None)
        
        # Load with or without config name
        if config_name:
            dataset = load_dataset(name, config_name, split=f"{split}[:{num_samples}]")
        else:
            dataset = load_dataset(name, split=f"{split}[:{num_samples}]")
        
        # Format based on type
        if format_type == 'alpaca':
            dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
        elif format_type == 'oasst':
            dataset = dataset.map(format_oasst, remove_columns=dataset.column_names)
        elif format_type == 'gsm8k':
            dataset = dataset.map(format_gsm8k, remove_columns=dataset.column_names)
        
        # Filter out empty responses
        dataset = dataset.filter(lambda x: len(x['instruction']) > 0 and len(x['response']) > 0)
        
        # Add task type label
        dataset = dataset.map(lambda x: {**x, 'task_type': dataset_config.get('task_type', 'general')})
        
        logger.info(f"  ✅ Loaded {len(dataset)} valid samples from {name}")
        return dataset
        
    except Exception as e:
        logger.error(f"  ❌ Failed to load {name}: {e}")
        return None

def load_mixed_datasets(config_path, mode='validation'):
    """Load and mix multiple datasets"""
    config = load_data_config(config_path)
    
    mode = config.get('mode', mode)
    logger.info(f"Loading datasets in {mode.upper()} mode")
    logger.info("=" * 80)
    
    datasets_to_mix = []
    
    for dataset_config in config['datasets']:
        # Check if enabled
        if mode == 'validation':
            # In validation mode, only use first enabled dataset
            if dataset_config.get('enabled', False):
                dataset = load_single_dataset(dataset_config, mode='validation')
                if dataset is not None:
                    datasets_to_mix.append(dataset)
                logger.info(f"\n⚠️  VALIDATION MODE: Using only first dataset")
                logger.info(f"   To use all datasets, change mode to 'full' in config")
                break
        else:
            # In full mode, use all enabled datasets
            if dataset_config.get('enabled', False):
                dataset = load_single_dataset(dataset_config, mode='full')
                if dataset is not None:
                    datasets_to_mix.append(dataset)
    
    if not datasets_to_mix:
        raise ValueError("No datasets loaded! Check your configuration.")
    
    # Concatenate and shuffle
    logger.info(f"\n📊 Mixing {len(datasets_to_mix)} dataset(s)...")
    mixed_dataset = concatenate_datasets(datasets_to_mix)
    
    if config['processing'].get('shuffle', True):
        seed = config['processing'].get('seed', 42)
        mixed_dataset = mixed_dataset.shuffle(seed=seed)
    
    logger.info(f"✅ Final mixed dataset: {len(mixed_dataset)} samples")
    logger.info("=" * 80)
    
    return mixed_dataset, config

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples with instruction masking"""
    instructions = examples['instruction']
    responses = examples['response']
    
    all_input_ids = []
    all_labels = []
    all_attention_mask = []
    
    for instruction, response in zip(instructions, responses):
        # Build prompt
        instruction_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
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
        
        # Smart truncation (75/25 rule)
        max_instruction_length = int(max_length * 0.75)
        
        if len(instruction_tokens) > max_instruction_length:
            instruction_tokens = instruction_tokens[:max_instruction_length]
        
        remaining_space = max_length - len(instruction_tokens)
        
        if len(response_tokens) > remaining_space:
            response_tokens = response_tokens[:remaining_space]
        
        # Combine
        input_ids = instruction_tokens + response_tokens
        labels = [-100] * len(instruction_tokens) + response_tokens[:]
        
        # Pad
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

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank <= 0 else logging.WARN,
    )
    
    # Load data configuration
    logger.info(f"\n{'='*80}")
    logger.info("Mixed Dataset SFT Training for MoE Expert Specialization")
    logger.info(f"{'='*80}\n")
    
    mixed_dataset, data_config = load_mixed_datasets(data_args.data_config)
    
    # Split train/eval
    split_dataset = mixed_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}\n")
    
    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}...")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    
    # Enable Flash Attention 2 for faster training
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
    
    model = LlamaMoEForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # Apply LoRA
    if model_args.use_qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load tokenizer (same approach as train_sft_reasoning.py)
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
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4,
        desc="Tokenizing train"
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=2,
        desc="Tokenizing eval"
    )
    
    # Training
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    logger.info("\n🚀 Starting training...")
    
    # Check if we should resume from checkpoint
    resume_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        resume_checkpoint = training_args.resume_from_checkpoint
        logger.info(f"   Resuming from checkpoint: {resume_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    logger.info("\n💾 Saving model...")
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("\n✅ Training complete!")

if __name__ == "__main__":
    train()

