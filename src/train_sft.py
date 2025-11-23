import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    HfArgumentParser
)
from datasets import load_dataset
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.modeling_llama_moe import LlamaMoEForCausalLM
except ImportError:
    from modeling_llama_moe import LlamaMoEForCausalLM

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="models/Llama-3.1-8B-MoE-Upcycled")
    use_flash_attn: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(
        default="facebook/natural_reasoning", 
        metadata={"help": "HuggingFace dataset path or local path"}
    )
    max_seq_length: int = field(default=2048)
    val_size: float = field(default=0.05, metadata={"help": "Validation set size if not provided"})

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

def compute_metrics(eval_preds):
    """Compute metrics for evaluation."""
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    
    return {
        "accuracy": (preds == labels).astype(np.float32).mean().item()
    }

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load Model
    logger.info(f"Loading MoE model from {model_args.model_name_or_path}...")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if training_args.bf16 else torch.float16,
        "trust_remote_code": True,
    }
    if model_args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = LlamaMoEForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        **model_kwargs
    )
    
    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load Tokenizer
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

    # Load Dataset
    logger.info(f"Loading dataset: {data_args.data_path}")
    try:
        dataset = load_dataset(data_args.data_path)
    except Exception as e:
        logger.warning(f"Failed to load {data_args.data_path}: {e}. Trying local load or verifying dataset name.")
        raise e

    # Select split
    if "train" in dataset:
        full_dataset = dataset["train"]
    else:
        full_dataset = dataset

    # Split dataset into train and eval(50K samples, can be tuned)
    MAX_SAMPLES = 50000 
    if len(full_dataset) > MAX_SAMPLES:
        logger.info(f"Dataset too large ({len(full_dataset)}), selecting random {MAX_SAMPLES} samples...")
        full_dataset = full_dataset.shuffle(seed=42).select(range(MAX_SAMPLES))

    logger.info(f"Splitting dataset into Train/Val (Val size: {data_args.val_size})...")
    dataset_split = full_dataset.train_test_split(test_size=data_args.val_size, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # Data Processing Function (SFT Format)
    def formatting_func(example):
        input_text = example.get("question", "")
        output_text = ""
        
        if "responses" in example and isinstance(example["responses"], list) and len(example["responses"]) > 0:
            # Use the first response(high quality answer)
            first_response = example["responses"][0]
            if isinstance(first_response, dict) and "response" in first_response:
                output_text = first_response["response"]
            else:
                output_text = str(first_response)
        elif "reference_answer" in example:
            output_text = example["reference_answer"]
            
        prompt = f"### Instruction:\n{input_text}\n\n### Response:\n{output_text}"
        return prompt + tokenizer.eos_token

    def tokenize_function(examples):
        texts = [formatting_func(ex) if isinstance(ex, dict) else formatting_func(examples) for ex in (examples if isinstance(examples, list) else [examples])]
        if isinstance(examples, dict):
            texts = []
            for i in range(len(examples[list(examples.keys())[0]])):
                ex = {k: examples[k][i] for k in examples}
                texts.append(formatting_func(ex))
                
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
        )

    logger.info("Tokenizing train dataset...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=8
    )
    
    logger.info("Tokenizing eval dataset...")
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=4
    )

    logger.info(f"Training samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Done!")

if __name__ == "__main__":
    train()
