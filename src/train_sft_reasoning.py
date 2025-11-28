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
    HfArgumentParser,
    BitsAndBytesConfig
)
from datasets import load_dataset
import numpy as np
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from peft.tuners.lora import LoraLayer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
try:
    from src.configuration_llama_moe import LlamaMoEConfig
    from src.modeling_llama_moe import LlamaMoEForCausalLM
except ImportError:
    from configuration_llama_moe import LlamaMoEConfig
    from modeling_llama_moe import LlamaMoEForCausalLM

# 注册自定义模型配置和模型类，确保 AutoConfig 能识别 "llama_moe"
AutoConfig.register("llama_moe", LlamaMoEConfig)
AutoModelForCausalLM.register(LlamaMoEConfig, LlamaMoEForCausalLM)
AutoTokenizer.register(LlamaMoEConfig, fast_tokenizer_class=PreTrainedTokenizerFast)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="models/Llama-3.1-8B-MoE-Upcycled")
    use_flash_attn: bool = field(default=True)
    use_lora: bool = field(default=True, metadata={"help": "Use LoRA for parameter-efficient training"})
    use_qlora: bool = field(default=False, metadata={"help": "Use QLoRA (4-bit + LoRA) training"})
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    num_experts_to_train: Optional[int] = field(default=None, metadata={"help": "Number of experts to train (freeze others), e.g., 4 out of 8"})

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
    report_to: str = field(default="wandb") # 显式默认 wandb

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

    if "wandb" in training_args.report_to:
        if training_args.run_name is None:
            training_args.run_name = f"Llama-MoE-SFT-{os.path.basename(data_args.data_path)}"
        if training_args.local_rank <= 0:
            wandb.init(project="Llama-MoE-SFT", name=training_args.run_name)

    logger.info(f"Loading MoE model from {model_args.model_name_or_path}...")
    
    model_kwargs = {
        "trust_remote_code": True,
    }
    if model_args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if model_args.use_qlora:
        logger.info("Using QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["dtype"] = torch.bfloat16
    else:
        model_kwargs["dtype"] = torch.bfloat16 if training_args.bf16 else torch.float16

    model = LlamaMoEForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        **model_kwargs
    )
    
    # Freeze experts if specified
    if model_args.num_experts_to_train is not None:
        logger.info(f"Freezing experts: training only first {model_args.num_experts_to_train} experts per layer")
        frozen_params = 0
        trainable_params = 0
        
        # Iterate through all layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer_idx, layer in enumerate(model.model.layers):
                # Check if this layer has MoE experts
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                    experts = layer.mlp.experts
                    num_experts = len(experts)
                    
                    for expert_idx in range(num_experts):
                        expert = experts[expert_idx]
                        
                        if expert_idx >= model_args.num_experts_to_train:
                            # Freeze this expert
                            for param in expert.parameters():
                                param.requires_grad = False
                                frozen_params += param.numel()
                        else:
                            # Count trainable expert params
                            for param in expert.parameters():
                                trainable_params += param.numel()
                    
                    if layer_idx == 0:
                        logger.info(f"  Layer {layer_idx}: {num_experts} experts found, training first {model_args.num_experts_to_train}")
        
        logger.info(f"  Frozen expert parameters: {frozen_params:,}")
        logger.info(f"  Trainable expert parameters: {trainable_params:,}")
        logger.info(f"  Memory saved: ~{frozen_params * 2 / 1e9:.2f} GB (bf16)")
    
    # Apply LoRA (with or without quantization)
    if model_args.use_lora or model_args.use_qlora:
        if model_args.use_qlora:
            logger.info("Preparing model for QLoRA training (4-bit + LoRA)...")
            model = prepare_model_for_kbit_training(model)
        else:
            logger.info("Using LoRA training (bf16, no quantization)...")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=model_args.lora_r, 
            lora_alpha=model_args.lora_alpha, 
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", # Attention
                "gate_proj", "up_proj", "down_proj",    # Experts / MLP
            ],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        
        # Apply DeepSeek-MoE's mixed precision strategy
        # Key: Convert gate and norm modules to float32 for numerical stability and trainability
        logger.info("Applying mixed precision strategy (DeepSeek-MoE style)...")
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

        model.print_trainable_parameters()
    
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
    def tokenize_function(examples):
        # Handle batched processing
        if isinstance(examples, dict):
            # Batched mode: examples is a dict with lists
            batch_size = len(examples[list(examples.keys())[0]])
            questions = []
            responses = []
            
            for i in range(batch_size):
                question = examples["question"][i] if "question" in examples else ""
                
                # Extract response
                response = ""
                if "responses" in examples:
                    resp_data = examples["responses"][i]
                    if isinstance(resp_data, list) and len(resp_data) > 0:
                        first_response = resp_data[0]
                        if isinstance(first_response, dict) and "response" in first_response:
                            response = first_response["response"]
                        else:
                            response = str(first_response)
                    else:
                        response = str(resp_data)
                elif "reference_answer" in examples:
                    response = examples["reference_answer"][i]
                
                questions.append(question)
                responses.append(response)
        else:
            # Single example mode
            questions = [examples.get("question", "")]
            response = ""
            if "responses" in examples and isinstance(examples["responses"], list) and len(examples["responses"]) > 0:
                first_response = examples["responses"][0]
                if isinstance(first_response, dict) and "response" in first_response:
                    response = first_response["response"]
                else:
                    response = str(first_response)
            elif "reference_answer" in examples:
                response = examples["reference_answer"]
            responses = [response]
        
        # Build prompts and tokenize efficiently
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        
        for question, response in zip(questions, responses):
            # Tokenize instruction and response separately (more efficient)
            instruction_text = f"### Instruction:\n{question}\n\n### Response:\n"
            
            # Tokenize instruction part (will be masked in labels)
            instruction_tokens = tokenizer(
                instruction_text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            
            # Tokenize response part (will be used for loss)
            response_tokens = tokenizer(
                response + tokenizer.eos_token,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            
            # Smart truncation: ensure we always have response tokens for training
            # Reserve at least 25% of max_seq_length for response
            max_instruction_length = int(data_args.max_seq_length * 0.75)
            min_response_length = data_args.max_seq_length - max_instruction_length
            
            # Truncate instruction if too long
            if len(instruction_tokens) > max_instruction_length:
                instruction_tokens = instruction_tokens[:max_instruction_length]
            
            # Calculate remaining space for response
            remaining_space = data_args.max_seq_length - len(instruction_tokens)
            
            # Truncate response if needed
            if len(response_tokens) > remaining_space:
                response_tokens = response_tokens[:remaining_space]
            
            # Combine
            input_ids = instruction_tokens + response_tokens
            
            # Create labels: -100 for instruction, actual tokens for response
            labels = [-100] * len(instruction_tokens) + response_tokens[:]
            
            # Pad to max_length
            padding_length = data_args.max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = [1] * len(instruction_tokens + response_tokens) + [0] * padding_length
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    logger.info("Tokenizing train dataset (with progress bar)...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=8,
        desc="Tokenizing Train" # 显示进度条
    )
    
    logger.info("Tokenizing eval dataset (with progress bar)...")
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=4,
        desc="Tokenizing Eval" # 显示进度条
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
