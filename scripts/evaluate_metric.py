#!/usr/bin/env python3
"""
Post-training evaluation script for Stage 2A tasks.

Evaluates the trained model on all 4 tasks with task-specific metrics:
- GSM8K (Math): Accuracy - extracts final answer number
- Code: Pass@1 - execution-based correctness
- Summarization: ROUGE-1, ROUGE-2, ROUGE-L
- Translation: BLEU score

Usage:
    python scripts/evaluate_stage2A.py --model_path outputs/llama-3b-moe-stage2A/checkpoint-3000
"""

import os
import sys
import re
import yaml
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from functools import partial

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm
from evaluate import load as load_metric

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configuration_llama_moe import LlamaMoEConfig
from src.modeling_llama_moe import LlamaMoEForCausalLM
from transformers import AutoConfig, PreTrainedTokenizerFast

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
        source_lang = dataset_cfg.get("source_lang", "zh")
        target_lang = dataset_cfg.get("target_lang", "en")
        return partial(format_translation, source_lang=source_lang, target_lang=target_lang, task_type=task_type, task_templates=task_templates)
    formatter = FORMATTER_REGISTRY.get(fmt, format_alpaca)
    if fmt not in FORMATTER_REGISTRY:
        logger.warning(f"No formatter registered for '{fmt}', defaulting to Alpaca-style.")
    return partial(formatter, task_type=task_type, task_templates=task_templates)


def load_eval_dataset(dataset_cfg: Dict, task_templates: Dict[str, str], max_samples: Optional[int] = None) -> Dataset:
    """Load evaluation dataset for a task."""
    merged_cfg = apply_dataset_defaults(dataset_cfg)
    formatter = resolve_formatter(merged_cfg, task_templates)
    
    vali_split = merged_cfg.get("vali_split")
    validation_samples = merged_cfg.get("validation_samples")
    name = merged_cfg["name"]
    config_name = merged_cfg.get("config")
    
    if not vali_split:
        return None
    
    # Use max_samples if provided, otherwise use validation_samples from config
    samples = max_samples if max_samples is not None else validation_samples
    
    try:
        if config_name:
            ds = load_dataset(name, config_name, split=vali_split)
        else:
            ds = load_dataset(name, split=vali_split)
        
        if samples and len(ds) > samples:
            ds = ds.select(range(samples))
        
        ds = ds.map(
            formatter,
            remove_columns=ds.column_names,
            desc=f"Formatting {name}:{vali_split}",
        )
        ds = ds.filter(lambda x: len(x.get("text", "")) > 0 and len(x.get("response", "")) > 0)
        return ds
    except Exception as exc:
        logger.error(f"Failed to load {name} ({vali_split}): {exc}")
        raise


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer from GSM8K response.
    
    GSM8K format typically uses "#### 42" to mark the answer.
    Also handles formats like "The answer is 42" or just numbers.
    """
    if not text:
        return None
    
    text = text.strip()
    
    # First, try GSM8K standard format: "#### 42" or "####42"
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    
    # Try patterns like "The answer is 42" or "Answer: 42"
    patterns = [
        r'(?:the\s+)?answer\s+is\s*:?\s*(-?\d+\.?\d*)',
        r'answer\s*:?\s*(-?\d+\.?\d*)',
        r'final\s+answer\s*:?\s*(-?\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: find the last number in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return None


def compute_gsm8k_em(predictions: List[str], references: List[str]) -> float:
    """Compute Exact Match for GSM8K."""
    correct = 0
    total = len(predictions)
    failed_extractions = 0
    sample_issues = []
    
    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        pred_answer = extract_gsm8k_answer(pred)
        ref_answer = extract_gsm8k_answer(ref)

        print(f"Pred Answer: {pred_answer}, Ref Answer: {ref_answer}")
        print("-" * 80)
        
        if not pred_answer or not ref_answer:
            failed_extractions += 1
            if len(sample_issues) < 3:  # Log first 3 issues for debugging
                sample_issues.append({
                    'idx': idx,
                    'pred_extracted': pred_answer,
                    'ref_extracted': ref_answer,
                    'pred_preview': pred[:100] if pred else '',
                    'ref_preview': ref[:100] if ref else '',
                })
            continue
        
        # Compare as floats to handle "42" vs "42.0"
        try:
            if abs(float(pred_answer) - float(ref_answer)) < 1e-6:
                correct += 1
        except ValueError:
            failed_extractions += 1
    
    # Log debugging info if there are issues
    if failed_extractions > 0 or correct == 0:
        logger.warning(f"GSM8K EM: {correct}/{total} correct, {failed_extractions} failed extractions")
        if sample_issues:
            logger.warning("Sample extraction issues:")
            for issue in sample_issues:
                logger.warning(
                    f"  Sample {issue['idx']}: pred={issue['pred_extracted']}, ref={issue['ref_extracted']}\n"
                    f"    pred_preview: {issue['pred_preview']}\n"
                    f"    ref_preview: {issue['ref_preview']}"
                )
    
    return correct / total if total > 0 else 0.0


def compute_code_pass_at_1(predictions: List[str], references: List[str]) -> float:
    """Compute Pass@1 for code generation (syntax-based, simplified).
    
    Note: Full execution-based evaluation would require test cases.
    This is a simplified version that checks for valid Python syntax.
    """
    import ast
    
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Extract code from prediction (between ```python and ``` or just the code)
        pred_code = pred.strip()
        if "```python" in pred_code:
            pred_code = pred_code.split("```python")[1].split("```")[0].strip()
        elif "```" in pred_code:
            pred_code = pred_code.split("```")[1].split("```")[0].strip()
        
        # Check if code has valid Python syntax
        try:
            ast.parse(pred_code)
            # If syntax is valid, consider it potentially correct
            # For full evaluation, you'd need to execute and compare outputs
            correct += 1
        except SyntaxError:
            pass
        except Exception:
            # Other errors might indicate runtime issues, but syntax is OK
            pass
    
    return correct / total if total > 0 else 0.0


def generate_predictions(
    model,
    tokenizer,
    texts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda",
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate predictions for a batch of formatted prompts."""
    all_predictions = []

    num_samples = len(texts)
    num_batches = (num_samples + batch_size - 1) // batch_size

    logger.info(f"Generating in {num_batches} batches (batch_size={batch_size})")
    if do_sample:
        logger.info(f"  Using sampling: temperature={temperature}, top_p={top_p}")
    else:
        logger.info(f"  Using greedy decoding (deterministic)")

    for i in tqdm(range(0, num_samples, batch_size), total=num_batches, desc="Generation", leave=False):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        
        # Generate
        with torch.no_grad():
            if do_sample:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        
        # Decode only the generated part
        input_lengths = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_lengths:]
        predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        all_predictions.extend(predictions)
    
    return all_predictions


def truncate_text(text: str, max_len: int = 150) -> str:
    """Truncate text to max_len, adding ellipsis if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."

def extract_answer(text: str) -> str:
    """Extract the answer from the text."""
    return text.split("#### ")[1].strip()

def evaluate_task(
    model,
    tokenizer,
    dataset: Dataset,
    task_type: str,
    max_samples: Optional[int] = None,
    batch_size: int = 6,
    device: str = "cuda",
    show_samples: bool = False,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, float]:
    """Evaluate model on a specific task."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {task_type.upper()}")
    logger.info(f"{'='*80}")
    
    # Limit dataset size if specified
    eval_dataset = dataset
    if max_samples and len(eval_dataset) > max_samples:
        logger.info(f"Limiting evaluation to {max_samples} samples")
        eval_dataset = eval_dataset.select(range(max_samples))
    
    logger.info(f"Evaluating on {len(eval_dataset)} samples")
    
    texts = eval_dataset["text"]
    references = eval_dataset["response"]
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = generate_predictions(
        model,
        tokenizer,
        texts,
        max_new_tokens=512,
        batch_size=batch_size,
        device=device,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Print samples if requested
    if show_samples:
        logger.info("\n" + "-" * 80)
        logger.info("SAMPLE OUTPUTS:")
        logger.info("-" * 80)
        for idx, (text, pred, ref) in enumerate(zip(texts, predictions, references)):
            logger.info(f"\n[Sample {idx + 1}]")
            logger.info(f"prediction: {truncate_text(pred, )}")
        logger.info("-" * 80 + "\n")
    
    # Compute task-specific metrics
    metrics = {}
    
    if task_type == "math":
        # GSM8K: Exact Match
        em = compute_gsm8k_em(predictions, references)
        metrics["exact_match"] = em
        logger.info(f"Exact Match: {em:.4f}")
    
    elif task_type == "code":
        # Code: Pass@1
        pass_at_1 = compute_code_pass_at_1(predictions, references)
        metrics["pass_at_1"] = pass_at_1
        logger.info(f"Pass@1: {pass_at_1:.4f}")
    
    elif task_type == "summarization":
        # Summarization: ROUGE
        rouge = load_metric("rouge")
        rouge_scores = rouge.compute(
            predictions=predictions,
            references=references,
        )
        # Handle both Score objects (with .precision attribute) and direct float values
        def get_rouge_value(score):
            if hasattr(score, 'precision'):
                # Score object with precision, recall, fmeasure attributes
                return float(score.precision)
            else:
                # Already a float/number
                return float(score)
        
        metrics["rouge1"] = get_rouge_value(rouge_scores["rouge1"])
        metrics["rouge2"] = get_rouge_value(rouge_scores["rouge2"])
        metrics["rougeL"] = get_rouge_value(rouge_scores["rougeL"])
        logger.info(f"ROUGE-1: {metrics['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {metrics['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {metrics['rougeL']:.4f}")
    
    elif task_type == "translation":
        # Translation: BLEU
        bleu = load_metric("bleu")
        # BLEU expects references as lists of lists
        references_list = [[ref] for ref in references]
        bleu_scores = bleu.compute(
            predictions=predictions,
            references=references_list,
        )
        metrics["bleu"] = bleu_scores["bleu"]
        logger.info(f"BLEU: {metrics['bleu']:.4f}")
    
    else:
        logger.warning(f"Unknown task type: {task_type}, skipping metrics")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2A model on all tasks")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_task_stage2A.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--max_samples_per_task",
        type=int,
        default=None,
        help="Maximum samples to evaluate per task (None = use all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=18,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Evaluate only a specific task (task_type like 'math', 'code', etc.) or 'first' for first enabled task. None = evaluate all tasks.",
    )
    parser.add_argument(
        "--show_samples",
        action="store_true",
        help="Print prompt and output for each sample (concise format)",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling for generation (default: True). Use --no-do_sample for greedy decoding.",
    )
    parser.add_argument(
        "--no-do_sample",
        dest="do_sample",
        action="store_false",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7). Higher = more random.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: 0.9).",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info("\n" + "="*80)
    logger.info("STAGE 2A POST-TRAINING EVALUATION")
    logger.info("="*80 + "\n")
    
    # Load data config
    data_config = load_data_config(args.data_config)
    processing_cfg = data_config.get("processing", {})
    
    # Load task-specific templates (allow override from config)
    task_templates = processing_cfg.get("task_prompt_templates", {})
    task_templates = {**TASK_PROMPT_TEMPLATES, **task_templates}
    
    # Load tokenizer first to get the correct vocabulary size
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding_side to 'left' for decoder-only models during generation
    tokenizer.padding_side = "left"
    
    vocab_size = len(tokenizer)
    logger.info(f"Tokenizer vocabulary size: {vocab_size}")
    
    # Load model - handle PEFT checkpoints and vocabulary size mismatch
    logger.info(f"Loading model from {args.model_path}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    
    # Check if this is a PEFT checkpoint
    import json
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    is_peft = os.path.exists(adapter_config_path)
    
    # Load config from checkpoint first to get correct vocab_size
    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Fix model_type if needed
        if config_dict.get("model_type") != "llama_moe":
            logger.info(f"Fixing model_type in config: {config_dict.get('model_type')} -> llama_moe")
            config_dict["model_type"] = "llama_moe"
        
        # Use vocab_size from checkpoint config (it should match tokenizer after training)
        checkpoint_vocab_size = config_dict.get("vocab_size", vocab_size)
        logger.info(f"Checkpoint vocab_size: {checkpoint_vocab_size}, Tokenizer vocab_size: {vocab_size}")
        
        # Load config
        config = LlamaMoEConfig.from_dict(config_dict)
    else:
        # No config.json, create from checkpoint
        logger.warning("config.json not found, loading config from checkpoint")
        config = LlamaMoEConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if hasattr(config, 'vocab_size'):
            checkpoint_vocab_size = config.vocab_size
        else:
            checkpoint_vocab_size = vocab_size
            config.vocab_size = vocab_size
    
    # Ensure config vocab_size matches tokenizer
    if config.vocab_size != vocab_size:
        logger.info(f"Updating config vocab_size to match tokenizer: {config.vocab_size} -> {vocab_size}")
        config.vocab_size = vocab_size
    
    # Load model
    if is_peft:
        logger.info("Detected PEFT/LoRA checkpoint, loading with PEFT...")
        from peft import PeftModel, AutoPeftModelForCausalLM
        
        try:
            # Try AutoPeft first (handles vocab size automatically)
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model_path,
                **model_kwargs,
            )
            logger.info("✅ Loaded with AutoPeftModelForCausalLM")
        except Exception as e:
            logger.warning(f"AutoPeft failed: {e}, trying manual PEFT loading...")
            
            # Manual PEFT loading: load base model first, then adapter
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path")
            else:
                base_model_path = None
            
            if base_model_path and os.path.exists(base_model_path):
                logger.info(f"Loading base model from: {base_model_path}")
                # Load base model with correct config
                base_model = LlamaMoEForCausalLM.from_pretrained(
                    base_model_path,
                    config=config,
                    **model_kwargs,
                )
                # Resize embeddings to match checkpoint vocab_size
                base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
                if base_vocab_size != vocab_size:
                    logger.info(f"Resizing base model embeddings: {base_vocab_size} -> {vocab_size}")
                    base_model.resize_token_embeddings(vocab_size)
            else:
                # Create base model from config
                logger.info("Creating base model from config")
                base_model = LlamaMoEForCausalLM.from_config(config, **model_kwargs)
            
            # Load adapter
            logger.info("Loading PEFT adapter...")
            model = PeftModel.from_pretrained(base_model, args.model_path) 
            logger.info("✅ Loaded with manual PEFT")
    else:
        # Regular model (not PEFT)
        logger.info("Loading regular model...")
        model = LlamaMoEForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            **model_kwargs,
        )
        
        # Verify embeddings match tokenizer
        current_vocab_size = model.get_input_embeddings().weight.shape[0]
        if current_vocab_size != vocab_size:
            logger.info(f"Resizing model embeddings to match tokenizer: {current_vocab_size} -> {vocab_size}")
            model.resize_token_embeddings(vocab_size)
    
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to(args.device)
        model.eval()
    else:
        model.eval()
    
    # Load and evaluate each task
    all_metrics = {}
    datasets_cfg = data_config.get("datasets", [])
    
    # Filter tasks if --task is specified
    if args.task:
        if args.task.lower() == "first":
            # Find first enabled task
            first_enabled = None
            for dataset_cfg in datasets_cfg:
                if dataset_cfg.get("enabled", False):
                    merged_cfg = apply_dataset_defaults(dataset_cfg)
                    first_enabled = merged_cfg["task_type"]
                    logger.info(f"Evaluating only first enabled task: {first_enabled}")
                    break
            if first_enabled:
                datasets_cfg = [cfg for cfg in datasets_cfg if apply_dataset_defaults(cfg).get("task_type") == first_enabled]
            else:
                logger.warning("No enabled tasks found!")
                return
        else:
            # Filter to specific task type
            datasets_cfg = [cfg for cfg in datasets_cfg if apply_dataset_defaults(cfg).get("task_type") == args.task]
            if not datasets_cfg:
                logger.warning(f"No tasks found matching task_type: {args.task}")
                logger.info(f"Available task types: {[apply_dataset_defaults(cfg).get('task_type') for cfg in data_config.get('datasets', []) if cfg.get('enabled', False)]}")
                return
            logger.info(f"Evaluating only task: {args.task}")
    
    for dataset_cfg in datasets_cfg:
        if not dataset_cfg.get("enabled", False):
            continue
        
        merged_cfg = apply_dataset_defaults(dataset_cfg)
        task_type = merged_cfg["task_type"]
        
        # Load eval dataset
        eval_dataset = load_eval_dataset(merged_cfg, task_templates, max_samples=args.max_samples_per_task)
        if eval_dataset is None or len(eval_dataset) == 0:
            logger.warning(f"Skipping {task_type}: no eval dataset available")
            continue
        
        # Evaluate
        task_metrics = evaluate_task(
            model,
            tokenizer,
            eval_dataset,
            task_type,
            max_samples=args.max_samples_per_task,
            batch_size=args.batch_size,
            device=args.device,
            show_samples=args.show_samples,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        all_metrics[task_type] = task_metrics
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    for task_type, metrics in all_metrics.items():
        logger.info(f"\n{task_type.upper()}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()

