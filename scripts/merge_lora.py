#!/usr/bin/env python3
"""
Merge LoRA adapters with base model to create a standalone fine-tuned model
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def merge_lora_to_base(base_model_path, lora_model_path, output_dir):
    """
    Merge LoRA adapters with base model
    
    Args:
        base_model_path: Path to the base Llama model
        lora_model_path: Path to the LoRA checkpoint directory
        output_dir: Path to save the merged model
    """
    print("=" * 80)
    print("Merging LoRA Adapters with Base Model")
    print("=" * 80)
    print(f"\nBase model: {base_model_path}")
    print(f"LoRA adapters: {lora_model_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA model
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # Merge LoRA weights into base model
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ Merge complete!")
    print(f"Merged model saved to: {output_dir}")
    print("\nYou can now use this model like any other HuggingFace model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the base Llama model"
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        required=True,
        help="Path to the LoRA checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the merged model"
    )
    
    args = parser.parse_args()
    
    merge_lora_to_base(args.base_model, args.lora_model, args.output_dir)

if __name__ == "__main__":
    main()

