#!/usr/bin/env python3
"""
Inference script for Llama MoE models
Supports both base MoE models and fine-tuned checkpoints
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from src.configuration_llama_moe import LlamaMoEConfig
from src.modeling_llama_moe import LlamaMoEForCausalLM

# Register MoE model
AutoConfig.register("llama_moe", LlamaMoEConfig)
AutoModelForCausalLM.register(LlamaMoEConfig, LlamaMoEForCausalLM)

def find_best_checkpoint(output_dir):
    """Find the latest/best checkpoint in output directory"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    # Look for checkpoint directories
    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append((step, item))
            except:
                continue
    
    if checkpoints:
        # Return the latest checkpoint
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return str(checkpoints[0][1])
    
    # Check if output_dir itself is a valid model
    if (output_path / "config.json").exists():
        return str(output_path)
    
    return None

def load_model(model_path, device="cuda", load_in_4bit=False):
    """Load model and tokenizer"""
    
    print(f"📂 Loading model from: {model_path}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"   ✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    except Exception as e:
        print(f"   ❌ Failed to load tokenizer: {e}")
        return None, None
    
    # Prepare model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto" if device == "cuda" else "cpu",
    }
    
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"   Using 4-bit quantization")
    
    # Load model
    try:
        # Check if this is a PEFT checkpoint
        is_peft = (Path(model_path) / "adapter_config.json").exists()
        
        if is_peft:
            # Load PEFT model
            from peft import PeftModel, AutoPeftModelForCausalLM
            
            print(f"   Detected PEFT/LoRA checkpoint")
            try:
                # Try AutoPeft first
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                print(f"   ✅ Model loaded (PEFT/LoRA with AutoPeft)")
            except Exception as e:
                print(f"   AutoPeft failed: {e}")
                print(f"   Trying manual PEFT loading...")
                
                # Load base model first, then adapter
                # Get base model path from adapter config
                import json
                adapter_config_path = Path(model_path) / "adapter_config.json"
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path", "models/Llama-3.2-3B-Instruct-MoE-8x")
                
                print(f"   Loading base model: {base_model_path}")
                base_model = LlamaMoEForCausalLM.from_pretrained(
                    base_model_path,
                    **model_kwargs
                )
                
                print(f"   Loading adapter from: {model_path}")
                model = PeftModel.from_pretrained(base_model, model_path)
                print(f"   ✅ Model loaded (PEFT/LoRA manual)")
        else:
            # Load base MoE model - try AutoModel first for better compatibility
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                print(f"   ✅ Model loaded (Base MoE model via AutoModel)")
            except Exception as e:
                print(f"   AutoModel failed: {e}, trying LlamaMoEForCausalLM...")
                model = LlamaMoEForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                print(f"   ✅ Model loaded (Base MoE model via LlamaMoEForCausalLM)")
        
        model.eval()
        
        # Ensure model has generate method
        if not hasattr(model, 'generate'):
            print(f"   ⚠️  Warning: Model doesn't have generate method, adding it...")
            from transformers import GenerationMixin
            # This should not happen, but just in case
        
        return model, tokenizer
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate response for a given prompt"""
    
    # Format prompt (using Llama-3 Instruct format)
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    return response

def interactive_mode(model, tokenizer):
    """Interactive chat mode"""
    print()
    print("=" * 80)
    print("🤖 Interactive Mode - Type 'quit' or 'exit' to stop")
    print("=" * 80)
    print()
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not prompt:
                continue
            
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, prompt)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference with Llama MoE model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model or checkpoint (default: auto-detect)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 Llama MoE Inference")
    print("=" * 80)
    print()
    
    # Auto-detect model path if not provided
    if args.model_path is None:
        # Try to find trained model
        possible_paths = [
            "outputs/validation-mixed-sft",
            "outputs/llama-3b-moe-mixed-sft",
            "models/Llama-3.2-3B-Instruct-MoE-8x",
        ]
        
        print("🔍 Auto-detecting model path...")
        for path in possible_paths:
            checkpoint = find_best_checkpoint(path)
            if checkpoint:
                args.model_path = checkpoint
                print(f"   Found: {args.model_path}")
                break
            elif Path(path).exists() and (Path(path) / "config.json").exists():
                args.model_path = path
                print(f"   Found: {args.model_path}")
                break
        
        if args.model_path is None:
            print("   ❌ No model found!")
            print()
            print("Please specify a model path with --model_path")
            print("Example:")
            print("  python3 src/inference.py --model_path outputs/validation-mixed-sft")
            sys.exit(1)
    
    print()
    
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        load_in_4bit=args.load_in_4bit
    )
    
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    print()
    print("✅ Model ready!")
    print()
    
    # Single prompt mode or interactive mode
    if args.prompt:
        print(f"Prompt: {args.prompt}")
        print()
        print("Response:")
        response = generate_response(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(response)
    else:
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()

