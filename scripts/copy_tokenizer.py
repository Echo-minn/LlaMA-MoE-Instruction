#!/usr/bin/env python3
"""
Helper script to download or copy tokenizer files
Supports both local copy and HuggingFace download
Useful when converting models and tokenizer files are missing
"""

import os
import sys
import shutil
from pathlib import Path

def copy_tokenizer_from_local(source_model_path, target_model_path):
    """Copy tokenizer files from local source to target"""
    
    # Tokenizer files to copy
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",  # For sentencepiece tokenizers
    ]
    
    source_path = Path(source_model_path)
    target_path = Path(target_model_path)
    
    # Check if source exists
    if not source_path.exists():
        return False, f"Source path does not exist: {source_path}"
    
    # Create target directory if needed
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📋 Copying tokenizer files:")
    print(f"   From: {source_path}")
    print(f"   To:   {target_path}")
    print()
    
    copied_count = 0
    for filename in tokenizer_files:
        source_file = source_path / filename
        target_file = target_path / filename
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                print(f"   ✅ Copied: {filename}")
                copied_count += 1
            except Exception as e:
                print(f"   ❌ Failed to copy {filename}: {e}")
        else:
            print(f"   ⊘  Not found: {filename}")
    
    print()
    if copied_count > 0:
        return True, f"Successfully copied {copied_count} tokenizer file(s)"
    else:
        return False, "No tokenizer files were copied"

def download_tokenizer_from_hf(source_model_name, target_model_path):
    """Download tokenizer from HuggingFace"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return False, "transformers library not installed"
    
    target_path = Path(target_model_path)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading tokenizer from HuggingFace:")
    print(f"   From: {source_model_name}")
    print(f"   To:   {target_path}")
    print()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source_model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(target_path)
        print(f"   ✅ Downloaded and saved tokenizer")
        print(f"   Vocab size: {len(tokenizer)}")
        return True, "Successfully downloaded tokenizer"
    except Exception as e:
        return False, f"Failed to download tokenizer: {e}"

def copy_tokenizer(source, target):
    """Smart tokenizer copy - tries local copy first, then HuggingFace download"""
    
    # Check if source looks like a HuggingFace model name
    is_hf_name = "/" in source and not os.path.exists(source)
    
    if is_hf_name:
        # Download from HuggingFace
        success, message = download_tokenizer_from_hf(source, target)
    else:
        # Try local copy first
        success, message = copy_tokenizer_from_local(source, target)
        
        # If local copy failed and source looks like it could be a model name, try HF
        if not success and not os.path.exists(source):
            print(f"\n⚠️  Local copy failed: {message}")
            print(f"   Trying to download from HuggingFace as '{source}'...")
            success, message = download_tokenizer_from_hf(source, target)
    
    print()
    if success:
        print(f"✅ {message}")
        return True
    else:
        print(f"❌ {message}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 copy_tokenizer.py <source> <target_model_path>")
        print()
        print("Source can be:")
        print("  - Local model path: models/Llama-3.1-8B")
        print("  - HuggingFace model: meta-llama/Llama-3.2-3B-Instruct")
        print()
        print("Examples:")
        print("  # Copy from local model")
        print("  python3 copy_tokenizer.py models/Llama-3.1-8B models/Llama-3.2-3B-Instruct-MoE-8x")
        print()
        print("  # Download from HuggingFace")
        print("  python3 copy_tokenizer.py meta-llama/Llama-3.2-3B-Instruct models/Llama-3.2-3B-Instruct-MoE-8x")
        print()
        sys.exit(1)
    
    source = sys.argv[1]
    target_path = sys.argv[2]
    
    success = copy_tokenizer(source, target_path)
    
    if success:
        print()
        print("🎉 Done! You can now use the model for training.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

