#!/usr/bin/env python3
"""
Convert Llama-3.2-3B-Instruct to MoE with 8 experts
Based on Sparse Upcycling method
"""

import sys
import os
import torch
import copy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configuration_llama_moe import LlamaMoEConfig
from src.modeling_llama_moe import LlamaMoEForCausalLM

def download_model(model_name):
    """Download model if not exists"""
    from huggingface_hub import snapshot_download
    
    print(f"📥 Downloading {model_name}...")
    try:
        cache_dir = snapshot_download(
            repo_id=model_name,
            cache_dir="models/",
            resume_download=True
        )
        print(f"✅ Downloaded to: {cache_dir}")
        return cache_dir
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def upcycle_llama_3b():
    """Convert Llama-3.2-3B-Instruct to MoE"""
    
    # Configuration
    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    local_path = "models/Llama-3.2-3B-Instruct"
    save_path = "models/Llama-3.2-3B-Instruct-MoE-8x"
    
    num_experts = 8
    experts_per_tok = 2
    
    print("=" * 80)
    print("Llama-3.2-3B-Instruct → MoE Upcycling")
    print("=" * 80)
    print(f"Base model:      {base_model_name}")
    print(f"Num experts:     {num_experts}")
    print(f"Experts per tok: {experts_per_tok}")
    print(f"Output path:     {save_path}")
    print("=" * 80)
    print()
    
    # 1. Load base model
    print("📂 Step 1: Loading base model...")
    
    # Try local path first
    if os.path.exists(local_path):
        model_path = local_path
        print(f"   Using local model: {local_path}")
    else:
        print(f"   Local model not found, trying HuggingFace...")
        model_path = base_model_name
    
    try:
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print(f"   ✅ Loaded model: {llama_model.config.hidden_size}d hidden size")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        print(f"   Please download the model first:")
        print(f"   huggingface-cli download {base_model_name} --local-dir {local_path}")
        return False
    
    # Also load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"   ✅ Loaded tokenizer: vocab size {len(tokenizer)}")
    except Exception as e:
        print(f"   ⚠️  Tokenizer loading failed: {e}")
        tokenizer = None
    
    # 2. Create MoE Config
    print()
    print("⚙️  Step 2: Configuring MoE architecture...")
    
    moe_config = LlamaMoEConfig.from_pretrained(model_path)
    
    # MoE specific settings
    moe_config.n_routed_experts = num_experts
    moe_config.num_experts_per_tok = experts_per_tok
    
    # For 3B model, adjust expert size
    # Original FFN intermediate_size for 3B is typically 8192
    # We reduce it for each expert to save memory
    original_intermediate = moe_config.intermediate_size
    moe_config.moe_intermediate_size = original_intermediate // 2  # 4096 for 3B
    
    print(f"   Original model config:")
    print(f"     Hidden size:           {moe_config.hidden_size}")
    print(f"     Num layers:            {moe_config.num_hidden_layers}")
    print(f"     Intermediate size:     {original_intermediate}")
    print(f"     Attention heads:       {moe_config.num_attention_heads}")
    print()
    print(f"   MoE config:")
    print(f"     Num experts:           {moe_config.n_routed_experts}")
    print(f"     Experts per token:     {moe_config.num_experts_per_tok}")
    print(f"     Expert intermediate:   {moe_config.moe_intermediate_size}")
    print(f"     Total expert params:   ~{moe_config.num_hidden_layers * num_experts * moe_config.moe_intermediate_size * moe_config.hidden_size * 3 / 1e9:.2f}B")
    
    # 3. Create MoE model
    print()
    print("🏗️  Step 3: Building MoE model structure...")
    
    moe_model = LlamaMoEForCausalLM(moe_config)
    moe_model.to(torch.bfloat16)
    
    total_params = sum(p.numel() for p in moe_model.parameters())
    print(f"   ✅ MoE model created: {total_params / 1e9:.2f}B parameters")
    
    # 4. Transplant weights (Sparse Upcycling)
    print()
    print("🔄 Step 4: Transplanting weights (Sparse Upcycling)...")
    print("   This will copy base model weights to all experts...")
    
    llama_sd = llama_model.state_dict()
    moe_sd = moe_model.state_dict()
    
    copied_keys = 0
    skipped_keys = 0
    new_keys = 0
    
    # Progress bar
    pbar = tqdm(list(moe_sd.keys()), desc="   Copying weights", ncols=100)
    
    with torch.no_grad():
        for key in pbar:
            # Case A: Router/Gate weights (newly added) -> Keep random init
            if "mlp.gate.weight" in key:
                new_keys += 1
                pbar.set_postfix({"status": "new (router)"})
                continue
            
            # Case B: Expert weights -> Broadcast from original FFN
            if "mlp.experts." in key:
                # Key format: model.layers.X.mlp.experts.Y.gate_proj.weight
                # Source:      model.layers.X.mlp.gate_proj.weight
                
                parts = key.split('.')
                layer_idx = parts[2]
                expert_idx = parts[4]
                proj_name = parts[-2]  # gate_proj, up_proj, down_proj
                
                source_key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
                
                if source_key in llama_sd:
                    src_tensor = llama_sd[source_key]
                    target_tensor = moe_sd[key]
                    
                    # Check if slicing needed
                    if src_tensor.shape != target_tensor.shape:
                        if proj_name in ["gate_proj", "up_proj"]:
                            # Slice rows (output dimension)
                            target_tensor.data.copy_(src_tensor.data[:target_tensor.shape[0], :])
                        elif proj_name == "down_proj":
                            # Slice columns (input dimension)
                            target_tensor.data.copy_(src_tensor.data[:, :target_tensor.shape[1]])
                    else:
                        # Dimensions match
                        target_tensor.data.copy_(src_tensor.data)
                    
                    # Add small noise to break symmetry
                    noise = torch.randn_like(target_tensor) * 0.001
                    target_tensor.data.add_(noise)
                    
                    copied_keys += 1
                    pbar.set_postfix({"status": f"expert {expert_idx}/{num_experts}"})
                else:
                    skipped_keys += 1
                    pbar.set_postfix({"status": "skipped (no source)"})
            
            # Case C: Other layers (Attention, Norm, Embeddings) -> Direct copy
            elif key in llama_sd:
                moe_sd[key].data.copy_(llama_sd[key].data)
                copied_keys += 1
                pbar.set_postfix({"status": "copied"})
            else:
                skipped_keys += 1
                pbar.set_postfix({"status": "skipped"})
    
    print()
    print(f"   ✅ Weight transplantation complete:")
    print(f"      Copied:  {copied_keys} keys")
    print(f"      New:     {new_keys} keys (routers)")
    print(f"      Skipped: {skipped_keys} keys")
    
    # 5. Save model
    print()
    print(f"💾 Step 5: Saving MoE model to {save_path}...")
    
    os.makedirs(save_path, exist_ok=True)
    moe_model.save_pretrained(save_path, safe_serialization=True)
    
    # Save tokenizer if loaded
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        print(f"   ✅ Tokenizer saved")
    
    # Save config
    moe_config.save_pretrained(save_path)
    print(f"   ✅ Config saved")
    
    print()
    print("=" * 80)
    print("✅ Upcycling Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"  1. Verify model: ls -lh {save_path}")
    print(f"  2. Start training with: bash scripts/run_sft_moe.sh")
    print()
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Llama-3.2-3B-Instruct to MoE")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--experts_per_tok", type=int, default=2, help="Experts per token")
    
    args = parser.parse_args()
    
    success = upcycle_llama_3b()
    
    if success:
        print("🎉 Success! Ready for fine-tuning.")
        sys.exit(0)
    else:
        print("❌ Conversion failed. Please check errors above.")
        sys.exit(1)

