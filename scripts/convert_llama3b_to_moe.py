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
    save_path = "models/Llama-3.2-3B-Instruct-MoE"
    
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
    print("📂 Step 1: Loading base model from HuggingFace...")
    print(f"   Model: {base_model_name}")
    print(f"   (This will download the model if not cached)")
    
    model_path = base_model_name
    
    try:
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print(f"   ✅ Loaded model: {llama_model.config.hidden_size}d hidden size")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        print(f"   Make sure you have HuggingFace access and valid credentials.")
        print(f"   Run: huggingface-cli login")
        return False
    
    # Load tokenizer from the same source as the model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"   ✅ Loaded tokenizer: vocab size {len(tokenizer)}")
    except Exception as e:
        print(f"   ❌ Failed to load tokenizer: {e}")
        print(f"   The model was loaded but tokenizer failed.")
        print(f"   This should not happen if you downloaded from HuggingFace.")
        return False
    
    # 2. Create MoE Config
    print()
    print("⚙️  Step 2: Configuring MoE architecture...")
    
    moe_config = LlamaMoEConfig.from_pretrained(model_path)
    
    # MoE specific settings
    moe_config.n_routed_experts = num_experts
    moe_config.num_experts_per_tok = experts_per_tok
    
    # Keep the same intermediate size as original model
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
    
    # Explicitly (re)initialize all router weights with small random std for near-uniform routing
    # This is run before any weight transplant; router weights are not copied from base model.
    def _init_router_weights(model, std: float = 1e-4):
        import torch.nn as nn
        router_count = 0
        for module in model.modules():
            gate = getattr(module, "gate", None)
            if gate is not None and hasattr(gate, "weight"):
                nn.init.normal_(gate.weight, mean=0.0, std=std)
                router_count += 1
        return router_count, std
    
    router_count, router_std = _init_router_weights(moe_model, std=1e-4)
    print(f"   ✅ Routers initialized: {router_count} (std={router_std})")
    
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
    
    # 4.5. Verify router initialization
    print()
    print("🔍 Step 4.5: Verifying router initialization...")
    
    router_count = 0
    router_stats = []
    
    with torch.no_grad():
        for name, module in moe_model.named_modules():
            if hasattr(module, 'gate') and hasattr(module.gate, 'weight'):
                router_weight = module.gate.weight.data
                router_count += 1
                
                # Check weight statistics
                weight_mean = router_weight.mean().item()
                weight_std = router_weight.std().item()
                weight_min = router_weight.min().item()
                weight_max = router_weight.max().item()
                
                router_stats.append({
                    'name': name,
                    'mean': weight_mean,
                    'std': weight_std,
                    'min': weight_min,
                    'max': weight_max,
                })
                
        
        # Find first MoE layer to test
        test_layer = None
        for layer in moe_model.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                test_layer = layer.mlp
                break
        
        if test_layer is not None:
            # Create sample hidden states
            batch_size = 4
            seq_len = 16
            hidden_size = moe_config.hidden_size
            sample_hidden = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
            
            # Get router output
            gate = test_layer.gate
            topk_idx, topk_weight, aux_loss = gate(sample_hidden)
            
            # Compute expert selection probabilities
            n_experts = gate.n_routed_experts
            n_tokens = batch_size * seq_len
            
            # Count how many times each expert is selected
            expert_counts = torch.zeros(n_experts, dtype=torch.long)
            expert_counts.scatter_add_(0, topk_idx.view(-1), torch.ones_like(topk_idx.view(-1)))
            
            # Compute average probability per expert
            expert_probs = expert_counts.float() / (n_tokens * gate.top_k)
            expected_prob = 1.0 / n_experts
            
            print(f"      Tested {n_tokens} tokens with {gate.top_k} experts per token")
            print(f"      Expert selection probabilities:")
            for i, prob in enumerate(expert_probs):
                deviation = abs(prob.item() - expected_prob)
                print(f"        Expert {i}: {prob.item():.6f} (expected: {expected_prob:.6f}, dev: {deviation:.6f})")
            
            max_deviation = (expert_probs - expected_prob).abs().max().item()
            print(f"      Max deviation from uniform: {max_deviation:.6f}")
            
            if max_deviation < 0.01:
                print(f"   ✅ Router produces approximately uniform expert selection")
            else:
                print(f"   ⚠️  Router may not produce uniform selection (deviation: {max_deviation:.6f})")
        else:
            print("   ⚠️  Could not find MoE layer to test")
    
    print(f"   ✅ Verified {router_count} router(s)")
    
    # 5. Save model
    print()
    print(f"💾 Step 5: Saving MoE model to {save_path}...")
    
    os.makedirs(save_path, exist_ok=True)
    moe_model.save_pretrained(save_path, safe_serialization=True)
    
    # Save tokenizer
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
    
    # Check if tokenizer files exist
    tokenizer_files_exist = (
        os.path.exists(os.path.join(save_path, "tokenizer.json")) or
        os.path.exists(os.path.join(save_path, "tokenizer_config.json"))
    )
    
    if not tokenizer_files_exist:
        print("⚠️  WARNING: Tokenizer files are missing!")
        print()
        print("This should not happen. To fix, run:")
        print(f"  python3 scripts/copy_tokenizer.py meta-llama/Llama-3.2-3B-Instruct {save_path}")
        print()
    
    print("Next steps:")
    print(f"  1. Verify model: ls -lh {save_path}")
    if not tokenizer_files_exist:
        print(f"  2. Copy tokenizer files (see warning above)")
        print(f"  3. Start training with: bash scripts/run_mixed_sft.sh")
    else:
        print(f"  2. Start training with: bash scripts/run_mixed_sft.sh")
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

