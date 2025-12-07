#!/usr/bin/env python3
"""
Standalone script to verify router initialization in a converted MoE model.
Can be run on an already-converted model to check if routers are initialized correctly.
"""

import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modeling_llama_moe import LlamaMoEForCausalLM


def verify_router_initialization(model_path: str, verbose: bool = True):
    """
    Verify router initialization in a MoE model.
    
    Args:
        model_path: Path to the MoE model
        verbose: Whether to print detailed information
    
    Returns:
        dict: Verification results
    """
    print("=" * 80)
    print("Router Initialization Verification")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print()
    
    # Load model
    print("📂 Loading model...")
    try:
        model = LlamaMoEForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True
        )
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return {"success": False, "error": str(e)}
    
    config = model.config
    n_experts = config.n_routed_experts
    hidden_size = config.hidden_size
    
    print(f"   Model config:")
    print(f"     Num experts: {n_experts}")
    print(f"     Hidden size: {hidden_size}")
    print()
    
    results = {
        "success": True,
        "router_count": 0,
        "routers": [],
        "weight_stats": {},
        "uniformity_test": {}
    }
    
    # Check router weights
    print("🔍 Checking router weight statistics...")
    router_count = 0
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'gate') and hasattr(module.gate, 'weight'):
                router_weight = module.gate.weight.data.float()  # Convert to float for stats
                router_count += 1
                
                weight_mean = router_weight.mean().item()
                weight_std = router_weight.std().item()
                weight_min = router_weight.min().item()
                weight_max = router_weight.max().item()
                
                expected_std = 0.001
                std_ratio = weight_std / expected_std
                std_ok = 0.5 <= std_ratio <= 2.0
                
                router_info = {
                    "name": name,
                    "mean": weight_mean,
                    "std": weight_std,
                    "min": weight_min,
                    "max": weight_max,
                    "std_ok": std_ok,
                    "std_ratio": std_ratio,
                }
                results["routers"].append(router_info)
        
        results["router_count"] = router_count
        
        # Find first MoE layer
        test_layer = None
        test_layer_name = None
        for name, layer in model.named_modules():
            if hasattr(layer, 'gate') and hasattr(layer.gate, 'weight'):
                test_layer = layer
                test_layer_name = name
                break
        
        if test_layer is None:
            print("   ❌ Could not find MoE layer to test")
            results["success"] = False
            return results
        
        gate = test_layer.gate
        n_experts = gate.n_routed_experts
        top_k = gate.top_k
        
        # Create sample hidden states
        batch_size = 8
        seq_len = 32
        sample_hidden = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
        
        # Get router output
        topk_idx, topk_weight, aux_loss = gate(sample_hidden)
        
        # Compute expert selection statistics
        n_tokens = batch_size * seq_len
        
        # Count expert selections
        expert_counts = torch.zeros(n_experts, dtype=torch.long)
        expert_counts.scatter_add_(0, topk_idx.view(-1), torch.ones_like(topk_idx.view(-1)))
        
        # Compute probabilities
        expert_probs = expert_counts.float() / (n_tokens * top_k)
        expected_prob = 1.0 / n_experts
        
        # Compute statistics
        max_deviation = (expert_probs - expected_prob).abs().max().item()
        mean_deviation = (expert_probs - expected_prob).abs().mean().item()
        std_deviation = expert_probs.std().item()
        
        uniformity_ok = max_deviation < 0.01
        
        results["uniformity_test"] = {
            "test_layer": test_layer_name,
            "n_tokens": n_tokens,
            "top_k": top_k,
            "expert_probs": expert_probs.tolist(),
            "expected_prob": expected_prob,
            "max_deviation": max_deviation,
            "mean_deviation": mean_deviation,
            "std_deviation": std_deviation,
            "uniformity_ok": uniformity_ok,
        }
        
        if verbose:
            print(f"   Tested {n_tokens} tokens with top_k={top_k}")
            print(f"   Expert selection probabilities:")
            for i, prob in enumerate(expert_probs):
                deviation = abs(prob.item() - expected_prob)
                marker = "✅" if deviation < 0.01 else "⚠️"
                print(f"     {marker} Expert {i}: {prob.item():.6f} (expected: {expected_prob:.6f}, dev: {deviation:.6f})")
            
            print()
            print(f"   Statistics:")
            print(f"     Max deviation:  {max_deviation:.6f}")
            print(f"     Mean deviation: {mean_deviation:.6f}")
            print(f"     Std deviation:  {std_deviation:.6f}")
            
            if uniformity_ok:
                print(f"   ✅ Router produces approximately uniform expert selection")
            else:
                print(f"   ⚠️  Router may not produce uniform selection")
                print(f"      (Max deviation {max_deviation:.6f} exceeds threshold 0.01)")
    
    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_routers_ok = all(r["std_ok"] for r in results["routers"])
    uniformity_ok = results["uniformity_test"].get("uniformity_ok", False)
    
    if all_routers_ok and uniformity_ok:
        print("✅ All routers are initialized correctly!")
        print("   - Router weights have correct statistics")
        print("   - Router produces uniform expert selection")
        results["success"] = True
    else:
        print("⚠️  Some issues detected:")
        if not all_routers_ok:
            print("   - Some router weights have incorrect statistics")
        if not uniformity_ok:
            print("   - Router does not produce uniform expert selection")
        results["success"] = False
    
    print()
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify router initialization in MoE model")
    parser.add_argument("model_path", type=str, help="Path to MoE model")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    results = verify_router_initialization(args.model_path, verbose=not args.quiet)
    
    sys.exit(0 if results["success"] else 1)

