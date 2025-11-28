#!/usr/bin/env python3
"""
MoE Routing Sanity Check
========================
This script verifies that different tasks route to different experts.
Checks for routing collapse where 1-2 experts dominate all tasks.

Usage:
    python scripts/check_routing_sanity.py --model_path models/Llama-3.2-3B-Instruct-MoE-8x
"""

import torch
import argparse
from transformers import AutoTokenizer
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling_llama_moe import LlamaMoEForCausalLM
from configuration_llama_moe import LlamaMoEConfig


# Sample prompts for different tasks
TASK_PROMPTS = {
    "math": [
        "### Instruction:\nSolve this math problem: If John has 5 apples and buys 3 more, how many apples does he have?\n\n### Response:",
        "### Instruction:\nWhat is 15 multiplied by 8?\n\n### Response:",
        "### Instruction:\nCalculate the area of a circle with radius 5.\n\n### Response:",
        "### Instruction:\nSolve for x: 2x + 5 = 15\n\n### Response:",
        "### Instruction:\nWhat is the sum of the first 10 natural numbers?\n\n### Response:",
    ],
    "code": [
        "### Instruction:\nWrite a Python function to calculate the factorial of a number.\n\n### Response:",
        "### Instruction:\nCreate a function that reverses a string in Python.\n\n### Response:",
        "### Instruction:\nWrite a Python code to sort a list of numbers.\n\n### Response:",
        "### Instruction:\nHow do you read a file in Python?\n\n### Response:",
        "### Instruction:\nWrite a function to find the maximum element in a list.\n\n### Response:",
    ],
    "general": [
        "### Instruction:\nWhat are the benefits of regular exercise?\n\n### Response:",
        "### Instruction:\nExplain what photosynthesis is.\n\n### Response:",
        "### Instruction:\nDescribe the water cycle.\n\n### Response:",
        "### Instruction:\nWhat is the capital of France?\n\n### Response:",
        "### Instruction:\nList three healthy breakfast options.\n\n### Response:",
    ],
    "conversation": [
        "### Instruction:\nHello! How are you today?\n\n### Response:",
        "### Instruction:\nCan you help me with a question?\n\n### Response:",
        "### Instruction:\nThank you for your help!\n\n### Response:",
        "### Instruction:\nWhat's your favorite hobby?\n\n### Response:",
        "### Instruction:\nTell me a fun fact.\n\n### Response:",
    ],
}


class RoutingMonitor:
    """Captures expert routing decisions during forward pass"""
    
    def __init__(self, model):
        self.model = model
        self.routing_data = []
        self.hooks = []
        
    def register_hooks(self):
        """Register hooks to capture routing decisions"""
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                hook = layer.mlp.gate.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.hooks.append(hook)
    
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # output is (topk_idx, topk_weight, aux_loss)
            topk_idx, topk_weight, aux_loss = output
            self.routing_data.append({
                'layer': layer_idx,
                'expert_ids': topk_idx.cpu().numpy(),
                'weights': topk_weight.float().cpu().numpy(),  # Convert bfloat16 to float32 first
            })
        return hook
    
    def clear(self):
        """Clear collected routing data"""
        self.routing_data = []
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def analyze_routing(task_name: str, routing_data: List[Dict]) -> Dict:
    """Analyze routing decisions for a task"""
    all_expert_ids = []
    layer_expert_counts = defaultdict(Counter)
    
    for data in routing_data:
        layer = data['layer']
        expert_ids = data['expert_ids'].flatten()
        all_expert_ids.extend(expert_ids.tolist())
        layer_expert_counts[layer].update(expert_ids.tolist())
    
    # Overall expert distribution
    expert_counter = Counter(all_expert_ids)
    
    # Calculate statistics
    total_routings = len(all_expert_ids)
    expert_distribution = {
        expert_id: count / total_routings 
        for expert_id, count in expert_counter.items()
    }
    
    # Check for collapse
    top_2_experts = expert_counter.most_common(2)
    if len(top_2_experts) >= 2:
        top_2_percentage = sum(count for _, count in top_2_experts) / total_routings
    else:
        top_2_percentage = 1.0
    
    return {
        'task': task_name,
        'total_routings': total_routings,
        'expert_distribution': expert_distribution,
        'expert_counter': expert_counter,
        'top_2_percentage': top_2_percentage,
        'layer_counts': dict(layer_expert_counts),
    }


def print_routing_stats(stats: Dict):
    """Pretty print routing statistics"""
    print(f"\n{'='*60}")
    print(f"Task: {stats['task'].upper()}")
    print(f"{'='*60}")
    print(f"Total routing decisions: {stats['total_routings']}")
    print(f"\nExpert Usage Distribution:")
    
    # Sort by expert ID
    expert_dist = stats['expert_distribution']
    for expert_id in sorted(expert_dist.keys()):
        percentage = expert_dist[expert_id] * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"  Expert {expert_id}: {percentage:5.1f}% {bar}")
    
    # Check for collapse
    print(f"\n⚠️  Top 2 experts account for: {stats['top_2_percentage']*100:.1f}% of routings")
    if stats['top_2_percentage'] > 0.7:
        print("  🔴 WARNING: Routing may be collapsing to few experts!")
    elif stats['top_2_percentage'] > 0.5:
        print("  🟡 CAUTION: High concentration in top 2 experts")
    else:
        print("  ✅ GOOD: Routing is well distributed")
    
    # Show activated expert IDs
    top_experts = stats['expert_counter'].most_common(3)
    print(f"\n🎯 Most activated experts: {[f'E{eid}({cnt})' for eid, cnt in top_experts]}")


def compare_tasks(all_stats: List[Dict]):
    """Compare routing patterns across tasks"""
    print(f"\n{'='*80}")
    print("CROSS-TASK ROUTING COMPARISON")
    print(f"{'='*80}")
    
    # Find which experts are dominant for each task
    task_top_experts = {}
    for stats in all_stats:
        top_3 = stats['expert_counter'].most_common(3)
        task_top_experts[stats['task']] = [eid for eid, _ in top_3]
    
    print("\nTop 3 Experts per Task:")
    for task, experts in task_top_experts.items():
        print(f"  {task:15s}: {experts}")
    
    # Check if different tasks use different experts
    unique_top_experts = set()
    for experts in task_top_experts.values():
        unique_top_experts.update(experts[:2])  # Top 2 per task
    
    print(f"\n📊 Unique experts activated across all tasks: {sorted(unique_top_experts)}")
    print(f"   ({len(unique_top_experts)} out of 8 experts)")
    
    if len(unique_top_experts) <= 2:
        print("\n  🔴 CRITICAL: Only 1-2 experts dominate all tasks!")
        print("     → Routing has COLLAPSED. Router is not learning task-specific routing.")
    elif len(unique_top_experts) <= 4:
        print("\n  🟡 WARNING: Limited expert diversity across tasks")
        print("     → Consider increasing load balancing loss or training longer")
    else:
        print("\n  ✅ GOOD: Multiple experts are being used across different tasks")
        print("     → Router is learning task-specific routing patterns")
    
    # Check if tasks have distinct routing patterns
    all_same = len(set(tuple(experts[:2]) for experts in task_top_experts.values())) == 1
    if all_same:
        print("\n  🔴 All tasks route to the SAME experts → No specialization!")
    else:
        print("\n  ✅ Different tasks show different routing patterns → Specialization emerging!")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Check MoE routing sanity')
    parser.add_argument('--model_path', type=str, 
                       default='models/Llama-3.2-3B-Instruct-MoE-8x',
                       help='Path to the MoE model')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on (cuda:0, cuda:4, cpu, etc.)')
    parser.add_argument('--tasks', type=str, nargs='+', 
                       default=['math', 'code', 'general', 'conversation'],
                       help='Tasks to test')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("MoE ROUTING SANITY CHECK")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Tasks: {', '.join(args.tasks)}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"\n❌ ERROR: Model not found at {args.model_path}")
        print("Please provide the correct path to your trained MoE model.")
        return
    
    # Load model
    print(f"\n📥 Loading model on {args.device}...")
    try:
        # Set the specific GPU before loading
        if args.device.startswith('cuda'):
            device_id = args.device.split(':')[1] if ':' in args.device else '0'
            torch.cuda.set_device(int(device_id))
        
        model = LlamaMoEForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map=args.device,
        )
        model.eval()
        print(f"✅ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load tokenizer
    print("📥 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Setup routing monitor
    monitor = RoutingMonitor(model)
    monitor.register_hooks()
    
    # Test each task
    all_stats = []
    for task_name in args.tasks:
        if task_name not in TASK_PROMPTS:
            print(f"\n⚠️  Unknown task: {task_name}, skipping...")
            continue
        
        print(f"\n🔍 Testing task: {task_name.upper()}")
        prompts = TASK_PROMPTS[task_name]

        monitor.clear()
        
        # Process each prompt
        for i, prompt in enumerate(prompts, 1):
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            
            # Forward pass (routing data captured by hooks)
            # Disable cache to avoid past_key_values issues
            outputs = model(**inputs, use_cache=False)
            
            print(f"  Sample {i}/{len(prompts)}: {len(monitor.routing_data)} routing decisions captured")
        
        # Analyze routing for this task
        stats = analyze_routing(task_name, monitor.routing_data)
        all_stats.append(stats)
        print_routing_stats(stats)
    
    # Compare across tasks
    if len(all_stats) > 1:
        compare_tasks(all_stats)
    
    # Cleanup
    monitor.remove_hooks()
    
    print(f"\n{'='*80}")
    print("✅ Routing sanity check complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

