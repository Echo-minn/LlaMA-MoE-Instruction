#!/usr/bin/env python3
"""
Test script to verify router initialization and expert selection probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

def test_kaiming_uniform_init():
    """Test current initialization (Kaiming uniform)"""
    print("=" * 80)
    print("Testing Kaiming Uniform Initialization")
    print("=" * 80)
    
    n_experts = 8
    hidden_size = 4096
    batch_size = 32
    seq_len = 128
    
    # Initialize router weights with Kaiming uniform (current method)
    weight = torch.empty((n_experts, hidden_size))
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    
    # Create random hidden states (simulating model input)
    hidden_states = torch.randn(batch_size * seq_len, hidden_size)
    
    # Compute logits and probabilities
    logits = F.linear(hidden_states, weight, None)  # (batch*seq, n_experts)
    scores = logits.softmax(dim=-1)  # (batch*seq, n_experts)
    
    # Compute average probability per expert
    avg_probs = scores.mean(dim=0)  # (n_experts,)
    
    print(f"\nRouter weight shape: {weight.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"\nAverage probability per expert:")
    for i, prob in enumerate(avg_probs):
        print(f"  Expert {i}: {prob.item():.6f} (expected: {1/n_experts:.6f})")
    
    print(f"\nStatistics:")
    print(f"  Mean: {avg_probs.mean().item():.6f}")
    print(f"  Std:  {avg_probs.std().item():.6f}")
    print(f"  Min:  {avg_probs.min().item():.6f}")
    print(f"  Max:  {avg_probs.max().item():.6f}")
    print(f"  Range: {(avg_probs.max() - avg_probs.min()).item():.6f}")
    
    # Check if probabilities are uniform
    expected_prob = 1.0 / n_experts
    max_deviation = (avg_probs - expected_prob).abs().max().item()
    print(f"\n  Max deviation from uniform: {max_deviation:.6f}")
    
    if max_deviation < 0.01:
        print("  ✅ Probabilities are approximately uniform")
    else:
        print("  ❌ Probabilities are NOT uniform")
    
    return avg_probs


def test_zero_init():
    """Test zero initialization (should give uniform probabilities)"""
    print("\n" + "=" * 80)
    print("Testing Zero Initialization (should be uniform)")
    print("=" * 80)
    
    n_experts = 8
    hidden_size = 4096
    batch_size = 32
    seq_len = 128
    
    # Initialize router weights to zero
    weight = torch.zeros((n_experts, hidden_size))
    
    # Create random hidden states
    hidden_states = torch.randn(batch_size * seq_len, hidden_size)
    
    # Compute logits and probabilities
    logits = F.linear(hidden_states, weight, None)
    scores = logits.softmax(dim=-1)
    
    # Compute average probability per expert
    avg_probs = scores.mean(dim=0)
    
    print(f"\nAverage probability per expert:")
    for i, prob in enumerate(avg_probs):
        print(f"  Expert {i}: {prob.item():.6f} (expected: {1/n_experts:.6f})")
    
    print(f"\n  Max deviation from uniform: {(avg_probs - 1/n_experts).abs().max().item():.6f}")
    
    return avg_probs


def test_small_random_init():
    """Test small random initialization"""
    print("\n" + "=" * 80)
    print("Testing Small Random Initialization")
    print("=" * 80)
    
    n_experts = 8
    hidden_size = 4096
    batch_size = 32
    seq_len = 128
    
    # Initialize with very small random values
    weight = torch.empty((n_experts, hidden_size))
    nn.init.normal_(weight, mean=0.0, std=0.001)  # Very small std
    
    hidden_states = torch.randn(batch_size * seq_len, hidden_size)
    logits = F.linear(hidden_states, weight, None)
    scores = logits.softmax(dim=-1)
    avg_probs = scores.mean(dim=0)
    
    print(f"\nAverage probability per expert:")
    for i, prob in enumerate(avg_probs):
        print(f"  Expert {i}: {prob.item():.6f} (expected: {1/n_experts:.6f})")
    
    max_deviation = (avg_probs - 1/n_experts).abs().max().item()
    print(f"\n  Max deviation from uniform: {max_deviation:.6f}")
    
    return avg_probs


if __name__ == "__main__":
    print("\n🔍 Testing Router Initialization Methods\n")
    
    # Test current method
    kaiming_probs = test_kaiming_uniform_init()
    
    # Test zero init (baseline)
    zero_probs = test_zero_init()
    
    # Test small random init
    small_probs = test_small_random_init()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nCurrent Kaiming Uniform initialization:")
    print(f"  Does NOT guarantee uniform expert probabilities")
    print(f"  Experts will have different selection probabilities initially")
    print(f"  This may lead to imbalanced expert usage at the start of training")
    print("\nRecommendation:")
    print("  Consider initializing router weights to zero or very small values")
    print("  to ensure uniform expert selection probabilities initially.")

