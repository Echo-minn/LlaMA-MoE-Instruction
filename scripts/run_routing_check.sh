#!/bin/bash
#
# Quick runner for routing sanity check
# Usage: bash scripts/run_routing_check.sh [model_path] [gpu_id]
#

# Default model path and GPU
MODEL_PATH="${1:-models/Llama-3.2-3B-Instruct-MoE-8x}"
GPU_ID="${2:-0}"

echo "=========================================="
echo "MoE Routing Sanity Check"
echo "=========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "GPU: cuda:$GPU_ID"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    echo ""
    echo "Usage:"
    echo "  bash scripts/run_routing_check.sh [model_path] [gpu_id]"
    echo ""
    echo "Example:"
    echo "  bash scripts/run_routing_check.sh models/Llama-3.2-3B-Instruct-MoE-8x 4"
    echo "  bash scripts/run_routing_check.sh outputs/llama-3b-moe-mixed-sft/checkpoint-1000 0"
    exit 1
fi

# Run the sanity check
python scripts/check_routing_sanity.py \
    --model_path "$MODEL_PATH" \
    --device "cuda:$GPU_ID" \
    --tasks math code general conversation

echo ""
echo "=========================================="
echo "Check complete!"
echo "=========================================="

