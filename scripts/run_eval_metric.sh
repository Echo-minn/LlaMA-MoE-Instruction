#!/bin/bash

# Post-training evaluation script for Stage 2A
# Evaluates the trained model on all 4 tasks with task-specific metrics
#
# Usage:
#   bash scripts/run_eval_stage2A.sh outputs/llama-3b-moe-stage2A/checkpoint-3000
#   bash scripts/run_eval_stage2A.sh outputs/llama-3b-moe-stage2A/checkpoint-3000 --task first
#   bash scripts/run_eval_stage2A.sh outputs/llama-3b-moe-stage2A/checkpoint-3000 --task math --show_samples
#   bash scripts/run_eval_stage2A.sh outputs/llama-3b-moe-stage2A/checkpoint-3000 --max_samples_per_task 100

set -euo pipefail

echo "=========================================="
echo "Stage 2A: Post-Training Evaluation"
echo "=========================================="
echo ""

MODEL_PATH=""
if [ -n "${1:-}" ]; then
    MODEL_PATH="$1"
else
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/llama-3b-moe-stage2A}"
    LATEST_CHECKPOINT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        MODEL_PATH="$LATEST_CHECKPOINT"
        echo "📂 Auto-detected latest checkpoint:"
        echo "   $MODEL_PATH"
    else
        echo "❌ No checkpoint found in $OUTPUT_DIR"
        echo "   Provide an explicit path, e.g.:"
        echo "   bash scripts/run_eval_stage2A.sh outputs/llama-3b-moe-stage2A/checkpoint-3000"
        exit 1
    fi
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Checkpoint not found: $MODEL_PATH"
    exit 1
fi

# GPU configuration (edit as needed)
GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=warning

# Evaluation config
DATA_CONFIG="configs/data_task_stage2A.yaml"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-}"
BATCH_SIZE="${BATCH_SIZE:-18}"

# Collect all remaining arguments (everything after MODEL_PATH)
shift  # Remove MODEL_PATH from arguments
EXTRA_ARGS=("$@")

echo ""
echo "📋 Evaluation Configuration:"
echo "   Model:        $MODEL_PATH"
echo "   Data config:  $DATA_CONFIG"
echo "   GPU:          $GPU_IDS"
echo "   Batch size:   $BATCH_SIZE"
if [ -n "$MAX_SAMPLES_PER_TASK" ]; then
    echo "   Max samples:  $MAX_SAMPLES_PER_TASK per task"
else
    echo "   Max samples:  All available"
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "   Extra args:   ${EXTRA_ARGS[*]}"
fi
echo ""

# Build command
CMD="python scripts/evaluate_metric.py"
CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --data_config \"$DATA_CONFIG\""
CMD="$CMD --batch_size $BATCH_SIZE"
if [ -n "$MAX_SAMPLES_PER_TASK" ]; then
    CMD="$CMD --max_samples_per_task $MAX_SAMPLES_PER_TASK"
fi
# Add any extra arguments (pass them through to Python script)
for arg in "${EXTRA_ARGS[@]}"; do
    CMD="$CMD \"$arg\""
done

echo "🚀 Running evaluation..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation Complete!"
else
    echo "❌ Evaluation Failed!"
    echo "Exit code: $EXIT_CODE"
fi
echo "=========================================="
echo ""

exit $EXIT_CODE

