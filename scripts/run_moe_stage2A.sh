#!/bin/bash

# Stage 2A: Task-Grouped MoE Training (Math, Code, Summ, Translation)
# Sequential task cycling with optional freezing of non-MoE adapters
#
# Usage:
#   bash scripts/run_moe_stage2A.sh                        # Auto-detect latest Stage 1.5 ckpt
#   bash scripts/run_moe_stage2A.sh outputs/.../checkpoint-4500
#
# Resume training with different learning rate:
#   RESUME=outputs/llama-3b-moe-stage2A/checkpoint-600 LEARNING_RATE=5e-6 bash scripts/run_moe_stage2A.sh

set -euo pipefail

echo "Stage 2A: MoE Expert Specialization"
echo ""

STAGE1P5_CHECKPOINT=""
if [ -n "${1:-}" ]; then
    STAGE1P5_CHECKPOINT="$1"
else
    STAGE1_OUTPUT="outputs/llama-3b-moe-mixed-sft"
    LATEST_CHECKPOINT=$(ls -d $STAGE1_OUTPUT/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        STAGE1P5_CHECKPOINT="$LATEST_CHECKPOINT"
    else
        echo "❌ No checkpoint in $STAGE1_OUTPUT"
        echo "Usage: bash scripts/run_moe_stage2A.sh <checkpoint_path>"
        exit 1
    fi
fi

if [ ! -d "$STAGE1P5_CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $STAGE1P5_CHECKPOINT"
    exit 1
fi

# GPU configuration (edit as needed)
GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=warning
# Use expandable CUDA allocator segments to reduce fragmentation-related OOMs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Stage 2A config
MODEL_PATH="$STAGE1P5_CHECKPOINT"
DATA_CONFIG="configs/data_task_stage2A.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/llama-3b-moe-stage2A}"
RUN_NAME="${RUN_NAME:-moe-stage2A}"

MAX_STEPS=${MAX_STEPS:-3000}      # 4 tasks × 300 steps × 2 cycles + buffer
STEPS_PER_TASK=${STEPS_PER_TASK:-300}  # Steps per task before switching (for task cycling)
BATCH_SIZE=${BATCH_SIZE:-15}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-6}  # Very small batch for evaluation to avoid OOM (was 4, reduced to 1)
GRAD_ACCUM=${GRAD_ACCUM:-2}
# Increased from 1.5e-6 to 3e-6 for better convergence (loss was not decreasing)
LEARNING_RATE=${LEARNING_RATE:-3e-6}
# Reduced from 0.10 to 0.05 so warmup ends earlier and cosine decay can start properly
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
AUX_LOSS_ALPHA=${AUX_LOSS_ALPHA:-0.01}
SAVE_STEPS=${SAVE_STEPS:-300}
EVAL_STEPS=${EVAL_STEPS:-300}
EVAL_LOSS_ONLY=${EVAL_LOSS_ONLY:-True}  # Only compute eval loss (skip predictions/metrics) to avoid OOM
FREEZE_NON_MOE=${FREEZE_NON_MOE:-True}
GRADIENT_CKPT=${GRADIENT_CKPT:-True}

# Resume from checkpoint (only if explicitly specified)
RESUME_FROM_CHECKPOINT=""
if [ -n "${RESUME:-}" ]; then
    RESUME_FROM_CHECKPOINT="$RESUME"
    if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
        echo "❌ Checkpoint not found: $RESUME_FROM_CHECKPOINT"
        exit 1
    fi
fi

echo "📋 Config: Model=$MODEL_PATH | Output=$OUTPUT_DIR | GPUs=$GPU_IDS"
echo "   LR=$LEARNING_RATE | Warmup=$WARMUP_RATIO | Steps=$MAX_STEPS"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "   Resume: $RESUME_FROM_CHECKPOINT (LR→$LEARNING_RATE)"
fi
echo ""

printf "Continue with Stage 2A training? (y/n) "
read -r REPLY
if [[ "$REPLY" != "y" && "$REPLY" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

# Build resume argument if checkpoint is specified
RESUME_ARG=""
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_ARG="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
fi

echo "🚀 Starting training..."
echo ""

deepspeed --include localhost:$GPU_IDS \
    scripts/train_moe_stage2A.py \
    --deepspeed distributed-training/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    --data_config "$DATA_CONFIG" \
    --steps_per_task $STEPS_PER_TASK \
    --eval_loss_only $EVAL_LOSS_ONLY \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    $RESUME_ARG \
    --bf16 True \
    --use_qlora True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --aux_loss_alpha $AUX_LOSS_ALPHA \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio $WARMUP_RATIO \
    --max_steps $MAX_STEPS \
    --logging_steps 30 \
    --log_level warning \
    --disable_tqdm False \
    --report_to wandb \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_total_limit 3 \
    --gradient_checkpointing $GRADIENT_CKPT \
    --max_seq_length 1024 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --ddp_find_unused_parameters False \
    --ddp_bucket_cap_mb 25 \
    --freeze_non_moe_lora $FREEZE_NON_MOE

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete: $OUTPUT_DIR"
else
    echo "❌ Training failed (exit: $TRAINING_EXIT_CODE)"
fi

exit $TRAINING_EXIT_CODE

