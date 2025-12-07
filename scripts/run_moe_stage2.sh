#!/bin/bash

# Stage 2: Task-Grouped MoE Training for Expert Specialization
# Sequential task training with frozen attention LoRA
#
# Usage:
#   bash scripts/run_moe_stage2.sh                    # Auto-resume from latest stage 1 checkpoint
#   bash scripts/run_moe_stage2.sh checkpoint-4500    # Resume from specific checkpoint

echo "=========================================="
echo "Stage 2: MoE Expert Specialization Training"
echo "=========================================="
echo ""

# Parse command line arguments
STAGE1_CHECKPOINT=""
if [ -n "$1" ]; then
    STAGE1_CHECKPOINT="$1"
else
    # Auto-detect latest checkpoint from stage 1
    STAGE1_OUTPUT="outputs/llama-3b-moe-mixed-sft"
    LATEST_CHECKPOINT=$(ls -d $STAGE1_OUTPUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        STAGE1_CHECKPOINT="$LATEST_CHECKPOINT"
        echo "📂 Auto-detected latest Stage 1 checkpoint:"
        echo "   $STAGE1_CHECKPOINT"
    else
        echo "❌ No Stage 1 checkpoint found in $STAGE1_OUTPUT"
        echo ""
        echo "Usage:"
        echo "  bash scripts/run_moe_stage2.sh [checkpoint_path]"
        echo ""
        echo "Example:"
        echo "  bash scripts/run_moe_stage2.sh outputs/llama-3b-moe-mixed-sft/checkpoint-4500"
        exit 1
    fi
fi

# Check if checkpoint exists
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $STAGE1_CHECKPOINT"
    exit 1
fi

# GPU Configuration
GPU_IDS="4,5,6,7"  # 🔧 Modify based on available GPUs
NUM_GPUS=4

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=warning

# Model Configuration
MODEL_PATH="$STAGE1_CHECKPOINT"
DATA_CONFIG="configs/data_task_grouped.yaml"
OUTPUT_DIR="outputs/llama-3b-moe-stage3"
RUN_NAME="moe-stage3"

# Stage 2 Hyperparameters
MAX_STEPS=1500        # 5 tasks × 300 steps × 2 cycles
BATCH_SIZE=13         # Training batch size
GRAD_ACCUM=2          # Same as stage 1
LEARNING_RATE=3e-6    # Lower than stage 1 (8e-6)
WARMUP_RATIO=0.15     # Longer than stage 1 (0.1)
AUX_LOSS_ALPHA=0.01   # 10x higher than stage 1 (0.001)
SAVE_STEPS=100        # Save at end of each task
EVAL_STEPS=300        # Evaluate at end of each task
GRADIENT_CKPT="True"

echo ""
echo "📋 Stage 2 Configuration:"
echo "   Model: $MODEL_PATH"
echo "   Data config: $DATA_CONFIG"
echo "   Output: $OUTPUT_DIR"
echo "   GPUs: $GPU_IDS"
echo "   Max steps: $MAX_STEPS (5 tasks × 300 steps × 2 cycles)"
echo ""


# Confirm before starting
printf "Continue with Stage 2 training? (y/n) "
read -r REPLY
if [ "$REPLY" != "y" ] && [ "$REPLY" != "Y" ]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "🚀 Starting Stage 2 training..."
echo ""

# Run training
deepspeed --include localhost:$GPU_IDS \
    scripts/train_moe_stage2.py \
    --deepspeed distributed-training/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --data_config $DATA_CONFIG \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --bf16 True \
    --use_qlora True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --aux_loss_alpha $AUX_LOSS_ALPHA \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio $WARMUP_RATIO \
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
    --ddp_bucket_cap_mb 25

TRAINING_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Stage 2 Training Complete!"
    echo "=========================================="
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
else
    echo "❌ Stage 2 Training Failed!"
    echo "=========================================="
    echo ""
    echo "Exit code: $TRAINING_EXIT_CODE"
    echo "Please check the error messages above for details."
    echo ""
    exit $TRAINING_EXIT_CODE
fi

