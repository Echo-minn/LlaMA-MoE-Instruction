#!/bin/bash

# Mixed Dataset SFT Training Script
# Supports validation mode (quick test) and full mode (production)
# Supports resuming from checkpoint
#
# Usage:
#   bash scripts/run_mixed_sft.sh                          # Start new training
#   bash scripts/run_mixed_sft.sh --resume                 # Auto-resume from latest checkpoint
#   bash scripts/run_mixed_sft.sh --resume checkpoint-500  # Resume from specific checkpoint

echo "=========================================="
echo "Mixed Dataset SFT Training"
echo "=========================================="
echo ""

# Parse command line arguments
RESUME_FROM_CHECKPOINT=""
if [ "$1" == "--resume" ] || [ "$1" == "-r" ]; then
    if [ -n "$2" ]; then
        RESUME_FROM_CHECKPOINT="$2"
    else
        RESUME_FROM_CHECKPOINT="auto"
    fi
fi

# GPU Configuration
GPU_IDS="4,5,6,7"  # 🔧 Modify based on available GPUs
NUM_GPUS=4

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=warning

# Model Configuration
MODEL_PATH="models/Llama-3.2-3B-Instruct-MoE-8x"
DATA_CONFIG="configs/data_mix_instruction.yaml"

# Check mode from config
MODE=$(python3 -c "import yaml; print(yaml.safe_load(open('$DATA_CONFIG'))['mode'])")

if [ "$MODE" == "validation" ]; then
    echo "🔍 VALIDATION MODE"
    echo "   Using: First dataset only (5000 samples)"
    echo "   Purpose: Quick pipeline test"
    echo "   Time: ~30 minutes"
    echo ""
    OUTPUT_DIR="outputs/validation-mixed-sft"
    MAX_STEPS=500
    BATCH_SIZE=12
    GRAD_ACCUM=1
    GRADIENT_CKPT="False"
    SAVE_STEPS=100
    EVAL_STEPS=100
elif [ "$MODE" == "full" ]; then
    echo "🚀 FULL TRAINING MODE (Optimized with Flash Attention 2)"
    echo "   Using: All enabled datasets (~45K samples)"
    echo "   Purpose: Production model"
    echo ""
    OUTPUT_DIR="outputs/llama-3b-moe-mixed-sft"
    MAX_STEPS=5000
    BATCH_SIZE=12
    GRAD_ACCUM=2
    GRADIENT_CKPT="True"
    SAVE_STEPS=500
    EVAL_STEPS=1000     # Reduce evaluation frequency for faster training
else
    echo "❌ Invalid mode in config: $MODE"
    echo "   Please set mode to 'validation' or 'full'"
    exit 1
fi

echo "📋 Configuration:"
echo "   Model: $MODEL_PATH"
echo "   Data config: $DATA_CONFIG"
echo "   Output: $OUTPUT_DIR"
echo "   GPUs: $GPU_IDS"
echo "   Max steps: $MAX_STEPS"
echo "   Batch size: $NUM_GPUS × $BATCH_SIZE × $GRAD_ACCUM = $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"

# Check for existing checkpoints if resume requested
if [ "$RESUME_FROM_CHECKPOINT" == "auto" ]; then
    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        RESUME_FROM_CHECKPOINT="$LATEST_CHECKPOINT"
        echo "   Resume: Yes (auto-detected: $RESUME_FROM_CHECKPOINT)"
    else
        echo "   Resume: No checkpoints found, starting from scratch"
        RESUME_FROM_CHECKPOINT=""
    fi
elif [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "   Resume: Yes (from: $RESUME_FROM_CHECKPOINT)"
else
    echo "   Resume: No (starting from scratch)"
fi
echo ""

# Confirm before starting
if [ "$MODE" == "full" ]; then
    echo "⚠️  Full training will take 2-3 hours"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

echo ""
echo "🚀 Starting training..."
echo ""

# Build resume argument
RESUME_ARG=""
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_ARG="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
fi

# Run training
deepspeed --include localhost:$GPU_IDS \
    scripts/train_moe_stage1.py \
    --deepspeed distributed-training/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --data_config $DATA_CONFIG \
    --output_dir $OUTPUT_DIR \
    $RESUME_ARG \
    --bf16 True \
    --use_qlora True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 8e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 50 \
    --log_level warning \
    --disable_tqdm False \
    --report_to wandb \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_total_limit 2 \
    --gradient_checkpointing $GRADIENT_CKPT \
    --max_seq_length 1024 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --ddp_find_unused_parameters False \
    --ddp_bucket_cap_mb 25

# Check if training succeeded
TRAINING_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training Complete!"
    echo "=========================================="
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
else
    echo "❌ Training Failed!"
    echo "=========================================="
    echo ""
    echo "Exit code: $TRAINING_EXIT_CODE"
    echo "Please check the error messages above for details."
    echo ""
    exit $TRAINING_EXIT_CODE
fi

# Only show next steps if training succeeded
if [ $TRAINING_EXIT_CODE -eq 0 ] && [ "$MODE" == "validation" ]; then
    echo "Next steps:"
    echo "  1. Check if training completed successfully"
    echo "  2. Review logs and loss curves"
    echo "  3. If all good, switch to full mode:"
    echo "     Edit $DATA_CONFIG"
    echo "     Change: mode: \"validation\" → mode: \"full\""
    echo "     Run: bash scripts/run_mixed_sft.sh"
fi

