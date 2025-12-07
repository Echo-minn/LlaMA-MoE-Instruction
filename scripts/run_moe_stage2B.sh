#!/bin/bash

# Stage 2B: Task-Grouped MoE Training (Math, Code, Summ, Translation)

set -eu
# Use pipefail only if bash supports it (bash 3.0+)
(set -o pipefail 2>/dev/null) && set -o pipefail || true

echo "Stage 2B: MoE Expert Specialization"
echo ""

# GPU configuration (edit as needed)
GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=warning
# Use expandable CUDA allocator segments to reduce fragmentation-related OOMs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Stage 2B config
MODEL_PATH="models/Llama-3.2-3B-Instruct-MoE-8x"
# Optional: If you want to reuse adapters from a previous training run (e.g., collapsed routing model)
# Set ADAPTER_PATH to the path containing adapter_config.json (e.g., "models/llama-3b-moe-stage2")
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATA_CONFIG="configs/data_task_stage2A.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/llama-3b-moe-stage1}"
RUN_NAME="${RUN_NAME:-moe-stage1}"

MAX_STEPS=${MAX_STEPS:-2400}      # 4 tasks × 300 steps × 2 cycles + buffer
STEPS_PER_TASK=${STEPS_PER_TASK:-300}
BATCH_SIZE=${BATCH_SIZE:-15}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.10}
AUX_LOSS_ALPHA=${AUX_LOSS_ALPHA:-0.08}  # Increased from 0.01 to maintain routing separation
AUX_LOSS_INITIAL=${AUX_LOSS_INITIAL:-0.12}  # Optional: start higher and decay (e.g., 0.1)
AUX_LOSS_WARMUP=${AUX_LOSS_WARMUP:-400}  # Steps to keep at initial before decay
AUX_LOSS_DECAY_STEPS=${AUX_LOSS_DECAY_STEPS:-}  # Steps to decay (empty = decay over all training)
SAVE_STEPS=${SAVE_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-50}
EVAL_LOSS_ONLY=${EVAL_LOSS_ONLY:-True}
FREEZE_NON_MOE=${FREEZE_NON_MOE:-True}
GRADIENT_CKPT=${GRADIENT_CKPT:-True}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}

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
if [ -n "$ADAPTER_PATH" ]; then
    echo "   Adapter: $ADAPTER_PATH (reusing existing adapters)"
fi
echo "   LR=$LEARNING_RATE | Warmup=$WARMUP_RATIO | Steps=$MAX_STEPS"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "   Resume: $RESUME_FROM_CHECKPOINT (LR→$LEARNING_RATE)"
fi
if [ "$TENSOR_PARALLEL" -gt 1 ]; then
    if [ $((NUM_GPUS % TENSOR_PARALLEL)) -ne 0 ]; then
        echo "❌ Error: Tensor parallelism degree ($TENSOR_PARALLEL) must divide number of GPUs ($NUM_GPUS)"
        exit 1
    fi
    DATA_PARALLEL=$((NUM_GPUS / TENSOR_PARALLEL))
    echo "   Tensor Parallelism: TP=$TENSOR_PARALLEL, DP=$DATA_PARALLEL (Total GPUs: $NUM_GPUS)"
fi
echo ""

printf "Continue with Stage 2B training? (y/n) "
read -r REPLY
if [ "$REPLY" != "y" ] && [ "$REPLY" != "Y" ]; then
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

# Build adapter argument if specified
ADAPTER_ARG=""
if [ -n "$ADAPTER_PATH" ]; then
    ADAPTER_ARG="--adapter_path $ADAPTER_PATH"
fi

deepspeed --include localhost:$GPU_IDS \
    scripts/train_moe_stage2B.py \
    --deepspeed distributed-training/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    $ADAPTER_ARG \
    --data_config "$DATA_CONFIG" \
    --steps_per_task $STEPS_PER_TASK \
    --shuffle_tasks True \
    --eval_loss_only $EVAL_LOSS_ONLY \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    $RESUME_ARG \
    --bf16 True \
    --use_qlora True \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --aux_loss_alpha $AUX_LOSS_ALPHA \
    ${AUX_LOSS_INITIAL:+--aux_loss_initial $AUX_LOSS_INITIAL} \
    ${AUX_LOSS_WARMUP:+--aux_loss_warmup_steps $AUX_LOSS_WARMUP} \
    ${AUX_LOSS_DECAY_STEPS:+--aux_loss_decay_steps $AUX_LOSS_DECAY_STEPS} \
    --pretraining_tp $TENSOR_PARALLEL \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio $WARMUP_RATIO \
    --max_steps $MAX_STEPS \
    --logging_steps 25 \
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

