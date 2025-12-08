#!/bin/bash
# Stage 2: Task-Grouped MoE Training for Expert Specialization

set -eu
(set -o pipefail 2>/dev/null) && set -o pipefail || true

echo "Stage 2: MoE Expert Specialization"
echo ""

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-outputs/llama-3b-moe-stage1/checkpoint-1800}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_task_stage2A.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/llama-3b-moe-stage2}"
RUN_NAME="${RUN_NAME:-moe-stage2}"

MAX_STEPS=${MAX_STEPS:-1600}
STEPS_PER_TASK=${STEPS_PER_TASK:-200}
BATCH_SIZE=${BATCH_SIZE:-15}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LEARNING_RATE=${LEARNING_RATE:-2e-7}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
AUX_LOSS_ALPHA=${AUX_LOSS_ALPHA:-0.05}
SAVE_STEPS=${SAVE_STEPS:-400}
EVAL_STEPS=${EVAL_STEPS:-50}
EVAL_LOSS_ONLY=${EVAL_LOSS_ONLY:-True}
FREEZE_NON_MOE=${FREEZE_NON_MOE:-True}
GRADIENT_CKPT=${GRADIENT_CKPT:-True}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}

RESUME_FROM_CHECKPOINT=""
if [ -n "${RESUME:-}" ]; then
    RESUME_FROM_CHECKPOINT="$RESUME"
    [ ! -d "$RESUME_FROM_CHECKPOINT" ] && { echo "❌ Checkpoint not found: $RESUME_FROM_CHECKPOINT"; exit 1; }
fi

echo "📋 Config: Model=$MODEL_PATH | Output=$OUTPUT_DIR | GPUs=$GPU_IDS"
[ -n "$ADAPTER_PATH" ] && echo "   Adapter: $ADAPTER_PATH"
echo "   LR=$LEARNING_RATE | Warmup=$WARMUP_RATIO | Steps=$MAX_STEPS"
[ -n "$RESUME_FROM_CHECKPOINT" ] && echo "   Resume: $RESUME_FROM_CHECKPOINT"
if [ "$TENSOR_PARALLEL" -gt 1 ]; then
    [ $((NUM_GPUS % TENSOR_PARALLEL)) -ne 0 ] && { echo "❌ Error: TP ($TENSOR_PARALLEL) must divide GPUs ($NUM_GPUS)"; exit 1; }
    echo "   Tensor Parallelism: TP=$TENSOR_PARALLEL, DP=$((NUM_GPUS / TENSOR_PARALLEL))"
fi
echo ""

read -p "Continue with Stage 2 training? (y/n) " -r REPLY
[[ ! $REPLY =~ ^[Yy]$ ]] && { echo "Cancelled."; exit 0; }

RESUME_ARG=""
[ -n "$RESUME_FROM_CHECKPOINT" ] && RESUME_ARG="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"

ADAPTER_ARG=""
[ -n "$ADAPTER_PATH" ] && ADAPTER_ARG="--adapter_path $ADAPTER_PATH"

echo "🚀 Starting training..."
echo ""

deepspeed --include localhost:$GPU_IDS \
    scripts/train_moe_stage2.py \
    --deepspeed distributed-training/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    $ADAPTER_ARG \
    --data_config "$DATA_CONFIG" \
    --steps_per_task $STEPS_PER_TASK \
    --intra_group_shuffle True \
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
    --logging_steps 40 \
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

[ $TRAINING_EXIT_CODE -eq 0 ] && echo "✅ Training complete: $OUTPUT_DIR" || echo "❌ Training failed (exit: $TRAINING_EXIT_CODE)"

exit $TRAINING_EXIT_CODE
