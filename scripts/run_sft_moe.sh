#!/bin/bash

# 环境变量设置
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 路径设置
MODEL_PATH="models/Llama-3.1-8B-MoE-Upcycled"
DATA_PATH="facebook/natural_reasoning" # 或者替换为你下载好的本地数据路径
OUTPUT_DIR="outputs/Llama-MoE-SFT-Reasoning"

# 训练参数
MAX_STEPS=2000
BATCH_SIZE_PER_GPU=16    # 增大 Batch Size 提高利用率
GRAD_ACCUM=2            # 调整累积步数，保持总 Batch Size = 64 (8*2*4gpus)
LEARNING_RATE=2e-5

# 启动 DeepSpeed
deepspeed --include localhost:5,7 src/train_sft.py \
    --deepspeed distributed-training/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --logging_steps 20 \
    --report_to wandb \
    --save_strategy "steps" \
    --save_steps 500 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_total_limit 2 \
    --gradient_checkpointing True \
    --max_seq_length 2048

