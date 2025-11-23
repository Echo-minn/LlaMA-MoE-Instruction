#!/bin/bash

# 环境变量设置
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=warning

# 路径设置
MODEL_PATH="models/Llama-3.1-8B-MoE-Upcycled"
DATA_PATH="facebook/natural_reasoning"
OUTPUT_DIR="outputs/Llama-MoE-SFT-Reasoning"

# 训练参数
MAX_STEPS=2000
BATCH_SIZE_PER_GPU=4    # 保持4，专注于max_length优化
GRAD_ACCUM=2            # 4卡 * 4 * 2 = 32 总 Batch Size
LEARNING_RATE=2e-5

# 启动 DeepSpeed
deepspeed --include localhost:4,5,6,7 src/train_sft.py \
    --deepspeed distributed-training/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --use_lora True \
    --use_qlora False \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_experts_to_train 4 \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --logging_steps 50 \
    --log_level warning \
    --disable_tqdm False \
    --report_to wandb \
    --save_strategy "steps" \
    --save_steps 500 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_total_limit 2 \
    --gradient_checkpointing False \
    --max_seq_length 1440 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True

