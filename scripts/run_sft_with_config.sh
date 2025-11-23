#!/bin/bash

# SFT Training Script with YAML Configuration Support
# Usage: bash scripts/run_sft_with_config.sh [config_name]
# Example: bash scripts/run_sft_with_config.sh sft_default

# Get config name from argument or use default
CONFIG_NAME=${1:-sft_default}
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -1 configs/*.yaml | xargs -n 1 basename | sed 's/.yaml//'
    exit 1
fi

echo "📋 Using configuration: $CONFIG_FILE"

# Parse YAML using Python
python3 << 'EOF'
import yaml
import os
import sys

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Set environment variables
env = config.get('environment', {})
for key, value in env.items():
    os.environ[key] = str(value).lower() if isinstance(value, bool) else str(value)
    print(f"export {key}={os.environ[key]}")

# Print configuration for bash script
model = config['model']
data = config['data']
training = config['training']
deepspeed = config['deepspeed']
hardware = config['hardware']

# Create bash variables
print(f"MODEL_PATH='{model['model_name_or_path']}'")
print(f"DATA_PATH='{data['data_path']}'")
print(f"OUTPUT_DIR='{training['output_dir']}'")
print(f"MAX_STEPS={training.get('max_steps', 'null')}")
print(f"BATCH_SIZE={training['per_device_train_batch_size']}")
print(f"GRAD_ACCUM={training['gradient_accumulation_steps']}")
print(f"LEARNING_RATE={training['learning_rate']}")
print(f"MAX_SEQ_LENGTH={data['max_seq_length']}")
print(f"GPUS='{hardware['gpus']}'")
print(f"DEEPSPEED_CONFIG='{deepspeed['config_file']}'")

# Model arguments
print(f"USE_LORA={str(model['use_lora']).lower()}")
print(f"USE_QLORA={str(model['use_qlora']).lower()}")
print(f"LORA_R={model['lora_r']}")
print(f"LORA_ALPHA={model['lora_alpha']}")
print(f"LORA_DROPOUT={model['lora_dropout']}")
if model.get('num_experts_to_train') is not None:
    print(f"NUM_EXPERTS={model['num_experts_to_train']}")
else:
    print(f"NUM_EXPERTS=null")

# Training arguments
print(f"BF16={str(training['bf16']).lower()}")
print(f"GRADIENT_CHECKPOINTING={str(training['gradient_checkpointing']).lower()}")
print(f"LOGGING_STEPS={training['logging_steps']}")
print(f"SAVE_STEPS={training['save_steps']}")
print(f"EVAL_STEPS={training['eval_steps']}")
print(f"SAVE_TOTAL_LIMIT={training['save_total_limit']}")
print(f"REPORT_TO='{training['report_to']}'")
print(f"LOG_LEVEL='{training['log_level']}'")
print(f"DATALOADER_WORKERS={hardware['dataloader_num_workers']}")

# Optional: num_train_epochs
if training.get('num_train_epochs'):
    print(f"NUM_EPOCHS={training['num_train_epochs']}")
else:
    print(f"NUM_EPOCHS=1")

EOF

# Source the Python output
eval $(python3 << 'EOF'
import yaml
import sys

with open(sys.argv[1], 'r') as f:
    config = yaml.safe_load(f)

env = config.get('environment', {})
for key, value in env.items():
    val = str(value).lower() if isinstance(value, bool) else str(value)
    print(f"export {key}={val}")

model = config['model']
data = config['data']
training = config['training']
deepspeed = config['deepspeed']
hardware = config['hardware']

print(f"export MODEL_PATH='{model['model_name_or_path']}'")
print(f"export DATA_PATH='{data['data_path']}'")
print(f"export OUTPUT_DIR='{training['output_dir']}'")
print(f"export MAX_STEPS={training.get('max_steps') if training.get('max_steps') else -1}")
print(f"export BATCH_SIZE={training['per_device_train_batch_size']}")
print(f"export GRAD_ACCUM={training['gradient_accumulation_steps']}")
print(f"export LEARNING_RATE={training['learning_rate']}")
print(f"export MAX_SEQ_LENGTH={data['max_seq_length']}")
print(f"export GPUS='{hardware['gpus']}'")
print(f"export DEEPSPEED_CONFIG='{deepspeed['config_file']}'")
print(f"export USE_LORA={str(model['use_lora']).capitalize()}")
print(f"export USE_QLORA={str(model['use_qlora']).capitalize()}")
print(f"export LORA_R={model['lora_r']}")
print(f"export LORA_ALPHA={model['lora_alpha']}")
print(f"export LORA_DROPOUT={model['lora_dropout']}")
print(f"export NUM_EXPERTS={model.get('num_experts_to_train', 'null')}")
print(f"export BF16={str(training['bf16']).capitalize()}")
print(f"export GRADIENT_CHECKPOINTING={str(training['gradient_checkpointing']).capitalize()}")
print(f"export LOGGING_STEPS={training['logging_steps']}")
print(f"export SAVE_STEPS={training['save_steps']}")
print(f"export EVAL_STEPS={training['eval_steps']}")
print(f"export SAVE_TOTAL_LIMIT={training['save_total_limit']}")
print(f"export REPORT_TO='{training['report_to']}'")
print(f"export LOG_LEVEL='{training['log_level']}'")
print(f"export DATALOADER_WORKERS={hardware['dataloader_num_workers']}")
print(f"export NUM_EPOCHS={training.get('num_train_epochs', 1)}")
EOF
$CONFIG_FILE
)

echo ""
echo "🚀 Starting training with the following configuration:"
echo "   Model: $MODEL_PATH"
echo "   Data: $DATA_PATH"
echo "   Batch size: $BATCH_SIZE × $GRAD_ACCUM (effective: $((BATCH_SIZE * GRAD_ACCUM * 4)))"
echo "   Max steps: $MAX_STEPS"
echo "   Experts to train: $NUM_EXPERTS"
echo "   GPUs: $GPUS"
echo ""

# Build deepspeed command
CMD="deepspeed --include localhost:$GPUS src/train_sft.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --bf16 $BF16 \
    --use_lora $USE_LORA \
    --use_qlora $USE_QLORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT"

# Add num_experts_to_train if specified
if [ "$NUM_EXPERTS" != "null" ]; then
    CMD="$CMD --num_experts_to_train $NUM_EXPERTS"
fi

# Add max_steps or num_train_epochs
if [ "$MAX_STEPS" != "-1" ]; then
    CMD="$CMD --max_steps $MAX_STEPS"
fi
CMD="$CMD --num_train_epochs $NUM_EPOCHS"

# Continue building command
CMD="$CMD \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --logging_steps $LOGGING_STEPS \
    --log_level $LOG_LEVEL \
    --disable_tqdm False \
    --report_to $REPORT_TO \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --max_seq_length $MAX_SEQ_LENGTH \
    --dataloader_num_workers $DATALOADER_WORKERS \
    --dataloader_pin_memory True"

# Execute
echo "Executing: $CMD"
echo ""
eval $CMD

