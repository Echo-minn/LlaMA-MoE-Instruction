# LLaMA MoE with DP/EP and QLoRA training

## Download the model

1. Wandb login with `wandb login`

2. Huggingface login with `huggingface-cli login`

3. Download the model
```bash
bash scripts/get_init_model.sh
```

4. Convert the model to MoE
```bash
bash scripts/convert_llama3b_to_moe.sh
```

## Train the model
```bash
bash scripts/run_moe_stage1.sh
bash scripts/run_moe_stage2.sh
bash scripts/run_moe_stage3.sh
```

5. Evaluate the model
```bash
bash scripts/run_eval_metric.sh
```

6. Run routing sanity check
```bash
python scripts/check_routing_sanity.py
```