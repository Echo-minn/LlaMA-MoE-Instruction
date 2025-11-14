Llama-AgentOS v1
================

An AgentOS-style fine-tuning setup using Meta Llama 3.2 3B Instruct to predict the next agent message (including tool calls) given a dialogue history. Trains on `neulab/agent-data-collection` and evaluates with perplexity and simple rollout tasks.

Quickstart
----------
1) Create env and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Inspect data:

```bash
python scripts/explore_data.py --max_convs 3
```

3) Prepare splits (reproducible IDs):

```bash
python scripts/prepare_splits.py --seed 42 --train_ratio 0.8 --dev_ratio 0.1
```

4) Train LoRA (QLoRA config in `configs/model_lora.yaml`, training config in `configs/train.yaml`):

```bash
python scripts/train_lora_agent.py \
  --data_config configs/data.yaml \
  --model_config configs/model_lora.yaml \
  --train_config configs/train.yaml \
  --output_dir outputs/checkpoints/llama-agentos-qlora
```

5) Eval perplexity on held-out:

```bash
python scripts/eval_perplexity.py \
  --split outputs/splits/dev.jsonl \
  --checkpoint outputs/checkpoints/llama-agentos-qlora
```

6) Qualitative rollouts (simple mock OS env):

```bash
python scripts/eval_rollouts.py \
  --checkpoint outputs/checkpoints/llama-agentos-qlora \
  --num_tasks 5
```

Structure
---------
- `configs/`: data, model LoRA/quantization, training configs
- `prompts/`: system prompt and optional tool schema
- `src/`: dataset loading, formatting, metrics, mock eval env, utils
- `scripts/`: CLI entry points for data exploration, splits, training, evaluation
- `notebooks/`: report-ready EDA and examples
- `outputs/`: logs, checkpoints, evals, samples, splits

Notes and Tips
--------------
- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Dataset: `neulab/agent-data-collection`
- Single-GPU friendly: default configs use 4-bit loading and LoRA to reduce memory.
- On macOS without CUDA, 4-bit via `bitsandbytes` may not be supported. You can set `load_in_4bit: false` in `configs/model_lora.yaml` to run in full precision or try CPU inference for small experiments.
- Tool-call syntax: `<tool=name>{"arg":"value"}</tool>` (see `prompts/tool_schema.json`). The formatter ensures the next-agent-target includes these tags when present in data.

License
-------
This repository scaffolding is provided as-is for academic use. Check licenses of upstream models and datasets before use. 

