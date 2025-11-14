import argparse
import os
from typing import Any, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from src.formatting_agent import build_history_prompt
from src.metrics_agent import compute_perplexity
from src.utils import load_yaml_config, read_jsonl, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="outputs/splits/dev.jsonl")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_config", type=str, default="configs/data.yaml")
    ap.add_argument("--max_eval", type=int, default=256)
    args = ap.parse_args()

    data_conf = load_yaml_config(args.data_config)
    set_seed(42)

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    system_prompt = ""
    if os.path.exists("prompts/system_prompt.txt"):
        with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

    convs = read_jsonl(args.split)
    texts: List[str] = []
    for conv in convs[: args.max_eval]:
        turns = conv.get("conversations", [])
        if not turns:
            continue
        prompt = build_history_prompt(system_prompt, turns[:-1], max_context_turns=int(data_conf["formatting"]["max_context_turns"]))
        texts.append(prompt)
    ppl = compute_perplexity(
        model,
        tokenizer,
        texts,
        max_length=int(data_conf["filters"]["max_input_tokens"]),
        batch_size=2,
    )
    print(f"Perplexity: {ppl:.3f} on {len(texts)} prompts")


if __name__ == "__main__":
    main()


