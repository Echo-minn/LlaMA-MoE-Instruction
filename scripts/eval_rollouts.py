import argparse
import json
from typing import Any, Dict, List, Tuple

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from src.eval_env import MockOSEnv
from src.utils import ensure_dir


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt) :].strip()


def demo_tasks() -> List[Dict[str, Any]]:
    return [
        {"instruction": "List files in the current directory.", "goal": "list dir"},
        {"instruction": "Create a file notes.txt with content 'hello agent'.", "goal": "write then read"},
        {"instruction": "Read the file notes.txt.", "goal": "read file"},
    ]


def run_agent_on_task(model, tokenizer, env: MockOSEnv, instruction: str) -> Dict[str, Any]:
    prompt = f"[SYSTEM]: You are an OS agent.\n[USER]: {instruction}"
    reply = generate(model, tokenizer, prompt)
    # For now, we do not auto-execute tool calls. We log raw output.
    return {"prompt": prompt, "reply": reply}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--num_tasks", type=int, default=3)
    args = ap.parse_args()

    ensure_dir("outputs/samples")
    env = MockOSEnv()

    # Base model
    base_id = "meta-llama/Llama-3.2-3B-Instruct"
    base_tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True, use_fast=True)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_tok.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(base_id, trust_remote_code=True, device_map="auto")

    # Finetuned
    ft_tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=True)
    if ft_tok.pad_token is None:
        ft_tok.pad_token = ft_tok.eos_token
    ft_tok.padding_side = "left"
    ft_model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True, device_map="auto")

    tasks = demo_tasks()[: args.num_tasks]
    transcripts: Dict[str, List[Dict[str, Any]]] = {"base": [], "finetuned": []}
    for t in tasks:
        transcripts["base"].append(run_agent_on_task(base_model, base_tok, env, t["instruction"]))
        transcripts["finetuned"].append(run_agent_on_task(ft_model, ft_tok, env, t["instruction"]))

    with open("outputs/samples/rollouts.json", "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    print("Saved outputs/samples/rollouts.json")


if __name__ == "__main__":
    main()


