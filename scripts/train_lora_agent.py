import argparse
import os
from typing import Any, Dict, List

import torch  # type: ignore
from datasets import Dataset  # type: ignore
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data_agent import filter_conversations
from src.formatting_agent import format_dataset_examples
from src.utils import ensure_dir, load_yaml_config, read_jsonl, set_seed


def try_load_model(
    model_name: str,
    load_in_4bit: bool,
    bnb_4bit_compute_dtype: str,
    bnb_4bit_use_double_quant: bool,
    bnb_4bit_quant_type: str,
):
    kwargs: Dict[str, Any] = {"device_map": "auto"}
    if load_in_4bit:
        try:
            kwargs.update(
                dict(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                )
            )
        except Exception:
            kwargs["load_in_4bit"] = False
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    return model


def build_training_args(train_conf: Dict[str, Any], output_dir: str, logging_dir: str) -> TrainingArguments:
    t = train_conf["train"]
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=t.get("bf16", False),
        fp16=t.get("fp16", False),
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        report_to=t.get("report_to", None),
        evaluation_strategy="steps",
        save_strategy="steps",
    )
    return args


def tokenize_examples(tokenizer, examples: List[Dict[str, str]], max_length_input: int, max_length_target: int):
    eos = tokenizer.eos_token or ""
    input_texts = [e["input_text"] for e in examples]
    target_texts = [e["target_text"] + eos for e in examples]
    enc_inputs = tokenizer(
        input_texts,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=max_length_input,
    )
    enc_targets = tokenizer(
        target_texts,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=max_length_target,
    )
    model_inputs: Dict[str, List[List[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for inp_ids, tgt_ids in zip(enc_inputs["input_ids"], enc_targets["input_ids"]):
        input_ids = inp_ids + tgt_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(inp_ids) + tgt_ids[:]
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    return model_inputs


def build_hf_dataset(tokenizer, examples: List[Dict[str, str]], max_input: int, max_target: int) -> Dataset:
    tokenized = tokenize_examples(tokenizer, examples, max_input, max_target)
    ds = Dataset.from_dict(tokenized)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_config", type=str, default="configs/data.yaml")
    ap.add_argument("--model_config", type=str, default="configs/model_lora.yaml")
    ap.add_argument("--train_config", type=str, default="configs/train.yaml")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    data_conf = load_yaml_config(args.data_config)
    model_conf = load_yaml_config(args.model_config)
    train_conf = load_yaml_config(args.train_config)
    set_seed(train_conf.get("seed", 42))

    output_dir = args.output_dir or train_conf.get("output_dir", "outputs/checkpoints/run")
    logging_dir = train_conf.get("logging_dir", "outputs/logs")
    ensure_dir(output_dir)
    ensure_dir(logging_dir)

    tok_name = data_conf.get("tokenizer_name", model_conf["base_model_id"])
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = try_load_model(
        model_conf["base_model_id"],
        bool(model_conf.get("load_in_4bit", False)),
        str(model_conf.get("bnb_4bit_compute_dtype", "float16")),
        bool(model_conf.get("bnb_4bit_use_double_quant", True)),
        str(model_conf.get("bnb_4bit_quant_type", "nf4")),
    )
    model.config.use_cache = False

    lora_conf = model_conf.get("lora", {})
    lora = LoraConfig(
        r=int(lora_conf.get("r", 16)),
        lora_alpha=int(lora_conf.get("alpha", 32)),
        lora_dropout=float(lora_conf.get("dropout", 0.05)),
        target_modules=lora_conf.get("target_modules", None),
        bias=lora_conf.get("bias", "none"),
        task_type=lora_conf.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora)

    train_path = data_conf["splits"]["train"]
    dev_path = data_conf["splits"]["dev"]
    with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    def load_and_format(path: str) -> List[Dict[str, str]]:
        convs = read_jsonl(path)
        convs = filter_conversations(
            convs,
            max_turns=int(data_conf["filters"]["max_turns"]),
            max_input_tokens=int(data_conf["filters"]["max_input_tokens"]),
            tokenizer=tokenizer,
            drop_if_missing_agent_next=bool(data_conf["filters"].get("drop_if_missing_agent_next", True)),
        )
        return format_dataset_examples(convs, system_prompt, max_context_turns=int(data_conf["formatting"]["max_context_turns"]))

    if os.path.exists(train_path) and os.path.exists(dev_path):
        train_examples = load_and_format(train_path)
        dev_examples = load_and_format(dev_path)
    else:
        from src.data_agent import load_hf_conversations
        convs = load_hf_conversations(data_conf["dataset_name"], split="train")
        train_examples = format_dataset_examples(convs, system_prompt, max_context_turns=int(data_conf["formatting"]["max_context_turns"]))
        dev_examples = train_examples[: min(len(train_examples), 1024)]

    max_inp = int(data_conf["filters"]["max_input_tokens"])
    max_tgt = int(data_conf["filters"]["max_target_tokens"])
    ds_train = build_hf_dataset(tokenizer, train_examples, max_inp, max_tgt)
    ds_dev = build_hf_dataset(tokenizer, dev_examples, max_inp, max_tgt)

    targs = build_training_args(train_conf, output_dir, logging_dir)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training completed.")


if __name__ == "__main__":
    main()


