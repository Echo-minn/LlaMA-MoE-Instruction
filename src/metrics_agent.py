import math
import re
from typing import Iterable, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase  # type: ignore
import torch  # type: ignore


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    max_length: int = 4096,
    batch_size: int = 2,
) -> float:
    model.eval()
    losses: List[float] = []
    device = next(model.parameters()).device
    batch: List[str] = []
    with torch.no_grad():
        for text in texts:
            batch.append(text)
            if len(batch) < batch_size:
                continue
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            labels = enc["input_ids"].clone()
            outputs = model(**enc, labels=labels)
            loss = outputs.loss.detach().float()
            losses.append(loss.item())
            batch = []
        if batch:
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            labels = enc["input_ids"].clone()
            outputs = model(**enc, labels=labels)
            loss = outputs.loss.detach().float()
            losses.append(loss.item())
    if not losses:
        return float("inf")
    mean_loss = sum(losses) / len(losses)
    return float(math.exp(mean_loss))


TOOL_OPEN_RE = re.compile(r"<tool=([a-zA-Z0-9_\-]+)>")


def has_balanced_tool_tags(text: str) -> bool:
    opens = [m.start() for m in re.finditer(r"<tool=", text)]
    closes = [m.start() for m in re.finditer(r"</tool>", text)]
    return len(opens) == len(closes)


def tool_format_accuracy(pred_texts: Iterable[str]) -> float:
    """
    Simple syntactic validity: balanced <tool=...> and </tool> tags.
    """
    preds = list(pred_texts)
    if not preds:
        return 0.0
    valid = 0
    for t in preds:
        if has_balanced_tool_tags(t):
            valid += 1
    return valid / len(preds)


