from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore


def load_hf_conversations(dataset_name: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads conversations from Hugging Face datasets. Expects a schema compatible with:
    - {'id': str, 'system': str, 'conversations': [{'role': 'user'|'assistant'|'tool', 'content': str}]}
    """
    if split is None:
        dset = load_dataset(dataset_name)
        data_splits: List[Dataset] = []
        for key in sorted(dset.keys()):
            data_splits.append(dset[key])
        merged: Dataset = Dataset.from_list([row for ds in data_splits for row in ds])
        return list(merged)
    else:
        ds: Dataset = load_dataset(dataset_name, split=split)  # type: ignore
        return list(ds)


def approximate_token_count(text: str) -> int:
    # Rough heuristic when tokenizer is unavailable
    # average ~4 chars per token in English text
    return max(1, len(text) // 4)


def conversation_token_length(turns: List[Dict[str, Any]], tokenizer: Optional[PreTrainedTokenizerBase]) -> int:
    text = "\n".join([t.get("content", "") for t in turns])
    if tokenizer is None:
        return approximate_token_count(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def filter_conversations(
    conversations: Iterable[Dict[str, Any]],
    max_turns: int,
    max_input_tokens: int,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    drop_if_missing_agent_next: bool = True,
) -> List[Dict[str, Any]]:
    """
    Filters conversations by turn count and approximate token budget.
    Optionally drops dialogues where there is no agent turn following a user/tool turn.
    """
    kept: List[Dict[str, Any]] = []
    for conv in conversations:
        turns = conv.get("conversations", [])
        if not isinstance(turns, list) or len(turns) == 0:
            continue
        if len(turns) > max_turns:
            turns = turns[:max_turns]
        if conversation_token_length(turns, tokenizer) > max_input_tokens:
            continue
        if drop_if_missing_agent_next:
            has_following_agent = False
            for i in range(len(turns) - 1):
                if turns[i].get("role") in ("user", "tool") and turns[i + 1].get("role") == "assistant":
                    has_following_agent = True
                    break
            if not has_following_agent:
                continue
        kept.append({**conv, "conversations": turns})
    return kept


