from typing import Any, Dict, Iterable, List, Optional, Tuple


TOOL_OPEN_PREFIX = "<tool="
TOOL_CLOSE = "</tool>"


def build_history_prompt(system_prompt: str, turns: List[Dict[str, str]], max_context_turns: int = 12) -> str:
    """
    Build a simple chat-style prompt:
    [SYSTEM]: ...
    [USER]: ...
    [ASSISTANT]: ...
    [TOOL]: <tool=...>{...}</tool>
    """
    ctx = turns[-max_context_turns:] if max_context_turns > 0 else turns
    lines: List[str] = []
    if system_prompt:
        lines.append(f"[SYSTEM]: {system_prompt}".strip())
    for t in ctx:
        role = t.get("role", "").strip().lower()
        content = t.get("content", "")
        if role == "user":
            lines.append(f"[USER]: {content}")
        elif role == "assistant":
            lines.append(f"[ASSISTANT]: {content}")
        elif role == "tool":
            lines.append(f"[TOOL]: {content}")
        else:
            lines.append(f"[{role.upper()}]: {content}")
    return "\n".join(lines)


def create_next_agent_examples(
    conversation: Dict[str, Any],
    system_prompt: str,
    max_context_turns: int = 12,
) -> List[Dict[str, Any]]:
    """
    Produces one example per assistant turn:
      - input: system + history up to the turn before the assistant message
      - target: the assistant message itself (may include tool markup)
    """
    turns: List[Dict[str, str]] = conversation.get("conversations", [])
    conv_id = conversation.get("id", None)
    examples: List[Dict[str, Any]] = []
    for i, turn in enumerate(turns):
        if turn.get("role") != "assistant":
            continue
        history = turns[:i]
        inp = build_history_prompt(system_prompt, history, max_context_turns=max_context_turns)
        tgt = turn.get("content", "")
        examples.append(
            {
                "id": conv_id,
                "turn_index": i,
                "input_text": inp,
                "target_text": tgt,
            }
        )
    return examples


def format_dataset_examples(
    conversations: Iterable[Dict[str, Any]],
    system_prompt: str,
    max_context_turns: int = 12,
) -> List[Dict[str, Any]]:
    all_examples: List[Dict[str, Any]] = []
    for conv in conversations:
        all_examples.extend(
            create_next_agent_examples(conv, system_prompt=system_prompt, max_context_turns=max_context_turns)
        )
    return all_examples


