import argparse
import json
from typing import Any, Dict, List

from src.data_agent import load_hf_conversations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="neulab/agent-data-collection")
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--max_convs", type=int, default=3)
    args = ap.parse_args()

    convs: List[Dict[str, Any]] = load_hf_conversations(args.dataset, split=args.split)
    for i, conv in enumerate(convs[: args.max_convs]):
        print("=" * 80)
        print(f"id: {conv.get('id')}")
        print(f"system: {conv.get('system')!r}")
        for t in conv.get("conversations", []):
            print(f"{t.get('role')}: {t.get('content')}")
    print(f"\nDisplayed {min(args.max_convs, len(convs))} / {len(convs)} conversations.")


if __name__ == "__main__":
    main()


