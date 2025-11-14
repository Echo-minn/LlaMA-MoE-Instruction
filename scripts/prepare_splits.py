import argparse
import random
from typing import Any, Dict, List, Tuple

from src.data_agent import load_hf_conversations
from src.utils import ensure_dir, save_jsonl


def split_ids(ids: List[str], train_ratio: float, dev_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rnd = random.Random(seed)
    ids_copy = ids[:]
    rnd.shuffle(ids_copy)
    n = len(ids_copy)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train_ids = ids_copy[:n_train]
    dev_ids = ids_copy[n_train : n_train + n_dev]
    test_ids = ids_copy[n_train + n_dev :]
    return train_ids, dev_ids, test_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="neulab/agent-data-collection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    args = ap.parse_args()

    convs = load_hf_conversations(args.dataset, split=None)
    ids = [str(c.get("id", i)) for i, c in enumerate(convs)]
    train_ids, dev_ids, test_ids = split_ids(ids, args.train_ratio, args.dev_ratio, args.seed)
    id_to_conv = {str(c.get("id", i)): c for i, c in enumerate(convs)}

    ensure_dir("outputs/splits")
    save_jsonl("outputs/splits/train.jsonl", [id_to_conv[i] for i in train_ids])
    save_jsonl("outputs/splits/dev.jsonl", [id_to_conv[i] for i in dev_ids])
    save_jsonl("outputs/splits/test.jsonl", [id_to_conv[i] for i in test_ids])
    print(f"Saved: train={len(train_ids)} dev={len(dev_ids)} test={len(test_ids)}")


if __name__ == "__main__":
    main()


