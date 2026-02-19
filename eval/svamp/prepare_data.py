#!/usr/bin/env python3
"""
Prepare SVAMP (harder arithmetic story problems) for ACE framework.

Dataset: ChilleD/SVAMP (HuggingFace)

SVAMP is designed to be adversarially modified to challenge
semantic understanding + arithmetic reasoning (harder than GSM8K/ASDiv-style).

Important: The dataset includes an 'Equation' field. We MUST NOT include it
in the model input to avoid leakage. We only use Body + Question.

Splits (deterministic):
  - Train: from HF train split, first N after shuffle
  - Val:   from remaining HF train split
  - Test:  HF test split

Usage:
  python -m eval.svamp.prepare_data
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List


SEED = 42
OUTPUT_DIR = "./eval/svamp/data"

TRAIN_N = 500
VAL_N = 200
TRAIN_SMALL_N = 100


def _normalize_target(ans: Any) -> str:
    if ans is None:
        return ""
    if isinstance(ans, (int, float)):
        return str(ans)
    s = str(ans).strip()
    s = s.replace(",", "")
    return s


def process_sample(ex: Dict[str, Any]) -> Dict[str, str]:
    body = str(ex.get("Body", "")).strip()
    question = str(ex.get("Question", "")).strip()
    answer = ex.get("Answer", None)

    q_text = (
        f"{question}\n\n"
        "Answer with ONLY the final numeric answer. "
        "Do not include units or any explanation."
    ).strip()

    return {
        "context": body,
        "question": q_text,
        "target": _normalize_target(answer),
    }


def save_jsonl(samples: List[Dict[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples -> {path}")


def main() -> None:
    from datasets import load_dataset

    print("=" * 60)
    print("SVAMP — Data Preparation for ACE")
    print("=" * 60)

    print("\nLoading dataset from HuggingFace: ChilleD/SVAMP ...")
    ds = load_dataset("ChilleD/SVAMP")
    print(f"Splits: {list(ds.keys())}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} samples")

    train_raw = list(ds["train"])
    test_raw = list(ds["test"])

    rng = random.Random(SEED)
    rng.shuffle(train_raw)

    train_n = min(TRAIN_N, len(train_raw))
    val_n = min(VAL_N, max(0, len(train_raw) - train_n))

    train_split = train_raw[:train_n]
    val_split = train_raw[train_n:train_n + val_n]
    test_split = test_raw

    # Process
    train = [process_sample(x) for x in train_split if process_sample(x)["target"]]
    val = [process_sample(x) for x in val_split if process_sample(x)["target"]]
    test = [process_sample(x) for x in test_split if process_sample(x)["target"]]

    print(f"\nSplit sizes: train={len(train)} val={len(val)} test={len(test)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_jsonl(train, os.path.join(OUTPUT_DIR, "svamp_train.jsonl"))
    save_jsonl(val, os.path.join(OUTPUT_DIR, "svamp_val.jsonl"))
    save_jsonl(test, os.path.join(OUTPUT_DIR, "svamp_test.jsonl"))

    train_small = train[: min(TRAIN_SMALL_N, len(train))]
    save_jsonl(train_small, os.path.join(OUTPUT_DIR, "svamp_train_100.jsonl"))

    config = {
        "svamp": {
            "train_data": "./eval/svamp/data/svamp_train.jsonl",
            "val_data": "./eval/svamp/data/svamp_val.jsonl",
            "test_data": "./eval/svamp/data/svamp_test.jsonl",
        },
        "svamp_small": {
            "train_data": "./eval/svamp/data/svamp_train_100.jsonl",
            "val_data": "./eval/svamp/data/svamp_val.jsonl",
            "test_data": "./eval/svamp/data/svamp_test.jsonl",
        },
    }
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved config -> {config_path}")

    if train:
        ex = train[0]
        print(f"\n{'=' * 60}\nExample\n{'=' * 60}")
        print(f"Context: {ex['context']}")
        print(f"\nQuestion:\n{ex['question']}")
        print(f"\nTarget: {ex['target']}")

    print(f"\n{'=' * 60}\nDone.\n{'=' * 60}")


if __name__ == "__main__":
    main()

