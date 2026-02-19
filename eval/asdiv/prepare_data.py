#!/usr/bin/env python3
"""
Prepare ASDiv (Arithmetic Story Problems) for ACE framework.

Dataset: yimingzhang/asdiv (HuggingFace)

Goal:
  - Create ACE-friendly JSONL files with fields: context, question, target
  - Provide a *small but meaningful* split so we can quickly run a baseline
    and then run offline training to check for ACE gain.

Splits (default, deterministic with SEED=42):
  - Train: 500
  - Val:   200
  - Test:  200
  - Small train subset: 100

If the dataset has fewer samples than requested, sizes are clipped.

Usage:
  python -m eval.asdiv.prepare_data
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional


SEED = 42
OUTPUT_DIR = "./eval/asdiv/data"

TRAIN_N = 500
VAL_N = 200
TEST_N = 200
TRAIN_SMALL_N = 100


def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _normalize_target(ans: Any) -> str:
    """
    Store the target as a compact string; numeric correctness is handled by the
    task DataProcessor (robust extraction + tolerance).
    """
    if ans is None:
        return ""
    if isinstance(ans, (int, float)):
        return str(ans)
    s = str(ans).strip()
    # common cleanup
    s = s.replace(",", "")
    return s


def process_sample(ex: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert one ASDiv example to ACE JSONL sample.

    ASDiv field names vary across releases; we try a few common candidates.
    """
    body = _first_present(ex, ["body", "Body", "context", "Context", "story", "Story", "problem", "Problem"])
    question = _first_present(ex, ["question", "Question", "q", "Q"])
    answer = _first_present(ex, ["answer", "Answer", "ans", "Ans", "target", "Target", "label", "Label"])

    if body is None and question is None:
        # fallback: stringify everything except answer-ish keys
        filtered = {k: v for k, v in ex.items() if k.lower() not in {"answer", "ans", "target", "label"}}
        body = json.dumps(filtered, ensure_ascii=False)

    if question is None:
        question = ""

    # Compose prompt: keep context separate to match other ACE math tasks
    q_text = (
        f"{question}\n\n"
        "Answer with ONLY the final numeric answer. "
        "Do not include units or any explanation."
    ).strip()

    return {
        "context": (str(body).strip() if body is not None else ""),
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
    print("ASDiv — Data Preparation for ACE")
    print("=" * 60)

    print("\nLoading dataset from HuggingFace: yimingzhang/asdiv ...")
    ds = load_dataset("yimingzhang/asdiv")
    print(f"Splits: {list(ds.keys())}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} samples")

    # Prefer a single pool split to build our own deterministic split
    if "train" in ds:
        pool = list(ds["train"])
        # Some datasets have dev/test too — include them for more variety if present
        for extra in ["validation", "dev", "test"]:
            if extra in ds:
                pool.extend(list(ds[extra]))
    else:
        # take first available split
        first = list(ds.keys())[0]
        pool = list(ds[first])

    print(f"\nTotal raw pool size: {len(pool)}")

    # Process + filter minimally
    all_samples: List[Dict[str, str]] = []
    missing_target = 0
    for ex in pool:
        s = process_sample(ex)
        if not s["target"]:
            missing_target += 1
            continue
        all_samples.append(s)

    print(f"Processed samples: {len(all_samples)} (dropped missing target: {missing_target})")

    rng = random.Random(SEED)
    rng.shuffle(all_samples)

    n_total = len(all_samples)
    train_n = min(TRAIN_N, n_total)
    val_n = min(VAL_N, max(0, n_total - train_n))
    test_n = min(TEST_N, max(0, n_total - train_n - val_n))

    train = all_samples[:train_n]
    val = all_samples[train_n:train_n + val_n]
    test = all_samples[train_n + val_n:train_n + val_n + test_n]

    print(f"\nSplit sizes: train={len(train)} val={len(val)} test={len(test)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_jsonl(train, os.path.join(OUTPUT_DIR, "asdiv_train.jsonl"))
    save_jsonl(val, os.path.join(OUTPUT_DIR, "asdiv_val.jsonl"))
    save_jsonl(test, os.path.join(OUTPUT_DIR, "asdiv_test.jsonl"))

    train_small = train[: min(TRAIN_SMALL_N, len(train))]
    save_jsonl(train_small, os.path.join(OUTPUT_DIR, "asdiv_train_100.jsonl"))

    # Save sample_config.json
    config = {
        "asdiv": {
            "train_data": "./eval/asdiv/data/asdiv_train.jsonl",
            "val_data": "./eval/asdiv/data/asdiv_val.jsonl",
            "test_data": "./eval/asdiv/data/asdiv_test.jsonl",
        },
        "asdiv_small": {
            "train_data": "./eval/asdiv/data/asdiv_train_100.jsonl",
            "val_data": "./eval/asdiv/data/asdiv_val.jsonl",
            "test_data": "./eval/asdiv/data/asdiv_test.jsonl",
        },
    }
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved config -> {config_path}")

    # Print example
    if train:
        ex = train[0]
        print(f"\n{'=' * 60}\nExample\n{'=' * 60}")
        print(f"Context (first 400 chars): {ex['context'][:400]}...")
        print(f"\nQuestion:\n{ex['question']}")
        print(f"\nTarget: {ex['target']}")

    print(f"\n{'=' * 60}\nDone.\n{'=' * 60}")


if __name__ == "__main__":
    main()

