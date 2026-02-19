#!/usr/bin/env python3
"""
Prepare LSAT Analytical Reasoning (AGIEval subset) for ACE framework.

Source: AGIEval benchmark — lsat-ar.jsonl
  https://github.com/ruixiangcui/AGIEval/tree/main/data/v1

LSAT-AR contains 230 analytical reasoning (logic puzzle) questions from the
Law School Admission Test (LSAT).  Each question presents a constraint setup
(e.g. scheduling, ordering, grouping) and asks the model to determine which
arrangement satisfies or violates the given rules.

This is the hardest LSAT sub-task: GPT-4-class models score ~40-55% 0-shot,
significantly below human performance, making it an ideal benchmark for ACE
to demonstrate meaningful gain through reflection-based learning.

All questions are single-choice (A–E) with one correct answer.

Split (deterministic, seed=42):
  Train : 150  — used for ACE offline training
  Val   :  40  — used for validation during training
  Test  :  40  — held-out final evaluation

Usage:
  python -m eval.lsat_ar.prepare_data
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import Any, Dict, List

SEED = 42
RAW_FILE = "./eval/agieval/raw_data/lsat-ar.jsonl"
OUTPUT_DIR = "./eval/lsat_ar/data"

TRAIN_N = 150
VAL_N = 40
# Remaining ~40 goes to test


def _format_options(options: List[str]) -> str:
    """Return options joined on separate lines (already include '(A) ...')."""
    return "\n".join(options)


def process_sample(ex: Dict[str, Any]) -> Dict[str, str]:
    """Convert a raw AGIEval LSAT-AR example to ACE format.

    ACE format:
      context  — constraint setup / puzzle description
      question — question stem + formatted choices (A–E) + answer instruction
      target   — correct letter (A–E, uppercase)
    """
    passage = (ex.get("passage") or "").strip()
    question_stem = (ex.get("question") or "").strip()
    options: List[str] = ex.get("options") or []
    label = ex.get("label") or ""

    options_text = _format_options(options)

    question_full = (
        f"{question_stem}\n\n"
        f"{options_text}\n\n"
        "Answer with ONLY the letter of the correct option (A, B, C, D, or E). "
        "Do not include any explanation."
    )

    return {
        "context": passage,
        "question": question_full,
        "target": str(label).strip().upper(),
    }


def save_jsonl(samples: List[Dict[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples -> {path}")


def main() -> None:
    print("=" * 60)
    print("LSAT-AR (AGIEval) — Data Preparation for ACE")
    print("=" * 60)

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"Raw file not found: {RAW_FILE}\n"
            "Run the download step first (see eval/lsat_lr/prepare_data.py)."
        )

    with open(RAW_FILE, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    print(f"\nLoaded {len(raw)} raw examples from {RAW_FILE}")

    # Deterministic shuffle then split
    rng = random.Random(SEED)
    rng.shuffle(raw)

    train_raw = raw[:TRAIN_N]
    val_raw   = raw[TRAIN_N: TRAIN_N + VAL_N]
    test_raw  = raw[TRAIN_N + VAL_N:]

    print(f"Split: train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")

    train = [process_sample(x) for x in train_raw]
    val   = [process_sample(x) for x in val_raw]
    test  = [process_sample(x) for x in test_raw]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_jsonl(train, os.path.join(OUTPUT_DIR, "lsat_ar_train.jsonl"))
    save_jsonl(val,   os.path.join(OUTPUT_DIR, "lsat_ar_val.jsonl"))
    save_jsonl(test,  os.path.join(OUTPUT_DIR, "lsat_ar_test.jsonl"))

    # Label distribution check
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        dist = Counter(s["target"] for s in split)
        print(f"  {split_name} label dist: {dict(sorted(dist.items()))}")

    # sample_config.json
    config = {
        "lsat_ar": {
            "train_data": "./eval/lsat_ar/data/lsat_ar_train.jsonl",
            "val_data":   "./eval/lsat_ar/data/lsat_ar_val.jsonl",
            "test_data":  "./eval/lsat_ar/data/lsat_ar_test.jsonl",
        },
    }
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved config -> {config_path}")

    # Show one example
    ex = train[0]
    print(f"\n{'=' * 60}\nExample\n{'=' * 60}")
    print(f"Context (first 400 chars):\n{ex['context'][:400]}...")
    print(f"\nQuestion:\n{ex['question']}")
    print(f"\nTarget: {ex['target']}")
    print(f"\n{'=' * 60}\nDone.\n{'=' * 60}")


if __name__ == "__main__":
    main()
