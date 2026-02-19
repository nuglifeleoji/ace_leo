#!/usr/bin/env python3
"""
Prepare LSAT Logical Reasoning (AGIEval subset) for ACE framework.

Source: AGIEval benchmark — lsat-lr.jsonl
  https://github.com/ruixiangcui/AGIEval/tree/main/data/v1

LSAT-LR contains 510 logical reasoning questions from the Law School Admission
Test (LSAT).  Each question presents a short argument or passage and asks the
model to identify an assumption, weaken/strengthen the argument, draw an
inference, resolve a paradox, etc.  All questions are single-choice (A–E) with
a single correct answer, making evaluation trivially exact.

Split (deterministic, seed=42):
  Train : 300  — used for ACE offline training
  Val   : 100  — used for validation during training
  Test  : 110  — held-out final evaluation

Usage:
  python -m eval.lsat_lr.prepare_data
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

SEED = 42
RAW_FILE = "./eval/agieval/raw_data/lsat-lr.jsonl"
OUTPUT_DIR = "./eval/lsat_lr/data"

TRAIN_N = 300
VAL_N = 100
# Remaining goes to test (~110)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_options(options: List[str]) -> str:
    """Return options joined on separate lines (they already include '(A) ...')."""
    return "\n".join(options)


def process_sample(ex: Dict[str, Any]) -> Dict[str, str]:
    """Convert a raw AGIEval LSAT-LR example to ACE format.

    ACE format:
      context  — the argument / passage text
      question — question stem + formatted choices + answer instruction
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LSAT-LR (AGIEval) — Data Preparation for ACE")
    print("=" * 60)

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"Raw file not found: {RAW_FILE}\n"
            "Please run the download step first:\n"
            "  python -m eval.lsat_lr.download"
        )

    with open(RAW_FILE, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    print(f"\nLoaded {len(raw)} raw examples from {RAW_FILE}")

    # Shuffle deterministically then split
    rng = random.Random(SEED)
    rng.shuffle(raw)

    train_raw = raw[:TRAIN_N]
    val_raw = raw[TRAIN_N: TRAIN_N + VAL_N]
    test_raw = raw[TRAIN_N + VAL_N:]

    print(f"Split: train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")

    train = [process_sample(x) for x in train_raw]
    val = [process_sample(x) for x in val_raw]
    test = [process_sample(x) for x in test_raw]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_jsonl(train, os.path.join(OUTPUT_DIR, "lsat_lr_train.jsonl"))
    save_jsonl(val, os.path.join(OUTPUT_DIR, "lsat_lr_val.jsonl"))
    save_jsonl(test, os.path.join(OUTPUT_DIR, "lsat_lr_test.jsonl"))

    # Label distribution check (should be roughly uniform across A–E)
    from collections import Counter
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        dist = Counter(s["target"] for s in split)
        print(f"  {split_name} label dist: {dict(sorted(dist.items()))}")

    # sample_config.json
    config = {
        "lsat_lr": {
            "train_data": "./eval/lsat_lr/data/lsat_lr_train.jsonl",
            "val_data":   "./eval/lsat_lr/data/lsat_lr_val.jsonl",
            "test_data":  "./eval/lsat_lr/data/lsat_lr_test.jsonl",
        },
    }
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved config -> {config_path}")

    # Show one example
    ex = train[0]
    print(f"\n{'=' * 60}\nExample\n{'=' * 60}")
    print(f"Context:\n{ex['context']}")
    print(f"\nQuestion:\n{ex['question']}")
    print(f"\nTarget: {ex['target']}")
    print(f"\n{'=' * 60}\nDone.\n{'=' * 60}")


if __name__ == "__main__":
    main()
