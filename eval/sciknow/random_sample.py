"""
Random sampling baseline for SciKnowEval Chemistry L3.

Generates random subsets of sizes [5, 10, 20, 30, 40, 50, 80] from the
1000 training samples and updates sample_config.json accordingly.

Usage:
    python -m eval.sciknow.random_sample
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "./eval/sciknow/data/sciknow_chem_l3_train.jsonl"
OUTPUT_DIR  = "./eval/sciknow/data"
CONFIG_PATH = "./eval/sciknow/data/sample_config.json"
VAL_PATH    = "./eval/sciknow/data/sciknow_chem_l3_val.jsonl"
TEST_PATH   = "./eval/sciknow/data/sciknow_chem_l3_test.jsonl"

SIZES = [5, 10, 20, 30, 40, 50, 80]
SEED  = 42


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} samples → {path}")


def update_config(sizes: List[int]) -> None:
    config: Dict = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)

    for k in sizes:
        config[f"sciknow_chem_l3_random{k}"] = {
            "train_data": f"./eval/sciknow/data/sciknow_train_random{k}.jsonl",
            "val_data":   VAL_PATH,
            "test_data":  TEST_PATH,
        }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


def main() -> None:
    print("=" * 60)
    print("  SciKnowEval — Random Subset Sampling")
    print("=" * 60)

    train_data = load_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_data)} training samples")

    rng = random.Random(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for k in SIZES:
        subset = rng.sample(train_data, k)
        out_path = os.path.join(OUTPUT_DIR, f"sciknow_train_random{k}.jsonl")
        save_jsonl(subset, out_path)

    update_config(SIZES)
    print("Done.")


if __name__ == "__main__":
    main()
