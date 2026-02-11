#!/usr/bin/env python3
"""
Prepare CLUTRR data for ACE framework.

CLUTRR (Compositional Language Understanding with Textual Reasoning
for Unseen Relationships) tests kinship reasoning.

Given a story describing family relationships, the model must infer
the relationship between two specified people by composing multiple
relationship edges (multi-hop reasoning).

Uses kendrivp/CLUTRR_v1_extracted which has 2-10 hop reasoning chains.
We focus on 3-6 hops where models are likely to struggle but can learn.

Each sample:
  - context: Story describing family interactions
  - question: "What is the relationship of X to Y?" + instruction
  - target: Kinship relation (e.g., "grandson", "aunt", "daughter-in-law")

Data split (stratified by hop count):
  - Train: 200 samples
  - Val:   100 samples
  - Test:  200 samples
  - Train_small: 50 subset for quick experiments

Usage:
    python -m eval.clutrr.prepare_data
"""
import os
import json
import random
from collections import Counter
from datasets import load_dataset

SEED = 42
OUTPUT_DIR = "./eval/clutrr/data"

# All valid kinship relations in CLUTRR
VALID_RELATIONS = [
    "father", "mother", "son", "daughter",
    "grandfather", "grandmother", "grandson", "granddaughter",
    "brother", "sister", "uncle", "aunt", "nephew", "niece",
    "father-in-law", "mother-in-law", "son-in-law", "daughter-in-law",
    "husband", "wife"
]


def get_hop_count(f_comb: str) -> int:
    """Get the number of reasoning hops from the f_comb field."""
    return f_comb.count("-") + 1


def process_sample(example: dict) -> dict:
    """Convert a single CLUTRR example into ACE format."""
    story = example["story"]
    query = example["query"]  # e.g., "('Donald', 'Gilbert')"
    answer = example["target_text"]  # e.g., "nephew"
    f_comb = example["f_comb"]  # e.g., "son-mother-son-uncle-son"
    hop_count = get_hop_count(f_comb)

    # Parse query to get person names
    # query format: "('Name1', 'Name2')"
    query_clean = query.strip("()").replace("'", "").replace('"', '')
    parts = [p.strip() for p in query_clean.split(",")]
    person1 = parts[0] if len(parts) >= 1 else "Person1"
    person2 = parts[1] if len(parts) >= 2 else "Person2"

    question = (
        f"Based on the story above, what is the family relationship of "
        f"{person1} to {person2}?\n\n"
        f"Answer with ONLY the relationship word (e.g., father, mother, "
        f"grandson, aunt, nephew, daughter-in-law, etc.). "
        f"Do not include any explanation."
    )

    return {
        "context": story,
        "question": question,
        "target": answer,
        "hop_count": hop_count,
        "reasoning_chain": f_comb,
        "query": query,
    }


def save_jsonl(samples: list, path: str):
    """Save samples to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples to {path}")


def main():
    print("=" * 60)
    print("CLUTRR — Data Preparation for ACE")
    print("=" * 60)

    print("\nLoading CLUTRR from HuggingFace (kendrivp/CLUTRR_v1_extracted)...")
    ds = load_dataset("kendrivp/CLUTRR_v1_extracted")

    # Use test split which has all hop levels
    raw_data = list(ds["test"])
    print(f"Total test samples: {len(raw_data)}")

    # Focus on 3-6 hops (where models struggle but can learn)
    # 2-hop is likely too easy, 7+ hops has very few samples
    filtered = []
    for ex in raw_data:
        hop = get_hop_count(ex["f_comb"])
        if 3 <= hop <= 6:
            filtered.append(ex)

    print(f"Filtered to 3-6 hops: {len(filtered)} samples")

    # Process all samples
    all_samples = [process_sample(ex) for ex in filtered]

    # Statistics
    hop_dist = Counter(s["hop_count"] for s in all_samples)
    ans_dist = Counter(s["target"] for s in all_samples)
    story_lens = [len(s["context"]) for s in all_samples]
    print(f"\nHop count distribution: {dict(sorted(hop_dist.items()))}")
    print(f"Answer distribution (top 10): {dict(sorted(ans_dist.items(), key=lambda x:-x[1])[:10])}")
    print(f"Story length: min={min(story_lens)}, max={max(story_lens)}, avg={sum(story_lens)//len(story_lens)}")

    # Stratified split by hop count: 200 train / 100 val / 200 test
    rng = random.Random(SEED)

    samples_by_hop = {}
    for s in all_samples:
        h = s["hop_count"]
        if h not in samples_by_hop:
            samples_by_hop[h] = []
        samples_by_hop[h].append(s)

    for h in samples_by_hop:
        rng.shuffle(samples_by_hop[h])

    target_train = 200
    target_val = 100
    target_test = 200
    total_target = target_train + target_val + target_test

    train_final, val_final, test_final = [], [], []

    for h in sorted(samples_by_hop.keys()):
        pool = samples_by_hop[h]
        prop = len(pool) / len(all_samples)

        n_train = max(1, int(target_train * prop))
        n_val = max(1, int(target_val * prop))
        n_test = max(1, int(target_test * prop))

        # Don't exceed available samples
        total_needed = n_train + n_val + n_test
        if total_needed > len(pool):
            n_train = min(n_train, len(pool) // 3)
            n_val = min(n_val, len(pool) // 3)
            n_test = min(n_test, len(pool) // 3)

        train_final.extend(pool[:n_train])
        val_final.extend(pool[n_train:n_train + n_val])
        test_final.extend(pool[n_train + n_val:n_train + n_val + n_test])

    # Trim to target sizes
    train_final = train_final[:target_train]
    val_final = val_final[:target_val]
    test_final = test_final[:target_test]

    print(f"\n{'=' * 60}")
    print("Data Split")
    print(f"{'=' * 60}")
    for name, samples in [("Train", train_final), ("Val", val_final), ("Test", test_final)]:
        hop = Counter(s["hop_count"] for s in samples)
        ans = Counter(s["target"] for s in samples)
        print(f"  {name:5s}: {len(samples)} samples | hops: {dict(sorted(hop.items()))} | "
              f"top answers: {dict(sorted(ans.items(), key=lambda x:-x[1])[:5])}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving to {OUTPUT_DIR}/...")

    save_jsonl(train_final, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val_final, os.path.join(OUTPUT_DIR, "val.jsonl"))
    save_jsonl(test_final, os.path.join(OUTPUT_DIR, "test.jsonl"))

    train_small = train_final[:50]
    save_jsonl(train_small, os.path.join(OUTPUT_DIR, "train_50.jsonl"))

    # Example
    print(f"\n{'=' * 60}")
    print("Example Sample")
    print(f"{'=' * 60}")
    ex = train_final[0]
    print(f"Hops: {ex['hop_count']} | Chain: {ex['reasoning_chain']}")
    print(f"Story: {ex['context']}")
    print(f"Question: {ex['question']}")
    print(f"Target: {ex['target']}")

    # Save config
    config = {
        "clutrr": {
            "train_data": "./eval/clutrr/data/train.jsonl",
            "val_data": "./eval/clutrr/data/val.jsonl",
            "test_data": "./eval/clutrr/data/test.jsonl"
        },
        "clutrr_small": {
            "train_data": "./eval/clutrr/data/train_50.jsonl",
            "val_data": "./eval/clutrr/data/val.jsonl",
            "test_data": "./eval/clutrr/data/test.jsonl"
        }
    }
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved config to {config_path}")

    print(f"\n{'=' * 60}")
    print("Done! Ready for ACE training.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
