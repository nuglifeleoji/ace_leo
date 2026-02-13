#!/usr/bin/env python3
"""
Prepare LogiQA 2.0 data for ACE framework.

LogiQA 2.0 is a logical reasoning benchmark derived from Chinese civil
service exams (translated to English). It covers:
  - Sufficient Conditional Reasoning
  - Necessary Conditional Reasoning
  - Conjunctive Reasoning
  - Disjunctive Reasoning
  - Categorical Reasoning

Each sample:
  - context: Passage describing a scenario
  - question: Logical reasoning question + 4 options (A-D)
  - target: Correct answer letter (A, B, C, or D)

Data split (stratified by reasoning type):
  - Train: 200 samples
  - Val:   100 samples
  - Test:  200 samples
  - Train_small: 50 subset for quick experiments

Usage:
    python -m eval.logiqa.prepare_data
"""
import os
import json
import random
from collections import Counter
from datasets import load_dataset

SEED = 42
OUTPUT_DIR = "./eval/logiqa/data"
LETTERS = "ABCD"


def parse_entry(raw_text: str) -> dict:
    """Parse the JSON string inside the 'text' column."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def get_primary_type(type_dict: dict) -> str:
    """Get the primary reasoning type for stratification."""
    if not type_dict:
        return "unknown"
    for typ, val in type_dict.items():
        if val:
            return typ
    return "unknown"


def process_sample(entry: dict) -> dict:
    """Convert a parsed LogiQA entry into ACE format."""
    passage = entry["text"]
    question_text = entry["question"]
    options = entry["options"]
    answer_idx = entry["answer"]  # 0-indexed integer
    answer_letter = LETTERS[answer_idx]
    type_dict = entry.get("type", {})

    # Format options
    formatted_options = "\n".join(
        f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options)
    )

    question = (
        f"Passage: {passage}\n\n"
        f"Question: {question_text}\n\n"
        f"{formatted_options}\n\n"
        f"Answer with ONLY the letter of the correct choice (A, B, C, or D). "
        f"Do not include any explanation."
    )

    return {
        "context": "",  # passage is embedded in question for this task
        "question": question,
        "target": answer_letter,
        "reasoning_type": get_primary_type(type_dict),
        "original_answer_idx": answer_idx,
    }


def save_jsonl(samples: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples to {path}")


def main():
    print("=" * 60)
    print("LogiQA 2.0 — Data Preparation for ACE")
    print("=" * 60)

    print("\nLoading LogiQA 2.0 from HuggingFace (datatune/LogiQA2.0)...")
    ds = load_dataset("datatune/LogiQA2.0")
    print(f"Splits: {list(ds.keys())}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} samples")

    # Use the train split (63K+ samples) since it has all answers
    # We'll create our own train/val/test from it
    raw_data = ds["train"]
    print(f"\nUsing train split: {len(raw_data)} raw samples")

    # Parse and filter valid entries
    all_entries = []
    parse_errors = 0
    missing_answer = 0

    for item in raw_data:
        entry = parse_entry(item["text"])
        if entry is None:
            parse_errors += 1
            continue
        if "answer" not in entry or "options" not in entry:
            missing_answer += 1
            continue
        if len(entry["options"]) != 4:
            continue  # Only keep 4-option questions
        all_entries.append(entry)

    print(f"Valid entries: {len(all_entries)} (parse errors: {parse_errors}, "
          f"missing answer: {missing_answer})")

    # Process all samples
    all_samples = [process_sample(e) for e in all_entries]

    # Statistics
    type_dist = Counter(s["reasoning_type"] for s in all_samples)
    ans_dist = Counter(s["target"] for s in all_samples)
    q_lens = [len(s["question"]) for s in all_samples]

    print(f"\nReasoning type distribution:")
    for typ, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {typ}: {cnt}")
    print(f"\nAnswer distribution: {dict(sorted(ans_dist.items()))}")
    print(f"Question length: min={min(q_lens)}, max={max(q_lens)}, "
          f"avg={sum(q_lens)//len(q_lens)}")

    # Stratified split by reasoning type: 200 train / 100 val / 200 test
    rng = random.Random(SEED)

    samples_by_type = {}
    for s in all_samples:
        t = s["reasoning_type"]
        if t not in samples_by_type:
            samples_by_type[t] = []
        samples_by_type[t].append(s)

    for t in samples_by_type:
        rng.shuffle(samples_by_type[t])

    target_train = 200
    target_val = 100
    target_test = 200

    train_final, val_final, test_final = [], [], []

    for t in sorted(samples_by_type.keys()):
        pool = samples_by_type[t]
        prop = len(pool) / len(all_samples)

        n_train = max(1, round(target_train * prop))
        n_val = max(1, round(target_val * prop))
        n_test = max(1, round(target_test * prop))

        train_final.extend(pool[:n_train])
        val_final.extend(pool[n_train:n_train + n_val])
        test_final.extend(pool[n_train + n_val:n_train + n_val + n_test])

    # Shuffle and trim
    rng.shuffle(train_final)
    rng.shuffle(val_final)
    rng.shuffle(test_final)
    train_final = train_final[:target_train]
    val_final = val_final[:target_val]
    test_final = test_final[:target_test]

    print(f"\n{'=' * 60}")
    print("Data Split")
    print(f"{'=' * 60}")
    for name, samples in [("Train", train_final), ("Val", val_final),
                           ("Test", test_final)]:
        types = Counter(s["reasoning_type"] for s in samples)
        ans = Counter(s["target"] for s in samples)
        print(f"  {name:5s}: {len(samples)} samples")
        print(f"         answer dist: {dict(sorted(ans.items()))}")
        print(f"         types: {dict(sorted(types.items(), key=lambda x:-x[1]))}")

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
    print(f"Type: {ex['reasoning_type']}")
    print(f"Question (first 500 chars):\n{ex['question'][:500]}...")
    print(f"\nTarget: {ex['target']}")

    # Save config
    config = {
        "logiqa": {
            "train_data": "./eval/logiqa/data/train.jsonl",
            "val_data": "./eval/logiqa/data/val.jsonl",
            "test_data": "./eval/logiqa/data/test.jsonl"
        },
        "logiqa_small": {
            "train_data": "./eval/logiqa/data/train_50.jsonl",
            "val_data": "./eval/logiqa/data/val.jsonl",
            "test_data": "./eval/logiqa/data/test.jsonl"
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
