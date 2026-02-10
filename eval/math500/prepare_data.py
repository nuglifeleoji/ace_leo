"""
MATH-500 - Data Preparation for ACE

Downloads the MATH-500 benchmark from HuggingFace (HuggingFaceH4/MATH-500)
and creates train/val/test splits.

Dataset: HuggingFaceH4/MATH-500
- 500 math problems across 7 subjects and 5 difficulty levels
- Each problem has a step-by-step solution and a final answer
- Answers can be numeric (318) or symbolic LaTeX (182)

Split design (stratified by subject × level):
  - Train: 200 samples (~40%)
  - Val:   100 samples (~20%)
  - Test:  200 samples (~40%)
  - Train_small: 50 samples (subset of train for quick experiments)

Usage:
    python -m eval.math500.prepare_data
"""
import os
import json
import random
from collections import Counter, defaultdict


OUTPUT_DIR = "./eval/math500/data"
SEED = 42


def main():
    print("=" * 60)
    print("MATH-500 - Data Preparation for ACE")
    print("=" * 60)

    print("\nLoading MATH-500 dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded {len(ds)} math problems")

    rng = random.Random(SEED)

    # Process all samples
    all_samples = []
    for i, item in enumerate(ds):
        problem = item["problem"]
        solution = item["solution"]
        answer = item["answer"]
        subject = item["subject"]
        level = item["level"]

        # Build the question: problem + instruction to box the answer
        question = (
            f"{problem}\n\n"
            f"Please solve this step by step. "
            f"Put your final answer in \\boxed{{}}."
        )

        all_samples.append({
            "context": "",  # No separate context for math problems
            "question": question,
            "target": answer,
            "subject": subject,
            "level": level,
            "solution": solution,
            "unique_id": item["unique_id"]
        })

    # Statistics
    print(f"\nSubject distribution:")
    for s, c in Counter(s["subject"] for s in all_samples).most_common():
        print(f"  {s}: {c}")

    print(f"\nLevel distribution:")
    for l, c in sorted(Counter(s["level"] for s in all_samples).items()):
        print(f"  Level {l}: {c}")

    # Stratified split by (subject, level)
    strata = defaultdict(list)
    for s in all_samples:
        key = (s["subject"], s["level"])
        strata[key].append(s)

    train_samples = []
    val_samples = []
    test_samples = []

    target_train = 200
    target_val = 100
    target_test = 200

    for key in sorted(strata.keys()):
        samples = strata[key]
        rng.shuffle(samples)

        n = len(samples)
        # Proportional allocation
        n_train = max(1, round(n * target_train / len(all_samples)))
        n_val = max(1, round(n * target_val / len(all_samples)))
        # Rest goes to test
        n_test = n - n_train - n_val

        if n_test < 0:
            # Very small stratum, just split evenly
            n_train = n // 3
            n_val = n // 3
            n_test = n - n_train - n_val

        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train + n_val])
        test_samples.extend(samples[n_train + n_val:])

    # Trim to exact target sizes if we overshot
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    train_samples = train_samples[:target_train]
    val_samples = val_samples[:target_val]
    test_samples = test_samples[:target_test]

    print("\n" + "=" * 60)
    print("Data Split")
    print("=" * 60)
    for name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        subj_dist = Counter(s["subject"] for s in samples)
        lvl_dist = Counter(s["level"] for s in samples)
        print(f"  {name:5s}: {len(samples)} samples")
        print(f"         Subjects: {dict(sorted(subj_dist.items()))}")
        print(f"         Levels: {dict(sorted(lvl_dist.items()))}")

    # Save data files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = os.path.join(OUTPUT_DIR, f"math500_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved {len(samples)} samples to {path}")

    # Small train set (50 samples, stratified by subject)
    subject_samples = defaultdict(list)
    for s in train_samples:
        subject_samples[s["subject"]].append(s)

    train_small = []
    per_subject = max(1, 50 // len(subject_samples))
    for subj in sorted(subject_samples.keys()):
        available = subject_samples[subj]
        n_take = min(len(available), per_subject)
        train_small.extend(rng.sample(available, n_take))

    # Top up to 50 if needed
    remaining = [s for s in train_samples if s not in train_small]
    if len(train_small) < 50 and remaining:
        extra = rng.sample(remaining, min(50 - len(train_small), len(remaining)))
        train_small.extend(extra)

    train_small = train_small[:50]
    path = os.path.join(OUTPUT_DIR, "math500_train_50.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for s in train_small:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(train_small)} samples to {path} (small train set)")

    # Example
    print("\n" + "=" * 60)
    print("Example Sample")
    print("=" * 60)
    ex = train_samples[0]
    print(f"Subject: {ex['subject']}, Level: {ex['level']}")
    print(f"Question:\n{ex['question'][:300]}...")
    print(f"\nTarget: {ex['target']}")
    print(f"Solution (first 200 chars): {ex['solution'][:200]}...")

    # Save config
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    config_data = {
        "math500": {
            "train_data": "./eval/math500/data/math500_train.jsonl",
            "val_data": "./eval/math500/data/math500_val.jsonl",
            "test_data": "./eval/math500/data/math500_test.jsonl"
        },
        "math500_small": {
            "train_data": "./eval/math500/data/math500_train_50.jsonl",
            "val_data": "./eval/math500/data/math500_val.jsonl",
            "test_data": "./eval/math500/data/math500_test.jsonl"
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    print(f"\nSaved config to {config_path}")

    print("\n" + "=" * 60)
    print("Done! Ready for ACE training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
