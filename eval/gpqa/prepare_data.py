#!/usr/bin/env python3
"""
GPQA Diamond - Data Preparation for ACE

Downloads the GPQA Diamond dataset from HuggingFace and prepares it for ACE training.

GPQA (Graduate-Level Google-Proof Q&A) Diamond is a challenging multiple-choice
benchmark with 198 expert-level science questions (Physics, Chemistry, Biology).
Each question has 4 answer choices (A/B/C/D).

Since GPQA only has a single split of 198 samples, we create:
  - Train: 80 samples (~40%)
  - Val: 40 samples (~20%)
  - Test: 78 samples (~40%)
  - Train_small: 50 samples (subset of train for quick experiments)

Stratified by High-level domain (Physics, Chemistry, Biology) to maintain
subject balance across splits.

Usage:
    HF_TOKEN=hf_xxx python -m eval.gpqa.prepare_data
"""
import os
import json
import random
from datasets import load_dataset
from collections import Counter

OUTPUT_DIR = "./eval/gpqa/data"
SEED = 42
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def main():
    print("=" * 60)
    print("GPQA Diamond - Data Preparation for ACE")
    print("=" * 60)

    print("\nLoading GPQA Diamond dataset from HuggingFace...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", token=HF_TOKEN)["train"]
    print(f"Loaded {len(ds)} samples")

    rng = random.Random(SEED)

    all_samples = []
    for i, item in enumerate(ds):
        question_text = item["Question"]
        correct_answer = item["Correct Answer"]
        incorrect_1 = item["Incorrect Answer 1"]
        incorrect_2 = item["Incorrect Answer 2"]
        incorrect_3 = item["Incorrect Answer 3"]
        domain = item["High-level domain"]
        subdomain = item["Subdomain"]

        # Shuffle answer choices to randomize correct answer position
        choices = [
            ("A", correct_answer),
            ("B", incorrect_1),
            ("C", incorrect_2),
            ("D", incorrect_3),
        ]
        # Use a seeded shuffle per question so it's reproducible
        item_rng = random.Random(SEED + i)
        item_rng.shuffle(choices)

        # Find which letter the correct answer ended up at
        correct_letter = None
        for letter, text in choices:
            if text == correct_answer:
                correct_letter = letter
                break

        # Reassign letters after shuffle
        shuffled_choices = [(chr(65 + j), text) for j, (_, text) in enumerate(choices)]
        for letter, text in shuffled_choices:
            if text == correct_answer:
                correct_letter = letter
                break

        # Format choices
        formatted_choices = "\n".join(
            [f"{letter}) {text}" for letter, text in shuffled_choices]
        )

        # Construct the full question
        full_question = (
            f"{question_text}\n\n"
            f"{formatted_choices}\n\n"
            f"Answer with ONLY the letter of the correct choice (A, B, C, or D)."
        )

        all_samples.append({
            "context": "",  # GPQA questions are self-contained
            "question": full_question,
            "target": correct_letter,
            "domain": domain,
            "subdomain": subdomain,
            "answer_text": correct_answer,
            "n_choices": 4,
        })

    # Print statistics
    print(f"\nQuestion length: min={min(len(s['question']) for s in all_samples)}, "
          f"max={max(len(s['question']) for s in all_samples)}, "
          f"avg={sum(len(s['question']) for s in all_samples) / len(all_samples):.0f}")

    domain_dist = Counter(s["domain"] for s in all_samples)
    print(f"Domain distribution: {domain_dist}")

    answer_dist = Counter(s["target"] for s in all_samples)
    print(f"Answer distribution (after shuffle): {answer_dist}")

    # Stratified split by domain
    samples_by_domain = {k: [s for s in all_samples if s["domain"] == k] for k in domain_dist.keys()}

    train_samples = []
    val_samples = []
    test_samples = []

    # Target sizes
    target_train_size = 80
    target_val_size = 40
    target_test_size = 78  # remainder

    for domain_key in sorted(samples_by_domain.keys()):
        current_samples = samples_by_domain[domain_key]
        rng.shuffle(current_samples)

        # Proportional allocation
        prop_train = target_train_size / len(all_samples)
        prop_val = target_val_size / len(all_samples)

        n_train = max(1, round(len(current_samples) * prop_train))
        n_val = max(1, round(len(current_samples) * prop_val))

        train_samples.extend(current_samples[:n_train])
        val_samples.extend(current_samples[n_train: n_train + n_val])
        test_samples.extend(current_samples[n_train + n_val:])

    # Trim to exact sizes if over
    if len(train_samples) > target_train_size:
        train_samples = rng.sample(train_samples, target_train_size)
    if len(val_samples) > target_val_size:
        val_samples = rng.sample(val_samples, target_val_size)
    if len(test_samples) > target_test_size:
        test_samples = rng.sample(test_samples, target_test_size)

    print("\n" + "=" * 60)
    print("Data Split")
    print("=" * 60)
    for name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        domains = Counter(s["domain"] for s in samples)
        answers = Counter(s["target"] for s in samples)
        print(f"  {name:5s}: {len(samples)} samples | domains={domains} | answers={answers}")

    # Save data
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = os.path.join(OUTPUT_DIR, f"gpqa_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved {len(samples)} samples to {path}")

    # Small train set
    train_small = rng.sample(train_samples, min(len(train_samples), 50))
    path = os.path.join(OUTPUT_DIR, "gpqa_train_50.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for s in train_small:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(train_small)} samples to {path} (small train set)")

    # Print example
    print("\n" + "=" * 60)
    print("Example Sample")
    print("=" * 60)
    ex = train_samples[0]
    print(f"Domain: {ex['domain']}, Subdomain: {ex.get('subdomain', 'N/A')}")
    print(f"Question (first 300 chars):\n{ex['question'][:300]}...")
    print(f"\nTarget: {ex['target']}")

    # Save config
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    config_data = {
        "gpqa": {
            "train_data": "./eval/gpqa/data/gpqa_train.jsonl",
            "val_data": "./eval/gpqa/data/gpqa_val.jsonl",
            "test_data": "./eval/gpqa/data/gpqa_test.jsonl"
        },
        "gpqa_small": {
            "train_data": "./eval/gpqa/data/gpqa_train_50.jsonl",
            "val_data": "./eval/gpqa/data/gpqa_val.jsonl",
            "test_data": "./eval/gpqa/data/gpqa_test.jsonl"
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
