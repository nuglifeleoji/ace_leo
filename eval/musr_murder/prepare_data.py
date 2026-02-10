#!/usr/bin/env python3
"""
Prepare MuSR Murder Mysteries data for ACE framework.

Downloads the MuSR dataset from HuggingFace and converts the
murder_mysteries subtask into ACE-compatible JSONL samples.

Murder Mysteries (250 samples):
  Given a detective narrative, analyze evidence, motives, and alibis
  to identify the most likely murderer. Binary choice (2 suspects).

Each sample is formulated as:
  - context: Full detective narrative (3,700-7,300 chars)
  - question: "Who is the most likely murderer?" + lettered choices + instruction
  - target: Correct answer letter (A or B)

Data split:
  - Train: 100 samples
  - Val:   50 samples
  - Test:  100 samples
  - Train_small: 50 subset for quick experiments

Usage:
    python -m eval.musr.prepare_data
"""
import os
import ast
import json
import random
from collections import Counter

SEED = 42
OUTPUT_DIR = "./eval/musr/data"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_choices(choices_list: list) -> str:
    """
    Format choices as lettered options.

    Input:  ['Mackenzie', 'Ana']
    Output: 'A) Mackenzie\nB) Ana'
    """
    lines = []
    for i, choice in enumerate(choices_list):
        lines.append(f"{LETTERS[i]}) {choice}")
    return "\n".join(lines)


def get_answer_letter(answer_index: int) -> str:
    """Convert numeric index to letter (0 -> 'A', 1 -> 'B', etc.)."""
    return LETTERS[answer_index]


def process_sample(example: dict) -> dict:
    """
    Convert a single MuSR murder mystery example into ACE format.

    Args:
        example: Raw example from HuggingFace dataset

    Returns:
        Dict with keys: context, question, target, answer_choice, n_choices
    """
    narrative = example["narrative"]
    question_text = example["question"]
    choices_str = example["choices"]
    answer_index = example["answer_index"]
    answer_choice = example["answer_choice"]

    # Parse choices from string representation
    choices_list = ast.literal_eval(choices_str)

    # Format the choices with letters
    formatted_choices = format_choices(choices_list)

    # Build the question with choices and instruction
    question = (
        f"{question_text}\n\n"
        f"{formatted_choices}\n\n"
        f"Answer with ONLY the letter of the correct choice (e.g., A or B)."
    )

    # Target is just the letter
    answer_letter = get_answer_letter(answer_index)

    return {
        "context": narrative,
        "question": question,
        "target": answer_letter,
        "answer_choice": answer_choice,
        "n_choices": len(choices_list),
    }


def save_jsonl(samples: list, path: str):
    """Save samples to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=True) + "\n")
    print(f"  Saved {len(samples)} samples to {path}")


def main():
    from datasets import load_dataset

    print("=" * 60)
    print("MuSR Murder Mysteries - Data Preparation for ACE")
    print("=" * 60)

    # Load dataset (only murder_mysteries)
    print("\nLoading MuSR dataset from HuggingFace...")
    ds = load_dataset("TAUR-Lab/MuSR")
    data = ds["murder_mysteries"]
    print(f"Loaded {len(data)} murder mystery samples")

    # Process all samples
    all_samples = []
    for example in data:
        sample = process_sample(example)
        all_samples.append(sample)

    # Statistics
    context_lens = [len(s["context"]) for s in all_samples]
    ans_dist = Counter(s["target"] for s in all_samples)
    print(f"\nContext length: min={min(context_lens)}, max={max(context_lens)}, "
          f"avg={sum(context_lens)//len(context_lens)}")
    print(f"Answer distribution: {dict(sorted(ans_dist.items()))}")
    print(f"All samples have {all_samples[0]['n_choices']} choices (binary)")

    # Split: 100 train / 50 val / 100 test
    rng = random.Random(SEED)
    rng.shuffle(all_samples)

    train_samples = all_samples[:100]
    val_samples = all_samples[100:150]
    test_samples = all_samples[150:250]

    print(f"\n{'='*60}")
    print(f"Data Split")
    print(f"{'='*60}")
    for name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        ans = Counter(s["target"] for s in samples)
        lens = [len(s["context"]) for s in samples]
        print(f"  {name:5s}: {len(samples)} samples | "
              f"A={ans.get('A',0)}, B={ans.get('B',0)} | "
              f"context: {min(lens)}-{max(lens)} chars")

    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving to {OUTPUT_DIR}/...")

    save_jsonl(train_samples, os.path.join(OUTPUT_DIR, "musr_train.jsonl"))
    save_jsonl(val_samples, os.path.join(OUTPUT_DIR, "musr_val.jsonl"))
    save_jsonl(test_samples, os.path.join(OUTPUT_DIR, "musr_test.jsonl"))

    # Create smaller training subset (50 samples) for quick experiments
    train_small = train_samples[:50]
    save_jsonl(train_small, os.path.join(OUTPUT_DIR, "musr_train_50.jsonl"))

    # Print example
    print(f"\n{'='*60}")
    print(f"Example Sample")
    print(f"{'='*60}")
    ex = train_samples[0]
    print(f"Context (first 400 chars):\n{ex['context'][:400]}...")
    print(f"\nQuestion:\n{ex['question']}")
    print(f"\nTarget: {ex['target']} ({ex['answer_choice']})")

    # Save sample_config.json
    config = {
        "musr": {
            "train_data": "./eval/musr/data/musr_train.jsonl",
            "val_data": "./eval/musr/data/musr_val.jsonl",
            "test_data": "./eval/musr/data/musr_test.jsonl"
        },
        "musr_small": {
            "train_data": "./eval/musr/data/musr_train_50.jsonl",
            "val_data": "./eval/musr/data/musr_val.jsonl",
            "test_data": "./eval/musr/data/musr_test.jsonl"
        }
    }
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved config to {config_path}")

    print(f"\n{'='*60}")
    print(f"Done! Ready for ACE training.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
