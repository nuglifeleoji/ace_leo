#!/usr/bin/env python3
"""
Prepare MuSR Object Placements data for ACE framework.

Downloads the MuSR dataset from HuggingFace and converts the
object_placements subtask into ACE-compatible JSONL samples.

Object Placements (256 samples):
  Given a narrative about people and objects in various locations,
  track object movements and predict where a person would look
  for a specific object. 2-5 choices (mostly 3-4).

  Key reasoning skills:
  - Track object movements chronologically
  - Distinguish between who knows about movements vs who doesn't
  - Identify the last known position from a character's perspective

Each sample is formulated as:
  - context: Full narrative story (3,300-7,200 chars)
  - question: "Which location is most likely..." + lettered choices + instruction
  - target: Correct answer letter (A, B, C, D, or E)

Data split:
  - Train: 103 samples
  - Val:   53 samples
  - Test:  100 samples
  - Train_small: 50 subset for quick experiments

Usage:
    python -m eval.musr_location.prepare_data
"""
import os
import ast
import json
import random
from collections import Counter

SEED = 42
OUTPUT_DIR = "./eval/musr_location/data"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_choices(choices_list: list) -> str:
    """
    Format choices as lettered options.

    Input:  ['piano', "producer's desk", 'recording booth']
    Output: "A) piano\nB) producer's desk\nC) recording booth"
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
    Convert a single MuSR object placement example into ACE format.

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
    n_choices = len(choices_list)

    # Format the choices with letters
    formatted_choices = format_choices(choices_list)

    # Build letter options string for instruction
    letter_options = ", ".join(LETTERS[:n_choices])

    # Build the question with choices and instruction
    question = (
        f"{question_text}\n\n"
        f"{formatted_choices}\n\n"
        f"Answer with ONLY the letter of the correct choice ({letter_options})."
    )

    # Target is just the letter
    answer_letter = get_answer_letter(answer_index)

    return {
        "context": narrative,
        "question": question,
        "target": answer_letter,
        "answer_choice": answer_choice,
        "n_choices": n_choices,
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
    print("MuSR Object Placements - Data Preparation for ACE")
    print("=" * 60)

    # Load dataset (only object_placements)
    print("\nLoading MuSR dataset from HuggingFace...")
    ds = load_dataset("TAUR-Lab/MuSR")
    data = ds["object_placements"]
    print(f"Loaded {len(data)} object placement samples")

    # Process all samples
    all_samples = []
    for example in data:
        sample = process_sample(example)
        all_samples.append(sample)

    # Statistics
    context_lens = [len(s["context"]) for s in all_samples]
    ans_dist = Counter(s["target"] for s in all_samples)
    n_choices_dist = Counter(s["n_choices"] for s in all_samples)
    print(f"\nContext length: min={min(context_lens)}, max={max(context_lens)}, "
          f"avg={sum(context_lens)//len(context_lens)}")
    print(f"Answer distribution: {dict(sorted(ans_dist.items()))}")
    print(f"Number of choices: {dict(sorted(n_choices_dist.items()))}")

    # Split: 103 train / 53 val / 100 test
    rng = random.Random(SEED)
    rng.shuffle(all_samples)

    train_samples = all_samples[:103]
    val_samples = all_samples[103:156]
    test_samples = all_samples[156:256]

    print(f"\n{'='*60}")
    print(f"Data Split")
    print(f"{'='*60}")
    for name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        ans = Counter(s["target"] for s in samples)
        nc = Counter(s["n_choices"] for s in samples)
        lens = [len(s["context"]) for s in samples]
        print(f"  {name:5s}: {len(samples)} samples | "
              f"answers={dict(sorted(ans.items()))} | "
              f"choices={dict(sorted(nc.items()))} | "
              f"context: {min(lens)}-{max(lens)} chars")

    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving to {OUTPUT_DIR}/...")

    save_jsonl(train_samples, os.path.join(OUTPUT_DIR, "location_train.jsonl"))
    save_jsonl(val_samples, os.path.join(OUTPUT_DIR, "location_val.jsonl"))
    save_jsonl(test_samples, os.path.join(OUTPUT_DIR, "location_test.jsonl"))

    # Create smaller training subset (50 samples) for quick experiments
    train_small = train_samples[:50]
    save_jsonl(train_small, os.path.join(OUTPUT_DIR, "location_train_50.jsonl"))

    # Print example
    print(f"\n{'='*60}")
    print(f"Example Sample")
    print(f"{'='*60}")
    ex = train_samples[0]
    print(f"Context (first 400 chars): {ex['context'][:400]}...")
    print(f"\nQuestion:\n{ex['question']}")
    print(f"\nTarget: {ex['target']} ({ex['answer_choice']})")
    print(f"Choices: {ex['n_choices']}")

    # Save sample_config.json
    config = {
        "musr_location": {
            "train_data": "./eval/musr_location/data/location_train.jsonl",
            "val_data": "./eval/musr_location/data/location_val.jsonl",
            "test_data": "./eval/musr_location/data/location_test.jsonl"
        },
        "musr_location_small": {
            "train_data": "./eval/musr_location/data/location_train_50.jsonl",
            "val_data": "./eval/musr_location/data/location_val.jsonl",
            "test_data": "./eval/musr_location/data/location_test.jsonl"
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
