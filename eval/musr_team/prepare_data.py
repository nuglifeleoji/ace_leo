"""
MuSR Team Allocation - Data Preparation for ACE

Downloads the team_allocation split from MuSR and creates train/val/test splits.

Dataset: TAUR-Lab/MuSR (team_allocation split)
- 250 samples total, 3-choice multiple choice (A/B/C)
- Context: ~2350-3841 chars (team constraint narratives)
- Task: Allocate people to roles based on constraints

Split design:
  - Train: 100 samples (~40%)
  - Val:    50 samples (~20%)
  - Test:  100 samples (~40%)
  - Train_small: 50 samples (subset of train)

Stratified by answer letter (A/B/C) to maintain balanced distributions.

Usage:
    python -m eval.musr_team.prepare_data
"""
import os
import json
import ast
import random
from collections import Counter


OUTPUT_DIR = "./eval/musr_team/data"
SEED = 42


def main():
    print("=" * 60)
    print("MuSR Team Allocation - Data Preparation for ACE")
    print("=" * 60)

    print("\nLoading MuSR dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("TAUR-Lab/MuSR", split="team_allocation")
    print(f"Loaded {len(ds)} team allocation samples")

    rng = random.Random(SEED)

    all_samples = []
    for i, item in enumerate(ds):
        narrative = item["narrative"]
        question_text = item["question"]
        choices_str = item["choices"]
        answer_index = item["answer_index"]

        # choices is a string repr of a Python list, use ast.literal_eval
        choices = ast.literal_eval(choices_str)

        # Format choices for the question
        formatted_choices = "\n".join(
            [f"{chr(65 + j)}) {choice}" for j, choice in enumerate(choices)]
        )

        # Determine the correct answer letter
        correct_letter = chr(65 + answer_index)
        correct_choice_text = choices[answer_index]

        # Construct the full question for the model
        full_question = (
            f"{question_text}\n\n"
            f"{formatted_choices}\n\n"
            f"Answer with ONLY the letter of the correct choice (A, B, or C)."
        )

        all_samples.append({
            "context": narrative,
            "question": full_question,
            "target": correct_letter,
            "subtask": "team_allocation",
            "answer_choice": correct_choice_text,
            "n_choices": len(choices)
        })

    print(f"\nContext length: min={min(len(s['context']) for s in all_samples)}, "
          f"max={max(len(s['context']) for s in all_samples)}, "
          f"avg={sum(len(s['context']) for s in all_samples) / len(all_samples):.0f}")

    answer_dist = Counter(s["target"] for s in all_samples)
    print(f"Answer distribution: {dict(sorted(answer_dist.items()))}")
    print(f"All samples have {all_samples[0]['n_choices']} choices (3-way)")

    # -----------------------------------------------------------
    # Stratified split by answer (A/B/C)
    # -----------------------------------------------------------
    samples_by_answer = {
        k: [s for s in all_samples if s["target"] == k]
        for k in sorted(answer_dist.keys())
    }

    train_samples = []
    val_samples = []
    test_samples = []

    target_train_size = 100
    target_val_size = 50
    target_test_size = 100

    for answer_key in sorted(samples_by_answer.keys()):
        current_samples = samples_by_answer[answer_key]
        rng.shuffle(current_samples)

        # Distribute samples proportionally
        prop_train = target_train_size / len(all_samples)
        prop_val = target_val_size / len(all_samples)

        n_train_key = int(len(current_samples) * prop_train)
        n_val_key = int(len(current_samples) * prop_val)

        train_samples.extend(current_samples[:n_train_key])
        val_samples.extend(current_samples[n_train_key:n_train_key + n_val_key])
        test_samples.extend(current_samples[n_train_key + n_val_key:])

    # Ensure exact target sizes
    train_samples = rng.sample(train_samples, min(len(train_samples), target_train_size))
    val_samples = rng.sample(val_samples, min(len(val_samples), target_val_size))
    test_samples = rng.sample(test_samples, min(len(test_samples), target_test_size))

    print("\n" + "=" * 60)
    print("Data Split")
    print("=" * 60)
    for name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        ans_dist = Counter(s["target"] for s in samples)
        ctx_min = min(len(s["context"]) for s in samples)
        ctx_max = max(len(s["context"]) for s in samples)
        print(f"  {name:5s}: {len(samples)} samples | "
              f"A={ans_dist.get('A',0)}, B={ans_dist.get('B',0)}, C={ans_dist.get('C',0)} | "
              f"context: {ctx_min}-{ctx_max} chars")

    # -----------------------------------------------------------
    # Save data files
    # -----------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = os.path.join(OUTPUT_DIR, f"team_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved {len(samples)} samples to {path}")

    # Small train set (50 samples)
    train_small = rng.sample(train_samples, min(len(train_samples), 50))
    path = os.path.join(OUTPUT_DIR, "team_train_50.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for s in train_small:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(train_small)} samples to {path} (small train set)")

    # -----------------------------------------------------------
    # Example sample
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example Sample")
    print("=" * 60)
    ex = train_samples[0]
    print(f"Context (first 400 chars):\n{ex['context'][:400]}...\n")
    print(f"Question:\n{ex['question']}")
    print(f"\nTarget: {ex['target']} ({ex['answer_choice']})")
    print(f"Choices: {ex['n_choices']}")

    # -----------------------------------------------------------
    # Save config
    # -----------------------------------------------------------
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    config_data = {
        "musr_team": {
            "train_data": "./eval/musr_team/data/team_train.jsonl",
            "val_data": "./eval/musr_team/data/team_val.jsonl",
            "test_data": "./eval/musr_team/data/team_test.jsonl"
        },
        "musr_team_small": {
            "train_data": "./eval/musr_team/data/team_train_50.jsonl",
            "val_data": "./eval/musr_team/data/team_val.jsonl",
            "test_data": "./eval/musr_team/data/team_test.jsonl"
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
