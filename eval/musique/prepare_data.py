#!/usr/bin/env python3
"""
Prepare MuSiQue data for ACE framework.

MuSiQue (Multihop Questions via Single-hop Question Composition) is a
multi-hop question answering benchmark requiring 2-4 hops of reasoning.

Each sample has ~20 paragraphs as context, a multi-hop question, and
a short text answer. The model must find and combine information across
multiple paragraphs to answer correctly.

Data split (stratified by hop type):
  - Train: 200 samples
  - Val:   100 samples
  - Test:  200 samples (from HF validation split)
  - Train_small: 50 subset for quick experiments

Usage:
    python -m eval.musique.prepare_data
"""
import os
import json
import random
from collections import Counter

SEED = 42
OUTPUT_DIR = "./eval/musique/data"


def get_hop_type(sample_id: str) -> str:
    """Extract hop type from sample id (e.g., '2hop__482757_12019' -> '2hop')."""
    return sample_id.split("__")[0]


def format_context(paragraphs: list) -> str:
    """
    Format paragraphs into a readable context string.

    Each paragraph has a title and text. We format as:
      [Title 1]
      Paragraph text...

      [Title 2]
      Paragraph text...
    """
    parts = []
    for p in paragraphs:
        title = p.get("title", "")
        text = p.get("paragraph_text", "")
        parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts)


def process_sample(example: dict) -> dict:
    """
    Convert a MuSiQue example into ACE format.

    Args:
        example: Raw example from HuggingFace dataset

    Returns:
        Dict with context, question, target, and metadata
    """
    paragraphs = example["paragraphs"]
    question = example["question"]
    answer = example["answer"]
    answer_aliases = example.get("answer_aliases", [])
    sample_id = example["id"]
    decomposition = example.get("question_decomposition", [])

    # Build readable context from paragraphs
    context = format_context(paragraphs)

    # Build question with instruction
    full_question = (
        f"{question}\n\n"
        f"Based on the provided passages, give a short and precise answer. "
        f"Answer with ONLY the answer itself, no explanation."
    )

    hop_type = get_hop_type(sample_id)
    n_hops = len(decomposition) if decomposition else int(hop_type[0])

    return {
        "context": context,
        "question": full_question,
        "target": answer,
        "answer_aliases": answer_aliases,
        "hop_type": hop_type,
        "n_hops": n_hops,
        "sample_id": sample_id,
    }


def save_jsonl(samples: list, path: str):
    """Save samples to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples to {path}")


def stratified_sample(samples: list, n: int, rng: random.Random) -> list:
    """
    Sample n items from samples, stratified by hop_type.
    """
    by_hop = {}
    for s in samples:
        hop = s["hop_type"]
        by_hop.setdefault(hop, []).append(s)

    result = []
    total = len(samples)
    for hop in sorted(by_hop.keys()):
        hop_samples = by_hop[hop]
        rng.shuffle(hop_samples)
        # Proportional allocation
        k = max(1, round(n * len(hop_samples) / total))
        result.extend(hop_samples[:k])

    # Adjust if we got too many or too few
    rng.shuffle(result)
    if len(result) > n:
        result = result[:n]
    elif len(result) < n:
        # Fill from remaining
        used_ids = {s["sample_id"] for s in result}
        remaining = [s for s in samples if s["sample_id"] not in used_ids]
        rng.shuffle(remaining)
        result.extend(remaining[:n - len(result)])

    return result


def main():
    from datasets import load_dataset

    print("=" * 60)
    print("MuSiQue - Data Preparation for ACE")
    print("=" * 60)

    print("\nLoading MuSiQue dataset from HuggingFace...")
    ds = load_dataset("dgslibisey/MuSiQue")
    print(f"Train split: {len(ds['train'])} samples")
    print(f"Validation split: {len(ds['validation'])} samples")

    # Process all samples
    print("\nProcessing samples...")
    train_all = [process_sample(ex) for ex in ds["train"]]
    val_all = [process_sample(ex) for ex in ds["validation"]]

    # Statistics
    hop_dist_train = Counter(s["hop_type"] for s in train_all)
    hop_dist_val = Counter(s["hop_type"] for s in val_all)
    print(f"\nTrain hop distribution: {dict(sorted(hop_dist_train.items()))}")
    print(f"Val hop distribution:   {dict(sorted(hop_dist_val.items()))}")

    ctx_lens = [len(s["context"]) for s in train_all]
    ans_lens = [len(s["target"]) for s in train_all]
    print(f"\nContext length: min={min(ctx_lens)}, max={max(ctx_lens)}, avg={sum(ctx_lens)//len(ctx_lens)}")
    print(f"Answer length: min={min(ans_lens)}, max={max(ans_lens)}, avg={sum(ans_lens)//len(ans_lens)}")

    # Split: 200 train / 100 val from HF train, 200 test from HF validation
    rng = random.Random(SEED)

    # Shuffle HF train and take 300 samples (200 train + 100 val)
    rng.shuffle(train_all)
    train_pool = stratified_sample(train_all, 300, rng)

    # Split the 300 into train/val
    train_samples = train_pool[:200]
    val_samples = train_pool[200:300]

    # Test from HF validation (take 200, stratified)
    test_samples = stratified_sample(val_all, 200, rng)

    print(f"\n{'='*60}")
    print(f"Data Split")
    print(f"{'='*60}")
    for name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        hops = Counter(s["hop_type"] for s in samples)
        ctx = [len(s["context"]) for s in samples]
        print(f"  {name:5s}: {len(samples)} samples | "
              f"hops={dict(sorted(hops.items()))} | "
              f"context: {min(ctx)}-{max(ctx)} chars")

    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving to {OUTPUT_DIR}/...")

    save_jsonl(train_samples, os.path.join(OUTPUT_DIR, "musique_train.jsonl"))
    save_jsonl(val_samples, os.path.join(OUTPUT_DIR, "musique_val.jsonl"))
    save_jsonl(test_samples, os.path.join(OUTPUT_DIR, "musique_test.jsonl"))

    # Small training subset
    train_small = train_samples[:50]
    save_jsonl(train_small, os.path.join(OUTPUT_DIR, "musique_train_50.jsonl"))

    # Print example
    print(f"\n{'='*60}")
    print(f"Example Sample")
    print(f"{'='*60}")
    ex = train_samples[0]
    print(f"Hop type: {ex['hop_type']} ({ex['n_hops']} hops)")
    print(f"Context (first 400 chars):\n{ex['context'][:400]}...")
    print(f"\nQuestion:\n{ex['question']}")
    print(f"\nTarget: {ex['target']}")
    if ex['answer_aliases']:
        print(f"Aliases: {ex['answer_aliases']}")

    # Save sample_config.json
    config = {
        "musique": {
            "train_data": "./eval/musique/data/musique_train.jsonl",
            "val_data": "./eval/musique/data/musique_val.jsonl",
            "test_data": "./eval/musique/data/musique_test.jsonl"
        },
        "musique_small": {
            "train_data": "./eval/musique/data/musique_train_50.jsonl",
            "val_data": "./eval/musique/data/musique_val.jsonl",
            "test_data": "./eval/musique/data/musique_test.jsonl"
        }
    }
    config_path = os.path.join(OUTPUT_DIR, "sample_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved config to {config_path}")

    print(f"\n{'='*60}")
    print("Done! Ready for ACE training.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
