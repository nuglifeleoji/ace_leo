#!/usr/bin/env python3
"""
Prepare CL-bench Technical Standards data for ACE.

Extracts the 201 "Technical Standards" samples from the CL-bench Rule System
Application data, converts them to ACE format, and splits into train/val/test.

Key design decisions:
- **Context-ID-grouped splitting**: Samples sharing the same context_id (same source
  document, e.g., a dishwasher manual or an API spec) are kept in the same split.
  This prevents data leakage where the model might memorize document-specific patterns
  during training and exploit them at test time.
- Rubric-based targets: each sample's "target" is a formatted list of rubrics that
  define what a correct answer should contain.

Data profile (201 samples, 40 unique context_ids):
  - Messages per sample: 2–8 (median 2)
  - Rubrics per sample: 3–75 (mean 17.2)
  - Context chars: 744–240K (median 4.6K)
"""
import os
import json
import random
from typing import List, Dict, Any
from collections import defaultdict


def load_raw_data(data_path: str) -> List[Dict[str, Any]]:
    """Load raw CL-bench data from a JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found: {data_path}")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_messages_to_context(messages: List[Dict]) -> str:
    """Converts a list of OpenAI chat messages into a single string context."""
    formatted_context = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        formatted_context.append(f"[{role}]\n{content}\n")
    return "\n".join(formatted_context).strip()


def format_rubrics_to_target(rubrics: List[str]) -> str:
    """Converts a list of rubrics into a single string target."""
    if not rubrics:
        return "No evaluation criteria provided."
    formatted_target = f"Evaluation criteria ({len(rubrics)} rubrics):\n"
    for rubric in rubrics:
        formatted_target += f"- {rubric}\n"
    return formatted_target.strip()


def group_split_by_context_id(
    samples: List[Dict],
    train_ratio: float = 0.50,
    val_ratio: float = 0.13,
    seed: int = 42
) -> tuple:
    """
    Split samples by context_id groups so that all questions from the same
    source document stay in the same split.

    Args:
        samples: List of processed samples with metadata.context_id
        train_ratio: Fraction of samples for training (approximate)
        val_ratio: Fraction of samples for validation (approximate)
        seed: Random seed for reproducibility

    Returns:
        (train_samples, val_samples, test_samples)
    """
    # Group samples by context_id
    groups = defaultdict(list)
    for s in samples:
        cid = s['metadata']['context_id']
        groups[cid].append(s)

    # Sort context_ids by group size (descending) for more balanced allocation
    sorted_cids = sorted(groups.keys(), key=lambda c: -len(groups[c]))

    # Shuffle with seed for reproducibility (after sorting for determinism)
    rng = random.Random(seed)
    rng.shuffle(sorted_cids)

    total = len(samples)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    train_samples, val_samples, test_samples = [], [], []
    train_count, val_count = 0, 0

    for cid in sorted_cids:
        group = groups[cid]
        if train_count < train_target:
            train_samples.extend(group)
            train_count += len(group)
        elif val_count < val_target:
            val_samples.extend(group)
            val_count += len(group)
        else:
            test_samples.extend(group)

    return train_samples, val_samples, test_samples


def main():
    output_dir = "./eval/CLbench_technical/data"
    os.makedirs(output_dir, exist_ok=True)

    # Load the raw Technical Standards data (already extracted by CLbench download)
    raw_data_path = "./eval/CLbench/data/rule_system_technical_standards.jsonl"
    raw_samples = load_raw_data(raw_data_path)
    print(f"Loaded {len(raw_samples)} raw Technical Standards samples\n")

    # Convert to ACE format
    processed_samples = []
    context_lengths = []
    question_lengths = []
    rubrics_per_sample = []

    for sample in raw_samples:
        messages = sample['messages']
        rubrics = sample['rubrics']
        metadata = sample['metadata']

        # Last user message = question; rest = context
        question = messages[-1]['content']
        context_messages = messages[:-1]
        context = format_messages_to_context(context_messages)
        target = format_rubrics_to_target(rubrics)

        processed_sample = {
            "context": context,
            "question": question,
            "target": target,
            "metadata": {
                "task_id": metadata['task_id'],
                "context_id": metadata['context_id'],
                "context_category": metadata['context_category'],
                "sub_category": metadata['sub_category'],
                "num_rubrics": len(rubrics),
                "rubrics": rubrics
            }
        }
        processed_samples.append(processed_sample)
        context_lengths.append(len(context))
        question_lengths.append(len(question))
        rubrics_per_sample.append(len(rubrics))

    print("Data profile:")
    print(f"  Context lengths: min={min(context_lengths):,}, max={max(context_lengths):,}, "
          f"median={sorted(context_lengths)[len(context_lengths)//2]:,}")
    print(f"  Question lengths: min={min(question_lengths):,}, max={max(question_lengths):,}, "
          f"median={sorted(question_lengths)[len(question_lengths)//2]:,}")
    print(f"  Rubrics per sample: min={min(rubrics_per_sample)}, max={max(rubrics_per_sample)}, "
          f"mean={sum(rubrics_per_sample)/len(rubrics_per_sample):.1f}\n")

    # Split by context_id groups (~50% train, ~13% val, ~37% test)
    train_samples, val_samples, test_samples = group_split_by_context_id(
        processed_samples, train_ratio=0.50, val_ratio=0.13, seed=42
    )

    print(f"Context-ID-grouped split: {len(train_samples)} train / "
          f"{len(val_samples)} val / {len(test_samples)} test")

    # Report context_id distribution
    for split_name, split_data in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        cids = set(s['metadata']['context_id'] for s in split_data)
        avg_rubrics = sum(s['metadata']['num_rubrics'] for s in split_data) / max(len(split_data), 1)
        print(f"  {split_name}: {len(split_data)} samples, {len(cids)} unique contexts, "
              f"avg rubrics: {avg_rubrics:.1f}")

    # Verify no context_id leakage between splits
    train_cids = set(s['metadata']['context_id'] for s in train_samples)
    val_cids = set(s['metadata']['context_id'] for s in val_samples)
    test_cids = set(s['metadata']['context_id'] for s in test_samples)
    assert train_cids.isdisjoint(val_cids), "Train/val context_id overlap!"
    assert train_cids.isdisjoint(test_cids), "Train/test context_id overlap!"
    assert val_cids.isdisjoint(test_cids), "Val/test context_id overlap!"
    print("  ✅ No context_id leakage between splits\n")

    # Save splits
    for name, data in [("technical_train", train_samples),
                       ("technical_val", val_samples),
                       ("technical_test", test_samples)]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, 'w', encoding='utf-8') as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} samples to {path}")

    # Create 50-sample training set (subset of train, also grouped by context_id)
    if len(train_samples) > 50:
        # Pick context_ids until we have ~50 samples
        train_groups = defaultdict(list)
        for s in train_samples:
            train_groups[s['metadata']['context_id']].append(s)

        rng = random.Random(42)
        train_cid_list = list(train_groups.keys())
        rng.shuffle(train_cid_list)

        train_50 = []
        for cid in train_cid_list:
            if len(train_50) >= 50:
                break
            train_50.extend(train_groups[cid])
        # Trim to exactly 50 if slightly over
        train_50 = train_50[:50]
    else:
        train_50 = train_samples

    path_50 = os.path.join(output_dir, "technical_train_50.jsonl")
    with open(path_50, 'w', encoding='utf-8') as f:
        for sample in train_50:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(train_50)} samples to {path_50}")

    # Create sample_config.json
    config = {
        "technical": {
            "train_data": "./eval/CLbench_technical/data/technical_train.jsonl",
            "val_data": "./eval/CLbench_technical/data/technical_val.jsonl",
            "test_data": "./eval/CLbench_technical/data/technical_test.jsonl"
        },
        "technical_small": {
            "train_data": "./eval/CLbench_technical/data/technical_train_50.jsonl",
            "val_data": "./eval/CLbench_technical/data/technical_val.jsonl",
            "test_data": "./eval/CLbench_technical/data/technical_test.jsonl"
        }
    }
    config_path = os.path.join(output_dir, "sample_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"\nSaved config to {config_path}")

    # Show example
    print("\n" + "=" * 60)
    print("Example ACE sample:")
    print("=" * 60)
    ex = processed_samples[0]
    print(f"  context_id: {ex['metadata']['context_id']}")
    print(f"  context length: {len(ex['context']):,} chars")
    print(f"  question length: {len(ex['question']):,} chars")
    print(f"  num_rubrics: {ex['metadata']['num_rubrics']}")
    print(f"  question preview: {ex['question'][:200]}...")
    print(f"  target preview: {ex['target'][:300]}...")
    print("\nDone!")


if __name__ == "__main__":
    main()
