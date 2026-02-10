#!/usr/bin/env python3
"""
Prepare CL-bench Procedural Task Execution data for ACE.

Converts the raw CL-bench format (messages + rubrics) into ACE-compatible JSONL:
  - context: system prompt + prior conversation turns + procedure document
  - question: the last user message
  - target: rubrics joined as evaluation criteria
  - metadata: task_id, context_id, sub_category, rubrics list

Splits: 200 train / 71 val / 200 test (stratified by sub_category)
Also creates a 50-sample train subset for quick experiments.
"""
import json
import os
import random
from collections import Counter, defaultdict

random.seed(42)

INPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'procedural_all.jsonl')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')

TRAIN_SIZE = 200
VAL_SIZE = 71
TEST_SIZE = 200
SMALL_TRAIN_SIZE = 50


def load_raw_data(path):
    """Load raw CL-bench JSONL data."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} raw samples from {path}")
    return samples


def convert_to_ace_format(sample):
    """
    Convert a CL-bench sample to ACE format.

    For single-turn (system + user):
      context = system prompt
      question = user message (contains both procedure doc and question)

    For multi-turn (system + user + assistant + user + ...):
      context = system prompt + all prior turns (user/assistant pairs)
      question = last user message
    """
    messages = sample['messages']
    rubrics = sample['rubrics']
    metadata = sample['metadata']

    if len(messages) == 2:
        # Single-turn: system + user
        context = messages[0]['content']
        question = messages[1]['content']
    else:
        # Multi-turn: build context from all messages except the last user message
        context_parts = []
        context_parts.append(f"[System Instructions]\n{messages[0]['content']}")

        for i in range(1, len(messages) - 1):
            role = messages[i]['role'].capitalize()
            context_parts.append(f"\n[{role}]\n{messages[i]['content']}")

        context = "\n".join(context_parts)
        question = messages[-1]['content']

    # Target = rubrics formatted as evaluation criteria
    rubrics_text = "\n".join(f"- {r}" for r in rubrics)
    target = f"Evaluation criteria ({len(rubrics)} rubrics):\n{rubrics_text}"

    return {
        "context": context,
        "question": question,
        "target": target,
        "metadata": {
            "task_id": metadata['task_id'],
            "context_id": metadata['context_id'],
            "sub_category": metadata['sub_category'],
            "rubrics": rubrics,
            "num_rubrics": len(rubrics),
            "num_messages": len(messages)
        }
    }


def stratified_split(samples, train_size, val_size, test_size):
    """Split samples stratified by sub_category."""
    by_category = defaultdict(list)
    for s in samples:
        by_category[s['metadata']['sub_category']].append(s)

    total = train_size + val_size + test_size
    train, val, test = [], [], []

    for cat, cat_samples in by_category.items():
        random.shuffle(cat_samples)
        n = len(cat_samples)

        # Proportional split
        n_train = round(n * train_size / total)
        n_val = round(n * val_size / total)
        n_test = n - n_train - n_val

        # Ensure at least 1 in each split
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            while n_train + n_val + n_test > n:
                if n_train > n_val and n_train > n_test:
                    n_train -= 1
                elif n_val > n_test:
                    n_val -= 1
                else:
                    n_test -= 1

        train.extend(cat_samples[:n_train])
        val.extend(cat_samples[n_train:n_train + n_val])
        test.extend(cat_samples[n_train + n_val:n_train + n_val + n_test])

        print(f"  {cat}: {n_train} train / {n_val} val / {n_test} test (total {n})")

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_jsonl(data, path):
    """Save data as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {path}")


def main():
    # Load raw data
    raw_samples = load_raw_data(INPUT_PATH)

    # Convert to ACE format
    print("\nConverting to ACE format...")
    ace_samples = [convert_to_ace_format(s) for s in raw_samples]

    # Show stats
    context_lens = [len(s['context']) for s in ace_samples]
    question_lens = [len(s['question']) for s in ace_samples]
    rubric_counts = [s['metadata']['num_rubrics'] for s in ace_samples]
    print(f"  Context lengths: min={min(context_lens):,}, max={max(context_lens):,}, "
          f"median={sorted(context_lens)[len(context_lens)//2]:,}")
    print(f"  Question lengths: min={min(question_lens):,}, max={max(question_lens):,}, "
          f"median={sorted(question_lens)[len(question_lens)//2]:,}")
    print(f"  Rubrics per sample: min={min(rubric_counts)}, max={max(rubric_counts)}, "
          f"mean={sum(rubric_counts)/len(rubric_counts):.1f}")

    # Split
    print(f"\nSplitting {len(ace_samples)} samples: {TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE}...")
    train, val, test = stratified_split(ace_samples, TRAIN_SIZE, VAL_SIZE, TEST_SIZE)
    print(f"\nActual split: {len(train)} train / {len(val)} val / {len(test)} test")

    # Save full splits
    save_jsonl(train, os.path.join(OUTPUT_DIR, 'procedural_train.jsonl'))
    save_jsonl(val, os.path.join(OUTPUT_DIR, 'procedural_val.jsonl'))
    save_jsonl(test, os.path.join(OUTPUT_DIR, 'procedural_test.jsonl'))

    # Save small train subset (50 samples)
    small_train = train[:SMALL_TRAIN_SIZE]
    save_jsonl(small_train, os.path.join(OUTPUT_DIR, 'procedural_train_50.jsonl'))

    # Create sample_config.json
    config = {
        "procedural": {
            "train_data": "./eval/CLbench_procedural/data/procedural_train.jsonl",
            "val_data": "./eval/CLbench_procedural/data/procedural_val.jsonl",
            "test_data": "./eval/CLbench_procedural/data/procedural_test.jsonl"
        },
        "procedural_small": {
            "train_data": "./eval/CLbench_procedural/data/procedural_train_50.jsonl",
            "val_data": "./eval/CLbench_procedural/data/procedural_val.jsonl",
            "test_data": "./eval/CLbench_procedural/data/procedural_test.jsonl"
        }
    }
    config_path = os.path.join(OUTPUT_DIR, 'sample_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved config to {config_path}")

    # Show example
    print(f"\n{'='*60}")
    print("Example ACE sample:")
    print(f"{'='*60}")
    ex = ace_samples[0]
    print(f"  sub_category: {ex['metadata']['sub_category']}")
    print(f"  context length: {len(ex['context']):,} chars")
    print(f"  question length: {len(ex['question']):,} chars")
    print(f"  num_rubrics: {ex['metadata']['num_rubrics']}")
    print(f"  question preview: {ex['question'][:200]}...")
    print(f"  target preview: {ex['target'][:300]}...")

    print("\nDone!")


if __name__ == '__main__':
    main()
