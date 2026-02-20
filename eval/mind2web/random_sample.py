#!/usr/bin/env python3
"""
Generate 5 random samplings of 10 training examples from Mind2Web training data.

Usage:
    python -m eval.mind2web.random_sample
"""
import os
import json
import random

TRAIN_PATH = "./eval/mind2web/data/mind2web_train.jsonl"
OUTPUT_DIR = "./eval/mind2web/data"
CONFIG_PATH = "./eval/mind2web/data/sample_config.json"

SEEDS = [0, 1, 2, 3, 4]
N_SAMPLES = 10


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} samples → {path}")


def main():
    train_data = load_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_data)} training samples from {TRAIN_PATH}")

    # Load existing config
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    for seed in SEEDS:
        rng = random.Random(seed)
        selected = rng.sample(range(len(train_data)), N_SAMPLES)
        subset = [train_data[i] for i in selected]

        name = f"mind2web_random{N_SAMPLES}_seed{seed}"
        out_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        save_jsonl(subset, out_path)

        config[name] = {
            "train_data": f"./eval/mind2web/data/{name}.jsonl",
            "val_data": "./eval/mind2web/data/mind2web_val.jsonl",
            "test_data": "./eval/mind2web/data/mind2web_test.jsonl",
        }
        print(f"  Indices (seed={seed}): {selected}")

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH} with {len(SEEDS)} random sampling configs.")

    # Print run commands
    print("\n" + "="*60)
    print("  Next: run ACE for each random sampling:")
    print("="*60)
    for seed in SEEDS:
        name = f"mind2web_random{N_SAMPLES}_seed{seed}"
        print(f"\n  python -m eval.mind2web.run \\")
        print(f"    --task_name {name} \\")
        print(f"    --mode offline --skip_initial_test \\")
        print(f"    --eval_steps {N_SAMPLES} \\")
        print(f"    --save_path results/{name}")


if __name__ == "__main__":
    main()
