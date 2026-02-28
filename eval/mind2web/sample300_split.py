#!/usr/bin/env python3
"""
Step 1: Randomly sample 300 training examples.
Step 2: Run DeepSeek-V3 (Together AI) zero-shot inference on all 300.
Step 3: Split into two subsets of 100:
          - mind2web_correct100:   100 samples the base LLM got RIGHT
          - mind2web_incorrect100: 100 samples the base LLM got WRONG
Step 4: Register both subsets in sample_config.json.

Usage:
    python -m eval.mind2web.sample300_split
    python -m eval.mind2web.sample300_split --sample_size 300 --select_n 100 --seed 42 --workers 30
"""
import os
import json
import time
import random
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import openai

from .data_processor import DataProcessor

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR    = "./eval/mind2web/data"
TRAIN_PATH  = f"{DATA_DIR}/mind2web_train.jsonl"
CFG_PATH    = f"{DATA_DIR}/sample_config.json"
CACHE_PATH  = f"{DATA_DIR}/sample300_correctness.json"  # separate from Qwen cache

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"

SYSTEM_PROMPT = (
    "You are an expert web navigation agent. "
    "Given a webpage with candidate HTML elements and a navigation task, "
    "select the correct element and specify the action.\n\n"
    "Output format (one line only):\n"
    "  [element_idx] ACTION [tag] element_text: value\n\n"
    "Where:\n"
    "  - element_idx: integer index of the chosen element\n"
    "  - ACTION: one of CLICK, TYPE, SELECT\n"
    "  - For CLICK: no value needed  (e.g. \"[3] CLICK [button] Submit\")\n"
    "  - For TYPE:  value is the text to type  "
    "(e.g. \"[7] TYPE [input] Search: New York\")\n"
    "  - For SELECT: value is the option  "
    "(e.g. \"[2] SELECT [select] Sort by: Price\")\n\n"
    "Output ONLY the action line, no explanation."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_cache(path: str) -> Dict[str, int]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, int], path: str):
    with open(path, "w") as f:
        json.dump(cache, f)


# ── Inference ─────────────────────────────────────────────────────────────────

def evaluate_sample(
    orig_idx: int,
    sample: Dict,
    client: openai.OpenAI,
    model: str,
    processor: DataProcessor,
) -> Tuple[int, int]:
    """Returns (orig_idx, correct) where correct ∈ {0, 1}.  On error returns -1."""
    user_msg = f"{sample['context']}\n\n{sample['question']}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=64,
            temperature=0.0,
            timeout=90,
        )
        prediction = resp.choices[0].message.content.strip()
        correct = 1 if processor.answer_is_correct(prediction, sample["target"]) else 0
    except Exception as e:
        print(f"  [idx={orig_idx}] Error: {e}")
        correct = -1
    return orig_idx, correct


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample 300 training items, eval with DeepSeek-V3, split correct/incorrect."
    )
    parser.add_argument("--sample_size", type=int, default=300,
                        help="Number of training samples to evaluate (default 300)")
    parser.add_argument("--select_n", type=int, default=100,
                        help="Number of correct / incorrect samples to select (default 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default 42)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Together AI model name")
    parser.add_argument("--workers", type=int, default=20,
                        help="Parallel API workers (default 20)")
    args = parser.parse_args()

    # Setup Together AI client
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set in .env")
    client = openai.OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

    # Load full training data
    all_data = load_jsonl(TRAIN_PATH)
    processor = DataProcessor("mind2web")

    # Sample 300 random training examples (reproducible)
    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(all_data)), args.sample_size))
    print(f"{'='*60}")
    print(f"  Mind2Web — Sample {args.sample_size} → Split correct/incorrect")
    print(f"{'='*60}")
    print(f"  Model       : {args.model}")
    print(f"  Sample seed : {args.seed}")
    print(f"  Sample size : {args.sample_size}")
    print(f"  Select N    : {args.select_n}")
    print()

    # Resume from cache
    cache = load_cache(CACHE_PATH)
    todo = [(i, all_data[i]) for i in sampled_indices if str(i) not in cache]

    print(f"  Already evaluated : {len(cache)}")
    print(f"  Remaining todo    : {len(todo)}")
    print()

    # Run inference on todo items
    if todo:
        completed = 0
        errors    = 0
        start     = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(evaluate_sample, i, s, client, args.model, processor): i
                for i, s in todo
            }
            for future in as_completed(futures):
                orig_idx, correct = future.result()
                if correct >= 0:
                    cache[str(orig_idx)] = correct
                else:
                    errors += 1
                completed += 1

                if completed % 20 == 0 or completed == len(todo):
                    elapsed   = time.time() - start
                    rate      = completed / max(elapsed, 1e-6)
                    remaining = (len(todo) - completed) / max(rate, 1e-6)
                    valid_vals = [v for v in cache.values() if v >= 0]
                    acc = sum(valid_vals) / max(len(valid_vals), 1)
                    print(f"  [{completed:>4}/{len(todo)}]  acc={acc:.1%}  "
                          f"errors={errors}  ETA={remaining/60:.0f}min")
                    save_cache(cache, CACHE_PATH)

        save_cache(cache, CACHE_PATH)

    # ── Compute results for the sampled set ───────────────────────────────────
    results = {i: cache[str(i)] for i in sampled_indices if str(i) in cache and cache[str(i)] >= 0}
    correct_idxs   = [i for i, c in results.items() if c == 1]
    incorrect_idxs = [i for i, c in results.items() if c == 0]

    print()
    print(f"{'='*60}")
    print(f"  DeepSeek-V3 on {len(results)} sampled examples:")
    print(f"    Correct   : {len(correct_idxs)} ({len(correct_idxs)/len(results):.1%})")
    print(f"    Incorrect : {len(incorrect_idxs)} ({len(incorrect_idxs)/len(results):.1%})")
    print(f"{'='*60}")

    if len(correct_idxs) < args.select_n:
        print(f"  WARNING: only {len(correct_idxs)} correct samples, "
              f"selecting all instead of {args.select_n}")
    if len(incorrect_idxs) < args.select_n:
        print(f"  WARNING: only {len(incorrect_idxs)} incorrect samples, "
              f"selecting all instead of {args.select_n}")

    # Take first select_n (already random-ordered from the sampling)
    rng2 = random.Random(args.seed + 1)
    selected_correct   = rng2.sample(correct_idxs,   min(args.select_n, len(correct_idxs)))
    selected_incorrect = rng2.sample(incorrect_idxs, min(args.select_n, len(incorrect_idxs)))

    n_c  = len(selected_correct)
    n_ic = len(selected_incorrect)
    print(f"\n  Selected correct   : {n_c}")
    print(f"  Selected incorrect : {n_ic}")

    # ── Save JSONL files ──────────────────────────────────────────────────────
    def write_jsonl(indices, path):
        with open(path, "w", encoding="utf-8") as f:
            for idx in sorted(indices):
                f.write(json.dumps(all_data[idx], ensure_ascii=False) + "\n")

    correct_path   = f"{DATA_DIR}/mind2web_correct{n_c}.jsonl"
    incorrect_path = f"{DATA_DIR}/mind2web_incorrect{n_ic}.jsonl"
    write_jsonl(selected_correct,   correct_path)
    write_jsonl(selected_incorrect, incorrect_path)
    print(f"\n  Saved: {correct_path}")
    print(f"  Saved: {incorrect_path}")

    # ── Print sample stats ────────────────────────────────────────────────────
    def op_dist(indices):
        from collections import Counter
        ops = Counter()
        for i in indices:
            q = all_data[i]["question"]
            for op in ("CLICK", "TYPE", "SELECT"):
                if op in q:
                    ops[op] += 1
                    break
        return dict(ops)

    print(f"\n  Correct{n_c} op dist   : {op_dist(selected_correct)}")
    print(f"  Incorrect{n_ic} op dist: {op_dist(selected_incorrect)}")

    # ── Register in sample_config.json ────────────────────────────────────────
    cfg = {}
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r") as f:
            cfg = json.load(f)

    correct_name   = f"mind2web_correct{n_c}"
    incorrect_name = f"mind2web_incorrect{n_ic}"

    cfg[correct_name] = {
        "train_data": f"eval/mind2web/data/mind2web_correct{n_c}.jsonl",
        "val_data":   "eval/mind2web/data/mind2web_val.jsonl",
        "test_data":  "eval/mind2web/data/mind2web_test.jsonl",
        "description": f"Top {n_c} samples DeepSeek-V3 answered correctly (seed={args.seed})"
    }
    cfg[incorrect_name] = {
        "train_data": f"eval/mind2web/data/mind2web_incorrect{n_ic}.jsonl",
        "val_data":   "eval/mind2web/data/mind2web_val.jsonl",
        "test_data":  "eval/mind2web/data/mind2web_test.jsonl",
        "description": f"Top {n_ic} samples DeepSeek-V3 answered incorrectly (seed={args.seed})"
    }

    with open(CFG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n  Registered '{correct_name}' and '{incorrect_name}' in sample_config.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
