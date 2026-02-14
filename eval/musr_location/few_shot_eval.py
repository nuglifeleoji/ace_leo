#!/usr/bin/env python3
"""
Few-shot ICL evaluation for MuSR Location (Object Placements).

Tests whether simple multi-shot in-context learning can match ACE performance.
If few-shot ICL already achieves comparable accuracy, ACE may be overkill
for this task.

Experiment:
  - k = 0, 1, 3, 5, 10, 20 shot examples from train set
  - Each k-shot runs 3 seeds for variance estimation
  - Same model (DeepSeek-V3.1 via SambaNova) as ACE baseline
  - Evaluate on the same 100-sample test set

Usage:
    python -m eval.musr_location.few_shot_eval [--shots 0 1 3 5 10 20] [--seeds 3]
"""
import os
import json
import time
import random
import argparse
import openai
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# ── Data Loading ────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── Prompt Construction ─────────────────────────────────────────

def build_few_shot_prompt(
    test_sample: Dict,
    demo_samples: List[Dict],
    k: int,
) -> str:
    """
    Build a few-shot prompt.

    Format:
      [System instruction]
      --- Example 1 ---
      Story: ...
      Question: ...
      Answer: A
      --- Example 2 ---
      ...
      --- Your Turn ---
      Story: ...
      Question: ...
      Answer:
    """
    parts = []

    # System instruction
    parts.append(
        "You are an expert at tracking object locations in stories. "
        "Read the story carefully, track where each object is moved, "
        "and select the correct answer.\n"
    )

    # Few-shot demonstrations
    for i, demo in enumerate(demo_samples[:k]):
        parts.append(f"--- Example {i+1} ---")
        parts.append(f"Story:\n{demo['context']}\n")
        parts.append(f"Question:\n{demo['question']}\n")
        parts.append(f"Answer: {demo['target']}\n")

    # Test question
    if k > 0:
        parts.append("--- Your Turn ---")
    parts.append(f"Story:\n{test_sample['context']}\n")
    parts.append(f"Question:\n{test_sample['question']}\n")
    parts.append("Answer:")

    return "\n".join(parts)


# ── Answer Extraction ───────────────────────────────────────────

import re

def extract_letter(text: str) -> str:
    """Extract answer letter from model output (same logic as data_processor)."""
    if not text:
        return ""
    text = text.strip()

    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()

    m = re.match(r'^([A-Ea-e])\s*[\)\.:\s]', text)
    if m:
        return m.group(1).upper()

    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\**([A-Ea-e])\**', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r'\*\*([A-Ea-e])\*\*', text)
    if m:
        return m.group(1).                                                                                                                                                                                                                                                  upper()

    m = re.search(r'\b([A-E])\b', text)
    if m:
        return m.group(1)

    return ""


# ── Single Sample Evaluation ────────────────────────────────────

def evaluate_one(
    client: openai.OpenAI,
    model: str,
    test_sample: Dict,
    demo_samples: List[Dict],
    k: int,
    idx: int,
    max_tokens: int = 64,
    max_retries: int = 5,
) -> Dict:
    """Evaluate a single test sample with k-shot prompt."""
    prompt = build_few_shot_prompt(test_sample, demo_samples, k)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            answer_text = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                answer_text = ""
                print(f"  [WARN] Sample {idx} failed after {max_retries} retries: {e}")

    pred_letter = extract_letter(answer_text)
    gt_letter = test_sample["target"].strip().upper()

    return {
        "index": idx,
        "predicted": pred_letter,
        "ground_truth": gt_letter,
        "correct": pred_letter == gt_letter,
        "raw_answer": answer_text[:200],
    }


# ── Main Experiment ─────────────────────────────────────────────

def run_experiment(
    k: int,
    seed: int,
    train_data: List[Dict],
    test_data: List[Dict],
    client: openai.OpenAI,
    model: str,
    max_workers: int = 10,
) -> Dict:
    """Run one k-shot experiment with given seed."""
    rng = random.Random(seed)

    # Sample k demonstrations (fixed for all test samples in this run)
    demos = rng.sample(train_data, min(k, len(train_data))) if k > 0 else []

    print(f"\n{'='*50}")
    print(f"  {k}-shot | seed={seed} | {len(test_data)} test samples")
    print(f"{'='*50}")

    results = []
    correct_so_far = 0
    done_so_far = 0

    def _eval(idx):
        return evaluate_one(client, model, test_data[idx], demos, k, idx)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval, i): i for i in range(len(test_data))}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done_so_far += 1
            correct_so_far += int(r["correct"])
            if done_so_far % 10 == 0 or done_so_far == len(test_data):
                acc = correct_so_far / done_so_far
                print(f"  [{done_so_far}/{len(test_data)}] running acc: {acc:.2%}")

    results.sort(key=lambda r: r["index"])

    correct = sum(r["correct"] for r in results)
    total = len(results)
    accuracy = correct / total if total else 0.0

    print(f"  Result: {correct}/{total} = {accuracy:.4f}")

    return {
        "k": k,
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Few-shot ICL eval for MuSR Location")
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3, 5, 10, 20],
                        help="Number of shots to test (default: 0 1 3 5 10 20)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per k (default: 3)")
    parser.add_argument("--model", type=str, default="DeepSeek-V3.1",
                        help="Model name (default: DeepSeek-V3.1)")
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai"])
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Parallel workers (default: 10)")
    parser.add_argument("--save_path", type=str,
                        default="results/musr_location_few_shot",
                        help="Directory to save results")
    args = parser.parse_args()

    # ── Setup API client ──
    provider_config = {
        "sambanova": ("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
        "together":  ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        "openai":    ("https://api.openai.com/v1",   "OPENAI_API_KEY"),
    }
    base_url, key_env = provider_config[args.api_provider]
    api_key = os.getenv(key_env, "")
    if not api_key:
        raise ValueError(f"{key_env} not set in .env")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # ── Load data ──
    train_data = load_jsonl("./eval/musr_location/data/location_train.jsonl")
    test_data  = load_jsonl("./eval/musr_location/data/location_test.jsonl")
    print(f"Train: {len(train_data)} samples")
    print(f"Test:  {len(test_data)} samples")

    # ── Run experiments ──
    all_results = []
    summary = {}

    for k in args.shots:
        k_results = []
        for seed_idx in range(args.seeds):
            seed = 42 + seed_idx
            result = run_experiment(
                k=k, seed=seed,
                train_data=train_data, test_data=test_data,
                client=client, model=args.model,
                max_workers=args.max_workers,
            )
            k_results.append(result)
            all_results.append(result)

        accuracies = [r["accuracy"] for r in k_results]
        mean_acc = sum(accuracies) / len(accuracies)
        min_acc = min(accuracies)
        max_acc = max(accuracies)

        summary[f"{k}-shot"] = {
            "mean_accuracy": round(mean_acc, 4),
            "min_accuracy": round(min_acc, 4),
            "max_accuracy": round(max_acc, 4),
            "per_seed": [round(a, 4) for a in accuracies],
        }
        print(f"\n>>> {k}-shot mean: {mean_acc:.4f} (range: {min_acc:.4f} - {max_acc:.4f})")

    # ── Print summary table ──
    print(f"\n{'='*60}")
    print(f"  FEW-SHOT ICL RESULTS — MuSR Location (Object Placements)")
    print(f"{'='*60}")
    print(f"  Model: {args.model} | Provider: {args.api_provider}")
    print(f"  Test set: {len(test_data)} samples | Seeds: {args.seeds}")
    print(f"{'='*60}")
    print(f"  {'Shots':<8} {'Mean':>8} {'Min':>8} {'Max':>8}  Per-seed")
    print(f"  {'-'*52}")

    for k in args.shots:
        s = summary[f"{k}-shot"]
        seeds_str = ", ".join(f"{a:.2%}" for a in s["per_seed"])
        print(f"  {k:<8} {s['mean_accuracy']:>7.2%} {s['min_accuracy']:>7.2%} "
              f"{s['max_accuracy']:>7.2%}  [{seeds_str}]")

    # Reference: ACE results
    print(f"\n  --- ACE Reference ---")
    print(f"  {'Baseline (0-shot, no playbook)':<35} {'44.00%':>8}")
    print(f"  {'ACE trained (best playbook)':<35} {'63.00%':>8}")
    print(f"{'='*60}")

    # ── Save results ──
    os.makedirs(args.save_path, exist_ok=True)
    output = {
        "experiment": "few_shot_icl_musr_location",
        "model": args.model,
        "api_provider": args.api_provider,
        "shots_tested": args.shots,
        "num_seeds": args.seeds,
        "test_size": len(test_data),
        "train_size": len(train_data),
        "summary": summary,
        "ace_reference": {
            "baseline_0shot": 0.44,
            "ace_trained": 0.63,
        },
    }

    summary_path = os.path.join(args.save_path, "few_shot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Save detailed results (without raw answers to keep file small)
    detail_path = os.path.join(args.save_path, "few_shot_detailed.json")
    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Detailed results saved to {detail_path}")


if __name__ == "__main__":
    main()
