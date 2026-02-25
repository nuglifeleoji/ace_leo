#!/usr/bin/env python3
"""
Few-shot ICL evaluation for Mind2Web web navigation task.

Tests whether simple multi-shot in-context learning can match ACE performance.

Design:
  - Demos include question + answer only (no candidate context), to avoid
    exploding prompt length (~200 elements per context).
  - Test item uses full context + question.
  - k = 0, 5, 10 shots, 3 seeds each for variance estimation.
  - Same model (DeepSeek-V3.1 via SambaNova) and test set (1805 steps) as ACE.

Usage:
    python -m eval.mind2web.few_shot_eval
    python -m eval.mind2web.few_shot_eval --shots 0 5 10 --seeds 3
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

from .data_processor import DataProcessor

load_dotenv()

SYSTEM_PROMPT = """You are an expert web navigation agent. Given a webpage with candidate HTML elements and a navigation task, select the correct element and specify the action.

Output format (one line only):
  [element_idx] ACTION [tag] element_text: value

Where:
  - element_idx: integer index of the chosen element
  - ACTION: one of CLICK, TYPE, SELECT
  - For CLICK: no value needed (e.g., "[3] CLICK [button] Submit")
  - For TYPE: value is the text to type (e.g., "[7] TYPE [input] Search: New York")
  - For SELECT: value is the option to select (e.g., "[2] SELECT [select] Sort by: Price")

Output ONLY the action line, no explanation."""


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_prompt(
    test_sample: Dict,
    demo_samples: List[Dict],
    k: int,
) -> str:
    """
    Build a few-shot prompt for Mind2Web.

    Demos show only question + answer (no candidate context, to keep prompts short).
    Test item has full context + question.
    """
    parts = []

    # Few-shot demonstrations
    if k > 0 and demo_samples:
        parts.append("Here are some examples of web navigation actions:\n")
        for i, demo in enumerate(demo_samples[:k]):
            parts.append(f"--- Example {i+1} ---")
            # Include question (task + history) but NOT the full candidate context
            parts.append(f"Task & History:\n{demo['question']}\n")
            parts.append(f"Action: {demo['target']}\n")
        parts.append("--- Your Turn ---\n")

    # Full test item (context + question)
    parts.append(f"Candidate elements:\n{test_sample['context']}\n")
    parts.append(f"Task & History:\n{test_sample['question']}\n")
    parts.append("Action:")

    return "\n".join(parts)


def call_llm(client, model: str, prompt: str, max_tokens: int = 128) -> str:
    """Call LLM and return response text."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  [ERROR] LLM call failed: {e}")
                return ""
    return ""


def evaluate_one(
    client, model: str,
    test_sample: Dict,
    demos: List[Dict],
    k: int,
    idx: int,
    processor: DataProcessor,
) -> Dict:
    prompt = build_prompt(test_sample, demos, k)
    prediction = call_llm(client, model, prompt)
    correct = processor.answer_is_correct(prediction, test_sample["target"])
    return {
        "index": idx,
        "correct": correct,
        "prediction": prediction,
        "target": test_sample["target"],
    }


def run_experiment(
    k: int,
    seed: int,
    train_data: List[Dict],
    test_data: List[Dict],
    client,
    model: str,
    processor: DataProcessor,
    max_workers: int = 20,
) -> Dict:
    print(f"\n  Running {k}-shot (seed={seed})...")
    rng = random.Random(seed)
    demos = rng.sample(train_data, min(k, len(train_data))) if k > 0 else []

    results = []
    done_so_far = 0
    correct_so_far = 0

    def _eval(idx):
        return evaluate_one(client, model, test_data[idx], demos, k, idx, processor)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval, i): i for i in range(len(test_data))}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done_so_far += 1
            correct_so_far += int(r["correct"])
            if done_so_far % 50 == 0 or done_so_far == len(test_data):
                acc = correct_so_far / done_so_far
                print(f"  Progress: {done_so_far}/{len(test_data)}, Accuracy: {acc:.3f}")

    results.sort(key=lambda r: r["index"])
    correct = sum(r["correct"] for r in results)
    total = len(results)
    accuracy = correct / total if total else 0.0
    print(f"  📊 Final Accuracy: {accuracy:.3f} ({correct}/{total})")

    return {
        "k": k,
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Few-shot ICL eval for Mind2Web")
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 5, 10],
                        help="Number of shots to test (default: 0 5 10)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per k (default: 3)")
    parser.add_argument("--model", type=str, default="DeepSeek-V3.1",
                        help="Model name (default: DeepSeek-V3.1)")
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai"])
    parser.add_argument("--max_workers", type=int, default=20,
                        help="Parallel workers (default: 20)")
    parser.add_argument("--save_path", type=str,
                        default="results/mind2web_few_shot",
                        help="Directory to save results")
    parser.add_argument("--train_path", type=str,
                        default="./eval/mind2web/data/mind2web_train.jsonl")
    parser.add_argument("--test_path", type=str,
                        default="./eval/mind2web/data/mind2web_test.jsonl")
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
    processor = DataProcessor("mind2web")
    raw_train = load_jsonl(args.train_path)
    raw_test  = load_jsonl(args.test_path)
    train_data = processor.process_task_data(raw_train)
    test_data  = processor.process_task_data(raw_test)
    print(f"Train: {len(train_data)} samples")
    print(f"Test:  {len(test_data)} samples")

    # ── Run experiments ──
    all_results = []
    summary = {}

    for k in args.shots:
        k_results = []
        seeds_to_run = [42] if k == 0 else [42 + i for i in range(args.seeds)]
        for seed in seeds_to_run:
            result = run_experiment(
                k=k, seed=seed,
                train_data=train_data, test_data=test_data,
                client=client, model=args.model,
                processor=processor,
                max_workers=args.max_workers,
            )
            k_results.append(result)
            all_results.append(result)

        accuracies = [r["accuracy"] for r in k_results]
        mean_acc = sum(accuracies) / len(accuracies)
        min_acc  = min(accuracies)
        max_acc  = max(accuracies)

        summary[f"{k}-shot"] = {
            "mean_accuracy": round(mean_acc, 4),
            "min_accuracy":  round(min_acc, 4),
            "max_accuracy":  round(max_acc, 4),
            "per_seed":      [round(a, 4) for a in accuracies],
        }
        print(f"\n>>> {k}-shot mean: {mean_acc:.4f} "
              f"(range: {min_acc:.4f} - {max_acc:.4f})")

    # ── Print summary table ──
    print(f"\n{'='*62}")
    print(f"  FEW-SHOT ICL RESULTS — Mind2Web")
    print(f"{'='*62}")
    print(f"  Model: {args.model} | Provider: {args.api_provider}")
    print(f"  Test set: {len(test_data)} steps | Seeds: {args.seeds}")
    print(f"{'='*62}")
    print(f"  {'Shots':<8} {'Mean':>8} {'Min':>8} {'Max':>8}  Per-seed")
    print(f"  {'-'*54}")
    for k in args.shots:
        s = summary[f"{k}-shot"]
        seeds_str = ", ".join(f"{a:.3f}" for a in s["per_seed"])
        print(f"  {k:<8} {s['mean_accuracy']:>7.3f} {s['min_accuracy']:>7.3f} "
              f"{s['max_accuracy']:>7.3f}  [{seeds_str}]")

    # Reference: best ACE results so far
    print(f"\n  --- ACE Reference ---")
    print(f"  {'cluster15_lesson (best)':<35} {'0.373':>8}")
    print(f"  {'random10_seed0 (best random)':<35} {'0.345':>8}")
    print(f"{'='*62}")

    # ── Save results ──
    os.makedirs(args.save_path, exist_ok=True)
    output = {
        "experiment": "few_shot_icl_mind2web",
        "model": args.model,
        "api_provider": args.api_provider,
        "shots_tested": args.shots,
        "num_seeds": args.seeds,
        "test_size": len(test_data),
        "train_size": len(train_data),
        "summary": summary,
        "ace_reference": {
            "cluster15_lesson": 0.373,
            "best_random10": 0.345,
        },
    }
    summary_path = os.path.join(args.save_path, "few_shot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
