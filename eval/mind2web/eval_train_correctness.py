#!/usr/bin/env python3
"""
Run Qwen2.5-7B-Instruct-Turbo (Together AI, zero-shot, no playbook) on all
Mind2Web training samples to get per-sample correctness labels.

Output: eval/mind2web/data/train_correctness.json
  { "0": 1, "1": 0, "2": 1, ... }   (sample index → 1=correct, 0=wrong)

Supports resuming: already-completed entries are skipped on rerun.

Usage:
    python -m eval.mind2web.eval_train_correctness
    python -m eval.mind2web.eval_train_correctness --workers 30
"""
import os
import json
import time
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import openai

from .data_processor import DataProcessor

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_PATH    = "./eval/mind2web/data/mind2web_train.jsonl"
CACHE_PATH    = "./eval/mind2web/data/train_correctness.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"

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


# ── IO ────────────────────────────────────────────────────────────────────────

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
    idx: int,
    sample: Dict,
    client: openai.OpenAI,
    model: str,
    processor: DataProcessor,
) -> Tuple[int, int]:
    """Evaluate one training sample.  Returns (idx, correct) where correct ∈ {0,1,-1}.
    -1 means API error (will be retried or skipped)."""
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
            timeout=60,
        )
        prediction = resp.choices[0].message.content.strip()
        correct = 1 if processor.answer_is_correct(prediction, sample["target"]) else 0
    except Exception as e:
        print(f"  [idx={idx}] Error: {e}")
        correct = -1
    return idx, correct


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base Qwen on all Mind2Web training samples (0-shot)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Together AI model name")
    parser.add_argument("--workers", type=int, default=30,
                        help="Parallel API workers")
    parser.add_argument("--batch_save", type=int, default=200,
                        help="Save cache every N completions")
    args = parser.parse_args()

    # Setup Together AI client
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set in .env")
    client = openai.OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

    # Load data and processor
    train_data = load_jsonl(TRAIN_PATH)
    processor  = DataProcessor("mind2web")

    # Resume from cache
    cache = load_cache(CACHE_PATH)
    todo  = [(i, s) for i, s in enumerate(train_data) if str(i) not in cache]

    print(f"{'='*60}")
    print(f"  Mind2Web training set — base LLM correctness eval")
    print(f"{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Workers : {args.workers}")
    print(f"  Total   : {len(train_data)}")
    print(f"  Done    : {len(cache)}")
    print(f"  Todo    : {len(todo)}")
    print()

    if not todo:
        total_v = sum(1 for v in cache.values() if v >= 0)
        total_c = sum(v for v in cache.values() if v >= 0)
        print(f"Already complete!  accuracy = {total_c}/{total_v} = {total_c/total_v:.3f}")
        return

    completed = 0
    errors    = 0
    start     = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_sample, i, s, client, args.model, processor): i
            for i, s in todo
        }
        for future in as_completed(futures):
            idx, correct = future.result()
            if correct >= 0:
                cache[str(idx)] = correct
            else:
                errors += 1
            completed += 1

            if completed % args.batch_save == 0:
                save_cache(cache, CACHE_PATH)
                elapsed   = time.time() - start
                rate      = completed / elapsed
                remaining = (len(todo) - completed) / max(rate, 1e-6)
                valid     = {v for v in cache.values() if v >= 0}
                pct_c     = sum(cache[str(k)] for k in cache if cache[str(k)] >= 0) / max(len(valid), 1) * 100
                print(f"  [{completed:>5}/{len(todo)}]  acc={pct_c:.1f}%  "
                      f"errors={errors}  rate={rate:.1f}/s  ETA={remaining/60:.0f}min")

    # Final save
    save_cache(cache, CACHE_PATH)

    total_valid   = sum(1 for v in cache.values() if v >= 0)
    total_correct = sum(v for v in cache.values() if v >= 0)
    print()
    print(f"{'='*60}")
    print(f"  Done!  {total_valid}/{len(train_data)} evaluated  ({errors} errors)")
    print(f"  Base LLM accuracy on train set: "
          f"{total_correct}/{total_valid} = {total_correct/max(total_valid,1):.3f}")
    print(f"  Saved → {CACHE_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
