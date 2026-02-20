#!/usr/bin/env python3
"""
LLM-based training subset selection for Mind2Web using Llama 3.1 8B (Together AI).

Unlike embedding-based methods (K-means, DPP, Herding) that rely on vector distances,
this method uses a small LLM's *semantic understanding* to reason about task difficulty,
real-world importance, action-type coverage, and learning value.

Method — Two-stage tournament:
    Stage 1  (Batch coarse filter):
        All 4477 training samples are summarized into compact 1-line descriptions
        and fed to the LLM in batches of BATCH_SIZE (default 100).  For each batch
        the LLM returns the top TOP_PER_BATCH (default 5) global indices, yielding
        roughly N/BATCH_SIZE × TOP_PER_BATCH finalist candidates (~225 for defaults).

    Stage 2  (Final selection):
        The finalists are shown to the LLM in one call.  The LLM reasons about
        coverage (action types, domains, task complexity) and returns the final
        TOP_K (default 10) indices with justification.

Prerequisite:
    TOGETHER_API_KEY must be set in your .env file or environment.

Usage:
    # Default: Llama-3.1-8B, pick top 10
    python -m eval.mind2web.llm_select

    # Custom settings
    python -m eval.mind2web.llm_select --top_k 10 --batch_size 80 --top_per_batch 4

    # Use a different Together model
    python -m eval.mind2web.llm_select --model meta-llama/Llama-3.2-3B-Instruct-Turbo
"""
import os
import re
import json
import time
import argparse
from typing import List, Dict, Tuple
from collections import Counter

import openai
from dotenv import load_dotenv

# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_PATH   = "./eval/mind2web/data/mind2web_train.jsonl"
OUTPUT_DIR   = "./eval/mind2web/data"
CONFIG_PATH  = "./eval/mind2web/data/sample_config.json"

DEFAULT_MODEL         = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEFAULT_TOP_K         = 10
DEFAULT_BATCH_SIZE    = 100
DEFAULT_TOP_PER_BATCH = 5

RETRY_ATTEMPTS = 3
RETRY_DELAY    = 4   # seconds between retries


# ── IO ───────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} samples → {path}")


# ── Compact Summary ───────────────────────────────────────────────────────────

def make_compact_summary(item: Dict, global_idx: int) -> str:
    """
    Create a compact one-line summary for a training sample.

    Format:
        [IDX] Domain/Website | OP:Value | Step S/T | "Task description (truncated)"

    Example:
        [142] Travel/kayak | SELECT=Economy | Step 2/5 | "Book cheapest flight NYC to LAX"
    """
    domain    = item.get("domain", "?")
    website   = item.get("website", "?")
    op_dict   = item.get("operation", {})
    op_type   = op_dict.get("op", "?")
    op_value  = (op_dict.get("value") or "")[:25]
    step_idx  = item.get("step_idx", 0)
    total_steps = item.get("total_steps", 1)

    # Extract task description from question field
    task_desc = ""
    for line in item.get("question", "").split("\n"):
        if line.startswith("Task:"):
            task_desc = line.replace("Task:", "").strip()[:65]
            break

    op_str = op_type
    if op_value:
        op_str += f"={op_value}"

    return (
        f"[{global_idx}] {domain}/{website} | {op_str} "
        f"| Step {step_idx + 1}/{total_steps} | \"{task_desc}\""
    )


# ── LLM Helper ───────────────────────────────────────────────────────────────

def call_llm(
    client: openai.OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
) -> str:
    """
    Call the Together AI LLM (OpenAI-compatible endpoint) with retry logic.

    Returns the response text, or "" on persistent failure.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"    [Retry {attempt + 1}/{RETRY_ATTEMPTS}] {exc} — waiting {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [FAILED after {RETRY_ATTEMPTS} attempts] {exc}")
    return ""


def parse_indices(text: str, lo: int, hi: int) -> List[int]:
    """
    Extract integer indices from LLM response text.

    Only returns integers in [lo, hi).  Preserves first-occurrence order
    and deduplicates.
    """
    seen: set = set()
    result: List[int] = []
    for tok in re.findall(r"\b(\d+)\b", text):
        idx = int(tok)
        if lo <= idx < hi and idx not in seen:
            result.append(idx)
            seen.add(idx)
    return result


# ── Stage 1: Batch Coarse Filter ─────────────────────────────────────────────

_STAGE1_SYSTEM = """\
You are an expert in web automation and AI training-data curation.
Your task is to identify the MOST REPRESENTATIVE and educationally valuable training
examples for a web navigation model that learns to:
  - Select the correct HTML element from ~200 candidates on a live webpage
  - Predict the action: CLICK, TYPE, or SELECT (with value)

Criteria for a "good" training example:
  1. REPRESENTATIVENESS  — covers common, real-world web navigation patterns
  2. DIVERSITY VALUE      — represents a pattern not well-covered by simpler examples
  3. APPROPRIATE DIFFICULTY — not trivially easy, not impossibly ambiguous
  4. GENERALIZABILITY     — the strategy learned transfers to other websites/tasks
"""

_STAGE1_USER = """\
Below are {n} training examples from the Mind2Web dataset.
Each line: [global_index] Domain/Website | ActionType:Value | Step S/TotalSteps | "Task"

{summaries}

Select the {top_n} MOST representative and educationally valuable examples from this batch.
Consider coverage of different action types, real-world task importance, and learning value.

Respond with ONLY a comma-separated list of the chosen global indices.
Example: 42, 137, 891, 1203, 2045\
"""


def stage1_batch_filter(
    client: openai.OpenAI,
    model: str,
    train_data: List[Dict],
    summaries: List[str],
    batch_size: int,
    top_per_batch: int,
) -> List[int]:
    """
    Stage 1: Process all training data in batches and collect finalist indices.

    Returns a deduplicated list of finalist global indices (in first-seen order).
    """
    n_total   = len(train_data)
    starts    = list(range(0, n_total, batch_size))
    n_batches = len(starts)
    finalists: List[int] = []
    seen: set = set()

    print(f"\n  Stage 1 — Coarse filter")
    print(f"  {n_total} samples | {n_batches} batches of {batch_size} | "
          f"top-{top_per_batch} per batch → up to {n_batches * top_per_batch} finalists\n")

    for batch_num, start in enumerate(starts, 1):
        end   = min(start + batch_size, n_total)
        batch = summaries[start:end]

        user_prompt = _STAGE1_USER.format(
            n=len(batch),
            summaries="\n".join(batch),
            top_n=top_per_batch,
        )

        response = call_llm(
            client, model,
            system=_STAGE1_SYSTEM,
            user=user_prompt,
            max_tokens=128,
        )

        selected = parse_indices(response, lo=start, hi=end)

        # Fallback: evenly spaced if LLM returned nothing valid
        if not selected:
            step = max(1, (end - start) // top_per_batch)
            selected = list(range(start, end, step))[:top_per_batch]
            print(f"  Batch {batch_num:3}/{n_batches} [{start:4}:{end:4}] "
                  f"⚠ parse failed → fallback {selected}")
        else:
            print(f"  Batch {batch_num:3}/{n_batches} [{start:4}:{end:4}] "
                  f"✓ selected {selected}")

        for idx in selected:
            if idx not in seen:
                finalists.append(idx)
                seen.add(idx)

    print(f"\n  Stage 1 done: {len(finalists)} unique finalists")
    return finalists


# ── Stage 2: Final Selection ──────────────────────────────────────────────────

_STAGE2_SYSTEM = """\
You are a curriculum designer for a web navigation AI system.
You have pre-screened a pool of candidate training examples.
Your job is to select the final {top_k} examples that together form the BEST
possible {top_k}-example training set.

The ideal set should:
  1. Cover ALL three action types: CLICK, TYPE, SELECT
  2. Span multiple web domains: Travel, Shopping, Finance, Social, Government, etc.
  3. Include a range of task complexity (short vs. long multi-step tasks)
  4. Teach transferable strategies — each example should offer a distinct lesson
  5. Minimise redundancy — no two examples should teach the same thing
"""

_STAGE2_USER = """\
You are selecting the final {top_k} training examples for a web navigation model.
These {n} candidates were pre-screened from a pool of {total} total samples.

{summaries}

Choose the best {top_k} examples that together give maximal coverage and learning value.
Briefly state your selection strategy (2–3 sentences), then output:

SELECTED: <comma-separated global indices, exactly {top_k} values>

Example final line:
SELECTED: 42, 137, 891, 1203, 2045, 3102, 4001, 512, 888, 2234\
"""


def stage2_final_select(
    client: openai.OpenAI,
    model: str,
    finalists: List[int],
    summaries: List[str],
    top_k: int,
    total_n: int,
) -> List[int]:
    """
    Stage 2: Show all finalists to the LLM and pick the final top_k.

    Returns a list of exactly top_k global indices.
    """
    finalist_summaries = [summaries[i] for i in finalists]

    user_prompt = _STAGE2_USER.format(
        top_k=top_k,
        n=len(finalists),
        total=total_n,
        summaries="\n".join(finalist_summaries),
    )
    system = _STAGE2_SYSTEM.format(top_k=top_k)

    print(f"\n  Stage 2 — Final selection of {top_k} from {len(finalists)} finalists")
    response = call_llm(
        client, model,
        system=system,
        user=user_prompt,
        max_tokens=600,
    )

    print(f"\n  {'─'*55}")
    print("  LLM reasoning & selection:")
    print(f"  {'─'*55}")
    for line in response.splitlines():
        print(f"  {line}")
    print(f"  {'─'*55}")

    # Parse "SELECTED: ..." line
    selected: List[int] = []
    for line in response.splitlines():
        if "SELECTED:" in line.upper():
            after = line[line.upper().index("SELECTED:") + len("SELECTED:"):].strip()
            selected = parse_indices(after, lo=0, hi=len(summaries))
            break

    # Fallback: any valid finalist index in full response
    if not selected:
        finalist_set = set(finalists)
        selected = [i for i in parse_indices(response, lo=0, hi=len(summaries))
                    if i in finalist_set]

    # Ensure exactly top_k, padding from finalists if needed
    selected_set = set(selected)
    if len(selected) < top_k:
        print(f"  ⚠ Only parsed {len(selected)}/{top_k} indices — padding from finalists")
        for idx in finalists:
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) == top_k:
                break

    return selected[:top_k]


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_selection(train_data: List[Dict], selected_indices: List[int]) -> None:
    """Print diversity statistics for the selected subset."""
    domains  = Counter()
    ops      = Counter()
    websites: set = set()
    positions: List[float] = []

    for i in selected_indices:
        item = train_data[i]
        domains[item.get("domain", "?")] += 1
        ops[item.get("operation", {}).get("op", "?")] += 1
        websites.add(item.get("website", "?"))
        total = max(1, item.get("total_steps", 1))
        positions.append(item.get("step_idx", 0) / total)

    avg_pos = sum(positions) / len(positions) if positions else 0.0
    print(f"\n  Domains    : {dict(domains)}")
    print(f"  Operations : {dict(ops)}")
    print(f"  Unique websites: {len(websites)} / {len(selected_indices)}")
    print(f"  Avg step position (0=start, 1=end): {avg_pos:.2f}")


# ── Config Update ─────────────────────────────────────────────────────────────

def update_config(top_k: int) -> None:
    config: Dict = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    key = f"mind2web_llm{top_k}"
    config[key] = {
        "train_data": f"./eval/mind2web/data/mind2web_train_llm{top_k}.jsonl",
        "val_data":   "./eval/mind2web/data/mind2web_val.jsonl",
        "test_data":  "./eval/mind2web/data/mind2web_test.jsonl",
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\n  Updated {CONFIG_PATH}  →  added \"{key}\"")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-based training subset selection for Mind2Web"
    )
    parser.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Final number of samples to select (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Samples per batch in Stage 1 (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--top_per_batch", type=int, default=DEFAULT_TOP_PER_BATCH,
        help=f"Finalists to pick per Stage-1 batch (default: {DEFAULT_TOP_PER_BATCH})"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Together AI model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max_finalists", type=int, default=None,
        help="Cap Stage-2 finalist pool to this many (default: use all Stage-1 outputs)"
    )
    args = parser.parse_args()

    load_dotenv()

    # ── Connect to Together AI ────────────────────────────────────────────────
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment or .env file")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    )

    print("\n" + "=" * 60)
    print("  LLM-based Training Subset Selection — Mind2Web")
    print("=" * 60)
    print(f"  Model          : {args.model}")
    print(f"  Final top-K    : {args.top_k}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Top / batch    : {args.top_per_batch}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_data = load_jsonl(TRAIN_PATH)
    print(f"\n  Loaded {len(train_data)} training samples")

    # ── Build compact summaries ───────────────────────────────────────────────
    print("  Building compact summaries ...")
    summaries = [make_compact_summary(item, i) for i, item in enumerate(train_data)]

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    finalists = stage1_batch_filter(
        client, args.model,
        train_data, summaries,
        batch_size=args.batch_size,
        top_per_batch=args.top_per_batch,
    )

    if args.max_finalists and len(finalists) > args.max_finalists:
        finalists = finalists[:args.max_finalists]
        print(f"  Capped finalist pool to {args.max_finalists}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    selected_indices = stage2_final_select(
        client, args.model,
        finalists, summaries,
        top_k=args.top_k,
        total_n=len(train_data),
    )

    # ── Report ────────────────────────────────────────────────────────────────
    report_selection(train_data, selected_indices)

    print(f"\n  {'='*55}")
    print(f"  Final {args.top_k} selected samples:")
    print(f"  {'='*55}")
    for rank, idx in enumerate(selected_indices, 1):
        print(f"  #{rank:2}: {summaries[idx]}")

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subset   = [train_data[i] for i in selected_indices]
    out_path = os.path.join(OUTPUT_DIR, f"mind2web_train_llm{args.top_k}.jsonl")
    save_jsonl(subset, out_path)

    meta = {
        "method":           "llm_select",
        "model":            args.model,
        "top_k":            args.top_k,
        "batch_size":       args.batch_size,
        "top_per_batch":    args.top_per_batch,
        "selected_indices": selected_indices,
        "finalist_indices": finalists,
        "n_total":          len(train_data),
    }
    meta_path = os.path.join(OUTPUT_DIR, f"llm{args.top_k}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {meta_path}")

    update_config(args.top_k)

    print(f"\n  {'='*60}")
    print(f"  DONE — {args.top_k} samples selected via LLM")
    print(f"  {'='*60}")
    print(f"\n  Next: run ACE training:")
    print(f"    python -m eval.mind2web.run \\")
    print(f"      --task_name mind2web_llm{args.top_k} \\")
    print(f"      --mode offline --skip_initial_test \\")
    print(f"      --eval_steps {args.top_k} \\")
    print(f"      --save_path results/mind2web_llm{args.top_k}")


if __name__ == "__main__":
    main()
