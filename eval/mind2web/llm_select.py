#!/usr/bin/env python3
"""
LLM-based training subset selection for Mind2Web.

Supports Together AI (Llama 3.x) and OpenAI (GPT-4o-mini / GPT-4o).

Selection strategy (data-driven v8 — based on empirical analysis of 6 random seeds):
  1. MID-STAGE FOCUS     — ≥6/15 examples must be mid-stage steps (strongest predictor)
  2. SHORT TASKS         — prefer total_steps in [4,12]; at most 2 tasks with >15 steps
  3. NO SELECT           — SELECT degrades performance; include ≤1, ideally 0
  4. CLICK-HEAVY         — 80%+ CLICK is correct; ≥2 TYPE for input-field coverage
  5. DOMAIN BALANCE      — Travel≤55%, with Shopping and Entertainment well represented

Method — Two-stage tournament:
  Stage 1 (Batch coarse filter):
    All 4477 training samples are summarized into compact descriptors
    and fed to the LLM in batches. For each batch, the LLM returns the
    top TOP_PER_BATCH finalists → ~220 candidates.

  Stage 2 (Self-determined final selection):
    The finalists are shown in one call. The LLM reasons about coverage and
    selects the MINIMUM SUFFICIENT subset (8–25 examples), writing a coverage
    report for CLICK/TYPE/SELECT patterns learned, and explaining why it stops.

Prerequisites:
  TOGETHER_API_KEY or OPENAI_API_KEY must be set in .env or environment.

Usage:
    # Default: Together AI Llama-3.3-70B, pick top 20
    python -m eval.mind2web.llm_select --top_k 20

    # Use OpenAI GPT-4o-mini
    python -m eval.mind2web.llm_select --provider openai --model gpt-4o-mini --top_k 20

    # Use strong Together model for Stage 2
    python -m eval.mind2web.llm_select --top_k 20 --stage2_model meta-llama/Llama-3.3-70B-Instruct-Turbo
"""
import os
import re
import json
import time
import argparse
from typing import List, Dict, Optional
from collections import Counter

import openai
from dotenv import load_dotenv

# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_PATH  = "./eval/mind2web/data/mind2web_train.jsonl"
OUTPUT_DIR  = "./eval/mind2web/data"
CONFIG_PATH = "./eval/mind2web/data/sample_config.json"

# Together AI defaults
DEFAULT_MODEL_TOGETHER       = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_MODEL_TOGETHER_FAST  = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# OpenAI defaults
DEFAULT_MODEL_OPENAI         = "gpt-4o-mini"

DEFAULT_TOP_K         = 20
DEFAULT_BATCH_SIZE    = 100
DEFAULT_TOP_PER_BATCH = 5

RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5   # seconds between retries


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


# ── Distribution Helpers ──────────────────────────────────────────────────────

def compute_op_distribution(train_data: List[Dict], top_k: int) -> str:
    """
    Compute operation type distribution of the full training set and return
    a formatted string showing the distribution and target counts for top_k.
    """
    ops = Counter(item.get("operation", {}).get("op", "CLICK") for item in train_data)
    total = sum(ops.values())
    lines = ["Full training-set operation distribution "
             f"({total} samples) — your selection should reflect this:"]
    for op, count in ops.most_common():
        pct = count / total
        target = round(pct * top_k)
        target = max(1, target)  # at least 1 of each present op type
        lines.append(f"  {op:<8}: {count:5d} / {pct*100:5.1f}%  "
                     f"→ target in your {top_k} selections: ~{target} (±2)")
    return "\n".join(lines)


# ── Compact Summary ───────────────────────────────────────────────────────────

def _difficulty_hint(item: Dict) -> str:
    """
    Heuristic difficulty label based on task length and step position ONLY.
    Operation type is intentionally excluded — a CLICK identifying the correct
    element among many candidates is just as hard as a TYPE/SELECT.
      easy   — short task, early step
      medium — mid-range steps or medium-length tasks
      hard   — late-stage steps (>= 2/3 into task) or long tasks (>= 15 steps)
    """
    step  = item.get("step_idx", 0)
    total = max(1, item.get("total_steps", 1))
    pos   = step / (total - 1) if total > 1 else 0.0

    score = 0
    if total >= 15:    score += 2
    elif total >= 8:   score += 1
    if pos >= 0.67:    score += 2
    elif pos >= 0.33:  score += 1
    # NOTE: op type intentionally removed — difficulty is about element identification,
    # not operation label.

    if score <= 1:   return "easy"
    elif score <= 3: return "medium"
    else:            return "hard"


def _is_keystone(item: Dict) -> bool:
    """
    A 'keystone' step is the core decision point of a task — not pure navigation,
    not the very first step of a trivial task.

    Determined purely by task length and step position (operation type excluded
    to avoid biasing the LLM toward TYPE/SELECT):
      - total_steps in [4, 20]  (complex enough to transfer)
      - step position in [0.2, 0.8]  (past setup, before wrap-up)
    Steps from very short tasks (<=3) are never keystone.
    """
    step  = item.get("step_idx", 0)
    total = max(1, item.get("total_steps", 1))
    pos   = step / (total - 1) if total > 1 else 0.0

    if total <= 3:
        return False
    # NOTE: op type intentionally excluded — CLICK can be just as much a keystone
    # decision as TYPE/SELECT.
    if 4 <= total <= 20 and 0.20 <= pos <= 0.80:
        return True
    return False


def make_compact_summary(item: Dict, global_idx: int) -> str:
    """
    Create a compact one-line summary for Stage 1 batch screening.

    Format:
        [IDX] Domain/Website | OP:Value | Step S/T(pos_label) | diff:LEVEL | "Task"
    """
    domain   = item.get("domain", "?")
    website  = item.get("website", "?")
    op_dict  = item.get("operation", {})
    op_type  = op_dict.get("op", "?")
    op_value = (op_dict.get("value") or "")[:20]
    step     = item.get("step_idx", 0)
    total    = item.get("total_steps", 1)
    diff     = _difficulty_hint(item)

    task_desc = ""
    for line in item.get("question", "").split("\n"):
        if line.startswith("Task:"):
            task_desc = line.replace("Task:", "").strip()[:65]
            break

    op_str = op_type + (f":{op_value}" if op_value else "")
    pos_label = ("early" if step / max(total - 1, 1) < 0.33
                 else "mid" if step / max(total - 1, 1) < 0.67 else "late")
    keystone_marker = "★" if _is_keystone(item) else " "

    return (
        f"[{global_idx}]{keystone_marker}{domain}/{website} | {op_str} "
        f"| Step {step+1}/{total}({pos_label}) | diff:{diff} | \"{task_desc}\""
    )


def make_rich_summary(item: Dict, global_idx: int) -> str:
    """
    Create a richer summary for Stage 2 final selection — includes
    the first few candidate elements so the LLM can judge ambiguity.
    """
    base   = make_compact_summary(item, global_idx)
    op_val = (item.get("operation", {}).get("value") or "")

    # Extract a snippet of the page context (candidate elements)
    question = item.get("question", "")
    ctx_lines = []
    in_ctx = False
    for line in question.split("\n"):
        if "Candidate elements" in line or line.startswith("[0]"):
            in_ctx = True
        if in_ctx:
            ctx_lines.append(line.strip())
        if len(ctx_lines) >= 4:
            break
    ctx_snippet = " | ".join(ctx_lines[:3]) if ctx_lines else ""

    target = f" → target:{op_val[:30]}" if op_val else ""
    ctx    = f"\n      ctx: {ctx_snippet}" if ctx_snippet else ""
    return f"{base}{target}{ctx}"


# ── LLM Client Factory ────────────────────────────────────────────────────────

def make_client(provider: str) -> openai.OpenAI:
    load_dotenv()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env or environment")
        return openai.OpenAI(api_key=api_key)
    else:  # together
        api_key = os.getenv("TOGETHER_API_KEY", "")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in .env or environment")
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )


# ── LLM Helper ───────────────────────────────────────────────────────────────

def call_llm(
    client: openai.OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
) -> str:
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
                print(f"    [Retry {attempt+1}/{RETRY_ATTEMPTS}] {exc} — waiting {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [FAILED after {RETRY_ATTEMPTS} attempts] {exc}")
    return ""


def parse_indices(text: str, lo: int, hi: int) -> List[int]:
    seen, result = set(), []
    for tok in re.findall(r"\b(\d+)\b", text):
        idx = int(tok)
        if lo <= idx < hi and idx not in seen:
            result.append(idx)
            seen.add(idx)
    return result


# ── Stage 1: Batch Coarse Filter ─────────────────────────────────────────────

_STAGE1_SYSTEM = """\
You are pre-screening training examples for a web navigation AI curriculum.
Your goal: identify examples that could be CANONICAL — the clearest, most
representative, and most transferable examples of real web navigation.

A CANONICAL candidate:
  1. REPRESENTATIVE — the navigation pattern appears frequently across many sites
  2. CLEAN REASONING — the correct element is identifiable by clear logic (not luck)
  3. TRANSFERABLE — the lesson applies broadly, not just to this one website
  4. REAL-WORLD TASK — booking, shopping, searching, forms, media management

Actively REJECT:
  - Target element is trivially obvious (only button or text box on the page)
  - Step 1 of a very short task (total_steps <= 3) — too little context to generalise
  - Highly site-specific steps that don't generalise beyond this workflow
  - Two examples from the same website teaching the same navigation pattern

Do NOT favour any operation type (CLICK / TYPE / SELECT) — all three appear
equally in real web navigation and all can be equally canonical.
"""

_STAGE1_USER = """\
Below are {n} training examples. Format per line:
  [global_index]★? Domain/Website | Action:Value | Step(position) | diff:LEVEL | "Task"
  (★ = keystone step — core decision point of the task)

{summaries}

Select the {top_n} BEST examples from this batch.
Focus on transferable element-identification reasoning. Do NOT favour any operation
type (CLICK/TYPE/SELECT) over another — judge solely by learning value.

Respond with ONLY a comma-separated list of chosen global indices.
Example: 42, 137, 891\
"""


def stage1_batch_filter(
    client: openai.OpenAI,
    model: str,
    train_data: List[Dict],
    summaries: List[str],
    batch_size: int,
    top_per_batch: int,
) -> List[int]:
    n_total   = len(train_data)
    starts    = list(range(0, n_total, batch_size))
    n_batches = len(starts)
    finalists: List[int] = []
    seen: set = set()

    print(f"\n  Stage 1 — Coarse filter")
    print(f"  {n_total} samples | {n_batches} batches of ≤{batch_size} | "
          f"top-{top_per_batch}/batch → up to {n_batches * top_per_batch} finalists\n")

    for batch_num, start in enumerate(starts, 1):
        end   = min(start + batch_size, n_total)
        batch = summaries[start:end]

        user_prompt = _STAGE1_USER.format(
            n=len(batch),
            summaries="\n".join(batch),
            top_n=top_per_batch,
        )
        response = call_llm(client, model,
                            system=_STAGE1_SYSTEM,
                            user=user_prompt,
                            max_tokens=128)

        selected = parse_indices(response, lo=start, hi=end)

        if not selected:
            step = max(1, (end - start) // top_per_batch)
            selected = list(range(start, end, step))[:top_per_batch]
            print(f"  Batch {batch_num:3}/{n_batches} [{start:4}:{end:4}] "
                  f"⚠ fallback → {selected}")
        else:
            print(f"  Batch {batch_num:3}/{n_batches} [{start:4}:{end:4}] "
                  f"✓ {selected}")

        for idx in selected:
            if idx not in seen:
                finalists.append(idx)
                seen.add(idx)

    print(f"\n  Stage 1 done: {len(finalists)} unique finalists")
    return finalists


# ── Stage 2: Final Selection ──────────────────────────────────────────────────

_STAGE2_SYSTEM = """\
You are a professor selecting CANONICAL examples for a web navigation AI curriculum
— the essential textbook examples every student must master.

━━ WHAT MAKES AN EXAMPLE "CANONICAL" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  REPRESENTATIVE  — captures a pattern that appears frequently in real web tasks
  CLEAN & CLEAR   — the correct action follows from logical reasoning, not guessing
  GENERALIZABLE   — the strategy transfers to other websites and task types
  WELL-POSITIONED — at a meaningful decision point in the task flow (not trivial setup)

━━ OPERATION TYPE GUIDANCE (data-driven, not a hard target) ━━━━━━━━━━━━━━━
  CLICK is the dominant operation (~80% of real web navigation). Your selection
  will and SHOULD be CLICK-heavy — that is correct and expected.
  • TYPE  — teaches input-field identification. Include ≥ 2 TYPE examples.
  • SELECT — rare, hard to generalize, and empirically degrades performance when
    over-represented. Include AT MOST 1 SELECT example; ideally 0.

━━ TASK LENGTH (empirically important) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRONGLY prefer tasks with total_steps in [4, 12].
  Short-to-medium tasks produce the cleanest learning signal — each step's
  relationship to the goal is unambiguous. Long tasks (>15 steps) dilute the
  signal and should be included only when they contain an exceptional mid-stage
  step. Include AT MOST 2 tasks with total_steps > 15.

━━ DIFFICULTY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Medium difficulty as the baseline. Add a few hard examples only when they
  cover a pattern not otherwise represented. Avoid trivially easy examples
  (total_steps ≤ 3). Avoid exotic edge cases that don't transfer.

━━ HARD CONSTRAINTS (enforce strictly — check each one before submitting) ━━━
  H1. WEBSITE UNIQUENESS — exactly 1 step per website. Count your websites.
      If two selected examples share the same website, replace one. No exceptions.
  H2. TASK LENGTH — at most 2 examples with total_steps > 15.
  H3. DOMAIN BALANCE — no single domain (Travel / Shopping / Entertainment)
      may exceed 55% of your selection (≤8 out of 15).
  H4. STEP POSITION — MID steps are the primary learning signal. Your selection
      must include AT LEAST 6 mid-stage steps (position label = "mid").
      Also include ≥ 3 early and ≥ 3 late to cover task start/end patterns.
      Target distribution: early ≈ 4, mid ≈ 7, late ≈ 4.
  H5. ANTI-CONFUSION — check the rich summary's candidate elements. Reject
      examples where 2+ candidates share nearly identical text/label/type.
      The model cannot learn a clean strategy from ambiguous cases.
"""

_STAGE2_USER = """\
Select exactly {top_k} canonical training examples for a web navigation model.
{n} candidates were pre-screened from {total} total samples.
(★ = keystone step — core decision point of the task)

━━ TRAINING SET OPERATION DISTRIBUTION (for reference only) ━━━━━━━━━━━━━━━━
{op_distribution}
(Use this as context, not as an exact target. CLICK will be your majority;
 ensure ≥2 TYPE and ≥2 SELECT for skill coverage.)

── Compact summary of all candidates ──────────────────────────────────────────
{compact_summaries}

── Rich detail (operation target + first 3 candidate elements) ─────────────────
{rich_summaries}
────────────────────────────────────────────────────────────────────────────────

Before finalising, verify your selection against ALL hard constraints (H1–H5):
  ✓ H1: every website appears at most once → list your websites and check
  ✓ H2: at most 2 examples with total_steps > 15 → count long tasks
  ✓ H3: no domain exceeds 55% → count Travel / Shopping / Entertainment
  ✓ H4: mid-stage count ≥ 6 → count early / mid / late explicitly
  ✓ H5: no high-confusion examples → check rich summaries for identical candidates

Write a brief rationale (3–4 sentences) and your constraint verification,
then output:

SELECTED: <comma-separated global indices, exactly {top_k} values>
"""


def stage2_final_select(
    client: openai.OpenAI,
    model: str,
    finalists: List[int],
    compact_summaries: List[str],
    rich_summaries: List[str],
    top_k: int,
    total_n: int,
    train_data: List[Dict],
) -> List[int]:
    finalist_compact = "\n".join(compact_summaries[i] for i in finalists)
    finalist_rich    = "\n".join(rich_summaries[i]    for i in finalists)
    op_dist_str      = compute_op_distribution(train_data, top_k)

    user_prompt = _STAGE2_USER.format(
        top_k=top_k,
        n=len(finalists),
        total=total_n,
        op_distribution=op_dist_str,
        compact_summaries=finalist_compact,
        rich_summaries=finalist_rich,
    )

    print(f"\n  Stage 2 — Final selection of {top_k} canonical examples "
          f"from {len(finalists)} finalists")
    response = call_llm(client, model,
                        system=_STAGE2_SYSTEM,
                        user=user_prompt,
                        max_tokens=1000)

    print(f"\n  {'─'*60}")
    print("  LLM rationale & selection:")
    print(f"  {'─'*60}")
    for line in response.splitlines():
        print(f"  {line}")
    print(f"  {'─'*60}")

    # Parse "SELECTED: ..." line
    selected: List[int] = []
    for line in response.splitlines():
        if "SELECTED:" in line.upper():
            after   = line[line.upper().index("SELECTED:") + len("SELECTED:"):].strip()
            selected = parse_indices(after, lo=0, hi=total_n)
            break

    # Fallback: any valid finalist index found anywhere in the response
    if not selected:
        finalist_set = set(finalists)
        selected = [i for i in parse_indices(response, lo=0, hi=total_n)
                    if i in finalist_set]

    # Pad to top_k if needed
    if len(selected) < top_k:
        print(f"  ⚠ Parsed only {len(selected)}/{top_k} — padding from finalists")
        sel_set = set(selected)
        for idx in finalists:
            if idx not in sel_set:
                selected.append(idx)
                sel_set.add(idx)
            if len(selected) == top_k:
                break

    return selected[:top_k]


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_selection(train_data: List[Dict], selected_indices: List[int]) -> None:
    domains, ops, websites = Counter(), Counter(), set()
    positions, diffs, task_lengths = [], [], []
    keystones = 0

    for i in selected_indices:
        item = train_data[i]
        domains[item.get("domain", "?")] += 1
        ops[item.get("operation", {}).get("op", "?")] += 1
        websites.add(item.get("website", "?"))
        total = max(1, item.get("total_steps", 1))
        positions.append(item.get("step_idx", 0) / (total - 1) if total > 1 else 0.0)
        task_lengths.append(total)
        diffs.append(_difficulty_hint(item))
        if _is_keystone(item):
            keystones += 1

    diff_counts = Counter(diffs)
    early = sum(1 for p in positions if p < 0.33)
    mid   = sum(1 for p in positions if 0.33 <= p < 0.67)
    late  = sum(1 for p in positions if p >= 0.67)
    avg_pos = sum(positions) / len(positions) if positions else 0.0
    avg_len = sum(task_lengths) / len(task_lengths) if task_lengths else 0.0
    in_range = sum(1 for l in task_lengths if 6 <= l <= 15)

    print(f"\n  Domains         : {dict(domains)}")
    print(f"  Operations      : {dict(ops)}")
    print(f"  Difficulty      : {dict(diff_counts)}")
    print(f"  Unique websites : {len(websites)}/{len(selected_indices)}")
    print(f"  Step position   : avg={avg_pos:.2f}  early={early} mid={mid} late={late}")
    print(f"  Task length     : avg={avg_len:.1f}  in[6,15]={in_range}/{len(selected_indices)}")
    print(f"  Keystone steps  : {keystones}/{len(selected_indices)} ({'★' * keystones})")


# ── Config Update ─────────────────────────────────────────────────────────────

def update_config(top_k: int, suffix: str = "") -> None:
    config: Dict = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    key = f"mind2web_llm{top_k}{suffix}"
    config[key] = {
        "train_data": f"./eval/mind2web/data/mind2web_train_llm{top_k}{suffix}.jsonl",
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
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of canonical examples to select (default: {DEFAULT_TOP_K})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Samples per Stage-1 batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--top_per_batch", type=int, default=DEFAULT_TOP_PER_BATCH,
                        help=f"Finalists per Stage-1 batch (default: {DEFAULT_TOP_PER_BATCH})")
    parser.add_argument("--provider", type=str, default="together",
                        choices=["together", "openai"],
                        help="LLM provider (default: together)")
    parser.add_argument("--model", type=str, default=None,
                        help="Stage-1 model name (default: provider default)")
    parser.add_argument("--stage2_model", type=str, default=None,
                        help="Stage-2 model (default: same as --model)")
    parser.add_argument("--max_finalists", type=int, default=None,
                        help="Cap Stage-2 finalist pool size")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix for output filename, e.g. '_v2'")
    args = parser.parse_args()

    # Resolve model names
    if args.model is None:
        args.model = (DEFAULT_MODEL_OPENAI if args.provider == "openai"
                      else DEFAULT_MODEL_TOGETHER)
    if args.stage2_model is None:
        args.stage2_model = args.model

    client = make_client(args.provider)

    print("\n" + "=" * 65)
    print("  LLM-based Training Subset Selection — Mind2Web")
    print("=" * 65)
    print(f"  Provider       : {args.provider}")
    print(f"  Stage-1 model  : {args.model}")
    print(f"  Stage-2 model  : {args.stage2_model}")
    print(f"  Final top-K    : {args.top_k}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Top/batch      : {args.top_per_batch}")
    print("=" * 65)

    train_data = load_jsonl(TRAIN_PATH)
    print(f"\n  Loaded {len(train_data)} training samples")

    print("  Building compact & rich summaries ...")
    compact_sums = [make_compact_summary(item, i) for i, item in enumerate(train_data)]
    rich_sums    = [make_rich_summary(item, i)    for i, item in enumerate(train_data)]

    # Stage 1
    finalists = stage1_batch_filter(
        client, args.model,
        train_data, compact_sums,
        batch_size=args.batch_size,
        top_per_batch=args.top_per_batch,
    )

    if args.max_finalists and len(finalists) > args.max_finalists:
        finalists = finalists[:args.max_finalists]
        print(f"  Capped finalist pool to {args.max_finalists}")

    # Stage 2 — optionally use a different (stronger) model
    stage2_client = client
    if args.stage2_model != args.model:
        stage2_client = make_client(args.provider)

    selected_indices = stage2_final_select(
        stage2_client, args.stage2_model,
        finalists, compact_sums, rich_sums,
        top_k=args.top_k,
        total_n=len(train_data),
        train_data=train_data,
    )

    report_selection(train_data, selected_indices)

    print(f"\n  {'='*60}")
    print(f"  Final {args.top_k} selected samples:")
    print(f"  {'='*60}")
    for rank, idx in enumerate(selected_indices, 1):
        print(f"  #{rank:2}: {compact_sums[idx]}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix_str = args.suffix

    subset   = [train_data[i] for i in selected_indices]
    out_path = os.path.join(OUTPUT_DIR, f"mind2web_train_llm{args.top_k}{suffix_str}.jsonl")
    save_jsonl(subset, out_path)

    meta = {
        "method":           "llm_select_canonical",
        "provider":         args.provider,
        "stage1_model":     args.model,
        "stage2_model":     args.stage2_model,
        "top_k":            args.top_k,
        "batch_size":       args.batch_size,
        "top_per_batch":    args.top_per_batch,
        "selected_indices": selected_indices,
        "finalist_indices": finalists,
        "n_total":          len(train_data),
    }
    meta_path = os.path.join(OUTPUT_DIR, f"llm{args.top_k}{suffix_str}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")

    update_config(args.top_k, suffix_str)

    print(f"\n  {'='*65}")
    print(f"  DONE — {args.top_k} samples selected via LLM ({args.provider})")
    print(f"  {'='*65}")
    print(f"\n  Next: run ACE training:")
    print(f"    python -m eval.mind2web.run \\")
    print(f"      --task_name mind2web_llm{args.top_k}{suffix_str} \\")
    print(f"      --mode offline --skip_initial_test \\")
    print(f"      --eval_steps {args.top_k} \\")
    print(f"      --save_path results/mind2web_llm{args.top_k}{suffix_str}")


if __name__ == "__main__":
    main()
