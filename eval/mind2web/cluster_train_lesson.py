#!/usr/bin/env python3
"""
Lesson Generation for Mind2Web Training Subset Selection (3.1 Playbook Output).

Core idea:
    Run a "mini ACE" on every training sample using a small LLM (Qwen 2.5 7B via
    Together AI), simulating what the ACE curator would ADD to a playbook after
    seeing that example.  Each sample produces one "playbook lesson" sentence.

    These lessons are then embedded with OpenAI text-embedding-3-small and saved
    for downstream clustering (not done in this script).

Why this is better than embedding task descriptions:
    Two tasks with very different descriptions can teach the *same* navigation
    rule (e.g., "click the dropdown to filter results" appears in travel AND
    shopping tasks).  By clustering on what ACE *learns* rather than on what the
    task *says*, we select maximally diverse strategies instead of diverse domains.

Pipeline (this script):
    1. For each of the 4477 training samples, call Qwen 2.5 7B with an ACE-curator-
       style prompt: "After seeing this correctly-solved step, what one generalizable
       playbook rule would you add?"
       → saves to  eval/mind2web/data/lesson_cache.json  (resumable)

    2. Embed all 4477 lesson sentences with text-embedding-3-small (1536-dim).
       → saves to  eval/mind2web/data/lesson_embeddings.npy

Downstream (separate script, to be written later):
    K-means cluster lesson embeddings → select centroid-nearest sample per cluster.

Usage:
    # Generate lessons + embeddings for all training samples
    python -m eval.mind2web.cluster_train_lesson

    # Force re-generate all lessons (ignore cache)
    python -m eval.mind2web.cluster_train_lesson --regen_lessons

    # Force re-embed (ignore cached .npy)
    python -m eval.mind2web.cluster_train_lesson --regen_embeddings
"""
import os
import json
import time
import argparse
import numpy as np
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_PATH       = "./eval/mind2web/data/mind2web_train.jsonl"
LESSON_CACHE     = "./eval/mind2web/data/lesson_cache.json"
LESSON_EMB_PATH  = "./eval/mind2web/data/lesson_embeddings.npy"
OUTPUT_DIR       = "./eval/mind2web/data"

TOGETHER_MODEL   = "Qwen/Qwen2.5-7B-Instruct-Turbo"
EMBED_MODEL      = "text-embedding-3-small"

LESSON_SLEEP     = 0.5   # seconds between Together AI calls
EMBED_BATCH_SIZE = 200
EMBED_BATCH_SLEEP= 0.3
RETRY_ATTEMPTS   = 4
RETRY_DELAY      = 10


# ── IO ────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_lesson_cache() -> Dict[int, str]:
    if os.path.exists(LESSON_CACHE):
        with open(LESSON_CACHE, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return {}


def save_lesson_cache(cache: Dict[int, str]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LESSON_CACHE, "w") as f:
        json.dump({str(k): v for k, v in sorted(cache.items())}, f,
                  indent=2, ensure_ascii=False)


# ── LLM Clients ───────────────────────────────────────────────────────────────

def make_together_client():
    import openai
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set in .env")
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
        timeout=30.0,   # hard 30s timeout per call — prevents hanging
    )


def make_openai_client():
    import openai
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return openai.OpenAI(api_key=api_key)


# ── Lesson Generation (mini ACE curator simulation) ───────────────────────────

# We simulate the ACE Curator: given a correctly-solved navigation step,
# what ONE generalizable rule would you add to the playbook?
_CURATOR_SYSTEM = """\
You are the curator of a web navigation playbook.
You are shown a single correctly-solved web navigation step.
Your job: output exactly ONE concise, generalizable playbook rule (1 sentence)
that this example teaches — a rule that would help navigate OTHER websites too.
The rule must be specific enough to be actionable (not "click the right button").
Output ONLY the rule sentence. No preamble, no numbering."""

_CURATOR_USER_TMPL = """\
Task goal: {task}
Website: {website}
Step {step_n} of {total_steps} — {op}{value_part}
Target element: {target}
Action history so far: {history}

What ONE generalizable web-navigation rule does this step teach?"""


def _build_curator_prompt(sample: Dict) -> str:
    op_info = sample.get("operation", {})
    op = str(op_info.get("op", "CLICK"))
    value = str(op_info.get("value", "") or "")
    value_part = f' "{value[:60]}"' if value and value != "None" else ""

    # Extract target element from context (line with [correct] marker)
    target = sample.get("target_element", "")
    if not target:
        import re
        ctx = sample.get("context", "")
        m = re.search(r"\[correct\](.*?)(?:\n|\Z)", ctx, re.IGNORECASE)
        target = m.group(1).strip()[:120] if m else ctx[:100]

    # Action history: last line(s) of question before "Select the"
    question = sample.get("question", "")
    history = ""
    if "Actions completed:" in question:
        history_block = question.split("Actions completed:")[-1]
        history_block = history_block.split("Select the")[0].strip()
        # Last 2 history lines
        lines = [l.strip() for l in history_block.split("\n") if l.strip()]
        history = " → ".join(lines[-2:]) if lines else "none"
    else:
        history = "none"

    task = sample.get("task", "")
    if not task:
        # Extract from question "Task: ..."
        import re
        m = re.search(r"Task:\s*(.*?)(?:\n|Website:)", question, re.DOTALL)
        task = m.group(1).strip()[:200] if m else question[:200]

    return _CURATOR_USER_TMPL.format(
        task=task[:200],
        website=sample.get("website", "?"),
        step_n=sample.get("step_idx", 0) + 1,
        total_steps=sample.get("total_steps", 1),
        op=op,
        value_part=value_part,
        target=target[:120],
        history=history[:200],
    )


def generate_lessons(
    train_data: List[Dict],
    cache: Dict[int, str],
    client,
    regen: bool = False,
) -> Dict[int, str]:
    """
    For each training sample, ask Qwen 2.5 7B (ACE curator simulation):
    "What one generalizable playbook rule does this step teach?"
    
    Saves checkpoint every 100 samples.  Skips already-cached indices.
    """
    to_generate = [i for i in range(len(train_data))
                   if regen or i not in cache]
    total = len(to_generate)

    if total == 0:
        print(f"  All {len(train_data)} lessons already cached ✓")
        return cache

    print(f"  Generating lessons for {total} samples  "
          f"(already cached: {len(train_data) - total})")
    print(f"  Model: {TOGETHER_MODEL}\n")

    start_t = time.time()
    for done, idx in enumerate(to_generate):
        sample = train_data[idx]
        user_msg = _build_curator_prompt(sample)

        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = client.chat.completions.create(
                    model=TOGETHER_MODEL,
                    messages=[
                        {"role": "system", "content": _CURATOR_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=80,
                    temperature=0.0,
                )
                lesson = resp.choices[0].message.content.strip()
                # Strip common model prefixes
                for pfx in ("Rule: ", "Lesson: ", "Playbook rule: ",
                             "The rule is: ", "Generalizable rule: "):
                    if lesson.lower().startswith(pfx.lower()):
                        lesson = lesson[len(pfx):].strip()
                # Take only first sentence if multiple
                lesson = lesson.split("\n")[0].split(". ")[0]
                if not lesson.endswith("."):
                    lesson += "."
                cache[idx] = lesson
                break
            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    print(f"    [Retry {attempt+1}/{RETRY_ATTEMPTS}] idx={idx}: {e}"
                          f" — waiting {RETRY_DELAY}s")
                    time.sleep(RETRY_DELAY)
                else:
                    # Fallback to task description
                    fallback = sample.get("task", sample.get("question", ""))[:100]
                    cache[idx] = fallback
                    print(f"    [ERROR] idx={idx}: fallback used")

        time.sleep(LESSON_SLEEP)

        # Progress + checkpoint
        if (done + 1) % 100 == 0 or (done + 1) == total:
            elapsed = time.time() - start_t
            rate = (done + 1) / elapsed
            eta = (total - done - 1) / rate if rate > 0 else 0
            print(f"  [{done+1:>5}/{total}]  {rate:.1f} samples/s  "
                  f"ETA {eta/60:.1f}min  "
                  f"last: \"{cache[idx][:65]}\"")
            save_lesson_cache(cache)

    save_lesson_cache(cache)
    print(f"\n  Done. Lesson cache → {LESSON_CACHE}")
    return cache


# ── Lesson Embedding ──────────────────────────────────────────────────────────

def embed_lessons(lessons: List[str], client) -> np.ndarray:
    """
    Embed all lesson sentences with text-embedding-3-small (1536-dim).
    Lessons are short (~15 tokens each) → fast and cheap (~$0.01 for 4477).
    """
    n = len(lessons)
    print(f"  Embedding {n} lessons with {EMBED_MODEL}...")
    all_embs = []
    i = 0
    start_t = time.time()

    while i < n:
        batch = lessons[i: i + EMBED_BATCH_SIZE]
        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                embs = [item.embedding
                        for item in sorted(resp.data, key=lambda x: x.index)]
                all_embs.extend(embs)
                break
            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    print(f"    [Retry {attempt+1}/{RETRY_ATTEMPTS}] batch {i}: {e}"
                          f" — waiting {RETRY_DELAY}s")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        i += EMBED_BATCH_SIZE
        if i % 1000 < EMBED_BATCH_SIZE or i >= n:
            elapsed = time.time() - start_t
            print(f"    [{min(i, n):>5}/{n}]  {elapsed:.1f}s")
        time.sleep(EMBED_BATCH_SLEEP)

    arr = np.array(all_embs, dtype=np.float32)
    print(f"  Embedding shape: {arr.shape}")
    return arr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate ACE-curator-style lessons + embeddings for all Mind2Web training samples"
    )
    parser.add_argument(
        "--regen_lessons", action="store_true",
        help="Force re-generate lessons even if cache exists"
    )
    parser.add_argument(
        "--regen_embeddings", action="store_true",
        help="Force re-embed lessons even if .npy cache exists"
    )
    args = parser.parse_args()

    # ── Load training data ────────────────────────────────────────
    train_data = load_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_data)} training samples from {TRAIN_PATH}\n")

    # ── Step 1: Generate lessons (Qwen 2.5 7B) ───────────────────
    print("=" * 60)
    print("  Step 1 — Generate playbook lessons (Qwen 2.5 7B)")
    print("=" * 60)

    together_client = make_together_client()
    cache = load_lesson_cache()
    cache = generate_lessons(train_data, cache, together_client,
                             regen=args.regen_lessons)

    # Build ordered lesson list (fill any gaps with fallback)
    lessons = []
    missing = 0
    for i in range(len(train_data)):
        if i in cache:
            lessons.append(cache[i])
        else:
            lessons.append(
                train_data[i].get("task", train_data[i].get("question", ""))[:100]
            )
            missing += 1
    if missing:
        print(f"  [WARN] {missing} samples used fallback lesson (not in cache)")

    # Sample printout
    print("\n  Sample generated lessons:")
    for idx in [0, 100, 500, 1000, 2000, 3000, 4000]:
        if idx < len(train_data):
            d = train_data[idx]
            op = str(d.get("operation", {}).get("op", "?"))
            site = d.get("website", "?")
            print(f"    [{idx:>4}] ({op:6s}) {site:20s}  \"{lessons[idx][:70]}\"")

    # ── Step 2: Embed lessons ─────────────────────────────────────
    print()
    print("=" * 60)
    print("  Step 2 — Embed lessons (text-embedding-3-small)")
    print("=" * 60)

    openai_client = make_openai_client()

    if os.path.exists(LESSON_EMB_PATH) and not args.regen_embeddings:
        embeddings = np.load(LESSON_EMB_PATH)
        print(f"  Loaded cached embeddings: {embeddings.shape}")
        if embeddings.shape[0] != len(lessons):
            print(f"  [WARN] Size mismatch — re-embedding...")
            embeddings = embed_lessons(lessons, openai_client)
            np.save(LESSON_EMB_PATH, embeddings)
            print(f"  Saved → {LESSON_EMB_PATH}")
    else:
        embeddings = embed_lessons(lessons, openai_client)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(LESSON_EMB_PATH, embeddings)
        print(f"  Saved → {LESSON_EMB_PATH}")

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Lessons cache : {LESSON_CACHE}  ({len(cache)} entries)")
    print(f"  Embeddings    : {LESSON_EMB_PATH}  {embeddings.shape}")
    print()
    print("  Next step: run clustering (to be implemented)")
    print("    python -m eval.mind2web.cluster_lesson_select --clusters 10 15 20")


if __name__ == "__main__":
    main()
