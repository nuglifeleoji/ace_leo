"""
Lesson-based clustering for SciKnowEval Chemistry L3.

Step 1 (this script):
  For each training sample, ask a small LLM (Qwen2.5-7B-Instruct-Turbo) to
  extract ONE concise, generalizable "chemistry lesson" — a rule or principle
  that the solved problem teaches — then embed all lessons with
  sentence-transformers.

Output:
  - eval/sciknow/data/sciknow_lessons.jsonl    (one lesson per sample)
  - eval/sciknow/data/sciknow_lesson_embeddings.npy

Usage:
    python -m eval.sciknow.lesson_generate
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH      = "./eval/sciknow/data/sciknow_chem_l3_train.jsonl"
LESSONS_PATH    = "./eval/sciknow/data/sciknow_lessons.jsonl"
EMBED_PATH      = "./eval/sciknow/data/sciknow_lesson_embeddings.npy"
EMBED_MODEL     = "all-MiniLM-L6-v2"

LESSON_MODEL    = "Qwen/Qwen2.5-7B-Instruct-Turbo"
API_PROVIDER    = "together"
MAX_WORKERS     = 16
MAX_TOKENS      = 128
RETRY_DELAY     = 2


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a chemistry tutor reviewing solved problems.
You are shown a single correctly-answered chemistry question.
Your job: output exactly ONE concise, generalizable chemistry lesson (1 sentence) \
that this problem teaches — a rule, principle, or reasoning strategy that would \
help solve OTHER chemistry problems too.
The rule must be specific enough to be actionable (not "understand chemistry").
Output ONLY the lesson sentence. No preamble, no numbering."""

_USER_TMPL = """\
Task type: {task}
Question: {question}
Correct answer: {answer}

What ONE generalizable chemistry lesson does this problem teach?"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(client, sample: Dict) -> str:
    question = sample.get("question", "")
    answer   = sample.get("target", "")
    task     = sample.get("task", "")
    prompt   = _USER_TMPL.format(task=task, question=question[:1000], answer=answer)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MAX_TOKENS and LESSON_MODEL,  # evaluated as LESSON_MODEL
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Attempt {attempt+1}/3] API error: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main() -> None:
    print("=" * 60)
    print("  SciKnowEval — Lesson Generation (Qwen2.5-7B)")
    print("=" * 60)

    train_data = load_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_data)} training samples")

    # ── resume: skip already-generated lessons ────────────────────────────────
    done_ids: set = set()
    existing_lessons: List[Dict] = []
    if os.path.exists(LESSONS_PATH):
        existing_lessons = load_jsonl(LESSONS_PATH)
        done_ids = {item["idx"] for item in existing_lessons}
        print(f"Resuming — {len(done_ids)} lessons already done")

    # ── set up Together client ────────────────────────────────────────────────
    if API_PROVIDER == "together":
        from together import Together
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        client  = Together(api_key=api_key)
    else:
        raise ValueError(f"Unsupported API_PROVIDER: {API_PROVIDER}")

    # ── generate lessons in parallel ──────────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    todo = [(i, s) for i, s in enumerate(train_data) if i not in done_ids]
    print(f"Generating {len(todo)} lessons with {MAX_WORKERS} workers ...")

    results: List[Dict] = list(existing_lessons)

    def _worker(args):
        idx, sample = args
        lesson = _call_llm(client, sample)
        return idx, lesson

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_worker, item): item[0] for item in todo}
        for i, fut in enumerate(as_completed(futures)):
            idx, lesson = fut.result()
            results.append({"idx": idx, "lesson": lesson,
                             "task": train_data[idx].get("task", "")})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(todo)} done ...")

    # sort by original index
    results.sort(key=lambda x: x["idx"])

    # ── save lessons ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LESSONS_PATH), exist_ok=True)
    with open(LESSONS_PATH, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} lessons → {LESSONS_PATH}")

    # ── sample output ─────────────────────────────────────────────────────────
    print("\nSample lessons:")
    for item in results[:3]:
        print(f"  [{item['task']}] {item['lesson'][:100]}")

    # ── embed lessons ─────────────────────────────────────────────────────────
    print(f"\nEmbedding lessons with {EMBED_MODEL} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    texts = [item["lesson"] or "no lesson" for item in results]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(EMBED_PATH, embeddings)
    print(f"Saved lesson embeddings → {EMBED_PATH}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
