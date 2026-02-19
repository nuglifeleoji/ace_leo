#!/usr/bin/env python3
"""
Generate semantic embeddings for LSAT-LR training data.

Uses OpenAI text-embedding-3-large (same as mind2web) to embed each
sample's passage + question.  Embeddings are saved as a .npy file for
reuse in downstream clustering experiments.

Cost estimate: ~300 samples × ~300 tokens avg = ~90K tokens → ~$0.01

Output:
    eval/lsat_lr/data/embeddings.npy        — (N, 3072) float32 array
    eval/lsat_lr/data/embeddings_meta.json  — metadata

Usage:
    python -m eval.lsat_lr.embed_train
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv

load_dotenv()

TRAIN_PATH    = "./eval/lsat_lr/data/lsat_lr_train.jsonl"
OUTPUT_DIR    = "./eval/lsat_lr/data"
EMBED_PATH    = os.path.join(OUTPUT_DIR, "embeddings.npy")
PARTIAL_PATH  = os.path.join(OUTPUT_DIR, "embeddings_partial.npy")
META_PATH     = os.path.join(OUTPUT_DIR, "embeddings_meta.json")
EMBED_MODEL   = "text-embedding-3-large"
MAX_CHARS     = 8000   # passage ≤ 800 chars + question ≤ 400 chars → well within limit
BATCH_SIZE    = 100


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_text(sample: Dict) -> str:
    """Combine passage + question stem for embedding (options excluded)."""
    passage  = (sample.get("context") or "").strip()
    question = (sample.get("question") or "").strip()
    # Only use the question stem (before the options block)
    q_stem = question.split("\n\n(A)")[0].split("\n(A)")[0].strip()
    combined = f"Passage: {passage}\n\nQuestion: {q_stem}"
    return combined[:MAX_CHARS]


def embed_samples(train_data: List[Dict]) -> np.ndarray:
    import openai

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")

    client = openai.OpenAI(api_key=api_key)
    texts  = [build_text(s) for s in train_data]

    all_embeddings: list = []
    total_tokens   = 0
    start          = time.time()

    # Resume from partial checkpoint
    start_idx = 0
    if os.path.exists(PARTIAL_PATH):
        partial   = np.load(PARTIAL_PATH)
        start_idx = partial.shape[0]
        all_embeddings = list(partial)
        print(f"  Resuming from checkpoint: {start_idx} done")

    print(f"Embedding {len(texts)} samples with {EMBED_MODEL} ...")
    batch_size = BATCH_SIZE
    i = start_idx

    while i < len(texts):
        batch = texts[i: i + batch_size]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        except Exception as e:
            err = str(e)
            if "max_tokens_per_request" in err and len(batch) > 1:
                batch_size = max(1, len(batch) // 2)
                print(f"  [WARN] batch too large → reduce to {batch_size}")
                continue
            elif "rate_limit" in err.lower() or "429" in err:
                print("  [RATE LIMIT] waiting 5s...")
                time.sleep(5)
                continue
            else:
                raise

        all_embeddings.extend([item.embedding for item in resp.data])
        total_tokens += resp.usage.total_tokens
        i += len(batch)

        elapsed = time.time() - start
        print(f"  [{i:>3}/{len(texts)}] tokens={total_tokens:,}  "
              f"rate={i/elapsed:.1f} samples/s", flush=True)

        if i % 200 == 0 or i >= len(texts):
            np.save(PARTIAL_PATH, np.array(all_embeddings, dtype=np.float32))

        time.sleep(0.3)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"\nDone!  shape={embeddings.shape}  tokens={total_tokens:,}  "
          f"cost≈${total_tokens/1_000_000*0.13:.4f}")
    return embeddings, total_tokens


def main() -> None:
    if os.path.exists(EMBED_PATH):
        X = np.load(EMBED_PATH)
        print(f"Embeddings already exist: {EMBED_PATH}  shape={X.shape}")
        print("Use --force to regenerate.")
        return

    train_data = load_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_data)} samples from {TRAIN_PATH}")

    embeddings, total_tokens = embed_samples(train_data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(EMBED_PATH, embeddings)
    print(f"Saved → {EMBED_PATH}")

    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)

    meta = {
        "model":         EMBED_MODEL,
        "n_samples":     len(train_data),
        "embedding_dim": embeddings.shape[1],
        "total_tokens":  total_tokens,
        "max_chars":     MAX_CHARS,
        "train_path":    TRAIN_PATH,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {META_PATH}")


if __name__ == "__main__":
    main()
