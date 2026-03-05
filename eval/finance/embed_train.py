#!/usr/bin/env python3
"""
Generate semantic embeddings for FiNer training data.

FiNer is an XBRL named entity tagging task: each sample contains 4 financial
text sentences, each with one entity to be tagged with a US GAAP concept.

The context field is dominated by a long boilerplate XBRL tag-list header
(~2000 chars, identical across all samples).  To obtain *meaningful* semantic
embeddings we strip the boilerplate and keep only:
    • the numbered financial sentences (the actual content)
    • the ground-truth XBRL tags (to enrich semantic signal)

Cost estimate:  1000 samples × ~600 tokens avg ≈ 600K tokens → ~$0.08

Output:
    eval/finance/data/finer_embeddings.npy       — (N, 3072) float32 array
    eval/finance/data/finer_embeddings_meta.json — metadata

Usage:
    python -m eval.finance.embed_train [--batch_size 100] [--force]
"""

from __future__ import annotations

import json
import os
import re
import time
import argparse
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_PATH   = "./eval/finance/data/finer_train_batched_1000_samples.jsonl"
OUTPUT_DIR   = "./eval/finance/data"
EMBED_PATH   = os.path.join(OUTPUT_DIR, "finer_embeddings.npy")
PARTIAL_PATH = os.path.join(OUTPUT_DIR, "finer_embeddings_partial.npy")
META_PATH    = os.path.join(OUTPUT_DIR, "finer_embeddings_meta.json")
EMBED_MODEL  = "text-embedding-3-large"
MAX_CHARS    = 8000   # financial sentences are short; well under 8191-token limit


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_financial_sentences(context: str) -> str:
    """
    Strip the boilerplate XBRL tag-list header and return only the
    numbered financial questions.

    The context format is:
        "You are XBRL expert.  Here is a list of US GAAP tags options: ...
         ...Provide nothing else.
         1. What is best tag for entity \"X\" in sentence: \"Y\"?
         2. ...
         Output US GAAP tags:"

    We extract everything from the first numbered item up to
    "Output US GAAP tags:", which is the semantically meaningful part.
    """
    # Try to find the start of numbered questions
    match = re.search(r'\n1\.\s+What is best tag', context)
    if match:
        sentences_part = context[match.start():].strip()
        # Remove trailing "Output US GAAP tags:" if present
        sentences_part = re.sub(r'\nOutput US GAAP tags:.*$', '', sentences_part,
                                flags=re.DOTALL).strip()
        return sentences_part

    # Fallback: strip everything before "Provide nothing else."
    marker = "Provide nothing else."
    idx = context.find(marker)
    if idx != -1:
        return context[idx + len(marker):].strip()

    # Last resort: return truncated full context
    return context[:MAX_CHARS]


def build_embedding_text(sample: Dict) -> str:
    """
    Build the text to embed for one FiNer training sample.

    We use only the extracted financial sentences (the numbered questions).
    The boilerplate XBRL tag list and the ground-truth labels are excluded,
    so the embedding reflects purely the financial text content.
    """
    context   = sample.get("context", "")
    sentences = extract_financial_sentences(context)
    return sentences[:MAX_CHARS]


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_samples(train_data: List[Dict], batch_size: int = 100) -> tuple:
    """
    Embed all training samples using OpenAI text-embedding-3-large.

    Returns:
        (embeddings np.ndarray of shape (N, 3072), total_tokens int)
    """
    import openai

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to your .env file.\n"
            "Get one at https://platform.openai.com/api-keys"
        )
    client = openai.OpenAI(api_key=api_key)

    texts = [build_embedding_text(s) for s in train_data]

    # Quick sanity check on extraction quality
    print(f"\nSample embedding text (first sample, first 400 chars):")
    print(texts[0][:400])
    print("...\n")

    all_embeddings: list = []
    total_tokens   = 0
    start          = time.time()

    # Resume from partial checkpoint if available
    start_idx = 0
    if os.path.exists(PARTIAL_PATH):
        partial   = np.load(PARTIAL_PATH)
        start_idx = partial.shape[0]
        all_embeddings = list(partial)
        print(f"  Resuming from checkpoint: {start_idx} samples done")

    print(f"Embedding {len(texts)} samples with {EMBED_MODEL} ...")
    print(f"  Max chars per sample : {MAX_CHARS}")
    print(f"  Batch size           : {batch_size}")
    print(f"  Starting from index  : {start_idx}\n")

    current_batch = batch_size
    i = start_idx

    while i < len(texts):
        batch = texts[i: i + current_batch]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        except Exception as e:
            err = str(e)
            if "max_tokens_per_request" in err and len(batch) > 1:
                current_batch = max(1, len(batch) // 2)
                print(f"  [WARN] Batch too large → reducing to {current_batch}")
                continue
            elif "rate_limit" in err.lower() or "429" in err:
                print("  [RATE LIMIT] waiting 5s ...")
                time.sleep(5)
                continue
            elif "maximum context length" in err.lower():
                for j in range(len(batch)):
                    if len(batch[j]) > 6000:
                        batch[j] = batch[j][:6000]
                        texts[i + j] = batch[j]
                print(f"  [WARN] Truncated long sample(s) at index {i}")
                continue
            else:
                raise

        all_embeddings.extend([item.embedding for item in resp.data])
        total_tokens += resp.usage.total_tokens
        i += len(batch)

        elapsed = time.time() - start
        rate    = i / elapsed if elapsed > 0 else 0
        print(f"  [{i:>5}/{len(texts)}] "
              f"tokens: {total_tokens:,} | rate: {rate:.1f} samples/s")

        # Checkpoint every 200 samples
        if i % 200 == 0 or i >= len(texts):
            np.save(PARTIAL_PATH, np.array(all_embeddings, dtype=np.float32))

        time.sleep(0.3)   # stay under rate limits

    embeddings = np.array(all_embeddings, dtype=np.float32)
    elapsed    = time.time() - start

    print(f"\nEmbedding complete!")
    print(f"  Shape          : {embeddings.shape}")
    print(f"  Total tokens   : {total_tokens:,}")
    print(f"  Estimated cost : ${total_tokens / 1_000_000 * 0.13:.4f}")
    print(f"  Time           : {elapsed:.1f}s")

    return embeddings, total_tokens


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate semantic embeddings for FiNer training data"
    )
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if cache exists")
    args = parser.parse_args()

    if os.path.exists(EMBED_PATH) and not args.force:
        emb = np.load(EMBED_PATH)
        print(f"Embeddings already exist: {EMBED_PATH}")
        print(f"  Shape: {emb.shape}")
        print("Use --force to regenerate.")
        return

    train_data = load_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_data)} training samples from {TRAIN_PATH}")

    embeddings, total_tokens = embed_samples(train_data, batch_size=args.batch_size)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(EMBED_PATH, embeddings)
    print(f"\nSaved embeddings → {EMBED_PATH}")

    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)

    meta = {
        "model":         EMBED_MODEL,
        "n_samples":     len(train_data),
        "embedding_dim": embeddings.shape[1],
        "total_tokens":  total_tokens,
        "max_chars":     MAX_CHARS,
        "train_path":    TRAIN_PATH,
        "embedding_text": "financial_sentences only (boilerplate and target tags excluded)",
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {META_PATH}")


if __name__ == "__main__":
    main()
