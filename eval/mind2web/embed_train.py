#!/usr/bin/env python3
"""
Generate semantic embeddings for Mind2Web training data.

Uses OpenAI text-embedding-3-large to embed each sample's question + context.
Embeddings are saved as a .npy file for reuse in downstream experiments
(clustering, similarity analysis, etc.).

Cost estimate: ~4477 samples × ~2700 tokens avg = ~12M tokens → ~$1.60

Output:
    eval/mind2web/data/embeddings.npy       — (N, 3072) float32 array
    eval/mind2web/data/embeddings_meta.json  — metadata (model, timestamp, etc.)

Usage:
    python -m eval.mind2web.embed_train [--batch_size 100]
"""
import os
import json
import time
import argparse
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────

TRAIN_PATH = "./eval/mind2web/data/mind2web_train.jsonl"
OUTPUT_DIR = "./eval/mind2web/data"
EMBEDDING_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
PARTIAL_PATH = os.path.join(OUTPUT_DIR, "embeddings_partial.npy")
META_PATH = os.path.join(OUTPUT_DIR, "embeddings_meta.json")
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_CHARS = 24000  # ~6000 tokens, stay well under 8191 token limit


# ── Data Loading ────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── Embedding ───────────────────────────────────────────────────

def build_embedding_text(sample: Dict) -> str:
    """
    Combine question + context into a single text for embedding.

    Question is placed first (more semantically important for task identity),
    followed by truncated context (page structure / candidates).
    """
    question = sample.get("question", "")
    context = sample.get("context", "")
    combined = f"Question: {question}\n\nContext: {context}"
    if len(combined) > MAX_CHARS:
        combined = combined[:MAX_CHARS]
    return combined


def embed_samples(
    train_data: List[Dict],
    batch_size: int = 100,
) -> np.ndarray:
    """
    Embed all training samples using OpenAI text-embedding-3-large.

    Processes in batches with progress reporting.

    Returns:
        np.ndarray of shape (N, 3072), dtype float32
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
    all_embeddings = []
    total_tokens = 0
    start_time = time.time()

    # Resume from partial checkpoint if available
    start_idx = 0
    if os.path.exists(PARTIAL_PATH):
        partial = np.load(PARTIAL_PATH)
        start_idx = partial.shape[0]
        all_embeddings = list(partial)
        print(f"  Resuming from checkpoint: {start_idx} samples already done")

    print(f"Embedding {len(texts)} samples with {EMBEDDING_MODEL}...")
    print(f"  Max chars per sample: {MAX_CHARS}")
    print(f"  Batch size: {batch_size}")
    print(f"  Starting from index: {start_idx}")

    i = start_idx
    current_batch_size = batch_size
    while i < len(texts):
        batch = texts[i : i + current_batch_size]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
        except Exception as e:
            err_str = str(e)
            if "max_tokens_per_request" in err_str and len(batch) > 1:
                current_batch_size = max(1, len(batch) // 2)
                print(f"  [WARN] Batch too large, reducing to {current_batch_size}")
                continue
            elif "rate_limit" in err_str.lower() or "429" in err_str:
                wait = 5
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
                continue
            elif "maximum context length" in err_str.lower():
                # Single sample too long, truncate more aggressively
                for j in range(len(batch)):
                    if len(batch[j]) > 16000:
                        batch[j] = batch[j][:16000]
                        texts[i + j] = batch[j]
                print(f"  [WARN] Truncated long sample(s) in batch at index {i}")
                continue
            else:
                raise

        batch_embs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embs)
        total_tokens += response.usage.total_tokens

        i += len(batch)
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        print(f"  [{i:>5}/{len(texts)}] "
              f"tokens so far: {total_tokens:,} | "
              f"rate: {rate:.1f} samples/s")

        # Save partial checkpoint every 500 samples
        if i % 500 == 0 or i >= len(texts):
            partial = np.array(all_embeddings, dtype=np.float32)
            np.save(PARTIAL_PATH, partial)

        # Small delay to avoid rate limits (~1M TPM limit)
        time.sleep(0.5)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    elapsed = time.time() - start_time

    print(f"\nEmbedding complete!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Estimated cost: ${total_tokens / 1_000_000 * 0.13:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    return embeddings, total_tokens


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Mind2Web training data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=100,
        help="Batch size for OpenAI API calls (default: 100)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-generate embeddings even if cache exists"
    )
    args = parser.parse_args()

    # Check if already cached
    if os.path.exists(EMBEDDING_PATH) and not args.force:
        emb = np.load(EMBEDDING_PATH)
        print(f"Embeddings already exist: {EMBEDDING_PATH}")
        print(f"  Shape: {emb.shape}")
        print(f"  Use --force to regenerate")
        return

    # Load data
    train_data = load_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_data)} training samples from {TRAIN_PATH}")

    # Generate embeddings
    embeddings, total_tokens = embed_samples(train_data, batch_size=args.batch_size)

    # Save embeddings
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(EMBEDDING_PATH, embeddings)
    print(f"\nSaved embeddings to {EMBEDDING_PATH}")

    # Clean up partial checkpoint
    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)
        print(f"Removed partial checkpoint")

    # Save metadata
    meta = {
        "model": EMBEDDING_MODEL,
        "n_samples": len(train_data),
        "embedding_dim": embeddings.shape[1],
        "total_tokens": total_tokens,
        "max_chars": MAX_CHARS,
        "train_path": TRAIN_PATH,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()
