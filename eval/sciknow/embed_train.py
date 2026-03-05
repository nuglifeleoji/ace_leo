"""
Generate semantic embeddings for SciKnowEval Chemistry L3 training data.

Uses sentence-transformers to encode each sample's question + answer into a
fixed-length vector, saved as sciknow_embeddings.npy.

Usage:
    python -m eval.sciknow.embed_train
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "./eval/sciknow/data/sciknow_chem_l3_train.jsonl"
EMBED_PATH  = "./eval/sciknow/data/sciknow_embeddings.npy"
MODEL_NAME  = "all-MiniLM-L6-v2"
MAX_CHARS   = 2000


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_embedding_text(sample: Dict) -> str:
    """Combine question text and correct answer for richer semantic signal."""
    question = sample.get("question", "")
    target   = sample.get("target", "")
    task     = sample.get("task", "")
    text = f"[{task}] {question}"
    if target:
        text += f"\nAnswer: {target}"
    return text[:MAX_CHARS]


def main() -> None:
    print("=" * 60)
    print("  SciKnowEval — Generating Semantic Embeddings")
    print("=" * 60)

    train_data = load_jsonl(TRAIN_PATH)
    print(f"Loaded {len(train_data)} training samples")

    texts = [build_embedding_text(s) for s in train_data]
    print(f"Sample text[0]: {texts[0][:120]!r}")

    print(f"\nLoading SentenceTransformer: {MODEL_NAME} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding ...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Embedding shape: {embeddings.shape}")

    os.makedirs(os.path.dirname(EMBED_PATH), exist_ok=True)
    np.save(EMBED_PATH, embeddings)
    print(f"\nSaved embeddings → {EMBED_PATH}")


if __name__ == "__main__":
    main()
