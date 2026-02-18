#!/usr/bin/env python3
"""
Kernel Herding (prototype / representative) subset selection for Mind2Web.

Why herding?
------------
For ACE-style learning, selecting *representative* samples (prototypes) can be
more helpful than maximizing diversity (e.g. DPP). Kernel herding selects a
subset whose empirical mean in feature space matches the full-data mean.

This implementation uses linear-kernel herding over normalized embeddings:
    k(x, y) = x · y   (cosine similarity after L2-normalization)

Prerequisite
------------
    python -m eval.mind2web.embed_train    # generates embeddings.npy

Usage
-----
    # Default subset sizes: 10, 15, 20, 30, 50, 80
    python -m eval.mind2web.herding_select

    # Custom sizes
    python -m eval.mind2web.herding_select --sizes 10 20 50
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np

# ── Config ──────────────────────────────────────────────────────

TRAIN_PATH = "./eval/mind2web/data/mind2web_train.jsonl"
EMBEDDING_PATH = "./eval/mind2web/data/embeddings.npy"
OUTPUT_DIR = "./eval/mind2web/data"
CONFIG_PATH = "./eval/mind2web/data/sample_config.json"

DEFAULT_SIZES = [10, 15, 20, 30, 50, 80]


# ── IO ──────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
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
    print(f"    Saved {len(data)} samples → {path}")


# ── Herding ─────────────────────────────────────────────────────

def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def linear_kernel_herding(embeddings: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """
    Linear kernel herding (mean matching) greedy selection.

    Step t chooses:
        i* = argmax_i  x_i · (m - sum_S/(t+1))
    where m is the full-data mean embedding and sum_S is the sum of selected
    embeddings (after L2-normalization).
    """
    X = normalize_rows(embeddings.astype(np.float32, copy=False))
    N, _ = X.shape
    if k > N:
        raise ValueError(f"k={k} > N={N}")

    rng = np.random.default_rng(seed)
    order = np.arange(N)
    rng.shuffle(order)  # tie-breaker

    m = X.mean(axis=0)  # (d,)
    selected: List[int] = []
    selected_mask = np.zeros(N, dtype=bool)
    selected_sum = np.zeros_like(m)

    # Precompute X @ m for step 0
    Xm = X @ m  # (N,)

    for t in range(k):
        if t == 0:
            scores = Xm.copy()
        else:
            residual = m - (selected_sum / (t + 1))
            scores = X @ residual

        scores[selected_mask] = -np.inf
        best = order[np.argmax(scores[order])]
        selected.append(int(best))
        selected_mask[best] = True
        selected_sum += X[best]

        if (t + 1) % 10 == 0 or (t + 1) == k:
            approx_mean = selected_sum / (t + 1)
            err = float(np.linalg.norm(m - approx_mean))
            print(f"      Step {t+1}/{k}: picked {best}, mean-error ||m-μ_S||={err:.6f}")

    return selected


def report_selection(train_data: List[Dict], selected_indices: List[int], embeddings: np.ndarray, k: int) -> None:
    """Print basic distribution stats for the selected subset."""
    domains = Counter(train_data[i].get("domain", "?") for i in selected_indices)
    ops = Counter(str(train_data[i].get("operation", {}).get("op", "?")) for i in selected_indices)
    websites = set(train_data[i].get("website", "?") for i in selected_indices)
    print(f"    Domains: {dict(domains)}")
    print(f"    Operations: {dict(ops)}")
    print(f"    Unique websites: {len(websites)}/{k}")

    # Similarity stats (sanity)
    X = normalize_rows(embeddings.astype(np.float32, copy=False))
    sel = np.array(selected_indices, dtype=int)
    sims = (X[sel] @ X[sel].T)
    upper = sims[np.triu_indices_from(sims, k=1)]
    if upper.size:
        print(
            f"    Pairwise cosine-sim: mean={upper.mean():.4f}, "
            f"min={upper.min():.4f}, max={upper.max():.4f}"
        )


# ── Config update ───────────────────────────────────────────────

def update_config(sizes: List[int]) -> None:
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

    for k in sizes:
        config[f"mind2web_herd{k}"] = {
            "train_data": f"./eval/mind2web/data/mind2web_train_herd{k}.jsonl",
            "val_data": "./eval/mind2web/data/mind2web_val.jsonl",
            "test_data": "./eval/mind2web/data/mind2web_test.jsonl",
        }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Main ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kernel herding (prototype) subset selection for Mind2Web"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help=f"Subset sizes to generate (default: {DEFAULT_SIZES})",
    )
    parser.add_argument("--seed", type=int, default=0, help="Tie-break seed (default: 0)")
    args = parser.parse_args()

    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDING_PATH}.\n"
            f"Run: python -m eval.mind2web.embed_train"
        )

    embeddings = np.load(EMBEDDING_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")

    train_data = load_jsonl(TRAIN_PATH)
    if len(train_data) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(train_data)} samples vs {embeddings.shape[0]} embeddings"
        )
    print(f"Loaded {len(train_data)} training samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for k in args.sizes:
        if k > len(train_data):
            print(f"\n  [SKIP] k={k} > {len(train_data)} training samples")
            continue

        print(f"\n{'='*60}")
        print(f"  Herding: Selecting {k} representative training samples")
        print(f"{'='*60}")

        selected = linear_kernel_herding(embeddings, k=k, seed=args.seed)
        report_selection(train_data, selected, embeddings, k=k)

        subset = [train_data[i] for i in selected]
        out_path = os.path.join(OUTPUT_DIR, f"mind2web_train_herd{k}.jsonl")
        save_jsonl(subset, out_path)

        meta = {
            "method": "linear_kernel_herding",
            "k": k,
            "seed": args.seed,
            "selected_indices": selected,
            "n_total": len(train_data),
            "kernel": "linear/cosine(normalized)",
        }
        meta_path = os.path.join(OUTPUT_DIR, f"herd{k}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    update_config(args.sizes)

    print(f"\n{'='*60}")
    print("  DONE — Generated herding subsets")
    print(f"{'='*60}")
    print("\n  Next: Run ACE training for each subset:\n")
    for k in args.sizes:
        print("    python -m eval.mind2web.run \\")
        print(f"      --task_name mind2web_herd{k} \\")
        print(f"      --mode offline --eval_steps {k} \\")
        print(f"      --skip_initial_test \\")
        print(f"      --save_path results/mind2web_herd{k}")
        print()


if __name__ == "__main__":
    main()

