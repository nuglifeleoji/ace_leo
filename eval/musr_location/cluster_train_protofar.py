#!/usr/bin/env python3
"""
K-means + Prototype & Mid-far (Quantile) Selection for MuSR Location.

Goal
----
Select 2 samples per cluster:
  - 1 prototype sample closest to the cluster centroid (representative)
  - 1 "mid-far" sample inside the same cluster at a distance quantile
    (contrastive / expands coverage without picking extreme outliers)

This aims to improve upon pure "2 nearest to centroid" by adding a boundary-ish
example per cluster while keeping samples reasonably on-distribution.

Prerequisite
------------
    python -m eval.musr_location.embed_train    # generates embeddings.npy

Usage
-----
    # Default: K = 5, 10, 15, 20, 30
    python -m eval.musr_location.cluster_train_protofar

    # Custom cluster sizes + quantile
    python -m eval.musr_location.cluster_train_protofar --clusters 20 30 --far_quantile 0.85
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# ── Config ──────────────────────────────────────────────────────

TRAIN_PATH = "./eval/musr_location/data/location_train.jsonl"
EMBEDDING_PATH = "./eval/musr_location/data/embeddings.npy"
OUTPUT_DIR = "./eval/musr_location/data"
CONFIG_PATH = "./eval/musr_location/data/sample_config.json"
DEFAULT_CLUSTERS = [5, 10, 15, 20, 30]


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


# ── Selection ───────────────────────────────────────────────────

def cluster_and_select_protofar(
    embeddings: np.ndarray,
    k: int,
    far_quantile: float = 0.85,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Run K-means and select 2 samples per cluster:
      - nearest to centroid
      - within-cluster distance at `far_quantile` (mid-far)

    Returns:
        (selected_indices, labels, dists_to_centroid)
    """
    from sklearn.cluster import KMeans

    if not (0.0 < far_quantile < 1.0):
        raise ValueError("--far_quantile must be in (0, 1)")

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Precompute distances to centroids for all points
    dists = np.linalg.norm(embeddings - centroids[labels], axis=1)

    selected: List[int] = []
    selected_set = set()

    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue

        # Prototype: nearest
        proto = int(idx[np.argmin(dists[idx])])
        selected.append(proto)
        selected_set.add(proto)

        # Mid-far: pick element at quantile of distance distribution
        if idx.size >= 2:
            # sort cluster members by dist asc
            order = idx[np.argsort(dists[idx])]
            qpos = int(round(far_quantile * (len(order) - 1)))
            qpos = max(0, min(qpos, len(order) - 1))
            far = int(order[qpos])

            # Ensure distinct; if collided, pick nearest to that quantile position around it.
            if far in selected_set:
                # try neighbors around qpos
                found = None
                for delta in range(1, len(order)):
                    for j in (qpos - delta, qpos + delta):
                        if 0 <= j < len(order) and int(order[j]) not in selected_set:
                            found = int(order[j])
                            break
                    if found is not None:
                        break
                if found is not None:
                    far = found
                else:
                    far = None

            if far is not None:
                selected.append(far)
                selected_set.add(far)

    return selected, labels, dists


def report_selection(
    train_data: List[Dict],
    selected_indices: List[int],
    labels: np.ndarray,
    dists: np.ndarray,
    k: int,
    far_quantile: float,
) -> None:
    cluster_sizes = np.bincount(labels, minlength=k)
    print(
        f"    Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
        f"mean={cluster_sizes.mean():.1f}, median={np.median(cluster_sizes):.1f}"
    )
    print(f"    Selected: {len(selected_indices)} samples (~2×{k})")
    print(f"    far_quantile: {far_quantile}")

    # Answer distribution
    answers = Counter(train_data[i].get("target", "?") for i in selected_indices)
    n_choices = Counter(train_data[i].get("n_choices", 4) for i in selected_indices)
    print(f"    Selected answers: {dict(answers)}")
    print(f"    N choices: {dict(n_choices)}")

    # Distance sanity
    sel_d = dists[np.array(selected_indices, dtype=int)]
    print(
        f"    Dist-to-centroid (selected): min={sel_d.min():.4f}, "
        f"median={np.median(sel_d):.4f}, mean={sel_d.mean():.4f}, max={sel_d.max():.4f}"
    )


# ── Config update ───────────────────────────────────────────────

def update_config(cluster_sizes: List[int]) -> None:
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

    for k in cluster_sizes:
        task = f"musr_location_protofar2x{k}"
        config[task] = {
            "train_data": f"./eval/musr_location/data/location_train_protofar2x{k}.jsonl",
            "val_data": "./eval/musr_location/data/location_val.jsonl",
            "test_data": "./eval/musr_location/data/location_test.jsonl",
        }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Main ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuSR Location: K-means + prototype & mid-far (quantile) selection"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=DEFAULT_CLUSTERS,
        help=f"Cluster sizes to generate (default: {DEFAULT_CLUSTERS})",
    )
    parser.add_argument(
        "--far_quantile",
        type=float,
        default=0.85,
        help="Within-cluster distance quantile for the mid-far point (default: 0.85)",
    )
    parser.add_argument("--seed", type=int, default=42, help="K-means seed (default: 42)")
    args = parser.parse_args()

    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDING_PATH}.\n"
            f"Run: python -m eval.musr_location.embed_train"
        )

    embeddings = np.load(EMBEDDING_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")

    train_data = load_jsonl(TRAIN_PATH)
    if len(train_data) != embeddings.shape[0]:
        raise ValueError(f"Mismatch: {len(train_data)} samples vs {embeddings.shape[0]} embeddings")
    print(f"Loaded {len(train_data)} training samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for k in args.clusters:
        if k * 2 > len(train_data):
            print(f"\n  [SKIP] K={k} (2×K={2*k}) > {len(train_data)} training samples")
            continue

        print(f"\n{'='*60}")
        print(f"  K={k}: selecting prototype + mid-far per cluster (≈ {2*k} samples)")
        print(f"{'='*60}")

        selected, labels, dists = cluster_and_select_protofar(
            embeddings=embeddings,
            k=k,
            far_quantile=args.far_quantile,
            seed=args.seed,
        )
        report_selection(train_data, selected, labels, dists, k=k, far_quantile=args.far_quantile)

        subset = [train_data[i] for i in selected]
        out_path = os.path.join(OUTPUT_DIR, f"location_train_protofar2x{k}.jsonl")
        save_jsonl(subset, out_path)

        meta = {
            "method": "kmeans_protofar",
            "k": k,
            "seed": args.seed,
            "far_quantile": args.far_quantile,
            "selected_indices": selected,
            "n_total": len(train_data),
        }
        meta_path = os.path.join(OUTPUT_DIR, f"protofar2x{k}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    update_config(args.clusters)

    print(f"\n{'='*60}")
    print("  DONE — Generated proto+mid-far subsets")
    print(f"{'='*60}")
    print("\n  Next: Run ACE training:\n")
    for k in args.clusters:
        print("    python -m eval.musr_location.run \\")
        print(f"      --task_name musr_location_protofar2x{k} \\")
        print("      --mode offline \\")
        print(f"      --eval_steps {2*k} \\")
        print("      --skip_initial_test \\")
        print(f"      --save_path results/musr_location_protofar2x{k}")
        print()


if __name__ == "__main__":
    main()

