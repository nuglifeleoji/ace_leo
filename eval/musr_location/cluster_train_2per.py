#!/usr/bin/env python3
"""
2-per-Cluster Selection for MuSR Location Training Subset.

Instead of selecting 1 sample per cluster (closest to centroid), this selects
the 2 samples closest to each centroid. This tests the hypothesis that
having more within-cluster samples improves learning while maintaining
the same cluster coverage.

Prerequisite:
    python -m eval.musr_location.embed_train    # generates embeddings.npy

Usage:
    # Default: K = 5, 10, 15, 20 → selects 10, 20, 30, 40 samples
    python -m eval.musr_location.cluster_train_2per

    # Custom cluster sizes
    python -m eval.musr_location.cluster_train_2per --clusters 5 10
"""
import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

# ── Config ──────────────────────────────────────────────────────

TRAIN_PATH = "./eval/musr_location/data/location_train.jsonl"
EMBEDDING_PATH = "./eval/musr_location/data/embeddings.npy"
OUTPUT_DIR = "./eval/musr_location/data"
CONFIG_PATH = "./eval/musr_location/data/sample_config.json"
DEFAULT_CLUSTERS = [5, 10, 15, 20]
SAMPLES_PER_CLUSTER = 2


# ── Data Loading ────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"    Saved {len(data)} samples -> {path}")


# ── Clustering (2-per-cluster) ─────────────────────────────────

def cluster_and_select_n(
    embeddings: np.ndarray,
    train_data: List[Dict],
    k: int,
    n_per_cluster: int = 2,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray]:
    """
    Run K-means and select the N samples closest to each centroid.

    Returns:
        (selected_indices, labels) — indices into train_data, cluster assignments
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    selected_indices = []
    for cid in range(k):
        mask = labels == cid
        indices = np.where(mask)[0]
        dists = np.linalg.norm(embeddings[indices] - centroids[cid], axis=1)
        # Sort by distance and take top-N (or all if cluster is smaller)
        n_select = min(n_per_cluster, len(indices))
        top_n = indices[np.argsort(dists)[:n_select]]
        selected_indices.extend(int(idx) for idx in top_n)

    return selected_indices, labels


def report_selection(
    train_data: List[Dict],
    selected_indices: List[int],
    labels: np.ndarray,
    k: int,
    n_per_cluster: int,
):
    """Print statistics about the selected subset."""
    cluster_sizes = np.bincount(labels)
    print(f"    Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}, median={np.median(cluster_sizes):.1f}")
    print(f"    Selected: {len(selected_indices)} samples "
          f"({k} clusters x {n_per_cluster} per cluster)")

    # Answer distribution in selected subset
    answers = Counter(train_data[i].get("target", "?") for i in selected_indices)
    n_choices = Counter(train_data[i].get("n_choices", 4) for i in selected_indices)

    print(f"    Selected answers: {dict(answers)}")
    print(f"    N choices: {dict(n_choices)}")

    # Compare with full training set distribution
    all_answers = Counter(d.get("target", "?") for d in train_data)
    print(f"    (Full train answers: {dict(all_answers)})")


# ── Config Update ───────────────────────────────────────────────

def update_config(cluster_sizes: List[int], n_per_cluster: int):
    """Add 2-per-cluster configs to sample_config.json."""
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    for k in cluster_sizes:
        n_total = k * n_per_cluster
        task_name = f"musr_location_cluster{n_per_cluster}x{k}"
        config[task_name] = {
            "train_data": f"./eval/musr_location/data/location_train_cluster{n_per_cluster}x{k}.jsonl",
            "val_data": "./eval/musr_location/data/location_val.jsonl",
            "test_data": "./eval/musr_location/data/location_test.jsonl",
        }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cluster MuSR Location training data and select 2 samples per cluster"
    )
    parser.add_argument(
        "--clusters", type=int, nargs="+", default=DEFAULT_CLUSTERS,
        help=f"Cluster sizes to generate (default: {DEFAULT_CLUSTERS})"
    )
    parser.add_argument(
        "--n_per_cluster", type=int, default=SAMPLES_PER_CLUSTER,
        help=f"Number of samples per cluster (default: {SAMPLES_PER_CLUSTER})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for K-means (default: 42)"
    )
    args = parser.parse_args()

    n_per = args.n_per_cluster

    # Load embeddings
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDING_PATH}.\n"
            f"Run: python -m eval.musr_location.embed_train"
        )
    embeddings = np.load(EMBEDDING_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load training data
    train_data = load_jsonl(TRAIN_PATH)
    assert len(train_data) == embeddings.shape[0], \
        f"Mismatch: {len(train_data)} samples vs {embeddings.shape[0]} embeddings"
    print(f"Loaded {len(train_data)} training samples")
    print(f"Selection strategy: {n_per} samples per cluster")

    # Run clustering for each K
    for k in args.clusters:
        n_total = k * n_per
        if n_total > len(train_data):
            print(f"\n  [SKIP] K={k} x {n_per} = {n_total} > {len(train_data)} training samples")
            continue

        print(f"\n{'='*60}")
        print(f"  K={k}, {n_per}-per-cluster: Selecting {n_total} training samples")
        print(f"{'='*60}")

        selected, labels = cluster_and_select_n(
            embeddings, train_data, k, n_per_cluster=n_per, seed=args.seed
        )
        report_selection(train_data, selected, labels, k, n_per)

        # Save subset
        subset = [train_data[i] for i in selected]
        out_path = os.path.join(OUTPUT_DIR, f"location_train_cluster{n_per}x{k}.jsonl")
        save_jsonl(subset, out_path)

        # Save metadata
        meta = {
            "k": k,
            "n_per_cluster": n_per,
            "n_selected": len(selected),
            "seed": args.seed,
            "selected_indices": selected,
            "n_total": len(train_data),
        }
        meta_path = os.path.join(OUTPUT_DIR, f"cluster{n_per}x{k}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Update config
    update_config(args.clusters, n_per)

    # Print next steps
    print(f"\n{'='*60}")
    print(f"  DONE — Generated {len(args.clusters)} training subsets")
    print(f"{'='*60}")
    print(f"\n  Next: Run ACE training for each:\n")
    for k in args.clusters:
        n_total = k * n_per
        print(f"    python -m eval.musr_location.run \\")
        print(f"      --task_name musr_location_cluster{n_per}x{k} \\")
        print(f"      --mode offline --eval_steps {n_total} \\")
        print(f"      --save_path results/musr_location_cluster{n_per}x{k}")
        print()


if __name__ == "__main__":
    main()
