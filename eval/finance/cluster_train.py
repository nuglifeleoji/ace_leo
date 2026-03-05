#!/usr/bin/env python3
"""
K-means clustering for FiNer training subset selection.

Uses pre-computed embeddings (from embed_train.py) to select semantically
representative training subsets via K-means clustering.  One sample closest
to each cluster centroid is selected as the "representative".

Hypothesis: A small, diverse subset of FiNer training samples (covering
distinct XBRL tagging patterns / financial domains) can provide ACE with
sufficiently varied examples to build an effective playbook.

Prerequisite:
    python -m eval.finance.embed_train    # generates finer_embeddings.npy

Usage:
    # Default: K = 5, 10, 20, 30, 40, 50, 80
    python -m eval.finance.cluster_train

    # Custom sizes
    python -m eval.finance.cluster_train --clusters 10 20 50

    # With t-SNE visualization
    python -m eval.finance.cluster_train --visualize
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_PATH   = "./eval/finance/data/finer_train_batched_1000_samples.jsonl"
EMBED_PATH   = "./eval/finance/data/finer_embeddings.npy"
OUTPUT_DIR   = "./eval/finance/data"
CONFIG_PATH  = "./eval/finance/data/sample_config.json"
VAL_PATH     = "./eval/finance/data/finer_val_batched_500_samples.jsonl"
TEST_PATH    = "./eval/finance/data/finer_test_subset_006_seed42.jsonl"

DEFAULT_CLUSTERS = [5, 10, 20, 30, 40, 50, 80]


# ── I/O Helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
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


# ── Clustering ───────────────────────────────────────────────────────────────

def cluster_and_select(
    embeddings: np.ndarray,
    train_data: List[Dict],
    k: int,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray]:
    """
    Run K-means and return the index of the sample closest to each centroid.

    Returns:
        (selected_indices, labels)
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_

    selected_indices: List[int] = []
    for cid in range(k):
        mask  = labels == cid
        idxs  = np.where(mask)[0]
        dists = np.linalg.norm(embeddings[idxs] - centroids[cid], axis=1)
        best  = int(idxs[np.argmin(dists)])
        selected_indices.append(best)

    return selected_indices, labels


def report_selection(
    train_data: List[Dict],
    selected_indices: List[int],
    labels: np.ndarray,
    k: int,
) -> None:
    """Print statistics about the selected subset."""
    cluster_sizes = np.bincount(labels)
    print(f"    Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}, median={np.median(cluster_sizes):.1f}")

    # Analyse the XBRL tag distribution of selected samples
    all_tags: List[str] = []
    for idx in selected_indices:
        target = train_data[idx].get("target", "")
        for tag in target.split(","):
            tag = tag.strip()
            if tag:
                all_tags.append(tag)

    tag_counter = Counter(all_tags)
    top_tags = tag_counter.most_common(10)
    print(f"    Top-10 XBRL tags in selected subset: {top_tags}")
    print(f"    Unique XBRL tags in subset          : {len(tag_counter)}")

    # Compare with full training distribution
    all_train_tags: List[str] = []
    for item in train_data:
        for tag in item.get("target", "").split(","):
            tag = tag.strip()
            if tag:
                all_train_tags.append(tag)
    print(f"    Unique XBRL tags in full train      : {len(Counter(all_train_tags))}")


# ── Config Update ─────────────────────────────────────────────────────────────

def update_config(cluster_sizes: List[int]) -> None:
    """Add finer_cluster<K> entries to sample_config.json."""
    config: Dict = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)

    for k in cluster_sizes:
        config[f"finer_cluster{k}"] = {
            "train_data": f"./eval/finance/data/finer_train_cluster{k}.jsonl",
            "val_data":   VAL_PATH,
            "test_data":  TEST_PATH,
        }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(
    embeddings: np.ndarray,
    train_data: List[Dict],
    cluster_sizes: List[int],
) -> None:
    """Generate t-SNE visualization (saved to finer_cluster_visualization.png)."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping visualization (install matplotlib + scikit-learn)")
        return

    print("\nGenerating t-SNE (may take a minute) ...")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    n = len(cluster_sizes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, cluster_sizes):
        selected, labels = cluster_and_select(embeddings, train_data, k)
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=labels, alpha=0.2, s=3, cmap="tab20")
        ax.scatter(coords[selected, 0], coords[selected, 1],
                   c="red", s=50, marker="x", linewidths=1.5,
                   label=f"{k} reps", zorder=5)
        ax.set_title(f"K={k}")
        ax.legend(fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("FiNer Training Data — Semantic Clusters (XBRL tagging)", fontsize=13)
    plt.tight_layout()

    viz_path = os.path.join(OUTPUT_DIR, "finer_cluster_visualization.png")
    plt.savefig(viz_path, dpi=150)
    print(f"Saved visualization → {viz_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster FiNer training data and select representative subsets"
    )
    parser.add_argument("--clusters", type=int, nargs="+",
                        default=DEFAULT_CLUSTERS,
                        help=f"Cluster sizes (default: {DEFAULT_CLUSTERS})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  FiNer — K-means Cluster Subset Selection")
    print("=" * 60)

    # Load embeddings
    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(
            f"Embeddings not found: {EMBED_PATH}\n"
            "Run first:  python -m eval.finance.embed_train"
        )
    embeddings = np.load(EMBED_PATH)
    print(f"\nLoaded embeddings: {embeddings.shape}")

    # Load training data
    train_data = load_jsonl(TRAIN_PATH)
    assert len(train_data) == embeddings.shape[0], (
        f"Mismatch: {len(train_data)} samples vs {embeddings.shape[0]} embeddings"
    )
    print(f"Loaded {len(train_data)} training samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run clustering for each K
    for k in sorted(args.clusters):
        print(f"\n{'='*55}")
        print(f"  K={k}: Selecting {k} representative training samples")
        print(f"{'='*55}")

        selected_indices, labels = cluster_and_select(
            embeddings, train_data, k, seed=args.seed
        )
        report_selection(train_data, selected_indices, labels, k)

        # Save subset
        subset   = [train_data[i] for i in selected_indices]
        out_path = os.path.join(OUTPUT_DIR, f"finer_train_cluster{k}.jsonl")
        save_jsonl(subset, out_path)

        # Save metadata
        meta_path = os.path.join(OUTPUT_DIR, f"finer_cluster{k}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "k":               k,
                "seed":            args.seed,
                "selected_indices": selected_indices,
                "n_total":         len(train_data),
            }, f, indent=2)

    # Update config
    update_config(args.clusters)

    # Visualization
    if args.visualize:
        visualize(embeddings, train_data, args.clusters)

    # Print next steps
    print(f"\n{'='*60}")
    print(f"  DONE — Generated {len(args.clusters)} training subsets")
    print(f"{'='*60}")
    print(f"\n  Next: Run ACE training for each cluster size:\n")
    for k in sorted(args.clusters):
        print(f"    python -m eval.finance.run \\")
        print(f"      --task_name finer_cluster{k} \\")
        print(f"      --mode offline --eval_steps {k} \\")
        print(f"      --save_path results/finer_cluster{k}")
        print()


if __name__ == "__main__":
    main()
