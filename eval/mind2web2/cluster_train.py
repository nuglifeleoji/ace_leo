#!/usr/bin/env python3
"""
Semantic Clustering for Mind2Web2 Training Subset Selection.

Uses pre-computed embeddings (from embed_train.py) to select semantically
representative training subsets via K-means clustering.

Prerequisite:
    python -m eval.mind2web2.embed_train    # generates embeddings.npy

Usage:
    # Default: K = 5, 10, 15, 20, 30, 50, 80
    python -m eval.mind2web2.cluster_train

    # Custom cluster sizes
    python -m eval.mind2web2.cluster_train --clusters 10 20 50

    # Also generate t-SNE visualization
    python -m eval.mind2web2.cluster_train --visualize
"""
import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

# ── Config ──────────────────────────────────────────────────────

TRAIN_PATH = "./eval/mind2web2/data/mind2web2_train.jsonl"
EMBEDDING_PATH = "./eval/mind2web2/data/embeddings.npy"
OUTPUT_DIR = "./eval/mind2web2/data"
CONFIG_PATH = "./eval/mind2web2/data/sample_config.json"
DEFAULT_CLUSTERS = [5, 10, 15, 20, 30, 50, 80]


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
    print(f"    Saved {len(data)} samples → {path}")


# ── Clustering ──────────────────────────────────────────────────

def cluster_and_select(
    embeddings: np.ndarray,
    train_data: List[Dict],
    k: int,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray]:
    """
    Run K-means and select the sample closest to each centroid.

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
        best = indices[np.argmin(dists)]
        selected_indices.append(int(best))

    return selected_indices, labels


def report_selection(
    train_data: List[Dict],
    selected_indices: List[int],
    labels: np.ndarray,
    k: int,
):
    """Print statistics about the selected subset."""
    cluster_sizes = np.bincount(labels)
    print(f"    Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}, median={np.median(cluster_sizes):.1f}")

    domains = Counter(train_data[i].get("domain", "?") for i in selected_indices)
    ops = Counter(
        str(train_data[i].get("operation", {}).get("op", "?"))
        for i in selected_indices
    )
    websites = set(train_data[i].get("website", "?") for i in selected_indices)

    print(f"    Domains: {dict(domains)}")
    print(f"    Operations: {dict(ops)}")
    print(f"    Unique websites: {len(websites)}/{k}")

    # Compare with full training set distribution
    all_domains = Counter(d.get("domain", "?") for d in train_data)
    print(f"    (Full train domains: {dict(all_domains)})")


# ── Config Update ───────────────────────────────────────────────

def update_config(cluster_sizes: List[int]):
    """Add cluster configs to sample_config.json."""
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    for k in cluster_sizes:
        config[f"mind2web2_cluster{k}"] = {
            "train_data": f"./eval/mind2web2/data/mind2web2_train_cluster{k}.jsonl",
            "val_data": "./eval/mind2web2/data/mind2web2_val.jsonl",
            "test_data": "./eval/mind2web2/data/mind2web2_test.jsonl",
        }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Visualization ───────────────────────────────────────────────

def visualize(
    embeddings: np.ndarray,
    train_data: List[Dict],
    cluster_sizes: List[int],
):
    """Generate t-SNE visualization with cluster assignments and selected reps."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping visualization (install matplotlib and scikit-learn)")
        return

    print("\nGenerating t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    n_plots = len(cluster_sizes)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, k in zip(axes, cluster_sizes):
        selected, labels = cluster_and_select(embeddings, train_data, k)

        ax.scatter(coords[:, 0], coords[:, 1],
                   c=labels, alpha=0.2, s=3, cmap="tab20")
        ax.scatter(coords[selected, 0], coords[selected, 1],
                   c="red", s=40, marker="x", linewidths=1.5,
                   label=f"{k} reps", zorder=5)
        ax.set_title(f"K={k}")
        ax.legend(fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Mind2Web2 Training Data — Semantic Clusters", fontsize=14)
    plt.tight_layout()

    viz_path = os.path.join(OUTPUT_DIR, "cluster_visualization.png")
    plt.savefig(viz_path, dpi=150)
    print(f"Saved visualization → {viz_path}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cluster Mind2Web2 training data and select representative subsets"
    )
    parser.add_argument(
        "--clusters", type=int, nargs="+", default=DEFAULT_CLUSTERS,
        help=f"Cluster sizes to generate (default: {DEFAULT_CLUSTERS})"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate t-SNE visualization"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for K-means (default: 42)"
    )
    args = parser.parse_args()

    # Load embeddings
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDING_PATH}.\n"
            f"Run: python -m eval.mind2web2.embed_train"
        )
    embeddings = np.load(EMBEDDING_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load training data
    train_data = load_jsonl(TRAIN_PATH)
    assert len(train_data) == embeddings.shape[0], \
        f"Mismatch: {len(train_data)} samples vs {embeddings.shape[0]} embeddings"
    print(f"Loaded {len(train_data)} training samples")

    # Run clustering for each K
    for k in args.clusters:
        print(f"\n{'='*55}")
        print(f"  K={k}: Selecting {k} representative training samples")
        print(f"{'='*55}")

        selected, labels = cluster_and_select(embeddings, train_data, k, seed=args.seed)
        report_selection(train_data, selected, labels, k)

        # Save subset
        subset = [train_data[i] for i in selected]
        out_path = os.path.join(OUTPUT_DIR, f"mind2web2_train_cluster{k}.jsonl")
        save_jsonl(subset, out_path)

        # Save metadata
        meta = {
            "k": k,
            "seed": args.seed,
            "selected_indices": selected,
            "n_total": len(train_data),
        }
        meta_path = os.path.join(OUTPUT_DIR, f"cluster{k}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

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
    for k in args.clusters:
        print(f"    python -m eval.mind2web2.run \\")
        print(f"      --task_name mind2web2_cluster{k} \\")
        print(f"      --mode offline --eval_steps {k} \\")
        print(f"      --save_path results/mind2web2_cluster{k}")
        print()


if __name__ == "__main__":
    main()
