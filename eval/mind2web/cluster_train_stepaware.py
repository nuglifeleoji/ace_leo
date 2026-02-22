#!/usr/bin/env python3
"""
Step-Position-Aware Clustering for Mind2Web Training Subset Selection.

Problem with vanilla K-means on embeddings:
    Early-step samples (step 0, 1, 2) have very similar embeddings
    (they all look like "initial homepage navigation"), so cluster
    centroids crowd into this dense region. As a result, ~90% of
    selected samples are early steps — the playbook never learns
    mid-task or late-task interaction patterns.

Solution:
    Augment each embedding with a normalized step-position feature
    before running K-means. This pulls centroids to spread across
    the full task lifecycle (early → mid → late), while still
    capturing semantic diversity.

    augmented[i] = concat(norm(embedding[i]), pos_weight * pos[i])
    where pos[i] = step_idx / (total_steps - 1)  ∈ [0, 1]

Usage:
    # Generate step-aware clusters for k=10,15,20
    python -m eval.mind2web.cluster_train_stepaware --clusters 10 15 20

    # Tune position weight (0=pure semantic, 1=heavy position influence)
    python -m eval.mind2web.cluster_train_stepaware --clusters 20 --pos_weight 0.5
"""
import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

# ── Config ──────────────────────────────────────────────────────

TRAIN_PATH = "./eval/mind2web/data/mind2web_train.jsonl"
EMBEDDING_PATH = "./eval/mind2web/data/embeddings.npy"
OUTPUT_DIR = "./eval/mind2web/data"
CONFIG_PATH = "./eval/mind2web/data/sample_config.json"
DEFAULT_CLUSTERS = [10, 15, 20]
DEFAULT_POS_WEIGHT = 0.5


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


# ── Feature Engineering ─────────────────────────────────────────

def build_position_features(train_data: List[Dict]) -> np.ndarray:
    """
    Compute normalized step position for each sample.
    pos = step_idx / max(total_steps - 1, 1)  ∈ [0, 1]
    """
    positions = []
    for d in train_data:
        step = d.get("step_idx", 0)
        total = d.get("total_steps", 1)
        pos = step / max(total - 1, 1)
        positions.append(pos)
    return np.array(positions, dtype=np.float32).reshape(-1, 1)


def build_augmented_embeddings(
    embeddings: np.ndarray,
    train_data: List[Dict],
    pos_weight: float,
) -> np.ndarray:
    """
    Augment embeddings with scaled step position feature.

    The embeddings are L2-normalized first so all semantic dimensions
    are on the same scale. The position feature is then added with
    a controllable weight.

    Args:
        embeddings: (N, D) raw embeddings
        train_data: list of N training samples
        pos_weight: how much to weight the position feature
                    (0 = pure semantic, 1 = equal to full norm embedding)

    Returns:
        (N, D+1) augmented embeddings
    """
    # L2-normalize embeddings so they lie on unit hypersphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
    emb_normed = embeddings / norms

    # Normalized step position
    pos_feat = build_position_features(train_data)  # (N, 1), range [0, 1]

    # Scale position feature by pos_weight
    pos_feat_scaled = pos_feat * pos_weight

    # Concatenate
    augmented = np.concatenate([emb_normed, pos_feat_scaled], axis=1)
    return augmented


# ── Clustering ──────────────────────────────────────────────────

def cluster_and_select(
    augmented_embeddings: np.ndarray,
    original_embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray]:
    """
    Run K-means on augmented embeddings, then select the sample
    closest to each centroid in the ORIGINAL embedding space
    (so the selected sample is the most semantically representative
    within its position-aware cluster).

    Returns:
        (selected_indices, labels)
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(augmented_embeddings)
    centroids_aug = kmeans.cluster_centers_

    selected_indices = []
    for cid in range(k):
        mask = labels == cid
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        # Distance in augmented space (consistent with how clusters were formed)
        dists = np.linalg.norm(augmented_embeddings[indices] - centroids_aug[cid], axis=1)
        best = indices[np.argmin(dists)]
        selected_indices.append(int(best))

    return selected_indices, labels


# ── Reporting ───────────────────────────────────────────────────

def report_selection(
    train_data: List[Dict],
    selected_indices: List[int],
    labels: np.ndarray,
    k: int,
    pos_weight: float,
):
    """Print statistics including step position distribution."""
    cluster_sizes = np.bincount(labels)
    print(f"    Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}")

    domains = Counter(train_data[i].get("domain", "?") for i in selected_indices)
    ops = Counter(
        str(train_data[i].get("operation", {}).get("op", "?"))
        for i in selected_indices
    )
    websites = set(train_data[i].get("website", "?") for i in selected_indices)

    print(f"    Domains: {dict(domains)}")
    print(f"    Operations: {dict(ops)}")
    print(f"    Unique websites: {len(websites)}/{k}")

    # Step position analysis (the key metric we're improving)
    positions = []
    for i in selected_indices:
        d = train_data[i]
        step = d.get("step_idx", 0)
        total = d.get("total_steps", 1)
        positions.append(step / max(total - 1, 1))

    avg_pos = np.mean(positions)
    early = sum(1 for p in positions if p < 0.33)
    mid = sum(1 for p in positions if 0.33 <= p < 0.67)
    late = sum(1 for p in positions if p >= 0.67)
    print(f"    Step positions: avg={avg_pos:.2f}, early={early}, mid={mid}, late={late}")
    print(f"    (pos_weight={pos_weight}; vanilla cluster20 was: avg=0.11, early=18, mid=1, late=1)")


# ── Config Update ───────────────────────────────────────────────

def update_config(cluster_sizes: List[int], pos_weight: float):
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    tag = f"_stepaware{str(pos_weight).replace('.', '')}"
    for k in cluster_sizes:
        name = f"mind2web_cluster{k}{tag}"
        config[name] = {
            "train_data": f"./eval/mind2web/data/mind2web_train_cluster{k}{tag}.jsonl",
            "val_data": "./eval/mind2web/data/mind2web_val.jsonl",
            "test_data": "./eval/mind2web/data/mind2web_test.jsonl",
        }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step-position-aware cluster selection for Mind2Web"
    )
    parser.add_argument(
        "--clusters", type=int, nargs="+", default=DEFAULT_CLUSTERS,
        help=f"Cluster sizes to generate (default: {DEFAULT_CLUSTERS})"
    )
    parser.add_argument(
        "--pos_weight", type=float, default=DEFAULT_POS_WEIGHT,
        help=f"Weight for step-position feature (default: {DEFAULT_POS_WEIGHT}). "
             "0 = pure semantic (same as vanilla), 1 = position as strong as embedding norm."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for K-means (default: 42)"
    )
    args = parser.parse_args()

    # Load
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDING_PATH}.\n"
            f"Run: python -m eval.mind2web.embed_train"
        )
    embeddings = np.load(EMBEDDING_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")

    train_data = load_jsonl(TRAIN_PATH)
    assert len(train_data) == embeddings.shape[0], \
        f"Mismatch: {len(train_data)} samples vs {embeddings.shape[0]} embeddings"
    print(f"Loaded {len(train_data)} training samples")

    # Build augmented embeddings (once, reuse for all k)
    print(f"\nBuilding position-augmented embeddings (pos_weight={args.pos_weight})...")
    augmented = build_augmented_embeddings(embeddings, train_data, args.pos_weight)
    print(f"Augmented embedding shape: {augmented.shape}  "
          f"(original {embeddings.shape[1]}D + 1 position dim)")

    # Position distribution in full train set (for reference)
    all_positions = [
        d.get("step_idx", 0) / max(d.get("total_steps", 1) - 1, 1)
        for d in train_data
    ]
    print(f"Full train position distribution: "
          f"early={sum(1 for p in all_positions if p < 0.33)}, "
          f"mid={sum(1 for p in all_positions if 0.33 <= p < 0.67)}, "
          f"late={sum(1 for p in all_positions if p >= 0.67)}")

    tag = f"_stepaware{str(args.pos_weight).replace('.', '')}"

    for k in args.clusters:
        print(f"\n{'='*60}")
        print(f"  K={k}: Step-aware cluster selection (pos_weight={args.pos_weight})")
        print(f"{'='*60}")

        selected, labels = cluster_and_select(augmented, embeddings, k, seed=args.seed)
        report_selection(train_data, selected, labels, k, args.pos_weight)

        # Save subset
        subset = [train_data[i] for i in selected]
        name = f"mind2web_train_cluster{k}{tag}"
        out_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        save_jsonl(subset, out_path)

        # Save metadata
        meta = {
            "method": "step_aware_kmeans",
            "k": k,
            "pos_weight": args.pos_weight,
            "seed": args.seed,
            "selected_indices": selected,
            "n_total": len(train_data),
            "step_positions": [
                train_data[i].get("step_idx", 0) / max(train_data[i].get("total_steps", 1) - 1, 1)
                for i in selected
            ],
        }
        meta_path = os.path.join(OUTPUT_DIR, f"cluster{k}{tag}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"    Saved metadata → {meta_path}")

    update_config(args.clusters, args.pos_weight)

    print(f"\n{'='*60}")
    print(f"  DONE — Generated {len(args.clusters)} step-aware training subsets")
    print(f"{'='*60}")
    print(f"\n  Next: Run ACE training:\n")
    for k in args.clusters:
        name = f"mind2web_cluster{k}{tag}"
        print(f"    python -m eval.mind2web.run \\")
        print(f"      --task_name {name} \\")
        print(f"      --mode offline --eval_steps {k} \\")
        print(f"      --save_path results/{name}")
        print()


if __name__ == "__main__":
    main()
