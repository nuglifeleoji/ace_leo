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

Website Deduplication (--dedup, default ON):
    After K-means selects one representative per cluster, check for
    duplicate websites. If two clusters would pick the same website,
    the cluster with the worse centroid-distance yields its pick and
    searches for the next-best candidate from a different website.
    This preserves semantic representativeness while maximising the
    number of distinct websites in the final selection.

Usage:
    # Generate step-aware clusters for k=10,15,20 (with dedup)
    python -m eval.mind2web.cluster_train_stepaware --clusters 10 15 20

    # Disable website deduplication
    python -m eval.mind2web.cluster_train_stepaware --clusters 15 --no_dedup

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
    train_data: List[Dict],
    k: int,
    seed: int = 42,
    dedup_website: bool = True,
) -> Tuple[List[int], np.ndarray]:
    """
    Run K-means on augmented embeddings, then select the sample
    closest to each centroid in the augmented space.

    When dedup_website=True (default), a post-processing pass ensures
    every selected sample comes from a distinct website:
      - Clusters are processed in order of their best sample's distance
        to the centroid (closest = highest-quality cluster gets first pick).
      - If a cluster's best candidate shares a website with an already-chosen
        sample, the cluster falls back to its 2nd-best candidate, then 3rd,
        etc., until a unique website is found.
      - If a cluster has no candidates with a unique website, it keeps its
        original best pick (fallback, logged as a warning).

    Returns:
        (selected_indices, labels)
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(augmented_embeddings)
    centroids_aug = kmeans.cluster_centers_

    # For each cluster, rank all members by distance to centroid (ascending)
    cluster_candidates: Dict[int, List[Tuple[float, int]]] = {}
    for cid in range(k):
        mask = labels == cid
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        dists = np.linalg.norm(
            augmented_embeddings[indices] - centroids_aug[cid], axis=1
        )
        # sorted: best (smallest dist) first
        sorted_pairs = sorted(zip(dists.tolist(), indices.tolist()), key=lambda x: x[0])
        cluster_candidates[cid] = sorted_pairs

    if not dedup_website:
        # Simple: just take the best from each cluster
        selected_indices = [pairs[0][1] for pairs in cluster_candidates.values()]
        return selected_indices, labels

    # ── Website-dedup greedy assignment ──────────────────────────────────────
    # Process clusters in order of their #1 candidate's distance (best quality
    # clusters get priority to claim their preferred website first).
    sorted_clusters = sorted(
        cluster_candidates.items(), key=lambda kv: kv[1][0][0]  # sort by best dist
    )

    used_websites: set = set()
    selected_by_cid: Dict[int, int] = {}

    for cid, candidates in sorted_clusters:
        chosen_idx = None
        for _dist, idx in candidates:
            website = train_data[idx].get("website", "__unknown__")
            if website not in used_websites:
                chosen_idx = idx
                used_websites.add(website)
                break
        if chosen_idx is None:
            # All candidates share a website with an already-chosen sample;
            # fall back to the original best to avoid dropping the cluster.
            chosen_idx = candidates[0][1]
            fallback_site = train_data[chosen_idx].get("website", "?")
            print(f"      [dedup] cluster {cid}: no unique website found "
                  f"(fallback → {fallback_site})")
        selected_by_cid[cid] = chosen_idx

    # Return in cluster-id order (deterministic)
    selected_indices = [selected_by_cid[cid] for cid in sorted(selected_by_cid)]
    return selected_indices, labels


# ── Reporting ───────────────────────────────────────────────────

def report_selection(
    train_data: List[Dict],
    selected_indices: List[int],
    labels: np.ndarray,
    k: int,
    pos_weight: float,
    dedup_website: bool = True,
):
    """Print statistics including step position distribution and website uniqueness."""
    cluster_sizes = np.bincount(labels)
    print(f"    Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}")

    domains = Counter(train_data[i].get("domain", "?") for i in selected_indices)
    ops = Counter(
        str(train_data[i].get("operation", {}).get("op", "?"))
        for i in selected_indices
    )
    website_list = [train_data[i].get("website", "?") for i in selected_indices]
    unique_websites = len(set(website_list))
    dup_sites = {w: c for w, c in Counter(website_list).items() if c > 1}

    print(f"    Domains   : {dict(domains)}")
    print(f"    Operations: {dict(ops)}")
    dedup_flag = " (dedup ON)" if dedup_website else " (dedup OFF)"
    print(f"    Unique websites: {unique_websites}/{k}{dedup_flag}"
          + (f"  duplicates={dup_sites}" if dup_sites else "  ✓ all unique"))

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
    print(f"    (pos_weight={pos_weight}; vanilla cluster20 had: avg=0.11, early=18, mid=1, late=1)")

    # Per-sample detail
    for i in selected_indices:
        d = train_data[i]
        step = d.get("step_idx", 0)
        total = d.get("total_steps", 1)
        pos = step / max(total - 1, 1)
        pos_l = "E" if pos < 0.33 else ("M" if pos < 0.67 else "L")
        op = d.get("operation", {}).get("op", "?")
        site = d.get("website", "?")
        print(f"      [{pos_l}] {site:20s} step{step+1}/{total}  {op}")


# ── Config Update ───────────────────────────────────────────────

def make_tag(pos_weight: float, dedup: bool) -> str:
    """Build a short tag string that encodes the method variant."""
    tag = f"_stepaware{str(pos_weight).replace('.', '')}"
    if dedup:
        tag += "_dedup"
    return tag


def update_config(cluster_sizes: List[int], pos_weight: float, dedup: bool):
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    tag = make_tag(pos_weight, dedup)
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
    parser.add_argument(
        "--no_dedup", action="store_true",
        help="Disable website-level deduplication post-processing (default: dedup ON)"
    )
    args = parser.parse_args()
    args.dedup = not args.no_dedup

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
    print(f"Website deduplication: {'ON' if args.dedup else 'OFF'}")

    tag = make_tag(args.pos_weight, args.dedup)

    for k in args.clusters:
        print(f"\n{'='*60}")
        print(f"  K={k}: Step-aware cluster selection "
              f"(pos_weight={args.pos_weight}, dedup={'ON' if args.dedup else 'OFF'})")
        print(f"{'='*60}")

        selected, labels = cluster_and_select(
            augmented, embeddings, train_data, k,
            seed=args.seed, dedup_website=args.dedup,
        )
        report_selection(train_data, selected, labels, k, args.pos_weight, args.dedup)

        # Save subset
        subset = [train_data[i] for i in selected]
        name = f"mind2web_train_cluster{k}{tag}"
        out_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        save_jsonl(subset, out_path)

        # Save metadata
        meta = {
            "method": "step_aware_kmeans_dedup" if args.dedup else "step_aware_kmeans",
            "k": k,
            "pos_weight": args.pos_weight,
            "dedup_website": args.dedup,
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

    update_config(args.clusters, args.pos_weight, args.dedup)

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
