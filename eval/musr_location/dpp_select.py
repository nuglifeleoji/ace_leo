#!/usr/bin/env python3
"""
DPP-based Diverse Subset Selection for MuSR Location.

Uses pre-computed embeddings (from embed_train.py) and Determinantal Point
Processes (DPP) to select diverse training subsets. This is an alternative
to K-means centroid selection (cluster_train.py).

Method: Greedy MAP k-DPP
    At each step, select the item that maximizes the marginal gain in
    log-determinant of the selected submatrix L_S. This greedily
    approximates the MAP solution of k-DPP (maximizing det(L_S) over
    all subsets S of size k).

    The key difference from K-means:
    - K-means selects "typical" points near cluster centroids
    - DPP selects "diverse" points that maximize coverage / volume
      in embedding space (the determinant measures the volume of the
      parallelepiped spanned by the selected vectors)

Prerequisite:
    python -m eval.musr_location.embed_train    # generates embeddings.npy

Usage:
    # Default: K = 5, 10, 15, 20, 30
    python -m eval.musr_location.dpp_select

    # Custom cluster sizes
    python -m eval.musr_location.dpp_select --clusters 5 10 20
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
DEFAULT_CLUSTERS = [5, 10, 15, 20, 30]


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


# ── DPP Core ────────────────────────────────────────────────────

def build_kernel(embeddings: np.ndarray) -> np.ndarray:
    """
    Build the L-ensemble kernel matrix from embeddings.

    Uses cosine similarity (dot product of unit vectors).
    Since OpenAI text-embedding-3-large outputs are already normalized,
    L = X @ X^T gives cosine similarity, which is PSD by construction.

    A small regularization term is added for numerical stability.
    """
    # Normalize to unit vectors (should already be, but just in case)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X = embeddings / norms

    L = X @ X.T

    # Small regularization for numerical stability
    L += 1e-8 * np.eye(len(L))

    return L


def greedy_map_dpp(L: np.ndarray, k: int) -> List[int]:
    """
    Greedy MAP inference for k-DPP.

    At each step, select the item i that maximizes:
        log det(L_{S ∪ {i}}) - log det(L_S)

    This is equivalent to maximizing the Schur complement:
        L_{ii} - L_{i,S} @ L_{S,S}^{-1} @ L_{S,i}

    which is the conditional variance of item i given the already-
    selected items. This greedy approach provides a (1-1/e)-approximation
    to the NP-hard MAP k-DPP problem.

    For N=103, the naive O(k * N * k^2) approach is efficient enough.

    Args:
        L: N×N positive semi-definite kernel matrix
        k: number of items to select

    Returns:
        List of k selected indices
    """
    N = L.shape[0]
    selected = []
    selected_set = set()

    for step in range(k):
        best_item = -1
        best_logdet = -np.inf

        for i in range(N):
            if i in selected_set:
                continue

            candidate = selected + [i]
            L_sub = L[np.ix_(candidate, candidate)]
            sign, logdet = np.linalg.slogdet(L_sub)

            if sign > 0 and logdet > best_logdet:
                best_logdet = logdet
                best_item = i

        if best_item == -1:
            # Fallback: pick first remaining item
            for i in range(N):
                if i not in selected_set:
                    best_item = i
                    break

        selected.append(best_item)
        selected_set.add(best_item)

        if (step + 1) % 5 == 0 or step == k - 1:
            print(f"      Step {step+1}/{k}: selected item {best_item}, "
                  f"log-det = {best_logdet:.4f}")

    return selected


# ── Analysis & Reporting ────────────────────────────────────────

def report_selection(
    train_data: List[Dict],
    selected_indices: List[int],
    k: int,
    L: np.ndarray,
):
    """Print statistics about the DPP-selected subset."""
    # Log-determinant of the selected submatrix
    L_sub = L[np.ix_(selected_indices, selected_indices)]
    sign, logdet = np.linalg.slogdet(L_sub)
    print(f"    Final log-det(L_S): {logdet:.4f}")

    # Pairwise similarity statistics of selected items
    sims = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            sims.append(L[selected_indices[i], selected_indices[j]])
    sims = np.array(sims)
    print(f"    Pairwise cosine-sim of selected: "
          f"mean={sims.mean():.4f}, min={sims.min():.4f}, max={sims.max():.4f}")

    # Answer distribution in selected subset
    answers = Counter(train_data[i].get("target", "?") for i in selected_indices)
    print(f"    Selected answers: {dict(answers)}")

    # Compare with full training set distribution
    all_answers = Counter(d.get("target", "?") for d in train_data)
    print(f"    (Full train answers: {dict(all_answers)})")


def compare_with_kmeans(
    selected_dpp: List[int],
    k: int,
    embeddings: np.ndarray,
    L: np.ndarray,
):
    """Compare DPP selection with K-means selection."""
    # Load K-means metadata if available
    kmeans_meta_path = os.path.join(OUTPUT_DIR, f"cluster{k}_meta.json")
    if not os.path.exists(kmeans_meta_path):
        print(f"    (K-means meta not found for K={k}, skipping comparison)")
        return

    with open(kmeans_meta_path, "r") as f:
        kmeans_meta = json.load(f)
    selected_kmeans = kmeans_meta["selected_indices"]

    # Log-det comparison
    L_dpp = L[np.ix_(selected_dpp, selected_dpp)]
    L_km = L[np.ix_(selected_kmeans, selected_kmeans)]
    _, logdet_dpp = np.linalg.slogdet(L_dpp)
    _, logdet_km = np.linalg.slogdet(L_km)

    print(f"\n    ┌─ DPP vs K-means (K={k}) ─────────────────────")
    print(f"    │ log-det(L)  DPP: {logdet_dpp:.4f}  vs  K-means: {logdet_km:.4f}")
    print(f"    │ DPP {'wins' if logdet_dpp > logdet_km else 'loses'} "
          f"by {abs(logdet_dpp - logdet_km):.4f}")

    # Overlap
    overlap = set(selected_dpp) & set(selected_kmeans)
    print(f"    │ Overlap: {len(overlap)}/{k} items in common")

    # Average pairwise distance
    def avg_pairwise_dist(indices):
        pts = embeddings[indices]
        from scipy.spatial.distance import pdist
        return np.mean(pdist(pts, 'cosine'))

    try:
        dist_dpp = avg_pairwise_dist(selected_dpp)
        dist_km = avg_pairwise_dist(selected_kmeans)
        print(f"    │ Avg pairwise cosine-dist  DPP: {dist_dpp:.4f}  vs  K-means: {dist_km:.4f}")
    except ImportError:
        pass

    print(f"    └──────────────────────────────────────────────")


# ── Config Update ───────────────────────────────────────────────

def update_config(cluster_sizes: List[int]):
    """Add DPP configs to sample_config.json."""
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    for k in cluster_sizes:
        config[f"musr_location_dpp{k}"] = {
            "train_data": f"./eval/musr_location/data/location_train_dpp{k}.jsonl",
            "val_data": "./eval/musr_location/data/location_val.jsonl",
            "test_data": "./eval/musr_location/data/location_test.jsonl",
        }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DPP-based diverse subset selection for MuSR Location"
    )
    parser.add_argument(
        "--clusters", type=int, nargs="+", default=DEFAULT_CLUSTERS,
        help=f"Subset sizes to select (default: {DEFAULT_CLUSTERS})"
    )
    args = parser.parse_args()

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

    # Build kernel matrix (once, reuse for all K)
    print("\nBuilding kernel matrix L = X @ X^T ...")
    L = build_kernel(embeddings)
    print(f"Kernel matrix shape: {L.shape}, "
          f"mean={L.mean():.4f}, diag_mean={np.diag(L).mean():.6f}")

    # Run Greedy MAP k-DPP for each K
    for k in args.clusters:
        if k > len(train_data):
            print(f"\n  [SKIP] K={k} > {len(train_data)} training samples")
            continue

        print(f"\n{'='*60}")
        print(f"  Greedy MAP k-DPP: Selecting {k} diverse training samples")
        print(f"{'='*60}")

        selected = greedy_map_dpp(L, k)
        report_selection(train_data, selected, k, L)
        compare_with_kmeans(selected, k, embeddings, L)

        # Save subset
        subset = [train_data[i] for i in selected]
        out_path = os.path.join(OUTPUT_DIR, f"location_train_dpp{k}.jsonl")
        save_jsonl(subset, out_path)

        # Save metadata
        meta = {
            "method": "greedy_map_dpp",
            "k": k,
            "selected_indices": selected,
            "n_total": len(train_data),
            "kernel": "cosine",
        }
        L_sub = L[np.ix_(selected, selected)]
        _, logdet = np.linalg.slogdet(L_sub)
        meta["log_det"] = float(logdet)

        meta_path = os.path.join(OUTPUT_DIR, f"dpp{k}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Update config
    update_config(args.clusters)

    # Print next steps
    print(f"\n{'='*60}")
    print(f"  DONE — Generated {len(args.clusters)} DPP training subsets")
    print(f"{'='*60}")
    print(f"\n  Next: Run ACE training for each DPP subset:\n")
    for k in args.clusters:
        print(f"    python -m eval.musr_location.run \\")
        print(f"      --task_name musr_location_dpp{k} \\")
        print(f"      --mode offline --eval_steps {k} \\")
        print(f"      --save_path results/musr_location_dpp{k}")
        print()


if __name__ == "__main__":
    main()
