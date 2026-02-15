#!/usr/bin/env python3
"""
DPP-based Diverse Subset Selection for Mind2Web.

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
    python -m eval.mind2web.embed_train    # generates embeddings.npy

Usage:
    # Default: K = 10, 15, 20, 30, 50, 80
    python -m eval.mind2web.dpp_select

    # Custom subset sizes
    python -m eval.mind2web.dpp_select --clusters 10 20 50
"""
import os
import json
import argparse
import time
import numpy as np
from typing import List, Dict
from collections import Counter

# ── Config ──────────────────────────────────────────────────────

TRAIN_PATH = "./eval/mind2web/data/mind2web_train.jsonl"
EMBEDDING_PATH = "./eval/mind2web/data/embeddings.npy"
OUTPUT_DIR = "./eval/mind2web/data"
CONFIG_PATH = "./eval/mind2web/data/sample_config.json"
DEFAULT_CLUSTERS = [10, 15, 20, 30, 50, 80]


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

    For large N (4477), we use the Schur complement approach for efficiency
    instead of recomputing determinants from scratch.

    Args:
        L: N×N positive semi-definite kernel matrix
        k: number of items to select

    Returns:
        List of k selected indices
    """
    N = L.shape[0]
    selected = []
    selected_set = set()

    # We maintain L_S_inv incrementally for efficiency
    L_S_inv = None

    for step in range(k):
        best_item = -1
        best_gain = -np.inf

        for i in range(N):
            if i in selected_set:
                continue

            if not selected:
                # First item: marginal gain is just L[i,i]
                gain = L[i, i]
            else:
                # Schur complement: L_ii - L_{i,S} @ L_{S,S}^{-1} @ L_{S,i}
                L_iS = L[i, selected]  # (m,) vector
                gain = L[i, i] - L_iS @ L_S_inv @ L_iS

            if gain > best_gain:
                best_gain = gain
                best_item = i

        if best_item == -1:
            # Fallback: pick first remaining item
            for i in range(N):
                if i not in selected_set:
                    best_item = i
                    break

        selected.append(best_item)
        selected_set.add(best_item)

        # Update L_S_inv using block matrix inversion formula
        if len(selected) == 1:
            L_S_inv = np.array([[1.0 / L[best_item, best_item]]])
        else:
            # Block inversion: given L_S_inv for old S, compute new inverse
            # after adding best_item
            L_iS = L[best_item, selected[:-1]]  # (m-1,) vector
            schur = L[best_item, best_item] - L_iS @ L_S_inv @ L_iS
            schur_inv = 1.0 / schur

            # New inverse in block form
            v = L_S_inv @ L_iS  # (m-1,) vector
            new_top_left = L_S_inv + schur_inv * np.outer(v, v)
            new_top_right = -schur_inv * v
            new_bottom_left = -schur_inv * v
            new_bottom_right = schur_inv

            m = len(selected)
            new_inv = np.zeros((m, m))
            new_inv[:m-1, :m-1] = new_top_left
            new_inv[:m-1, m-1] = new_top_right
            new_inv[m-1, :m-1] = new_bottom_left
            new_inv[m-1, m-1] = new_bottom_right
            L_S_inv = new_inv

        if (step + 1) % 5 == 0 or step == k - 1:
            L_sub = L[np.ix_(selected, selected)]
            _, logdet = np.linalg.slogdet(L_sub)
            print(f"      Step {step+1}/{k}: selected item {best_item}, "
                  f"log-det = {logdet:.4f}")

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

    # Domain distribution in selected subset
    domains = Counter()
    operations = Counter()
    websites = set()
    for i in selected_indices:
        item = train_data[i]
        others = item.get("others", {})
        domains[others.get("domain", "?")] += 1
        operations[others.get("operation", "?")] += 1
        websites.add(others.get("website", "?"))
    print(f"    Domains: {dict(domains)}")
    print(f"    Operations: {dict(operations)}")
    print(f"    Unique websites: {len(websites)}/{k}")


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
        config[f"mind2web_dpp{k}"] = {
            "train_data": f"./eval/mind2web/data/mind2web_train_dpp{k}.jsonl",
            "val_data": "./eval/mind2web/data/mind2web_val.jsonl",
            "test_data": "./eval/mind2web/data/mind2web_test.jsonl",
        }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DPP-based diverse subset selection for Mind2Web"
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
            f"Run: python -m eval.mind2web.embed_train"
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
    t0 = time.time()
    L = build_kernel(embeddings)
    print(f"Kernel matrix shape: {L.shape}, "
          f"mean={L.mean():.4f}, diag_mean={np.diag(L).mean():.6f} "
          f"(built in {time.time()-t0:.1f}s)")

    # Run Greedy MAP k-DPP for each K
    for k in args.clusters:
        if k > len(train_data):
            print(f"\n  [SKIP] K={k} > {len(train_data)} training samples")
            continue

        print(f"\n{'='*60}")
        print(f"  Greedy MAP k-DPP: Selecting {k} diverse training samples")
        print(f"  (N={len(train_data)}, complexity ≈ O({k}×{len(train_data)}))")
        print(f"{'='*60}")

        t0 = time.time()
        selected = greedy_map_dpp(L, k)
        elapsed = time.time() - t0
        print(f"    Selection completed in {elapsed:.1f}s")

        report_selection(train_data, selected, k, L)
        compare_with_kmeans(selected, k, embeddings, L)

        # Save subset
        subset = [train_data[i] for i in selected]
        out_path = os.path.join(OUTPUT_DIR, f"mind2web_train_dpp{k}.jsonl")
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
        print(f"    python -m eval.mind2web.run \\")
        print(f"      --task_name mind2web_dpp{k} \\")
        print(f"      --mode offline --eval_steps {k} \\")
        print(f"      --save_path results/mind2web_dpp{k}")
        print()


if __name__ == "__main__":
    main()
