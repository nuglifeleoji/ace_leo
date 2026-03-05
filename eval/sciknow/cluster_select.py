"""
K-means clustering for SciKnowEval Chemistry L3 training subset selection.

Uses pre-computed embeddings (from embed_train.py) to select semantically
representative training subsets via K-means. One sample closest to each
cluster centroid is selected as the representative.

Prerequisite:
    python -m eval.sciknow.embed_train

Usage:
    python -m eval.sciknow.cluster_select
    python -m eval.sciknow.cluster_select --clusters 10 20 50
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "./eval/sciknow/data/sciknow_chem_l3_train.jsonl"
EMBED_PATH  = "./eval/sciknow/data/sciknow_embeddings.npy"
OUTPUT_DIR  = "./eval/sciknow/data"
CONFIG_PATH = "./eval/sciknow/data/sample_config.json"
VAL_PATH    = "./eval/sciknow/data/sciknow_chem_l3_val.jsonl"
TEST_PATH   = "./eval/sciknow/data/sciknow_chem_l3_test.jsonl"

DEFAULT_CLUSTERS = [5, 10, 20, 30, 40, 50, 80]


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


def cluster_and_select(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray]:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_

    selected: List[int] = []
    for cid in range(k):
        mask  = labels == cid
        idxs  = np.where(mask)[0]
        dists = np.linalg.norm(embeddings[idxs] - centroids[cid], axis=1)
        selected.append(int(idxs[np.argmin(dists)]))

    return selected, labels


def report(train_data: List[Dict], selected: List[int], labels: np.ndarray) -> None:
    sizes = np.bincount(labels)
    print(f"    Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
          f"mean={sizes.mean():.1f}")
    task_cnt = Counter(train_data[i]["task"] for i in selected)
    print(f"    Task distribution in subset: {dict(task_cnt)}")


def update_config(cluster_sizes: List[int]) -> None:
    config: Dict = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)

    for k in cluster_sizes:
        key = f"sciknow_chem_l3_cluster{k}"
        config[key] = {
            "train_data": f"./eval/sciknow/data/sciknow_train_cluster{k}.jsonl",
            "val_data":   VAL_PATH,
            "test_data":  TEST_PATH,
        }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, nargs="+", default=DEFAULT_CLUSTERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  SciKnowEval — K-means Cluster Subset Selection")
    print("=" * 60)

    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(
            f"Embeddings not found: {EMBED_PATH}\n"
            "Run first:  python -m eval.sciknow.embed_train"
        )
    embeddings = np.load(EMBED_PATH)
    train_data = load_jsonl(TRAIN_PATH)
    assert len(train_data) == embeddings.shape[0]
    print(f"Loaded embeddings {embeddings.shape}, {len(train_data)} samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for k in sorted(args.clusters):
        print(f"\n{'='*55}")
        print(f"  K={k}: selecting {k} representative samples")
        print(f"{'='*55}")
        selected, labels = cluster_and_select(embeddings, k, seed=args.seed)
        report(train_data, selected, labels)

        subset = [train_data[i] for i in selected]
        save_jsonl(subset, os.path.join(OUTPUT_DIR, f"sciknow_train_cluster{k}.jsonl"))

        with open(os.path.join(OUTPUT_DIR, f"sciknow_cluster{k}_meta.json"), "w") as f:
            json.dump({"k": k, "seed": args.seed, "selected_indices": selected,
                       "n_total": len(train_data)}, f, indent=2)

    update_config(args.clusters)
    print("\nDone.")


if __name__ == "__main__":
    main()
