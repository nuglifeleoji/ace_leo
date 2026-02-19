#!/usr/bin/env python3
"""
Build a 1-sample "patch" training set from KMeans-20 playbook errors.

Workflow
--------
1) Load the error list produced by `cluster20_plus_error10.py`
2) Cluster the incorrect samples in embedding space (default k=5)
3) Pick ONE representative patch sample:
   - from the largest error cluster
   - closest to that cluster centroid (prototype)
4) Write a 1-sample train JSONL + add a `sample_config.json` entry.

This is meant for quick experiments:
  Start from the strong `musr_location_cluster20` playbook, then train on 1
  representative failure case to see if it improves generalization.

Usage
-----
python -m eval.musr_location.patch_from_kmeans20_errors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


DATA_DIR = Path("./eval/musr_location/data")
TRAIN_JSONL = DATA_DIR / "location_train.jsonl"
EMB_PATH = DATA_DIR / "embeddings.npy"
CONFIG_PATH = DATA_DIR / "sample_config.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_patch_sample(
    embeddings: np.ndarray,
    wrong_abs_indices: List[int],
    n_clusters: int,
    seed: int,
) -> Tuple[int, Dict[str, Any]]:
    """
    Cluster wrong samples and pick ONE representative sample.
    Returns (chosen_abs_index, debug_info).
    """
    if len(wrong_abs_indices) == 0:
        raise ValueError("wrong_abs_indices is empty")

    X = embeddings[np.array(wrong_abs_indices, dtype=int)]
    n_clusters = min(n_clusters, X.shape[0])

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    # cluster sizes
    sizes = np.bincount(labels, minlength=n_clusters)
    largest = int(np.argmax(sizes))

    # pick nearest-to-centroid within largest cluster
    idx = np.where(labels == largest)[0]
    dists = np.linalg.norm(X[idx] - centroids[largest], axis=1)
    chosen_pos = int(idx[np.argmin(dists)])
    chosen_abs = int(wrong_abs_indices[chosen_pos])

    debug = {
        "n_wrong": len(wrong_abs_indices),
        "n_clusters": n_clusters,
        "seed": seed,
        "cluster_sizes": sizes.tolist(),
        "largest_cluster": largest,
        "chosen_pos_in_wrong_list": chosen_pos,
        "chosen_abs_index": chosen_abs,
        "chosen_dist_to_centroid": float(np.min(dists)),
    }
    return chosen_abs, debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 1-sample patch set from kmeans20 errors")
    parser.add_argument("--kmeans_k", type=int, default=20)
    parser.add_argument("--error_meta", type=str, default="cluster20_err10_meta.json")
    parser.add_argument("--error_clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(f"Missing {TRAIN_JSONL}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Missing {EMB_PATH}")

    meta_path = DATA_DIR / args.error_meta
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path} (run cluster20_plus_error10 first)")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    wrong_abs = list(map(int, meta.get("wrong_abs_indices", [])))
    if not wrong_abs:
        raise ValueError("No wrong_abs_indices found in meta")

    train = load_jsonl(TRAIN_JSONL)
    emb = np.load(EMB_PATH)
    if len(train) != emb.shape[0]:
        raise ValueError(f"Mismatch train={len(train)} vs embeddings={emb.shape[0]}")

    chosen_abs, debug = pick_patch_sample(
        embeddings=emb,
        wrong_abs_indices=wrong_abs,
        n_clusters=args.error_clusters,
        seed=args.seed,
    )

    task_name = f"musr_location_cluster{args.kmeans_k}_patch1_i{chosen_abs}"
    out_jsonl = DATA_DIR / f"location_train_cluster{args.kmeans_k}_patch1_i{chosen_abs}.jsonl"
    save_jsonl([train[chosen_abs]], out_jsonl)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg[task_name] = {
        "train_data": str(out_jsonl),
        "val_data": "./eval/musr_location/data/location_val.jsonl",
        "test_data": "./eval/musr_location/data/location_test.jsonl",
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=4), encoding="utf-8")

    patch_meta = {
        "task_name": task_name,
        "chosen_abs_index": chosen_abs,
        "kmeans_base": f"cluster{args.kmeans_k}",
        "source_error_meta": args.error_meta,
        "debug": debug,
        "playbook_path": meta.get("playbook_path"),
    }
    meta_out = DATA_DIR / f"cluster{args.kmeans_k}_patch1_meta.json"
    meta_out.write_text(json.dumps(patch_meta, indent=2), encoding="utf-8")

    print(f"Chosen patch sample abs_index={chosen_abs}")
    print(f"Wrote: {out_jsonl}")
    print(f"Updated: {CONFIG_PATH} with task {task_name}")
    print(f"Wrote meta: {meta_out}")
    print("\nNext:")
    print("  python -m eval.musr_location.run \\")
    print(f"    --task_name {task_name} \\")
    print("    --mode offline --eval_steps 1 --skip_initial_test \\")
    print("    --initial_playbook_path <cluster20_best_playbook.txt> \\")
    print(f"    --save_path results/{task_name}")


if __name__ == "__main__":
    main()

