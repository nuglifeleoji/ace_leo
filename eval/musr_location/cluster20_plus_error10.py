#!/usr/bin/env python3
"""
MuSR Location: KMeans-20 + Error-Mining-10 (clustered errors) subset builder.

Build a 30-sample training set:
  - 20 samples from existing K-means selection (cluster20_meta.json)
  - 10 additional samples mined from the remaining train set that the
    KMeans-20 best playbook gets wrong, clustered into 5 clusters and selected
    as 2-per-cluster (prototype + mid-far quantile) for coverage.

This implements the workflow:
  1) Train using KMeans-20 subset → get best_playbook.txt
  2) Evaluate best_playbook on remaining train samples → collect incorrect ones
  3) Cluster incorrect ones (k=5) and select 10 representatives
  4) Combine with the original 20 and write a new train JSONL + config entry

Usage:
  python -m eval.musr_location.cluster20_plus_error10

  # Custom parameters
  python -m eval.musr_location.cluster20_plus_error10 \\
    --kmeans_k 20 --error_clusters 5 --per_error_cluster 2 \\
    --far_quantile 0.85 --max_workers 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ace import ACE
from utils import evaluate_test_set

from .data_processor import DataProcessor


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


def load_best_playbook_for_task(results_dir: Path, task_dir_name: str) -> Optional[Path]:
    """
    Find the newest best_playbook.txt under results/<task_dir_name>/*/best_playbook.txt
    """
    base = results_dir / task_dir_name
    if not base.exists():
        return None
    pbs = sorted(base.glob("*/best_playbook.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pbs[0] if pbs else None


def select_proto_plus_midfar(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    per_cluster: int,
    far_quantile: float,
) -> List[int]:
    """
    For each cluster id, select:
      - 1 nearest to centroid (prototype)
      - (per_cluster-1) mid-far points at distance quantiles inside the cluster

    For our use case per_cluster=2 → returns 2 per cluster when possible.
    """
    if per_cluster < 1:
        raise ValueError("--per_error_cluster must be >= 1")
    if not (0.0 < far_quantile < 1.0):
        raise ValueError("--far_quantile must be in (0, 1)")

    dists = np.linalg.norm(embeddings - centroids[labels], axis=1)
    selected: List[int] = []
    selected_set = set()

    k = centroids.shape[0]
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue

        # prototype
        proto = int(idx[np.argmin(dists[idx])])
        selected.append(proto)
        selected_set.add(proto)

        if per_cluster == 1 or idx.size == 1:
            continue

        order = idx[np.argsort(dists[idx])]

        # pick remaining points by quantiles around far_quantile (spread slightly)
        # e.g., for per_cluster=2, just far_quantile; for larger, use linspace.
        qs = np.linspace(far_quantile, min(0.95, far_quantile + 0.10), per_cluster - 1)
        for q in qs:
            pos = int(round(q * (len(order) - 1)))
            pos = max(0, min(pos, len(order) - 1))
            cand = int(order[pos])
            if cand in selected_set:
                # find nearest neighbor not selected
                found = None
                for delta in range(1, len(order)):
                    for j in (pos - delta, pos + delta):
                        if 0 <= j < len(order) and int(order[j]) not in selected_set:
                            found = int(order[j])
                            break
                    if found is not None:
                        break
                if found is None:
                    continue
                cand = found
            selected.append(cand)
            selected_set.add(cand)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build musr_location kmeans20 + error10 subset")
    parser.add_argument("--kmeans_k", type=int, default=20, help="Base K-means subset size (default: 20)")
    parser.add_argument("--error_clusters", type=int, default=5, help="KMeans clusters over error pool (default: 5)")
    parser.add_argument("--per_error_cluster", type=int, default=2, help="Selected per error cluster (default: 2)")
    parser.add_argument("--far_quantile", type=float, default=0.85, help="Mid-far quantile inside error cluster (default: 0.85)")
    parser.add_argument("--seed", type=int, default=42, help="KMeans seed (default: 42)")
    parser.add_argument("--max_workers", type=int, default=5, help="Parallel workers for error evaluation (default: 5)")
    parser.add_argument("--api_provider", type=str, default="sambanova", choices=["sambanova", "together", "openai"])
    parser.add_argument("--generator_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--reflector_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--curator_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(f"Missing {TRAIN_JSONL}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Missing {EMB_PATH} (run embed_train first)")

    train = load_jsonl(TRAIN_JSONL)
    emb = np.load(EMB_PATH)
    if len(train) != emb.shape[0]:
        raise ValueError(f"Mismatch train={len(train)} vs embeddings={emb.shape[0]}")

    # Load kmeans indices
    meta_path = DATA_DIR / f"cluster{args.kmeans_k}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path} (run cluster_train first)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base_indices = list(map(int, meta["selected_indices"]))
    base_set = set(base_indices)
    if len(base_indices) != args.kmeans_k:
        print(f"[WARN] cluster meta has {len(base_indices)} indices, expected {args.kmeans_k}")

    # Determine remaining pool
    pool_indices = [i for i in range(len(train)) if i not in base_set]
    pool_samples_raw = [train[i] for i in pool_indices]

    # Load playbook from cluster20 run
    results_dir = Path("./results")
    pb_path = load_best_playbook_for_task(results_dir, f"musr_location_cluster{args.kmeans_k}")
    if pb_path is None:
        raise FileNotFoundError(
            f"Could not find best_playbook for results/musr_location_cluster{args.kmeans_k}/*/best_playbook.txt"
        )
    playbook = pb_path.read_text(encoding="utf-8")
    print(f"Using playbook: {pb_path}")
    print(f"Pool size (remaining train): {len(pool_samples_raw)}")

    # Evaluate playbook on pool to collect incorrect samples
    dp = DataProcessor(task_name="musr_location")
    pool_samples = dp.process_task_data(pool_samples_raw)

    ace = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=playbook,
    )

    # Use evaluate_test_set directly to get per-sample errors
    test_results, error_log = evaluate_test_set(
        data_processor=dp,
        generator=ace.generator,
        playbook=playbook,
        test_samples=pool_samples,
        max_tokens=args.max_tokens,
        log_dir=None,
        max_workers=args.max_workers,
        use_json_mode=False,
    )

    errors = error_log.get("errors", [])
    wrong_pool_positions = [int(e["index"]) for e in errors]
    wrong_abs_indices = [pool_indices[pos] for pos in wrong_pool_positions]
    print(f"Incorrect in pool: {len(wrong_abs_indices)}/{len(pool_indices)}")

    if len(wrong_abs_indices) == 0:
        raise RuntimeError("No incorrect samples found in pool; cannot build error-mined subset.")

    # Cluster errors and select representatives
    err_emb = emb[np.array(wrong_abs_indices, dtype=int)]
    if err_emb.shape[0] < args.error_clusters:
        print(f"[WARN] only {err_emb.shape[0]} errors; reducing error_clusters to {err_emb.shape[0]}")
        args.error_clusters = err_emb.shape[0]

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=args.error_clusters, random_state=args.seed, n_init=10, max_iter=300)
    err_labels = km.fit_predict(err_emb)
    err_centroids = km.cluster_centers_

    selected_err_positions = select_proto_plus_midfar(
        embeddings=err_emb,
        labels=err_labels,
        centroids=err_centroids,
        per_cluster=args.per_error_cluster,
        far_quantile=args.far_quantile,
    )

    # Map back to absolute indices
    selected_err_abs = [wrong_abs_indices[p] for p in selected_err_positions]

    # Enforce exactly 10 (or as close as possible)
    target_err_n = args.error_clusters * args.per_error_cluster
    if len(selected_err_abs) > target_err_n:
        selected_err_abs = selected_err_abs[:target_err_n]

    # De-dup, keep order
    seen = set()
    selected_err_abs2 = []
    for i in selected_err_abs:
        if i not in seen and i not in base_set:
            seen.add(i)
            selected_err_abs2.append(int(i))
    selected_err_abs = selected_err_abs2

    print(f"Selected error samples: {len(selected_err_abs)} (target {target_err_n})")

    # Combine
    combined_indices = base_indices + selected_err_abs
    combined_indices = list(dict.fromkeys(map(int, combined_indices)))  # preserve order
    out_k = len(combined_indices)
    print(f"Combined train set size: {out_k} (= {len(base_indices)} + {len(selected_err_abs)})")

    out_train = [train[i] for i in combined_indices]

    out_jsonl = DATA_DIR / f"location_train_cluster{args.kmeans_k}_err{len(selected_err_abs)}.jsonl"
    save_jsonl(out_train, out_jsonl)
    print(f"Wrote: {out_jsonl}")

    # Update sample_config
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    task_name = f"musr_location_cluster{args.kmeans_k}_err{len(selected_err_abs)}"
    cfg[task_name] = {
        "train_data": str(out_jsonl),
        "val_data": "./eval/musr_location/data/location_val.jsonl",
        "test_data": "./eval/musr_location/data/location_test.jsonl",
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    print(f"Updated {CONFIG_PATH} with task {task_name}")

    # Save meta
    meta_out = {
        "task_name": task_name,
        "base_method": f"kmeans{args.kmeans_k}",
        "base_indices": base_indices,
        "pool_size": len(pool_indices),
        "pool_accuracy": test_results.get("accuracy"),
        "wrong_abs_indices": wrong_abs_indices,
        "error_clusters": args.error_clusters,
        "per_error_cluster": args.per_error_cluster,
        "far_quantile": args.far_quantile,
        "selected_error_indices": selected_err_abs,
        "combined_indices": combined_indices,
        "playbook_path": str(pb_path),
    }
    meta_path_out = DATA_DIR / f"cluster{args.kmeans_k}_err{len(selected_err_abs)}_meta.json"
    meta_path_out.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Wrote meta: {meta_path_out}")

    print("\nNext:")
    print(f"  python -m eval.musr_location.run --task_name {task_name} --mode offline --eval_steps {out_k} --skip_initial_test --save_path results/{task_name}")
    print(f"  # then eval_only on test with best_playbook")


if __name__ == "__main__":
    main()

