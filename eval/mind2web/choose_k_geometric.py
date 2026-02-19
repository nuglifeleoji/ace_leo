#!/usr/bin/env python3
"""
Pick an (approximately) optimal K for K-means using *geometric* criteria only.

This script ignores downstream task performance and uses only embedding-space
signals:
  - Inertia (SSE) vs K and an elbow/knee estimate (max distance to chord)
  - Silhouette score (computed on a random sample for speed)
  - Clustering stability across random seeds (NMI/ARI)

Outputs:
  - A JSON report with all metrics
  - A PNG/PDF plot for quick inspection

Example:
  python -m eval.mind2web.choose_k_geometric \\
    --embeddings eval/mind2web/data/embeddings.npy \\
    --ks 10 15 20 30 50 80 120 200 \\
    --silhouette_sample 2000 \\
    --stability_seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class KMetrics:
    k: int
    inertia: float
    silhouette: float | None
    stability_nmi: float | None
    stability_ari: float | None


def _max_distance_to_chord(xs: np.ndarray, ys: np.ndarray) -> int:
    """
    Knee/elbow by maximum perpendicular distance to the line connecting endpoints.
    Returns the index of the knee point.
    """
    if len(xs) < 3:
        return int(np.argmax(ys))

    # normalize to [0,1] for numerical stability
    x = (xs - xs.min()) / (xs.max() - xs.min() + 1e-12)
    y = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    v = p2 - p1
    v_norm = np.linalg.norm(v) + 1e-12

    # distance from point to line
    dists = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        # area of parallelogram / base length
        dist = abs(np.cross(v, p - p1)) / v_norm
        dists.append(dist)
    return int(np.argmax(dists))


def _fit_kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    use_minibatch: bool,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    if use_minibatch:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=batch_size,
            n_init="auto",
            max_iter=300,
            reassignment_ratio=0.01,
        )
    else:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)

    km.fit(X)
    return km.labels_.copy(), float(km.inertia_)


def _silhouette_on_sample(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> float:
    from sklearn.metrics import silhouette_score

    n = len(X)
    if n <= 2:
        return float("nan")
    if len(set(labels)) < 2:
        return float("nan")

    if sample_size <= 0 or sample_size >= n:
        return float(silhouette_score(X, labels, metric="euclidean"))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    return float(silhouette_score(X[idx], labels[idx], metric="euclidean"))


def _stability(labels_by_seed: List[np.ndarray]) -> Tuple[float, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if len(labels_by_seed) < 2:
        return float("nan"), float("nan")

    nmis, aris = [], []
    for a, b in combinations(labels_by_seed, 2):
        nmis.append(normalized_mutual_info_score(a, b))
        aris.append(adjusted_rand_score(a, b))
    return float(np.mean(nmis)), float(np.mean(aris))


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometric K selection for K-means")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .npy")
    parser.add_argument("--ks", type=int, nargs="+", required=True, help="Candidate K values")
    parser.add_argument("--use_minibatch", action="store_true", help="Use MiniBatchKMeans")
    parser.add_argument("--batch_size", type=int, default=2048, help="MiniBatchKMeans batch size")
    parser.add_argument("--primary_seed", type=int, default=42, help="Seed for primary fit per K")
    parser.add_argument(
        "--stability_seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for stability (default: 0 1 2)",
    )
    parser.add_argument(
        "--silhouette_sample",
        type=int,
        default=2000,
        help="Sample size for silhouette (0=full) (default: 2000)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional: subsample embeddings to this many points for all computations (0=all)",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="results/kmeans_geometric_k",
        help="Output prefix (default: results/kmeans_geometric_k)",
    )
    args = parser.parse_args()

    X = np.load(args.embeddings)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {X.shape}")
    n, d = X.shape
    print(f"Loaded embeddings: {X.shape} from {args.embeddings}")

    if args.max_points and args.max_points > 0 and args.max_points < n:
        rng = np.random.default_rng(args.primary_seed)
        idx = rng.choice(n, size=args.max_points, replace=False)
        X = X[idx]
        n = len(X)
        print(f"Subsampled to {n} points for all computations")

    ks = sorted(set(args.ks))
    for k in ks:
        if k < 2:
            raise ValueError(f"K must be >=2, got {k}")
        if k >= n:
            raise ValueError(f"K must be < N ({n}), got {k}")

    all_metrics: List[KMetrics] = []

    for k in ks:
        print(f"\n=== K={k} ===")

        # Primary fit for inertia + silhouette
        labels, inertia = _fit_kmeans(
            X, k=k, seed=args.primary_seed, use_minibatch=args.use_minibatch, batch_size=args.batch_size
        )
        print(f"inertia={inertia:.4e}")

        sil = _silhouette_on_sample(X, labels, sample_size=args.silhouette_sample, seed=args.primary_seed)
        if math.isnan(sil):
            sil_out = None
            print("silhouette=nan (insufficient clusters or points)")
        else:
            sil_out = float(sil)
            print(f"silhouette={sil_out:.4f} (sample={args.silhouette_sample if args.silhouette_sample>0 else 'full'})")

        # Stability across seeds (fit on same X)
        labels_by_seed = []
        for s in args.stability_seeds:
            lb, _ = _fit_kmeans(
                X, k=k, seed=s, use_minibatch=args.use_minibatch, batch_size=args.batch_size
            )
            labels_by_seed.append(lb)
        nmi, ari = _stability(labels_by_seed)
        stab_nmi = None if math.isnan(nmi) else float(nmi)
        stab_ari = None if math.isnan(ari) else float(ari)
        if stab_nmi is not None:
            print(f"stability: NMI={stab_nmi:.3f} ARI={stab_ari:.3f} (seeds={args.stability_seeds})")

        all_metrics.append(
            KMetrics(k=k, inertia=float(inertia), silhouette=sil_out, stability_nmi=stab_nmi, stability_ari=stab_ari)
        )

    # Knee on log inertia
    xs = np.array([m.k for m in all_metrics], dtype=float)
    ys = np.log(np.array([m.inertia for m in all_metrics], dtype=float) + 1e-12)
    knee_idx = _max_distance_to_chord(xs, ys)
    knee_k = int(all_metrics[knee_idx].k)

    # Build report
    report = {
        "embeddings": args.embeddings,
        "n_points": n,
        "dim": d,
        "ks": ks,
        "primary_seed": args.primary_seed,
        "stability_seeds": args.stability_seeds,
        "use_minibatch": bool(args.use_minibatch),
        "batch_size": args.batch_size,
        "silhouette_sample": args.silhouette_sample,
        "metrics": [
            {
                "k": m.k,
                "inertia": m.inertia,
                "log_inertia": float(np.log(m.inertia + 1e-12)),
                "silhouette": m.silhouette,
                "stability_nmi": m.stability_nmi,
                "stability_ari": m.stability_ari,
            }
            for m in all_metrics
        ],
        "recommendation": {
            "knee_k_log_inertia": knee_k,
            "note": "This is a geometric elbow estimate only; validate with downstream performance.",
        },
    }

    out_json = args.out_prefix + ".json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report -> {out_json}")
    print(f"Recommended K (knee on log inertia): {knee_k}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # inertia
    axes[0].plot(xs, [m.inertia for m in all_metrics], "o-", color="#1565C0")
    axes[0].set_title("Inertia (SSE) vs K")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].axvline(knee_k, color="#E65100", linestyle="--", linewidth=2, label=f"knee={knee_k}")
    axes[0].legend()

    # silhouette
    sils = [m.silhouette if m.silhouette is not None else np.nan for m in all_metrics]
    axes[1].plot(xs, sils, "s--", color="#7B1FA2")
    axes[1].set_title("Silhouette vs K (sampled)")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].axvline(knee_k, color="#E65100", linestyle="--", linewidth=2)

    # stability
    nmis = [m.stability_nmi if m.stability_nmi is not None else np.nan for m in all_metrics]
    aris = [m.stability_ari if m.stability_ari is not None else np.nan for m in all_metrics]
    axes[2].plot(xs, nmis, "^-", color="#00897B", label="NMI")
    axes[2].plot(xs, aris, "v-", color="#C62828", label="ARI")
    axes[2].set_title("Stability across seeds")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Score")
    axes[2].axvline(knee_k, color="#E65100", linestyle="--", linewidth=2)
    axes[2].legend()

    plt.tight_layout()
    out_png = args.out_prefix + ".png"
    out_pdf = args.out_prefix + ".pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot -> {out_png} / {out_pdf}")


if __name__ == "__main__":
    main()

