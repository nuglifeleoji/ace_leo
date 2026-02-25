#!/usr/bin/env python3
"""
Lesson-based K-means Clustering & Selection for Mind2Web.

Uses pre-computed lesson embeddings (from cluster_train_lesson.py) to cluster
training samples by the *strategy they teach*, not by task description.

Usage:
    python -m eval.mind2web.cluster_lesson_select --clusters 10 15 20 30
    python -m eval.mind2web.cluster_lesson_select --clusters 20 --no_dedup
"""
import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_PATH      = "./eval/mind2web/data/mind2web_train.jsonl"
LESSON_CACHE    = "./eval/mind2web/data/lesson_cache.json"
LESSON_EMB_PATH = "./eval/mind2web/data/lesson_embeddings.npy"
OUTPUT_DIR      = "./eval/mind2web/data"
CONFIG_PATH     = "./eval/mind2web/data/sample_config.json"

DEFAULT_CLUSTERS = [10, 15, 20, 30]


# ── IO ────────────────────────────────────────────────────────────────────────

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


def load_lessons(n: int) -> List[str]:
    if not os.path.exists(LESSON_CACHE):
        raise FileNotFoundError(f"Lesson cache not found: {LESSON_CACHE}\n"
                                f"Run: python -m eval.mind2web.cluster_train_lesson")
    with open(LESSON_CACHE, "r") as f:
        raw = json.load(f)
    lessons = []
    for i in range(n):
        lessons.append(raw.get(str(i), f"(missing lesson for sample {i})"))
    return lessons


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_and_select(
    embeddings: np.ndarray,
    train_data: List[Dict],
    k: int,
    seed: int = 42,
    dedup_website: bool = True,
) -> Tuple[List[int], np.ndarray]:
    """
    K-means on lesson embeddings (L2-normalized), select centroid-nearest sample
    per cluster. Optionally deduplicates by website.
    """
    from sklearn.cluster import KMeans

    # L2-normalize → cosine-like clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_n = embeddings / norms

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(emb_n)
    centroids = kmeans.cluster_centers_

    # Rank candidates within each cluster by distance to centroid
    cluster_candidates: Dict[int, List[Tuple[float, int]]] = {}
    for cid in range(k):
        idx_arr = np.where(labels == cid)[0]
        if len(idx_arr) == 0:
            continue
        dists = np.linalg.norm(emb_n[idx_arr] - centroids[cid], axis=1)
        sorted_pairs = sorted(zip(dists.tolist(), idx_arr.tolist()))
        cluster_candidates[cid] = sorted_pairs

    if not dedup_website:
        return [pairs[0][1] for pairs in cluster_candidates.values()], labels

    # Greedy website-dedup: best clusters claim their preferred website first
    sorted_clusters = sorted(cluster_candidates.items(),
                             key=lambda kv: kv[1][0][0])
    used_websites: set = set()
    selected_by_cid: Dict[int, int] = {}

    for cid, candidates in sorted_clusters:
        chosen = None
        for _dist, idx in candidates:
            site = train_data[idx].get("website", "__unknown__")
            if site not in used_websites:
                chosen = idx
                used_websites.add(site)
                break
        if chosen is None:
            chosen = candidates[0][1]
            print(f"      [dedup] cluster {cid}: no unique website, "
                  f"fallback → {train_data[chosen].get('website','?')}")
        selected_by_cid[cid] = chosen

    return [selected_by_cid[cid] for cid in sorted(selected_by_cid)], labels


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_selection(
    train_data: List[Dict],
    lessons: List[str],
    selected: List[int],
    labels: np.ndarray,
    k: int,
    dedup: bool,
):
    csizes = np.bincount(labels)
    print(f"    Cluster sizes : min={csizes.min()}, max={csizes.max()}, "
          f"mean={csizes.mean():.1f}")

    domains = Counter(train_data[i].get("domain", "?") for i in selected)
    ops     = Counter(str(train_data[i].get("operation", {}).get("op", "?"))
                      for i in selected)
    sites   = [train_data[i].get("website", "?") for i in selected]
    n_uniq  = len(set(sites))
    dup_sites = {w: c for w, c in Counter(sites).items() if c > 1}

    positions = []
    for i in selected:
        d = train_data[i]
        step  = d.get("step_idx", 0)
        total = d.get("total_steps", 1)
        positions.append(step / max(total - 1, 1))
    avg_pos = float(np.mean(positions))
    early = sum(1 for p in positions if p < 0.33)
    mid   = sum(1 for p in positions if 0.33 <= p < 0.67)
    late  = sum(1 for p in positions if p >= 0.67)
    avg_len = float(np.mean([train_data[i].get("total_steps", 1) for i in selected]))

    print(f"    Domains       : {dict(domains)}")
    print(f"    Operations    : {dict(ops)}")
    dedup_str = " (dedup ON)" if dedup else " (dedup OFF)"
    print(f"    Unique websites: {n_uniq}/{k}{dedup_str}"
          + (f"  dups={dup_sites}" if dup_sites else "  ✓"))
    print(f"    Step positions : avg={avg_pos:.2f}  early={early}  mid={mid}  late={late}")
    print(f"    Task length    : avg={avg_len:.1f}")
    print()
    print(f"    {'#':>3}  {'website':20s}  {'op':6s}  {'step':9s}  lesson")
    print(f"    {'─'*90}")
    for rank, i in enumerate(selected, 1):
        d     = train_data[i]
        step  = d.get("step_idx", 0)
        total = d.get("total_steps", 1)
        pos   = step / max(total - 1, 1)
        pos_l = "E" if pos < 0.33 else ("M" if pos < 0.67 else "L")
        op    = str(d.get("operation", {}).get("op", "?"))
        site  = d.get("website", "?")
        lesson = lessons[i][:55] if i < len(lessons) else ""
        print(f"    {rank:>3}  {site:20s}  {op:6s}  s{step+1}/{total}({pos_l})    {lesson}")


# ── Config Update ─────────────────────────────────────────────────────────────

def update_config(ks: List[int], suffix: str = ""):
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    for k in ks:
        name = f"mind2web_cluster{k}_lesson{suffix}"
        config[name] = {
            "train_data": f"./eval/mind2web/data/mind2web_train_cluster{k}_lesson{suffix}.jsonl",
            "val_data":   "./eval/mind2web/data/mind2web_val.jsonl",
            "test_data":  "./eval/mind2web/data/mind2web_test.jsonl",
        }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lesson-based K-means selection for Mind2Web"
    )
    parser.add_argument("--clusters", type=int, nargs="+", default=DEFAULT_CLUSTERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to output filenames and task names (e.g. '_rerun')")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable website-level deduplication (default: ON)")
    args = parser.parse_args()
    dedup = not args.no_dedup

    # Load
    if not os.path.exists(LESSON_EMB_PATH):
        raise FileNotFoundError(
            f"Lesson embeddings not found: {LESSON_EMB_PATH}\n"
            "Run: python -m eval.mind2web.cluster_train_lesson"
        )
    embeddings = np.load(LESSON_EMB_PATH)
    train_data = load_jsonl(TRAIN_PATH)
    lessons    = load_lessons(len(train_data))

    assert embeddings.shape[0] == len(train_data), \
        f"Mismatch: {embeddings.shape[0]} embeddings vs {len(train_data)} samples"

    print(f"Loaded {len(train_data)} samples, embeddings {embeddings.shape}, "
          f"lessons {len(lessons)}")

    for k in args.clusters:
        print()
        print("=" * 60)
        print(f"  K={k}  lesson-based clustering  (dedup={'ON' if dedup else 'OFF'})")
        print("=" * 60)

        selected, labels = cluster_and_select(
            embeddings, train_data, k, seed=args.seed, dedup_website=dedup
        )
        report_selection(train_data, lessons, selected, labels, k, dedup)

        # Save subset
        subset   = [train_data[i] for i in selected]
        out_path = os.path.join(OUTPUT_DIR, f"mind2web_train_cluster{k}_lesson{args.suffix}.jsonl")
        save_jsonl(subset, out_path)

        # Save metadata
        meta = {
            "method": "lesson_kmeans_dedup" if dedup else "lesson_kmeans",
            "k": k, "seed": args.seed, "suffix": args.suffix, "dedup_website": dedup,
            "selected_indices": selected,
            "lessons": [lessons[i] for i in selected],
            "n_total": len(train_data),
        }
        with open(os.path.join(OUTPUT_DIR, f"cluster{k}_lesson{args.suffix}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    update_config(args.clusters, suffix=args.suffix)

    print()
    print("=" * 60)
    print(f"  DONE — {len(args.clusters)} lesson-based subsets generated")
    print("=" * 60)
    for k in args.clusters:
        print(f"    mind2web_cluster{k}_lesson{args.suffix}  →  "
              f"eval/mind2web/data/mind2web_train_cluster{k}_lesson{args.suffix}.jsonl")


if __name__ == "__main__":
    main()
