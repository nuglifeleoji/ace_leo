#!/usr/bin/env python3
"""
Correctness-Augmented Lesson Clustering for Mind2Web.

Combines lesson embeddings (what a sample *teaches*) with base-LLM correctness
labels (what the model *doesn't know yet*) to select training samples that:
  1. Cover diverse strategies  (lesson embedding)
  2. Prefer examples the model currently fails on  (correctness weight)

Concretely:
    augmented_emb[i] = [normalize(lesson_emb[i]),  correctness_weight * (1 - correct[i])]

Setting correctness_weight > 0 pushes "wrong" samples to be preferred within
each semantic cluster.  correctness_weight=0 degenerates to plain lesson clustering.

Prerequisites:
    python -m eval.mind2web.cluster_train_lesson        # generates lesson_embeddings.npy
    python -m eval.mind2web.eval_train_correctness      # generates train_correctness.json

Usage:
    python -m eval.mind2web.cluster_correctness_select --clusters 10 15 20
    python -m eval.mind2web.cluster_correctness_select --clusters 15 --weight 20
    python -m eval.mind2web.cluster_correctness_select --clusters 15 --no_dedup
"""
import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_PATH        = "./eval/mind2web/data/mind2web_train.jsonl"
LESSON_CACHE      = "./eval/mind2web/data/lesson_cache.json"
LESSON_EMB_PATH   = "./eval/mind2web/data/lesson_embeddings.npy"
CORRECT_CACHE     = "./eval/mind2web/data/train_correctness.json"
OUTPUT_DIR        = "./eval/mind2web/data"
CONFIG_PATH       = "./eval/mind2web/data/sample_config.json"

DEFAULT_CLUSTERS  = [10, 15, 20, 30]
DEFAULT_WEIGHT    = 15.0   # scale for correctness dimension


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
        raise FileNotFoundError(f"Lesson cache not found: {LESSON_CACHE}")
    with open(LESSON_CACHE, "r") as f:
        raw = json.load(f)
    return [raw.get(str(i), f"(missing lesson {i})") for i in range(n)]


def load_correctness(n: int) -> np.ndarray:
    """Load correctness labels.  Missing entries default to 0 (assumed wrong)."""
    if not os.path.exists(CORRECT_CACHE):
        raise FileNotFoundError(
            f"Correctness cache not found: {CORRECT_CACHE}\n"
            "Run: python -m eval.mind2web.eval_train_correctness"
        )
    with open(CORRECT_CACHE, "r") as f:
        raw = json.load(f)
    labels = np.array([int(raw.get(str(i), 0)) for i in range(n)], dtype=np.float32)
    n_correct = int(labels.sum())
    print(f"  Correctness labels: {n_correct}/{n} correct "
          f"({n_correct/n*100:.1f}%)  missing={n - len(raw)}")
    return labels


# ── Clustering ────────────────────────────────────────────────────────────────

def build_augmented_embeddings(
    lesson_emb: np.ndarray,
    correct: np.ndarray,
    weight: float,
) -> np.ndarray:
    """
    Concatenate L2-normalised lesson embeddings with a weighted
    'difficulty' signal:  (1 - correct) → 1 if model got it wrong, 0 if right.

    augmented[i] = [lesson_emb_normalised[i], weight * (1 - correct[i])]
    """
    norms = np.linalg.norm(lesson_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_n = lesson_emb / norms                           # (N, D)
    difficulty = (1.0 - correct).reshape(-1, 1) * weight  # (N, 1)
    return np.hstack([emb_n, difficulty])                 # (N, D+1)


def cluster_and_select(
    embeddings: np.ndarray,
    train_data: List[Dict],
    k: int,
    seed: int = 42,
    dedup_website: bool = True,
) -> Tuple[List[int], np.ndarray]:
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    cluster_candidates: Dict[int, List[Tuple[float, int]]] = {}
    for cid in range(k):
        idx_arr = np.where(labels == cid)[0]
        if len(idx_arr) == 0:
            continue
        dists = np.linalg.norm(embeddings[idx_arr] - centroids[cid], axis=1)
        cluster_candidates[cid] = sorted(zip(dists.tolist(), idx_arr.tolist()))

    if not dedup_website:
        return [pairs[0][1] for pairs in cluster_candidates.values()], labels

    # Greedy website-dedup: tightest clusters claim preferred website first
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
    correct: np.ndarray,
    selected: List[int],
    labels: np.ndarray,
    k: int,
    dedup: bool,
    weight: float,
):
    csizes = np.bincount(labels)
    print(f"    Cluster sizes : min={csizes.min()}, max={csizes.max()}, "
          f"mean={csizes.mean():.1f}")

    domains   = Counter(train_data[i].get("domain", "?") for i in selected)
    ops       = Counter(str(train_data[i].get("operation", {}).get("op", "?"))
                        for i in selected)
    sites     = [train_data[i].get("website", "?") for i in selected]
    n_uniq    = len(set(sites))
    dup_sites = {w: c for w, c in Counter(sites).items() if c > 1}

    n_wrong   = sum(1 for i in selected if correct[i] == 0)
    n_right   = sum(1 for i in selected if correct[i] == 1)

    positions = []
    for i in selected:
        d = train_data[i]
        step  = d.get("step_idx", 0)
        total = d.get("total_steps", 1)
        positions.append(step / max(total - 1, 1))
    avg_pos = float(np.mean(positions))
    early   = sum(1 for p in positions if p < 0.33)
    mid     = sum(1 for p in positions if 0.33 <= p < 0.67)
    late    = sum(1 for p in positions if p >= 0.67)
    avg_len = float(np.mean([train_data[i].get("total_steps", 1) for i in selected]))

    print(f"    Domains       : {dict(domains)}")
    print(f"    Operations    : {dict(ops)}")
    dedup_str = " (dedup ON)" if dedup else " (dedup OFF)"
    print(f"    Unique websites: {n_uniq}/{k}{dedup_str}"
          + (f"  dups={dup_sites}" if dup_sites else "  ✓"))
    print(f"    Correctness   : wrong={n_wrong}  right={n_right}  "
          f"(weight={weight})")
    print(f"    Step positions : avg={avg_pos:.2f}  "
          f"early={early}  mid={mid}  late={late}")
    print(f"    Task length    : avg={avg_len:.1f}")
    print()
    print(f"    {'#':>3}  {'website':20s}  {'op':6s}  {'step':9s}  {'C':1s}  lesson")
    print(f"    {'─'*95}")
    for rank, i in enumerate(selected, 1):
        d     = train_data[i]
        step  = d.get("step_idx", 0)
        total = d.get("total_steps", 1)
        pos   = step / max(total - 1, 1)
        pos_l = "E" if pos < 0.33 else ("M" if pos < 0.67 else "L")
        op    = str(d.get("operation", {}).get("op", "?"))
        site  = d.get("website", "?")
        c_str = "✓" if correct[i] == 1 else "✗"
        lesson = lessons[i][:50] if i < len(lessons) else ""
        print(f"    {rank:>3}  {site:20s}  {op:6s}  s{step+1}/{total}({pos_l})  "
              f"{c_str}  {lesson}")


# ── Config Update ─────────────────────────────────────────────────────────────

def update_config(ks: List[int], suffix: str = ""):
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    for k in ks:
        name = f"mind2web_cluster{k}_correctness{suffix}"
        config[name] = {
            "train_data": f"./eval/mind2web/data/mind2web_train_cluster{k}_correctness{suffix}.jsonl",
            "val_data":   "./eval/mind2web/data/mind2web_val.jsonl",
            "test_data":  "./eval/mind2web/data/mind2web_test.jsonl",
        }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {CONFIG_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Correctness-augmented lesson clustering for Mind2Web"
    )
    parser.add_argument("--clusters", type=int, nargs="+", default=DEFAULT_CLUSTERS)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--weight",   type=float, default=DEFAULT_WEIGHT,
                        help="Scale for difficulty dimension (default: 15). "
                             "0 = plain lesson clustering. Higher → stronger "
                             "preference for samples the model gets wrong.")
    parser.add_argument("--suffix",   type=str,   default="",
                        help="Suffix appended to output filenames/task names")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable website-level deduplication (default: ON)")
    args = parser.parse_args()
    dedup = not args.no_dedup

    # Load resources
    if not os.path.exists(LESSON_EMB_PATH):
        raise FileNotFoundError(
            f"Lesson embeddings not found: {LESSON_EMB_PATH}\n"
            "Run: python -m eval.mind2web.cluster_train_lesson"
        )
    lesson_emb = np.load(LESSON_EMB_PATH)
    train_data = load_jsonl(TRAIN_PATH)
    lessons    = load_lessons(len(train_data))
    correct    = load_correctness(len(train_data))

    assert lesson_emb.shape[0] == len(train_data), (
        f"Mismatch: {lesson_emb.shape[0]} embeddings vs {len(train_data)} samples"
    )

    # Build augmented embeddings once (shared across all k values)
    aug_emb = build_augmented_embeddings(lesson_emb, correct, args.weight)

    print(f"\nLoaded {len(train_data)} samples, lesson emb {lesson_emb.shape}, "
          f"aug emb {aug_emb.shape}")
    print(f"Correctness weight = {args.weight}  "
          f"(0=lesson-only, higher=harder-sample preference)")

    for k in args.clusters:
        print()
        print("=" * 65)
        print(f"  K={k}  correctness-augmented lesson clustering  "
              f"(weight={args.weight}, dedup={'ON' if dedup else 'OFF'})")
        print("=" * 65)

        selected, labels = cluster_and_select(
            aug_emb, train_data, k, seed=args.seed, dedup_website=dedup
        )
        report_selection(train_data, lessons, correct, selected, labels, k,
                         dedup, args.weight)

        # Save selected subset
        subset   = [train_data[i] for i in selected]
        out_path = os.path.join(
            OUTPUT_DIR,
            f"mind2web_train_cluster{k}_correctness{args.suffix}.jsonl"
        )
        save_jsonl(subset, out_path)

        # Save metadata
        meta = {
            "method":           "correctness_lesson_kmeans_dedup" if dedup
                                else "correctness_lesson_kmeans",
            "k":                k,
            "seed":             args.seed,
            "correctness_weight": args.weight,
            "suffix":           args.suffix,
            "dedup_website":    dedup,
            "selected_indices": selected,
            "lessons":          [lessons[i] for i in selected],
            "correctness":      [int(correct[i]) for i in selected],
            "n_total":          len(train_data),
        }
        meta_path = os.path.join(
            OUTPUT_DIR,
            f"cluster{k}_correctness{args.suffix}_meta.json"
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    update_config(args.clusters, suffix=args.suffix)

    print()
    print("=" * 65)
    print(f"  DONE — {len(args.clusters)} correctness-augmented subsets generated")
    print("=" * 65)
    for k in args.clusters:
        print(f"    mind2web_cluster{k}_correctness{args.suffix}")


if __name__ == "__main__":
    main()
