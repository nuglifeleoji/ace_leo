#!/usr/bin/env python3
"""
Plot Mind2Web experiment results: Random vs Cluster comparison.
Generates val and test accuracy comparison charts.
"""
import json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

BASE = Path("results")
OUT  = Path("results/mind2web_comparison.png")

# ── Data collection ──────────────────────────────────────────────────────────

def load(path):
    try:
        d = json.load(open(path))
        if "training_results" in d:
            return d["training_results"].get("best_validation_accuracy"), "val"
        if "test_results" in d:
            return d["test_results"].get("accuracy"), "test"
    except:
        pass
    return None, None

data = defaultdict(lambda: {"val": [], "test": []})  # key=(k, method)

def seed_method(p):
    m = re.search(r"seed(\d+)", p)
    return f"random_s{m.group(1)}" if m else None

def collect_k(glob_pat, k, cluster_subfolder_re):
    for p in sorted(BASE.glob(glob_pat)):
        acc, split = load(p)
        if acc is None:
            continue
        # Use only the immediate subfolder name (e.g. "cluster10", "random10_seed0")
        parts = p.parts  # e.g. ('results','mind2web_cluster10_vs_random','cluster10','ace_run_...','final_results.json')
        subfolder = parts[2] if len(parts) > 2 else ""
        if re.fullmatch(cluster_subfolder_re, subfolder):
            data[(k, "cluster")][split].append(acc)
        elif re.search(r"seed(\d+)", subfolder):
            method = seed_method(subfolder)
            if method:
                data[(k, method)][split].append(acc)

# k=10  (subfolder names: "cluster10", "cluster10_test")
collect_k("mind2web_cluster10_vs_random/**/final_results.json", 10, r"cluster10(_test)?")
# k=15
collect_k("mind2web_cluster15_vs_random15/**/final_results.json", 15, r"cluster15(_test)?")
# k=20
collect_k("mind2web_cluster20_vs_random20/**/final_results.json", 20, r"cluster20(_test)?")

# random20 seed4-7 (extra)
for p in sorted(BASE.glob("mind2web_random_more/**/final_results.json")):
    acc, split = load(p)
    if acc is None: continue
    m = re.search(r"random(\d+)_seed(\d+)", str(p))
    if m:
        k, s = int(m.group(1)), int(m.group(2))
        data[(k, f"random_s{s}")][split].append(acc)

# Deduplicate (keep unique values only)
for key in data:
    for split in ("val", "test"):
        data[key][split] = list(dict.fromkeys(data[key][split]))

# ── Build plot data ───────────────────────────────────────────────────────────

ks = [10, 15, 20]

def get_random_scores(k, split):
    scores = []
    for (kk, method), v in data.items():
        if kk == k and method.startswith("random_"):
            if v[split]:
                scores.extend(v[split])
    return scores

def get_cluster_score(k, split):
    v = data.get((k, "cluster"), {})
    vals = v.get(split, [])
    return vals[0] if vals else None

# ── Plotting ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
fig.suptitle("Mind2Web: Random Sampling vs Vanilla K-means Clustering",
             fontsize=14, fontweight="bold", y=1.01)

RAND_COLOR    = "#4C72B0"
CLUSTER_COLOR = "#DD8452"
MEAN_COLOR    = "#2ecc71"

for ax, split in zip(axes, ["val", "test"]):
    rand_means, rand_stds = [], []
    cluster_vals          = []
    all_randoms           = []   # list of lists

    for k in ks:
        r = get_random_scores(k, split)
        c = get_cluster_score(k, split)
        rand_means.append(np.mean(r) if r else np.nan)
        rand_stds.append(np.std(r)   if r else np.nan)
        cluster_vals.append(c if c is not None else np.nan)
        all_randoms.append(r)

    x = np.array(ks, dtype=float)

    # ── scatter random seeds ──────────────────────────────────────────────────
    for i, (k, scores) in enumerate(zip(ks, all_randoms)):
        jitter = np.random.default_rng(42).uniform(-0.4, 0.4, len(scores))
        xs = np.full(len(scores), k) + jitter
        ax.scatter(xs, scores, color=RAND_COLOR, alpha=0.55, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5)

    # ── mean of random ────────────────────────────────────────────────────────
    valid = [(k, m, s) for k, m, s in zip(ks, rand_means, rand_stds) if not np.isnan(m)]
    if valid:
        kv, mv, sv = zip(*valid)
        ax.plot(kv, mv, "o--", color=RAND_COLOR, linewidth=2, markersize=7,
                zorder=4, label="Random (mean ± std)")
        ax.fill_between(kv,
                        [m - s for m, s in zip(mv, sv)],
                        [m + s for m, s in zip(mv, sv)],
                        alpha=0.15, color=RAND_COLOR)

    # ── cluster line ─────────────────────────────────────────────────────────
    valid_c = [(k, c) for k, c in zip(ks, cluster_vals) if not np.isnan(c)]
    if valid_c:
        kc, cc = zip(*valid_c)
        ax.plot(kc, cc, "s-", color=CLUSTER_COLOR, linewidth=2.5, markersize=9,
                zorder=5, label="K-means Cluster")
        for k_v, c_v in zip(kc, cc):
            ax.annotate(f"{c_v:.3f}", (k_v, c_v),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=CLUSTER_COLOR, fontweight="bold")

    # ── annotate random mean ──────────────────────────────────────────────────
    if valid:
        for k_v, m_v in zip(kv, mv):
            ax.annotate(f"{m_v:.3f}", (k_v, m_v),
                        textcoords="offset points", xytext=(-28, 4),
                        fontsize=8, color=RAND_COLOR, fontweight="bold")

    ax.set_title(f"{'Validation' if split == 'val' else 'Test'} Accuracy",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Set Size (k)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"k={k}" for k in ks], fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.legend(fontsize=9, loc="lower right")

    # y-axis range: tight around data
    all_vals = [v for scores in all_randoms for v in scores] + \
               [c for c in cluster_vals if not np.isnan(c)]
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        pad = (ymax - ymin) * 0.3
        ax.set_ylim(ymin - pad, ymax + pad)

# Add note about val result availability
axes[0].text(0.02, 0.02,
             "Note: val = best val acc during training\n(not all seeds have val logged)",
             transform=axes[0].transAxes, fontsize=7.5, color="gray",
             verticalalignment="bottom")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'k':>4}  {'method':20s}  {'val':>8}  {'test':>8}")
print("="*65)
for k in ks:
    c_val  = get_cluster_score(k, "val")
    c_test = get_cluster_score(k, "test")
    cv_str = f"{c_val:.4f}" if c_val is not None else "N/A"
    ct_str = f"{c_test:.4f}" if c_test is not None else "N/A"
    print(f"{k:>4}  {'cluster':20s}  {cv_str:>8}  {ct_str:>8}")
    r_scores_val  = get_random_scores(k, "val")
    r_scores_test = get_random_scores(k, "test")
    for (kk, method), v in sorted(data.items()):
        if kk != k or not method.startswith("random_"):
            continue
        rv = v["val"][0]  if v["val"]  else None
        rt = v["test"][0] if v["test"] else None
        rv_s = f"{rv:.4f}" if rv is not None else "N/A"
        rt_s = f"{rt:.4f}" if rt is not None else "N/A"
        print(f"{k:>4}  {method:20s}  {rv_s:>8}  {rt_s:>8}")
    if r_scores_test:
        rmv = f"{np.mean(r_scores_val):.4f}" if r_scores_val else "N/A"
        print(f"{k:>4}  {'random MEAN':20s}  {rmv:>8}  {np.mean(r_scores_test):.4f}")
    print("-"*65)
