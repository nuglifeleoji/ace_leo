#!/usr/bin/env python3
"""
Plot Mind2Web: Lesson Clustering vs Vanilla Clustering vs Random (scatter).
Generates a side-by-side val + test accuracy comparison chart.
"""
import json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

BASE = Path("results")
OUT  = Path("results/mind2web_lesson_vs_cluster.png")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_acc(path, split):
    try:
        d = json.load(open(path))
        if split == "val" and "training_results" in d:
            return d["training_results"].get("best_validation_accuracy")
        if split == "test" and "test_results" in d:
            return d["test_results"].get("accuracy")
    except:
        pass
    return None

# ── Vanilla Cluster ───────────────────────────────────────────────────────────

vanilla_test = {}
vanilla_val  = {}

# k=10,15,20 from vs_random experiment folders
for k, folder, subfolder in [
    (10, "mind2web_cluster10_vs_random",    "cluster10"),
    (15, "mind2web_cluster15_vs_random15",  "cluster15"),
    (20, "mind2web_cluster20_vs_random20",  "cluster20"),
]:
    vp = sorted(BASE.glob(f"{folder}/{subfolder}/ace_run_*_offline/final_results.json"))
    if vp:
        v = load_acc(vp[-1], "val")
        if v is not None:
            vanilla_val[k] = v
    tp = sorted(BASE.glob(f"{folder}/{subfolder}_test/ace_run_*_eval_only/final_results.json"))
    if tp:
        t = load_acc(tp[-1], "test")
        if t is not None:
            vanilla_test[k] = t

# k=30,40,50 from standalone folders (backfill)
for k in [30, 40, 50]:
    # val: from training run
    vp = sorted(BASE.glob(f"mind2web_cluster{k}/ace_run_*_offline/final_results.json"))
    if vp:
        v = load_acc(vp[-1], "val")
        if v is not None:
            vanilla_val[k] = v
    # test: from test eval folder
    tp = sorted(BASE.glob(f"mind2web_cluster{k}_test/ace_run_*_eval_only/final_results.json"))
    if tp:
        t = load_acc(tp[-1], "test")
        if t is not None:
            vanilla_test[k] = t

print("Vanilla cluster test:", vanilla_test)
print("Vanilla cluster val:", vanilla_val)

# ── Lesson Cluster ────────────────────────────────────────────────────────────

lesson_test = {}
lesson_val  = {}

for k in [10, 15, 20, 30, 40, 50]:
    # val: from training final_results
    # k=20 prefer rerun (seed=123), k=10 use retrained version
    val_dirs = [f"mind2web_cluster{k}_lesson"]
    if k == 20:
        val_dirs = ["mind2web_cluster20_lesson_rerun", "mind2web_cluster20_lesson"]
    for sfx in val_dirs:
        vp = sorted(BASE.glob(f"mind2web_lesson_cluster/{sfx}/ace_run_*_offline/final_results.json"))
        if vp:
            v = load_acc(vp[-1], "val")
            if v is not None:
                lesson_val[k] = v
                break

    # test: k=20 prefer rerun; k=10 prefer test2
    if k == 20:
        test_dirs = ["mind2web_cluster20_lesson_rerun_test", "mind2web_cluster20_lesson_test"]
    elif k == 10:
        test_dirs = ["mind2web_cluster10_lesson_test2", "mind2web_cluster10_lesson_test"]
    else:
        test_dirs = [f"mind2web_cluster{k}_lesson_test"]
    for td in test_dirs:
        tp = sorted(BASE.glob(f"mind2web_lesson_cluster/{td}/ace_run_*_eval_only/final_results.json"))
        if tp:
            t = load_acc(tp[-1], "test")
            if t is not None:
                lesson_test[k] = t
                break

print("Lesson cluster test:", lesson_test)
print("Lesson cluster val:", lesson_val)

# ── Random seeds (scatter reference) ─────────────────────────────────────────

random_test = defaultdict(list)
random_val  = defaultdict(list)

# From vs_random folders
for folder, k in [
    ("mind2web_cluster10_vs_random",   10),
    ("mind2web_cluster15_vs_random15", 15),
    ("mind2web_cluster20_vs_random20", 20),
]:
    for p in sorted(BASE.glob(f"{folder}/random{k}_seed*/ace_run_*_offline/final_results.json")):
        v = load_acc(p, "val")
        if v is not None:
            random_val[k].append(v)
    for p in sorted(BASE.glob(f"{folder}/random{k}_seed*_test/ace_run_*_eval_only/final_results.json")):
        t = load_acc(p, "test")
        if t is not None:
            random_test[k].append(t)

# From random_more folder
for p in sorted(BASE.glob("mind2web_random_more/**/final_results.json")):
    m = re.search(r"random(\d+)_seed(\d+)", str(p))
    if not m: continue
    k = int(m.group(1))
    if "eval_only" in str(p):
        t = load_acc(p, "test")
        if t is not None:
            random_test[k].append(t)
    elif "offline" in str(p):
        v = load_acc(p, "val")
        if v is not None:
            random_val[k].append(v)

print("Random test:", {k: v for k, v in sorted(random_test.items())})

# ── Plot ──────────────────────────────────────────────────────────────────────

VANILLA_COLOR = "#DD8452"   # orange
LESSON_COLOR  = "#2ecc71"   # green
RANDOM_COLOR  = "#4C72B0"   # blue

ks_lesson  = sorted(lesson_test.keys())
ks_vanilla = sorted(vanilla_test.keys())
all_ks     = sorted(set(ks_lesson) | set(ks_vanilla))

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle("Mind2Web: Lesson Clustering vs Vanilla K-means vs Random",
             fontsize=14, fontweight="bold", y=1.01)

for ax, (split, v_dict, l_dict) in zip(axes, [
    ("Test",       vanilla_test, lesson_test),
    ("Validation", vanilla_val,  lesson_val),
]):
    # ── Vanilla cluster line ──────────────────────────────────────────────────
    vc_ks = [k for k in ks_vanilla if k in v_dict]
    vc_vs = [v_dict[k] for k in vc_ks]
    if vc_ks:
        ax.plot(vc_ks, vc_vs, "s-", color=VANILLA_COLOR, linewidth=2.5,
                markersize=9, zorder=5, label="Vanilla K-means")
        for kv, vv in zip(vc_ks, vc_vs):
            ax.annotate(f"{vv:.3f}", (kv, vv),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=VANILLA_COLOR, fontweight="bold")

    # ── Lesson cluster line ───────────────────────────────────────────────────
    lc_ks = [k for k in ks_lesson if k in l_dict]
    lc_vs = [l_dict[k] for k in lc_ks]
    if lc_ks:
        ax.plot(lc_ks, lc_vs, "D-", color=LESSON_COLOR, linewidth=2.5,
                markersize=8, zorder=5, label="Lesson Clustering")
        for kv, lv in zip(lc_ks, lc_vs):
            ax.annotate(f"{lv:.3f}", (kv, lv),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color=LESSON_COLOR, fontweight="bold")

    ax.set_title(f"{split} Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Set Size (k)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks], fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.legend(fontsize=9, loc="lower right")

    # y-axis range
    all_vals = vc_vs + lc_vs
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        pad = (ymax - ymin) * 0.35
        ax.set_ylim(ymin - pad, ymax + pad)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"  Mind2Web — Lesson Cluster vs Vanilla Cluster (Test Accuracy)")
print("="*65)
print(f"{'k':>4}  {'Lesson':>10}  {'Vanilla':>10}  {'Random mean':>12}")
print("-"*65)
for k in all_ks:
    lv = f"{lesson_test[k]:.4f}"  if k in lesson_test  else "  N/A  "
    vv = f"{vanilla_test[k]:.4f}" if k in vanilla_test else "  N/A  "
    rs = random_test.get(k, [])
    rv = f"{np.mean(rs):.4f}" if rs else "  N/A  "
    print(f"{k:>4}  {lv:>10}  {vv:>10}  {rv:>12}")
print("="*65)
