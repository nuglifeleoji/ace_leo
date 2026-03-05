"""
Plot FiNer curriculum selection comparison:
  Semantic (V2), Lesson-based, Random sampling
  vs Baseline LLM and Full Train ACE.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── result paths ──────────────────────────────────────────────
RESULTS_DIR = "/workspace/ace_leo/results"
K_VALUES    = [5, 10, 20, 30, 40, 50, 80]

def load_acc(path):
    """Return accuracy*100 if final_results.json exists, else None."""
    try:
        f = None
        for root, _, files in os.walk(path):
            if "final_results.json" in files:
                f = os.path.join(root, "final_results.json")
                break
        if f is None:
            return None
        d = json.load(open(f))
        return round(d["test_results"]["accuracy"] * 100, 1)
    except Exception:
        return None

def gather(series_prefix):
    xs, ys = [], []
    for k in K_VALUES:
        acc = load_acc(f"{RESULTS_DIR}/{series_prefix}_{k}_test")
        if acc is not None:
            xs.append(k)
            ys.append(acc)
    return xs, ys

# ── gather data ───────────────────────────────────────────────
xs_sem,  ys_sem  = gather("finer_cluster_v2")
xs_les,  ys_les  = gather("finer_lesson")
xs_rnd,  ys_rnd  = gather("finer_random")

# Full train ACE (1000 samples)
full_train_acc = load_acc(f"{RESULTS_DIR}/finer_cluster_v2")
BASELINE = 70.7   # LLM without playbook

print("Semantic:", list(zip(xs_sem, ys_sem)))
print("Lesson  :", list(zip(xs_les, ys_les)))
print("Random  :", list(zip(xs_rnd, ys_rnd)))
print(f"Full Train: {full_train_acc}")

# ── plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# colour / style
C_SEM = "#2563EB"   # blue
C_LES = "#16A34A"   # green
C_RND = "#D97706"   # amber

# three curves
ax.plot(xs_sem, ys_sem, "o-",  color=C_SEM, lw=2,   ms=7,  label="Semantic Clustering (V2)")
ax.plot(xs_les, ys_les, "s--", color=C_LES, lw=2,   ms=7,  label="Lesson-based Clustering")
ax.plot(xs_rnd, ys_rnd, "^:", color=C_RND, lw=2,   ms=7,  label="Random Sampling")

# baseline reference line
ax.axhline(BASELINE, color="gray", lw=1.5, ls="--", label=f"Baseline LLM ({BASELINE}%)")

# full train reference line (only if result is available)
if full_train_acc is not None:
    ax.axhline(full_train_acc, color="#DC2626", lw=1.5, ls="-.",
               label=f"ACE Full Train / 1000 ({full_train_acc}%)")
else:
    ax.axhline(78.3, color="#DC2626", lw=1.5, ls="-.",
               label="ACE Full Train / 1000 (78.3%)")

# annotate peaks
for xs, ys, c in [(xs_sem, ys_sem, C_SEM),
                  (xs_les, ys_les, C_LES),
                  (xs_rnd, ys_rnd, C_RND)]:
    if ys:
        best_idx = int(np.argmax(ys))
        ax.annotate(f"{ys[best_idx]}%",
                    xy=(xs[best_idx], ys[best_idx]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=c, fontweight="bold")

ax.set_xlabel("Number of training examples (K)", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("FiNer: Curriculum Selection Methods Comparison", fontsize=13, fontweight="bold")
ax.set_xticks(K_VALUES)
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.set_ylim(68, max(max(ys_sem or [0]), max(ys_les or [0]), max(ys_rnd or [0]), 78.3) + 2)
ax.legend(fontsize=9.5, loc="lower right")
ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout()

out = f"{RESULTS_DIR}/finer_comparison_all.png"
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
