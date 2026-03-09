"""
analyze_phased.py  — Comprehensive analysis of Phased curriculum results.

Groups methods by family, computes mean±std across seeds, and outputs
a formatted leaderboard comparing phased strategies to random baseline.
"""

import os
import json
import glob
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


RESULTS_DIR = "/workspace/ace_leo/results/finer_curriculum"
LOG_DIR     = RESULTS_DIR


def parse_log(log_path: str) -> Dict:
    """Parse a run log file to extract key metrics."""
    try:
        with open(log_path) as f:
            content = f.read()
    except FileNotFoundError:
        return {}

    result = {"log_path": log_path, "name": os.path.basename(log_path)}

    # Baseline test accuracy
    m = re.search(r'\[test-baseline\] acc=([0-9.]+)', content)
    if m:
        result["baseline_acc"] = float(m.group(1))

    # Final test accuracy
    m = re.search(r'\[test-final\] acc=([0-9.]+)', content)
    if m:
        result["final_acc"] = float(m.group(1))

    if "baseline_acc" in result and "final_acc" in result:
        result["delta"] = result["final_acc"] - result["baseline_acc"]
        result["rel_delta"] = result["delta"] / result["baseline_acc"]
        result["completed"] = True
    else:
        result["completed"] = False

    # Val curve
    val_accs = re.findall(r'\[val-step\d+\] acc=([0-9.]+)', content)
    result["val_curve"] = [float(v) for v in val_accs]

    # Steps used
    n_curator = content.count("Running Curator at step")
    result["steps"] = n_curator

    return result


def load_all_results() -> List[Dict]:
    """Load all run log files."""
    logs = glob.glob(os.path.join(LOG_DIR, "run_*.log"))
    results = []
    for log in logs:
        r = parse_log(log)
        if r:
            results.append(r)
    return results


def assign_family(name: str) -> Tuple[str, str]:
    """Return (family, config) for a run name."""
    n = name.replace("run_", "").replace(".log", "")

    if re.match(r"^random", n):
        seed = re.search(r"_s(\d+)", n)
        return "random", f"s{seed.group(1) if seed else '42'}"
    if re.match(r"^phased_\d+", n):
        pct = re.search(r"phased_(\d+)", n)
        return f"phased_pct", f"{pct.group(1) if pct else '40'}pct"
    if re.match(r"^phased_thompson", n):
        return "phased_thompson", ""
    if re.match(r"^phased_earlystop", n):
        return "phased_earlystop", ""
    if re.match(r"^phased\d", n) or n.startswith("phased_40") or n.startswith("phased_50"):
        pct = re.search(r"(\d+)_best", n)
        return "phased_bestpb", f"{pct.group(1) if pct else ''}pct"
    if re.match(r"^sp_e(\d+)_s(\d+)", n):
        easy = re.search(r"sp_e(\d+)", n)
        seed = re.search(r"_s(\d+)", n)
        return "stratified_phased", f"e{easy.group(1) if easy else '?'}_{seed.group(1) if seed else 's42'}"
    if re.match(r"^gp_e", n):
        return "general_phased", n
    if re.match(r"^gpv2_", n):
        return "general_phased_v2", n
    if re.match(r"^bp_w", n):
        return "bayesian_phased", n
    if re.match(r"^po_", n):
        return "phase_ordered", n
    if "thompson" in n:
        return "thompson_only", n
    if "easy_first" in n:
        return "easy_first", n
    if "stratified" in n and "phased" not in n:
        return "stratified_only", n
    if "hard_first" in n:
        return "hard_first", n
    if "ucb" in n:
        return "ucb", n
    return "other", n


def group_by_method(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group completed results by method family."""
    groups = defaultdict(list)
    for r in results:
        if not r.get("completed"):
            continue
        family, config = assign_family(r["name"])
        r["family"] = family
        r["config"] = config
        groups[family].append(r)
    return groups


def summarize_group(runs: List[Dict]) -> Dict:
    """Compute mean±std of delta and final_acc for a group."""
    deltas = [r["delta"] for r in runs]
    finals = [r["final_acc"] for r in runs]
    bases  = [r["baseline_acc"] for r in runs]
    n = len(runs)

    def mean(x): return sum(x) / len(x)
    def std(x):
        m = mean(x)
        return (sum((v - m)**2 for v in x) / len(x)) ** 0.5 if len(x) > 1 else 0.0

    return {
        "n": n,
        "delta_mean":  mean(deltas),
        "delta_std":   std(deltas),
        "final_mean":  mean(finals),
        "final_std":   std(finals),
        "base_mean":   mean(bases),
        "best_delta":  max(deltas),
        "best_final":  max(finals),
        "runs":        runs,
    }


def print_leaderboard(groups: Dict[str, List[Dict]]):
    summaries = {}
    for family, runs in groups.items():
        summaries[family] = summarize_group(runs)

    # Sort by mean delta (descending)
    sorted_families = sorted(summaries.items(),
                             key=lambda x: -x[1]["delta_mean"])

    print("\n" + "="*80)
    print("FiNER CURRICULUM LEADERBOARD  (completed runs)")
    print("="*80)
    print(f"{'Method':<28} {'N':>3}  {'Base':>6}  {'Final':>6}  "
          f"{'Δ mean':>8}  {'Δ std':>7}  {'Best Δ':>8}  {'Best final':>10}")
    print("-"*80)

    for family, s in sorted_families:
        print(f"  {family:<26} {s['n']:>3}  {s['base_mean']:.4f}  "
              f"{s['final_mean']:.4f}  {s['delta_mean']:>+8.4f}  "
              f"{s['delta_std']:>7.4f}  {s['best_delta']:>+8.4f}  "
              f"{s['best_final']:>10.4f}")

    # Random baseline reference
    if "random" in summaries:
        r_delta = summaries["random"]["delta_mean"]
        print(f"\n  Random baseline Δ = {r_delta:+.4f}")
        print(f"\n  Methods ABOVE random:")
        for family, s in sorted_families:
            if s["delta_mean"] > r_delta + 0.005:
                print(f"    {family}: +{s['delta_mean'] - r_delta:.4f} over random "
                      f"(+{(s['delta_mean']-r_delta)/abs(r_delta)*100:.1f}%)")
    print("="*80)


def print_phased_ratio_analysis(groups: Dict[str, List[Dict]]):
    """Print analysis of easy phase ratio effect."""
    print("\n" + "="*60)
    print("Easy Phase Ratio Analysis (FiNER, all seeds)")
    print("="*60)

    # Collect all phased runs by easy %
    ratio_runs = defaultdict(list)
    for r in sum(groups.values(), []):
        n = r["name"].lower()
        if "phased_" in n and not "thompson" not in n:
            m = re.search(r"phased_0?(\d+)", n)
            if m:
                pct = int(m.group(1))
                ratio_runs[pct].append(r)

    # Also include original phased methods
    for family, runs in groups.items():
        if "phased_pct" in family:
            for r in runs:
                m = re.search(r"phased_(\d+)", r["name"])
                if m:
                    ratio_runs[int(m.group(1))].append(r)

    for pct in sorted(ratio_runs.keys()):
        runs = ratio_runs[pct]
        if runs:
            s = summarize_group(runs)
            print(f"  easy={pct:3d}%: N={s['n']} "
                  f"Δ={s['delta_mean']:+.4f}±{s['delta_std']:.4f} "
                  f"final={s['final_mean']:.4f}")
    print("="*60)


def main():
    results = load_all_results()
    completed = [r for r in results if r.get("completed")]
    in_progress = [r for r in results if not r.get("completed")]

    print(f"\nLoaded {len(results)} logs: {len(completed)} completed, "
          f"{len(in_progress)} in progress")

    groups = group_by_method(results)
    print_leaderboard(groups)
    print_phased_ratio_analysis(groups)

    # Print val curves for key methods
    print("\n" + "="*60)
    print("Val Curves (key completed methods)")
    print("="*60)
    key_methods = ["random", "phased_pct", "stratified_phased",
                   "phased_thompson", "bayesian_phased"]
    for family in key_methods:
        for r in groups.get(family, []):
            if r["val_curve"]:
                curve = "  ".join(f"{v:.3f}" for v in r["val_curve"])
                print(f"  {r['name'][:45]}: [{curve}]")

    print("\n" + "="*60)
    print("Still running:")
    for r in sorted(in_progress, key=lambda x: -x.get("steps", 0)):
        vals = "  ".join(f"{v:.3f}" for v in r.get("val_curve", []))
        print(f"  {r['name'][:40]:40s} ({r.get('steps',0):3d} steps): [{vals}]")


if __name__ == "__main__":
    main()
