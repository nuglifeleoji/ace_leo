#!/usr/bin/env python3
"""
Analyze and compare all FiNER curriculum learning experiments.

Usage:
  cd /workspace/ace_leo
  python -m eval.finance.analyze_curriculum
  python -m eval.finance.analyze_curriculum --results_root results/finer_curriculum
"""

import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict


RESULTS_ROOT = "./results/finer_curriculum"


def load_all_results(root: str) -> List[Dict]:
    results = []
    for d in sorted(os.listdir(root)):
        rpath = os.path.join(root, d, "result.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            r["_dir"] = d
            results.append(r)
    return results


def best_per_selector(results: List[Dict]) -> Dict[str, Dict]:
    """Keep only the best run per selector name (by final test acc)."""
    best = {}
    for r in results:
        sel = r["selector"]
        if sel not in best or r["final_test_acc"] > best[sel]["final_test_acc"]:
            best[sel] = r
    return best


def print_leaderboard(best: Dict[str, Dict]):
    print("\n" + "="*70)
    print("LEADERBOARD  (best run per selector, sorted by final test acc)")
    print("="*70)
    print(f"{'Selector':<38} {'Baseline':>8} {'Final':>8} {'Delta':>8}  {'Steps':>5}")
    print("-"*70)
    for sel, r in sorted(best.items(), key=lambda x: -x[1]["final_test_acc"]):
        print(f"{sel:<38} {r['baseline_test_acc']:>8.4f} {r['final_test_acc']:>8.4f} "
              f"{r['delta']:>+8.4f}  {r['steps_used']:>5}")
    print("="*70)


def print_val_curves(best: Dict[str, Dict]):
    print("\n" + "="*70)
    print("VALIDATION CURVES  (val acc at each checkpoint)")
    print("="*70)
    # Collect all steps
    all_steps = sorted(set(
        vc["step"]
        for r in best.values()
        for vc in r.get("val_curve", [])
    ))
    header = f"{'Selector':<32}" + "".join(f" {s:>7}" for s in all_steps)
    print(header)
    print("-" * len(header))
    for sel, r in sorted(best.items(), key=lambda x: -x[1]["final_test_acc"]):
        step_map = {vc["step"]: vc["val_acc"] for vc in r.get("val_curve", [])}
        row = f"{sel:<32}" + "".join(
            f" {step_map[s]:>7.4f}" if s in step_map else "       "
            for s in all_steps
        )
        print(row)
    print("="*70)


def print_category_distribution(best: Dict[str, Dict]):
    print("\n" + "="*70)
    print("CATEGORY DISTRIBUTION  (fraction of steps per macro-category)")
    print("="*70)
    for sel, r in sorted(best.items(), key=lambda x: -x[1]["final_test_acc"]):
        step_log = r.get("step_log", [])
        if not step_log:
            continue
        cat_counts: Dict[str, int] = defaultdict(int)
        for s in step_log:
            cat_counts[s.get("cat", "?")] += 1
        total = sum(cat_counts.values())
        top = sorted(cat_counts.items(), key=lambda x: -x[1])[:5]
        dist = ", ".join(f"{c}={n/total:.0%}" for c, n in top)
        print(f"  {sel:<35} {dist}")
    print("="*70)


def print_auc_analysis(best: Dict[str, Dict]):
    """Learning efficiency: AUC under the validation curve."""
    print("\n" + "="*70)
    print("LEARNING EFFICIENCY  (AUC under val curve, normalized)")
    print("="*70)
    aucs = []
    for sel, r in best.items():
        vc = r.get("val_curve", [])
        if len(vc) < 2:
            continue
        steps = [v["step"] for v in vc]
        accs  = [v["val_acc"] for v in vc]
        auc = sum(
            (steps[i+1] - steps[i]) * (accs[i] + accs[i+1]) / 2
            for i in range(len(steps)-1)
        )
        max_steps = max(steps) if steps else 1
        norm_auc = auc / max_steps
        aucs.append((sel, norm_auc, max(accs), r["final_test_acc"]))
    for sel, auc, peak_val, final_test in sorted(aucs, key=lambda x: -x[1]):
        print(f"  {sel:<38} AUC={auc:.4f}  peak_val={peak_val:.4f}  "
              f"final_test={final_test:.4f}")
    print("="*70)


def print_error_pattern(best: Dict[str, Dict]):
    """How often the model was correct BEFORE training on each step."""
    print("\n" + "="*70)
    print("PRE-TRAIN CORRECTNESS PATTERN")
    print("="*70)
    for sel, r in sorted(best.items(), key=lambda x: -x[1]["final_test_acc"]):
        step_log = r.get("step_log", [])
        if not step_log:
            continue
        n_correct = sum(1 for s in step_log if s.get("correct", False))
        total = len(step_log)
        # Compute rolling 50-step error rate to see improvement
        w = 50
        windows = []
        for i in range(0, total - w + 1, w):
            chunk = step_log[i:i+w]
            err = 1 - sum(s.get("correct", False) for s in chunk) / w
            windows.append(f"{err:.2f}")
        print(f"  {sel:<35} base_err={1-n_correct/total:.3f}  "
              f"windows(50): {' → '.join(windows)}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=RESULTS_ROOT)
    args = parser.parse_args()

    results = load_all_results(args.results_root)
    if not results:
        print(f"No result.json files found in {args.results_root}")
        return

    print(f"Loaded {len(results)} runs from {args.results_root}")
    best = best_per_selector(results)
    print(f"Unique selectors: {len(best)}")

    print_leaderboard(best)
    print_val_curves(best)
    print_auc_analysis(best)
    print_category_distribution(best)
    print_error_pattern(best)


if __name__ == "__main__":
    main()
