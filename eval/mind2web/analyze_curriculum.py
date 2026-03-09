#!/usr/bin/env python3
"""
Analyze and compare Mind2Web curriculum learning experiments.

Usage:
  cd /workspace/ace_leo
  python -m eval.mind2web.analyze_curriculum
"""

import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict


RESULTS_ROOT = "./results/mind2web_curriculum"


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
    best = {}
    for r in results:
        sel = r["selector"]
        if sel not in best or r["final_test_acc"] > best[sel]["final_test_acc"]:
            best[sel] = r
    return best


def print_leaderboard(best: Dict[str, Dict]):
    print("\n" + "="*72)
    print("MIND2WEB CURRICULUM LEADERBOARD")
    print("="*72)
    print(f"{'Selector':<38} {'Baseline':>8} {'Final':>8} {'Delta':>8}  "
          f"{'BestVal':>8}  {'Steps':>5}")
    print("-"*72)
    for sel, r in sorted(best.items(), key=lambda x: -x[1]["final_test_acc"]):
        bv = r.get("best_val_acc", r.get("val_curve", [{}])[-1].get("val_acc", 0))
        print(f"{sel:<38} {r['baseline_test_acc']:>8.4f} {r['final_test_acc']:>8.4f} "
              f"{r['delta']:>+8.4f}  {bv:>8.4f}  {r['steps_used']:>5}")
    print("="*72)


def print_val_curves(best: Dict[str, Dict]):
    print("\n" + "="*72)
    print("VALIDATION CURVES")
    print("="*72)
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
    print("="*72)


def print_category_analysis(best: Dict[str, Dict]):
    print("\n" + "="*72)
    print("CATEGORY TRAINING DISTRIBUTION (domain×operation)")
    print("="*72)
    for sel, r in sorted(best.items(), key=lambda x: -x[1]["final_test_acc"]):
        step_log = r.get("step_log", [])
        if not step_log:
            continue
        cat_counts: Dict[str, int] = defaultdict(int)
        cat_correct: Dict[str, int] = defaultdict(int)
        for s in step_log:
            cat = s.get("cat", "?")
            cat_counts[cat] += 1
            cat_correct[cat] += int(s.get("correct", False))
        total = sum(cat_counts.values())
        top = sorted(cat_counts.items(), key=lambda x: -x[1])[:5]
        print(f"\n  {sel} (final={r['final_test_acc']:.4f})")
        for cat, cnt in top:
            err = 1 - cat_correct[cat] / cnt
            print(f"    {cat:<30} {cnt:3d} ({cnt/total:.0%})  err={err:.2f}")
    print("="*72)


def print_auc_analysis(best: Dict[str, Dict]):
    print("\n" + "="*72)
    print("LEARNING EFFICIENCY (AUC under val curve)")
    print("="*72)
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
        norm_auc = auc / max(steps) if steps else 0
        aucs.append((sel, norm_auc, max(accs), r["final_test_acc"]))
    for sel, auc, peak, final in sorted(aucs, key=lambda x: -x[1]):
        print(f"  {sel:<38} AUC={auc:.4f}  peak={peak:.4f}  final={final:.4f}")
    print("="*72)


def compare_finer_vs_mind2web():
    """Print side-by-side comparison if FiNER results available."""
    finer_root = "./results/finer_curriculum"
    if not os.path.exists(finer_root):
        return
    finer_results = []
    for d in sorted(os.listdir(finer_root)):
        rpath = os.path.join(finer_root, d, "result.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            finer_results.append(r)
    if not finer_results:
        return
    finer_best = {}
    for r in finer_results:
        sel = r["selector"]
        if sel not in finer_best or r["final_test_acc"] > finer_best[sel]["final_test_acc"]:
            finer_best[sel] = r
    # Only show selectors present in both
    m2w_results = load_all_results(RESULTS_ROOT)
    m2w_best = best_per_selector(m2w_results)
    common = set(finer_best) & set(m2w_best)
    if not common:
        return
    print("\n" + "="*72)
    print("CROSS-TASK COMPARISON  (FiNER vs Mind2Web)")
    print("="*72)
    print(f"{'Selector':<38} {'FiNER Δ':>8} {'M2W Δ':>8}")
    print("-"*72)
    for sel in sorted(common, key=lambda s: -finer_best[s]["delta"]):
        fd = finer_best[sel]["delta"]
        md = m2w_best[sel]["delta"]
        print(f"{sel:<38} {fd:>+8.4f} {md:>+8.4f}")
    print("="*72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=RESULTS_ROOT)
    args = parser.parse_args()

    results = load_all_results(args.results_root)
    if not results:
        print(f"No results yet in {args.results_root}")
        return

    print(f"Loaded {len(results)} runs from {args.results_root}")
    best = best_per_selector(results)

    print_leaderboard(best)
    print_val_curves(best)
    print_auc_analysis(best)
    print_category_analysis(best)
    compare_finer_vs_mind2web()


if __name__ == "__main__":
    main()
