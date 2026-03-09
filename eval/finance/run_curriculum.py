#!/usr/bin/env python3
"""
Curriculum learning comparison for FiNER XBRL tagging.

Runs ONE selector (specified via --selector) for a fixed budget of B steps,
then evaluates on the test set.  Launch multiple times in parallel or via
run_finer_curriculum.sh to compare all methods.

Usage:
  cd /workspace/ace_leo
  source .env

  # Pre-test once (shared across all selectors):
  python -m eval.finance.run_curriculum --selector pretest_only

  # Run each selector:
  python -m eval.finance.run_curriculum --selector random
  python -m eval.finance.run_curriculum --selector stratified
  python -m eval.finance.run_curriculum --selector hard_first
  python -m eval.finance.run_curriculum --selector easy_first
  python -m eval.finance.run_curriculum --selector ucb_cat
  python -m eval.finance.run_curriculum --selector error_focused
  python -m eval.finance.run_curriculum --selector hard_cat_first

Selectors available:
  random           Uniform random (baseline)
  stratified       Round-robin over 12 macro categories
  hard_first       Pre-test: wrong answers first
  easy_first       Pre-test: correct answers first
  ucb_cat          UCB1 bandit over macro categories
  error_focused    Greedy error-rate maximiser (rolling window)
  hard_cat_first   Hardest category (by pre-test acc) first, then next, etc.
  pretest_only     Just run the pre-test, cache, and exit.
"""

import os
import re
import json
import argparse
import random
from collections import defaultdict
from datetime import datetime
from typing import List, Dict

from ace import ACE
from eval.finance.data_processor import DataProcessor, load_data
from eval.finance.run_mastery_macro import parse_batch_to_singles, MACRO_CATEGORIES, get_macro_category
from eval.finance.curriculum.selectors import (
    RandomSelector, StratifiedSelector, HardFirstSelector, EasyFirstSelector,
    UCBCategorySelector, ErrorFocusedSelector, HardCategoryFirstSelector,
    PhasedEasyHardSelector, DiversityAwareSelector, HybridHardUCBSelector,
    ThompsonSamplingSelector, PhasedThompsonSelector,
    GeneralPhasedSelector, GeneralPhasedV2, PhaseOrderedSelector,
    BayesianPhasedSelector, StratifiedPhasedSelector,
)
from eval.finance.curriculum.offline_bandit import (
    OfflineBanditSelector, build_offline_bandit_from_results,
)
from eval.finance.curriculum.pretester import run_pretest
from eval.finance.curriculum.runner import CurriculumRunner


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_DATA   = "./eval/finance/data/finer_train_batched_1000_samples.jsonl"
VAL_DATA     = "./eval/finance/data/finer_val_batched_500_samples.jsonl"
TEST_DATA    = "./eval/finance/data/finer_test_subset_006_seed42.jsonl"

RESULTS_ROOT = "./results/finer_curriculum"

# Use a fixed 100-sample val subset for fast online monitoring
VAL_MONITOR_N = 100

# Number of training examples to pre-test (for difficulty scoring)
PRETEST_N = 600   # ~15% of pool; 600 / 30 workers ≈ 2 min


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selector",          type=str, required=True)
    p.add_argument("--budget",            type=int, default=200)
    p.add_argument("--eval_every",        type=int, default=25)
    p.add_argument("--api_provider",      type=str, default="together")
    p.add_argument("--model",             type=str, default="deepseek-ai/DeepSeek-V3.1")
    p.add_argument("--max_tokens",        type=int, default=4096)
    p.add_argument("--test_workers",      type=int, default=20)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--ucb_c",            type=float, default=1.0,
                   help="Exploration constant for UCB selector")
    p.add_argument("--phase_split",      type=float, default=0.4,
                   help="Fraction of budget for easy phase in PhasedEasyHard selector")
    p.add_argument("--easy_pct",         type=float, default=0.20)
    p.add_argument("--medium_pct",       type=float, default=0.60)
    p.add_argument("--hard_pct",         type=float, default=0.20)
    p.add_argument("--switch_patience",  type=int,   default=0,
                   help="GeneralPhased: val evals without improvement before switching phase")
    p.add_argument("--within_phase",     type=str,   default="diversity",
                   choices=["diversity", "random"],
                   help="Within-phase selection strategy")
    p.add_argument("--phase_order",      type=str,   default="EMH",
                   help="Phase ordering for PhaseOrderedSelector, e.g. EMH, EHM, EH")
    p.add_argument("--phase_fracs",      type=str,   default=None,
                   help="JSON string of phase fractions, e.g. '{\"E\":0.25,\"H\":0.25,\"M\":0.50}'")
    p.add_argument("--warmup_pct",       type=float, default=0.25,
                   help="BayesianPhased: fraction of budget for easy warmup phase")
    p.add_argument("--prior_strength",   type=float, default=2.0,
                   help="BayesianPhased: strength of pretest difficulty prior")
    p.add_argument("--use_best_for_test", action="store_true",
                   help="Use best-val playbook for final test (train full budget, no early stop)")
    p.add_argument("--val_patience",      type=int, default=0,
                   help="Early stopping: stop if val doesn't improve for N checks (0=disabled)")
    p.add_argument("--pretest_force",     action="store_true",
                   help="Force re-run pre-test even if cached")
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_and_explode_train(path: str, seed: int = 42) -> List[Dict]:
    """Load train data, explode to single-question format, add _id."""
    raw = load_data(path)
    singles = []
    for raw_s in raw:
        for s in parse_batch_to_singles(raw_s):
            singles.append(s)
    rng = random.Random(seed)
    rng.shuffle(singles)
    # Assign integer _id AFTER shuffle so it's consistent within a run
    for i, s in enumerate(singles):
        s["_id"] = i
    return singles


def sample_val_monitor(val_raw: List[Dict], data_processor, n: int, seed: int) -> List[Dict]:
    """Sample n batched val samples for online monitoring."""
    rng = random.Random(seed)
    subset = rng.sample(val_raw, min(n, len(val_raw)))
    return data_processor.process_task_data(subset)


# ── Selector factory ──────────────────────────────────────────────────────────

def build_selector(name: str, args, difficulty: Dict, cat_acc: Dict):
    if name == "random":
        return RandomSelector(seed=args.seed)
    elif name == "stratified":
        return StratifiedSelector(seed=args.seed)
    elif name == "hard_first":
        return HardFirstSelector(difficulty=difficulty, seed=args.seed)
    elif name == "easy_first":
        return EasyFirstSelector(difficulty=difficulty, seed=args.seed)
    elif name == "ucb_cat":
        return UCBCategorySelector(seed=args.seed, c=args.ucb_c)
    elif name == "error_focused":
        return ErrorFocusedSelector(seed=args.seed)
    elif name == "hard_cat_first":
        return HardCategoryFirstSelector(cat_accuracy=cat_acc, seed=args.seed)
    elif name == "phased":
        return PhasedEasyHardSelector(difficulty=difficulty, budget=args.budget,
                                      phase_split=args.phase_split, seed=args.seed)
    elif name == "diversity":
        return DiversityAwareSelector(seed=args.seed)
    elif name == "thompson":
        return ThompsonSamplingSelector(seed=args.seed)
    elif name == "ucb_cat_c03":
        return UCBCategorySelector(seed=args.seed, c=0.3)
    elif name == "hybrid_hard_ucb":
        return HybridHardUCBSelector(difficulty=difficulty, budget=args.budget,
                                     warmup_steps=50, c=0.5, seed=args.seed)
    elif name == "phased_thompson":
        return PhasedThompsonSelector(difficulty=difficulty, budget=args.budget,
                                      phase_split=args.phase_split, seed=args.seed)
    elif name == "general_phased":
        return GeneralPhasedSelector(
            difficulty=difficulty,
            budget=args.budget,
            easy_pct=args.easy_pct,
            medium_pct=args.medium_pct,
            hard_pct=args.hard_pct,
            switch_patience=args.switch_patience,
            within_phase_strategy=args.within_phase,
            seed=args.seed,
        )
    elif name == "general_phased_v2":
        return GeneralPhasedV2(
            difficulty=difficulty,
            budget=args.budget,
            easy_pct=args.easy_pct,
            medium_pct=args.medium_pct,
            hard_pct=args.hard_pct,
            switch_patience=args.switch_patience,
            seed=args.seed,
        )
    elif name == "phase_ordered":
        import json as _json
        fracs = _json.loads(args.phase_fracs) if args.phase_fracs else None
        return PhaseOrderedSelector(
            difficulty=difficulty,
            budget=args.budget,
            phase_order=args.phase_order,
            phase_fracs=fracs,
            switch_patience=args.switch_patience,
            seed=args.seed,
        )
    elif name == "stratified_phased":
        return StratifiedPhasedSelector(
            difficulty=difficulty,
            budget=args.budget,
            easy_pct=args.easy_pct,
            include_medium=True,
            seed=args.seed,
        )
    elif name == "bayesian_phased":
        return BayesianPhasedSelector(
            difficulty=difficulty,
            budget=args.budget,
            warmup_pct=args.warmup_pct,
            prior_strength=args.prior_strength,
            seed=args.seed,
        )
    elif name == "offline_bandit":
        model_path = os.path.join(RESULTS_ROOT, "offline_bandit_model.json")
        if not os.path.exists(model_path):
            print(f"[offline_bandit] Building model from Phase 1 results...")
            model = build_offline_bandit_from_results(
                results_root=RESULTS_ROOT,
                eval_every=args.eval_every,
                save_model_path=model_path,
            )
        else:
            with open(model_path) as f:
                model = json.load(f)
            print(f"[offline_bandit] Loaded model R²={model.get('r2', '?')}")
        return OfflineBanditSelector(model=model, budget=args.budget, seed=args.seed)
    else:
        raise ValueError(f"Unknown selector: {name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name   = f"{timestamp}_{args.selector}_b{args.budget}"
    save_path  = os.path.join(RESULTS_ROOT, run_name)
    log_dir    = os.path.join(save_path, "llm_logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    print(f"\n{'='*65}")
    print(f"FiNER Curriculum  selector={args.selector}  budget={args.budget}")
    print(f"Model: {args.model}")
    print(f"{'='*65}")

    # ── Load data ─────────────────────────────────────────────
    train_singles = load_and_explode_train(TRAIN_DATA, seed=args.seed)
    val_raw       = load_data(VAL_DATA)
    test_raw      = load_data(TEST_DATA)

    data_processor = DataProcessor(task_name="finer")
    val_monitor    = sample_val_monitor(val_raw, data_processor, VAL_MONITOR_N, args.seed)
    test_samples   = data_processor.process_task_data(test_raw)

    print(f"Train pool     : {len(train_singles)} singles")
    print(f"Val monitor    : {len(val_monitor)} batched samples")
    print(f"Test samples   : {len(test_samples)} batched samples")

    # ── ACE system (fresh for each run) ──────────────────────
    ace_system = ACE(
        api_provider    = args.api_provider,
        generator_model = args.model,
        reflector_model = args.model,
        curator_model   = args.model,
        max_tokens      = args.max_tokens,
    )

    config_params = {
        "max_num_rounds":    3,
        "curator_frequency": 1,
        "token_budget":      80000,
        "val_patience":      args.val_patience,
        "use_best_for_test": args.use_best_for_test,
        "task_name":         f"finer_curriculum_{args.selector}",
        "use_json_mode":     False,
        "no_ground_truth":   False,
        "save_dir":          save_path,
        "test_workers":      args.test_workers,
        "eval_steps":        9999,
        "save_steps":        9999,
        "use_bulletpoint_analyzer": False,
        "bulletpoint_analyzer_threshold": 0.90,
    }

    # ── Pre-test (if needed by selector) ─────────────────────
    needs_pretest = args.selector in ("hard_first", "easy_first", "hard_cat_first",
                                      "phased", "hybrid_hard_ucb", "pretest_only",
                                      "phased_thompson", "general_phased",
                                      "general_phased_v2", "phase_ordered",
                                      "bayesian_phased", "stratified_phased")

    difficulty: Dict[str, float] = {}
    cat_acc:    Dict[str, float] = {}

    if needs_pretest:
        pretest_pool = train_singles[:PRETEST_N]
        difficulty, cat_acc = run_pretest(
            pool          = pretest_pool,
            data_processor= data_processor,
            generator     = ace_system.generator,
            max_tokens    = 512,
            max_workers   = 30,
            log_dir       = log_dir,
            cache_tag     = f"b{args.budget}_n{PRETEST_N}",
            force         = args.pretest_force,
        )

    if args.selector == "pretest_only":
        print("\n[pretest_only] Done.")
        return

    # ── Build selector ────────────────────────────────────────
    selector = build_selector(args.selector, args, difficulty, cat_acc)

    # ── Save run config ───────────────────────────────────────
    with open(os.path.join(save_path, "run_config.json"), "w") as f:
        json.dump({
            "selector":    args.selector,
            "budget":      args.budget,
            "eval_every":  args.eval_every,
            "model":       args.model,
            "seed":        args.seed,
            "train_pool":  len(train_singles),
            "val_monitor": len(val_monitor),
            "test_samples":len(test_samples),
            "ucb_c":       args.ucb_c,
            "pretest_n":   PRETEST_N,
        }, f, indent=2)

    # ── Run curriculum ────────────────────────────────────────
    runner = CurriculumRunner(
        ace_system    = ace_system,
        selector      = selector,
        data_processor= data_processor,
        train_pool    = train_singles,
        val_samples   = val_monitor,
        test_samples  = test_samples,
        budget        = args.budget,
        eval_every    = args.eval_every,
        max_tokens    = args.max_tokens,
        log_dir       = log_dir,
        save_path     = save_path,
        config_params = config_params,
    )

    result = runner.run()

    # ── Append to global comparison table ────────────────────
    comparison_path = os.path.join(RESULTS_ROOT, "comparison.jsonl")
    with open(comparison_path, "a") as f:
        f.write(json.dumps({
            "selector":          result["selector"],
            "baseline_test_acc": result["baseline_test_acc"],
            "final_test_acc":    result["final_test_acc"],
            "delta":             result["delta"],
            "steps_used":        result["steps_used"],
            "runtime_minutes":   result["runtime_minutes"],
            "run_name":          run_name,
            "timestamp":         timestamp,
        }) + "\n")

    print(f"Appended to comparison log: {comparison_path}")


if __name__ == "__main__":
    main()
