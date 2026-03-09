#!/usr/bin/env python3
"""
Curriculum learning comparison for Mind2Web web navigation.

Mirrors eval/finance/run_curriculum.py but adapted for Mind2Web:
  - macro_category = domain  (Travel, Shopping, Entertainment, ...)
  - Difficulty from precomputed train_correctness.json (zero-shot baseline)
  - No batch explosion needed (each sample is already one navigation step)

Usage:
  cd /workspace/ace_leo
  source .env

  # Run each selector:
  python -m eval.mind2web.run_curriculum --selector random
  python -m eval.mind2web.run_curriculum --selector phased
  python -m eval.mind2web.run_curriculum --selector thompson
  python -m eval.mind2web.run_curriculum --selector easy_first
  python -m eval.mind2web.run_curriculum --selector hard_first
  python -m eval.mind2web.run_curriculum --selector stratified
  python -m eval.mind2web.run_curriculum --selector ucb_cat
  python -m eval.mind2web.run_curriculum --selector error_focused

Available selectors:
  random           Uniform random baseline
  stratified       Round-robin over domains
  hard_first       Easiest-samples-first (by zero-shot baseline)
  easy_first       Correct-samples-first (by zero-shot baseline)
  phased           Two-phase: easy 40% then hard 60%
  ucb_cat          UCB1 bandit over domains
  error_focused    Greedy: domain with highest recent error rate
  thompson         Thompson Sampling over domains
  diversity        Diversity-aware (underrepresented domains first)
  hybrid_hard_ucb  Hard-first warmup then UCB exploitation
"""

import os
import json
import random
import argparse
from collections import defaultdict
from datetime import datetime
from typing import List, Dict

from ace import ACE
from eval.mind2web.data_processor import DataProcessor, load_data
from eval.finance.curriculum.selectors import (
    RandomSelector, StratifiedSelector, HardFirstSelector, EasyFirstSelector,
    UCBCategorySelector, ErrorFocusedSelector, HardCategoryFirstSelector,
    PhasedEasyHardSelector, DiversityAwareSelector, HybridHardUCBSelector,
    ThompsonSamplingSelector, GeneralPhasedSelector, GeneralPhasedV2,
    PhaseOrderedSelector, BayesianPhasedSelector, StratifiedPhasedSelector,
)
from eval.mind2web.curriculum.pretester import run_pretest
from eval.finance.curriculum.runner import CurriculumRunner


# ── Config ─────────────────────────────────────────────────────────────────

TRAIN_DATA   = "./eval/mind2web/data/mind2web_train.jsonl"
VAL_DATA     = "./eval/mind2web/data/mind2web_val.jsonl"
TEST_DATA    = "./eval/mind2web/data/mind2web_test.jsonl"

RESULTS_ROOT = "./results/mind2web_curriculum"
VAL_MONITOR_N = 100


# ── macro_category mapping ─────────────────────────────────────────────────
# Mind2Web has only 3 domains (Travel, Shopping, Entertainment).
# We enrich with operation type to create 9 more granular categories:
#   Travel_CLICK, Travel_TYPE, Travel_SELECT,
#   Shopping_CLICK, Shopping_TYPE, Shopping_SELECT,
#   Entertainment_CLICK, ...
# This gives Thompson/UCB more structure to exploit.

def assign_macro_category(sample: Dict) -> str:
    """Map a Mind2Web sample to a domain × operation macro category."""
    domain = sample.get("domain", "Other") or "Other"
    op = (sample.get("operation") or {}).get("op", "CLICK")
    return f"{domain}_{op}"


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selector",      type=str, required=True)
    p.add_argument("--budget",        type=int, default=200)
    p.add_argument("--eval_every",    type=int, default=25)
    p.add_argument("--api_provider",  type=str, default="together")
    p.add_argument("--model",         type=str, default="deepseek-ai/DeepSeek-V3.1")
    p.add_argument("--max_tokens",    type=int, default=4096)
    p.add_argument("--test_workers",  type=int, default=20)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--ucb_c",         type=float, default=1.0)
    p.add_argument("--easy_pct",      type=float, default=0.20)
    p.add_argument("--medium_pct",    type=float, default=0.60)
    p.add_argument("--hard_pct",      type=float, default=0.20)
    p.add_argument("--switch_patience", type=int, default=0)
    p.add_argument("--within_phase",  type=str,  default="diversity",
                   choices=["diversity", "random"])
    p.add_argument("--phase_split",   type=float, default=0.4)
    p.add_argument("--phase_order",    type=str,  default="EMH")
    p.add_argument("--phase_fracs",    type=str,  default=None)
    p.add_argument("--warmup_pct",     type=float, default=0.25)
    p.add_argument("--prior_strength", type=float, default=2.0)
    p.add_argument("--use_best_for_test", action="store_true")
    p.add_argument("--val_patience",  type=int, default=0,
                   help="Early stopping patience (0=disabled)")
    p.add_argument("--pretest_force", action="store_true")
    p.add_argument("--train_subset",  type=int, default=0,
                   help="If >0 limit training pool to first N samples (fast debug)")
    p.add_argument("--test_subset",   type=int, default=441,
                   help="Limit test evaluation to N samples (0=full). Default=441 to match FiNER.")
    return p.parse_args()


# ── Data helpers ───────────────────────────────────────────────────────────

def load_and_prepare_train(path: str, seed: int = 42) -> List[Dict]:
    """Load Mind2Web train data, assign macro_category and _id."""
    raw = load_data(path)
    # Attach original line index BEFORE shuffle (for train_correctness.json lookup)
    for i, s in enumerate(raw):
        s["orig_idx"] = i
    rng = random.Random(seed)
    rng.shuffle(raw)
    for i, s in enumerate(raw):
        s["macro_category"] = assign_macro_category(s)
        s["_id"] = i          # sequential id after shuffle (for pretester cache key)
    return raw


def sample_val_monitor(val_raw: List[Dict], data_processor, n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    subset = rng.sample(val_raw, min(n, len(val_raw)))
    return data_processor.process_task_data(subset)


# ── Selector factory ───────────────────────────────────────────────────────

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
                                      phase_split=0.4, seed=args.seed)
    elif name == "diversity":
        return DiversityAwareSelector(seed=args.seed)
    elif name == "thompson":
        return ThompsonSamplingSelector(seed=args.seed)
    elif name == "ucb_cat_c03":
        return UCBCategorySelector(seed=args.seed, c=0.3)
    elif name == "hybrid_hard_ucb":
        return HybridHardUCBSelector(difficulty=difficulty, budget=args.budget,
                                     warmup_steps=50, c=0.5, seed=args.seed)
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
    else:
        raise ValueError(f"Unknown selector: {name}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{timestamp}_{args.selector}_b{args.budget}"
    save_path = os.path.join(RESULTS_ROOT, run_name)
    log_dir   = os.path.join(save_path, "llm_logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Mind2Web Curriculum  selector={args.selector}  budget={args.budget}")
    print(f"Model: {args.model}")
    print(f"{'='*65}")

    # ── Load data ────────────────────────────────────────────
    train_pool = load_and_prepare_train(TRAIN_DATA, seed=args.seed)
    if args.train_subset > 0:
        train_pool = train_pool[:args.train_subset]
        print(f"  [debug] train_pool truncated to {len(train_pool)}")

    val_raw  = load_data(VAL_DATA)
    test_raw = load_data(TEST_DATA)

    data_processor = DataProcessor(task_name="mind2web")
    val_monitor    = sample_val_monitor(val_raw, data_processor, VAL_MONITOR_N, args.seed)
    if args.test_subset > 0:
        rng_test = random.Random(args.seed + 99)
        test_raw = rng_test.sample(test_raw, min(args.test_subset, len(test_raw)))
    test_samples   = data_processor.process_task_data(test_raw)

    print(f"Train pool  : {len(train_pool)} samples")
    print(f"Val monitor : {len(val_monitor)} samples")
    print(f"Test samples: {len(test_samples)} samples")

    # Print domain distribution
    from collections import Counter
    domain_dist = Counter(s.get("macro_category", "Other") for s in train_pool)
    print(f"\nDomain distribution ({len(domain_dist)} domains):")
    for dom, cnt in sorted(domain_dist.items(), key=lambda x: -x[1])[:15]:
        print(f"  {dom:<25} {cnt:4d}")

    # ── ACE system ───────────────────────────────────────────
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
        "task_name":         f"mind2web_curriculum_{args.selector}",
        "use_json_mode":     False,
        "no_ground_truth":   False,
        "save_dir":          save_path,
        "test_workers":      args.test_workers,
        "eval_steps":        9999,
        "save_steps":        9999,
        "use_bulletpoint_analyzer": False,
    }

    # ── Pre-test / difficulty ─────────────────────────────────
    needs_pretest = args.selector in (
        "hard_first", "easy_first", "hard_cat_first",
        "phased", "hybrid_hard_ucb", "general_phased", "general_phased_v2",
        "phase_ordered", "bayesian_phased", "stratified_phased",
    )

    difficulty: Dict = {}
    cat_acc:    Dict = {}

    if needs_pretest:
        # Use precomputed train_correctness.json via orig_idx mapping
        difficulty, cat_acc = run_pretest(
            pool           = train_pool,
            data_processor = data_processor,
            generator      = ace_system.generator,
            max_tokens     = 512,
            max_workers    = 30,
            log_dir        = log_dir,
            cache_tag      = f"b{args.budget}_n{len(train_pool)}",
            force          = args.pretest_force,
            use_precomputed= True,
        )

    # ── Build selector ────────────────────────────────────────
    selector = build_selector(args.selector, args, difficulty, cat_acc)

    # ── Save run config ───────────────────────────────────────
    with open(os.path.join(save_path, "run_config.json"), "w") as f:
        json.dump({
            "selector":     args.selector,
            "budget":       args.budget,
            "eval_every":   args.eval_every,
            "model":        args.model,
            "seed":         args.seed,
            "train_pool":   len(train_pool),
            "val_monitor":  len(val_monitor),
            "test_samples": len(test_samples),
        }, f, indent=2)

    # ── Run curriculum ────────────────────────────────────────
    runner = CurriculumRunner(
        ace_system     = ace_system,
        selector       = selector,
        data_processor = data_processor,
        train_pool     = train_pool,
        val_samples    = val_monitor,
        test_samples   = test_samples,
        budget         = args.budget,
        eval_every     = args.eval_every,
        max_tokens     = args.max_tokens,
        log_dir        = log_dir,
        save_path      = save_path,
        config_params  = config_params,
    )

    result = runner.run()

    # ── Append to comparison table ────────────────────────────
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
