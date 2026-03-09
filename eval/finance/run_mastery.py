#!/usr/bin/env python3
"""
Mastery-Based Curriculum Learning for FiNER (XBRL tagging).

Strategy:
  1. Explode batched 4-question samples into single-question samples.
  2. Group single-question samples by their XBRL label.
  3. Sort label groups by frequency (most common labels first).
  4. For each label group, train sequentially (no repeats) until mastery:
       mastery = sum(sliding_window[-WINDOW:]) >= THRESHOLD
     Adaptive threshold: min(THRESHOLD, max(2, n_samples // 3))
  5. After all labels done, evaluate on the test set using the final playbook.

Usage:
  cd /workspace/ace_leo
  source .env
  python -m eval.finance.run_mastery \
      --save_path results/finer_mastery \
      --api_provider together \
      --generator_model deepseek-ai/DeepSeek-V3.1 \
      --mastery_threshold 5 \
      --mastery_window 8 \
      --test_workers 20
"""

import os
import re
import json
import argparse
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Dict, Any

from ace import ACE
from eval.finance.data_processor import DataProcessor, load_data
from utils import evaluate_test_set


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FiNER Mastery-Based Curriculum Training")
    p.add_argument("--save_path",         type=str, required=True)
    p.add_argument("--config_path",       type=str,
                   default="./eval/finance/data/sample_config.json")
    p.add_argument("--api_provider",      type=str, default="together",
                   choices=["sambanova", "together", "openai"])
    p.add_argument("--generator_model",   type=str, default="deepseek-ai/DeepSeek-V3.1")
    p.add_argument("--reflector_model",   type=str, default="deepseek-ai/DeepSeek-V3.1")
    p.add_argument("--curator_model",     type=str, default="deepseek-ai/DeepSeek-V3.1")
    p.add_argument("--max_tokens",        type=int, default=4096)
    p.add_argument("--mastery_threshold", type=int, default=5,
                   help="Correct answers needed in sliding window to declare mastery")
    p.add_argument("--mastery_window",    type=int, default=8,
                   help="Sliding window size for mastery check")
    p.add_argument("--max_num_rounds",    type=int, default=3)
    p.add_argument("--curator_frequency", type=int, default=1)
    p.add_argument("--playbook_token_budget", type=int, default=80000)
    p.add_argument("--test_workers",      type=int, default=20)
    p.add_argument("--initial_playbook_path", type=str, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Parsing: explode 4-question batches → single-question samples
# ─────────────────────────────────────────────────────────────

def parse_batch_to_singles(raw_sample: Dict) -> List[Dict]:
    """
    Given a raw FiNER sample with 4 batched questions, return up to 4
    single-question dicts:  {context, question, target, label}
    """
    context    = raw_sample.get("context", "")
    target_str = raw_sample.get("target", "")
    targets    = [t.strip() for t in target_str.split(",")]

    # Locate where numbered questions start (\n1. What is best tag …)
    q_positions = list(re.finditer(r'\n(\d+)\.\s+What is best tag', context))
    if not q_positions:
        return []

    # Header = everything before the first numbered question
    header = context[: q_positions[0].start()].strip()

    # Rewrite the header instruction for single-question format
    header_single = re.sub(
        r'Answer the following \d+ independent questions by providing only\s+'
        r'\d+ US GAAP tags answers in the order of the questions\.'
        r'.*?Provide nothing else\.',
        'Answer the following question by providing only 1 US GAAP tag. '
        'Provide nothing else.',
        header,
        flags=re.DOTALL,
    )

    singles = []
    for i, match in enumerate(q_positions):
        if i >= len(targets):
            break

        # Slice this question's text
        q_start = match.start() + 1          # skip leading \n
        if i + 1 < len(q_positions):
            q_end = q_positions[i + 1].start()
        else:
            # Last question: cut at "Output US GAAP tags:"
            tail_match = re.search(r'\nOutput US GAAP tags:', context[match.start():])
            q_end = (match.start() + tail_match.start()) if tail_match else len(context)

        q_text = context[q_start:q_end].strip()
        target = targets[i]

        # Build the single-question prompt (ACE format: context="", question=full_prompt)
        single_prompt = (
            f"{header_single}\n"
            f"{q_text}\n"
            f"Output US GAAP tag:"
        )

        singles.append({
            "context":  "",
            "question": single_prompt,
            "target":   target,
            "label":    target,          # XBRL tag name used for grouping
        })

    return singles


# ─────────────────────────────────────────────────────────────
# Mastery curriculum training loop
# ─────────────────────────────────────────────────────────────

def run_mastery_curriculum(
    ace_system:       ACE,
    all_singles:      List[Dict],
    data_processor,
    config_params:    Dict[str, Any],
    save_path:        str,
    usage_log_path:   str,
    log_dir:          str,
    mastery_threshold: int,
    mastery_window:    int,
) -> Dict[str, Any]:
    """
    Iterate over label groups in descending frequency order.
    For each group, train until mastery or samples exhausted.
    Returns per-label stats dict.
    """
    # ── Group samples by label ──────────────────────────────
    label_to_samples: Dict[str, List[Dict]] = defaultdict(list)
    for s in all_singles:
        label_to_samples[s["label"]].append(s)

    sorted_labels = sorted(label_to_samples, key=lambda l: -len(label_to_samples[l]))

    total_labels   = len(sorted_labels)
    global_step    = 0
    label_stats    = {}   # label → {n_used, mastered, n_available, threshold}

    print(f"\n{'='*60}")
    print(f"MASTERY CURRICULUM  ({total_labels} unique labels)")
    print(f"Window={mastery_window}, Threshold={mastery_threshold}")
    print(f"{'='*60}\n")

    playbook_dir = os.path.join(save_path, "intermediate_playbooks")
    os.makedirs(playbook_dir, exist_ok=True)

    for label_idx, label in enumerate(sorted_labels):
        samples     = label_to_samples[label]
        n_available = len(samples)
        threshold   = min(mastery_threshold, max(2, n_available // 3))
        window      = deque(maxlen=mastery_window)
        n_used      = 0
        mastered    = False

        print(f"\n[{label_idx+1}/{total_labels}] Label: {label}  "
              f"(n={n_available}, threshold={threshold})")

        for sample in samples:
            global_step += 1
            step_id = f"mastery_l{label_idx}_s{n_used}"

            _, _, tracking = ace_system._train_single_sample(
                task_dict      = sample,
                data_processor = data_processor,
                step_id        = step_id,
                epoch          = 1,
                step           = global_step,
                usage_log_path = usage_log_path,
                log_dir        = log_dir,
                config_params  = config_params,
                total_samples  = len(all_singles),
            )

            pre_correct = tracking["pre_train_result"]["is_correct"]
            window.append(int(pre_correct))
            n_used += 1

            # Log per-step result
            window_str = "".join("✓" if x else "✗" for x in window)
            print(f"  step {n_used:3d}/{n_available} | "
                  f"correct={pre_correct} | "
                  f"window=[{window_str}] {sum(window)}/{len(window)}")

            # Check mastery: only after window is at least threshold+1 filled
            if (len(window) >= min(mastery_window, threshold + 1)
                    and sum(window) >= threshold):
                mastered = True
                print(f"  ✅  MASTERED {label} after {n_used} samples!")
                break

        if not mastered:
            print(f"  ⚠️  Exhausted all {n_available} samples for {label} "
                  f"(best window: {sum(window)}/{len(window)})")

        label_stats[label] = {
            "n_used":       n_used,
            "n_available":  n_available,
            "threshold":    threshold,
            "mastered":     mastered,
            "final_window_correct": int(sum(window)),
        }

        # Save intermediate playbook after each label
        pb_path = os.path.join(
            playbook_dir, f"playbook_after_label_{label_idx:03d}.txt"
        )
        with open(pb_path, "w") as f:
            f.write(ace_system.playbook)

    return label_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Paths ────────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder  = f"ace_run_{timestamp}_finer_mastery"
    save_path   = os.path.join(args.save_path, run_folder)
    log_dir     = os.path.join(save_path, "detailed_llm_logs")
    playbook_dir = os.path.join(save_path, "intermediate_playbooks")
    os.makedirs(log_dir,      exist_ok=True)
    os.makedirs(playbook_dir, exist_ok=True)
    usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")

    print(f"\n{'='*60}")
    print(f"FiNER Mastery-Based Curriculum")
    print(f"Save path : {save_path}")
    print(f"Model     : {args.generator_model}")
    print(f"Mastery   : {args.mastery_threshold}/{args.mastery_window}")
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────────────────────
    with open(args.config_path) as f:
        task_config = json.load(f)

    finer_cfg        = task_config["finer"]
    train_raw        = load_data(finer_cfg["train_data"])
    val_raw          = load_data(finer_cfg["val_data"])
    test_raw         = load_data(finer_cfg["test_data"])

    data_processor = DataProcessor(task_name="finer")
    val_samples    = data_processor.process_task_data(val_raw)
    test_samples   = data_processor.process_task_data(test_raw)

    # ── Explode train batches → single-question samples ─────
    all_singles: List[Dict] = []
    for raw in train_raw:
        all_singles.extend(parse_batch_to_singles(raw))

    print(f"Train batches : {len(train_raw)}")
    print(f"Single Qs     : {len(all_singles)}")

    # Label frequency summary
    label_counts: Dict[str, int] = defaultdict(int)
    for s in all_singles:
        label_counts[s["label"]] += 1
    print(f"Unique labels : {len(label_counts)}")
    top5 = sorted(label_counts.items(), key=lambda x: -x[1])[:5]
    print("Top-5 labels  :", ", ".join(f"{l}({c})" for l, c in top5))

    # ── Load initial playbook (optional) ────────────────────
    initial_playbook = None
    if args.initial_playbook_path and os.path.exists(args.initial_playbook_path):
        with open(args.initial_playbook_path) as f:
            initial_playbook = f.read()
        print(f"Initial playbook: {args.initial_playbook_path}")
    else:
        print("Initial playbook: empty")

    # ── ACE system ───────────────────────────────────────────
    ace_system = ACE(
        api_provider      = args.api_provider,
        generator_model   = args.generator_model,
        reflector_model   = args.reflector_model,
        curator_model     = args.curator_model,
        max_tokens        = args.max_tokens,
        initial_playbook  = initial_playbook,
    )

    config_params = {
        "max_num_rounds":    args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "token_budget":      args.playbook_token_budget,
        "task_name":         "finer_mastery",
        "use_json_mode":     False,
        "no_ground_truth":   False,
        "save_dir":          args.save_path,
        "test_workers":      args.test_workers,
        "eval_steps":        9999,   # not used in custom loop
        "save_steps":        9999,
        "use_bulletpoint_analyzer": False,
        "bulletpoint_analyzer_threshold": 0.90,
    }

    # ── Save run config ──────────────────────────────────────
    with open(os.path.join(save_path, "run_config.json"), "w") as f:
        json.dump({
            "mastery_threshold": args.mastery_threshold,
            "mastery_window":    args.mastery_window,
            "generator_model":   args.generator_model,
            "api_provider":      args.api_provider,
            "train_batches":     len(train_raw),
            "single_questions":  len(all_singles),
        }, f, indent=2)

    # ── Baseline test (no playbook) ──────────────────────────
    print(f"\n{'='*60}")
    print("BASELINE TEST (empty playbook)")
    print(f"{'='*60}")
    baseline_results, _ = evaluate_test_set(
        data_processor = data_processor,
        generator      = ace_system.generator,
        playbook       = ace_system.playbook,
        test_samples   = test_samples,
        max_tokens     = args.max_tokens,
        log_dir        = log_dir,
        max_workers    = args.test_workers,
    )
    baseline_acc = baseline_results["accuracy"]
    print(f"Baseline Test Accuracy: {baseline_acc:.4f}")

    # ── Mastery curriculum training ──────────────────────────
    label_stats = run_mastery_curriculum(
        ace_system        = ace_system,
        all_singles       = all_singles,
        data_processor    = data_processor,
        config_params     = config_params,
        save_path         = save_path,
        usage_log_path    = usage_log_path,
        log_dir           = log_dir,
        mastery_threshold = args.mastery_threshold,
        mastery_window    = args.mastery_window,
    )

    # ── Save final playbook ──────────────────────────────────
    final_pb_path = os.path.join(save_path, "final_playbook.txt")
    with open(final_pb_path, "w") as f:
        f.write(ace_system.playbook)
    print(f"\nFinal playbook saved: {final_pb_path}")

    # ── Per-label mastery summary ────────────────────────────
    total_used    = sum(v["n_used"]    for v in label_stats.values())
    total_mastered = sum(1 for v in label_stats.values() if v["mastered"])
    print(f"\n{'='*60}")
    print(f"MASTERY SUMMARY")
    print(f"{'='*60}")
    print(f"Labels mastered : {total_mastered} / {len(label_stats)}")
    print(f"Total samples   : {total_used} / {len(all_singles)}")
    print(f"\nPer-label breakdown (sorted by n_used desc):")
    print(f"{'Label':<60} {'avail':>6} {'used':>5} {'thr':>4} {'mastered':>8}")
    print("-" * 90)
    for label, stat in sorted(label_stats.items(), key=lambda x: -x[1]["n_used"]):
        print(f"{label:<60} {stat['n_available']:>6} {stat['n_used']:>5} "
              f"{stat['threshold']:>4} {'✅' if stat['mastered'] else '❌':>8}")

    # Save label stats JSON
    stats_path = os.path.join(save_path, "label_mastery_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "label_stats":     label_stats,
            "total_used":      total_used,
            "total_mastered":  total_mastered,
            "total_labels":    len(label_stats),
            "baseline_acc":    baseline_acc,
        }, f, indent=2)

    # ── Final test eval ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL TEST (mastery playbook)")
    print(f"{'='*60}")
    final_results, _ = evaluate_test_set(
        data_processor = data_processor,
        generator      = ace_system.generator,
        playbook       = ace_system.playbook,
        test_samples   = test_samples,
        max_tokens     = args.max_tokens,
        log_dir        = log_dir,
        max_workers    = args.test_workers,
    )
    final_acc = final_results["accuracy"]
    print(f"Baseline Test Accuracy : {baseline_acc:.4f}")
    print(f"Final    Test Accuracy : {final_acc:.4f}")
    print(f"Delta                  : {final_acc - baseline_acc:+.4f}")

    # Update stats file with final acc
    with open(stats_path) as f:
        stats = json.load(f)
    stats["final_acc"] = final_acc
    stats["delta"]     = final_acc - baseline_acc
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
