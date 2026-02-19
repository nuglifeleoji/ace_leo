#!/usr/bin/env python3
"""
Run script for ACE on LSAT Logical Reasoning (AGIEval subset).

Setup:
  python -m eval.lsat_lr.prepare_data

Examples:
  # 0-shot baseline (no playbook)
  python -m eval.lsat_lr.run \\
      --task_name lsat_lr \\
      --mode eval_only \\
      --save_path results/lsat_lr_baseline

  # Offline training (300-sample train set)
  python -m eval.lsat_lr.run \\
      --task_name lsat_lr \\
      --mode offline \\
      --save_path results/lsat_lr_offline

  # Eval with a trained playbook
  python -m eval.lsat_lr.run \\
      --task_name lsat_lr \\
      --mode eval_only \\
      --initial_playbook_path results/lsat_lr_offline/playbook.md \\
      --save_path results/lsat_lr_eval
"""

import argparse
import json
import os

from ace import ACE
from .data_processor import DataProcessor, load_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACE System — LSAT Logical Reasoning")

    p.add_argument("--task_name", type=str, required=True,
                   help="Task key in sample_config.json (e.g. 'lsat_lr')")
    p.add_argument("--initial_playbook_path", type=str, default=None,
                   help="Path to an existing playbook (.md) to start from")
    p.add_argument("--mode", type=str, default="offline",
                   choices=["offline", "online", "eval_only"])

    # Model settings
    p.add_argument("--api_provider", type=str, default="sambanova",
                   choices=["sambanova", "together", "openai"])
    p.add_argument("--generator_model",  type=str, default="DeepSeek-V3.1")
    p.add_argument("--reflector_model",  type=str, default="DeepSeek-V3.1")
    p.add_argument("--curator_model",    type=str, default="DeepSeek-V3.1")

    # Training hyper-params
    p.add_argument("--num_epochs",             type=int,   default=1)
    p.add_argument("--max_num_rounds",         type=int,   default=3)
    p.add_argument("--curator_frequency",      type=int,   default=1)
    p.add_argument("--eval_steps",             type=int,   default=100)
    p.add_argument("--online_eval_frequency",  type=int,   default=15)
    p.add_argument("--save_steps",             type=int,   default=50)

    # Token / concurrency
    p.add_argument("--max_tokens",            type=int,   default=4096)
    p.add_argument("--playbook_token_budget", type=int,   default=80000)
    p.add_argument("--test_workers",          type=int,   default=20)

    # Misc
    p.add_argument("--json_mode",           action="store_true")
    p.add_argument("--no_ground_truth",     action="store_true")
    p.add_argument("--skip_initial_test",   action="store_true",
                   help="Skip initial test eval during offline training")
    p.add_argument("--use_bulletpoint_analyzer",        action="store_true")
    p.add_argument("--bulletpoint_analyzer_threshold",  type=float, default=0.90)
    p.add_argument("--save_path", type=str, required=True,
                   help="Directory to save playbook and results")

    return p.parse_args()


def preprocess_data(task_name: str, config: dict, mode: str):
    proc = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        if "test_data" not in config:
            raise ValueError(f"{mode} mode requires 'test_data' in config.")
        test_samples = proc.process_task_data(load_data(config["test_data"]))
        print(f"{'Online' if mode == 'online' else 'Eval-only'} mode: "
              f"testing on {len(test_samples)} examples")
    else:
        train_samples = proc.process_task_data(load_data(config["train_data"]))
        val_samples   = proc.process_task_data(load_data(config["val_data"]))
        test_samples  = []
        if "test_data" in config:
            test_samples = proc.process_task_data(load_data(config["test_data"]))
        print(f"Offline mode: train={len(train_samples)}  "
              f"val={len(val_samples)}  test={len(test_samples)}")

    return train_samples, val_samples, test_samples, proc


def load_initial_playbook(path: str | None) -> str | None:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


def main() -> None:
    args = parse_args()

    print(f"\n{'=' * 60}")
    print("ACE SYSTEM — LSAT Logical Reasoning (AGIEval)")
    print(f"{'=' * 60}")
    print(f"Task    : {args.task_name}")
    print(f"Mode    : {args.mode.upper().replace('_', ' ')}")
    print(f"Model   : {args.generator_model}")
    print(f"{'=' * 60}\n")

    config_path = "./eval/lsat_lr/data/sample_config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if args.task_name not in cfg:
        raise ValueError(
            f"Unknown task: '{args.task_name}'. Available: {list(cfg.keys())}"
        )

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name, cfg[args.task_name], args.mode
    )

    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")

    ace = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
    )

    run_test_samples = test_samples
    if args.mode == "offline" and args.skip_initial_test:
        print("⏭  Skipping initial test evaluation (--skip_initial_test)\n")
        run_test_samples = None

    ace_config = {
        "num_epochs":                    args.num_epochs,
        "max_num_rounds":                args.max_num_rounds,
        "curator_frequency":             args.curator_frequency,
        "eval_steps":                    args.eval_steps,
        "online_eval_frequency":         args.online_eval_frequency,
        "save_steps":                    args.save_steps,
        "playbook_token_budget":         args.playbook_token_budget,
        "task_name":                     args.task_name,
        "mode":                          args.mode,
        "json_mode":                     args.json_mode,
        "no_ground_truth":               args.no_ground_truth,
        "save_dir":                      args.save_path,
        "test_workers":                  args.test_workers,
        "initial_playbook_path":         args.initial_playbook_path,
        "use_bulletpoint_analyzer":      args.use_bulletpoint_analyzer,
        "bulletpoint_analyzer_threshold": args.bulletpoint_analyzer_threshold,
        "api_provider":                  args.api_provider,
    }

    results = ace.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=run_test_samples,
        data_processor=data_processor,
        config=ace_config,
    )
    print(f"\nFinal results: {results}")


if __name__ == "__main__":
    main()
