#!/usr/bin/env python3
"""
Run script for ACE on SVAMP.

Example:
  python -m eval.svamp.prepare_data

  # 0-shot baseline
  python -m eval.svamp.run --task_name svamp --mode eval_only --save_path results/svamp_baseline
"""

import argparse
import json
import os

from ace import ACE
from .data_processor import DataProcessor, load_data


def parse_args():
    p = argparse.ArgumentParser(description="ACE System - SVAMP")

    p.add_argument("--task_name", type=str, required=True)
    p.add_argument("--initial_playbook_path", type=str, default=None)
    p.add_argument("--mode", type=str, default="offline", choices=["offline", "online", "eval_only"])

    p.add_argument("--api_provider", type=str, default="sambanova", choices=["sambanova", "together", "openai"])
    p.add_argument("--generator_model", type=str, default="DeepSeek-V3.1")
    p.add_argument("--reflector_model", type=str, default="DeepSeek-V3.1")
    p.add_argument("--curator_model", type=str, default="DeepSeek-V3.1")

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_num_rounds", type=int, default=3)
    p.add_argument("--curator_frequency", type=int, default=1)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--online_eval_frequency", type=int, default=15)
    p.add_argument("--save_steps", type=int, default=50)

    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--playbook_token_budget", type=int, default=80000)
    p.add_argument("--test_workers", type=int, default=20)

    p.add_argument("--json_mode", action="store_true")
    p.add_argument("--no_ground_truth", action="store_true")

    p.add_argument("--use_bulletpoint_analyzer", action="store_true")
    p.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90)

    p.add_argument("--skip_initial_test", action="store_true")
    p.add_argument("--save_path", type=str, required=True)

    return p.parse_args()


def preprocess_data(task_name, config, mode):
    proc = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        if "test_data" not in config:
            raise ValueError(f"{mode} mode requires test data in config.")
        test_samples = proc.process_task_data(load_data(config["test_data"]))
        print(f"{'Online' if mode == 'online' else 'Eval only'} mode: Testing on {len(test_samples)} examples")
    else:
        train_samples = proc.process_task_data(load_data(config["train_data"]))
        val_samples = proc.process_task_data(load_data(config["val_data"]))
        test_samples = []
        if "test_data" in config:
            test_samples = proc.process_task_data(load_data(config["test_data"]))
        print(f"Offline mode: Training on {len(train_samples)} examples, validating on {len(val_samples)}, testing on {len(test_samples)}")

    return train_samples, val_samples, test_samples, proc


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("ACE SYSTEM - SVAMP")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"{'='*60}\n")

    with open("./eval/svamp/data/sample_config.json", "r") as f:
        cfg = json.load(f)

    if args.task_name not in cfg:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {list(cfg.keys())}")

    train_samples, val_samples, test_samples, data_processor = preprocess_data(args.task_name, cfg[args.task_name], args.mode)

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
        print("⏭️  Skipping test evaluation (--skip_initial_test)\n")
        print("   Run eval_only with --initial_playbook_path to test the learned playbook.\n")
        run_test_samples = None

    config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "online_eval_frequency": args.online_eval_frequency,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "task_name": args.task_name,
        "mode": args.mode,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "initial_playbook_path": args.initial_playbook_path,
        "use_bulletpoint_analyzer": args.use_bulletpoint_analyzer,
        "bulletpoint_analyzer_threshold": args.bulletpoint_analyzer_threshold,
        "api_provider": args.api_provider,
    }

    results = ace.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=run_test_samples,
        data_processor=data_processor,
        config=config,
    )
    print(f"\nFinal results: {results}")


if __name__ == "__main__":
    main()

