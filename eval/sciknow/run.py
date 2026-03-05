#!/usr/bin/env python3
"""
ACE training / evaluation script for SciKnowEval Chemistry (L3+L4 MCQ).

Usage examples:
  # Baseline LLM (no playbook)
  python -m eval.sciknow.run --task_name sciknow_chem --mode eval_only --save_path results/sciknow_baseline

  # ACE full train
  python -m eval.sciknow.run --task_name sciknow_chem --mode offline --save_path results/sciknow_full_train
"""
import os
import json
import argparse
from .data_processor import DataProcessor, load_data

from ace import ACE


def parse_args():
    parser = argparse.ArgumentParser(description="ACE – SciKnowEval Chemistry")

    # Task
    parser.add_argument("--task_name", type=str, default="sciknow_chem")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"])
    parser.add_argument("--config_path", type=str,
                        default="./eval/sciknow/data/sample_config.json")
    parser.add_argument("--initial_playbook_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)

    # Model
    parser.add_argument("--api_provider", type=str, default="together",
                        choices=["sambanova", "together", "openai"])
    parser.add_argument("--generator_model", type=str,
                        default="deepseek-ai/DeepSeek-V3.1")
    parser.add_argument("--reflector_model", type=str,
                        default="deepseek-ai/DeepSeek-V3.1")
    parser.add_argument("--curator_model", type=str,
                        default="deepseek-ai/DeepSeek-V3.1")

    # Training
    parser.add_argument("--num_epochs",          type=int,   default=1)
    parser.add_argument("--max_num_rounds",       type=int,   default=3)
    parser.add_argument("--curator_frequency",    type=int,   default=1)
    parser.add_argument("--eval_steps",           type=int,   default=100)
    parser.add_argument("--online_eval_frequency",type=int,   default=15)
    parser.add_argument("--save_steps",           type=int,   default=50)
    parser.add_argument("--max_tokens",           type=int,   default=8192)
    parser.add_argument("--playbook_token_budget",type=int,   default=80000)
    parser.add_argument("--test_workers",         type=int,   default=20)
    parser.add_argument("--json_mode",            action="store_true")
    parser.add_argument("--no_ground_truth",      action="store_true")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90)

    return parser.parse_args()


def preprocess_data(task_name, config, mode):
    processor = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = val_samples = None
        test_samples  = load_data(config["test_data"])
        test_samples  = processor.process_task_data(test_samples)
        print(f"{'Online' if mode=='online' else 'Eval-only'} mode: "
              f"{len(test_samples)} test examples")
    else:
        train_samples = processor.process_task_data(load_data(config["train_data"]))
        val_samples   = processor.process_task_data(load_data(config["val_data"]))
        test_samples  = processor.process_task_data(load_data(config["test_data"])) \
                        if "test_data" in config else []
        print(f"Offline mode: {len(train_samples)} train / "
              f"{len(val_samples)} val / {len(test_samples)} test")

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"ACE – SciKnowEval Chemistry (L3+L4 MCQ)")
    print(f"{'='*60}")
    print(f"Task  : {args.task_name}")
    print(f"Mode  : {args.mode.upper()}")
    print(f"Model : {args.generator_model}  API: {args.api_provider}")
    print(f"{'='*60}\n")

    with open(args.config_path, 'r') as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name, task_config[args.task_name], args.mode
    )

    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    print(f"Initial playbook: {'loaded from ' + args.initial_playbook_path if initial_playbook else 'empty'}\n")

    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
    )

    config = {
        'num_epochs':               args.num_epochs,
        'max_num_rounds':           args.max_num_rounds,
        'curator_frequency':        args.curator_frequency,
        'eval_steps':               args.eval_steps,
        'online_eval_frequency':    args.online_eval_frequency,
        'save_steps':               args.save_steps,
        'playbook_token_budget':    args.playbook_token_budget,
        'task_name':                args.task_name,
        'mode':                     args.mode,
        'json_mode':                args.json_mode,
        'no_ground_truth':          args.no_ground_truth,
        'save_dir':                 args.save_path,
        'test_workers':             args.test_workers,
        'initial_playbook_path':    args.initial_playbook_path,
        'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
        'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
        'api_provider':             args.api_provider,
    }

    ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config,
    )


if __name__ == "__main__":
    main()
