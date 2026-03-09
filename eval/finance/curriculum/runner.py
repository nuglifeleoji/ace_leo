"""
CurriculumRunner: fixed-budget training loop with any BaseSelector.

Per-step logging:
  - step, category, label, correct (pre-training), playbook_bullets
  - val accuracy every eval_every steps

Final output dict:
  {
    "selector": name,
    "baseline_test_acc": float,
    "final_test_acc": float,
    "delta": float,
    "steps_used": int,
    "val_curve": [ {"step": int, "val_acc": float}, ... ],
    "step_log": [ {"step": int, "cat": str, "label": str, "correct": bool}, ... ],
    "cat_stats": { cat: {...} },
  }
"""

import os
import json
import time
from typing import List, Dict, Any

from ace import ACE
from utils import evaluate_test_set

from .selectors import BaseSelector


class CurriculumRunner:

    def __init__(
        self,
        ace_system: ACE,
        selector: BaseSelector,
        data_processor,
        train_pool: List[Dict],        # single-question format
        val_samples: List[Dict],       # batched format, for online monitoring
        test_samples: List[Dict],      # batched format, for final eval
        budget: int = 200,
        eval_every: int = 25,
        max_tokens: int = 4096,
        log_dir: str = "./tmp_logs",
        save_path: str = "./tmp_save",
        config_params: Dict[str, Any] = None,
    ):
        self.ace = ace_system
        self.selector = selector
        self.dp = data_processor
        self.train_pool = list(train_pool)   # working copy
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.budget = budget
        self.eval_every = eval_every
        self.max_tokens = max_tokens
        self.log_dir = log_dir
        self.save_path = save_path
        self.config_params = config_params or {}

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        self._usage_log = os.path.join(save_path, "bullet_usage.jsonl")

    # ── Evaluation helpers ─────────────────────────────────────────────────

    def _eval(self, samples: List[Dict], prefix: str) -> float:
        results, _ = evaluate_test_set(
            data_processor=self.dp,
            generator=self.ace.generator,
            playbook=self.ace.playbook,
            test_samples=samples,
            max_tokens=self.max_tokens,
            log_dir=self.log_dir,
            max_workers=20,
            use_json_mode=False,
        )
        acc = results["accuracy"]
        print(f"  [{prefix}] acc={acc:.4f}  ({int(acc*len(samples))}/{len(samples)})")
        return acc

    # ── Main run ───────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        sel_name = self.selector.name
        print(f"\n{'='*65}")
        print(f"CurriculumRunner  selector={sel_name}  budget={self.budget}")
        print(f"{'='*65}")

        # Baseline test
        print("\n[Baseline test eval]")
        baseline_acc = self._eval(self.test_samples, "test-baseline")

        val_curve = [{"step": 0, "val_acc": self._eval(self.val_samples, "val-step0")}]
        step_log: List[Dict] = []

        # Track best playbook for optional early-stop / best-playbook restore
        best_val_acc = val_curve[0]["val_acc"]
        best_playbook = self.ace.playbook
        best_step = 0
        patience_counter = 0
        patience = self.config_params.get("val_patience", 0)          # 0 = disabled
        use_best_for_test = self.config_params.get("use_best_for_test", False)

        t0 = time.time()
        pool_snapshot = list(self.train_pool)   # selector sees full pool; pops internally

        for step in range(1, self.budget + 1):
            example = self.selector.select(pool_snapshot)
            if example is None:
                print(f"  [runner] Pool exhausted at step {step}. Stopping.")
                break

            # ACE training step
            _, _, tracking = self.ace._train_single_sample(
                task_dict      = example,
                data_processor = self.dp,
                step_id        = f"{sel_name}_s{step}",
                epoch          = 1,
                step           = step,
                usage_log_path = self._usage_log,
                log_dir        = self.log_dir,
                config_params  = self.config_params,
                total_samples  = self.budget,
            )

            pre_correct = tracking["pre_train_result"]["is_correct"]
            self.selector.update(example, pre_correct)

            cat   = example.get("macro_category", "Other")
            label = example.get("label", "?")
            n_bullets = self.ace.playbook.count("\n[") if self.ace.playbook else 0

            step_log.append({
                "step": step,
                "cat": cat,
                "label": label,
                "correct": pre_correct,
                "n_bullets": n_bullets,
            })

            elapsed = time.time() - t0
            print(f"  step {step:3d}/{self.budget} | {cat:<28} | {label[:35]:<35} | "
                  f"{'✓' if pre_correct else '✗'} | bullets={n_bullets} | "
                  f"elapsed={elapsed/60:.1f}m")

            # Periodic val evaluation
            if step % self.eval_every == 0:
                val_acc = self._eval(self.val_samples, f"val-step{step}")
                val_curve.append({"step": step, "val_acc": val_acc})

                # Track best playbook
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_playbook = self.ace.playbook
                    best_step = step
                    patience_counter = 0
                    print(f"  [best] New best val={best_val_acc:.4f} at step {step}")
                else:
                    patience_counter += 1

                # Notify selector of val result (for adaptive phase switching)
                if hasattr(self.selector, "notify_val"):
                    self.selector.notify_val(val_acc)

                # Early stopping
                if patience > 0 and patience_counter >= patience:
                    print(f"  [early-stop] No improvement for {patience} checks. "
                          f"Stopping at step {step}. Best was step {best_step}.")
                    break

        # Final evaluations — use best playbook if requested
        use_best = (patience > 0 or use_best_for_test)
        if use_best and best_step > 0 and best_playbook != self.ace.playbook:
            print(f"\n[Using best playbook from step {best_step} "
                  f"(val={best_val_acc:.4f}) for final eval]")
            self.ace.playbook = best_playbook

        print("\n[Final test eval]")
        final_acc = self._eval(self.test_samples, "test-final")

        # Save playbook
        pb_path = os.path.join(self.save_path, "final_playbook.txt")
        with open(pb_path, "w") as f:
            f.write(self.ace.playbook)

        # Also save best playbook separately if different
        if use_best and best_playbook:
            best_pb_path = os.path.join(self.save_path, "best_playbook.txt")
            with open(best_pb_path, "w") as f:
                f.write(best_playbook)
            print(f"  Best playbook (step {best_step}) saved → {best_pb_path}")

        result = {
            "selector":          sel_name,
            "baseline_test_acc": baseline_acc,
            "final_test_acc":    final_acc,
            "delta":             final_acc - baseline_acc,
            "steps_used":        len(step_log),
            "best_val_acc":      best_val_acc,
            "best_step":         best_step,
            "val_curve":         val_curve,
            "step_log":          step_log,
            "cat_stats":         self.selector.category_stats(),
            "runtime_minutes":   round((time.time() - t0) / 60, 2),
        }

        # Save JSON
        res_path = os.path.join(self.save_path, "result.json")
        with open(res_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*65}")
        print(f"RESULT  selector={sel_name}")
        print(f"  Baseline : {baseline_acc:.4f}")
        print(f"  Final    : {final_acc:.4f}")
        print(f"  Delta    : {final_acc - baseline_acc:+.4f}")
        print(f"  Saved    : {self.save_path}")
        print(f"{'='*65}\n")

        return result
