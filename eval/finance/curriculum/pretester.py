"""
Pre-test utility: run predictions on training pool WITHOUT ACE training.
Used to compute per-example and per-category difficulty scores.

Results are cached to disk so they don't need to be recomputed.
"""

import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

from utils import evaluate_single_test_sample, extract_answer


CACHE_DIR = "./eval/finance/data/pretest_cache"


def _pool_hash(pool: List[Dict]) -> str:
    """Stable hash of the pool for cache key."""
    ids = [s.get("_id", s["question"][:80]) for s in pool]
    return hashlib.md5(json.dumps(ids, sort_keys=True).encode()).hexdigest()[:12]


def run_pretest(
    pool: List[Dict],
    data_processor,
    generator,
    max_tokens: int = 512,
    max_workers: int = 30,
    log_dir: str = None,
    cache_tag: str = "default",
    force: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate the current model (empty playbook) on a pool of examples.

    Returns:
        example_difficulty: dict  _id → error score (1=wrong, 0=correct)
        cat_accuracy:       dict  macro_category → accuracy in [0,1]
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    ph = _pool_hash(pool)
    cache_path = os.path.join(CACHE_DIR, f"pretest_{cache_tag}_{ph}.json")

    if not force and os.path.exists(cache_path):
        print(f"[pretest] Loading cached results: {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["example_difficulty"], cached["cat_accuracy"]

    print(f"[pretest] Running pre-test on {len(pool)} examples "
          f"(workers={max_workers}) …")

    # Build args list for parallel evaluation
    args_list = [
        (i, s, generator, "", max_tokens, log_dir, False)
        for i, s in enumerate(pool)
    ]

    example_difficulty: Dict[str, float] = {}
    cat_correct: Dict[str, int] = {}
    cat_total:   Dict[str, int] = {}
    n_correct = 0

    def _eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, data_processor)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_wrapper, a): a for a in args_list}
        for future in as_completed(futures):
            res, err = future.result()
            if err or not res or not res.get("success"):
                continue
            idx = res["index"]
            s   = pool[idx]
            # Use s["_id"] if set, otherwise fall back to pool position index
            _id = s.get("_id", idx)
            correct = res["is_correct"]
            example_difficulty[_id] = 0.0 if correct else 1.0
            n_correct += int(correct)

            cat = s.get("macro_category", "Other")
            cat_correct[cat] = cat_correct.get(cat, 0) + int(correct)
            cat_total[cat]   = cat_total.get(cat, 0) + 1

    overall_acc = n_correct / len(pool) if pool else 0.0
    cat_accuracy = {
        cat: cat_correct[cat] / cat_total[cat]
        for cat in cat_total
    }

    print(f"[pretest] Overall acc: {overall_acc:.4f}")
    print("[pretest] Per-category accuracy:")
    for cat, acc in sorted(cat_accuracy.items(), key=lambda x: x[1]):
        n = cat_total[cat]
        print(f"  {cat:<30}  {acc:.3f}  (n={n})")

    # Cache
    with open(cache_path, "w") as f:
        json.dump({
            "example_difficulty": example_difficulty,
            "cat_accuracy": cat_accuracy,
            "overall_acc": overall_acc,
        }, f, indent=2)
    print(f"[pretest] Cached → {cache_path}")

    return example_difficulty, cat_accuracy
