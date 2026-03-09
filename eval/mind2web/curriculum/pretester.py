"""
Pre-test utility for Mind2Web: compute per-example difficulty from
existing train_correctness.json (zero-shot baseline) or by running
the generator live.

mind2web/data/train_correctness.json has already been computed and
contains {str(idx): 0_or_1} for all 4476 training samples.
We use this as a free difficulty cache — no API calls needed.
"""

import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

from utils import evaluate_single_test_sample

CACHE_DIR = "./eval/mind2web/data/pretest_cache"
PRECOMPUTED_PATH = "./eval/mind2web/data/train_correctness.json"


def load_precomputed_difficulty(
    pool: List[Dict],
) -> Tuple[Dict, Dict]:
    """
    Load difficulty from the precomputed train_correctness.json file.
    Keys in that file are str(original_line_index).
    Each sample must have '_id' equal to its original line index.

    Returns:
        example_difficulty: {_id -> 0.0 (easy) or 1.0 (hard)}
        cat_accuracy:       {domain -> accuracy}
    """
    if not os.path.exists(PRECOMPUTED_PATH):
        return {}, {}

    with open(PRECOMPUTED_PATH) as f:
        raw = json.load(f)

    example_difficulty: Dict = {}
    cat_correct: Dict[str, int] = {}
    cat_total:   Dict[str, int] = {}

    for s in pool:
        _id = s.get("_id")
        if _id is None:
            continue
        # Use orig_idx for lookup in train_correctness.json (which uses line index)
        lookup_key = str(s.get("orig_idx", _id))
        correctness = raw.get(lookup_key)
        if correctness is None:
            continue
        example_difficulty[_id] = 0.0 if correctness == 1 else 1.0
        cat = s.get("macro_category", "Other")
        cat_correct[cat] = cat_correct.get(cat, 0) + int(correctness == 1)
        cat_total[cat]   = cat_total.get(cat, 0) + 1

    cat_accuracy = {
        cat: cat_correct[cat] / cat_total[cat]
        for cat in cat_total if cat_total[cat] > 0
    }

    n_easy = sum(1 for v in example_difficulty.values() if v == 0.0)
    print(f"[pretest] Loaded precomputed: {len(example_difficulty)} samples, "
          f"{n_easy} easy ({n_easy/max(len(example_difficulty),1):.1%})")
    print("[pretest] Per-category accuracy (zero-shot baseline):")
    for cat, acc in sorted(cat_accuracy.items(), key=lambda x: x[1]):
        print(f"  {cat:<25}  {acc:.3f}  (n={cat_total[cat]})")

    return example_difficulty, cat_accuracy


def run_pretest(
    pool: List[Dict],
    data_processor,
    generator,
    max_tokens: int = 512,
    max_workers: int = 30,
    log_dir: str = None,
    cache_tag: str = "default",
    force: bool = False,
    use_precomputed: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Compute difficulty scores for a pool of Mind2Web samples.

    First tries the precomputed train_correctness.json; falls back to
    live evaluation if needed.
    """
    # Try precomputed first (free, covers entire training set)
    if use_precomputed and not force:
        diff, cat_acc = load_precomputed_difficulty(pool)
        if len(diff) >= len(pool) * 0.5:
            return diff, cat_acc
        print(f"[pretest] Precomputed coverage too low ({len(diff)}/{len(pool)}), "
              "falling back to live evaluation.")

    # Live evaluation fallback
    os.makedirs(CACHE_DIR, exist_ok=True)
    ids = [s.get("_id", i) for i, s in enumerate(pool)]
    ph = hashlib.md5(json.dumps(ids, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = os.path.join(CACHE_DIR, f"pretest_{cache_tag}_{ph}.json")

    if not force and os.path.exists(cache_path):
        print(f"[pretest] Loading cached results: {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["example_difficulty"], cached["cat_accuracy"]

    print(f"[pretest] Running live pre-test on {len(pool)} examples …")
    args_list = [
        (i, s, generator, "", max_tokens, log_dir, False)
        for i, s in enumerate(pool)
    ]

    example_difficulty: Dict = {}
    cat_correct: Dict[str, int] = {}
    cat_total:   Dict[str, int] = {}
    n_correct = 0

    def _eval_wrapper(a):
        return evaluate_single_test_sample(a, data_processor)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_wrapper, a): a for a in args_list}
        for fut in as_completed(futures):
            res, err = fut.result()
            if err or not res or not res.get("success"):
                continue
            idx = res["index"]
            s   = pool[idx]
            _id = s.get("_id", idx)
            correct = res["is_correct"]
            example_difficulty[_id] = 0.0 if correct else 1.0
            n_correct += int(correct)
            cat = s.get("macro_category", "Other")
            cat_correct[cat] = cat_correct.get(cat, 0) + int(correct)
            cat_total[cat]   = cat_total.get(cat, 0) + 1

    cat_accuracy = {cat: cat_correct[cat] / cat_total[cat] for cat in cat_total}
    print(f"[pretest] Overall acc: {n_correct/len(pool):.4f}")

    with open(cache_path, "w") as f:
        json.dump({"example_difficulty": example_difficulty,
                   "cat_accuracy": cat_accuracy}, f, indent=2)
    return example_difficulty, cat_accuracy
