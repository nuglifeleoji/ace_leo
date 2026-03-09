"""
Offline Bandit Selector (Phase 3).

Uses collected experience from Phase 1 runs to train a scoring model
that predicts expected reward (val accuracy improvement) for each category.

Training data format (from result.json step_log):
  features: [step/budget, cat_one_hot, rolling_err_rate_per_cat]
  target: val accuracy improvement at next checkpoint

Pipeline:
  1. Load all Phase 1 result.json files
  2. Construct (state, action, reward) triples
  3. Train a linear regression model to score (state, category) → expected reward
  4. Use this model online to select the best category at each step

This is the "offline warmstart" from the problem statement.
"""

import os
import json
import math
import random
import pickle
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple

from .selectors import BaseSelector

# ────────────────────────────────────────────────────────────────
# Feature extraction
# ────────────────────────────────────────────────────────────────

ALL_CATEGORIES = [
    "Debt_Financing", "Equity_Shares", "ShareBased_Compensation",
    "Tax", "Revenue_Contracts", "MA_BusinessCombinations",
    "Leases_RealEstate", "Intangibles_Goodwill_PPE", "Restructuring_Impairment",
    "Investments_Segments", "Compensation_Benefits", "Contingencies_Legal_Other",
    "Other",
]
N_CATS = len(ALL_CATEGORIES)
CAT_IDX = {c: i for i, c in enumerate(ALL_CATEGORIES)}


def state_action_features(
    step: int,
    budget: int,
    cat_totals: Dict[str, int],
    cat_errors: Dict[str, int],
    action_cat: str,
) -> List[float]:
    """
    Feature vector for (state, action):
      - step / budget                           (1)
      - category error rate (rolling)           (N_CATS)
      - log(1 + n_steps_for_action_cat)         (1)
      - action cat one-hot                      (N_CATS)
    Total: 1 + N_CATS + 1 + N_CATS = 2*N_CATS + 2
    """
    feats = [step / max(budget, 1)]

    for cat in ALL_CATEGORIES:
        n = cat_totals.get(cat, 0)
        e = cat_errors.get(cat, 0)
        err_rate = e / n if n > 0 else 0.5
        feats.append(err_rate)

    # Log steps for action cat
    feats.append(math.log1p(cat_totals.get(action_cat, 0)))

    # One-hot for action
    one_hot = [0.0] * N_CATS
    if action_cat in CAT_IDX:
        one_hot[CAT_IDX[action_cat]] = 1.0
    feats.extend(one_hot)

    return feats


def extract_training_data(
    result_paths: List[str],
    eval_every: int = 25,
) -> Tuple[List[List[float]], List[float]]:
    """
    Build (features, reward) dataset from Phase 1 results.

    For each step in [t, t+eval_every), compute:
      reward = val_acc[t+eval_every] - val_acc[t]

    Assign this reward to all steps in [t, t+eval_every).
    """
    X, y = [], []

    for path in result_paths:
        with open(path) as f:
            r = json.load(f)

        step_log = r.get("step_log", [])
        val_curve = r.get("val_curve", [])
        budget    = r.get("steps_used", 200)

        # Build val_acc lookup
        val_map = {pt["step"]: pt["val_acc"] for pt in val_curve}

        # Build rolling state
        cat_totals: Dict[str, int] = defaultdict(int)
        cat_errors: Dict[str, int] = defaultdict(int)

        # Segment steps into eval windows
        checkpoints = sorted(val_map.keys())

        for i, pt in enumerate(step_log):
            step = pt["step"]
            cat  = pt["cat"]

            # Find next checkpoint after this step
            next_ck = next((ck for ck in checkpoints if ck > step), None)
            prev_ck = max((ck for ck in checkpoints if ck <= step), default=None)

            if next_ck is None or prev_ck is None:
                cat_totals[cat] += 1
                cat_errors[cat] += int(not pt["correct"])
                continue

            val_before = val_map.get(prev_ck, r["baseline_test_acc"])
            val_after  = val_map.get(next_ck, val_before)
            reward     = val_after - val_before

            feats = state_action_features(step, budget, cat_totals, cat_errors, cat)
            X.append(feats)
            y.append(reward)

            cat_totals[cat] += 1
            cat_errors[cat] += int(not pt["correct"])

    return X, y


def train_linear_bandit(X: List[List[float]], y: List[float]) -> Dict:
    """
    Train a simple linear regression model: w^T x → reward.
    Uses ordinary least squares via numpy.
    """
    import numpy as np
    Xm = np.array(X, dtype=float)
    ym = np.array(y, dtype=float)

    # L2 regularization to avoid overfitting
    lam = 0.1
    n, d = Xm.shape
    A = Xm.T @ Xm + lam * np.eye(d)
    b = Xm.T @ ym
    w = np.linalg.solve(A, b)

    # Compute train R²
    y_pred = Xm @ w
    ss_res = np.sum((ym - y_pred) ** 2)
    ss_tot = np.sum((ym - np.mean(ym)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return {"weights": w.tolist(), "r2": float(r2), "n_train": n, "d": d}


# ────────────────────────────────────────────────────────────────
# Online selector using trained model
# ────────────────────────────────────────────────────────────────

class OfflineBanditSelector(BaseSelector):
    """
    Online selection using a pre-trained linear scoring model.
    At each step, scores all available categories and picks the best one.
    """

    def __init__(
        self,
        model: Dict,
        budget: int = 200,
        seed: int = 42,
        epsilon: float = 0.1,   # ε-greedy exploration
    ):
        import numpy as np
        self._w      = np.array(model["weights"])
        self._budget = budget
        self._eps    = epsilon
        self._rng    = random.Random(seed)
        self._step   = 0
        self._cat_totals: Dict[str, int] = defaultdict(int)
        self._cat_errors: Dict[str, int] = defaultdict(int)
        self._cat_pools: Optional[Dict[str, List[Dict]]] = None

    @property
    def name(self): return "offline_bandit"

    def _init(self, pool):
        self._cat_pools = defaultdict(list)
        for s in pool:
            self._cat_pools[s["macro_category"]].append(s)
        for cat in self._cat_pools:
            self._rng.shuffle(self._cat_pools[cat])

    def _score(self, cat: str) -> float:
        import numpy as np
        feats = state_action_features(
            self._step, self._budget, self._cat_totals, self._cat_errors, cat
        )
        return float(np.dot(self._w, feats))

    def select(self, pool):
        if self._cat_pools is None:
            self._init(pool)
        self._step += 1

        available = [c for c, s in self._cat_pools.items() if s]
        if not available:
            return None

        # ε-greedy
        if self._rng.random() < self._eps:
            cat = self._rng.choice(available)
        else:
            cat = max(available, key=self._score)

        return self._cat_pools[cat].pop(0)

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        self._cat_totals[cat] += 1
        self._cat_errors[cat] += int(not is_correct)

    def category_stats(self):
        return {
            cat: {
                "total":    self._cat_totals[cat],
                "error_rate": round(self._cat_errors[cat] / max(self._cat_totals[cat], 1), 3),
                "score":    round(self._score(cat), 5),
            }
            for cat in self._cat_totals
        }


def build_offline_bandit_from_results(
    results_root: str,
    eval_every: int = 25,
    save_model_path: Optional[str] = None,
) -> Dict:
    """
    End-to-end: load Phase 1 results, extract data, train model, return model dict.
    """
    result_paths = [
        os.path.join(results_root, d, "result.json")
        for d in os.listdir(results_root)
        if os.path.isfile(os.path.join(results_root, d, "result.json"))
    ]

    print(f"[offline_bandit] Loading {len(result_paths)} Phase 1 results...")
    X, y = extract_training_data(result_paths, eval_every=eval_every)
    print(f"[offline_bandit] Dataset: {len(X)} (state, action, reward) triples")

    model = train_linear_bandit(X, y)
    print(f"[offline_bandit] Linear model R²={model['r2']:.4f} "
          f"(n={model['n_train']}, d={model['d']})")

    if save_model_path:
        with open(save_model_path, "w") as f:
            json.dump(model, f, indent=2)
        print(f"[offline_bandit] Model saved → {save_model_path}")

    return model
