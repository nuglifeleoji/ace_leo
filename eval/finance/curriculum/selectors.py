"""
Curriculum selectors for FiNER XBRL tagging.

All selectors share the interface:
  selector.select(pool)         → next Dict (or None if exhausted)
  selector.update(ex, correct)  → None  (update internal state)
  selector.name                 → str

Pool items must have keys: context, question, target, label, macro_category.
"""

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────

class BaseSelector(ABC):

    @abstractmethod
    def select(self, pool: List[Dict]) -> Optional[Dict]:
        """Pop and return next example, or None if pool exhausted."""

    def update(self, example: Dict, is_correct: bool):
        """Called after each training step with the outcome."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def category_stats(self) -> Dict[str, Dict]:
        """Return per-category stats dict (for logging). Override if needed."""
        return {}


# ──────────────────────────────────────────────────────────────
# 1. Random (baseline)
# ──────────────────────────────────────────────────────────────

class RandomSelector(BaseSelector):
    """Uniform random sampling without replacement."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._queue: Optional[List[Dict]] = None

    @property
    def name(self): return "random"

    def _init(self, pool):
        self._queue = list(pool)
        self._rng.shuffle(self._queue)

    def select(self, pool):
        if self._queue is None:
            self._init(pool)
        return self._queue.pop(0) if self._queue else None


# ──────────────────────────────────────────────────────────────
# 2. Stratified round-robin over macro categories
# ──────────────────────────────────────────────────────────────

class StratifiedSelector(BaseSelector):
    """Cycles over macro categories alphabetically, random within each."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._cat_pools: Optional[Dict[str, List[Dict]]] = None
        self._cat_order: Optional[List[str]] = None
        self._cat_idx: int = 0

    @property
    def name(self): return "stratified"

    def _init(self, pool):
        self._cat_pools = defaultdict(list)
        for s in pool:
            self._cat_pools[s["macro_category"]].append(s)
        for cat in self._cat_pools:
            self._rng.shuffle(self._cat_pools[cat])
        self._cat_order = sorted(self._cat_pools.keys())
        self._cat_idx = 0

    def select(self, pool):
        if self._cat_pools is None:
            self._init(pool)
        n = len(self._cat_order)
        for _ in range(n):
            cat = self._cat_order[self._cat_idx % n]
            self._cat_idx += 1
            if self._cat_pools[cat]:
                return self._cat_pools[cat].pop(0)
        return None


# ──────────────────────────────────────────────────────────────
# 3. Hard-first (difficulty curriculum, pre-test based)
# ──────────────────────────────────────────────────────────────

class HardFirstSelector(BaseSelector):
    """Train on wrong examples first (sorted by pre-test error)."""

    def __init__(self, difficulty: Dict[str, float], seed: int = 42):
        """
        difficulty: maps example _id → score in [0,1] where 1 = always wrong.
        Examples without a score get score 0.5.
        """
        self._difficulty = difficulty
        self._rng = random.Random(seed)
        self._queue: Optional[List[Dict]] = None

    @property
    def name(self): return "hard_first"

    def _key(self, s: Dict) -> float:
        base = self._difficulty.get(s.get("_id", ""), 0.5)
        return -(base + self._rng.random() * 1e-6)

    def _init(self, pool):
        self._queue = sorted(pool, key=self._key)

    def select(self, pool):
        if self._queue is None:
            self._init(pool)
        return self._queue.pop(0) if self._queue else None


# ──────────────────────────────────────────────────────────────
# 4. Easy-first (clean insight generation)
# ──────────────────────────────────────────────────────────────

class EasyFirstSelector(BaseSelector):
    """Train on correct examples first → clean, stable insight generation."""

    def __init__(self, difficulty: Dict[str, float], seed: int = 42):
        self._difficulty = difficulty
        self._rng = random.Random(seed)
        self._queue: Optional[List[Dict]] = None

    @property
    def name(self): return "easy_first"

    def _key(self, s: Dict) -> float:
        base = self._difficulty.get(s.get("_id", ""), 0.5)
        return base + self._rng.random() * 1e-6   # ascending = easiest first

    def _init(self, pool):
        self._queue = sorted(pool, key=self._key)

    def select(self, pool):
        if self._queue is None:
            self._init(pool)
        return self._queue.pop(0) if self._queue else None


# ──────────────────────────────────────────────────────────────
# 5. UCB-Category bandit
# ──────────────────────────────────────────────────────────────

class UCBCategorySelector(BaseSelector):
    """
    UCB1 bandit over macro categories.

    score(cat) = error_rate(cat) + c * sqrt(log(N+1) / n_cat)

    High error rate → need more training here.
    High exploration bonus → underexplored categories get a chance.
    """

    def __init__(self, seed: int = 42, c: float = 1.0):
        self._rng = random.Random(seed)
        self.c = c
        self._cat_correct: Dict[str, int] = defaultdict(int)
        self._cat_total: Dict[str, int] = defaultdict(int)
        self._total_steps: int = 0
        self._cat_pools: Optional[Dict[str, List[Dict]]] = None

    @property
    def name(self): return f"ucb_cat_c{self.c}"

    def _init(self, pool):
        self._cat_pools = defaultdict(list)
        for s in pool:
            self._cat_pools[s["macro_category"]].append(s)
        for cat in self._cat_pools:
            self._rng.shuffle(self._cat_pools[cat])

    def _ucb(self, cat: str) -> float:
        n = self._cat_total[cat]
        if n == 0:
            return float("inf")
        err = 1.0 - self._cat_correct[cat] / n
        return err + self.c * math.sqrt(math.log(self._total_steps + 1) / n)

    def select(self, pool):
        if self._cat_pools is None:
            self._init(pool)
        available = [c for c, s in self._cat_pools.items() if s]
        if not available:
            return None
        best = max(available, key=self._ucb)
        return self._cat_pools[best].pop(0)

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        self._cat_correct[cat] += int(is_correct)
        self._cat_total[cat] += 1
        self._total_steps += 1

    def category_stats(self):
        return {
            cat: {
                "correct": self._cat_correct[cat],
                "total": self._cat_total[cat],
                "error_rate": round(1 - self._cat_correct[cat] / max(self._cat_total[cat], 1), 3),
                "ucb": round(self._ucb(cat), 4),
            }
            for cat in self._cat_total
        }


# ──────────────────────────────────────────────────────────────
# 6. Error-focused greedy (exploitation only, no exploration)
# ──────────────────────────────────────────────────────────────

class ErrorFocusedSelector(BaseSelector):
    """
    Online greedy: always select from the category with the highest
    recent error rate in a rolling window. No exploration bonus.
    """

    def __init__(self, window: int = 12, seed: int = 42):
        self._rng = random.Random(seed)
        self._window = window
        self._cat_win: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._cat_pools: Optional[Dict[str, List[Dict]]] = None

    @property
    def name(self): return f"error_focused_w{self._window}"

    def _init(self, pool):
        self._cat_pools = defaultdict(list)
        for s in pool:
            self._cat_pools[s["macro_category"]].append(s)
        for cat in self._cat_pools:
            self._rng.shuffle(self._cat_pools[cat])

    def _err(self, cat: str) -> float:
        w = self._cat_win[cat]
        return 1.0 - (sum(w) / len(w)) if w else 1.0  # unknown → treat as hard

    def select(self, pool):
        if self._cat_pools is None:
            self._init(pool)
        available = [c for c, s in self._cat_pools.items() if s]
        if not available:
            return None
        best = max(available, key=self._err)
        return self._cat_pools[best].pop(0)

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        self._cat_win[cat].append(int(is_correct))

    def category_stats(self):
        return {
            cat: {
                "window": list(self._cat_win[cat]),
                "error_rate": round(self._err(cat), 3),
            }
            for cat in self._cat_win
        }


# ──────────────────────────────────────────────────────────────
# 7. Category-difficulty curriculum (hardest category first,
#    pre-test based on category-level accuracy)
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# 8. Hybrid: Hard-first warmup → UCB-Category exploitation
# ──────────────────────────────────────────────────────────────

class HybridHardUCBSelector(BaseSelector):
    """
    Phase 1 (steps 1..warmup_steps): Hard-first (pre-test difficulty)
      → Build playbook with actual wrong-answer corrections.
    Phase 2 (steps warmup_steps+1..budget): UCB-Category
      → Adaptively focus on categories where errors remain.

    Motivation: Hard-first creates informative playbook quickly,
    then UCB exploits that playbook to fill remaining gaps.
    """

    def __init__(self, difficulty: Dict[str, float], budget: int = 200,
                 warmup_steps: int = 50, c: float = 0.5, seed: int = 42):
        self._difficulty    = difficulty
        self._warmup_steps  = warmup_steps
        self._budget        = budget
        self._step          = 0
        self._rng           = random.Random(seed)
        self._hard_queue:   Optional[List[Dict]] = None
        self._ucb_pools:    Optional[Dict[str, List[Dict]]] = None
        self._cat_correct:  Dict[str, int] = defaultdict(int)
        self._cat_total:    Dict[str, int] = defaultdict(int)
        self._total:        int = 0
        self.c              = c

    @property
    def name(self): return f"hybrid_hard{self._warmup_steps}_ucb_c{self.c}"

    def _init_hard(self, pool):
        def key(s):
            return -(self._difficulty.get(s.get("_id", ""), 0.5) + self._rng.random() * 1e-6)
        self._hard_queue = sorted(pool, key=key)

    def _init_ucb(self, pool):
        self._ucb_pools = defaultdict(list)
        for s in pool:
            self._ucb_pools[s["macro_category"]].append(s)
        for cat in self._ucb_pools:
            self._rng.shuffle(self._ucb_pools[cat])

    def _ucb_score(self, cat: str) -> float:
        n = self._cat_total[cat]
        if n == 0:
            return float("inf")
        err = 1.0 - self._cat_correct[cat] / n
        return err + self.c * math.sqrt(math.log(self._total + 1) / n)

    def select(self, pool):
        self._step += 1
        if self._step <= self._warmup_steps:
            # Phase 1: hard-first
            if self._hard_queue is None:
                self._init_hard(pool)
            return self._hard_queue.pop(0) if self._hard_queue else None
        else:
            # Phase 2: UCB
            if self._ucb_pools is None:
                remaining = [s for s in pool if s not in (self._hard_queue or [])]
                self._init_ucb(pool)   # re-init from full pool (hard_first already popped its items)
            available = [c for c, s in self._ucb_pools.items() if s]
            if not available:
                return None
            best = max(available, key=self._ucb_score)
            return self._ucb_pools[best].pop(0)

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        self._cat_correct[cat] += int(is_correct)
        self._cat_total[cat]   += 1
        self._total            += 1

    def category_stats(self):
        return {
            cat: {
                "correct": self._cat_correct[cat],
                "total":   self._cat_total[cat],
                "error_rate": round(1 - self._cat_correct[cat] / max(self._cat_total[cat], 1), 3),
            }
            for cat in self._cat_total
        }


# ──────────────────────────────────────────────────────────────
# 8b. Phased curriculum: Easy → Hard
# ──────────────────────────────────────────────────────────────

class PhasedEasyHardSelector(BaseSelector):  # formerly class 8, now 8b
    """
    Two-phase curriculum:
      Phase 1 (steps 1..phase_switch): Easy examples first
        → build clean, well-structured playbook quickly.
      Phase 2 (steps phase_switch+1..budget): Hard examples
        → test & refine playbook on confusing cases.

    Hypothesis: A clean foundation + targeted hard refinement > pure hard or pure easy.
    """

    def __init__(self, difficulty: Dict[str, float], budget: int = 200,
                 phase_split: float = 0.4, seed: int = 42):
        """
        phase_split: fraction of budget spent on easy examples (default 40%).
        """
        self._difficulty  = difficulty
        self._phase_split = phase_split
        self._budget      = budget
        self._rng         = random.Random(seed)
        self._step        = 0
        self._easy_queue: Optional[List[Dict]] = None
        self._hard_queue: Optional[List[Dict]] = None

    @property
    def name(self): return f"phased_easy_hard_{int(self._phase_split*100)}pct"

    def _init(self, pool):
        def score(s):
            return self._difficulty.get(s.get("_id", ""), 0.5) + self._rng.random() * 1e-6
        sorted_pool = sorted(pool, key=score)          # ascending = easiest first
        n_easy      = int(len(sorted_pool) * self._phase_split)
        self._easy_queue = sorted_pool[:n_easy]         # easiest half
        self._hard_queue = sorted_pool[n_easy:][::-1]   # hardest half, hard first

    def select(self, pool):
        if self._easy_queue is None:
            self._init(pool)
        self._step += 1
        phase_switch = int(self._budget * self._phase_split)
        if self._step <= phase_switch:
            if self._easy_queue:
                return self._easy_queue.pop(0)
            if self._hard_queue:
                return self._hard_queue.pop(0)
        else:
            if self._hard_queue:
                return self._hard_queue.pop(0)
            if self._easy_queue:
                return self._easy_queue.pop(0)
        return None


# ──────────────────────────────────────────────────────────────
# 9. Diversity-aware selector (playbook bullet coverage)
# ──────────────────────────────────────────────────────────────

class DiversityAwareSelector(BaseSelector):
    """
    Prefer categories that are UNDER-REPRESENTED in the current playbook.

    At each step:
      score(cat) = (1 / (n_cat_bullets + 1)) + ε·error_rate(cat)

    where n_cat_bullets ≈ proxy from playbook keyword counting.
    No access to ACE internals needed; we just count category keywords in playbook text.
    """

    def __init__(self, seed: int = 42, eps: float = 0.3):
        self._rng  = random.Random(seed)
        self._eps  = eps
        self._cat_correct: Dict[str, int] = defaultdict(int)
        self._cat_total:   Dict[str, int] = defaultdict(int)
        self._cat_pools:   Optional[Dict[str, List[Dict]]] = None
        # Track approximate playbook coverage by counting category appearances
        self._cat_bullets: Dict[str, int] = defaultdict(int)

    @property
    def name(self): return f"diversity_aware"

    def _init(self, pool):
        self._cat_pools = defaultdict(list)
        for s in pool:
            self._cat_pools[s["macro_category"]].append(s)
        for cat in self._cat_pools:
            self._rng.shuffle(self._cat_pools[cat])

    def _score(self, cat: str) -> float:
        diversity = 1.0 / (self._cat_bullets[cat] + 1)
        n = self._cat_total[cat]
        err = (1.0 - self._cat_correct[cat] / n) if n > 0 else 0.5
        return diversity + self._eps * err + self._rng.random() * 1e-5

    def select(self, pool):
        if self._cat_pools is None:
            self._init(pool)
        available = [c for c, s in self._cat_pools.items() if s]
        if not available:
            return None
        best = max(available, key=self._score)
        return self._cat_pools[best].pop(0)

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        self._cat_correct[cat] += int(is_correct)
        self._cat_total[cat]   += 1
        # Count each training step as +1 bullet for that category (proxy for coverage)
        self._cat_bullets[cat] += 1

    def category_stats(self):
        return {
            cat: {
                "correct": self._cat_correct[cat],
                "total":   self._cat_total[cat],
                "bullets": self._cat_bullets[cat],
                "score":   round(self._score(cat), 4),
            }
            for cat in set(list(self._cat_total) + list(self._cat_bullets))
        }


# ──────────────────────────────────────────────────────────────
# 10. HardCategoryFirst (pre-test based, category level)
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# 11. Thompson Sampling over macro categories
# ──────────────────────────────────────────────────────────────

class ThompsonSamplingSelector(BaseSelector):
    """
    Beta-Bernoulli Thompson Sampling over macro categories.

    Each category has a Beta(alpha, beta) posterior over its "error rate":
      alpha = 1 + n_errors
      beta  = 1 + n_correct
    At each step, sample a θ ~ Beta(alpha, beta) for each category,
    then pick the category with the highest sampled θ (= expected error rate).

    More principled than UCB: avoids hard-coded c parameter,
    naturally decays exploration as we gather more data.
    """

    def __init__(self, seed: int = 42):
        self._rng           = random.Random(seed)
        self._cat_correct:  Dict[str, int] = defaultdict(int)
        self._cat_errors:   Dict[str, int] = defaultdict(int)
        self._cat_pools:    Optional[Dict[str, List[Dict]]] = None

    @property
    def name(self): return "thompson_sampling"

    def _init(self, pool):
        self._cat_pools = defaultdict(list)
        for s in pool:
            self._cat_pools[s["macro_category"]].append(s)
        for cat in self._cat_pools:
            self._rng.shuffle(self._cat_pools[cat])

    def _sample_error_rate(self, cat: str) -> float:
        import random as stdlib_random
        alpha = 1 + self._cat_errors[cat]    # prior: 1 error seen
        beta  = 1 + self._cat_correct[cat]   # prior: 1 correct seen
        # Beta distribution via gamma sampling
        g1 = stdlib_random.gammavariate(alpha, 1)
        g2 = stdlib_random.gammavariate(beta, 1)
        return g1 / (g1 + g2) if (g1 + g2) > 0 else 0.5

    def select(self, pool):
        if self._cat_pools is None:
            self._init(pool)
        available = [c for c, s in self._cat_pools.items() if s]
        if not available:
            return None
        best = max(available, key=self._sample_error_rate)
        return self._cat_pools[best].pop(0)

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        if is_correct:
            self._cat_correct[cat] += 1
        else:
            self._cat_errors[cat] += 1

    def category_stats(self):
        return {
            cat: {
                "errors":  self._cat_errors[cat],
                "correct": self._cat_correct[cat],
                "error_rate": round(
                    self._cat_errors[cat] /
                    max(self._cat_errors[cat] + self._cat_correct[cat], 1), 3
                ),
            }
            for cat in set(list(self._cat_errors) + list(self._cat_correct))
        }


class PhasedThompsonSelector(BaseSelector):
    """
    Two-phase hybrid:
      Phase 1 (easy_split fraction of budget): train on easy examples
              (lowest difficulty score first) to warm up the playbook.
      Phase 2 (remaining budget): Thompson Sampling over macro categories
              to efficiently exploit & explore on hard examples.

    Combines the benefits of easy-first warmup (clean signal early) with
    the principled exploration of Thompson Sampling (focus on hard categories).
    """

    def __init__(self,
                 difficulty: Dict,
                 budget: int,
                 phase_split: float = 0.4,
                 seed: int = 42):
        self._difficulty  = difficulty
        self._budget      = budget
        self._phase_split = phase_split
        self._rng         = random.Random(seed)
        self._step        = 0
        self._phase       = 1
        self._easy_queue: Optional[List[Dict]] = None
        self._cat_pools:  Optional[Dict[str, List[Dict]]] = None
        self._cat_correct: Dict[str, int] = defaultdict(int)
        self._cat_errors:  Dict[str, int] = defaultdict(int)
        self._cat_total:   Dict[str, int] = defaultdict(int)

    @property
    def name(self): return f"phased_thompson_{int(self._phase_split*100)}pct"

    def _init_pools(self, pool):
        phase_switch = int(self._budget * self._phase_split)
        diff_sorted = sorted(pool, key=lambda s: self._difficulty.get(s.get("_id", -1), 0.5))
        easy_pool = diff_sorted[:phase_switch]
        hard_pool = diff_sorted[phase_switch:]
        self._rng.shuffle(easy_pool)
        # Build per-category pools for Thompson phase from hard examples
        self._cat_pools = defaultdict(list)
        for s in hard_pool:
            self._cat_pools[s.get("macro_category", "Other")].append(s)
        for cat in self._cat_pools:
            self._rng.shuffle(self._cat_pools[cat])
        self._easy_queue = easy_pool
        print(f"[PhasedThompson] Phase 1: {len(easy_pool)} easy, "
              f"Phase 2: {len(hard_pool)} hard across {len(self._cat_pools)} cats")

    def _sample_error_rate(self, cat: str) -> float:
        import random as stdlib_random
        alpha = 1 + self._cat_errors[cat]
        beta  = 1 + self._cat_correct[cat]
        g1 = stdlib_random.gammavariate(alpha, 1)
        g2 = stdlib_random.gammavariate(beta, 1)
        return g1 / (g1 + g2) if (g1 + g2) > 0 else 0.5

    def select(self, pool):
        if self._easy_queue is None:
            self._init_pools(pool)
        self._step += 1
        phase_switch = int(self._budget * self._phase_split)

        # Phase 1: easy examples
        if self._step <= phase_switch and self._easy_queue:
            return self._easy_queue.pop(0)

        # Phase 2: Thompson Sampling on hard pool
        self._phase = 2
        available = [c for c, s in self._cat_pools.items() if s]
        if not available:
            # Fall back to remaining easy examples if hard exhausted
            return self._easy_queue.pop(0) if self._easy_queue else None
        best = max(available, key=self._sample_error_rate)
        return self._cat_pools[best].pop(0)

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        self._cat_total[cat] += 1
        if is_correct:
            self._cat_correct[cat] += 1
        else:
            self._cat_errors[cat] += 1

    def category_stats(self):
        return {
            cat: {
                "correct": self._cat_correct[cat],
                "errors":  self._cat_errors[cat],
                "total":   self._cat_total[cat],
                "phase":   self._phase,
            }
            for cat in self._cat_total
        }


class HardCategoryFirstSelector(BaseSelector):
    """
    Orders categories by pre-test accuracy (hardest category first).
    Within a category, samples are randomised.
    Does NOT switch mid-category — exhausts all samples in one category
    before moving to next (no mastery threshold, fixed ordering).
    """

    def __init__(self, cat_accuracy: Dict[str, float], seed: int = 42):
        """
        cat_accuracy: maps category name → accuracy in [0,1] (from pre-test).
        Hardest = lowest accuracy → trains on this category first.
        """
        self._cat_acc = cat_accuracy
        self._rng = random.Random(seed)
        self._queue: Optional[List[Dict]] = None

    @property
    def name(self): return "hard_cat_first"

    def _init(self, pool):
        # Group by category
        cat_pools: Dict[str, List[Dict]] = defaultdict(list)
        for s in pool:
            cat_pools[s["macro_category"]].append(s)
        for cat in cat_pools:
            self._rng.shuffle(cat_pools[cat])
        # Sort categories by pre-test accuracy ascending (hardest first)
        sorted_cats = sorted(cat_pools.keys(),
                             key=lambda c: self._cat_acc.get(c, 0.5))
        self._queue = []
        for cat in sorted_cats:
            self._queue.extend(cat_pools[cat])

    def select(self, pool):
        if self._queue is None:
            self._init(pool)
        return self._queue.pop(0) if self._queue else None


# ──────────────────────────────────────────────────────────────
# GeneralPhasedSelector  — Generalizable 3-phase curriculum
# ──────────────────────────────────────────────────────────────

class GeneralPhasedSelector(BaseSelector):
    """
    Generalizable 3-phase curriculum learning selector.

    Philosophy
    ----------
    For ANY classification/prediction dataset:
      - Pre-test ~15-20% of training data (cheap, no labels needed)
      - "Easy"   = pretest correct  → model already knows how to do these
      - "Medium" = not pre-tested   → represents the true data distribution
      - "Hard"   = pretest wrong    → model consistently fails on these

    Training order: Easy → Medium → Hard
      Phase 1: Easy warmup.   Build playbook from clean signal.
      Phase 2: Medium bulk.   Reinforce on diverse, representative data.
      Phase 3: Hard targeted. Apply mature playbook to fix failure modes.

    Adaptive phase switching (optional)
    ------------------------------------
    If `switch_patience` > 0, the selector auto-switches to the next phase
    when val accuracy hasn't improved for `switch_patience` consecutive
    evaluations, even if the current phase is not exhausted.
    This prevents over-training on one difficulty tier.

    Parameters
    ----------
    difficulty:       dict  _id → 0.0 (easy) | 0.5 (unknown) | 1.0 (hard)
    budget:           total training steps
    easy_pct:         fraction of budget for Phase 1  (default 0.20)
    medium_pct:       fraction of budget for Phase 2  (default 0.60)
    hard_pct:         fraction of budget for Phase 3  (default 0.20)
    switch_patience:  val evals without improvement before phase switch
                      (0 = use fixed phase budgets, no adaptive switching)
    within_phase_strategy: 'random' | 'diversity'
                      How to select within each phase.
    seed:             random seed
    """

    def __init__(
        self,
        difficulty: Dict,
        budget: int,
        easy_pct: float = 0.20,
        medium_pct: float = 0.60,
        hard_pct: float = 0.20,
        switch_patience: int = 0,
        within_phase_strategy: str = "diversity",
        seed: int = 42,
    ):
        assert abs(easy_pct + medium_pct + hard_pct - 1.0) < 1e-6, \
            "Phase fractions must sum to 1.0"
        self._difficulty = difficulty
        self._budget = budget
        self._easy_pct = easy_pct
        self._medium_pct = medium_pct
        self._hard_pct = hard_pct
        self._switch_patience = switch_patience
        self._within_strategy = within_phase_strategy
        self._rng = random.Random(seed)

        # Phase budgets (steps)
        self._phase_budget = [
            int(budget * easy_pct),
            int(budget * medium_pct),
            budget - int(budget * easy_pct) - int(budget * medium_pct),
        ]

        # State
        self._phase = 0              # current phase index (0=easy, 1=medium, 2=hard)
        self._phase_step = 0         # steps used in current phase
        self._phase_patience = 0     # consecutive non-improving val evals
        self._best_val_in_phase = 0.0

        # Per-phase pools
        self._pools: Optional[List[List[Dict]]] = None

        # Per-category tracking (for diversity-aware within-phase selection)
        self._cat_seen: Dict[str, int] = defaultdict(int)
        self._cat_correct: Dict[str, int] = defaultdict(int)
        self._cat_total: Dict[str, int] = defaultdict(int)
        self._step = 0

    @property
    def name(self):
        return (f"general_phased_"
                f"e{int(self._easy_pct*100)}"
                f"m{int(self._medium_pct*100)}"
                f"h{int(self._hard_pct*100)}"
                f"_sw{self._switch_patience}"
                f"_{self._within_strategy}")

    def _init_pools(self, pool: List[Dict]):
        easy, medium, hard = [], [], []
        for s in pool:
            d = self._difficulty.get(s.get("_id", -1), 0.5)
            if d <= 0.0:
                easy.append(s)
            elif d >= 1.0:
                hard.append(s)
            else:
                medium.append(s)
        # Shuffle each pool
        for p in [easy, medium, hard]:
            self._rng.shuffle(p)
        self._pools = [easy, medium, hard]
        phase_names = ["easy", "medium", "hard"]
        for i, (p, n) in enumerate(zip(self._pools, phase_names)):
            print(f"  [GeneralPhased] Phase {i+1} ({n}): {len(p)} samples, "
                  f"budget={self._phase_budget[i]} steps")

    def _pick_diverse(self, pool: List[Dict]) -> Optional[Dict]:
        """Select the sample from pool whose category has been seen least."""
        if not pool:
            return None
        # Score each sample: prefer under-represented categories
        best_idx = min(
            range(len(pool)),
            key=lambda i: self._cat_seen.get(
                pool[i].get("macro_category", "Other"), 0
            ) + self._rng.random() * 0.1
        )
        return pool.pop(best_idx)

    def _pick_from_phase(self, phase_idx: int) -> Optional[Dict]:
        pool = self._pools[phase_idx]
        if not pool:
            return None
        if self._within_strategy == "diversity":
            return self._pick_diverse(pool)
        else:
            return pool.pop(0)

    def notify_val(self, val_acc: float):
        """
        Call after each val evaluation so the selector can adaptively
        switch phases. Called by CurriculumRunner if switch_patience > 0.
        """
        if self._switch_patience <= 0:
            return
        if val_acc > self._best_val_in_phase + 1e-4:
            self._best_val_in_phase = val_acc
            self._phase_patience = 0
        else:
            self._phase_patience += 1
            if self._phase_patience >= self._switch_patience and self._phase < 2:
                print(f"  [GeneralPhased] Phase {self._phase+1} plateau "
                      f"(patience={self._phase_patience}). "
                      f"Switching to phase {self._phase+2}.")
                self._phase += 1
                self._phase_step = 0
                self._phase_patience = 0
                self._best_val_in_phase = val_acc

    def select(self, pool: List[Dict]) -> Optional[Dict]:
        if self._pools is None:
            self._init_pools(pool)
        self._step += 1

        # Fixed-budget phase switching
        if self._switch_patience <= 0:
            while self._phase < 2:
                if self._phase_step < self._phase_budget[self._phase]:
                    break
                # Current phase budget exhausted → advance
                if self._pools[self._phase]:
                    pass  # still has samples but budget done
                self._phase += 1
                self._phase_step = 0
                self._best_val_in_phase = 0.0
                phase_names = ["easy", "medium", "hard"]
                if self._phase <= 2:
                    print(f"  [GeneralPhased] Entering Phase {self._phase+1} "
                          f"({phase_names[self._phase]}) at step {self._step}")

        # Try to pick from current phase; fall back to next if exhausted
        for offset in range(3):
            ph = self._phase + offset
            if ph > 2:
                break
            ex = self._pick_from_phase(ph)
            if ex is not None:
                self._phase_step += 1
                cat = ex.get("macro_category", "Other")
                self._cat_seen[cat] = self._cat_seen.get(cat, 0) + 1
                return ex

        return None  # all pools exhausted

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        self._cat_total[cat] += 1
        self._cat_correct[cat] += int(is_correct)

    def current_phase(self) -> int:
        return self._phase

    def category_stats(self) -> Dict:
        return {
            cat: {
                "seen":    self._cat_seen.get(cat, 0),
                "correct": self._cat_correct[cat],
                "total":   self._cat_total[cat],
            }
            for cat in set(list(self._cat_seen) + list(self._cat_total))
        }


# ──────────────────────────────────────────────────────────────
# GeneralPhasedV2  — 3-phase with per-phase Thompson Sampling
# ──────────────────────────────────────────────────────────────

class GeneralPhasedV2(BaseSelector):
    """
    Improved GeneralPhased: each phase uses Thompson Sampling over categories.

    Phase 1 (easy):   diversity-first to cover all categories cleanly.
    Phase 2 (medium): Thompson Sampling — focus on categories where model
                      still struggles after Phase 1 warmup.
    Phase 3 (hard):   Thompson Sampling — targeted hard-case coverage.

    The key insight: Phase 1 should be DIVERSE (see every category once),
    while Phase 2/3 should be ADAPTIVE (focus on weak spots revealed by
    Phase 1 training).

    Additional feature: 'smooth' difficulty — samples pretest-scored
    multiple times get a continuous difficulty in [0,1] rather than
    binary 0/1. If only one pretest pass is available, reverts to
    binary bucketing.
    """

    PHASE_NAMES = ["easy", "medium", "hard"]

    def __init__(
        self,
        difficulty: Dict,
        budget: int,
        easy_pct:   float = 0.20,
        medium_pct: float = 0.60,
        hard_pct:   float = 0.20,
        switch_patience: int = 2,
        seed: int = 42,
    ):
        assert abs(easy_pct + medium_pct + hard_pct - 1.0) < 1e-6
        self._diff = difficulty
        self._budget = budget
        self._phase_budget = [
            int(budget * easy_pct),
            int(budget * medium_pct),
            budget - int(budget * easy_pct) - int(budget * medium_pct),
        ]
        self._switch_patience = switch_patience
        self._rng = random.Random(seed)

        # Shared Thompson state (persists across phases)
        self._cat_correct:  Dict[str, int] = defaultdict(int)
        self._cat_errors:   Dict[str, int] = defaultdict(int)
        self._cat_seen:     Dict[str, int] = defaultdict(int)

        # Phase state
        self._phase = 0
        self._phase_step = 0
        self._phase_patience = 0
        self._best_val = 0.0

        # Per-phase category pools: {phase: {cat: [samples]}}
        self._cat_pools: Optional[List[Dict[str, List[Dict]]]] = None

    @property
    def name(self):
        pb = self._phase_budget
        return (f"general_phased_v2_"
                f"e{pb[0]}m{pb[1]}h{pb[2]}"
                f"_sw{self._switch_patience}")

    def _init_pools(self, pool: List[Dict]):
        easy, medium, hard = [], [], []
        for s in pool:
            d = self._diff.get(s.get("_id", -1), 0.5)
            if d <= 0.0:
                easy.append(s)
            elif d >= 1.0:
                hard.append(s)
            else:
                medium.append(s)

        def build_cat_pool(samples):
            cp = defaultdict(list)
            for s in samples:
                cp[s.get("macro_category", "Other")].append(s)
            for cat in cp:
                self._rng.shuffle(cp[cat])
            return cp

        self._cat_pools = [
            build_cat_pool(easy),
            build_cat_pool(medium),
            build_cat_pool(hard),
        ]
        for i, (cp, n) in enumerate(zip(self._cat_pools, self.PHASE_NAMES)):
            total = sum(len(v) for v in cp.values())
            print(f"  [GPv2] Phase {i+1} ({n}): {total} samples in {len(cp)} cats, "
                  f"budget={self._phase_budget[i]}")

    def _sample_thompson(self, cat: str) -> float:
        import random as _r
        a = 1 + self._cat_errors[cat]
        b = 1 + self._cat_correct[cat]
        g1 = _r.gammavariate(a, 1)
        g2 = _r.gammavariate(b, 1)
        return g1 / (g1 + g2) if (g1 + g2) > 0 else 0.5

    def _pick_phase1_diverse(self, cp: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Phase 1: prefer under-seen categories (build balanced easy coverage)."""
        available = [c for c, s in cp.items() if s]
        if not available:
            return None
        best = min(available,
                   key=lambda c: self._cat_seen.get(c, 0) + self._rng.random() * 0.1)
        return cp[best].pop(0)

    def _pick_thompson(self, cp: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Phase 2/3: Thompson Sampling over categories."""
        available = [c for c, s in cp.items() if s]
        if not available:
            return None
        best = max(available, key=self._sample_thompson)
        return cp[best].pop(0)

    def _advance_phase(self):
        if self._phase < 2:
            self._phase += 1
            self._phase_step = 0
            self._phase_patience = 0
            self._best_val = 0.0
            print(f"  [GPv2] → Phase {self._phase+1} "
                  f"({self.PHASE_NAMES[self._phase]})")

    def notify_val(self, val_acc: float):
        if self._switch_patience <= 0:
            return
        if val_acc > self._best_val + 1e-4:
            self._best_val = val_acc
            self._phase_patience = 0
        else:
            self._phase_patience += 1
            if self._phase_patience >= self._switch_patience:
                self._advance_phase()

    def select(self, pool: List[Dict]) -> Optional[Dict]:
        if self._cat_pools is None:
            self._init_pools(pool)

        # Fixed-budget switching
        if self._switch_patience <= 0:
            while (self._phase < 2 and
                   self._phase_step >= self._phase_budget[self._phase]):
                self._advance_phase()

        cp = self._cat_pools[self._phase]

        # Pick strategy depends on phase
        if self._phase == 0:
            ex = self._pick_phase1_diverse(cp)
        else:
            ex = self._pick_thompson(cp)

        # Fall back to next non-empty phase if current is exhausted
        if ex is None:
            for offset in range(1, 3):
                ph = self._phase + offset
                if ph > 2:
                    break
                cp2 = self._cat_pools[ph]
                ex = self._pick_thompson(cp2)
                if ex is not None:
                    break
            if ex is None:
                return None

        self._phase_step += 1
        cat = ex.get("macro_category", "Other")
        self._cat_seen[cat] = self._cat_seen.get(cat, 0) + 1
        return ex

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        if is_correct:
            self._cat_correct[cat] += 1
        else:
            self._cat_errors[cat] += 1

    def current_phase(self) -> int:
        return self._phase

    def category_stats(self) -> Dict:
        cats = set(list(self._cat_correct) + list(self._cat_errors))
        return {
            cat: {
                "correct": self._cat_correct[cat],
                "errors":  self._cat_errors[cat],
                "seen":    self._cat_seen.get(cat, 0),
            }
            for cat in cats
        }


# ──────────────────────────────────────────────────────────────
# PhaseOrderedSelector  — configurable phase ordering
# ──────────────────────────────────────────────────────────────

class PhaseOrderedSelector(BaseSelector):
    """
    Generalizable curriculum with configurable phase ordering.

    Supports any ordering of easy/medium/hard phases, e.g.:
      - EMH: Easy → Medium → Hard  (conservative: build then reinforce then fix)
      - EHM: Easy → Hard → Medium  (aggressive: build then fix then generalize)
      - MEH: Medium → Easy → Hard  (typical random-then-easy-then-hard)
      - HME: Hard → Medium → Easy  (anti-curriculum — for ablation)

    Within each phase:
      - 'easy'   phase: diversity-first (cover all categories)
      - 'medium' phase: Thompson Sampling (adaptive, focus on weak cats)
      - 'hard'   phase: Thompson Sampling (targeted hard coverage)

    Parameters
    ----------
    phase_order:  str, e.g. "EMH" or "EHM" or "MEH"
    phase_fracs:  dict, optional fractions per phase letter.
                  e.g. {"E": 0.25, "H": 0.25, "M": 0.50}
                  If None, splits budget evenly.
    switch_patience: int, val evals to wait before adaptive phase switch (0=fixed)
    """

    _PHASE_MAP = {"E": 0.0, "M": 0.5, "H": 1.0}

    def __init__(
        self,
        difficulty: Dict,
        budget: int,
        phase_order: str = "EMH",
        phase_fracs: Optional[Dict[str, float]] = None,
        switch_patience: int = 0,
        seed: int = 42,
    ):
        assert len(phase_order) in (2, 3), "phase_order must be 2 or 3 chars"
        self._diff = difficulty
        self._budget = budget
        self._order = phase_order.upper()
        self._switch_patience = switch_patience
        self._rng = random.Random(seed)

        # Phase fractions
        if phase_fracs is None:
            n = len(phase_order)
            frac = 1.0 / n
            phase_fracs = {c: frac for c in phase_order.upper()}

        # Compute budgets in sequence order
        cumsum = 0
        budgets = []
        for i, c in enumerate(self._order):
            if i < len(self._order) - 1:
                b = int(budget * phase_fracs[c])
            else:
                b = budget - cumsum
            budgets.append(b)
            cumsum += b
        self._phase_budgets = budgets

        # Thompson state (shared across phases)
        self._cat_correct:  Dict[str, int] = defaultdict(int)
        self._cat_errors:   Dict[str, int] = defaultdict(int)
        self._cat_seen:     Dict[str, int] = defaultdict(int)

        # Phase state
        self._phase_idx = 0           # index into self._order
        self._phase_step = 0
        self._phase_patience = 0
        self._best_val = 0.0

        # Pools: {threshold: {cat: [samples]}}
        # threshold 0.0=easy, 0.5=medium, 1.0=hard
        self._tier_pools: Optional[Dict[float, Dict[str, List[Dict]]]] = None

    @property
    def name(self) -> str:
        frac_str = "_".join(
            f"{c}{int(self._phase_budgets[i]/self._budget*100)}"
            for i, c in enumerate(self._order)
        )
        return f"phase_order_{self._order}_{frac_str}_sw{self._switch_patience}"

    def _init_pools(self, pool: List[Dict]):
        easy, medium, hard = [], [], []
        for s in pool:
            d = self._diff.get(s.get("_id", -1), 0.5)
            if d <= 0.0:
                easy.append(s)
            elif d >= 1.0:
                hard.append(s)
            else:
                medium.append(s)

        def build_cat_pool(samples):
            cp = defaultdict(list)
            for s in samples:
                cp[s.get("macro_category", "Other")].append(s)
            for cat in cp:
                self._rng.shuffle(cp[cat])
            return cp

        self._tier_pools = {
            0.0: build_cat_pool(easy),
            0.5: build_cat_pool(medium),
            1.0: build_cat_pool(hard),
        }
        label = {"E": "easy", "M": "medium", "H": "hard"}
        for i, c in enumerate(self._order):
            threshold = self._PHASE_MAP[c]
            total = sum(len(v) for v in self._tier_pools[threshold].values())
            print(f"  [PhaseOrder-{self._order}] Phase {i+1} ({label[c]}): "
                  f"{total} samples, budget={self._phase_budgets[i]}")

    def _sample_thompson(self, cat: str) -> float:
        a = 1 + self._cat_errors[cat]
        b = 1 + self._cat_correct[cat]
        g1 = random.gammavariate(a, 1)
        g2 = random.gammavariate(b, 1)
        return g1 / (g1 + g2) if (g1 + g2) > 0 else 0.5

    def _pick_diverse(self, cp: Dict[str, List[Dict]]) -> Optional[Dict]:
        available = [c for c, s in cp.items() if s]
        if not available:
            return None
        best = min(available,
                   key=lambda c: self._cat_seen.get(c, 0) + self._rng.random() * 0.1)
        return cp[best].pop(0)

    def _pick_thompson(self, cp: Dict[str, List[Dict]]) -> Optional[Dict]:
        available = [c for c, s in cp.items() if s]
        if not available:
            return None
        best = max(available, key=self._sample_thompson)
        return cp[best].pop(0)

    def _pick_from_tier(self, tier: float) -> Optional[Dict]:
        cp = self._tier_pools[tier]
        phase_char = self._order[self._phase_idx]
        # Easy phase: diversity; Medium/Hard: Thompson
        if phase_char == "E":
            return self._pick_diverse(cp)
        else:
            return self._pick_thompson(cp)

    def _advance_phase(self):
        if self._phase_idx < len(self._order) - 1:
            self._phase_idx += 1
            self._phase_step = 0
            self._phase_patience = 0
            self._best_val = 0.0
            c = self._order[self._phase_idx]
            label = {"E": "easy", "M": "medium", "H": "hard"}
            print(f"  [PhaseOrder-{self._order}] → Phase {self._phase_idx+1} "
                  f"({label[c]})")

    def notify_val(self, val_acc: float):
        if self._switch_patience <= 0:
            return
        if val_acc > self._best_val + 1e-4:
            self._best_val = val_acc
            self._phase_patience = 0
        else:
            self._phase_patience += 1
            if self._phase_patience >= self._switch_patience:
                self._advance_phase()

    def select(self, pool: List[Dict]) -> Optional[Dict]:
        if self._tier_pools is None:
            self._init_pools(pool)

        # Fixed-budget switching
        if self._switch_patience <= 0:
            while (self._phase_idx < len(self._order) - 1 and
                   self._phase_step >= self._phase_budgets[self._phase_idx]):
                self._advance_phase()

        tier = self._PHASE_MAP[self._order[self._phase_idx]]
        ex = self._pick_from_tier(tier)

        # Fall back to other tiers if current is exhausted
        if ex is None:
            for offset in range(1, len(self._order)):
                ph = self._phase_idx + offset
                if ph >= len(self._order):
                    break
                other_tier = self._PHASE_MAP[self._order[ph]]
                ex = self._pick_thompson(self._tier_pools[other_tier])
                if ex is not None:
                    break
        if ex is None:
            return None

        self._phase_step += 1
        cat = ex.get("macro_category", "Other")
        self._cat_seen[cat] = self._cat_seen.get(cat, 0) + 1
        return ex

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        if is_correct:
            self._cat_correct[cat] += 1
        else:
            self._cat_errors[cat] += 1

    def current_phase(self) -> int:
        return self._phase_idx

    def category_stats(self) -> Dict:
        cats = set(list(self._cat_correct) + list(self._cat_errors))
        return {
            cat: {
                "correct": self._cat_correct[cat],
                "errors":  self._cat_errors[cat],
                "seen":    self._cat_seen.get(cat, 0),
            }
            for cat in cats
        }


# ──────────────────────────────────────────────────────────────
# BayesianPhasedSelector  — pretest-informed Bayesian Thompson
# ──────────────────────────────────────────────────────────────

class BayesianPhasedSelector(BaseSelector):
    """
    Bayesian Phased Curriculum: uses pretest results as Beta-distribution priors.

    Core idea
    ---------
    Instead of hard phase boundaries, we assign each sample a difficulty-informed
    prior and use Thompson Sampling throughout.

    Prior calibration:
      - Easy (pretest correct):   Beta(1, 2)  → ~33% error rate prior
      - Unknown (not pretested):  Beta(1, 1)  → uniform 50% prior
      - Hard (pretest wrong):     Beta(2, 1)  → ~67% error rate prior

    Phase 1 (warmup_pct % of budget):
      Diversity-first over EASY examples only. Purpose: build clean playbook.

    Phase 2 (remaining budget):
      Thompson Sampling over ALL categories (easy + medium + hard),
      but hard examples have informative priors that drive initial exploration.
      As training proceeds, the posteriors are updated from observed outcomes,
      making the selection fully adaptive.

    This is more principled than fixed-boundary phasing because:
    1. No sharp transitions that can cause playbook instability
    2. The prior drives initial focus on hard examples naturally
    3. If "easy" categories later prove difficult (playbook changed them), the
       model adapts automatically

    Parameters
    ----------
    difficulty:     dict _id → float (0=easy, 0.5=unknown, 1=hard)
    budget:         total training steps
    warmup_pct:     fraction for diversity-easy Phase 1 (default 0.25)
    prior_strength: how strongly to weight pretest priors (default 2.0)
                    higher = more weight on pretest, lower = faster adaptation
    seed:           random seed
    """

    def __init__(
        self,
        difficulty: Dict,
        budget: int,
        warmup_pct: float = 0.25,
        prior_strength: float = 2.0,
        seed: int = 42,
    ):
        self._diff = difficulty
        self._budget = budget
        self._warmup_steps = int(budget * warmup_pct)
        self._prior_strength = prior_strength
        self._rng = random.Random(seed)

        # Category-level Thompson state
        # alpha = 1 + prior_errors + observed_errors
        # beta  = 1 + prior_correct + observed_correct
        self._cat_alpha: Dict[str, float] = defaultdict(lambda: 1.0)
        self._cat_beta:  Dict[str, float] = defaultdict(lambda: 1.0)
        self._cat_seen:  Dict[str, int]   = defaultdict(int)

        # Phase state
        self._step = 0
        self._phase = 0  # 0=warmup, 1=thompson

        # Pools: easy_cats (for Phase 1), all_cats (for Phase 2)
        self._easy_cat_pool: Optional[Dict[str, List[Dict]]] = None
        self._all_cat_pool:  Optional[Dict[str, List[Dict]]] = None
        self._warmup_pct = warmup_pct

    @property
    def name(self) -> str:
        return (f"bayesian_phased_w{int(self._warmup_pct*100)}"
                f"_pr{self._prior_strength}")

    def _sample_error_rate(self, cat: str) -> float:
        a = self._cat_alpha[cat]
        b = self._cat_beta[cat]
        g1 = random.gammavariate(a, 1)
        g2 = random.gammavariate(b, 1)
        return g1 / (g1 + g2) if (g1 + g2) > 0 else 0.5

    def _init_pools(self, pool: List[Dict]):
        easy_cats: Dict[str, List[Dict]] = defaultdict(list)
        all_cats:  Dict[str, List[Dict]] = defaultdict(list)

        for s in pool:
            d = self._diff.get(s.get("_id", -1), 0.5)
            cat = s.get("macro_category", "Other")
            all_cats[cat].append(s)
            if d <= 0.0:
                easy_cats[cat].append(s)

        # Apply difficulty priors to category Thompson parameters
        # Aggregate difficulty per category
        cat_difficulty: Dict[str, List[float]] = defaultdict(list)
        for s in pool:
            cat = s.get("macro_category", "Other")
            d = self._diff.get(s.get("_id", -1), 0.5)
            cat_difficulty[cat].append(d)

        for cat, diffs in cat_difficulty.items():
            avg_d = sum(diffs) / len(diffs)  # 0=easy, 0.5=unknown, 1=hard
            n = len(diffs)
            ps = self._prior_strength
            # Encode pretest difficulty as pseudo-observations
            prior_errors   = avg_d * ps
            prior_correct  = (1.0 - avg_d) * ps
            self._cat_alpha[cat] = 1.0 + prior_errors
            self._cat_beta[cat]  = 1.0 + prior_correct

        # Shuffle pools
        for cp in [easy_cats, all_cats]:
            for cat in cp:
                self._rng.shuffle(cp[cat])

        self._easy_cat_pool = dict(easy_cats)
        self._all_cat_pool  = dict(all_cats)

        easy_total = sum(len(v) for v in easy_cats.values())
        all_total  = sum(len(v) for v in all_cats.values())
        print(f"  [BayesianPhased] warmup={self._warmup_steps} steps "
              f"(easy_pool={easy_total}), then Thompson over {all_total} samples")
        print(f"  [BayesianPhased] Category priors (top-5 hardest):")
        cat_prior_err = {c: self._cat_alpha[c] / (self._cat_alpha[c] + self._cat_beta[c])
                         for c in self._all_cat_pool}
        for c, e in sorted(cat_prior_err.items(), key=lambda x: -x[1])[:5]:
            print(f"    {c}: prior_error={e:.3f}")

    def _pick_diverse_easy(self) -> Optional[Dict]:
        available = [c for c, s in self._easy_cat_pool.items() if s]
        if not available:
            return None
        best = min(available,
                   key=lambda c: self._cat_seen.get(c, 0) + self._rng.random() * 0.1)
        return self._easy_cat_pool[best].pop(0)

    def _pick_thompson_all(self) -> Optional[Dict]:
        available = [c for c, s in self._all_cat_pool.items() if s]
        if not available:
            return None
        best = max(available, key=self._sample_error_rate)
        return self._all_cat_pool[best].pop(0)

    def select(self, pool: List[Dict]) -> Optional[Dict]:
        if self._all_cat_pool is None:
            self._init_pools(pool)

        self._step += 1

        if self._phase == 0 and self._step > self._warmup_steps:
            self._phase = 1
            print(f"  [BayesianPhased] Phase 1 done at step {self._step}. "
                  f"→ Full Thompson Sampling.")

        if self._phase == 0:
            ex = self._pick_diverse_easy()
            if ex is None:  # easy pool exhausted early → switch
                self._phase = 1
                print(f"  [BayesianPhased] Easy pool exhausted at step {self._step}.")
                ex = self._pick_thompson_all()
        else:
            ex = self._pick_thompson_all()

        if ex is not None:
            cat = ex.get("macro_category", "Other")
            self._cat_seen[cat] = self._cat_seen.get(cat, 0) + 1
            # Remove from easy pool too (avoid duplicates)
            if cat in self._easy_cat_pool and self._easy_cat_pool[cat]:
                ep = self._easy_cat_pool[cat]
                # Filter out this specific sample if present
                ep[:] = [s for s in ep if s.get("_id") != ex.get("_id")]

        return ex

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        if is_correct:
            self._cat_beta[cat]  += 1.0
        else:
            self._cat_alpha[cat] += 1.0

    def notify_val(self, val_acc: float):
        pass  # No adaptive switching needed — priors handle it

    def current_phase(self) -> int:
        return self._phase

    def category_stats(self) -> Dict:
        return {
            cat: {
                "seen":     self._cat_seen.get(cat, 0),
                "alpha":    self._cat_alpha[cat],
                "beta":     self._cat_beta[cat],
                "est_err":  self._cat_alpha[cat]
                            / (self._cat_alpha[cat] + self._cat_beta[cat]),
            }
            for cat in self._all_cat_pool or {}
        }


# ──────────────────────────────────────────────────────────────
# StratifiedPhasedSelector  — Easy diversity + Hard stratified
# ──────────────────────────────────────────────────────────────

class StratifiedPhasedSelector(BaseSelector):
    """
    Generalizable 2-phase curriculum:
      Phase 1 (easy_pct): Diversity-first over easy examples.
      Phase 2 (rest):     Stratified round-robin over hard examples.

    Key insight from experiments:
      - Thompson Sampling in hard phase causes playbook pollution by
        over-exploiting the hardest 1-2 categories.
      - Sequential/stratified hard selection maintains category coverage
        and produces more consistent playbook entries.

    This is the conservative, robust baseline for phased curriculum.

    Parameters
    ----------
    difficulty:   dict _id → float (0=easy, 1=hard, 0.5=unknown)
    budget:       total steps
    easy_pct:     fraction for easy warmup phase (default 0.50)
    include_medium: if True, include untested samples in hard pool (default True)
    seed:         random seed
    """

    def __init__(
        self,
        difficulty: Dict,
        budget: int,
        easy_pct: float = 0.50,
        include_medium: bool = True,
        seed: int = 42,
    ):
        self._diff = difficulty
        self._budget = budget
        self._easy_budget = int(budget * easy_pct)
        self._easy_pct = easy_pct
        self._include_medium = include_medium
        self._rng = random.Random(seed)

        # Per-category pools
        self._easy_cat:  Optional[Dict[str, List[Dict]]] = None
        self._hard_cat:  Optional[Dict[str, List[Dict]]] = None

        # Diversity tracking
        self._cat_seen:   Dict[str, int] = defaultdict(int)
        self._cat_correct: Dict[str, int] = defaultdict(int)
        self._cat_total:   Dict[str, int] = defaultdict(int)

        # Phase state
        self._phase = 0         # 0=easy, 1=hard
        self._phase_step = 0
        self._step = 0

        # Stratified state: category round-robin queue
        self._hard_cat_order: List[str] = []
        self._hard_cat_idx: int = 0

    @property
    def name(self) -> str:
        m = "m" if self._include_medium else ""
        return (f"stratified_phased_e{int(self._easy_pct*100)}"
                f"h{100-int(self._easy_pct*100)}{m}")

    def _init_pools(self, pool: List[Dict]):
        easy: List[Dict] = []
        hard: List[Dict] = []

        for s in pool:
            d = self._diff.get(s.get("_id", -1), 0.5)
            if d <= 0.0:
                easy.append(s)
            elif d >= 1.0:
                hard.append(s)
            elif self._include_medium:
                hard.append(s)   # treat untested as hard fallback

        self._rng.shuffle(easy)
        self._rng.shuffle(hard)

        # Build per-category pools
        def to_cat_pool(samples):
            cp = defaultdict(list)
            for s in samples:
                cp[s.get("macro_category", "Other")].append(s)
            return cp

        self._easy_cat = to_cat_pool(easy)
        self._hard_cat = to_cat_pool(hard)

        # Stratified order: sort cats by number of hard examples (desc),
        # interleave evenly for round-robin
        hard_cat_sizes = [(c, len(v)) for c, v in self._hard_cat.items()]
        hard_cat_sizes.sort(key=lambda x: -x[1])
        self._hard_cat_order = [c for c, _ in hard_cat_sizes]

        print(f"  [StratPhased] easy_budget={self._easy_budget}/{self._budget}, "
              f"easy_pool={sum(len(v) for v in self._easy_cat.values())}, "
              f"hard_pool={sum(len(v) for v in self._hard_cat.values())} "
              f"in {len(self._hard_cat)} cats")

    def _pick_diverse_easy(self) -> Optional[Dict]:
        available = [c for c, s in self._easy_cat.items() if s]
        if not available:
            return None
        best = min(available,
                   key=lambda c: self._cat_seen.get(c, 0) + self._rng.random() * 0.1)
        return self._easy_cat[best].pop(0)

    def _pick_stratified_hard(self) -> Optional[Dict]:
        """Round-robin over hard categories, skip empty ones."""
        cats = self._hard_cat_order
        n = len(cats)
        if n == 0:
            return None
        for _ in range(n):
            cat = cats[self._hard_cat_idx % n]
            self._hard_cat_idx += 1
            if self._hard_cat.get(cat):
                return self._hard_cat[cat].pop(0)
        return None  # all hard pools exhausted

    def select(self, pool: List[Dict]) -> Optional[Dict]:
        if self._easy_cat is None:
            self._init_pools(pool)

        self._step += 1
        self._phase_step += 1

        # Phase switching
        if self._phase == 0 and self._phase_step > self._easy_budget:
            self._phase = 1
            self._phase_step = 1
            print(f"  [StratPhased] → Phase 2 (hard stratified) at step {self._step}")

        if self._phase == 0:
            ex = self._pick_diverse_easy()
            if ex is None:
                self._phase = 1
                self._phase_step = 1
                ex = self._pick_stratified_hard()
        else:
            ex = self._pick_stratified_hard()
            if ex is None:
                ex = self._pick_diverse_easy()  # fallback

        if ex is not None:
            cat = ex.get("macro_category", "Other")
            self._cat_seen[cat] = self._cat_seen.get(cat, 0) + 1

        return ex

    def update(self, example: Dict, is_correct: bool):
        cat = example.get("macro_category", "Other")
        self._cat_total[cat] += 1
        self._cat_correct[cat] += int(is_correct)

    def current_phase(self) -> int:
        return self._phase

    def category_stats(self) -> Dict:
        return {
            cat: {
                "seen":    self._cat_seen.get(cat, 0),
                "correct": self._cat_correct[cat],
                "total":   self._cat_total[cat],
            }
            for cat in set(list(self._cat_seen) + list(self._cat_total))
        }
