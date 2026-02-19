"""
Data processor for ASDiv (Arithmetic Story Problems).

Task: Given a story/context and a question, produce the final numeric answer.

Evaluation:
  - Robust numeric extraction from model output (handles commas, fractions,
    negative numbers, simple percentages).
  - Compare with ground truth using exact Fraction equality when possible,
    otherwise use a small float tolerance.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


_RE_FRACTION = re.compile(r"(?<!\d)(-?\d+)\s*/\s*(\d+)(?!\d)")
_RE_NUMBER = re.compile(r"(?<![\w/])(-?\d+(?:\.\d+)?)(?![\w/])")
_RE_PERCENT = re.compile(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*%(?!\d)")


def _strip_common(s: str) -> str:
    s = s.strip()
    # remove common wrappers
    s = s.replace(",", "")
    s = s.replace("$", "")
    return s


def _decimal_to_fraction(x: str) -> Optional[Fraction]:
    try:
        d = Decimal(x)
    except InvalidOperation:
        return None
    # limit denominator to keep things stable
    return Fraction(d).limit_denominator(10_000)


def _parse_numeric(s: str) -> Optional[Fraction]:
    """
    Parse a numeric string into a Fraction.
    Supports:
      - integers, decimals
      - simple fractions a/b
      - percentages like 12.5% (converted to /100)
    """
    if s is None:
        return None
    s = str(s)
    # ASDiv sometimes uses a derivation marker like: "<<10-2+2>>10"
    # Keep only the final answer part after the last ">>".
    if ">>" in s:
        s = s.split(">>")[-1]
    # Some datasets use "#### 10" style.
    if "####" in s:
        s = s.split("####")[-1]
    s = _strip_common(s)
    if not s:
        return None

    # percentage
    if s.endswith("%"):
        inner = s[:-1].strip()
        f = _decimal_to_fraction(inner)
        return (f / 100) if f is not None else None

    # fraction a/b
    m = _RE_FRACTION.fullmatch(s.replace(" ", ""))
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        if den == 0:
            return None
        return Fraction(num, den)

    # decimal or int
    f = _decimal_to_fraction(s)
    return f


def _extract_candidate_numbers(text: str) -> List[str]:
    """
    Extract numeric candidates from a model output string.
    We try to capture:
      - percentages
      - fractions
      - plain numbers
    """
    if not text:
        return []
    t = _strip_common(text.lower())

    # Prefer explicit "answer" section if present
    # e.g. "Final answer: 12.5"
    for key in ["final answer", "answer", "ans"]:
        if key in t:
            t = t.split(key, 1)[1]
            break

    cands: List[str] = []

    # percentages first
    cands.extend([m.group(1) + "%" for m in _RE_PERCENT.finditer(t)])
    # fractions
    cands.extend([f"{m.group(1)}/{m.group(2)}" for m in _RE_FRACTION.finditer(t)])
    # plain numbers
    cands.extend([m.group(1) for m in _RE_NUMBER.finditer(t)])

    # de-dup (preserve order)
    seen = set()
    out: List[str] = []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


@dataclass(frozen=True)
class NumericMatch:
    pred: Fraction
    truth: Fraction


def _numbers_match(pred: Fraction, truth: Fraction) -> bool:
    """Exact match if possible, else small tolerance on float."""
    if pred == truth:
        return True
    # fallback tolerance (handles e.g. 0.3333 vs 1/3)
    pf = float(pred)
    tf = float(truth)
    return abs(pf - tf) <= 1e-6 or abs(pf - tf) <= 1e-4 * max(1.0, abs(tf))


class DataProcessor:
    """ASDiv processor (numeric QA)."""

    def __init__(self, task_name: str):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for item in raw_data:
            processed.append(
                {
                    "context": item.get("context", ""),
                    "question": item.get("question", ""),
                    "target": item.get("target", ""),
                    "others": {"task": self.task_name},
                }
            )
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        truth_f = _parse_numeric(ground_truth)
        if truth_f is None:
            # last resort: exact normalized string
            return _strip_common(str(predicted)).strip().lower() == _strip_common(str(ground_truth)).strip().lower()

        cands = _extract_candidate_numbers(predicted)
        if not cands:
            return False

        # pick the last candidate (usually the final answer)
        for cand in reversed(cands):
            pred_f = _parse_numeric(cand)
            if pred_f is None:
                continue
            if _numbers_match(pred_f, truth_f):
                return True
        return False

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        if len(out) != len(target):
            raise ValueError("Predictions and ground truths must have the same length.")

        correct = 0
        parse_fail = 0
        for pred, gt in zip(out, target):
            if not _extract_candidate_numbers(pred):
                parse_fail += 1
            if self.answer_is_correct(pred, gt):
                correct += 1

        n = len(out) if out else 1
        print(f"  Total samples: {n}")
        print(f"  Correct: {correct}")
        print(f"  Parse failures (no numeric extracted): {parse_fail}")
        print(f"  Accuracy: {correct}/{n} = {correct/n:.4f}")
        return correct / n

