"""
Data processor for LSAT Analytical Reasoning (AGIEval subset).

Task: single-choice logic puzzle / constraint satisfaction (A–E).

LSAT-AR is the hardest LSAT sub-task.  Each puzzle provides a set of
constraints (ordering, grouping, scheduling rules) and asks whether a
given arrangement is possible, which element must/cannot occupy a slot,
or which additional fact changes the answer.

Evaluation: exact letter match (A–E) with the same flexible extraction
used in lsat_lr, adapted to ensure only A–E are accepted as valid answers
(not accidentally matching wider letters from reasoning text).

ACE is expected to learn constraint-processing strategies:
  - Draw a constraint table / diagram before answering
  - Eliminate options that violate any single rule
  - Identify "if … then" chains between constraints
  - Flag fixed vs. floating elements early
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List


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


_VALID_LETTERS = set("ABCDE")


class DataProcessor:
    """Processor for LSAT Analytical Reasoning multiple-choice questions.

    Identical answer-extraction logic to lsat_lr; separated into its own
    module so ACE can maintain a distinct playbook per task.
    """

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

    # ------------------------------------------------------------------
    # Data formatting
    # ------------------------------------------------------------------

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert raw JSONL data to standardised ACE format."""
        return [
            {
                "context":  item.get("context", ""),
                "question": item.get("question", ""),
                "target":   item.get("target", "").strip().upper(),
                "others":   {"task": self.task_name},
            }
            for item in raw_data
        ]

    # ------------------------------------------------------------------
    # Letter extraction
    # ------------------------------------------------------------------

    def _extract_letter(self, text: str) -> str:
        """Extract the answer letter (A–E) from raw model output.

        Tries patterns in decreasing specificity.
        """
        if not text:
            return ""

        text = text.strip()

        # 1. Bare single letter
        if len(text) == 1 and text.upper() in _VALID_LETTERS:
            return text.upper()

        # 2. Starts with letter + punctuation/space  e.g. "C)" / "C." / "B: "
        m = re.match(r"^\(?([A-Ea-e])\)?[\s\)\.:\-]", text)
        if m:
            return m.group(1).upper()

        # 3. "The answer is X" / "Answer: X"
        m = re.search(
            r"(?:the\s+)?answer\s*(?:is|:)\s*\**\(?([A-Ea-e])\)?",
            text, re.IGNORECASE
        )
        if m:
            return m.group(1).upper()

        # 4. "I choose / I select / option X"
        m = re.search(
            r"(?:choose|select|pick|option)\s+\(?([A-Ea-e])\)?",
            text, re.IGNORECASE
        )
        if m:
            return m.group(1).upper()

        # 5. Bold markdown **X** or __X__
        m = re.search(r"(?:\*\*|__)([A-Ea-e])(?:\*\*|__)", text)
        if m:
            return m.group(1).upper()

        # 6. Bracket form [X]
        m = re.search(r"\[([A-Ea-e])\]", text)
        if m:
            return m.group(1).upper()

        # 7. Parenthesised option "(C)" anywhere
        m = re.search(r"\(([A-Ea-e])\)", text)
        if m:
            return m.group(1).upper()

        # 8. Last resort: first standalone A–E at word boundary
        m = re.search(r"\b([A-Ea-e])\b", text)
        if m:
            return m.group(1).upper()

        return ""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Return True if extracted letter matches ground truth (A–E)."""
        return self._extract_letter(predicted) == ground_truth.strip().upper()

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """Compute accuracy with parse-failure diagnostics.

        Returns:
            Accuracy (float in [0, 1]).
        """
        if len(out) != len(target):
            raise ValueError(
                f"Length mismatch: predictions={len(out)} vs targets={len(target)}"
            )

        correct = 0
        parse_fail = 0

        for pred, gt in zip(out, target):
            letter = self._extract_letter(pred)
            if not letter:
                parse_fail += 1
            elif letter == gt.strip().upper():
                correct += 1

        n = len(out) or 1
        print(f"  Total samples  : {n}")
        print(f"  Correct        : {correct}")
        print(f"  Parse failures : {parse_fail}")
        print(f"  Accuracy       : {correct}/{n} = {correct / n:.4f}")
        return correct / n
