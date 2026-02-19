"""
Data processor for LSAT Logical Reasoning (AGIEval subset).

Task: single-choice logical reasoning (A–E).

Evaluation: exact letter match with flexible extraction to handle
common model output variations:
  - bare letter:          "C"
  - letter + punct:       "C)" / "C." / "C:"
  - English prefix:       "The answer is C" / "Answer: C"
  - bold markdown:        "**C**"
  - bracket:              "[C]"
  - option full text:     "(C) Some legal scholars …" → extracts "C"
  - last-resort fallback: first standalone A–E in the response

LSAT question types covered in this dataset:
  - Assumption (the argument depends on / requires)
  - Weaken / Strengthen
  - Inference / Must be true
  - Principle / Parallel reasoning
  - Paradox / Resolve the discrepancy
  - Flaw / Method of reasoning

ACE is expected to learn per-type strategies in its playbook.
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


# Valid answer set for LSAT-LR (5 options A–E)
_VALID_LETTERS = set("ABCDE")


class DataProcessor:
    """Processor for LSAT Logical Reasoning multiple-choice questions.

    Handles single-select questions with options A through E.
    Answer checking uses flexible letter extraction to accommodate
    diverse model output formats.
    """

    def __init__(self, task_name: str) -> None:
        """
        Args:
            task_name: Task identifier (e.g. 'lsat_lr').
        """
        self.task_name = task_name

    # ------------------------------------------------------------------
    # Data formatting
    # ------------------------------------------------------------------

    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert raw JSONL data into standardised ACE format.

        Input fields (from prepare_data.py output):
            context  — argument / passage text
            question — question stem + choices + answer instruction
            target   — correct letter (A–E)

        Output adds an 'others' dict with task metadata.
        """
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

        Tries patterns in decreasing specificity; returns the first match
        or "" if nothing can be extracted.
        """
        if not text:
            return ""

        text = text.strip()

        # 1. Bare single letter
        if len(text) == 1 and text.upper() in _VALID_LETTERS:
            return text.upper()

        # 2. Starts with letter followed by ) . : or whitespace
        #    e.g. "C)" / "C. Some legal…" / "E: …"
        m = re.match(r"^\(?([A-Ea-e])\)?[\s\)\.:\-]", text)
        if m:
            return m.group(1).upper()

        # 3. "The answer is X" / "Answer: X" / "answer is X"
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

        # 7. Option pattern in text: "(C)" anywhere
        m = re.search(r"\(([A-Ea-e])\)", text)
        if m:
            return m.group(1).upper()

        # 8. Last resort: first standalone A–E word boundary
        m = re.search(r"\b([A-Ea-e])\b", text)
        if m:
            return m.group(1).upper()

        return ""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Return True if the extracted letter matches the ground truth.

        Args:
            predicted:    Raw model output string.
            ground_truth: Correct letter (A–E, from dataset).
        """
        pred_letter = self._extract_letter(predicted)
        truth_letter = ground_truth.strip().upper()
        return pred_letter == truth_letter

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """Compute accuracy with parse-failure diagnostics.

        Args:
            out:    List of raw model predictions.
            target: List of ground-truth letters.

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
