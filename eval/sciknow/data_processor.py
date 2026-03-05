"""
DataProcessor for SciKnowEval Chemistry MCQ tasks.
Author: Leo Ji
"""
import os
import re
import json
from typing import List, Dict, Any, Tuple


def load_data(data_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def _extract_letter(text: str) -> str:
    """
    Extract the answer letter/word from a model response.

    Strategy (in order):
    1. If the response is JSON, read the 'final_answer' field directly.
       This avoids false matches from SMILES strings in the reasoning.
    2. Look for explicit answer patterns like 'answer is A', 'Answer: B'.
    3. Search the last 100 characters for a bare A/B/C/D letter.
    4. Fall back to the first bare A/B/C/D anywhere in the text.
    5. Handle Yes/No for true_or_false questions.
    """
    text = text.strip()

    # Strip markdown code fences (e.g. ```json ... ```)
    if text.startswith("```"):
        # Remove opening fence line and closing fence
        lines = text.split("\n")
        # Drop first line (```json or ```) and last fence line if present
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip().startswith("```"):
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    # Strategy 1: parse JSON and extract final_answer
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            fa = parsed.get("final_answer", None)
            if fa is not None:
                fa = str(fa).strip()
                # Normalise: single letter → upper, Yes/No → capitalise
                if fa.upper() in ("A", "B", "C", "D"):
                    return fa.upper()
                if fa.lower() in ("yes", "no"):
                    return fa.lower().capitalize()
                return fa
        except Exception:
            pass

    # Strategy 2a: JSON-style "final_answer": "X" pattern (catches invalid/truncated JSON
    # where final_answer field appears but json.loads failed due to earlier syntax errors)
    m = re.search(r'"final_answer"\s*:\s*"([A-Da-d])"', text)
    if m:
        return m.group(1).upper()

    # Strategy 2b: explicit answer patterns
    m = re.search(
        r"(?:answer\s+is|answer:|final\s+answer[:\s]+)\s*([A-Da-d])\b",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Strategy 3: last 100 characters
    tail = text[-100:]
    m = re.search(r"\b([A-Da-d])\b", tail)
    if m:
        return m.group(1).upper()

    # Strategy 4: first occurrence anywhere
    text_lower = text.lower()
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()

    return ""


class DataProcessor:
    """
    DataProcessor for SciKnowEval Chemistry MCQ tasks.

    Supported task names:
        - 'sciknow_chem'     (combined L3+L4 MCQ)
        - 'sciknow_chem_l3'  (L3 MCQ only)
    """

    def __init__(self, task_name: str):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw JSONL records to ACE standard format.

        Raw fields:
            question  – question text (MCQ already has choices appended)
            target    – 'A'/'B'/'C'/'D' or 'Yes'/'No'
            level     – 'L3' or 'L4'
            task      – sub-task name
            type      – 'mcq-4-choices' or 'true_or_false'

        ACE standard keys: context, question, target, (others)
        """
        processed = []
        for item in raw_data:
            q_type = item.get("type", "mcq-4-choices")
            if q_type == "mcq-4-choices":
                instruction = (
                    "Given a question and four options, select the correct answer. "
                    "Your answer should be exactly one letter: A, B, C, or D."
                )
            else:
                instruction = (
                    "Read the following statement about laboratory safety or chemistry. "
                    "Answer 'Yes' if it is correct, or 'No' if it is incorrect."
                )
            processed.append({
                "context":  instruction,
                "question": item.get("question", ""),
                "target":   item.get("target", ""),
                "level":    item.get("level", ""),
                "task":     item.get("task", ""),
                "type":     q_type,
            })
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Case-insensitive exact match after extracting the answer token.
        """
        pred = _extract_letter(predicted.strip())
        gt   = ground_truth.strip().upper()
        return pred.upper() == gt

    def evaluate_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> float:
        """
        Calculate exact-match accuracy across the full set.
        Also prints per-type breakdown if 'others' metadata is available.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")
        correct = sum(
            self.answer_is_correct(p, g) for p, g in zip(predictions, ground_truths)
        )
        return correct / len(predictions)
