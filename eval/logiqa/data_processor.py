"""
Data processor for LogiQA 2.0 (Logical Reasoning) task.

Task: Given a passage and a logical reasoning question, select the correct
answer from 4 options (A-D).

Covers reasoning types:
  - Sufficient Conditional Reasoning
  - Necessary Conditional Reasoning
  - Conjunctive Reasoning
  - Disjunctive Reasoning
  - Categorical Reasoning

Evaluation: Exact letter match (case-insensitive) with flexible extraction.
"""
import os
import re
import json
from typing import List, Dict, Any


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


class DataProcessor:
    """
    Processor for LogiQA 2.0 logical reasoning task.

    Multiple-choice with 4 options (A-D).
    """

    def __init__(self, task_name: str):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Convert raw data into standardized ACE format."""
        processed_data = []
        for item in raw_data:
            processed_item = {
                "context": item.get("context", ""),
                "question": item.get("question", ""),
                "target": item.get("target", ""),
                "others": {
                    "reasoning_type": item.get("reasoning_type", ""),
                    "task": self.task_name
                }
            }
            processed_data.append(processed_item)
        return processed_data

    def _extract_letter(self, text: str) -> str:
        """
        Extract answer letter (A-D) from model output.

        Handles various formats:
          - "A", "B)", "C.", "D) some text"
          - "The answer is A"
          - "**B**", "[C]"
        """
        if not text:
            return ""

        text = text.strip()

        # Direct single letter
        if len(text) == 1 and text.upper() in "ABCD":
            return text.upper()

        # Starts with letter + delimiter
        match = re.match(r'^([A-Da-d])\s*[\)\.:\s]', text)
        if match:
            return match.group(1).upper()

        # "The answer is X" or "Answer: X"
        match = re.search(
            r'(?:the\s+)?answer\s*(?:is|:)\s*\**([A-Da-d])\**',
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # **X** (bold)
        match = re.search(r'\*\*([A-Da-d])\*\*', text)
        if match:
            return match.group(1).upper()

        # [X]
        match = re.search(r'\[([A-Da-d])\]', text)
        if match:
            return match.group(1).upper()

        # "option X" or "choice X"
        match = re.search(
            r'(?:option|choice)\s*([A-Da-d])', text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Last resort: first standalone letter A-D
        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1)

        return ""

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        pred_letter = self._extract_letter(predicted)
        truth_letter = ground_truth.strip().upper()
        return pred_letter == truth_letter

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        if len(out) != len(target):
            raise ValueError(
                "Predictions and ground truths must have the same length."
            )

        correct_count = 0
        parse_failures = 0

        for predicted, ground_truth in zip(out, target):
            pred_letter = self._extract_letter(predicted)

            if not pred_letter:
                parse_failures += 1
                continue

            if pred_letter == ground_truth.strip().upper():
                correct_count += 1

        n = len(out) if out else 1

        print(f"  Total samples: {n}")
        print(f"  Correct: {correct_count}")
        print(f"  Parse failures (no letter extracted): {parse_failures}")
        print(f"  Accuracy: {correct_count}/{n} = {correct_count/n:.4f}")

        return correct_count / n
