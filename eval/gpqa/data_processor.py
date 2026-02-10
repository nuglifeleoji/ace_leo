import os
import re
import json
from typing import List, Dict, Any


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load and process data from a JSONL file.

    Args:
        data_path: Path to the JSONL file

    Returns:
        List of dictionaries containing the data
    """
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
    Processor for the GPQA (Graduate-Level Google-Proof Q&A) Diamond task.

    Handles 4-choice multiple-choice questions across Physics, Chemistry,
    and Biology domains. The model must select the correct lettered option
    (A, B, C, or D).
    """

    def __init__(self, task_name: str):
        """
        Initialize the data processor.

        Args:
            task_name: The name of the task (e.g., 'gpqa', 'gpqa_small')
        """
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw GPQA data into standardized format for ACE.

        Input format (from JSONL, produced by prepare_data.py):
            {
                "context": "",
                "question": "...\n\nA) ...\nB) ...\nC) ...\nD) ...\n\nAnswer with ONLY...",
                "target": "B",
                "domain": "Physics",
                "subdomain": "Quantum Mechanics",
                "answer_text": "...",
                "n_choices": 4
            }

        Output format (standardized for ACE):
            {
                "context": "",
                "question": "<question + choices + instruction>",
                "target": "B",
                "others": { ... metadata ... }
            }

        Args:
            raw_data: Raw data loaded from JSONL

        Returns:
            List of dicts in standardized format
        """
        processed_data = []

        for item in raw_data:
            processed_item = {
                "context": item.get("context", ""),
                "question": item.get("question", ""),
                "target": item.get("target", ""),
                "others": {
                    "domain": item.get("domain", ""),
                    "subdomain": item.get("subdomain", ""),
                    "answer_text": item.get("answer_text", ""),
                    "n_choices": item.get("n_choices", 4),
                    "task": self.task_name
                }
            }
            processed_data.append(processed_item)

        return processed_data

    def _extract_letter(self, text: str) -> str:
        """
        Extract the answer letter from model output.

        Handles various formats:
          - "A"
          - "A)"
          - "A."
          - "A) some text"
          - "The answer is A"
          - "Answer: B"
          - "**A**"
          - "I think it's C"

        Returns:
            Extracted letter (uppercase), or empty string if not found
        """
        if not text:
            return ""

        text = text.strip()

        # Direct single letter
        if len(text) == 1 and text.upper() in "ABCD":
            return text.upper()

        # Pattern: starts with letter followed by ) or . or space
        match = re.match(r'^([A-Da-d])\s*[\)\.:\s]', text)
        if match:
            return match.group(1).upper()

        # Pattern: "The answer is X" or "Answer: X" or "answer is X"
        match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\**([A-Da-d])\**', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern: **X** (bold letter)
        match = re.search(r'\*\*([A-Da-d])\*\*', text)
        if match:
            return match.group(1).upper()

        # Pattern: letter in brackets [X]
        match = re.search(r'\[([A-Da-d])\]', text)
        if match:
            return match.group(1).upper()

        # Pattern: "option X" or "choice X"
        match = re.search(r'(?:option|choice)\s+([A-Da-d])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Last resort: find last standalone A-D letter (models often put answer at end)
        matches = re.findall(r'\b([A-D])\b', text)
        if matches:
            return matches[-1]

        return ""

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.

        Uses flexible letter extraction to handle various model output formats.

        Args:
            predicted: Model's answer (may contain reasoning + letter)
            ground_truth: Ground truth letter (e.g., "A")

        Returns:
            bool: True if the extracted letter matches
        """
        pred_letter = self._extract_letter(predicted)
        truth_letter = ground_truth.strip().upper()

        return pred_letter == truth_letter

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """
        Calculate accuracy across multiple predictions.

        Args:
            out: List of model predictions
            target: List of ground truth targets

        Returns:
            Accuracy as float between 0 and 1
        """
        if len(out) != len(target):
            raise ValueError("Predictions and ground truths must have the same length.")

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
