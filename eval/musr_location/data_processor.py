"""
Data processor for MuSR (Multistep Soft Reasoning) task.

MuSR is a benchmark for evaluating multi-step reasoning in LLMs. It contains
three subtasks requiring different reasoning strategies:

1. Murder Mysteries: Analyze detective narratives to identify the killer
   - Requires: Tracking motives, opportunities, alibis, evidence
   - Strategy: Process of elimination, evidence weighing

2. Object Placements: Track object movements across locations
   - Requires: State tracking, understanding implicit movements
   - Strategy: Chronological tracking, last-known-position reasoning

3. Team Allocation: Assign people to roles based on constraints
   - Requires: Constraint satisfaction, preference matching
   - Strategy: Systematic elimination, constraint checking

Each sample is a multiple-choice question:
  - context: A narrative story (2,000-7,000 characters)
  - question: The question + lettered choices + answer instruction
  - target: The correct letter (A, B, C, ...)

Evaluation: Exact letter match (case-insensitive), with flexible parsing
to handle common model output variations like "A)", "A.", "The answer is A".
"""
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
    Processor for the MuSR multi-step reasoning task.

    Handles multiple-choice questions where the model must select the correct
    lettered option (A, B, C, ...) based on a narrative story.
    """

    def __init__(self, task_name: str):
        """
        Initialize the data processor.

        Args:
            task_name: The name of the task (e.g., 'musr', 'musr_small')
        """
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw MuSR data into standardized format for ACE.

        Input format (from JSONL, produced by prepare_data.py):
            {
                "context": "In an adrenaline inducing bungee jumping site...",
                "question": "Who is the most likely murderer?\n\nA) Mackenzie\nB) Ana\n\nAnswer with ONLY the letter...",
                "target": "A",
                "subtask": "murder_mysteries",
                "answer_choice": "Mackenzie",
                "n_choices": 2
            }

        Output format (standardized for ACE):
            {
                "context": "<narrative story>",
                "question": "<question + choices + instruction>",
                "target": "A",
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
                    "subtask": item.get("subtask", ""),
                    "answer_choice": item.get("answer_choice", ""),
                    "n_choices": item.get("n_choices", 0),
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
          - "A) Mackenzie"
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
        if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return text.upper()

        # Pattern: starts with letter followed by ) or . or space
        match = re.match(r'^([A-Za-z])\s*[\)\.:\s]', text)
        if match:
            letter = match.group(1).upper()
            if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return letter

        # Pattern: "The answer is X" or "Answer: X" or "answer is X"
        match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\**([A-Za-z])\**', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern: **X** (bold letter)
        match = re.search(r'\*\*([A-Za-z])\*\*', text)
        if match:
            letter = match.group(1).upper()
            if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return letter

        # Pattern: letter in brackets [X]
        match = re.search(r'\[([A-Za-z])\]', text)
        if match:
            letter = match.group(1).upper()
            if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return letter

        # Last resort: find first standalone uppercase letter that could be an option
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1)

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

        Reports overall accuracy and per-subtask breakdown is not possible
        here (since we don't have subtask info in just the strings), but
        the main accuracy is computed.

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
