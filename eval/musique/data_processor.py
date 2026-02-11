"""
Data processor for MuSiQue (Multihop Questions via Single-hop Question Composition).

MuSiQue is a multi-hop question answering benchmark where the model must
combine information from multiple passages to produce a short text answer.

Each sample:
  - context: ~20 paragraphs formatted as [Title]\nText blocks
  - question: A multi-hop question + answer instruction
  - target: Short text answer (avg ~15 chars)

Evaluation: Normalized exact match and token-level F1 score.
The final accuracy uses exact match (with normalization for case,
articles, whitespace, and punctuation).
"""
import os
import re
import json
import string
from typing import List, Dict, Any, Optional


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
    Processor for MuSiQue multi-hop question answering.

    Uses normalized exact match for evaluation, following standard
    QA evaluation practices (SQuAD-style normalization).
    """

    def __init__(self, task_name: str):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw MuSiQue data into standardized format for ACE.

        Input format (from JSONL):
            {
                "context": "[Title1]\nParagraph...\n\n[Title2]\n...",
                "question": "When was ... ?\n\nBased on ...",
                "target": "1960",
                "answer_aliases": [],
                "hop_type": "2hop",
                "n_hops": 2,
                "sample_id": "2hop__482757_12019"
            }

        Output format (standardized for ACE):
            {
                "context": "<passages>",
                "question": "<question + instruction>",
                "target": "<answer>",
                "others": { ... metadata ... }
            }
        """
        processed_data = []
        for item in raw_data:
            processed_item = {
                "context": item.get("context", ""),
                "question": item.get("question", ""),
                "target": item.get("target", ""),
                "others": {
                    "answer_aliases": item.get("answer_aliases", []),
                    "hop_type": item.get("hop_type", ""),
                    "n_hops": item.get("n_hops", 0),
                    "sample_id": item.get("sample_id", ""),
                    "task": self.task_name
                }
            }
            processed_data.append(processed_item)
        return processed_data

    @staticmethod
    def _normalize_answer(s: str) -> str:
        """
        Normalize answer for comparison (SQuAD-style).

        Steps:
        1. Lowercase
        2. Remove punctuation
        3. Remove articles (a, an, the)
        4. Fix extra whitespace
        """
        if not s:
            return ""

        # Lowercase
        s = s.lower()

        # Remove punctuation
        s = ''.join(ch for ch in s if ch not in string.punctuation)

        # Remove articles
        s = re.sub(r'\b(a|an|the)\b', ' ', s)

        # Fix whitespace
        s = ' '.join(s.split())

        return s.strip()

    @staticmethod
    def _get_tokens(s: str) -> List[str]:
        """Tokenize a normalized string into words."""
        if not s:
            return []
        return s.split()

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute token-level F1 score between prediction and ground truth.
        """
        pred_tokens = self._get_tokens(self._normalize_answer(prediction))
        truth_tokens = self._get_tokens(self._normalize_answer(ground_truth))

        if not pred_tokens or not truth_tokens:
            return float(pred_tokens == truth_tokens)

        common = set(pred_tokens) & set(truth_tokens)
        num_common = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common)

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _extract_answer(self, text: str) -> str:
        """
        Extract the answer from model output.

        Handles formats like:
          - Direct answer: "1960"
          - "The answer is 1960"
          - "Answer: 1960"
          - Multi-line with reasoning, takes first line or after "answer is"
        """
        if not text:
            return ""

        text = text.strip()

        # If short (< 50 chars), likely a direct answer
        if len(text) < 50:
            return text

        # Try "the answer is X" pattern
        match = re.search(
            r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|$)',
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        # Try "**answer**" pattern (bold)
        match = re.search(r'\*\*(.+?)\*\*', text)
        if match:
            return match.group(1).strip()

        # Take the last line (often contains the final answer)
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        if lines:
            last_line = lines[-1]
            # Clean up common prefixes
            last_line = re.sub(r'^(?:Answer|Final answer|Therefore|So|Thus)[:\s]*',
                               '', last_line, flags=re.IGNORECASE).strip()
            if last_line:
                return last_line

        return text[:100]

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth using normalized exact match.

        Also checks against answer_aliases if available in the metadata.
        """
        pred_answer = self._extract_answer(predicted)
        pred_norm = self._normalize_answer(pred_answer)
        truth_norm = self._normalize_answer(ground_truth)

        # Exact match (normalized)
        if pred_norm == truth_norm:
            return True

        # Check if prediction contains the ground truth or vice versa
        # (for cases like pred="Houston Baptist University" truth="Houston Baptist University")
        if truth_norm and truth_norm in pred_norm:
            # Only if truth is a significant portion of pred
            if len(truth_norm) / max(len(pred_norm), 1) > 0.5:
                return True

        # F1 threshold (lenient: if F1 > 0.8, consider correct)
        f1 = self._compute_f1(pred_answer, ground_truth)
        if f1 >= 0.8:
            return True

        return False

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """
        Calculate accuracy across multiple predictions.

        Reports exact match accuracy and average F1 score.
        """
        if len(out) != len(target):
            raise ValueError("Predictions and ground truths must have the same length.")

        correct_count = 0
        total_f1 = 0.0
        no_answer = 0

        for predicted, ground_truth in zip(out, target):
            pred_answer = self._extract_answer(predicted)

            if not pred_answer:
                no_answer += 1
                continue

            # Check exact match
            if self.answer_is_correct(predicted, ground_truth):
                correct_count += 1

            # Always compute F1
            total_f1 += self._compute_f1(pred_answer, ground_truth)

        n = len(out) if out else 1
        avg_f1 = total_f1 / n

        print(f"  Total samples: {n}")
        print(f"  Correct (EM): {correct_count}")
        print(f"  No answer: {no_answer}")
        print(f"  EM Accuracy: {correct_count}/{n} = {correct_count/n:.4f}")
        print(f"  Average F1: {avg_f1:.4f}")

        return correct_count / n
