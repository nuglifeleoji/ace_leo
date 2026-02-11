"""
Data processor for CLUTRR (Kinship Reasoning) task.

Task: Given a story about family members, infer the relationship between
two specified people by composing multiple relationship edges.

Answer format: A kinship relation word (e.g., "grandson", "aunt", "daughter-in-law")

Evaluation: Case-insensitive exact match with flexible extraction to handle
model output variations like "The relationship is grandson" or "Answer: nephew".
"""
import os
import re
import json
from typing import List, Dict, Any

# All valid kinship relations for extraction
VALID_RELATIONS = {
    "father", "mother", "son", "daughter",
    "grandfather", "grandmother", "grandson", "granddaughter",
    "brother", "sister", "uncle", "aunt", "nephew", "niece",
    "father-in-law", "mother-in-law", "son-in-law", "daughter-in-law",
    "husband", "wife", "cousin",
    "great-grandfather", "great-grandmother", "great-grandson", "great-granddaughter",
    "stepfather", "stepmother", "stepson", "stepdaughter",
    "half-brother", "half-sister",
}


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
    Processor for the CLUTRR kinship reasoning task.

    Handles extractive answers where the model should output a kinship
    relation word (e.g., "grandson", "aunt", "daughter-in-law").
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
                    "hop_count": item.get("hop_count", 0),
                    "reasoning_chain": item.get("reasoning_chain", ""),
                    "query": item.get("query", ""),
                    "task": self.task_name
                }
            }
            processed_data.append(processed_item)
        return processed_data

    def _extract_relation(self, text: str) -> str:
        """
        Extract kinship relation from model output.

        Handles various formats:
          - "grandson"
          - "The relationship is grandson"
          - "Answer: daughter-in-law"
          - "nephew."
          - "He is Donald's nephew"

        Returns:
            Extracted relation (lowercase), or the full text stripped if
            no known relation found.
        """
        if not text:
            return ""

        text = text.strip().lower()

        # Remove common prefixes
        text = re.sub(r'^(the\s+)?(answer|relationship|relation)\s*(is|:)\s*', '', text)
        text = re.sub(r'^(he|she|they)\s+(is|are)\s+(the\s+)?', '', text)
        text = re.sub(r"^[\w\s]+'s\s+", '', text)  # "Donald's nephew" -> "nephew"

        # Remove trailing punctuation
        text = text.rstrip('.,!;:')

        # Try to find a known relation in the text
        # Check compound relations first (e.g., "daughter-in-law")
        for rel in sorted(VALID_RELATIONS, key=len, reverse=True):
            if rel in text:
                return rel

        # Fallback: return cleaned text
        return text.strip()

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth."""
        pred_rel = self._extract_relation(predicted)
        truth_rel = ground_truth.strip().lower()
        return pred_rel == truth_rel

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """Calculate accuracy across multiple predictions."""
        if len(out) != len(target):
            raise ValueError("Predictions and ground truths must have the same length.")

        correct_count = 0
        extraction_details = {}

        for predicted, ground_truth in zip(out, target):
            pred_rel = self._extract_relation(predicted)
            truth_rel = ground_truth.strip().lower()

            if pred_rel == truth_rel:
                correct_count += 1

            # Track extraction for debugging
            key = f"{pred_rel} vs {truth_rel}"
            if pred_rel != truth_rel:
                extraction_details[key] = extraction_details.get(key, 0) + 1

        n = len(out) if out else 1

        print(f"  Total samples: {n}")
        print(f"  Correct: {correct_count}")
        print(f"  Accuracy: {correct_count}/{n} = {correct_count/n:.4f}")

        if extraction_details:
            print(f"  Top mismatches:")
            for k, v in sorted(extraction_details.items(), key=lambda x: -x[1])[:10]:
                print(f"    {k}: {v}")

        return correct_count / n
