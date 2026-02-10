import os
import json
import re
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
    Processor for CL-bench Technical Standards tasks.

    Each sample contains:
    - context: System instructions + technical document (product manuals, API specs,
      engineering standards, safety regulations, etc.)
    - question: A question requiring understanding and application of the technical
      standards described in the context
    - target: Evaluation rubrics defining what a correct answer should contain

    Sub-category: Technical Standards (all 201 samples)
    Data split: Context-ID-grouped (no document leakage between splits)

    Evaluation uses rubric-based keyword matching:
    - Extract key terms from each rubric (quoted strings, specific values,
      terms after "namely:", "such as", etc.)
    - Check if the model's response contains those terms
    - A rubric is satisfied if >= 50% of its key terms appear in the response
    - A sample is correct if >= 50% of its rubrics are satisfied
    """

    def __init__(self, task_name: str):
        """
        Initialize the data processor.

        Args:
            task_name: The name of the task (e.g., 'technical', 'technical_small')
        """
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw CL-bench Technical Standards data into standardized format for ACE.

        Input format (from JSONL):
            {
                "context": "<system instructions + technical document>",
                "question": "<last user message>",
                "target": "<rubrics as evaluation criteria>",
                "metadata": {...}
            }

        Output format (standardized for ACE):
            {
                "context": "<context>",
                "question": "<question>",
                "target": "<rubrics text>",
                "others": {
                    "rubrics": [...],
                    "num_rubrics": int,
                    "context_id": "...",
                    "task_id": "...",
                    "task": "..."
                }
            }

        Args:
            raw_data: Raw data loaded from JSONL

        Returns:
            List of dicts in standardized format
        """
        processed_data = []

        for item in raw_data:
            context = item.get('context', '')
            question = item.get('question', '')
            target = item.get('target', '')
            metadata = item.get('metadata', {})

            processed_item = {
                "context": context,
                "question": question,
                "target": target,
                "others": {
                    "rubrics": metadata.get('rubrics', []),
                    "num_rubrics": metadata.get('num_rubrics', 0),
                    "context_id": metadata.get('context_id', ''),
                    "task_id": metadata.get('task_id', ''),
                    "task": self.task_name
                }
            }
            processed_data.append(processed_item)

        return processed_data

    def _extract_key_terms(self, rubric: str) -> List[str]:
        """
        Extract key terms from a rubric for keyword matching.

        Extraction priority:
        1. Terms after "namely:", "such as", "for example", "e.g."
        2. Quoted strings (double-quoted)
        3. Specific numbers/values with units
        4. Technical terms (capitalized multi-word phrases)
        5. Fallback: significant nouns (>= 5 chars, not common words)
        """
        key_terms = []

        # 1. Terms after signal phrases
        for pattern in [
            r'namely[:\s]+(.+?)(?:\.|$)',
            r'such as\s+(.+?)(?:\.|$)',
            r'for example[,:\s]+(.+?)(?:\.|$)',
            r'e\.g\.[,:\s]+(.+?)(?:\.|$)',
            r'including\s+(.+?)(?:\.|$)',
            r'specifically[,:\s]+(.+?)(?:\.|$)',
        ]:
            matches = re.findall(pattern, rubric, re.IGNORECASE)
            for match in matches:
                terms = re.split(r',\s*|\s+and\s+', match)
                key_terms.extend(t.strip().strip('"\'') for t in terms if len(t.strip()) > 2)

        # 2. Quoted strings
        quoted = re.findall(r'"([^"]+)"', rubric)
        key_terms.extend(q for q in quoted if len(q) > 2)

        # 3. Numbers/values with optional units
        numbers = re.findall(
            r'\b\d+(?:\.\d+)?(?:\s*(?:min|minutes|hours|seconds|degrees|°[CF]|'
            r'%|percent|V|W|A|Hz|kHz|MHz|GHz|mm|cm|m|kg|g|lbs?|psi|bar|dB))?\b',
            rubric
        )
        key_terms.extend(numbers)

        # 4. Technical terms (capitalized phrases that aren't sentence starters)
        tech_terms = re.findall(r'(?<!\. )(?<!\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', rubric)
        key_terms.extend(t for t in tech_terms if len(t) > 5)

        # 5. Fallback: significant nouns
        if not key_terms:
            common_words = {
                'should', 'would', 'could', 'response', 'include', 'explain',
                'describe', 'provide', 'mention', 'state', 'about', 'based',
                'information', 'following', 'example', 'answer', 'question',
                'correctly', 'accurately', 'relevant', 'specific', 'appropriate',
                'manual', 'document', 'section', 'standard', 'specification'
            }
            words = re.findall(r'\b[A-Za-z]{5,}\b', rubric)
            key_terms = [w for w in words if w.lower() not in common_words][:3]

        return key_terms

    def _rubric_satisfied(self, prediction: str, rubric: str) -> bool:
        """
        Check if a single rubric is satisfied by the prediction.

        Uses keyword matching: extracts key terms from the rubric
        and checks if they appear in the prediction.
        """
        pred_lower = prediction.lower()
        key_terms = self._extract_key_terms(rubric)

        if not key_terms:
            # Fallback: word overlap check
            rubric_words = set(re.findall(r'\b\w{4,}\b', rubric.lower()))
            pred_words = set(re.findall(r'\b\w{4,}\b', pred_lower))
            common = {
                'should', 'would', 'could', 'response', 'include', 'explain',
                'describe', 'provide', 'mention', 'state', 'about', 'based',
                'information', 'following', 'example', 'answer', 'question'
            }
            rubric_words -= common
            if not rubric_words:
                return True  # Trivial rubric
            overlap = len(rubric_words & pred_words) / len(rubric_words)
            return overlap >= 0.3

        # Check if enough key terms appear in prediction
        found = sum(1 for term in key_terms if term.lower() in pred_lower)
        return found >= max(1, len(key_terms) * 0.5)

    def _compute_rubric_score(self, prediction: str, rubrics: List[str]) -> float:
        """
        Compute the fraction of rubrics satisfied by the prediction.
        """
        if not rubrics:
            return 0.0
        satisfied = sum(1 for r in rubrics if self._rubric_satisfied(prediction, r))
        return satisfied / len(rubrics)

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction satisfies the rubrics.

        Extracts rubrics from the ground_truth string and checks
        if the prediction satisfies >= 50% of them.

        Args:
            predicted: Model's answer
            ground_truth: Target string containing rubrics

        Returns:
            bool: True if >= 50% rubrics satisfied
        """
        predicted = predicted.strip()
        ground_truth = ground_truth.strip()

        # Extract individual rubrics from target string
        # Format: "Evaluation criteria (N rubrics):\n- rubric1\n- rubric2\n..."
        rubric_lines = []
        for line in ground_truth.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                rubric_lines.append(line[2:])

        if not rubric_lines:
            # Fallback: simple text overlap
            gt_words = set(ground_truth.lower().split())
            pred_words = set(predicted.lower().split())
            if not gt_words:
                return True
            overlap = len(gt_words & pred_words) / len(gt_words)
            return overlap >= 0.3

        score = self._compute_rubric_score(predicted, rubric_lines)
        return score >= 0.5

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """
        Calculate accuracy across multiple predictions.

        Reports two metrics:
        - Binary accuracy: fraction of samples with >= 50% rubrics satisfied
        - Average rubric satisfaction: mean fraction of rubrics satisfied per sample

        Args:
            out: List of model predictions
            target: List of ground truth targets (rubric strings)

        Returns:
            Accuracy as float between 0 and 1
        """
        if len(out) != len(target):
            raise ValueError("Predictions and ground truths must have the same length.")

        correct_count = 0
        for predicted, ground_truth in zip(out, target):
            if self.answer_is_correct(predicted, ground_truth):
                correct_count += 1

        accuracy = correct_count / len(out) if out else 0.0

        # Also compute average rubric satisfaction for more granular metric
        rubric_scores = []
        for predicted, ground_truth in zip(out, target):
            rubric_lines = []
            for line in ground_truth.strip().split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    rubric_lines.append(line[2:])
            if rubric_lines:
                score = self._compute_rubric_score(predicted.strip(), rubric_lines)
                rubric_scores.append(score)

        avg_rubric_score = sum(rubric_scores) / len(rubric_scores) if rubric_scores else 0.0
        print(f"  Accuracy (>=50% rubrics): {accuracy:.4f} ({correct_count}/{len(out)})")
        print(f"  Avg rubric satisfaction: {avg_rubric_score:.4f}")

        return accuracy
