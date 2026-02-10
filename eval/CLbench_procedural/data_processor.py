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
    Processor for CL-bench Procedural Task Execution tasks.

    Each sample contains:
    - context: System instructions + procedural documents (workflows, operational
      manuals, instructional guides, etc.)
    - question: A task that requires following/executing a procedure from the context
    - target: Evaluation rubrics defining what a correct execution should include

    Sub-categories:
    - Workflow Orchestration: Multi-agent workflow coordination
    - Operational Procedures: Following specific operational guidelines
    - Instructional Procedures: Creating step-by-step instructions

    Evaluation uses rubric-based keyword matching (same as Rule System Application):
    - Extract key terms from each rubric
    - Check if the model's response contains those terms
    - Score = fraction of rubrics satisfied (threshold >= 50%)
    """

    def __init__(self, task_name: str):
        """
        Initialize the data processor.

        Args:
            task_name: The name of the task (e.g., 'procedural', 'procedural_small')
        """
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw CL-bench Procedural data into standardized format for ACE.

        Input format (from JSONL):
            {
                "context": "<system instructions + procedure document>",
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
                    "sub_category": "...",
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
                    "sub_category": metadata.get('sub_category', ''),
                    "task_id": metadata.get('task_id', ''),
                    "context_id": metadata.get('context_id', ''),
                    "task": self.task_name
                }
            }
            processed_data.append(processed_item)

        return processed_data

    def _extract_key_terms(self, rubric: str) -> List[str]:
        """
        Extract key terms from a rubric for keyword matching.

        Looks for:
        - Terms after "namely:", "such as", "for example"
        - Quoted strings
        - Numbers and specific values
        - Key nouns (fallback)
        """
        key_terms = []

        # Extract terms after "namely:" or "such as" or "for example"
        for pattern in [r'namely[:\s]+(.+?)(?:\.|$)',
                        r'such as\s+(.+?)(?:\.|$)',
                        r'for example[,:\s]+(.+?)(?:\.|$)']:
            matches = re.findall(pattern, rubric, re.IGNORECASE)
            for match in matches:
                terms = re.split(r',\s*|\s+and\s+', match)
                key_terms.extend(t.strip().strip('"\'') for t in terms if len(t.strip()) > 2)

        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', rubric)
        key_terms.extend(q for q in quoted if len(q) > 2)

        # Extract specific numbers/values
        numbers = re.findall(
            r'\b\d+(?:\.\d+)?(?:\s*(?:min|minutes|hours|degrees|%|percent|steps?|slides?))?\b',
            rubric
        )
        key_terms.extend(numbers)

        # Fallback: extract key nouns
        if not key_terms:
            common_words = {
                'should', 'would', 'could', 'response', 'include', 'explain',
                'describe', 'provide', 'mention', 'state', 'about', 'based',
                'information', 'following', 'example', 'answer', 'question',
                'correctly', 'accurately', 'relevant', 'specific', 'appropriate'
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
            # Fallback: simple word overlap
            rubric_words = set(re.findall(r'\b\w{4,}\b', rubric.lower()))
            pred_words = set(re.findall(r'\b\w{4,}\b', pred_lower))
            common = {
                'should', 'would', 'could', 'response', 'include', 'explain',
                'describe', 'provide', 'mention', 'state', 'about', 'based',
                'information', 'following', 'example', 'answer', 'question'
            }
            rubric_words -= common
            if not rubric_words:
                return True
            overlap = len(rubric_words & pred_words) / len(rubric_words)
            return overlap >= 0.3

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
