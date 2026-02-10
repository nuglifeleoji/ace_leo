"""
Data processor for MATH-500 benchmark.

MATH-500 is a curated subset of 500 competition-level math problems spanning
7 subjects (Algebra, Geometry, Number Theory, etc.) and 5 difficulty levels.

Each sample:
  - context: "" (empty, no separate context)
  - question: The math problem + instruction to put answer in \\boxed{}
  - target: The ground truth answer (numeric or LaTeX expression)

Evaluation: Extract \\boxed{} answer from model output, normalize both
prediction and ground truth, then compare. Handles fractions, square roots,
scientific notation, and other LaTeX math expressions.
"""
import os
import re
import json
from typing import List, Dict, Any


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

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
    Processor for MATH-500 benchmark.

    Handles extraction and comparison of mathematical answers, including
    numeric values, fractions, algebraic expressions, and LaTeX notation.
    """

    def __init__(self, task_name: str):
        """
        Initialize the data processor.

        Args:
            task_name: The name of the task (e.g., 'math500', 'math500_small')
        """
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw MATH-500 data into standardized format for ACE.

        Input format (from JSONL):
            {
                "context": "",
                "question": "Convert the point (0,3)... Put your final answer in \\boxed{}.",
                "target": "\\left( 3, \\frac{\\pi}{2} \\right)",
                "subject": "Precalculus",
                "level": 2,
                "solution": "We have that r = ...",
                "unique_id": "test/precalculus/807.json"
            }

        Output format (standardized for ACE):
            {
                "context": "",
                "question": "<problem + instruction>",
                "target": "<answer>",
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
                    "subject": item.get("subject", ""),
                    "level": item.get("level", 0),
                    "unique_id": item.get("unique_id", ""),
                    "solution": item.get("solution", ""),
                    "task": self.task_name
                }
            }
            processed_data.append(processed_item)

        return processed_data

    def _extract_boxed(self, text: str) -> str:
        """
        Extract the content inside \\boxed{} from model output.

        Handles nested braces correctly.

        Args:
            text: Model output text

        Returns:
            Content inside \\boxed{}, or the last number/expression found
        """
        if not text:
            return ""

        # Find all \boxed{...} occurrences (handle nested braces)
        boxed_contents = []
        search_text = text
        while True:
            idx = search_text.find("\\boxed{")
            if idx == -1:
                # Also try \boxed without backslash (some models output it differently)
                idx = search_text.find("boxed{")
                if idx == -1:
                    break
                start = idx + 6
            else:
                start = idx + 7

            # Find matching closing brace
            depth = 1
            pos = start
            while pos < len(search_text) and depth > 0:
                if search_text[pos] == '{':
                    depth += 1
                elif search_text[pos] == '}':
                    depth -= 1
                pos += 1

            if depth == 0:
                content = search_text[start:pos - 1]
                boxed_contents.append(content)
                search_text = search_text[pos:]
            else:
                break

        if boxed_contents:
            # Return the last boxed content (usually the final answer)
            return boxed_contents[-1].strip()

        # Fallback: try to find "the answer is X" pattern
        match = re.search(r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|$)',
                          text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Last fallback: extract the last number or expression
        # Look for the last standalone number
        numbers = re.findall(r'(?:^|[\s=])(-?\d+(?:\.\d+)?(?:/\d+)?)', text)
        if numbers:
            return numbers[-1].strip()

        return text.strip()[-50:] if text.strip() else ""

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize a math answer for comparison.

        Handles:
        - Whitespace removal
        - LaTeX command cleanup
        - Fraction normalization
        - Common equivalent forms

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer string
        """
        if not answer:
            return ""

        s = answer.strip()

        # Remove \text{} wrappers but keep content
        s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
        # Remove \mathrm{} wrappers
        s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
        # Remove \left and \right
        s = s.replace("\\left", "").replace("\\right", "")
        # Remove \, (thin space) and other spacing
        s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "")
        s = s.replace("\\!", "")
        # Remove $ signs
        s = s.replace("$", "")
        # Remove \displaystyle
        s = s.replace("\\displaystyle", "")
        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s).strip()

        # Try to evaluate simple fractions
        s = self._try_eval_fraction(s)

        return s

    def _try_eval_fraction(self, s: str) -> str:
        """
        Try to evaluate \\frac{a}{b} to a decimal for comparison.
        If it can't be evaluated, return the cleaned string.
        """
        # Match \frac{numerator}{denominator}
        frac_match = re.match(r'^\\frac\{([^}]+)\}\{([^}]+)\}$', s.strip())
        if frac_match:
            try:
                num = float(frac_match.group(1))
                den = float(frac_match.group(2))
                if den != 0:
                    result = num / den
                    # Return as fraction string for exact comparison
                    return f"{num}/{den}" if num == int(num) and den == int(den) else str(result)
            except (ValueError, ZeroDivisionError):
                pass

        # Also handle simple a/b format
        simple_frac = re.match(r'^(-?\d+)/(\d+)$', s.strip())
        if simple_frac:
            try:
                num = int(simple_frac.group(1))
                den = int(simple_frac.group(2))
                if den != 0:
                    return f"{num}/{den}"
            except ValueError:
                pass

        return s

    def _normalize_numeric(self, s: str) -> float:
        """
        Try to convert a string to a float for numeric comparison.

        Returns None if not a simple number.
        """
        try:
            # Remove trailing periods or commas
            cleaned = s.strip().rstrip('.')
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _answers_equivalent(self, pred: str, truth: str) -> bool:
        """
        Check if two math answers are equivalent.

        Uses multiple comparison strategies:
        1. Exact string match (after normalization)
        2. Numeric comparison (with tolerance)
        3. Fraction equivalence

        Args:
            pred: Normalized predicted answer
            truth: Normalized ground truth answer

        Returns:
            True if answers are equivalent
        """
        # 1. Exact match
        if pred == truth:
            return True

        # 2. Case-insensitive match
        if pred.lower() == truth.lower():
            return True

        # 3. Numeric comparison
        pred_num = self._normalize_numeric(pred)
        truth_num = self._normalize_numeric(truth)
        if pred_num is not None and truth_num is not None:
            if abs(pred_num - truth_num) < 1e-6:
                return True

        # 4. Fraction comparison: convert both to float
        pred_frac = self._try_parse_fraction(pred)
        truth_frac = self._try_parse_fraction(truth)
        if pred_frac is not None and truth_frac is not None:
            if abs(pred_frac - truth_frac) < 1e-6:
                return True

        # 5. Remove all spaces and compare
        if pred.replace(" ", "") == truth.replace(" ", ""):
            return True

        return False

    def _try_parse_fraction(self, s: str) -> float:
        """Try to parse a fraction string to float."""
        # Handle \frac{a}{b}
        frac_match = re.match(r'^\\frac\{([^}]+)\}\{([^}]+)\}$', s.strip())
        if frac_match:
            try:
                return float(frac_match.group(1)) / float(frac_match.group(2))
            except (ValueError, ZeroDivisionError):
                pass

        # Handle a/b
        simple_match = re.match(r'^(-?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$', s.strip())
        if simple_match:
            try:
                return float(simple_match.group(1)) / float(simple_match.group(2))
            except (ValueError, ZeroDivisionError):
                pass

        # Handle plain number
        try:
            return float(s.strip())
        except ValueError:
            return None

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.

        Extracts \\boxed{} content from prediction, normalizes both sides,
        then compares using multiple strategies.

        Args:
            predicted: Model's full response
            ground_truth: Ground truth answer string

        Returns:
            bool: True if answers are equivalent
        """
        # Extract answer from prediction
        pred_answer = self._extract_boxed(predicted)
        # Normalize both
        pred_norm = self._normalize_answer(pred_answer)
        truth_norm = self._normalize_answer(ground_truth)

        return self._answers_equivalent(pred_norm, truth_norm)

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """
        Calculate accuracy across multiple predictions.

        Args:
            out: List of model predictions (full responses)
            target: List of ground truth answers

        Returns:
            Accuracy as float between 0 and 1
        """
        if len(out) != len(target):
            raise ValueError("Predictions and ground truths must have the same length.")

        correct_count = 0
        no_boxed = 0

        for predicted, ground_truth in zip(out, target):
            pred_answer = self._extract_boxed(predicted)

            if not pred_answer:
                no_boxed += 1
                continue

            pred_norm = self._normalize_answer(pred_answer)
            truth_norm = self._normalize_answer(ground_truth)

            if self._answers_equivalent(pred_norm, truth_norm):
                correct_count += 1

        n = len(out) if out else 1

        print(f"  Total samples: {n}")
        print(f"  Correct: {correct_count}")
        print(f"  No \\boxed{{}} found: {no_boxed}")
        print(f"  Accuracy: {correct_count}/{n} = {correct_count/n:.4f}")

        return correct_count / n
