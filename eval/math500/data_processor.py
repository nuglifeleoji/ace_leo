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
scientific notation, degree symbols, commas in numbers, and other LaTeX
math expressions.
"""
import os
import re
import json
from typing import List, Dict, Any, Optional


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
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Convert raw MATH-500 data into standardized format for ACE."""
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
        Handles nested braces correctly. Returns the last \\boxed{} found.
        """
        if not text:
            return ""

        boxed_contents = []
        search_text = text
        while True:
            idx = search_text.find("\\boxed{")
            if idx == -1:
                idx = search_text.find("boxed{")
                if idx == -1:
                    break
                start = idx + 6
            else:
                start = idx + 7

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
            return boxed_contents[-1].strip()

        # Fallback: "the answer is X"
        match = re.search(r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|$)',
                          text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Last fallback: last standalone number
        numbers = re.findall(r'(?:^|[\s=])(-?\d+(?:\.\d+)?(?:/\d+)?)', text)
        if numbers:
            return numbers[-1].strip()

        return text.strip()[-50:] if text.strip() else ""

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize a math answer for comparison.

        Handles:
        - LaTeX wrappers (\text{}, \mathrm{}, \left, \right, \dfrac -> \frac)
        - Spacing commands (\, \; \: \! \quad)
        - Dollar signs, \displaystyle, \boxed{}
        - Degree symbols (^\circ, °)
        - Thousand separators (commas and \! in numbers)
        - Base subscripts (e.g., 2516_8 vs 2516_{8})
        - Parenthesized labels like \text{(B)} -> B
        """
        if not answer:
            return ""

        s = answer.strip()

        # Remove outer \boxed{} if present
        if s.startswith("\\boxed{") and s.endswith("}"):
            s = s[7:-1].strip()

        # Remove \text{} but keep content
        s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)
        # Remove \mathrm{} but keep content
        s = re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', s)
        # Remove \textbf, \textit wrappers
        s = re.sub(r'\\textbf\s*\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\textit\s*\{([^}]*)\}', r'\1', s)

        # \dfrac -> \frac
        s = s.replace("\\dfrac", "\\frac")

        # Remove \left and \right
        s = s.replace("\\left", "").replace("\\right", "")

        # Remove thousand separators BEFORE general spacing removal
        # "10,\! 080" -> "10080", "58,500" -> "58500"
        # Step 1: Remove LaTeX comma+spacing combos: ,\! or ,\, or ,\; etc.
        # (Must happen before \\! is stripped by general spacing removal below)
        s = re.sub(r',\s*\\!\s*', '', s)   # ,\! (thin negative space)
        s = re.sub(r',\s*\\,\s*', '', s)   # ,\, (thin space)
        s = re.sub(r',\s*\\;\s*', '', s)   # ,\; (medium space)
        s = re.sub(r',\s*\\:\s*', '', s)   # ,\: (medium space)
        # Step 2: Remove commas that are thousand separators (digit,3-digit groups)
        # Apply repeatedly: "11,111,111,100" -> "11111111100"
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r'(\d),\s*(\d{3})(?=,\d|[^,\d]|$)', r'\1\2', s)

        # Remove spacing commands (after thousand separator handling)
        s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "")
        s = s.replace("\\!", "").replace("\\quad", " ").replace("\\qquad", " ")

        # Remove $ signs
        s = s.replace("$", "")

        # Remove \displaystyle
        s = s.replace("\\displaystyle", "")

        # Normalize degree: ^\circ -> °, then remove °
        s = re.sub(r'\^\\circ', '°', s)
        s = s.replace("°", "")

        # Remove \$ (dollar sign in LaTeX)
        s = s.replace("\\$", "")

        # Normalize subscripts: both 2516_8 and 2516_{8} -> 2516_8
        s = re.sub(r'_\{([^}]+)\}', r'_\1', s)

        # Normalize parenthesized answers: (B) -> B for single letter answers
        paren_match = re.match(r'^\(([A-Za-z])\)$', s.strip())
        if paren_match:
            s = paren_match.group(1)

        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s).strip()

        return s

    def _try_parse_number(self, s: str) -> Optional[float]:
        """Try to convert a string to a float. Returns None if not a number."""
        try:
            cleaned = s.strip().rstrip('.')
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _try_parse_fraction(self, s: str) -> Optional[float]:
        """
        Try to parse a fraction to a float.
        Handles: \frac{a}{b}, \frac ab (single-char), a/b, plain numbers.
        """
        s = s.strip()

        # \frac{a}{b}
        frac_match = re.match(r'^\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$', s)
        if frac_match:
            try:
                num = float(frac_match.group(1))
                den = float(frac_match.group(2))
                if den != 0:
                    return num / den
            except (ValueError, ZeroDivisionError):
                pass

        # \frac followed by two single characters/digits: \frac12 = 1/2
        frac_short = re.match(r'^\\frac\s*(\d)(\d)$', s)
        if frac_short:
            try:
                return int(frac_short.group(1)) / int(frac_short.group(2))
            except ZeroDivisionError:
                pass

        # a/b
        simple_match = re.match(r'^(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$', s)
        if simple_match:
            try:
                return float(simple_match.group(1)) / float(simple_match.group(2))
            except (ValueError, ZeroDivisionError):
                pass

        # Plain number
        try:
            return float(s)
        except ValueError:
            return None

    def _strip_latex_for_comparison(self, s: str) -> str:
        """
        Aggressively strip all LaTeX formatting for a final string comparison.
        Keeps only the mathematical content.
        """
        result = s

        # Remove all remaining backslash commands (but keep their content)
        # e.g., \sqrt{3} -> sqrt{3}, \pi -> pi
        result = re.sub(r'\\([a-zA-Z]+)', r'\1', result)

        # Remove all braces
        result = result.replace("{", "").replace("}", "")

        # Remove all spaces
        result = result.replace(" ", "")

        # Lowercase
        result = result.lower()

        return result

    def _answers_equivalent(self, pred: str, truth: str) -> bool:
        """
        Check if two math answers are equivalent using multiple strategies.

        1. Exact string match
        2. Case-insensitive
        3. No-space comparison
        4. Numeric comparison (with tolerance)
        5. Fraction equivalence
        6. Stripped LaTeX comparison
        7. Reordered list/set comparison (e.g., "-2, 1" vs "1,-2")
        """
        if not pred or not truth:
            return False

        # 1. Exact match
        if pred == truth:
            return True

        # 2. Case-insensitive
        if pred.lower() == truth.lower():
            return True

        # 3. No-space comparison
        pred_ns = pred.replace(" ", "")
        truth_ns = truth.replace(" ", "")
        if pred_ns == truth_ns:
            return True

        # 4. Numeric comparison
        pred_num = self._try_parse_number(pred)
        truth_num = self._try_parse_number(truth)
        if pred_num is not None and truth_num is not None:
            if abs(pred_num - truth_num) < 1e-6:
                return True

        # 5. Fraction comparison
        pred_frac = self._try_parse_fraction(pred)
        truth_frac = self._try_parse_fraction(truth)
        if pred_frac is not None and truth_frac is not None:
            if abs(pred_frac - truth_frac) < 1e-6:
                return True

        # 6. Aggressively stripped LaTeX comparison
        pred_stripped = self._strip_latex_for_comparison(pred)
        truth_stripped = self._strip_latex_for_comparison(truth)
        if pred_stripped == truth_stripped:
            return True

        # 7. Reordered list comparison: "-2, 1" vs "1,-2" or "1, -2"
        pred_items = self._parse_as_list(pred)
        truth_items = self._parse_as_list(truth)
        if pred_items is not None and truth_items is not None:
            if len(pred_items) == len(truth_items) and len(pred_items) > 1:
                if sorted(pred_items) == sorted(truth_items):
                    return True

        # 8. Handle "x \in [a,b]" vs "[a,b]" - extract interval
        pred_interval = self._extract_interval(pred)
        truth_interval = self._extract_interval(truth)
        if pred_interval and truth_interval:
            if pred_interval == truth_interval:
                return True

        # 9. Commutative addition: "a + b" vs "b + a" (split on + and compare as sets)
        pred_terms = self._parse_additive_terms(pred)
        truth_terms = self._parse_additive_terms(truth)
        if pred_terms is not None and truth_terms is not None:
            if len(pred_terms) == len(truth_terms) and len(pred_terms) > 1:
                if sorted(pred_terms) == sorted(truth_terms):
                    return True

        return False

    def _parse_as_list(self, s: str) -> Optional[list]:
        """
        Try to parse a string as a comma-separated list of values.
        Returns sorted list of stripped items, or None if not a list.
        """
        s = s.strip()
        # Remove outer parentheses/brackets/braces if present
        if (s.startswith("(") and s.endswith(")")) or \
           (s.startswith("[") and s.endswith("]")) or \
           (s.startswith("{") and s.endswith("}")):
            s = s[1:-1]

        # Must contain a comma to be a list
        if "," not in s:
            return None

        items = [item.strip() for item in s.split(",") if item.strip()]
        if len(items) < 2:
            return None

        # Try to normalize each item as a number for better comparison
        normalized = []
        for item in items:
            num = self._try_parse_number(item)
            if num is not None:
                normalized.append(str(num))
            else:
                normalized.append(item.strip().lower())

        return normalized

    def _extract_interval(self, s: str) -> Optional[str]:
        """Extract interval notation, stripping 'x \\in' prefix if present."""
        s = s.strip()
        # Remove "x \in " or "x ∈ " prefix
        s = re.sub(r'^[a-z]\s*(?:\\in|∈)\s*', '', s)
        # Check if it looks like an interval
        if re.match(r'^[\[\(].+,\s*.+[\]\)]$', s):
            return s.replace(" ", "")
        return None

    def _parse_additive_terms(self, s: str) -> Optional[list]:
        """
        Parse an expression into additive terms for commutative comparison.
        E.g., "2\sqrt{3} + 1" -> ["1", "2sqrt{3}"]
        Only works for simple sums (no subtraction reordering).
        """
        s = s.strip()
        # Only apply to expressions with '+' and no '=' or other operators
        if '+' not in s or '=' in s:
            return None
        # Don't apply to complex expressions with parentheses or brackets
        if '(' in s or '[' in s:
            return None

        terms = [t.strip().replace(" ", "").lower() for t in s.split('+')]
        if len(terms) < 2:
            return None
        # Normalize LaTeX in each term
        terms = [re.sub(r'\\([a-zA-Z]+)', r'\1', t) for t in terms]
        return terms

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.

        Extracts \\boxed{} content from prediction, normalizes both sides,
        then compares using multiple strategies.
        """
        pred_answer = self._extract_boxed(predicted)
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
