import os
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
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


class DataProcessor:
    """
    Processor for MemoryAgentBench Test-Time Learning tasks.
    
    Currently handles the recommendation system cluster:
    - recsys_redial_full: Movie recommendation from dialogue context
    
    Each sample has a large context (movie recommendation dialogues) and a question
    to answer. Answers are movie IDs (numeric strings).
    """
    
    def __init__(self, task_name: str):
        """
        Initialize the data processor.
        
        Args:
            task_name: The name of the task (e.g., 'test_time_learning')
        """
        self.task_name = task_name
    
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw MemoryAgentBench data into standardized format.
        
        Input format (from JSONL):
            {
                "context": "<large in-context examples>",
                "question": "<the actual question>",
                "answers": ["<answer_id>", ...],
                "source": "<cluster_name>",
                "qa_pair_id": "<unique_id>"
            }
        
        Output format (standardized for ACE):
            {
                "context": "<in-context examples>",
                "question": "<the question>",
                "target": "<answer_id>",   # first valid answer as target
                "others": {
                    "all_valid_answers": [...],
                    "source": "<cluster>",
                    "qa_pair_id": "<id>"
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
            answers = item.get('answers', [])
            source = item.get('source', '')
            qa_pair_id = item.get('qa_pair_id', '')
            
            # Use the first answer as the primary target
            target = answers[0] if answers else ''
            
            processed_item = {
                "context": context,
                "question": question,
                "target": target,
                "others": {
                    "all_valid_answers": answers,
                    "source": source,
                    "qa_pair_id": qa_pair_id,
                    "task": self.task_name
                }
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        
        For MemoryAgentBench, answers are numeric label IDs (as strings).
        A prediction is correct if it matches any of the valid answers.
        Since ground_truth here is the primary target (first answer),
        we do a normalized string comparison.
        
        Args:
            predicted: Model's answer
            ground_truth: Ground truth answer (primary target)
            
        Returns:
            bool: True if answer is correct
        """
        predicted = predicted.strip()
        ground_truth = ground_truth.strip()
        
        # Direct string match
        if predicted == ground_truth:
            return True
        
        # Try numeric comparison (e.g., "42" == "42.0")
        try:
            if int(float(predicted)) == int(float(ground_truth)):
                return True
        except (ValueError, TypeError):
            pass
        
        # Check if predicted answer appears in ground_truth 
        # (handles cases where model outputs extra text around the ID)
        if ground_truth in predicted.split():
            return True
        
        return False
    
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
        
        for predicted, ground_truth in zip(out, target):
            if self.answer_is_correct(predicted, ground_truth):
                correct_count += 1
        
        accuracy = correct_count / len(out) if out else 0.0
        return accuracy
