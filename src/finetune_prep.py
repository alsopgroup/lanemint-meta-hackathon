"""
Prepare fine-tuning data from benchmark tasks.

Converts tasks into training JSONL format with train/validation split.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


class FineTunePrep:
    """Prepares fine-tuning data from tasks."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible splits."""
        random.seed(seed)
    
    def load_tasks_with_answers(self, tasks_dir: Path) -> List[Dict]:
        """Load tasks with their expected answers."""
        tasks_dir = Path(tasks_dir)
        
        # Load task definitions
        with open(tasks_dir / 'tasks.json', 'r') as f:
            tasks = json.load(f)
        
        # Load answers
        answers_dir = tasks_dir / 'answers'
        for task in tasks:
            answer_file = answers_dir / f"{task['id']}_answer.json"
            with open(answer_file, 'r') as f:
                task['expected'] = json.load(f)
            
            # Load data
            data_file = tasks_dir / task['data_file']
            with open(data_file, 'r') as f:
                task['data'] = f.read()
        
        return tasks
    
    def format_for_training(self, task: Dict) -> Dict:
        """Format a single task as a training example."""
        messages = [
            {
                'role': 'system',
                'content': 'You are a financial data analyst. Analyze the provided data and return results as valid JSON only.'
            },
            {
                'role': 'user',
                'content': f"{task['prompt']}\n\nData:\n{task['data']}"
            },
            {
                'role': 'assistant',
                'content': json.dumps(task['expected'])
            }
        ]
        
        return {'messages': messages}
    
    def split_and_prepare(
        self,
        tasks_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.8
    ):
        """
        Split tasks into train/validation and prepare JSONL files.
        
        Args:
            tasks_dir: Directory containing tasks
            output_dir: Directory to save training files
            train_ratio: Fraction of tasks for training (default 0.8)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all tasks
        tasks = self.load_tasks_with_answers(tasks_dir)
        print(f"Loaded {len(tasks)} tasks")
        
        # Shuffle and split
        random.shuffle(tasks)
        split_idx = int(len(tasks) * train_ratio)
        train_tasks = tasks[:split_idx]
        val_tasks = tasks[split_idx:]
        
        print(f"Split: {len(train_tasks)} train, {len(val_tasks)} validation")
        
        # Write training JSONL
        train_file = output_dir / 'train.jsonl'
        with open(train_file, 'w') as f:
            for task in train_tasks:
                example = self.format_for_training(task)
                f.write(json.dumps(example) + '\n')
        print(f"Training data: {train_file}")
        
        # Write validation JSONL
        val_file = output_dir / 'validation.jsonl'
        with open(val_file, 'w') as f:
            for task in val_tasks:
                example = self.format_for_training(task)
                f.write(json.dumps(example) + '\n')
        print(f"Validation data: {val_file}")
        
        # Save validation task IDs for later evaluation
        val_ids_file = output_dir / 'validation_task_ids.json'
        with open(val_ids_file, 'w') as f:
            json.dump([t['id'] for t in val_tasks], f, indent=2)
        print(f"Validation task IDs: {val_ids_file}")
        
        return {
            'train_file': str(train_file),
            'validation_file': str(val_file),
            'num_train': len(train_tasks),
            'num_validation': len(val_tasks)
        }


def main():
    """CLI for preparing fine-tuning data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare fine-tuning data')
    parser.add_argument(
        '--tasks-dir',
        type=Path,
        default=Path('tasks'),
        help='Directory containing tasks (default: tasks/)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('finetune_data'),
        help='Directory for training files (default: finetune_data/)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Fraction of tasks for training (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    if not args.tasks_dir.exists():
        print(f"Error: Tasks directory not found: {args.tasks_dir}")
        print("Run: python -m src.generate_tasks first")
        return 1
    
    prep = FineTunePrep()
    result = prep.split_and_prepare(
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio
    )
    
    print("\n✓ Fine-tuning data prepared")
    print(f"  {result['num_train']} training examples")
    print(f"  {result['num_validation']} validation examples")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
