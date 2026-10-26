"""
Runner for executing tasks against language models.

Supports OpenAI-compatible APIs with dry-run mode for testing.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


class StubModel:
    """Stub model for dry-run testing."""

    def chat_completions_create(self, messages: list[dict], **kwargs) -> dict:
        """Return a mock response."""
        return {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'problems_found': ['duplicate_row', 'bad_price'],
                        'mean_return': 0.001234,
                        'std_return': 0.023456,
                        'total_return': 0.123456,
                        'final_price': 105.67,
                        'annualized_volatility': 0.234567,
                        'num_clean_days': 60
                    })
                }
            }]
        }


class TaskRunner:
    """Runs tasks against a language model."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        dry_run: bool = False
    ):
        """
        Initialize runner.

        Args:
            base_url: API base URL (default from META_BASE_URL env var)
            api_key: API key (default from META_API_KEY env var)
            model_name: Model name (default from META_MODEL_NAME env var)
            dry_run: If True, use stub model instead of real API
        """
        load_dotenv()

        self.dry_run = dry_run
        self.base_url = base_url or os.getenv('META_BASE_URL', 'https://api.meta.ai/v1')
        self.api_key = api_key or os.getenv('META_API_KEY') or os.getenv('MODEL_API_KEY', '')
        self.model_name = model_name or os.getenv('META_MODEL_NAME', 'muse-spark-1.3')

        if dry_run:
            print("[DRY RUN] Using stub model")
            self.client = StubModel()
        else:
            if not self.api_key:
                raise ValueError("META_API_KEY environment variable not set")

            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install -r requirements.txt"
                ) from None

    def load_tasks(self, tasks_dir: Path) -> list[dict]:
        """Load task definitions."""
        tasks_file = Path(tasks_dir) / 'tasks.json'
        with open(tasks_file) as f:
            return json.load(f)

    def load_data(self, tasks_dir: Path, data_file: str) -> str:
        """Load CSV data for a task."""
        data_path = Path(tasks_dir) / data_file
        with open(data_path) as f:
            return f.read()

    def run_task(self, task: dict, data: str) -> dict:
        """Run a single task."""
        messages = [
            {
                'role': 'system',
                'content': 'You are a financial data analyst. Analyze the provided data and return results as valid JSON only.'
            },
            {
                'role': 'user',
                'content': f"{task['prompt']}\n\nData:\n{data}"
            }
        ]

        if self.dry_run:
            response = self.client.chat_completions_create(messages=messages)
        else:
            extra = {}
            if os.getenv('META_REASONING_EFFORT'):
                extra['reasoning_effort'] = os.getenv('META_REASONING_EFFORT')
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                # muse-spark-1.3 is a reasoning model: reasoning tokens count against
                # max_tokens, so 1000 left no room for the answer (empty content).
                max_tokens=int(os.getenv('META_MAX_TOKENS', '16000')),
                **extra
            )

        return {
            'task_id': task['id'],
            'task_type': task['type'],
            'response': response['choices'][0]['message']['content'] if self.dry_run
                       else (response.choices[0].message.content or '')
        }

    def run_all_tasks(
        self,
        tasks_dir: Path,
        output_dir: Path,
        limit: int | None = None
    ) -> list[dict]:
        """
        Run all tasks and save results.

        Args:
            tasks_dir: Directory containing tasks and data
            output_dir: Directory to save results
            limit: Optional limit on number of tasks to run
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks = self.load_tasks(tasks_dir)
        if limit:
            tasks = tasks[:limit]

        print(f"Running {len(tasks)} tasks...")
        if self.dry_run:
            print("[DRY RUN] No actual API calls will be made")
        else:
            print(f"Model: {self.model_name}")
            print(f"Base URL: {self.base_url}")

        results = []
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] Running {task['id']}...")

            data = self.load_data(tasks_dir, task['data_file'])
            result = self.run_task(task, data)
            results.append(result)

            # Rate limiting (not needed for stub, but good practice)
            if not self.dry_run and i < len(tasks):
                time.sleep(1)

        # Save results
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        return results


def main():
    """CLI for running tasks."""
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmark tasks against a model')
    parser.add_argument(
        '--tasks-dir',
        type=Path,
        default=Path('tasks'),
        help='Directory containing tasks (default: tasks/)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Directory for results (default: output/)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of tasks to run'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Use stub model instead of real API'
    )
    parser.add_argument(
        '--base-url',
        help='API base URL (overrides META_BASE_URL env var)'
    )
    parser.add_argument(
        '--model',
        help='Model name (overrides META_MODEL_NAME env var)'
    )

    args = parser.parse_args()

    if not args.tasks_dir.exists():
        print(f"Error: Tasks directory not found: {args.tasks_dir}")
        print("Run: python -m src.generate_tasks first")
        return 1

    runner = TaskRunner(
        base_url=args.base_url,
        model_name=args.model,
        dry_run=args.dry_run
    )

    results = runner.run_all_tasks(
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        limit=args.limit
    )

    print(f"\n✓ Completed {len(results)} tasks")
    print("\nNext step: Score the results")
    print(f"  python -m src.scorer {args.output_dir}/results.json")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
