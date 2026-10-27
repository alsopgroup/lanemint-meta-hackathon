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
                'finish_reason': 'stop',
                'message': {
                    'content': json.dumps({
                        'problems': ['duplicate', 'bad_price'],
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
        dry_run: bool = False,
        max_tokens: int | None = None
    ):
        """
        Initialize runner.

        Args:
            base_url: API base URL (default from META_BASE_URL env var)
            api_key: API key (default from META_API_KEY env var)
            model_name: Model name (default from META_MODEL_NAME env var)
            dry_run: If True, use stub model instead of real API
            max_tokens: Output token cap override. When unset, the
                META_MAX_TOKENS env var wins; otherwise hard- and
                frontier-tier tasks (task_h_* / task_f_* ids) default to
                32000 and easy-tier tasks to 16000, because reasoning
                tokens count against the cap.
        """
        load_dotenv()

        self.max_tokens = max_tokens
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
                # No SDK retries: a timed-out long request must never be
                # silently re-sent (and billed) twice. The timeout is
                # generous enough for 32k-token reasoning responses.
                self.client = OpenAI(base_url=self.base_url,
                                     api_key=self.api_key,
                                     max_retries=0, timeout=300.0)
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

    def _max_tokens_for(self, task: dict) -> int:
        """Token cap for one task: explicit arg, then env, then tier default."""
        if self.max_tokens is not None:
            return self.max_tokens
        env = os.getenv('META_MAX_TOKENS')
        if env:
            return int(env)
        return 32000 if str(task.get('id', '')).startswith(('task_h_', 'task_f_')) \
            else 16000

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
                # muse-spark-1.3 is a reasoning model: reasoning tokens count
                # against max_tokens, so a low cap leaves no room for the
                # answer (empty content). Hard-tier tasks default to 32000.
                max_tokens=self._max_tokens_for(task),
                **extra
            )

        if self.dry_run:
            choice = response['choices'][0]
            content = choice['message']['content']
            finish_reason = choice.get('finish_reason')
            usage = response.get('usage') or {}
            prompt_tokens = usage.get('prompt_tokens')
            completion_tokens = usage.get('completion_tokens')
            total_tokens = usage.get('total_tokens')
        else:
            choice = response.choices[0]
            content = choice.message.content or ''
            finish_reason = getattr(choice, 'finish_reason', None)
            usage = getattr(response, 'usage', None)
            prompt_tokens = getattr(usage, 'prompt_tokens', None)
            completion_tokens = getattr(usage, 'completion_tokens', None)
            total_tokens = getattr(usage, 'total_tokens', None)

        return {
            'task_id': task['id'],
            'task_type': task['type'],
            'response': content,
            'finish_reason': finish_reason,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
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
        default=None,
        help='Directory containing tasks (default: tasks/, or the --tier default)'
    )
    parser.add_argument(
        '--tier',
        choices=['easy', 'hard', 'frontier'],
        default=None,
        help='Task tier; sets the default tasks directory (tasks/, tasks_hard/, '
             'tasks_frontier/) when --tasks-dir is not given'
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
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='Output token cap (overrides META_MAX_TOKENS; default 32000 for '
             'hard- and frontier-tier tasks, 16000 for easy-tier tasks)'
    )

    args = parser.parse_args()

    if args.tasks_dir is None:
        args.tasks_dir = {'hard': Path('tasks_hard'),
                          'frontier': Path('tasks_frontier')}.get(
                              args.tier, Path('tasks'))

    if not args.tasks_dir.exists():
        print(f"Error: Tasks directory not found: {args.tasks_dir}")
        print("Run: python -m src.generate_tasks first")
        return 1

    runner = TaskRunner(
        base_url=args.base_url,
        model_name=args.model,
        dry_run=args.dry_run,
        max_tokens=args.max_tokens
    )

    results = runner.run_all_tasks(
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        limit=args.limit
    )

    print(f"\n✓ Completed {len(results)} tasks")
    print("\nNext step: Score the results")
    print(f"  python -m src.scorer {args.output_dir}/results.json {args.tasks_dir}/answers")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
