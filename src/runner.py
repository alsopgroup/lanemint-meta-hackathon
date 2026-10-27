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
                # silently re-sent (and billed) twice. The timeout sits
                # well under provider context limits (Spark 1.3: ~1M
                # context, 131k max output; our cap stays 32k) while
                # allowing long reasoning responses to finish.
                self.client = OpenAI(base_url=self.base_url,
                                     api_key=self.api_key,
                                     max_retries=0, timeout=900.0)
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
            'model': self.model_name,
            'max_tokens': self._max_tokens_for(task),
            'finish_reason': finish_reason,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        }

    @staticmethod
    def _repair_partial_line(path: Path) -> None:
        """Truncate a trailing partial JSONL record left by a crash."""
        if not path.exists():
            return
        data = path.read_bytes()
        if not data or data.endswith(b'\n'):
            return
        cut = data.rfind(b'\n')
        with open(path, 'r+b') as f:
            f.truncate(cut + 1)

    @staticmethod
    def _error_status(exc: Exception) -> int | None:
        status = getattr(exc, 'status_code', None) or getattr(exc, 'status', None)
        return status if isinstance(status, int) and not isinstance(status, bool) \
            else None

    def _error_result(self, task: dict, exc: Exception) -> dict:
        # Record only the exception type and HTTP status: never the
        # exception text/body, which may echo request details.
        return {'task_id': task['id'], 'task_type': task.get('type'),
                'response': '', 'finish_reason': None,
                'model': self.model_name,
                'max_tokens': self._max_tokens_for(task),
                'prompt_tokens': None, 'completion_tokens': None,
                'total_tokens': None,
                'error': {'type': type(exc).__name__,
                          'status': self._error_status(exc)}}

    def run_all_tasks(
        self,
        tasks_dir: Path,
        output_dir: Path,
        limit: int | None = None,
        resume: bool = False
    ) -> list[dict]:
        """
        Run all tasks and save results.

        Results are appended to ``results.jsonl`` (flushed and fsynced
        after every task) and collected into ``results.json`` at the end,
        so one task's API failure cannot lose earlier paid responses: a
        failing task is recorded with only its error type and HTTP
        status. Every entry records the model and token cap used. With
        ``resume=True``, tasks already present without an error are
        skipped (refused if they were made with a different model) and
        error entries are retried; without it, any previous
        ``results.jsonl``/``results.json`` in the output dir is removed
        first so a fresh run never appends to a stale one.

        Args:
            tasks_dir: Directory containing tasks and data
            output_dir: Directory to save results
            limit: Optional limit on number of tasks to run
            resume: Skip tasks already successfully present in the output
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks = self.load_tasks(tasks_dir)
        if limit:
            tasks = tasks[:limit]
        task_ids = {t['id'] for t in tasks}

        jsonl_path = output_dir / 'results.jsonl'
        results_file = output_dir / 'results.json'
        done = {}
        if resume:
            # A crash can leave a partial last line: truncate back to
            # the last newline before loading or appending, so a new
            # record is never glued onto the fragment (which would make
            # that task unparseable and bill it again on the next resume).
            self._repair_partial_line(jsonl_path)
            if jsonl_path.exists():
                for line in jsonl_path.read_text().splitlines():
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # partial line from a crash mid-write
                    if entry.get('task_id') in task_ids:
                        done[entry['task_id']] = entry
            elif results_file.exists():
                for entry in json.loads(results_file.read_text()):
                    if entry.get('task_id') in task_ids:
                        done[entry['task_id']] = entry
            prior_models = {e.get('model') for e in done.values()
                            if e.get('model')}
            if prior_models and prior_models != {self.model_name}:
                raise ValueError(
                    f"--resume refused: {output_dir} contains results from "
                    f"model(s) {sorted(prior_models)}, not "
                    f"{self.model_name!r}; use a different --output-dir "
                    "or omit --resume to start fresh")
        else:
            # Fresh run: never append to a previous run's output.
            jsonl_path.unlink(missing_ok=True)
            results_file.unlink(missing_ok=True)

        print(f"Running {len(tasks)} tasks...")
        if self.dry_run:
            print("[DRY RUN] No actual API calls will be made")
        else:
            print(f"Model: {self.model_name}")
            print(f"Base URL: {self.base_url}")

        results_by_id = {}
        with open(jsonl_path, 'a') as jf:
            if resume and done and not jsonl_path.stat().st_size:
                for task in tasks:  # seed jsonl from an earlier results.json
                    if task['id'] in done:
                        jf.write(json.dumps(done[task['id']]) + '\n')
                jf.flush()
                os.fsync(jf.fileno())
            for i, task in enumerate(tasks, 1):
                existing = done.get(task['id'])
                if existing is not None and 'error' not in existing:
                    print(f"[{i}/{len(tasks)}] Skipping {task['id']} "
                          "(already in output)")
                    results_by_id[task['id']] = existing
                    continue
                print(f"[{i}/{len(tasks)}] Running {task['id']}...")
                try:
                    data = self.load_data(tasks_dir, task['data_file'])
                    result = self.run_task(task, data)
                except Exception as exc:  # one failure must not lose the run
                    result = self._error_result(task, exc)
                    print(f"[{i}/{len(tasks)}] {task['id']} failed: "
                          f"{result['error']['type']} "
                          f"(status {result['error']['status']})")
                results_by_id[task['id']] = result
                jf.write(json.dumps(result) + '\n')
                jf.flush()
                os.fsync(jf.fileno())

                # Rate limiting (not needed for stub, but good practice)
                if not self.dry_run and i < len(tasks):
                    time.sleep(1)

        results = [results_by_id[t['id']] for t in tasks
                   if t['id'] in results_by_id]
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
        '--resume',
        action='store_true',
        help='Skip tasks already successfully present in the output '
             'results.jsonl/results.json (error entries are retried); '
             'results are appended incrementally to results.jsonl'
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
        limit=args.limit,
        resume=args.resume
    )

    print(f"\n✓ Completed {len(results)} tasks")
    print("\nNext step: Score the results")
    print(f"  python -m src.scorer {args.output_dir}/results.json {args.tasks_dir}/answers")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
