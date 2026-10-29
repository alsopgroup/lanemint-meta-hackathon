"""
Fireworks SFT validation and job planning for Muse Glimmer 30B LoRA.

Validates JSONL datasets against the Fireworks supervised
fine-tuning rules (docs.fireworks.ai/fine-tuning/supervised-fine-tuning):
3 to 3,000,000 rows per dataset, ``messages`` with valid roles
(system first when present), a non-empty assistant message, and --
checked here so the platform never silently truncates -- every row
within ``--max-context-length`` tokens (default 32768). It then
prints, but never runs, the ``firectl`` commands that would upload
the datasets and start the LoRA job. Nothing in this module calls
Fireworks, the network, or a subprocess.
"""

import json
from pathlib import Path

from src.trace import count_tokens

BASE_MODEL = 'accounts/fireworks/models/muse-glimmer-30b'
VALID_ROLES = ('system', 'user', 'assistant')
MIN_ROWS, MAX_ROWS = 3, 3_000_000


class FireworksDataError(ValueError):
    """A dataset violates a Fireworks SFT rule."""


def _message_text(message: dict) -> str:
    content = message.get('content', '')
    if isinstance(content, list):
        content = "".join(
            str(part.get('text', '')) for part in content
            if isinstance(part, dict))
    return str(content) + str(message.get('reasoning_content', ''))


def row_tokens(example: dict) -> int:
    return sum(count_tokens(_message_text(m))
               for m in example.get('messages', []))


def validate_rows(rows: list, max_context_length: int,
                  source: str = 'dataset') -> list[str]:
    """Fireworks SFT rule violations for already-parsed rows."""
    errors = []
    if not MIN_ROWS <= len(rows) <= MAX_ROWS:
        errors.append(
            f"{source}: {len(rows)} rows, Fireworks SFT requires "
            f"{MIN_ROWS} to {MAX_ROWS} per dataset")
    for i, example in enumerate(rows, 1):
        where = f"{source} line {i}"
        if not isinstance(example, dict) \
                or not isinstance(example.get('messages'), list) \
                or not example['messages']:
            errors.append(f"{where}: missing or empty 'messages' array")
            continue
        roles = [m.get('role') for m in example['messages']
                 if isinstance(m, dict)]
        if len(roles) != len(example['messages']):
            errors.append(f"{where}: every message must be an object")
            continue
        bad = [r for r in roles if r not in VALID_ROLES]
        if bad:
            errors.append(f"{where}: invalid role(s) {bad}; valid "
                          f"roles are {list(VALID_ROLES)}")
        if 'system' in roles and roles[0] != 'system':
            errors.append(f"{where}: system message must be first")
        assistants = [m for m in example['messages']
                      if m.get('role') == 'assistant']
        if not assistants or not any(
                _message_text(m).strip() for m in assistants):
            errors.append(f"{where}: no non-empty assistant message")
        tokens = row_tokens(example)
        if tokens > max_context_length:
            errors.append(
                f"{where}: {tokens} tokens exceeds "
                f"--max-context-length {max_context_length}; "
                f"refusing rather than letting Fireworks truncate "
                f"the row")
    return errors


class FineTuneValidator:
    """Validates Fireworks SFT JSONL files and reports token stats."""

    def __init__(self, max_context_length: int = 32768):
        self.max_context_length = max_context_length

    def validate_jsonl(self, file_path: Path) -> dict:
        file_path = Path(file_path)
        if not file_path.exists():
            return {'valid': False, 'num_examples': 0,
                    'errors': [f'File not found: {file_path}'],
                    'max_tokens': 0, 'mean_tokens': 0.0}
        rows, errors = [], []
        with open(file_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    errors.append(f"{file_path} line {i}: invalid "
                                  f"JSON: {e}")
        errors.extend(validate_rows(rows, self.max_context_length,
                                    source=str(file_path)))
        tokens = [row_tokens(r) for r in rows
                  if isinstance(r, dict)]
        return {'valid': not errors, 'num_examples': len(rows),
                'errors': errors,
                'max_tokens': max(tokens) if tokens else 0,
                'mean_tokens': round(sum(tokens) / len(tokens), 1)
                if tokens else 0.0}

    def check_fine_tuning_data(self, train_file: Path,
                               validation_file: Path | None = None
                               ) -> dict:
        print("\n=== Fireworks SFT Data Validation ===\n")
        train_result = self.validate_jsonl(train_file)
        self._report('training', train_file, train_result)
        val_result = None
        if validation_file is not None:
            val_result = self.validate_jsonl(validation_file)
            self._report('validation', validation_file, val_result)
        return {'train': train_result, 'validation': val_result,
                'all_valid': train_result['valid']
                and (val_result['valid'] if val_result else True)}

    @staticmethod
    def _report(label, path, result):
        if result['valid']:
            print(f"✓ {label}: {path} "
                  f"({result['num_examples']} rows, max "
                  f"{result['max_tokens']} tokens, mean "
                  f"{result['mean_tokens']})")
        else:
            print(f"✗ {label}: {path}")
            for error in result['errors'][:20]:
                print(f"    {error}")


def default_names(target: str) -> tuple[str, str, str]:
    """Per-target Fireworks names, so answer and trace runs never
    collide: (train dataset, eval dataset, output model)."""
    return (f'glimmer-sft-{target}', f'glimmer-sft-{target}-eval',
            f'muse-glimmer-30b-lora-{target}')


def planned_commands(train_file: Path, validation_file: Path | None,
                     base_model: str = BASE_MODEL, lora_rank: int = 16,
                     epochs: int = 2, train_dataset: str = 'glimmer-sft',
                     eval_dataset: str = 'glimmer-sft-eval',
                     output_model: str = 'muse-glimmer-30b-lora',
                     max_context_length: int = 32768
                     ) -> list[str]:
    """The firectl commands for this dataset. Printed, never run."""
    commands = [f"firectl dataset create {train_dataset} {train_file}"]
    eval_flag = ''
    if validation_file is not None:
        commands.append(
            f"firectl dataset create {eval_dataset} {validation_file}")
        eval_flag = f" --evaluation-dataset {eval_dataset}"
    commands.append(
        f"firectl sftj create --base-model {base_model} "
        f"--dataset {train_dataset}{eval_flag} "
        f"--lora-rank {lora_rank} --epochs {epochs} "
        f"--max-context-length {max_context_length} "
        f"--output-model {output_model}")
    return commands


def main():
    """CLI: validate data and print the Fireworks plan (dry-run only)."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate Fireworks SFT data and print the '
                    'firectl plan (never executes anything)')
    parser.add_argument('--train-file', type=Path,
                        default=Path('finetune_data/train.jsonl'),
                        help='Training JSONL file')
    parser.add_argument('--validation-file', type=Path,
                        default=Path('finetune_data/validation.jsonl'),
                        help='Validation/evaluation JSONL file')
    parser.add_argument('--max-context-length', type=int, default=32768,
                        help='Per-row token limit; rows over it fail '
                             'validation (default: 32768)')
    parser.add_argument('--base-model', default=BASE_MODEL,
                        help=f'Base model (default: {BASE_MODEL})')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--target', choices=['answer', 'trace'],
                        default='answer',
                        help='Dataset target the files were built '
                             'with; selects the default dataset and '
                             'output model names (glimmer-sft-<target>'
                             ', muse-glimmer-30b-lora-<target>)')
    parser.add_argument('--train-dataset', default=None,
                        help='firectl dataset id for the training file '
                             '(default: per-target, see --target)')
    parser.add_argument('--eval-dataset', default=None,
                        help='firectl dataset id for the eval file '
                             '(default: per-target, see --target)')
    parser.add_argument('--output-model', default=None,
                        help='Output LoRA model id (default: '
                             'per-target, see --target)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Accepted for compatibility; this tool '
                             'only ever validates and prints commands')

    args = parser.parse_args()

    if not args.train_file.exists():
        print(f"Error: Training file not found: {args.train_file}")
        print("Run: python -m src.finetune_prep first")
        return 1

    manifest_path = args.train_file.parent / 'manifest.json'
    if manifest_path.exists():
        try:
            manifest_target = json.loads(
                manifest_path.read_text()).get('target')
        except (json.JSONDecodeError, OSError):
            manifest_target = None
        if manifest_target and manifest_target != args.target:
            print(f"Error: --target {args.target} does not match "
                  f"the dataset manifest target {manifest_target!r} "
                  f"in {manifest_path}; pass --target "
                  f"{manifest_target} so the printed Fireworks names "
                  f"match the data that was actually built")
            return 1

    validator = FineTuneValidator(
        max_context_length=args.max_context_length)
    result = validator.check_fine_tuning_data(
        train_file=args.train_file,
        validation_file=args.validation_file
        if args.validation_file.exists() else None)

    if not result['all_valid']:
        print("\n✗ Validation failed - fix errors before fine-tuning")
        return 1

    print("\n✓ All data files satisfy the Fireworks SFT rules")
    print("\nPlanned Fireworks commands (NOT executed; run them "
          "manually after review):")
    def_train, def_eval, def_model = default_names(args.target)
    for command in planned_commands(
            args.train_file,
            args.validation_file if args.validation_file.exists()
            else None,
            base_model=args.base_model, lora_rank=args.lora_rank,
            epochs=args.epochs,
            train_dataset=args.train_dataset or def_train,
            eval_dataset=args.eval_dataset or def_eval,
            output_model=args.output_model or def_model,
            max_context_length=args.max_context_length):
        print(f"  {command}")
    print("\nAfter training, evaluate on the locked seeds (42/1001/"
          "1002) at low reasoning effort, then tear down the "
          "deployment (see docs/BENCHMARK.md 'Fine-Tune on Fireworks').")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
