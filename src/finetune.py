"""
Fine-tuning configuration for local Muse Glimmer training.

Validates fine-tuning data and provides configuration templates for
local training with TRL (Hugging Face) or Axolotl.
"""

import json
from pathlib import Path


class FineTuneValidator:
    """Validates and reports on fine-tuning data preparation."""

    def validate_jsonl(self, file_path: Path) -> dict:
        """
        Validate JSONL format and structure.

        Returns statistics about the dataset.
        """
        if not file_path.exists():
            return {
                'valid': False,
                'error': f'File not found: {file_path}'
            }

        examples = []
        errors = []

        try:
            with open(file_path) as f:
                for i, line in enumerate(f, 1):
                    try:
                        example = json.loads(line)
                        if 'messages' not in example:
                            errors.append(f"Line {i}: missing 'messages' field")
                        else:
                            examples.append(example)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i}: {e}")
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }

        # Calculate statistics
        total_tokens_approx = sum(
            len(json.dumps(ex)) // 4  # Rough token estimate
            for ex in examples
        )

        return {
            'valid': len(errors) == 0,
            'num_examples': len(examples),
            'errors': errors,
            'total_tokens_approx': total_tokens_approx,
            'avg_tokens_per_example': total_tokens_approx // len(examples) if examples else 0
        }

    def check_fine_tuning_data(
        self,
        train_file: Path,
        validation_file: Path | None = None
    ) -> dict:
        """
        Validate fine-tuning data files.

        Args:
            train_file: Path to training JSONL
            validation_file: Path to validation JSONL (optional)
        """
        print("\n=== Fine-Tuning Data Validation ===\n")

        # Validate training file
        print(f"Checking training file: {train_file}")
        train_result = self.validate_jsonl(train_file)

        if train_result['valid']:
            print("✓ Valid JSONL format")
            print(f"  Examples: {train_result['num_examples']}")
            print(f"  Approx tokens: {train_result['total_tokens_approx']:,}")
            print(f"  Avg tokens/example: {train_result['avg_tokens_per_example']}")
        else:
            print(f"✗ Invalid: {train_result.get('error', 'See errors below')}")
            for error in train_result.get('errors', []):
                print(f"    {error}")

        # Validate validation file if provided
        val_result = None
        if validation_file and validation_file.exists():
            print(f"\nChecking validation file: {validation_file}")
            val_result = self.validate_jsonl(validation_file)

            if val_result['valid']:
                print("✓ Valid JSONL format")
                print(f"  Examples: {val_result['num_examples']}")
                print(f"  Approx tokens: {val_result['total_tokens_approx']:,}")
                print(f"  Avg tokens/example: {val_result['avg_tokens_per_example']}")
            else:
                print(f"✗ Invalid: {val_result.get('error', 'See errors below')}")
                for error in val_result.get('errors', []):
                    print(f"    {error}")

        # Print fine-tuning guidance
        if train_result['valid']:
            print("\n=== Local Fine-Tuning Guide ===\n")
            print("Target Model: Muse Glimmer 30B (Apache 2.0)")
            print("  Hugging Face: meta-models/Muse-Glimmer-30B")
            print("  GGUF variant: meta-models/Muse-Glimmer-30B-GGUF\n")

            print("Approach 1: Supervised Fine-Tuning (SFT)")
            print("  Framework: TRL (Hugging Face) or Axolotl")
            print("  Method: QLoRA for parameter-efficient training")
            print("  Data: Use train.jsonl as-is\n")

            print("Approach 2: GRPO (Group Relative Policy Optimization)")
            print("  Framework: TRL GRPOTrainer or Meta's TitanRL")
            print("  Reward: Use src/scorer.py as deterministic reward function")
            print("  Data: Sample from train.jsonl for policy optimization\n")

            print("Installation (optional dependencies):")
            print("  pip install transformers trl peft accelerate bitsandbytes")
            print("  pip install axolotl  # Alternative to TRL\n")

            print("Example TRL SFT command:")
            print("  # See TRL documentation for full configuration")
            print("  # https://huggingface.co/docs/trl/")

        return {
            'train': train_result,
            'validation': val_result,
            'all_valid': train_result['valid'] and (
                val_result['valid'] if val_result else True
            )
        }


def main():
    """CLI for fine-tuning data validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate fine-tuning data for local Muse Glimmer training'
    )
    parser.add_argument(
        '--train-file',
        type=Path,
        default=Path('finetune_data/train.jsonl'),
        help='Training JSONL file'
    )
    parser.add_argument(
        '--validation-file',
        type=Path,
        default=Path('finetune_data/validation.jsonl'),
        help='Validation JSONL file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Validate and show guidance (default: True, always enabled)'
    )

    args = parser.parse_args()

    if not args.train_file.exists():
        print(f"Error: Training file not found: {args.train_file}")
        print("Run: python -m src.finetune_prep first")
        return 1

    validator = FineTuneValidator()
    result = validator.check_fine_tuning_data(
        train_file=args.train_file,
        validation_file=args.validation_file if args.validation_file.exists() else None
    )

    if result['all_valid']:
        print("\n✓ All data files are valid and ready for local fine-tuning")
        return 0
    else:
        print("\n✗ Validation failed - fix errors before fine-tuning")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
