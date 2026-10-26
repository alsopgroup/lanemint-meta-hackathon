"""
Fine-tuning script for Meta Llama models.

This is a stub/template - actual fine-tuning requires approval
and will be run separately with hackathon credits.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


class FineTuner:
    """Handles fine-tuning jobs."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize fine-tuner."""
        load_dotenv()
        self.api_key = api_key or os.getenv('META_API_KEY', '')
        self.base_url = base_url or os.getenv('META_BASE_URL', 'https://api.meta.ai/v1')

    def estimate_cost(self, train_file: Path, validation_file: Path | None = None) -> dict:
        """
        Estimate fine-tuning cost.

        Based on Meta's pricing (approximate):
        - Training: ~$0.0001 per token
        - Validation: ~$0.00005 per token
        """
        def count_tokens_approx(file_path: Path) -> int:
            """Rough token count estimate (1 token ≈ 4 chars)."""
            with open(file_path) as f:
                content = f.read()
                return len(content) // 4

        train_tokens = count_tokens_approx(train_file)
        val_tokens = count_tokens_approx(validation_file) if validation_file else 0

        # Estimate costs (adjust based on actual Meta pricing)
        train_cost = train_tokens * 0.0001 / 1000  # per 1k tokens
        val_cost = val_tokens * 0.00005 / 1000
        total_cost = train_cost + val_cost

        return {
            'train_tokens': train_tokens,
            'validation_tokens': val_tokens,
            'estimated_train_cost_usd': round(train_cost, 4),
            'estimated_validation_cost_usd': round(val_cost, 4),
            'estimated_total_cost_usd': round(total_cost, 4),
            'note': 'This is a rough estimate. Actual costs depend on model size and final token counts.'
        }

    def create_finetune_job(
        self,
        train_file: Path,
        validation_file: Path | None,
        model: str = 'llama-3.1-8b',
        suffix: str = 'market-data',
        dry_run: bool = True
    ) -> dict:
        """
        Create a fine-tuning job.

        Args:
            train_file: Path to training JSONL
            validation_file: Path to validation JSONL (optional)
            model: Base model to fine-tune
            suffix: Custom suffix for fine-tuned model name
            dry_run: If True, only show what would be done
        """
        if dry_run:
            print("\n[DRY RUN] Would create fine-tuning job with:")
            print(f"  Base model: {model}")
            print(f"  Training file: {train_file}")
            if validation_file:
                print(f"  Validation file: {validation_file}")
            print(f"  Model suffix: {suffix}")
            print(f"  API base URL: {self.base_url}")

            estimate = self.estimate_cost(train_file, validation_file)
            print("\nCost estimate:")
            print(f"  Training tokens: ~{estimate['train_tokens']:,}")
            if estimate['validation_tokens'] > 0:
                print(f"  Validation tokens: ~{estimate['validation_tokens']:,}")
            print(f"  Estimated total cost: ${estimate['estimated_total_cost_usd']:.4f}")
            print("\n  Hackathon budget: $150.00")
            print(f"  Remaining after this run: ${150 - estimate['estimated_total_cost_usd']:.4f}")

            return {
                'status': 'dry_run',
                'estimate': estimate,
                'message': 'Set dry_run=False and ensure META_API_KEY is set to run actual fine-tuning'
            }

        if not self.api_key:
            raise ValueError(
                "META_API_KEY not set. Cannot create fine-tuning job.\n"
                "Set the key in .env file or META_API_KEY environment variable."
            )

        print("\n⚠️  CREATING REAL FINE-TUNING JOB")
        print("This will use hackathon credits.")

        try:
            from openai import OpenAI
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)

            # Upload training file
            print(f"\nUploading training file: {train_file}")
            with open(train_file, 'rb') as f:
                train_file_obj = client.files.create(file=f, purpose='fine-tune')

            # Upload validation file if provided
            val_file_id = None
            if validation_file:
                print(f"Uploading validation file: {validation_file}")
                with open(validation_file, 'rb') as f:
                    val_file_obj = client.files.create(file=f, purpose='fine-tune')
                val_file_id = val_file_obj.id

            # Create fine-tuning job
            print("\nCreating fine-tuning job...")
            job = client.fine_tuning.jobs.create(
                training_file=train_file_obj.id,
                validation_file=val_file_id,
                model=model,
                suffix=suffix
            )

            result = {
                'status': 'created',
                'job_id': job.id,
                'model': model,
                'fine_tuned_model': None,  # Will be set when job completes
                'status_url': f"{self.base_url}/fine_tuning/jobs/{job.id}",
                'message': f'Fine-tuning job created: {job.id}'
            }

            print(f"\n✓ Fine-tuning job created: {job.id}")
            print(f"Monitor status: {result['status_url']}")

            return result

        except ImportError:
            raise ImportError(
                "openai package required. Run: pip install -r requirements.txt"
            ) from None
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'message': f'Failed to create fine-tuning job: {e}'
            }


def main():
    """CLI for fine-tuning."""
    import argparse

    parser = argparse.ArgumentParser(description='Launch fine-tuning job')
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
        '--model',
        default='llama-3.1-8b',
        help='Base model to fine-tune (default: llama-3.1-8b)'
    )
    parser.add_argument(
        '--suffix',
        default='market-data',
        help='Suffix for fine-tuned model name (default: market-data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Show estimate without creating job (default: True)'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Actually create fine-tuning job (overrides --dry-run)'
    )

    args = parser.parse_args()

    if not args.train_file.exists():
        print(f"Error: Training file not found: {args.train_file}")
        print("Run: python -m src.finetune_prep first")
        return 1

    dry_run = not args.run

    tuner = FineTuner()
    result = tuner.create_finetune_job(
        train_file=args.train_file,
        validation_file=args.validation_file if args.validation_file.exists() else None,
        model=args.model,
        suffix=args.suffix,
        dry_run=dry_run
    )

    if result['status'] == 'dry_run':
        print("\n" + "="*60)
        print("DRY RUN COMPLETE")
        print("="*60)
        print("\nTo run actual fine-tuning:")
        print("  1. Set META_API_KEY in .env file")
        print("  2. Run with --run flag:")
        print("     python -m src.finetune --run")
        print("\n⚠️  This will use hackathon credits!")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
