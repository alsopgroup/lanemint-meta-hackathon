# Meta AI Developer Hackathon Benchmark

A self-contained benchmark and fine-tuning playground for evaluating language models on financial data quality tasks, built for Meta's AI Developer Hackathon (October 26 - November 4, 2026).

**Project by:** [LaneMint](https://lanemint.com)

## What It Is

An agent receives synthetic market data CSV files with intentional data quality problems (duplicate rows, missing days, bad prices, unadjusted stock splits). Each task includes a written prompt asking the model to identify issues and compute a metric (returns, volatility, backtest results). A deterministic scorer validates the responses.

The project runs a Meta Llama model before and after a light fine-tune on these tasks, using only the hackathon's $150 credit allocation.

## Key Features

- **~25 synthetic tasks** with known data quality issues
- **Deterministic scoring** with detailed per-problem feedback
- **Dry-run mode** for safe testing without API costs
- **Fine-tuning pipeline** with train/validation split
- **Cost estimation** before running paid operations
- **Zero external dependencies** on proprietary data or code

## Setup

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone and enter the repository
git clone https://github.com/alsopgroup/lanemint-meta-hackathon.git
cd lanemint-meta-hackathon

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy template)
cp .env.example .env
```

### Environment Configuration

Edit `.env` and set your Meta API credentials (do not commit this file):

```bash
META_API_KEY=your_api_key_here
META_BASE_URL=https://api.meta.ai/v1
META_MODEL_NAME=llama-3.1-8b
```

**Note**: `META_BASE_URL` is a placeholder. Confirm the actual API endpoint from the hackathon's official documentation before running any real API calls.

## Usage

### 1. Generate Tasks

Create synthetic market data tasks with quality issues:

```bash
python -m src.generate_tasks
```

This creates:
- `tasks/tasks.json` - Task definitions and prompts
- `tasks/task_*.csv` - Dirty market data CSV files
- `tasks/answers/` - Expected answers (hidden from model)

### 2. Run Tasks (Dry Run)

Test the full pipeline with a stub model (no API calls):

```bash
python -m src.runner --dry-run
```

Output saved to `output/results.json`.

### 3. Score Results

Evaluate model responses against expected answers:

```bash
python -m src.scorer output/results.json
```

Shows pass/fail for each task with detailed feedback.

### 4. Prepare Fine-Tuning Data

Split tasks into training and validation sets:

```bash
python -m src.finetune_prep
```

Creates:
- `finetune_data/train.jsonl` - Training examples (80%)
- `finetune_data/validation.jsonl` - Validation examples (20%)
- `finetune_data/validation_task_ids.json` - Held-out task IDs

### 5. Estimate Fine-Tuning Cost

Check cost before running (always use dry-run first):

```bash
python -m src.finetune --dry-run
```

### 6. Launch Fine-Tuning (Requires Approval)

⚠️ **This uses hackathon credits** - requires explicit approval:

```bash
python -m src.finetune --run
```

## Running Real Model Evaluation

Once you have Meta API access configured:

```bash
# Run baseline (before fine-tuning)
python -m src.runner --output-dir output/baseline

# Score baseline
python -m src.scorer output/baseline/results.json

# After fine-tuning completes, update META_MODEL_NAME in .env
# to your fine-tuned model, then:
python -m src.runner --output-dir output/finetuned

# Score fine-tuned results
python -m src.scorer output/finetuned/results.json
```

## Cost Estimate

Per full benchmark run with ~25 tasks:

- **Baseline run**: ~$0.50 (inference on 25 tasks)
- **Fine-tuning**: ~$2-5 (training tokens + validation)
- **Fine-tuned run**: ~$0.50 (inference on 25 tasks)

**Total estimated cost**: ~$3-6 per full cycle (well within $150 budget)

*Note: Actual costs depend on final model size, token counts, and Meta's pricing. Always run `--dry-run` first to see estimates.*

## Project Structure

```
.
├── src/
│   ├── generate_tasks.py    # Task and data generator
│   ├── scorer.py             # Deterministic scoring engine
│   ├── runner.py             # Model runner with dry-run mode
│   ├── finetune_prep.py      # Fine-tuning data preparation
│   └── finetune.py           # Fine-tuning job launcher
├── tests/
│   └── test_scorer.py        # Unit tests for scorer
├── tasks/                    # Generated tasks (gitignored)
├── output/                   # Model results (gitignored)
├── finetune_data/           # Training files (gitignored)
├── .env.example             # Environment template
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Data Quality Issues

The generator injects these types of problems:

1. **Duplicate rows** - Same date appears twice
2. **Missing days** - Gaps in trading day sequence
3. **Bad prices** - Obvious price errors (10x jump or 90% drop)
4. **Unadjusted stock splits** - Price discontinuities from splits

Each task contains 2-3 random issues. The model must identify and handle them appropriately.

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Tests cover scorer logic, JSON parsing, and tolerance checks.

## CI/CD

GitHub Actions runs tests automatically on every push:

```bash
# See .github/workflows/test.yml
```

No secrets or API keys required for CI.

## Development

### Adding New Task Types

Edit `src/generate_tasks.py` and add a method like:

```python
def generate_task_your_type(self) -> Dict:
    # Generate data with problems
    # Return task dict with prompt and expected answer
```

Register it in `generate_all_tasks()`.

### Adjusting Scoring

Edit scoring logic in `src/scorer.py`:

- Modify tolerance in `compare_float()`
- Adjust pass threshold (default: 80%)
- Add new field validators

## Hackathon Context

**Event**: Meta AI Developer Hackathon  
**Dates**: October 26 - November 4, 2026  
**Budget**: $150 in Meta API credits  
**Goal**: Demonstrate measurable improvement in data quality task performance through fine-tuning

---

Built by [LaneMint](https://lanemint.com) for Meta's AI Developer Hackathon 2026.
