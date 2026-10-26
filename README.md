# Meta AI Developer Hackathon Benchmark

[![Tests](https://github.com/alsopgroup/lanemint-meta-hackathon/actions/workflows/test.yml/badge.svg)](https://github.com/alsopgroup/lanemint-meta-hackathon/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> A synthetic benchmark for evaluating language models on financial data quality tasks

![LaneMint x Meta: synthetic market-data benchmark built with Meta's Muse models](docs/assets/banner.png)

Built for Meta's AI Developer Hackathon (October 26 - November 4, 2026) by [LaneMint](https://lanemint.com).

---

## Overview

Test language models on their ability to detect and handle data quality issues in financial time series. Each task provides synthetic market data with intentional problems (duplicate rows, missing days, bad prices, unadjusted stock splits) and asks the model to identify issues and compute metrics. A deterministic scorer validates responses with detailed feedback.

### Workflow

```mermaid
graph LR
    A[Generate Tasks] --> B[Run Model]
    B --> C[Score Results]
    A --> D[Prepare Fine-tuning Data]
    D --> E[Fine-tune Model Locally]
    E --> B
    C --> F[Deterministic Reward]
    F --> E
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#fce4ec
    style F fill:#fff3e0
```

### Key Features

- 🎯 **~25 synthetic tasks** with known data quality issues
- ✅ **Deterministic scoring** with detailed per-problem feedback
- 🧪 **Dry-run mode** for safe testing without API costs
- 🔧 **Local fine-tuning** targeting open-weights Muse Glimmer (30B)
- 🎁 **Scorer as reward** for GRPO-based reinforcement learning
- 🔓 **Zero external dependencies** on proprietary data or code

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/alsopgroup/lanemint-meta-hackathon.git
cd lanemint-meta-hackathon
pip install -r requirements.txt

# Generate tasks
python -m src.generate_tasks

# Run dry-run (no API calls)
python -m src.runner --dry-run

# Score results
python -m src.scorer output/results.json
```

See [full usage guide](#usage) below.

---

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
META_MODEL_NAME=muse-spark-1.3
```

The API is OpenAI-compatible. Available models: `muse-spark-1.3` (default), `muse-spark-1.2`, `muse-spark-1.1`.

---

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

Split tasks into training and validation sets for local fine-tuning:

```bash
python -m src.finetune_prep
```

Creates:
- `finetune_data/train.jsonl` - SFT training examples (80%)
- `finetune_data/validation.jsonl` - Validation examples (20%)
- `finetune_data/validation_task_ids.json` - Held-out task IDs

The JSONL format is compatible with TRL (Hugging Face) and Axolotl for QLoRA fine-tuning.

### 5. Fine-Tune Locally (Planned)

Fine-tune **Muse Glimmer** (30B dense, Apache 2.0) locally using the prepared data:

**Target model**: `meta-models/Muse-Glimmer-30B` (Hugging Face) or `meta-models/Muse-Glimmer-30B-GGUF`

**Approaches**:
- **Supervised Fine-Tuning (SFT)**: Use `train.jsonl` with TRL's `SFTTrainer` or Axolotl for QLoRA
- **GRPO (Group Relative Policy Optimization)**: Use `src/scorer.py` as a deterministic reward function with TRL's `GRPOTrainer` or Meta's TitanRL data-analyst recipe

```bash
# Estimate data size and check format (dry-run, no training)
python -m src.finetune --dry-run

# For actual fine-tuning, see TRL or Axolotl documentation
# Optional: install fine-tuning dependencies
# pip install -e ".[finetune]"
```

**Note**: Fine-tuning requires GPU resources. The scorer provides deterministic rewards for reinforcement learning approaches.

### 6. Running Real Model Evaluation

Once you have Meta API access configured:

```bash
# Run baseline with Muse Spark
python -m src.runner --output-dir output/baseline

# Score baseline
python -m src.scorer output/baseline/results.json

# After local fine-tuning, point to your fine-tuned model endpoint
# Update META_BASE_URL and META_MODEL_NAME in .env, then:
python -m src.runner --output-dir output/finetuned

# Score fine-tuned results
python -m src.scorer output/finetuned/results.json
```

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── test.yml              # CI/CD configuration
├── src/
│   ├── __init__.py
│   ├── generate_tasks.py         # Task and data generator
│   ├── scorer.py                 # Deterministic scoring engine (also usable as reward function)
│   ├── runner.py                 # Model runner with dry-run mode
│   ├── finetune_prep.py          # Fine-tuning data preparation (SFT/GRPO)
│   └── finetune.py               # Fine-tuning configuration and validation
├── tests/
│   ├── __init__.py
│   └── test_scorer.py            # Unit tests for scorer
├── tasks/                        # Generated tasks (gitignored)
├── output/                       # Model results (gitignored)
├── finetune_data/                # Training files (gitignored)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Data Quality Issues

The generator injects these types of problems:

1. **Duplicate rows** - Same date appears twice
2. **Missing days** - Gaps in trading day sequence
3. **Bad prices** - Obvious price errors (10x jump or 90% drop)
4. **Unadjusted stock splits** - Price discontinuities from splits

Each task contains 2-3 random issues. The model must identify and handle them appropriately.

---

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Tests cover scorer logic, JSON parsing, and tolerance checks.

### CI/CD

GitHub Actions runs tests automatically on every push. The workflow includes:
- Unit test execution
- Full pipeline dry-run
- Scoring validation
- Fine-tuning data preparation

No secrets or API keys required for CI.

---

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

---

## Hackathon: Build for Your World

**Track**: Build for Your World  
**Event**: Meta AI Developer Hackathon  
**Dates**: October 26 - November 4, 2026

### Why Market Data Quality?

Market data is never clean in the real world. Duplicate rows, missing trading days, bad price ticks, and unadjusted stock splits corrupt backtests and portfolio analytics. Manual cleaning is brittle and expensive.

### Our Approach

1. **Synthetic benchmark**: Generate realistic market data with known quality issues
2. **Test Muse Spark**: Evaluate Meta's Muse models on data quality detection
3. **Deterministic scoring**: Validate responses with precise, reproducible grading
4. **Local fine-tuning**: Use the same scorer as a reward function for fine-tuning Muse Glimmer (30B, open-weights) with GRPO

The deterministic scorer enables both evaluation and reinforcement learning, closing the loop from detection to improvement.

### Optional Tooling

Meta's **Muse Code** (coding agent) can be used as optional tooling for development workflows, but is not required for the benchmark itself.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

Built by [LaneMint](https://lanemint.com) for Meta's AI Developer Hackathon 2026.
