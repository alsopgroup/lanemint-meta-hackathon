#!/bin/sh
# Offline sample benchmark run — no API keys and no network calls.
#
# Documentation:
#   1. Generate a fresh, non-reserved easy-tier set (seed 123 — the
#      reserved evaluation seeds 42, 1001 and 1002 are not used, and no
#      shipped task data, seeds or results are touched: everything is
#      written to the container-local tasks_sample/ and output/ dirs).
#   2. Run the benchmark runner with --dry-run, which uses a stub model
#      instead of the Meta/Fireworks APIs, for the first 3 tasks.
#   3. Score those placeholder answers with the deterministic scorer.
#
# A zero pass rate is expected here: the stub returns placeholder
# answers, so this proves the generate -> run -> score pipeline works
# end to end, not that a model is accurate. For a live run, set
# META_API_KEY in the runtime environment (see .env.example) and call
# `python -m src.runner` without --dry-run.
set -eu

cd /workspace

echo "[benchmark-sample] Generating easy-tier sample tasks (seed 123, count 3)..."
python -m src.generate_tasks --tier easy --seed 123 --count 3 --output-dir tasks_sample

echo "[benchmark-sample] Running runner in --dry-run mode (stub model, offline)..."
python -m src.runner --tasks-dir tasks_sample --dry-run --limit 3 --output-dir output/sample

echo "[benchmark-sample] Scoring dry-run results..."
python -m src.scorer output/sample/results.json tasks_sample/answers

echo "[benchmark-sample] Done. Results: output/sample/results.json, scores: output/sample/scoring_summary.json"
