#!/bin/sh
# Benchmark container entrypoint.
#
# Documentation:
#   - With no arguments, run the offline sample pipeline (benchmark-sample):
#     generate a fresh easy-tier set, run the runner in --dry-run mode
#     (stub model, no API calls, no keys needed) and score the result.
#   - With arguments, execute them verbatim from /workspace, e.g.
#       python -m src.runner --tasks-dir tasks_sample --dry-run --limit 3
#       python -m src.scorer output/sample/results.json tasks_sample/answers
#   Secrets are never printed here; they are only read by the benchmark
#   code itself from the runtime environment.
set -eu

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec benchmark-sample
