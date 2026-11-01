#!/bin/bash
# usage: glimmer_loop.sh base|v3  — resumable 30-task sanity check on seed 5999 (serverless sampler, $0)
cd /workspace/lmh-runs/lora-v3; PY=/workspace/lmh-runs/fw-serverless-sft-full/cookbook/training/.venv/bin/python
for i in 1 2 3 4; do
  [ -f STOP_GLIMMER ] && exit 1
  timeout 90m $PY run.py --arm glimmer --glimmer-model $1 --seeds 5999 --workers 4 >> glimmer_$1.log 2>&1
  echo "$(date +%T) round $i exit $?" >> glimmer_$1.log
  n=$(python3 -c "import json;print(len({json.loads(l)['task_id'] for l in open('runs/glimmer-$1-tools-on/seed5999/results.jsonl') if not json.loads(l).get('error')}))" 2>/dev/null); [ "$n" = "30" ] && break
done; echo GLOOP_DONE >> glimmer_$1.log
