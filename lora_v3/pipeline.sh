#!/bin/bash
# Detached post-teacher pipeline: wait teacher -> build SFT -> train V3 ($0 guard) -> V3 sanity check on 5999.
cd /workspace/lmh-runs/lora-v3; PY=/workspace/lmh-runs/fw-serverless-sft-full/cookbook/training/.venv/bin/python
L=pipeline.log; say(){ echo "$(date +%T) $*" >> $L; }
until grep -q LOOP_DONE teacher_loop.log || [ -f runs/open-tools-on/BUDGET_STOP ]; do sleep 60; done
pkill -f teacher_loop.sh; pkill -f "run.py --arm open"; say teacher finished
if [ ! -f sft/.built ]; then $PY build_sft.py >> $L 2>&1 && touch sft/.built; fi
[ -f billing/train_baseline.json ] || $PY billing.py billing/train_baseline.json >> $L 2>&1
for i in 1 2 3; do grep -q '"run_done"' train_run/events.jsonl 2>/dev/null && break; [ -f train_run/STOP ] && { say TRAIN STOP; exit 1; }
  timeout 4h $PY train_v3.py >> train_v3.log 2>&1; say train attempt $i exit $?; done
grep -q '"run_done"' train_run/events.jsonl || { say training not done; exit 1; }
export LMH_V3_SAMPLER_PATH=$(grep '"sampler_saved"' train_run/events.jsonl | tail -1 | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['path'])")
say v3 sampler $LMH_V3_SAMPLER_PATH
while pgrep -f "glimmer_loop.sh base" >/dev/null; do sleep 60; done
./glimmer_loop.sh v3; say PIPELINE_DONE
