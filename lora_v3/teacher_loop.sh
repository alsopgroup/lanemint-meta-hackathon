#!/bin/bash
# Resumable teacher loop: two lanes (4 workers each = 8 concurrent, below the 429 threshold seen at ~30).
cd /workspace/lmh-runs/lora-v3
PY=/workspace/lmh-runs/fw-serverless-sft-full/cookbook/training/.venv/bin/python
lane() { for pass in 1 2; do for s in "$@"; do
  [ -f runs/open-tools-on/BUDGET_STOP ] && return
  timeout 40m $PY run.py --arm open --seeds $s --workers 4 >> teacher_loop.log 2>&1
  echo "$(date +%T) seed $s pass $pass exit $?" >> teacher_loop.log
done; done; }
lane 5102 5104 5106 5108 5110 5101 &
lane 5103 5105 5107 5109 &
wait; echo LOOP_DONE >> teacher_loop.log
