#!/bin/bash
# Resumable V3 held-out eval (free Glimmer serverless sampler) + billing guard (stop if total rises > $0.50)
cd /workspace/lmh-runs/tools-frontier-heldout; P=/workspace/lmh-runs/fw-serverless-sft-full/cookbook/training/.venv/bin/python
L=v3_heldout.log; say(){ echo "$(date +%T) $*" >> $L; }
B0=$(python3 -c "import json;print(sum(v['total'] for v in json.load(open('billing/v3_before.json')).values()))")
guard(){ while true; do sleep 600; $P /workspace/lmh-runs/lora-v3/billing.py billing/v3_guard.json >/dev/null 2>&1 || { say guard snapshot failed; continue; }
  d=$(python3 -c "import json;print(round(sum(v['total'] for v in json.load(open('billing/v3_guard.json')).values())-$B0,4))")
  n=$(cat runs/glimmer-v3-tools-on/seed*/results.jsonl 2>/dev/null | wc -l); say "guard delta=\$$d lines=$n"
  if python3 -c "import sys;sys.exit(0 if $d>0.5 else 1)"; then say "BILLING GUARD TRIPPED delta=$d"; touch STOP_V3; pkill -f "run.py --arm glimmer --glimmer-model v3"; exit; fi; done; }
guard & GP=$!
for i in 1 2 3 4; do [ -f STOP_V3 ] && break
  timeout 3h $P run.py --arm glimmer --glimmer-model v3 --seeds 42,1001,1002 --workers 4 --max-turns 12 >> glimmer_v3.log 2>&1; say "round $i exit $?"
  n=$(python3 -c "
import json,glob;print(sum(len({json.loads(l)['task_id'] for l in open(f) if not json.loads(l).get('error')}) for f in glob.glob('runs/glimmer-v3-tools-on/seed*/results.jsonl')))"); say "ok tasks $n"; [ "$n" = "90" ] && break
done; kill $GP; say V3_HELDOUT_DONE
