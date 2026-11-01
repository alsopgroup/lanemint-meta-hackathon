"""Score glimmer base vs v3 on sanity seed 5999 (local scorer, no API calls)."""
import json, sys, collections
sys.path.insert(0, "/workspace/lmh-pr25")
from src.scorer import TaskScorer
R = "/workspace/lmh-runs/lora-v3"; TD = f"{R}/tasks/seed5999"; S = TaskScorer(f"{TD}/answers")
MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
out = {}
for arm in ["base", "v3"]:
    latest = {}
    for l in open(f"{R}/runs/glimmer-{arm}-tools-on/seed5999/results.jsonl"):
        x = json.loads(l)
        if not x.get("error") or x["task_id"] not in latest: latest[x["task_id"]] = x
    c = collections.Counter(); bt = collections.defaultdict(collections.Counter)
    for tid, x in latest.items():
        sc = S.score_task(tid, x.get("response") or ""); p = bool(sc["passed"]); t = x["task_type"]
        c["n"] += 1; c["pass"] += p; c["err"] += bool(x.get("error"))
        c["turn_limit"] += bool(x.get("forced_final")) or (x.get("turns") or 0) >= MAXT
        c["empty_final"] += not (x.get("response") or "").strip()
        c["turns"] += x.get("turns") or 0
        bt[t]["n"] += 1; bt[t]["pass"] += p
    out[arm] = {"total": dict(c), "by_type": {k: dict(v) for k, v in sorted(bt.items())}}
print(json.dumps(out, indent=1))
