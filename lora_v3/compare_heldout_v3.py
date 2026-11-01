"""Counts-only comparison of arms on held-out seeds 42/1001/1002 (strict local scorer, no API calls)."""
import json, sys, collections
sys.path.insert(0, "/workspace/lmh-pr25")
from src.scorer import TaskScorer
R = "/workspace/lmh-runs/tools-frontier-heldout"; MAXT = 12
FAM = {"tools_frontier_convention_risk": "T1", "tools_frontier_fx_calendar": "T2", "tools_frontier_reconciliation": "T3", "tools_frontier_split_holdings": "T4", "tools_frontier_portfolio_construction": "T5"}
ARMS = {"glimmer-base": "glimmer-base-tools-on", "glimmer-v3": "glimmer-v3-tools-on", "spark13": "spark13-tools-on", "spark-open": "open-tools-on"}
out = {}
for name, arm in ARMS.items():
    c = collections.Counter(); bt = collections.defaultdict(collections.Counter); bs = collections.defaultdict(collections.Counter)
    for s in (42, 1001, 1002):
        S = TaskScorer(f"{R}/tasks/seed{s}/answers"); latest = {}
        try:
            for l in open(f"{R}/runs/{arm}/seed{s}/results.jsonl"): x = json.loads(l); latest[x["task_id"]] = x
        except FileNotFoundError: continue
        for tid, x in latest.items():
            p = bool(S.score_task(tid, x.get("response") or "")["passed"]); f = FAM[x["task_type"]]
            tl = bool(x.get("forced_final")) or (x.get("turns") or 0) >= MAXT; em = not (x.get("response") or "").strip()
            for C in (c, bt[f], bs[s]): C["n"] += 1; C["pass"] += p; C["turn_limit"] += tl; C["empty"] += em
            c["err"] += bool(x.get("error"))
    out[name] = {"total": dict(c), "by_type": {k: dict(bt[k]) for k in sorted(bt)}, "by_seed": {k: dict(v) for k, v in bs.items()}}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)
