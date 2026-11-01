"""Build multi-turn SFT rows from teacher transcripts: keep strict pass AND turns<=10 AND not forced.
Rows: system + user preamble (exact eval preamble) + assistant tool_call / tool result ... final JSON.
Validation split = seed 5110 (held out from training). Sanity seed 5999 never touched here."""
import json, sys, hashlib
from pathlib import Path
sys.path.insert(0, "/workspace/lmh-pr25")
from src.scorer import TaskScorer
from src.tool_runner import ToolTaskRunner
ROOT = Path("/workspace/lmh-runs/lora-v3"); ARM = sys.argv[1] if len(sys.argv) > 1 else "open-tools-on"
VAL_SEEDS = {5110}; MAXT = 10
out = {"train": [], "validation": []}; stats = {"total": 0, "pass": 0, "kept": 0, "drop_turns": 0, "drop_forced": 0, "drop_fail": 0, "drop_multi_tc": 0}
for sd in sorted((ROOT / "runs" / ARM).glob("seed*")):
    s = int(sd.name[4:]); assert 5101 <= s <= 5199
    TD = ROOT / "tasks" / sd.name; tasks = {t["id"]: t for t in json.loads((TD / "tasks.json").read_text())}
    S = TaskScorer(TD / "answers")
    r = ToolTaskRunner(base_url="https://api.fireworks.ai/inference/v1", api_key="unused", dry_run=True, model_name="x", reasoning_effort="low",
                       tools="python", max_turns=12, protocol="auto", tasks_dir=TD, transcript_dir=ROOT / "tmp_unused")
    latest = {}
    for l in open(sd / "results.jsonl"):
        x = json.loads(l); latest[x["task_id"]] = x
    for tid, res in latest.items():
        stats["total"] += 1
        sc = S.score_task(tid, res.get("response") or "")
        if not sc["passed"]: stats["drop_fail"] += 1; continue
        stats["pass"] += 1
        if res.get("forced_final"): stats["drop_forced"] += 1; continue
        if (res.get("turns") or 99) > MAXT: stats["drop_turns"] += 1; continue
        t = tasks[tid]; data = (TD / t["data_file"]).read_text()
        msgs = [{"role": "system", "content": "You are a financial data analyst. Analyze the provided data and return results as valid JSON only."},
                {"role": "user", "content": r._preamble(t, data)}]
        multi = False
        for l in open(sd / "transcripts" / f"{tid}.jsonl"):
            e = json.loads(l)
            if e["event"] != "message": continue
            m = dict(e["message"]); m.pop("reasoning_content", None)
            if m["role"] == "assistant" and m.get("tool_calls"):
                if len(m["tool_calls"]) > 1: multi = True
                m["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in m["tool_calls"]]
            msgs.append(m)
        if multi: stats["drop_multi_tc"] += 1; continue
        assert msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls")
        out["validation" if s in VAL_SEEDS else "train"].append({"messages": msgs, "meta": {"seed": s, "task_id": tid, "type": t["type"], "turns": res["turns"]}})
        stats["kept"] += 1
if len(out["validation"]) < 8:  # fallback: deterministic ~10% of train as validation
    import random; tr = out["train"]; random.Random(0).shuffle(tr); k = max(8, len(tr) // 10); out["validation"] += tr[:k]; out["train"] = tr[k:]
(ROOT / "sft").mkdir(exist_ok=True)
for k, rows in out.items():
    p = ROOT / "sft" / f"{k}.jsonl"; p.write_text("".join(json.dumps(x) + "\n" for x in rows))
    stats[f"{k}_rows"] = len(rows); stats[f"{k}_sha"] = hashlib.sha256(p.read_bytes()).hexdigest()
from collections import Counter
stats["by_type"] = Counter(x["meta"]["type"] for v in out.values() for x in v)
stats["turns_hist"] = Counter(x["meta"]["turns"] for v in out.values() for x in v)
(ROOT / "sft" / "manifest.json").write_text(json.dumps(stats, indent=1, default=dict)); print(json.dumps(stats, default=dict))
