"""Full tool-mode run (tools ON). Derived from toolmode-pilot/p2/pilot.py. Keys from env by name only; never printed/written.
Shared ledger with per-process in-flight worst-case reservations; every paid call pre-guarded: spent + inflight + worst <= STOP_AT."""
import argparse, copy, fcntl, json, os, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path("/workspace/lmh-runs/lora-v3"); LEDGER = ROOT / "ledger.json"; STOP_AT = 60.00  # ledger = list price + worst-case for errors (observed ~2.3x billed); real cap enforced by billing guard below
PRICES = {"spark13": (1.25e-6, 4.25e-6), "open": (1.25e-6, 4.25e-6), "glimmer": (0.0, 0.0)}
GLIMMER_LIST = (1.96e-6, 4.88e-6)
GMODELS = {"v2ep2": "alsopgroup/run-6c2bec1119fc4800a22f8cc1bebbafec/v2-ep2-0020f17b",
           "v1ep2": "alsopgroup/run-484cf3445bd54e9b9e88fa54a4f14ccc/v1-ep2-f817053a", "base": None,
           "v3": os.environ.get("LMH_V3_SAMPLER_PATH")}
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=list(PRICES)); ap.add_argument("--glimmer-model", default="v2ep2", choices=list(GMODELS))
ap.add_argument("--seeds", default="7101"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--tools-off", action="store_true"); ap.add_argument("--max-turns", type=int, default=12)
a = ap.parse_args()
SEEDS = {int(x) for x in a.seeds.split(",")}
FORBIDDEN = {42, 1001, 1002, 5001, 5002} | set(range(2001, 2041))
assert not (SEEDS & FORBIDDEN), "forbidden seed"
if a.arm == "glimmer": assert SEEDS == {5999}, "glimmer arms: sanity seed 5999 only"
else: assert 5999 not in SEEDS and all(5101 <= s <= 5199 for s in SEEDS), "teacher: training seeds 5101-5199 only"
sys.path.insert(0, "/workspace/lmh-pr25")
for k in ("META_MAX_TOKENS", "META_REASONING_EFFORT", "META_BASE_URL", "META_MODEL_NAME"): os.environ.pop(k, None)
from src.runner import TaskRunner
from src.tool_runner import ToolTaskRunner, OpenAIChatClient, estimate_prompt_tokens, _count_tokens
ARM = f"{a.arm}{'-' + a.glimmer_model if a.arm == 'glimmer' else ''}-tools-{'off' if a.tools_off else 'on'}"; assert not (a.tools_off and a.arm != "spark13"); AOUT = ROOT / "runs" / ARM; AOUT.mkdir(parents=True, exist_ok=True)
PIN, POUT = PRICES[a.arm]; PID = str(os.getpid()); tlock = threading.Lock(); TL = threading.local()
class BudgetStop(BaseException): pass
def log(*x):
    s = time.strftime("%H:%M:%S ") + " ".join(map(str, x)); print(s, flush=True)
    with tlock, open(AOUT / "driver.log", "a") as f: f.write(s + "\n")
def ledger_update(fn):
    with tlock, open(str(LEDGER) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"spent_usd": 0.0, "calls": 0, "by_arm": {}, "inflight": {}, "glimmer_list_shadow_usd": 0.0, "worst_case_charges": []}
        for p in list(led["inflight"]):
            if not Path(f"/proc/{p}").exists(): led["inflight"].pop(p)
        r = fn(led); LEDGER.write_text(json.dumps(led, indent=1)); return r
def pre_guard(est_in, max_tokens):
    worst = est_in * PIN + max_tokens * POUT
    def f(led):
        infl = sum(led["inflight"].values())
        if led["spent_usd"] + infl + worst > STOP_AT:
            raise BudgetStop(f"spent {led['spent_usd']:.4f} + inflight {infl:.4f} + worst {worst:.4f} > {STOP_AT}")
        led["inflight"][PID] = led["inflight"].get(PID, 0.0) + worst
    ledger_update(f); return worst
def record(usage, finish, secs, worst, error=None):
    pin, pout = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
    usd = pin * PIN + pout * POUT; shadow = (pin * GLIMMER_LIST[0] + pout * GLIMMER_LIST[1]) if a.arm == "glimmer" else 0.0
    def f(led):
        led["inflight"][PID] = max(0.0, led["inflight"].get(PID, 0.0) - worst)
        led["spent_usd"] += usd; led["calls"] += 1; led["glimmer_list_shadow_usd"] += shadow
        b = led["by_arm"].setdefault(ARM, {"usd": 0.0, "calls": 0, "prompt": 0, "completion": 0}); b["usd"] += usd; b["calls"] += 1; b["prompt"] += pin; b["completion"] += pout
        if error: led["worst_case_charges"].append({"arm": ARM, "seed": TL.seed, "task": TL.task, "turn": TL.turn, "reason": error, "usd": usd})
        return led["spent_usd"]
    total = ledger_update(f)
    with tlock, open(AOUT / "usage_log.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": TL.seed, "task_id": TL.task, "turn": TL.turn, "secs": round(secs, 1), "finish": finish,
                             "prompt_tokens": pin, "completion_tokens": pout, "usd": round(usd, 6), "ledger_total": round(total, 6), "error": error}) + "\n")
    return usd, total
def model_log(rec):
    with tlock, open(AOUT / "api_model_log.jsonl", "a") as fh: fh.write(json.dumps({"seed": TL.seed, "task_id": TL.task, "turn": TL.turn, **rec}) + "\n")

def make_glimmer():
    HERE = Path("/workspace/lmh-runs/fw-serverless-sft-full")
    sys.path.insert(0, str(HERE / "cookbook" / "training")); sys.path.insert(0, str(HERE / "cookbook"))
    from fireworks.training.sdk import FiretitanServiceClient
    from fireworks.training import sdk
    from training.utils.serverless import serverless_base_url
    from training.renderer.muse_glimmer import MuseGlimmerRenderer, MuseGlimmerOptions
    from training.renderer.tokenizer import get_tokenizer
    tok = get_tokenizer("meta-models/Muse-Glimmer-30B"); R = MuseGlimmerRenderer(tok, options=MuseGlimmerOptions(reasoning_strength="low")); STOP = R.get_stop_sequences()
    svc = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=serverless_base_url("https://api.fireworks.ai"))
    sess = str(svc.training_session_name or svc.training_session_id)
    with open(AOUT / "sessions.txt", "a") as f: f.write(sess + "\n")
    path = GMODELS[a.glimmer_model]
    sampler = svc.create_sampling_client(model_path=path, tokenizer=tok) if path else svc.create_sampling_client(base_model="accounts/fireworks/models/muse-glimmer-30b", tokenizer=tok)
    smodel = sampler.deployment_sampler.model
    if True:  # resume: v2ep2 sampler also hits 600s read hangs
        # base sampler hangs until the SDK's 600 s HTTP read timeout, then succeeds on retry; shorten so retries fire sooner
        import functools
        from fireworks.training.sdk import sampling as _smp
        _orig = _smp.DeploymentSampler.async_completions_stream
        if not getattr(_orig, "_lmh_patched", False):
            @functools.wraps(_orig)
            def _patched(self, *args, **kw):
                kw.setdefault("http_timeout", 240)
                return _orig(self, *args, **kw)
            _patched._lmh_patched = True
            _smp.DeploymentSampler.async_completions_stream = _patched
        log("base sampler http_timeout=240s (SDK default 600s)")
    log("glimmer session", sess, "sampler", smodel)
    def render_msgs(messages, tools):
        msgs = []
        for m in copy.deepcopy(messages):
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    try: fn["arguments"] = json.loads(fn["arguments"])
                    except Exception: fn["arguments"] = {"code": fn["arguments"]}
            msgs.append(m)
        if tools: msgs = R.create_conversation_prefix_with_tools(tools) + msgs
        return R.build_generation_prompt(msgs)
    class GlimmerClient:
        provider = "meta"  # forced final = drop tools + user nudge
        def chat(self, messages, *, tools=None, tool_choice=None, max_tokens=None, temperature=0.1):
            prompt = render_msgs(messages, tools); n_in = len(prompt.to_ints())
            params = sdk.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95, top_k=64, stop=STOP)
            res = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result(timeout=900)
            seq = res.sequences[0]; toks = list(seq.tokens)
            msg, term = R.parse_response(toks); oai = R.to_openai_message(msg)
            out = {"role": "assistant", "content": oai.get("content") or ""}
            if oai.get("reasoning_content"): out["reasoning_content"] = oai["reasoning_content"]
            if oai.get("tool_calls"): out["tool_calls"] = oai["tool_calls"]
            with tlock, open(AOUT / "raw_log.jsonl", "a") as f:
                f.write(json.dumps({"seed": TL.seed, "task_id": TL.task, "turn": TL.turn, "raw": tok.decode(toks)[:20000], "unparsed": str(msg.get("unparsed_tool_calls"))[:2000]}) + "\n")
            model_log({"sampler_model": smodel, "session": sess})
            out["_finish_reason"] = "length" if len(toks) >= max_tokens else "stop"
            out["_usage"] = {"prompt_tokens": n_in, "completion_tokens": len(toks), "total_tokens": n_in + len(toks)}
            return out
    def cleanup():
        try: sampler.close()
        except Exception as e: log("sampler close err", repr(e)[:200])
        try: svc.delete_training_session(); log("session deleted", sess)
        except Exception as e: log("SESSION DELETE ERROR", repr(e)[:300])
        try: svc.close()
        except Exception: pass
    return GlimmerClient(), cleanup

class Guarded:
    def __init__(self, inner): self.inner = inner; self.provider = inner.provider
    def chat(self, messages, *, tools=None, tool_choice=None, max_tokens=None):
        est = estimate_prompt_tokens(messages) + (_count_tokens(json.dumps(tools)) if tools else 0)
        worst = pre_guard(est, max_tokens); TL.turn += 1; t0 = time.time()
        try:
            for k in range(8):
                try:
                    out = self.inner.chat(messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens); break
                except Exception as e429:
                    n = type(e429).__name__
                    if k == 7 or n not in ("RateLimitError", "APITimeoutError", "APIConnectionError") or (n != "RateLimitError" and k >= 1): raise
                    if n != "RateLimitError":  # possibly billed: charge worst case, then retry the same turn
                        record({"prompt_tokens": est, "completion_tokens": max_tokens}, "error", time.time() - t0, 0.0, error=n)
                    log(f"  s{TL.seed} {TL.task} {n}, retry in {15 * (k + 1)}s"); time.sleep(15 * (k + 1))
        except Exception as e:
            usd, total = record({"prompt_tokens": est, "completion_tokens": max_tokens}, "error", time.time() - t0, worst, error=type(e).__name__)
            log(f"  s{TL.seed} {TL.task} turn {TL.turn} ERROR {type(e).__name__}; worst-case ${usd:.4f} ledger=${total:.4f}")
            raise
        u = out.get("_usage") or {}
        usd, total = record(u, out.get("_finish_reason"), time.time() - t0, worst)
        log(f"  s{TL.seed} {TL.task} turn {TL.turn} in={u.get('prompt_tokens')} out={u.get('completion_tokens')} tc={len(out.get('tool_calls') or [])} ${usd:.4f} ledger=${total:.4f}")
        return out

def wrap_sdk(client, base_url):
    orig = client.chat.completions.create
    def create(**kw):
        resp = orig(**kw)
        model_log({"base_url": base_url, "request_model": kw.get("model"), "response_model": getattr(resp, "model", None), "response_id_present": bool(getattr(resp, "id", None))})
        return resp
    client.chat.completions.create = create

cleanup = lambda: None
if a.arm == "spark13":
    base = TaskRunner(model_name="muse-spark-1.3", reasoning_effort="low"); assert base.base_url == "https://api.meta.ai/v1"
    wrap_sdk(base.client, base.base_url); MODEL = "muse-spark-1.3"
    inner = OpenAIChatClient(base.client, model_name=MODEL, provider="meta", reasoning_effort="low", call_timeout=900)
elif a.arm == "open":
    MODEL = "accounts/fireworks/models/muse-spark-open"
    base = TaskRunner(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"], model_name=MODEL, reasoning_effort="low")
    wrap_sdk(base.client, base.base_url)
    inner = OpenAIChatClient(base.client, model_name=MODEL, provider="fireworks", reasoning_effort="low", call_timeout=100)
else:
    MODEL = f"glimmer-{a.glimmer_model}"; inner, cleanup = make_glimmer()
client = Guarded(inner)
if a.tools_off:
    _create = base.client.chat.completions.create
    def _guarded_create(**kw):
        est = estimate_prompt_tokens(kw["messages"]); worst = pre_guard(est, kw["max_tokens"]); TL.turn += 1; t0 = time.time()
        try: resp = _create(**kw)
        except Exception as e:
            usd, total = record({"prompt_tokens": est, "completion_tokens": kw["max_tokens"]}, "error", time.time() - t0, worst, error=type(e).__name__); raise
        u = resp.usage; usd, total = record({"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}, resp.choices[0].finish_reason, time.time() - t0, worst)
        log(f"  s{TL.seed} {TL.task} off in={u.prompt_tokens} out={u.completion_tokens} ${usd:.4f} ledger=${total:.4f}"); return resp
    base.client.chat.completions.create = _guarded_create
def fw_open_spend():
    import httpx
    r = httpx.get("https://api.fireworks.ai/v1/accounts/alsopgroup/billing/summary", params={"startTime": "2026-10-01T00:00:00Z", "endTime": "2026-11-30T00:00:00Z"},
                  headers={"Authorization": "Bearer " + os.environ["FIREWORKS_API_KEY"]}, timeout=60); r.raise_for_status()
    tot = lambda items: sum(int(l["totalCost"].get("units", "0")) + l["totalCost"].get("nanos", 0) / 1e9 for l in items if "Spark Open" in str(l.get("groupingValue")))
    before = json.loads((ROOT / "billing" / "before_teacher.json").read_text())
    return tot(r.json().get("lineItems", [])) - sum(v["total"] for k, v in before.items() if "Spark Open" in k)
if a.arm == "open":
    _fw = fw_open_spend(); log(f"fireworks billed Spark Open since start: ${_fw:.2f}")
    assert _fw < 38.0, "billing guard: approved cap is $45 actual billed; stop new seeds at $38"
stop = threading.Event()
try:
    for s in [int(x) for x in a.seeds.split(",")]:
        TD = ROOT / "tasks" / f"seed{s}"; OUT = AOUT / f"seed{s}"; OUT.mkdir(parents=True, exist_ok=True)
        r = ToolTaskRunner(base_url="https://api.meta.ai/v1" if a.arm == "spark13" else "https://api.fireworks.ai/inference/v1", api_key="unused-here",
                           dry_run=(a.arm == "glimmer"), model_name=MODEL, reasoning_effort="low", tools="python", max_turns=a.max_turns, sandbox_timeout=30,
                           protocol="auto", chat_client=client, tasks_dir=TD, transcript_dir=OUT / "transcripts")
        tasks = json.loads((TD / "tasks.json").read_text())[: a.limit]
        prev = {}
        if (OUT / "results.jsonl").exists():
            for l in open(OUT / "results.jsonl"):
                x = json.loads(l); prev[x["task_id"]] = x  # last line per task wins
        def one(t):
            p = prev.get(t["id"])
            if p and (not p.get("error") or (p.get("attempt") or 1) >= 4): return p
            if stop.is_set(): return None
            attempt = ((p or {}).get("attempt") or 0) + 1
            TL.seed, TL.task, TL.turn = s, t["id"], 0; t0 = time.time()
            try:
                res = base.run_task(t, base.load_data(TD, t["data_dir"])) if a.tools_off else r.run_task(t, (TD / t["data_file"]).read_text())
            except BudgetStop as e:
                stop.set(); log("BUDGET STOP", e); (AOUT / "BUDGET_STOP").write_text(str(e)); return None
            except Exception as e:
                res = {"task_id": t["id"], "task_type": t["type"], "response": "", "error": type(e).__name__, "finish_reason": None, "model": MODEL}
                log(f"TASK ERROR s{s} {t['id']} {type(e).__name__}: {str(e)[:150]}")
            res["attempt"] = attempt; res["secs"] = round(time.time() - t0, 1); res["seed"] = s
            with tlock, open(OUT / "results.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
            log(f"DONE s{s} {t['id']} att={attempt} turns={res.get('turns')} tc={res.get('tool_calls')} forced={res.get('forced_final')} err={res.get('error')} {res['secs']}s")
            return res
        with ThreadPoolExecutor(a.workers) as ex: results = [x for x in ex.map(one, tasks) if x is not None]
        # results.json = latest record per task (rerun replaces the errored attempt)
        latest = {}
        for l in open(OUT / "results.jsonl"):
            x = json.loads(l); latest[x["task_id"]] = x
        (OUT / "results.json").write_text(json.dumps([latest[t["id"]] for t in tasks if t["id"] in latest], indent=1))
        if stop.is_set(): break
finally:
    cleanup(); log("ARM FINISHED", ARM)
