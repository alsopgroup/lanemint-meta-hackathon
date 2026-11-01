"""V3 (multi-turn tool trajectories, loss on ALL assistant messages, tool outputs masked). Derived from fw-serverless-sft-full/train.py.
Resume-safe Muse Glimmer 30B LoRA SFT on the Fireworks Serverless Training API.
usage: python train.py v1|v2
No sampling, no deployments. Never prints or writes FIREWORKS_API_KEY."""
import json, os, random, sys, time, traceback
from datetime import datetime, timedelta, timezone
import httpx
sys.path.insert(0, '/workspace/lmh-runs/fw-serverless-sft-full')
import common
import v3data
from common import sdk, BASE_MODEL, ACCOUNT, Log
from fireworks.training.sdk import FiretitanServiceClient, FireworksClient
from training.utils.serverless import enable_serverless_supervised_409_retry, serverless_base_url

KEY = "v3"; DS = dict(model="lmh-glimmer-tools-lora-v3")
from pathlib import Path
RUN = Path("/workspace/lmh-runs/lora-v3/train_run"); RUN.mkdir(parents=True, exist_ok=True)
HP = dict(base_model=BASE_MODEL, lora_rank=32, lora_alpha=64, train_attn=True, train_mlp=True, train_unembed=True,
          epochs=2, batch_size=8, learning_rate=5e-5, lr_schedule="constant", adam_beta1=0.9, adam_beta2=0.999,
          adam_eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0, loss="cross_entropy (sum over assistant-turn tokens)",
          train_on_what="per-assistant-message examples, LAST_ASSISTANT_MESSAGE each (tool/user/system masked)", renderer="muse_glimmer reasoning_strength=low", max_seq_len=common.MAX_SEQ,
          shuffle="random.Random(seed + epoch).shuffle per epoch", seed=0, state_save_every_steps=6, val_batch_size=4)
STATE = RUN / "state.json"
log = Log(RUN / "events.jsonl")
metrics_f = open(RUN / "metrics.jsonl", "a")
API_KEY = os.environ["FIREWORKS_API_KEY"]; H = {"Authorization": "Bearer " + API_KEY}
PROMO_DEADLINE = datetime(2026, 12, 1, tzinfo=timezone(timedelta(hours=-8)))

def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}
def save_state_file(st):
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1)); tmp.replace(STATE)

def billing():
    r = httpx.get(f"https://api.fireworks.ai/v1/accounts/{ACCOUNT}/billing/summary",
                  params={"startTime": "2026-10-01T00:00:00Z", "endTime": "2026-11-30T00:00:00Z"}, headers=H, timeout=60)
    r.raise_for_status(); out = {}
    for l in r.json().get("lineItems", []):
        c, u = l["totalCost"], l["unitAmount"]
        out[f'{l["category"]}|{l.get("groupingValue")}'] = dict(
            qty=l["quantity"], unit=int(u.get("units", "0")) + u.get("nanos", 0) / 1e9,
            total=int(c.get("units", "0")) + c.get("nanos", 0) / 1e9)
    return out

def cost_guard(where):
    """Abort if promo over, Glimmer training/sampler tokens carry a price, or on-demand GPU-hours rose vs baseline."""
    if datetime.now(timezone.utc) >= PROMO_DEADLINE: raise SystemExit("STOP: promo deadline passed")
    base = json.loads((RUN.parent / "billing" / "train_baseline.json").read_text())
    now = billing(); (RUN / f"billing_{where}.json").write_text(json.dumps(now, indent=1))
    bad = []
    for k, v in now.items():
        if ("Serverless Training Tokens" in k or "Glimmer" in k) and (v["unit"] > 0 or v["total"] > 0): bad.append((k, "nonzero", v))
        if "GPU" in k and v["total"] > base.get(k, {"total": 0.0})["total"] + 1e-9: bad.append((k, "gpu rose", v))
    log("cost_guard", where=where, ok=not bad, issues=bad, training_tokens={k: v for k, v in now.items() if "Training" in k})
    if bad:
        (RUN / "STOP").write_text(json.dumps(bad, default=str)); raise SystemExit(f"STOP: billing issue {bad}")

def val_loss(tc, val):
    nll = wsum = 0.0; s0 = time.time()
    for i in range(0, len(val), HP["val_batch_size"]):
        b = val[i:i + HP["val_batch_size"]]
        f = tc.forward(b, "cross_entropy").result()
        for j, d in enumerate(b):
            lp = f.loss_fn_outputs[j]["logprobs"].to_torch(); w = d.loss_fn_inputs["weights"].to_torch()
            assert len(lp) == len(w)
            nll += float(-(lp * w).sum()); wsum += float(w.sum())
    return nll / wsum, wsum, round(time.time() - s0, 1)

def promote(fw, session_name, save_name, model_id):
    cks = fw.list_training_session_checkpoints(session_name)
    tgt = [c for c in cks if c.get("promotable") and save_name in (c.get("checkpointName") or c.get("name") or "")]
    if not tgt: raise RuntimeError(f"no promotable checkpoint matching {save_name}: {[c.get('name') for c in cks]}")
    t = max(tgt, key=lambda c: c.get("createTime", ""))
    res = fw.promote_session_checkpoint(name=t["name"], output_model_id=model_id, base_model=BASE_MODEL)
    log("promoted", model=f"accounts/{ACCOUNT}/models/{model_id}", from_checkpoint=t["name"], state=res.get("state") if isinstance(res, dict) else None)
    return t["name"]

def main():
    if (RUN / "STOP").exists(): raise SystemExit("STOP flag present")
    st = load_state()
    if st.get("done"): log("already_done"); return 0
    if st.get("hp") and st["hp"] != HP: raise SystemExit(f"HP mismatch with existing state: {st['hp']}")
    st["hp"] = HP; st.setdefault("step", 0)
    # resume from the last saved optimizer state, not from the last logged step
    if st["step"] and not st.get(f"ep{HP['epochs']}_done"):
        st["step"] = st.get("last_state_step", 0) if st.get("last_state_path") else 0
    st.setdefault("attempts", 0); st["attempts"] += 1; save_state_file(st)
    (RUN / "hyperparameters.json").write_text(json.dumps(HP, indent=1))
    train, tstats = v3data.load("train"); val, vstats = v3data.load("validation")
    (RUN / "token_stats.json").write_text(json.dumps(tstats + vstats))
    log("skipped_over_max", n=sum(1 for s in tstats + vstats if s.get("skipped")))
    N = len(train); spe = N // HP["batch_size"]; assert spe >= 4
    total = spe * HP["epochs"]
    log("start", key=KEY, attempt=st["attempts"], resume_step=st["step"], steps_per_epoch=spe, total_steps=total,
        train_rows=N, val_rows=len(val), train_tokens=sum(s["tokens"] for s in tstats))
    cost_guard(f"attempt{st['attempts']}_start")
    orders = {}
    def order(e):
        if e not in orders:
            o = list(range(N)); random.Random(HP["seed"] + e).shuffle(o); orders[e] = o
        return orders[e]
    (RUN / "epoch_orders.json").write_text(json.dumps({e: order(e) for e in range(HP["epochs"])}))
    svc = FiretitanServiceClient(api_key=API_KEY, base_url=serverless_base_url("https://api.fireworks.ai"))
    enable_serverless_supervised_409_retry(svc)
    fw = FireworksClient(api_key=API_KEY, base_url="https://api.fireworks.ai")
    try:
        s0 = time.time()
        if st.get("last_state_path"):
            tc = svc.create_training_client_from_state_with_optimizer(st["last_state_path"])
            log("resumed_from_state", path=st["last_state_path"], step=st["step"], s=round(time.time() - s0, 1))
        else:
            assert st["step"] == 0
            tc = svc.create_lora_training_client(base_model=BASE_MODEL, rank=HP["lora_rank"], alpha=HP["lora_alpha"],
                                                 train_mlp=True, train_attn=True, train_unembed=True,
                                                 user_metadata={"purpose": "lmh-v3"})
            log("created_session", s=round(time.time() - s0, 1))
        session = svc.training_session_name
        st.setdefault("sessions", []).append({"session": session, "run": getattr(tc, "run_id", None), "attempt": st["attempts"]})
        save_state_file(st)
        r = httpx.get(f"https://api.fireworks.ai/v1/{session}", headers=H, timeout=60)
        log("session_meta", body=r.text[:1500])
        if "val_ep0" not in st:
            vl, w, s = val_loss(tc, val); st["val_ep0"] = vl; save_state_file(st)
            log("val_loss", epoch=0, loss=vl, weight=w, s=s); metrics_f.write(json.dumps({"step": 0, "val_loss": vl}) + "\n"); metrics_f.flush()
        adam = sdk.AdamParams(learning_rate=HP["learning_rate"], beta1=HP["adam_beta1"], beta2=HP["adam_beta2"],
                              eps=HP["adam_eps"], weight_decay=HP["weight_decay"], grad_clip_norm=HP["grad_clip_norm"])
        while st["step"] < total or not st.get(f"ep{HP['epochs']}_done"):
            step = st["step"]; e = step // spe if step < total else HP["epochs"] - 1
            if step < total and not (step > 0 and step % spe == 0 and not st.get(f"ep{step // spe}_done")):
                k = step % spe; idx = order(e)[k * HP["batch_size"]:(k + 1) * HP["batch_size"]]
                batch = [train[i] for i in idx]; t0 = time.time()
                fb = tc.forward_backward(batch, "cross_entropy").result(); t1 = time.time()
                opt = tc.optim_step(adam, emit_grad_norm_metrics=True).result()
                m = dict(fb.metrics or {}); om = dict(opt.metrics or {}); ws = common.weight_sum(batch)
                row = {"step": step + 1, "epoch": e + 1, "loss": m.get("loss:sum", 0.0) / ws, "loss_sum": m.get("loss:sum"),
                       "weight": ws, "tokens": common.ntokens(batch), "fb_s": round(t1 - t0, 1), "opt_s": round(time.time() - t1, 1),
                       "grad_norm": om.get("grad_norm"), "lr": om.get("lr:last"), "attempt": st["attempts"], "rows": idx}
                metrics_f.write(json.dumps(row) + "\n"); metrics_f.flush()
                log("step", **{k2: v for k2, v in row.items() if k2 != "rows"})
                st["step"] = step + 1
                if st["step"] % HP["state_save_every_steps"] == 0 and st["step"] % spe != 0:
                    p = tc.save_state(f"{KEY}-s{st['step']}").result().path
                    st["last_state_path"] = p; st["last_state_step"] = st["step"]; save_state_file(st); log("state_saved", path=p, step=st["step"])
                else:
                    save_state_file(st)
                continue
            # epoch boundary (st["step"] is a multiple of spe and epoch not yet finalized)
            ep = st["step"] // spe
            if st.get("last_state_step") != st["step"]:
                p = tc.save_state(f"{KEY}-ep{ep}-state").result().path
                st["last_state_path"] = p; st["last_state_step"] = st["step"]; save_state_file(st); log("state_saved", path=p, step=st["step"])
            if f"val_ep{ep}" not in st:
                vl, w, s = val_loss(tc, val); st[f"val_ep{ep}"] = vl; save_state_file(st)
                log("val_loss", epoch=ep, loss=vl, weight=w, s=s); metrics_f.write(json.dumps({"step": st["step"], "val_loss": vl}) + "\n"); metrics_f.flush()
            sname = f"{KEY}-ep{ep}"
            if not st.get(f"ep{ep}_model"):
                path = tc.save_weights_for_sampler(sname).result().path; log("sampler_saved", path=path)
                mid = f"{DS['model']}-ep{ep}"; ck = promote(fw, session, sname, mid)
                st[f"ep{ep}_model"] = f"accounts/{ACCOUNT}/models/{mid}"; st[f"ep{ep}_checkpoint"] = ck; save_state_file(st)
                if ep == HP["epochs"]:
                    try:
                        promote(fw, session, sname, DS["model"]); st["final_model"] = f"accounts/{ACCOUNT}/models/{DS['model']}"
                    except Exception as ex:
                        log("final_alias_promote_error", err=repr(ex)[:1000]); st["final_model"] = st[f"ep{ep}_model"]
            st[f"ep{ep}_done"] = True; save_state_file(st)
            cost_guard(f"ep{ep}_end")
        st["done"] = True; save_state_file(st); log("run_done", **{k: st.get(k) for k in ("val_ep0", "val_ep1", "val_ep2", "ep1_model", "ep2_model", "final_model")})
        return 0
    except BaseException as ex:
        log("error", step=st.get("step"), err=repr(ex)[:3000], tb=traceback.format_exc()[-3000:])
        return 2
    finally:
        try: svc.delete_training_session(); log("session_deleted")
        except Exception as ex: log("session_delete_error", err=repr(ex)[:300])
        try: svc.close(); fw.close()
        except Exception: pass

if __name__ == "__main__":
    sys.exit(main())
