import json, copy, hashlib, sys
sys.path.insert(0, "/workspace/lmh-pr25"); sys.path.insert(0, "/workspace/lmh-runs/fw-serverless-sft-full")
import common
from common import sdk
from training.renderer import TrainOnWhat
from src.tool_runner import RUN_PYTHON_TOOL
D = "/workspace/lmh-runs/lora-v3/sft"
def to_render(messages):
    """Same conversion as the eval's GlimmerClient.render_msgs (arguments JSON -> dict) + tools prefix."""
    tok, R = common.renderer(); msgs = []
    for m in copy.deepcopy(messages):
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try: fn["arguments"] = json.loads(fn["arguments"])
                except Exception: fn["arguments"] = {"code": fn["arguments"]}
        msgs.append(m)
    return R.create_conversation_prefix_with_tools([RUN_PYTHON_TOOL]) + msgs
def examples(messages):
    """One example per assistant message: prefix = exactly what build_generation_prompt sees at that turn
    (renderer lacks the extension property), loss on that assistant message only; tool/user/system masked."""
    for k, m in enumerate(messages):
        if m["role"] == "assistant": yield messages[:k + 1]
def load(split):
    tok, R = common.renderer(); rows = [json.loads(l) for l in open(f"{D}/{split}.jsonl") if l.strip()]
    datums, stats = [], []
    for i, row in enumerate(rows):
        for j, conv in enumerate(examples(row["messages"])):
            mi, w = R.build_supervised_example(to_render(conv), train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            t = [int(x) for x in mi.to_ints()]; w = [float(x) for x in w.tolist()]
            if len(t) > common.MAX_SEQ: stats.append({"i": i, "j": j, "tokens": len(t), "skipped": True}); continue
            stats.append({"i": i, "j": j, "tokens": len(t), "trained": sum(1 for x in w[1:] if x > 0), "over_max": False})
            datums.append(sdk.Datum(model_input=sdk.ModelInput.from_ints(t[:-1]), loss_fn_inputs={"target_tokens": t[1:], "weights": w[1:]}))
    return datums, stats
