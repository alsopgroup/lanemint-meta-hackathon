"""Tool-mode agent-loop tests with scripted fake clients (no network)."""

import copy
import json
from pathlib import Path

from src.runner import TaskRunner
from src.tool_runner import RUN_PYTHON_TOOL, ToolTaskRunner

FINAL = '{"problems": [], "mean_return": 0.5}'


class ScriptedClient:
    """Fake chat client: returns scripted assistant dicts in order.

    The last script entry repeats, so turn-cap tests can loop forever
    on tool calls until the harness forces a final answer (the script
    distinguishes forced calls via tools/tool_choice in `respond`).
    """

    provider = "fireworks"

    def __init__(self, script, forced_final=None):
        self.script = list(script)
        self.forced_final = forced_final
        self.calls = []

    def chat(self, messages, *, tools=None, tool_choice=None, max_tokens=None):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
        })
        is_forced = tool_choice == "none" or (
            tools is None and self.calls and
            any(m.get("content", "").startswith("No more tool calls")
                for m in messages if isinstance(m.get("content"), str)))
        if is_forced and self.forced_final is not None:
            return copy.deepcopy(self.forced_final)
        if len(self.script) > 1:
            return copy.deepcopy(self.script.pop(0))
        return copy.deepcopy(self.script[0])


def tool_call(code="print(1)", call_id="call_1", arguments=None):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "run_python",
                "arguments": arguments if arguments is not None
                else json.dumps({"code": code}),
            },
        }],
        "_usage": {"prompt_tokens": 10, "completion_tokens": 5,
                   "total_tokens": 15},
        "_finish_reason": "tool_calls",
    }


def final_msg(content=FINAL):
    return {
        "role": "assistant",
        "content": content,
        "_usage": {"prompt_tokens": 10, "completion_tokens": 5,
                   "total_tokens": 15},
        "_finish_reason": "stop",
    }


class FakeSandbox:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"stdout": "42\n", "stderr": "",
                                 "exit": 0, "timed_out": False,
                                 "truncated": False}

    def __call__(self, code, csv_path, workdir, limits):
        self.calls.append({"code": code, "csv_path": Path(csv_path),
                           "workdir": Path(workdir)})
        return dict(self.result)


TASK = {"id": "task_f_001", "type": "frontier_rolling_risk",
        "prompt": "Compute mean_return.", "data_file": "task_f_001.csv"}
DATA = "date,close\n2024-01-01,10\n2024-01-02,11\n"


def make_runner(client, sandbox=None, tmp_path=None, **kwargs):
    kwargs.setdefault("dry_run", True)
    runner = ToolTaskRunner(
        chat_client=client, sandbox_fn=sandbox or FakeSandbox(), **kwargs)
    if tmp_path is not None:
        runner.transcript_dir = Path(tmp_path) / "transcripts"
    return runner


# -- default (no-tool) path is unchanged ------------------------------------

def test_no_tool_request_is_byte_identical(monkeypatch):
    """With no --tools flag the plain runner sends exactly today's request."""
    from tests.test_runner import _runner_with_fake
    runner, fake = _runner_with_fake(monkeypatch)
    result = runner.run_task(
        {"id": "task_f_2001", "type": "frontier_rolling_risk",
         "prompt": "p"}, "data")
    assert fake.kwargs == {
        "model": runner.model_name,
        "messages": [
            {"role": "system",
             "content": "You are a financial data analyst. Analyze the "
                        "provided data and return results as valid JSON only."},
            {"role": "user", "content": "p\n\nData:\ndata"},
        ],
        "temperature": 0.1,
        "max_tokens": 32000,
    }
    assert set(result) == {
        "task_id", "task_type", "response", "model", "max_tokens",
        "reasoning_effort", "finish_reason", "prompt_tokens",
        "completion_tokens", "total_tokens"}


def test_cli_tools_flag_selects_tool_runner(monkeypatch, tmp_path):
    import sys

    from src import runner as runner_mod
    from src import tool_runner as tool_mod
    seen = {}

    class FakeToolRunner:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run_all_tasks(self, **kwargs):
            return []

    monkeypatch.setattr(tool_mod, "ToolTaskRunner", FakeToolRunner)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "runner", "--dry-run", "--tools", "python", "--max-turns", "5",
        "--sandbox-timeout", "7", "--tasks-dir", str(tasks),
        "--output-dir", str(tmp_path / "out")])
    assert runner_mod.main() == 0
    assert seen["tools"] == "python"
    assert seen["max_turns"] == 5
    assert seen["sandbox_timeout"] == 7.0
    assert seen["tasks_dir"] == tasks


def test_default_runner_class_unchanged_via_cli(monkeypatch, tmp_path):
    import sys

    from src import runner as runner_mod
    seen = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            seen["class"] = "TaskRunner"

        def run_all_tasks(self, **kwargs):
            return []

    monkeypatch.setattr(runner_mod, "TaskRunner", FakeRunner)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    monkeypatch.setattr(sys, "argv", ["runner", "--dry-run",
                                      "--tasks-dir", str(tasks)])
    assert runner_mod.main() == 0
    assert seen["class"] == "TaskRunner"


def test_openai_adapter_shapes_requests_per_provider():
    from types import SimpleNamespace

    from src.tool_runner import OpenAIChatClient, detect_provider

    assert detect_provider("https://api.meta.ai/v1") == "meta"
    assert detect_provider("https://api.fireworks.ai/inference/v1") \
        == "fireworks"
    assert detect_provider("https://example.invalid/v1") \
        == "openai-compatible"

    class FakeMessage:
        def model_dump(self):
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "run_python",
                                                 "arguments": "{}"}}],
                    "audio": None, "annotations": []}

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=FakeMessage(), finish_reason="tool_calls")],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2,
                                      total_tokens=5))

    sdk = SimpleNamespace(chat=SimpleNamespace(
        completions=FakeCompletions()))
    meta = OpenAIChatClient(sdk, model_name="m", provider="meta")
    assistant = meta.chat([{"role": "user", "content": "hi"}],
                          tools=[RUN_PYTHON_TOOL], tool_choice="auto",
                          max_tokens=100)
    assert "parallel_tool_calls" not in captured  # Meta: auto only
    assert captured["timeout"] == 900.0
    # Full tool-call message survives; SDK-only null fields do not.
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert "audio" not in assistant and "annotations" not in assistant
    assert assistant["_usage"]["total_tokens"] == 5

    fireworks = OpenAIChatClient(sdk, model_name="m", provider="fireworks")
    fireworks.chat([{"role": "user", "content": "hi"}],
                   tools=[RUN_PYTHON_TOOL], tool_choice="none",
                   max_tokens=100)
    assert captured["parallel_tool_calls"] is False
    assert captured["tool_choice"] == "none"


# -- native tool protocol -----------------------------------------------------

def test_single_tool_turn_then_final(tmp_path):
    client = ScriptedClient([tool_call(), final_msg()])
    sandbox = FakeSandbox()
    runner = make_runner(client, sandbox, tmp_path)
    result = runner.run_task(TASK, DATA)
    assert result["response"] == FINAL
    assert result["turns"] == 2
    assert result["tool_calls"] == 1
    assert result["tool_errors"] == 0
    assert result["forced_final"] is False
    assert result["protocol"] == "tools"
    assert result["prompt_tokens"] == 20
    assert result["completion_tokens"] == 10
    assert sandbox.calls[0]["code"] == "print(1)"
    # First call offered exactly the one tool, auto choice, no CSV pasted.
    first = client.calls[0]
    assert first["tools"] == [RUN_PYTHON_TOOL]
    assert first["tool_choice"] == "auto"
    user = first["messages"][1]["content"]
    assert "/data/task.csv" in user
    assert "CSV header: date,close" in user
    assert "\nData:\n" not in user  # preview only, not the pasted-CSV form
    # Tool result is returned as a tool message keyed by call id, and
    # the full assistant tool-call message was replayed before it.
    second_msgs = client.calls[1]["messages"]
    assert second_msgs[2]["tool_calls"][0]["id"] == "call_1"
    assert second_msgs[3]["role"] == "tool"
    assert second_msgs[3]["tool_call_id"] == "call_1"
    assert "42" in second_msgs[3]["content"]
    # Transcript was written with the tool code.
    transcript = (tmp_path / "transcripts" / "task_f_001.jsonl").read_text()
    assert "print(1)" in transcript


def test_multi_turn_tool_use(tmp_path):
    client = ScriptedClient(
        [tool_call("print(1)", "c1"), tool_call("print(2)", "c2"),
         final_msg()])
    sandbox = FakeSandbox()
    result = make_runner(client, sandbox, tmp_path).run_task(TASK, DATA)
    assert result["tool_calls"] == 2
    assert result["turns"] == 3
    assert [c["code"] for c in sandbox.calls] == ["print(1)", "print(2)"]


def test_turn_cap_forces_final_on_fireworks(tmp_path):
    client = ScriptedClient([tool_call()], forced_final=final_msg())
    runner = make_runner(client, tmp_path=tmp_path, max_turns=2)
    result = runner.run_task(TASK, DATA)
    assert result["forced_final"] is True
    assert result["response"] == FINAL
    assert len(client.calls) == 3
    assert client.calls[-1]["tool_choice"] == "none"
    assert client.calls[-1]["tools"] == [RUN_PYTHON_TOOL]


def test_turn_cap_forces_final_on_meta_without_tool_choice(tmp_path):
    client = ScriptedClient([tool_call()], forced_final=final_msg())
    client.provider = "meta"
    runner = make_runner(client, tmp_path=tmp_path, max_turns=1)
    result = runner.run_task(TASK, DATA)
    assert result["forced_final"] is True
    last = client.calls[-1]
    assert last["tools"] is None
    assert last["tool_choice"] is None
    assert any("No more tool calls" in (m.get("content") or "")
               for m in last["messages"])


def test_malformed_tool_arguments_counted_not_executed(tmp_path):
    bad = tool_call(arguments="{not json")
    client = ScriptedClient([bad, final_msg()])
    sandbox = FakeSandbox()
    result = make_runner(client, sandbox, tmp_path).run_task(TASK, DATA)
    assert result["tool_errors"] == 1
    assert result["tool_calls"] == 0
    assert sandbox.calls == []
    tool_reply = client.calls[1]["messages"][3]
    assert "malformed" in tool_reply["content"]


def test_two_tool_calls_in_one_turn_executes_only_first(tmp_path):
    both = tool_call("print(1)", "c1")
    both["tool_calls"].append({
        "id": "c2", "type": "function",
        "function": {"name": "run_python",
                     "arguments": json.dumps({"code": "print(2)"})}})
    client = ScriptedClient([both, final_msg()])
    sandbox = FakeSandbox()
    result = make_runner(client, sandbox, tmp_path).run_task(TASK, DATA)
    assert result["tool_calls"] == 1
    assert [c["code"] for c in sandbox.calls] == ["print(1)"]
    replies = client.calls[1]["messages"][3:]
    assert [r["tool_call_id"] for r in replies] == ["c1", "c2"]
    assert "Only one tool call per turn" in replies[1]["content"]


def test_non_json_and_list_finals_pass_through(tmp_path):
    for content in ("not json at all", "[1, 2, 3]"):
        client = ScriptedClient([final_msg(content)])
        result = make_runner(client, tmp_path=tmp_path).run_task(TASK, DATA)
        assert result["response"] == content
        assert result["forced_final"] is False


def test_context_guard_forces_final(tmp_path):
    client = ScriptedClient([tool_call()], forced_final=final_msg())
    runner = make_runner(client, tmp_path=tmp_path, context_token_limit=1)
    result = runner.run_task(TASK, DATA)
    assert result["forced_final"] is True
    assert result["tool_calls"] == 0
    assert len(client.calls) == 1
    assert client.calls[0]["tool_choice"] == "none"


def test_cost_guard_forces_final(tmp_path):
    client = ScriptedClient([tool_call(), tool_call("print(2)", "c2")],
                            forced_final=final_msg())
    runner = make_runner(client, tmp_path=tmp_path,
                         cost_cap_usd=1.0, input_cost_per_mtok=1.0,
                         output_cost_per_mtok=1.0)
    # After the first call, tighten the cap so the per-call guard trips.
    original = client.chat

    def tightening_chat(*a, **kw):
        out = original(*a, **kw)
        runner.cost_cap_usd = 0.0
        return out

    client.chat = tightening_chat
    result = runner.run_task(TASK, DATA)
    assert result["forced_final"] is True
    assert result["tool_calls"] == 1
    assert "cost_usd" in result


def test_tool_timeout_counted(tmp_path):
    sandbox = FakeSandbox({"stdout": "", "stderr": "", "exit": None,
                           "timed_out": True, "truncated": False})
    client = ScriptedClient([tool_call(), final_msg()])
    result = make_runner(client, sandbox, tmp_path).run_task(TASK, DATA)
    assert result["tool_timeouts"] == 1
    assert result["tool_calls"] == 1


# -- text protocol ------------------------------------------------------------

def test_text_protocol_fallback(tmp_path):
    text_turn = {"role": "assistant",
                 "content": "Let me compute.\n```python\nprint(6*7)\n```",
                 "_usage": {"prompt_tokens": 10, "completion_tokens": 5,
                            "total_tokens": 15},
                 "_finish_reason": "stop"}
    client = ScriptedClient([text_turn, final_msg()])
    sandbox = FakeSandbox()
    result = make_runner(client, sandbox, tmp_path).run_task(TASK, DATA)
    assert result["protocol"] == "text"
    assert result["tool_calls"] == 1
    assert sandbox.calls[0]["code"].strip() == "print(6*7)"
    second = client.calls[1]["messages"]
    assert "<tool_output>" in second[-1]["content"]
    assert "42" in second[-1]["content"]


def test_text_protocol_explicit_mode_sends_no_tools(tmp_path):
    text_turn = {"role": "assistant", "content": "```python\nprint(1)\n```"}
    client = ScriptedClient([text_turn, final_msg()])
    result = make_runner(client, tmp_path=tmp_path,
                         protocol="text").run_task(TASK, DATA)
    assert result["protocol"] == "text"
    assert client.calls[0]["tools"] is None


def test_pluggable_non_openai_client(tmp_path):
    """A client returning bare content dicts (raw-sampler shape) works."""

    class RawClient:
        provider = "raw"
        def __init__(self):
            self.n = 0

        def chat(self, messages, *, tools=None, tool_choice=None,
                 max_tokens=None):
            self.n += 1
            if self.n == 1:
                return {"role": "assistant",
                        "content": "```python\nprint(3)\n```"}
            return {"role": "assistant", "content": FINAL}

    runner = make_runner(RawClient(), tmp_path=tmp_path)
    result = runner.run_task(TASK, DATA)
    assert result["response"] == FINAL
    assert result["prompt_tokens"] is None  # no usage reported


# -- run_all_tasks integration: resume + scorer -------------------------------

def _tasks_dir(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(1, 4):
        (tasks_dir / f"task_{i:03d}.csv").write_text(DATA)
        tasks.append({"id": f"task_{i:03d}", "type": "return_calculation",
                      "prompt": "p", "data_file": f"task_{i:03d}.csv"})
    (tasks_dir / "tasks.json").write_text(json.dumps(tasks))
    return tasks_dir


class AlternatingClient:
    """Tool call on a fresh conversation, final once a tool reply exists."""

    provider = "fireworks"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, tools=None, tool_choice=None, max_tokens=None):
        self.calls.append(copy.deepcopy(messages))
        if any(m.get("role") == "tool" for m in messages):
            return final_msg()
        return tool_call(call_id=f"call_{len(self.calls)}")


def test_resume_skips_finished_tool_tasks(tmp_path):
    tasks_dir = _tasks_dir(tmp_path)
    out = tmp_path / "out"
    client = AlternatingClient()
    runner = make_runner(client, tmp_path=tmp_path)
    runner.run_all_tasks(tasks_dir, out)
    calls_after_first = len(client.calls)
    assert calls_after_first == 6  # 3 tasks x (tool turn + final)

    client2 = AlternatingClient()
    runner2 = make_runner(client2, tmp_path=tmp_path)
    results = runner2.run_all_tasks(tasks_dir, out, resume=True)
    assert client2.calls == []
    assert len(results) == 3
    assert all(r["tool_calls"] == 1 for r in results)
    assert (out / "transcripts" / "task_001.jsonl").exists()


def test_tool_results_score_with_unmodified_scorer(tmp_path):
    tasks_dir = _tasks_dir(tmp_path)
    answers = tmp_path / "answers"
    answers.mkdir()
    for i in range(1, 4):
        (answers / f"task_{i:03d}_answer.json").write_text(json.dumps(
            {"problems_found": [], "mean_return": 0.5}))
    out = tmp_path / "out"
    client = AlternatingClient()
    make_runner(client, tmp_path=tmp_path).run_all_tasks(tasks_dir, out)

    from src.scorer import TaskScorer
    summary = TaskScorer(answers).score_all(out / "results.json")
    assert summary["total_tasks"] == 3
    assert summary["passed"] == 3

    # results.jsonl stays scorer-schema compatible (task_id/response).
    lines = (out / "results.jsonl").read_text().strip().splitlines()
    entry = json.loads(lines[0])
    for key in ("task_id", "response", "turns", "tool_calls",
                "tool_errors", "forced_final", "protocol"):
        assert key in entry


def test_task_runner_base_still_scores_dry_run(tmp_path):
    # Guard against accidental coupling: plain runner untouched by import.
    runner = TaskRunner(dry_run=True)
    result = runner.run_task(
        {"id": "task_001", "type": "return_calculation", "prompt": "p"},
        DATA)
    assert "turns" not in result
