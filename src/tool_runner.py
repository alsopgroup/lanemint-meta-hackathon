"""Opt-in tool-mode runner: one sandboxed ``run_python`` tool per task.

The agent loop is provider-agnostic.  Anything implementing

    chat(messages, *, tools=None, tool_choice=None, max_tokens=None)
        -> assistant-message dict

can be plugged in as ``chat_client`` -- an OpenAI-compatible SDK
adapter (:class:`OpenAIChatClient`), a scripted fake in tests, or an
external raw-token sampler (e.g. one using a model-specific chat
renderer) that simply returns ``{"role": "assistant", "content": ...}``
and optionally uses the text protocol below.  Assistant dicts may carry
private ``_usage`` / ``_finish_reason`` keys; everything else is
replayed into the history verbatim, which the Meta API requires (it
returns HTTP 400 when a tool-call history is replayed without the full
assistant message).

Two protocols are supported:

* ``tools``/``auto`` -- native OpenAI-compatible ``tools`` with a single
  ``run_python(code)`` function; at most one call is executed per turn.
* ``text`` (also accepted as a fallback under ``auto``) -- the model
  writes one fenced ```python block, the harness executes it and
  replies with a ``<tool_output>`` user message.  Such runs are
  recorded with ``protocol="text"``.

Results keep every field the plain runner writes (so
``src.scorer`` consumes them unchanged) and add ``turns``,
``tool_calls``, ``tool_errors``, ``tool_timeouts``, ``forced_final``,
``protocol`` and, when pricing is configured, cost fields.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from src.runner import TaskRunner
from src.sandbox import SandboxLimits, run_python

_encoding = None


def _count_tokens(text: str) -> int:
    """tiktoken cl100k_base count when available, else chars/4."""
    global _encoding
    try:
        import tiktoken
        if _encoding is None:
            _encoding = tiktoken.get_encoding("cl100k_base")
        return len(_encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run Python code in a sandbox. The task CSV is read-only at "
            "/data/task.csv; pandas and numpy are available. Prints and "
            "stderr are returned. Stateless per call except files you "
            "write under /tmp, which persist for this task only."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute.",
                }
            },
            "required": ["code"],
        },
    },
}

_TEXT_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_META_HOSTS = ("api.meta.ai",)
_FIREWORKS_HOSTS = ("fireworks.ai",)


def detect_provider(base_url: str | None) -> str:
    """Classify an OpenAI-compatible endpoint for forced-final handling."""
    url = (base_url or "").lower()
    if any(h in url for h in _META_HOSTS):
        return "meta"
    if any(h in url for h in _FIREWORKS_HOSTS):
        return "fireworks"
    return "openai-compatible"


class ChatClient(Protocol):
    """Small provider-agnostic chat interface used by the agent loop."""

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             tool_choice: Any = None, max_tokens: int | None = None) -> dict:
        """Return one assistant message dict (see module docstring)."""
        ...


class OpenAIChatClient:
    """Adapter over an OpenAI SDK client (Fireworks / Meta / generic).

    ``client`` is the SDK object stored on :class:`TaskRunner`
    (``client.chat.completions.create``).  Per-call timeouts are passed
    through when the SDK accepts them; no retries are added here (the
    SDK client is already constructed with ``max_retries=0``).
    """

    def __init__(self, client: Any, *, model_name: str, provider: str,
                 temperature: float = 0.1,
                 reasoning_effort: str | None = None,
                 call_timeout: float | None = 900.0):
        self._client = client
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.call_timeout = call_timeout

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             tool_choice: Any = None, max_tokens: int | None = None) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if tools is not None:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            # One call per turn. Meta only documents tool_choice="auto"
            # and is not sent this extra field.
            if self.provider != "meta":
                kwargs["parallel_tool_calls"] = False
        if self.call_timeout is not None:
            kwargs["timeout"] = self.call_timeout
        # Exactly one API call: never re-issue a request from here, so a
        # provider-side failure can never be silently billed twice.
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        if hasattr(message, "model_dump"):
            raw = message.model_dump()
        else:  # pragma: no cover - plain namespace doubles
            raw = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", None),
                "tool_calls": getattr(message, "tool_calls", None),
            }
        # Replay the full assistant message (role, content and the
        # complete tool_calls, which Meta requires), without sending
        # back SDK-only null fields providers may reject.
        assistant = {k: raw[k] for k in
                     ("role", "content", "tool_calls", "reasoning_content",
                      "name")
                     if raw.get(k) is not None}
        assistant.setdefault("role", "assistant")
        assistant["_finish_reason"] = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        if usage is not None:
            assistant["_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return assistant


class _DryRunChatClient:
    """Chat client backed by the runner's dry-run stub model."""

    provider = "openai-compatible"

    def __init__(self, stub: Any):
        self._stub = stub

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             tool_choice: Any = None, max_tokens: int | None = None) -> dict:
        response = self._stub.chat_completions_create(messages=messages)
        choice = response["choices"][0]
        return {
            "role": "assistant",
            "content": choice["message"]["content"],
            "_finish_reason": choice.get("finish_reason"),
            "_usage": response.get("usage") or {},
        }


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Conservative prompt-size estimate (token count x 1.15)."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += _count_tokens(content)
        elif content is not None:
            total += _count_tokens(json.dumps(content, default=str))
        if msg.get("tool_calls"):
            total += _count_tokens(json.dumps(msg["tool_calls"], default=str))
    return math.ceil(total * 1.15)


def _csv_preview(data: str) -> tuple[str, int, list[str]]:
    """(header, data-row count, first 5 lines) for the tool preamble."""
    lines = data.splitlines()
    header = lines[0] if lines else ""
    return header, max(0, len([ln for ln in lines[1:] if ln.strip()])), lines[:5]


class ToolTaskRunner(TaskRunner):
    """TaskRunner that adds the sandboxed-Python agent loop.

    Only ``run_task`` is replaced; ``run_all_tasks`` (resume,
    results.jsonl streaming, per-family filters) is inherited, with a
    thin override that points transcripts at ``<output>/transcripts``.
    """

    def __init__(self, *args, tools: str | None = "python",
                 max_turns: int = 12, sandbox_timeout: float = 30.0,
                 protocol: str = "auto",
                 context_token_limit: int | None = None,
                 task_timeout_seconds: float = 20 * 60.0,
                 call_timeout: float | None = 900.0,
                 cost_cap_usd: float | None = None,
                 input_cost_per_mtok: float = 0.0,
                 output_cost_per_mtok: float = 0.0,
                 tasks_dir: Path | None = None,
                 transcript_dir: Path | None = None,
                 chat_client: Any = None, sandbox_fn: Any = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if tools not in (None, "python"):
            raise ValueError(f"[ToolTaskRunner] unsupported --tools value: {tools!r}")
        if protocol not in ("auto", "tools", "text"):
            raise ValueError(f"[ToolTaskRunner] unsupported protocol: {protocol!r}")
        self.tools_enabled = tools == "python"
        self.max_turns = max(0, int(max_turns))
        self.sandbox_timeout = float(sandbox_timeout)
        self.protocol_preference = protocol
        self.context_token_limit = context_token_limit
        self.task_timeout_seconds = float(task_timeout_seconds)
        self.call_timeout = call_timeout
        self.cost_cap_usd = cost_cap_usd
        self.input_cost_per_mtok = float(input_cost_per_mtok)
        self.output_cost_per_mtok = float(output_cost_per_mtok)
        self.tasks_dir = Path(tasks_dir) if tasks_dir else None
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self._sandbox_fn = sandbox_fn or run_python
        self._spent_usd = 0.0
        if chat_client is not None:
            self._chat_client = chat_client
        elif self.dry_run:
            self._chat_client = _DryRunChatClient(self.client)
        else:
            self._chat_client = OpenAIChatClient(
                self.client, model_name=self.model_name,
                provider=detect_provider(self.base_url),
                reasoning_effort=self._effective_reasoning_effort(),
                call_timeout=call_timeout)

    # -- inherited plumbing -------------------------------------------------

    def load_data(self, tasks_dir: Path, data_file: str) -> str:
        # Remember where task CSVs live so run_task can bind exactly
        # the one file named by the task being run.
        self.tasks_dir = Path(tasks_dir)
        return super().load_data(tasks_dir, data_file)

    def run_all_tasks(self, *args, **kwargs):
        output_dir = kwargs.get("output_dir") or (
            args[1] if len(args) > 1 else Path("output"))
        # Transcripts always live beside the results they produced;
        # direct run_task calls use the configured/default directory.
        self.transcript_dir = Path(output_dir) / "transcripts"
        return super().run_all_tasks(*args, **kwargs)

    # -- helpers ------------------------------------------------------------

    @property
    def _provider(self) -> str:
        return getattr(self._chat_client, "provider",
                       detect_provider(self.base_url))

    def _context_limit(self) -> int:
        if self.context_token_limit is not None:
            return int(self.context_token_limit)
        # Spark-class endpoints get the larger guard; everything else
        # uses the smaller (Glimmer-class) guard from the scope.
        return 200_000 if "spark" in (self.model_name or "").lower() else 90_000

    def _call_cost(self, usage: dict) -> float:
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        return (prompt / 1e6 * self.input_cost_per_mtok
                + completion / 1e6 * self.output_cost_per_mtok)

    def _worst_case_next_cost(self, messages: list[dict], task: dict) -> float:
        est_in = estimate_prompt_tokens(messages)
        return (est_in / 1e6 * self.input_cost_per_mtok
                + self._max_tokens_for(task) / 1e6 * self.output_cost_per_mtok)

    def _resolve_csv(self, task: dict, data: str,
                     tmp_root: Path) -> Path:
        if self.tasks_dir is not None and task.get("data_file"):
            candidate = self.tasks_dir / task["data_file"]
            if candidate.is_file():
                return candidate
        # Direct run_task calls (tests, external drivers): materialise
        # the already-loaded CSV under a task_*.csv name in a temp dir
        # outside the repo, and bind only that copy.
        name = f"{task.get('id', 'task')}.csv"
        if not name.startswith("task_"):
            name = f"task_{name}"
        path = tmp_root / name
        path.write_text(data)
        return path

    def _execute_tool(self, code: str, csv_path: Path, workdir: Path) -> dict:
        limits = SandboxLimits(timeout_seconds=self.sandbox_timeout)
        try:
            result = self._sandbox_fn(code, csv_path, workdir, limits)
        except TypeError:
            # External sandbox doubles with the short (code, csv) shape.
            result = self._sandbox_fn(code, csv_path)
        if not isinstance(result, dict):
            return {"stdout": str(result), "stderr": "", "exit": 0,
                    "timed_out": False, "truncated": False}
        return result

    @staticmethod
    def _tool_payload(result: dict) -> str:
        return json.dumps({
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit": result.get("exit"),
            "timed_out": bool(result.get("timed_out")),
            "truncated": bool(result.get("truncated")),
        })

    def _preamble(self, task: dict, data: str) -> str:
        header, row_count, first_lines = _csv_preview(data)
        lines = [
            task["prompt"],
            "",
            "Tool available: run_python(code) runs Python in a sandbox.",
            "The task data file is at /data/task.csv (read-only). "
            "Do not expect the CSV text in this prompt.",
            f"CSV header: {header}",
            f"CSV data rows (lines after the header, all blocks): {row_count}",
            "First lines of the file:",
            *first_lines,
            "pandas and numpy are available. Compute with the tool and "
            "report values at full precision (round only as the task "
            "states). Files written under /tmp persist for this task.",
        ]
        if self.protocol_preference == "text":
            lines.append(
                "To compute, reply with exactly one fenced ```python "
                "code block and no final answer yet. The result will "
                "come back in a <tool_output> message. When finished, "
                "reply with the final JSON answer only.")
        else:
            lines.append(
                "Make at most one tool call per turn. When finished, "
                "reply with the final JSON answer only.")
        return "\n".join(lines)

    # -- the agent loop -----------------------------------------------------

    def run_task(self, task: dict, data: str) -> dict:
        started = time.monotonic()
        max_tokens = self._max_tokens_for(task)
        messages: list[dict] = [
            {"role": "system",
             "content": "You are a financial data analyst. Analyze the "
                        "provided data and return results as valid JSON only."},
            {"role": "user", "content": self._preamble(task, data)},
        ]
        transcript: list[dict] = [
            {"event": "task_start", "task_id": task["id"],
             "protocol_preference": self.protocol_preference,
             "provider": self._provider}]
        stats = {"turns": 0, "tool_calls": 0, "tool_errors": 0,
                 "tool_timeouts": 0, "forced_final": False,
                 "protocol": "text" if self.protocol_preference == "text"
                 else "tools"}
        totals = {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0}
        saw_usage = False
        finish_reason = None
        content = ""
        spent_at_task_start = self._spent_usd

        def record_usage(assistant: dict) -> None:
            nonlocal saw_usage, finish_reason
            usage = assistant.get("_usage") or {}
            if usage:
                saw_usage = True
                for key in totals:
                    if isinstance(usage.get(key), (int, float)):
                        totals[key] += int(usage[key])
                self._spent_usd += self._call_cost(usage)
            finish_reason = assistant.get("_finish_reason", finish_reason)

        def call_model(forced: bool) -> dict:
            """One chat call; handles the provider-specific forced final."""
            if forced:
                stats["forced_final"] = True
                if self._provider == "meta" or stats["protocol"] == "text" \
                        or self.protocol_preference == "text":
                    # Meta supports only tool_choice="auto": drop the
                    # tools and nudge in a user message instead.
                    messages.append({
                        "role": "user",
                        "content": "No more tool calls. Return the "
                                   "final JSON answer now."})
                    transcript.append({"event": "message",
                                       "message": messages[-1]})
                    return self._chat_client.chat(
                        messages, tools=None, tool_choice=None,
                        max_tokens=max_tokens)
                return self._chat_client.chat(
                    messages, tools=[RUN_PYTHON_TOOL],
                    tool_choice="none", max_tokens=max_tokens)
            if self.protocol_preference == "text":
                return self._chat_client.chat(
                    messages, tools=None, tool_choice=None,
                    max_tokens=max_tokens)
            return self._chat_client.chat(
                messages, tools=[RUN_PYTHON_TOOL], tool_choice="auto",
                max_tokens=max_tokens)

        use_tools = self.tools_enabled
        try:
            with tempfile.TemporaryDirectory(
                    prefix="tool-task-") as tmp_name:
                tmp_root = Path(tmp_name)
                csv_path = self._resolve_csv(task, data, tmp_root) \
                    if use_tools else tmp_root / "task_unused.csv"
                workdir = tmp_root / "work"
                workdir.mkdir(exist_ok=True)

                forced_pending = False
                while True:
                    if not forced_pending:
                        # Guards run before every non-final call.
                        elapsed = time.monotonic() - started
                        over_context = estimate_prompt_tokens(messages) \
                            > self._context_limit()
                        over_cost = self.cost_cap_usd is not None and (
                            self._spent_usd
                            + self._worst_case_next_cost(messages, task)
                            > self.cost_cap_usd)
                        # max_turns counts model turns that may use
                        # the tool; the next call is the forced final.
                        if over_context or over_cost or \
                                elapsed > self.task_timeout_seconds or \
                                stats["turns"] >= self.max_turns:
                            forced_pending = True

                    is_forced = forced_pending
                    assistant = call_model(is_forced)
                    stats["turns"] += 1
                    record_usage(assistant)
                    public = {k: v for k, v in assistant.items()
                              if not k.startswith("_")}
                    public.setdefault("role", "assistant")
                    messages.append(public)
                    transcript.append({"event": "message", "message": public,
                                       "forced": is_forced})
                    content = public.get("content") or ""

                    if is_forced:
                        break

                    tool_calls = public.get("tool_calls") or []
                    if tool_calls and use_tools:
                        stats["protocol"] = "tools"
                        first, extras = tool_calls[0], tool_calls[1:]
                        self._handle_tool_call(
                            first, csv_path, workdir, messages,
                            transcript, stats)
                        for extra in extras:
                            # Every call id needs a tool reply; only the
                            # first call of a turn is ever executed.
                            messages.append({
                                "role": "tool", "tool_call_id":
                                    (extra.get("id") if isinstance(extra, dict)
                                     else None),
                                "content": "Only one tool call per turn is "
                                           "allowed; this call was not "
                                           "executed. Re-issue it next turn "
                                           "if still needed."})
                            transcript.append({
                                "event": "message", "message": messages[-1]})
                        continue

                    if use_tools:
                        block = _TEXT_BLOCK.search(content or "")
                        if block:
                            stats["protocol"] = "text"
                            result = self._execute_tool(
                                block.group(1), csv_path, workdir)
                            stats["tool_calls"] += 1
                            if result.get("timed_out"):
                                stats["tool_timeouts"] += 1
                            payload = self._tool_payload(result)
                            messages.append({
                                "role": "user",
                                "content": f"<tool_output>\n{payload}\n"
                                           "</tool_output>"})
                            transcript.append({
                                "event": "tool_result", "protocol": "text",
                                "code": block.group(1), "result": result})
                            transcript.append({
                                "event": "message", "message": messages[-1]})
                            continue
                    break
        finally:
            transcript.append({
                "event": "task_end", "task_id": task["id"], **stats,
                "finish_reason": finish_reason})
            self._write_transcript(task["id"], transcript)

        result = {
            "task_id": task["id"],
            "task_type": task["type"],
            "response": content,
            "model": self.model_name,
            "max_tokens": max_tokens,
            "reasoning_effort": self._effective_reasoning_effort(),
            "finish_reason": finish_reason,
            "prompt_tokens": totals["prompt_tokens"] if saw_usage else None,
            "completion_tokens": totals["completion_tokens"] if saw_usage
            else None,
            "total_tokens": totals["total_tokens"] if saw_usage else None,
            **stats,
        }
        if self.cost_cap_usd is not None or self.input_cost_per_mtok \
                or self.output_cost_per_mtok:
            result["cost_usd"] = round(
                self._spent_usd - spent_at_task_start, 8)
        return result

    def _handle_tool_call(self, call: dict, csv_path: Path, workdir: Path,
                          messages: list[dict], transcript: list[dict],
                          stats: dict) -> None:
        """Execute (or reject) one native tool call and append its reply."""
        call_id = call.get("id") if isinstance(call, dict) else None
        fn = (call.get("function") or {}) if isinstance(call, dict) else {}
        name = fn.get("name")
        raw_args = fn.get("arguments")
        code = None
        error = None
        if name != "run_python":
            error = f"unknown tool {name!r}; only run_python is available"
        else:
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) \
                    else raw_args
                if not isinstance(args, dict) or \
                        not isinstance(args.get("code"), str):
                    raise ValueError("arguments must be an object with a "
                                     "string 'code'")
                code = args["code"]
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                error = f"malformed run_python arguments: {exc}"

        if error is not None:
            stats["tool_errors"] += 1
            payload = json.dumps({"error": error})
            transcript.append({"event": "tool_error", "error": error,
                               "tool_call_id": call_id})
        else:
            try:
                result = self._execute_tool(code, csv_path, workdir)
            except Exception as exc:  # sandbox unavailable, bad CSV, ...
                stats["tool_errors"] += 1
                payload = json.dumps(
                    {"error": f"run_python failed: {type(exc).__name__}"})
                transcript.append({"event": "tool_error",
                                   "error": type(exc).__name__,
                                   "tool_call_id": call_id, "code": code})
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": payload})
                transcript.append({"event": "message",
                                   "message": messages[-1]})
                return
            stats["tool_calls"] += 1
            if result.get("timed_out"):
                stats["tool_timeouts"] += 1
            payload = self._tool_payload(result)
            transcript.append({"event": "tool_result", "protocol": "tools",
                               "tool_call_id": call_id, "code": code,
                               "result": result})
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": payload})
        transcript.append({"event": "message", "message": messages[-1]})

    def _write_transcript(self, task_id: str, transcript: list[dict]) -> None:
        if self.transcript_dir is None:
            return
        try:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            path = self.transcript_dir / f"{task_id}.jsonl"
            with open(path, "w") as fh:
                for entry in transcript:
                    fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            # A transcript failure must not lose a paid task result.
            print(f"[ToolTaskRunner] could not write transcript for {task_id}")
