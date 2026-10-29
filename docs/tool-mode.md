# Tool mode

Tool mode is an opt-in variant of the benchmark runner. Without
`--tools`, the runner sends exactly the same requests as before. With
`--tools python`, the model gets one sandboxed Python tool,
`run_python(code)`, so it can compute answers from the task CSV
instead of estimating numbers mentally. The scorer, tier generators,
tasks and answer files are unchanged, and final answers are still the
assistant's plain JSON content, scored the usual way.

## How a task runs

- The prompt is the unchanged task prompt plus a short preamble: the
  CSV is at `/data/task.csv`, with its header, row count and first
  lines; pandas and numpy are available; values should be computed
  with the tool and reported at full precision. The CSV is not pasted
  into the prompt.
- Multi-file (`tools_frontier`) tasks instead bind the whole task
  directory read-only at `/data/` (the `spec.md` methodology memo plus
  its data files, e.g. `/data/fx.csv`); the preamble lists the files
  with their sizes and no file contents are pasted into the prompt.
- The model makes at most one tool call per turn, up to `--max-turns`
  tool-using turns. After that (or when a guard trips) the harness
  makes one forced-final call and records `forced_final: true`.
- Each `run_python` call is a fresh bubblewrap process: no network;
  the child starts with an empty environment (`env -i`), and the
  host-side launcher chain (bubblewrap is PID 1 inside the sandbox,
  so its own `/proc/1/environ` matters) is itself started with a
  fixed `PATH`/`LANG`-only environment, with bubblewrap's options
  passed over an fd rather than argv, so no host secret value or
  host path is readable via `/proc/*/environ`, `/proc/*/cmdline` or
  open fds. Read-only system/Python binds (interpreter directories
  outside the system prefix, such as a venv or user site-packages,
  are bound at neutral `/opt/py` paths, and `PYTHONPATH` names only
  in-sandbox paths), only that task's CSV bound at `/data/task.csv`,
  everything except a per-task `/tmp` mounted read-only, and
  wall-clock/CPU/memory/process/file-size limits. Output is
  truncated head+tail. The sandbox never binds the repository,
  `answers/`, `tests/`, `src/`, the home directory or the workspace.
- Results stay scorer-compatible: `results.jsonl`/`results.json` keep
  the usual fields and add `turns`, `tool_calls`, `tool_errors`,
  `tool_timeouts`, `forced_final` and `protocol`. Full per-task
  transcripts (messages, tool code, tool outputs) are written to
  `<output-dir>/transcripts/<task_id>.jsonl`. Resume is per task: an
  interrupted task is rerun from scratch.

Guards: an estimated-prompt-token context guard (default 90,000, or
200,000 for Spark-named models; override with
`--context-token-limit`), a per-task wall-clock cap, per-call HTTP
timeouts with no SDK retries, and an optional per-call cost guard
(`--cost-cap-usd` together with `--input-cost-per-mtok` /
`--output-cost-per-mtok`), checked before every non-final call using
the worst-case output-token cost.

## Running it

Set the API key through the environment (for example `META_API_KEY`)
as usual; never put keys in commands, files or transcripts.

Fireworks (serverless or an on-demand deployment; both are
OpenAI-compatible). The forced final uses `tool_choice="none"`:

```bash
python -m src.runner --tier frontier --tools python \
  --max-turns 12 --sandbox-timeout 30 \
  --base-url https://api.fireworks.ai/inference/v1 \
  --model <model-id> --output-dir output/tool-fireworks
```

Meta API (Spark-class models). This endpoint supports only
`tool_choice="auto"`, so the forced final instead drops the tools and
sends a "return the final JSON answer now" user message; the harness
replays the full assistant tool-call messages, which this API
requires:

```bash
python -m src.runner --tier frontier --tools python \
  --max-turns 12 --sandbox-timeout 30 \
  --base-url https://api.meta.ai/v1 \
  --model <model-id> --output-dir output/tool-meta
```

Any other OpenAI-compatible endpoint works the same way via
`--base-url`/`--model`; unknown endpoints get the conservative
forced-final behaviour. Score tool-mode output exactly like a normal
run:

```bash
python -m src.scorer output/tool-meta/results.json tasks_frontier/answers
```

Useful flags: `--tool-protocol text` makes the model compute by
writing one fenced python code block per turn (the harness replies in
a `<tool_output>` message and records `protocol: "text"`); the
default `auto` accepts that form as a fallback alongside native tool
calls. `--dry-run --tools python` exercises the wiring with the stub
model and makes no API calls. The agent loop only depends on a small
`chat(messages, *, tools, tool_choice, max_tokens)` interface, so a
non-OpenAI client can be plugged into `ToolTaskRunner` from outside
this repository by passing `chat_client=...`.

Sandbox tests skip themselves when bubblewrap user namespaces are
unavailable; on Ubuntu CI runners the workflow enables them and
installs `bubblewrap`. All loop tests run offline against scripted
fake clients.
