"""Server-side Muse Spark client for the Muse Data Check web app.

Mirrors the OpenAI-compatible Meta API usage in ``src/runner.py`` at the
repo root, but asks the model for the demo's findings format instead of
benchmark answers. The API key is only ever read here, on the server.
"""

from __future__ import annotations

import json
import math
import os
import re
import time

ALLOWED_TYPES = {"dups", "gaps", "prices", "splits"}
MAX_FINDINGS = 50
MAX_TITLE_CHARS = 120
MAX_EXPLAIN_CHARS = 600
MAX_CSV_CHARS = 64_000
MAX_CSV_ROWS = 500
MAX_CSV_COLS = 20
DEFAULT_MAX_TOKENS = 4096
HARD_MAX_TOKENS = 8192
# Optional per-token pricing for the daily spend guard, supplied only
# via the environment (never hard-coded in the repo): both
# META_INPUT_COST_PER_MTOK and META_OUTPUT_COST_PER_MTOK must be set to
# valid non-negative numbers (cost per million tokens, in the budget
# currency) for spend to be estimable. When either is unset or invalid,
# cost is unknown (None): no spend estimate is produced and the guard
# in rate_limit.py enforces its call-count limits only — except that a
# budget explicitly set to 0 remains a hard block there. See README.md.

_SYSTEM_PROMPT = """You are a financial data quality analyst. You inspect CSV market data and report data quality problems.

Return valid JSON only, with exactly this shape:
{"findings": [{"type": "dups", "title": "Duplicate bar", "rows": [4, 5], "explain": "Plain-English explanation."}]}

Rules:
- "type" must be one of: "dups" (duplicate rows/bars), "gaps" (missing rows, out-of-order timestamps, stale/repeated quotes), "prices" (bad prices, one-row spikes, fat-finger decimals, high below low, zero/negative prices, zero volume), "splits" (unadjusted splits or reverse splits, i.e. a sustained level change by a clean ratio).
- "rows" are 1-based data-row numbers (the first row under the header is row 1). List every affected row.
- "title" is a short name for the problem. "explain" is one or two plain-English sentences saying what is wrong and what to do about it.
- Report only problems you actually find. If the data is clean, return {"findings": []}.
- Do not invent row numbers. Do not return anything except the JSON object."""


class MuseNotConfigured(RuntimeError):
    """Raised when no Meta API key is configured (simulated fallback)."""


class MuseUpstreamError(RuntimeError):
    """Raised when the upstream Muse call or its parsing fails."""


class MuseBadRequest(ValueError):
    """Raised when the submitted CSV exceeds the hard input caps."""


def live_config() -> dict:
    """Public, secret-free configuration for the frontend."""
    api_key = os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY")
    model = os.getenv("META_MODEL_NAME", "muse-spark-1.3")
    return {"live": bool(api_key), "mode": "live" if api_key else "simulated", "model": model}


def validate_csv(csv_text: str) -> tuple[str, int]:
    """Enforce hard input caps; returns (csv, data_row_count)."""
    if len(csv_text) > MAX_CSV_CHARS:
        raise MuseBadRequest(
            f"CSV is too large (max {MAX_CSV_CHARS} characters)")
    lines = [line for line in csv_text.replace("\r", "").split("\n") if line.strip()]
    if not lines:
        raise MuseBadRequest("CSV is empty")
    if len(lines) - 1 > MAX_CSV_ROWS:
        raise MuseBadRequest(f"CSV has too many rows (max {MAX_CSV_ROWS})")
    if lines[0].count(",") + 1 > MAX_CSV_COLS:
        raise MuseBadRequest(f"CSV has too many columns (max {MAX_CSV_COLS})")
    return "\n".join(lines), len(lines) - 1


def _cost_per_mtok() -> tuple[float, float] | None:
    """(input, output) cost per million tokens from the environment.

    Returns None — cost unknown — unless both pricing variables are
    set to finite, non-negative numbers. No pricing is ever embedded
    in the repo, and invalid values fail closed to "unknown" rather
    than being guessed.
    """
    raw_input = os.getenv("META_INPUT_COST_PER_MTOK")
    raw_output = os.getenv("META_OUTPUT_COST_PER_MTOK")
    if raw_input is None or raw_output is None:
        return None
    try:
        input_cost = float(raw_input)
        output_cost = float(raw_output)
    except ValueError:
        return None
    # float() accepts "nan"/"inf", which would poison budget maths.
    if not math.isfinite(input_cost) or not math.isfinite(output_cost) \
            or input_cost < 0 or output_cost < 0:
        return None
    return input_cost, output_cost


def _cost_usd(input_tokens: int, output_tokens: int) -> float | None:
    """Cost of a token pair, or None when pricing is not configured."""
    pricing = _cost_per_mtok()
    if pricing is None:
        return None
    input_per_mtok, output_per_mtok = pricing
    return (
        input_tokens * input_per_mtok / 1_000_000
        + output_tokens * output_per_mtok / 1_000_000
    )


def worst_case_cost_usd(csv_text: str) -> float | None:
    """Upper-bound cost of one check_csv call, for budget reservation.

    Input tokens are estimated at ~4 chars/token over the whole prompt
    (system prompt + header + CSV); output is capped at _max_tokens().
    With max_retries=0 there is exactly one billed attempt per call.
    Returns None when pricing is not configured via the environment:
    cost is then unknown and callers must not reserve spend against
    the daily budget (call-count limits still apply).
    """
    prompt_chars = len(_SYSTEM_PROMPT) + len(csv_text) + 64
    input_tokens = max(1, prompt_chars // 4)
    return _cost_usd(input_tokens, _max_tokens())


def _max_tokens() -> int:
    raw = os.getenv("META_MAX_TOKENS")
    if not raw:
        return DEFAULT_MAX_TOKENS
    try:
        return max(256, min(int(raw), HARD_MAX_TOKENS))
    except ValueError:
        return DEFAULT_MAX_TOKENS


def _safe_exc_summary(exc: BaseException) -> str:
    """Log only the exception type and HTTP status.

    Exception text (and JSON parse snippets) can echo request headers
    or response bodies that contain credentials; never write those.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"{name} status={status}"
    return name


def _parse_findings(content: str, num_rows: int) -> list[dict]:
    text = (content or "").strip()
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        text = text[start:end].strip() if end != -1 else text[start:].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip() if end != -1 else text[start:].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # Type only: parse-error text can include a snippet of the
        # response body, which may echo credentials.
        print(f"[_parse_findings] {_safe_exc_summary(exc)}")
        raise MuseUpstreamError("Model did not return valid JSON") from exc

    raw = parsed.get("findings") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        raise MuseUpstreamError("Model response has no findings list")

    findings: list[dict] = []
    for item in raw[:MAX_FINDINGS]:
        if not isinstance(item, dict):
            continue
        ftype = str(item.get("type") or item.get("step") or "").strip()
        if ftype not in ALLOWED_TYPES:
            continue
        raw_rows = item.get("rows")
        if isinstance(raw_rows, bool):
            raw_rows = []
        elif isinstance(raw_rows, (int, str)):
            raw_rows = [raw_rows]  # scalar row from the model: wrap, then validate below
        if not isinstance(raw_rows, list):
            raw_rows = []
        rows: list[int] = []
        for value in raw_rows:
            try:
                row = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= row <= max(num_rows, 1):
                rows.append(row)
        title = re.sub(r"\s+", " ", str(item.get("title") or "Problem")).strip()
        explain = re.sub(
            r"\s+", " ", str(item.get("explain") or item.get("explanation") or "")
        ).strip()
        findings.append(
            {
                "type": ftype,
                "title": (title or "Problem")[:MAX_TITLE_CHARS],
                "rows": sorted(set(rows)),
                "explain": explain[:MAX_EXPLAIN_CHARS],
            }
        )
    return findings


def check_csv(filename: str, csv_text: str) -> dict:
    """Call Muse Spark on a CSV and return validated findings.

    The model is pinned server-side (META_MODEL_NAME); callers cannot
    choose it. Raises MuseNotConfigured when no API key is set so the
    caller can tell the frontend to use its simulated fallback.
    """
    api_key = os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY")
    if not api_key:
        raise MuseNotConfigured("META_API_KEY is not set")
    if not csv_text or not csv_text.strip():
        raise MuseBadRequest("CSV is empty")

    base_url = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
    model_name = os.getenv("META_MODEL_NAME", "muse-spark-1.3")
    truncated, num_rows = validate_csv(csv_text)

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MuseUpstreamError("openai package is not installed") from exc

    # max_retries=0: one call = one billed attempt, so the budget
    # reservation in rate_limit.py is a true worst case per request.
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0)
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Filename: {filename or 'upload.csv'}\n"
                        f"Data rows: {num_rows}\n\nCSV:\n{truncated}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=_max_tokens(),
        )
    except Exception as exc:  # upstream/network errors become a clean 502
        # Type + status only: exception text can echo request headers
        # (including credentials) and must never be written to logs.
        print(f"[check_csv] {_safe_exc_summary(exc)}")
        raise MuseUpstreamError("Muse request failed") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)

    content = ""
    if response.choices:
        content = response.choices[0].message.content or ""
    findings = _parse_findings(content, num_rows)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = max(1, len(truncated) // 4)
    if completion_tokens is None:
        completion_tokens = max(1, len(content) // 4)
    # None when pricing is not configured: cost is unknown, not zero.
    estimated_cost = _cost_usd(prompt_tokens, completion_tokens)
    return {
        "findings": findings,
        "latency_ms": latency_ms,
        "model": model_name,
        "estimated_cost_usd": (
            round(estimated_cost, 6) if estimated_cost is not None else None
        ),
    }
