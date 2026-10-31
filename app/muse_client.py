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
DEFAULT_MAX_TOKENS = 8192
HARD_MAX_TOKENS = 16384
# muse-spark-1.3 is a reasoning model: without an explicit effort it can
# spend the whole completion budget on reasoning and return no content
# (finish_reason="length", message.content=None). "low" keeps that
# bounded; see _reasoning_effort() and META_REASONING_EFFORT in README.md.
DEFAULT_REASONING_EFFORT = "low"
ALLOWED_REASONING_EFFORTS = {"low", "medium", "high"}
# Explicit opt-out values omit reasoning_effort entirely, for
# non-reasoning models that would reject the parameter.
REASONING_EFFORT_OFF_VALUES = {"none", "off"}
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


def _reasoning_effort() -> str | None:
    """Reasoning effort for chat.completions.create, or None to omit it.

    From META_REASONING_EFFORT (case-insensitive): "low" (default),
    "medium" or "high". Invalid or empty values fall back to "low".
    An explicit "none"/"off" returns None so the parameter is omitted
    entirely, for non-reasoning models that would reject it.
    """
    raw = (os.getenv("META_REASONING_EFFORT") or "").strip().lower()
    if raw in REASONING_EFFORT_OFF_VALUES:
        return None
    if raw in ALLOWED_REASONING_EFFORTS:
        return raw
    return DEFAULT_REASONING_EFFORT


def _reasoning_tokens(usage: object) -> object:
    """Reasoning-token count from usage, when the API reports it.

    Read defensively (attribute access only) from
    usage.completion_tokens_details.reasoning_tokens, falling back to
    usage.reasoning_tokens; returns None when neither is present.
    """
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    tokens = getattr(details, "reasoning_tokens", None)
    if tokens is None:
        tokens = getattr(usage, "reasoning_tokens", None)
    return tokens


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
        # Fall back to the outermost {...} object in the text, so a
        # bare JSON object wrapped in leading/trailing prose still
        # parses. Only the extracted slice is retried, never logged.
        parsed = None
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            # Type only: parse-error text can include a snippet of the
            # response body, which may echo credentials.
            print(f"[_parse_findings] {_safe_exc_summary(exc)}")
            raise MuseUpstreamError("Model did not return valid JSON") from exc

    # Tolerate a top-level JSON list: it is the findings list itself.
    if isinstance(parsed, list):
        raw = parsed
    elif isinstance(parsed, dict):
        raw = parsed.get("findings")
    else:
        raw = None
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
    # 90s timeout: a reasoning call can take 25s+ end to end.
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=90.0, max_retries=0)
    started = time.perf_counter()
    # reasoning_effort bounds how much of the token budget the
    # reasoning model spends thinking before it answers; omit the
    # parameter entirely when META_REASONING_EFFORT=none/off.
    create_kwargs: dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Filename: {filename or 'upload.csv'}\n"
                    f"Data rows: {num_rows}\n\nCSV:\n{truncated}"
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": _max_tokens(),
    }
    reasoning_effort = _reasoning_effort()
    if reasoning_effort is not None:
        create_kwargs["reasoning_effort"] = reasoning_effort
    try:
        response = client.chat.completions.create(**create_kwargs)
    except Exception as exc:  # upstream/network errors become a clean 502
        # Type + status only: exception text can echo request headers
        # (including credentials) and must never be written to logs.
        print(f"[check_csv] {_safe_exc_summary(exc)}")
        raise MuseUpstreamError("Muse request failed") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)

    content = ""
    finish_reason = None
    if response.choices:
        content = response.choices[0].message.content or ""
        finish_reason = getattr(response.choices[0], "finish_reason", None)
    if not content.strip():
        # A reasoning model that exhausts its token budget returns no
        # content at all. Log only safe metadata (no body, no headers,
        # no key) and fail with a message that says what happened.
        reasoning_tokens = _reasoning_tokens(getattr(response, "usage", None))
        print(
            f"[check_csv] empty content finish_reason={finish_reason} "
            f"reasoning_tokens={reasoning_tokens if reasoning_tokens is not None else 'unknown'}"
        )
        if finish_reason == "length":
            raise MuseUpstreamError("Model ran out of tokens before answering")
        raise MuseUpstreamError("Model returned an empty response")
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


# ---------------------------------------------------------------------------
# Chart-image check (multimodal Muse Spark)
#
# Muse Spark 1.3 accepts image input through the same Chat Completions
# path used above: a ``user`` message whose content is a list of parts —
# a ``text`` part plus an ``image_url`` part carrying a base64 data URL
# (Meta Model API docs: "Image understanding" and the "Vision input"
# cookbook, https://dev.meta.ai/docs/image-understanding). Images go in
# user messages only. We inline base64 so user uploads never need a
# public URL. No other model/endpoint is substituted.
# ---------------------------------------------------------------------------

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_SERIES_POINTS = 120
MAX_SUMMARY_CHARS = 600
_IMAGE_MEDIA_TYPES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/webp": (b"RIFF",),  # also requires the WEBP tag; checked below
    "image/gif": (b"GIF87a", b"GIF89a"),
}

_IMAGE_SYSTEM_PROMPT = """You are a financial data quality analyst. You inspect an image of a price chart, statement, or data table and report visible data quality problems, and extract the visible series when you can read it.

Return valid JSON only, with exactly this shape:
{"summary": "One or two plain-English sentences.", "findings": [{"type": "prices", "title": "One-day spike", "explain": "Plain-English explanation of what is visibly wrong."}], "series": [{"date": "2024-01-02", "value": 101.25}]}

Rules:
- "type" must be one of: "prices" (one-day spikes, bad ticks, zero/negative prints), "gaps" (visible gaps in the series, flat/stale stretches with no movement), "splits" (a sustained level shift by a clean ratio, like an unadjusted split), "dups" (visibly repeated/duplicated bars or rows).
- Report only problems actually visible in the image. If the series looks clean, return an empty findings list.
- "series" is your best reading of the plotted/tabulated values (date or label plus numeric value), in order, at most 120 points. If you cannot read values reliably, return an empty list rather than inventing numbers.
- Do not return anything except the JSON object."""


def validate_image(image_bytes: bytes, media_type: str) -> str:
    """Enforce image caps and magic bytes; returns the normalised media type."""
    media = (media_type or "").split(";")[0].strip().lower()
    if media not in _IMAGE_MEDIA_TYPES:
        raise MuseBadRequest("Image must be PNG, JPEG, WebP, or GIF")
    if not image_bytes:
        raise MuseBadRequest("Image is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise MuseBadRequest(f"Image is too large (max {MAX_IMAGE_BYTES} bytes)")
    if not any(image_bytes.startswith(sig) for sig in _IMAGE_MEDIA_TYPES[media]):
        raise MuseBadRequest("Image bytes do not match the declared image type")
    if media == "image/webp" and image_bytes[8:12] != b"WEBP":
        raise MuseBadRequest("Image bytes do not match the declared image type")
    return media


def _parse_json_object(content: str) -> dict:
    """Parse a model JSON object (fenced/prose-wrapped tolerated)."""
    text = (content or "").strip()
    if "```" in text:
        start = text.find("```") + 3
        if text[start:start + 4].lower() == "json":
            start += 4
        end = text.find("```", start)
        text = text[start:end].strip() if end != -1 else text[start:].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        parsed = None
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            print(f"[_parse_image_analysis] {_safe_exc_summary(exc)}")
            raise MuseUpstreamError("Model did not return valid JSON") from exc
    if not isinstance(parsed, dict):
        raise MuseUpstreamError("Model response is not a JSON object")
    return parsed


def _parse_image_analysis(content: str) -> dict:
    """Validate a chart-image analysis: findings, summary, extracted series."""
    parsed = _parse_json_object(content)
    raw = parsed.get("findings")
    if not isinstance(raw, list):
        raise MuseUpstreamError("Model response has no findings list")
    findings: list[dict] = []
    for item in raw[:MAX_FINDINGS]:
        if not isinstance(item, dict):
            continue
        ftype = str(item.get("type") or "").strip()
        if ftype not in ALLOWED_TYPES:
            continue
        title = re.sub(r"\s+", " ", str(item.get("title") or "Problem")).strip()
        explain = re.sub(
            r"\s+", " ", str(item.get("explain") or item.get("explanation") or "")
        ).strip()
        findings.append(
            {
                "type": ftype,
                "title": (title or "Problem")[:MAX_TITLE_CHARS],
                "explain": explain[:MAX_EXPLAIN_CHARS],
            }
        )
    series: list[dict] = []
    raw_series = parsed.get("series")
    if isinstance(raw_series, list):
        for item in raw_series[:MAX_SERIES_POINTS]:
            if not isinstance(item, dict):
                continue
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            label = re.sub(
                r"\s+", " ", str(item.get("date") or item.get("label") or "")
            ).strip()
            series.append({"date": label[:32], "value": value})
    summary = re.sub(r"\s+", " ", str(parsed.get("summary") or "")).strip()
    return {
        "summary": summary[:MAX_SUMMARY_CHARS],
        "findings": findings,
        "series": series,
    }


def worst_case_image_cost_usd() -> float | None:
    """Worst-case reserve for one image call (cost unknown without pricing)."""
    return _cost_usd(max(1, len(_IMAGE_SYSTEM_PROMPT) // 4), _max_tokens())


def check_chart_image(filename: str, image_bytes: bytes, media_type: str) -> dict:
    """Call Muse Spark (image input) on a chart/table image.

    Mirrors check_csv's protections: model pinned server-side,
    reasoning_effort low by default, the shared max_tokens budget, one
    billed attempt (max_retries=0). Raises MuseNotConfigured with no key.
    """
    import base64

    api_key = os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY")
    if not api_key:
        raise MuseNotConfigured("META_API_KEY is not set")
    media = validate_image(image_bytes, media_type)

    base_url = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
    model_name = os.getenv("META_MODEL_NAME", "muse-spark-1.3")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MuseUpstreamError("openai package is not installed") from exc

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=90.0, max_retries=0)
    data_url = f"data:{media};base64," + base64.b64encode(image_bytes).decode()
    create_kwargs: dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _IMAGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Filename: {filename or 'chart.png'}\n"
                            "Identify visible data problems in this price chart "
                            "or table image and extract the series if readable."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": _max_tokens(),
    }
    reasoning_effort = _reasoning_effort()
    if reasoning_effort is not None:
        create_kwargs["reasoning_effort"] = reasoning_effort
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**create_kwargs)
    except Exception as exc:  # upstream/network errors become a clean 502
        print(f"[check_chart_image] {_safe_exc_summary(exc)}")
        raise MuseUpstreamError("Muse request failed") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)

    content = ""
    finish_reason = None
    if response.choices:
        content = response.choices[0].message.content or ""
        finish_reason = getattr(response.choices[0], "finish_reason", None)
    if not content.strip():
        reasoning_tokens = _reasoning_tokens(getattr(response, "usage", None))
        print(
            f"[check_chart_image] empty content finish_reason={finish_reason} "
            f"reasoning_tokens={reasoning_tokens if reasoning_tokens is not None else 'unknown'}"
        )
        if finish_reason == "length":
            raise MuseUpstreamError("Model ran out of tokens before answering")
        raise MuseUpstreamError("Model returned an empty response")
    result = _parse_image_analysis(content)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = max(1, len(_IMAGE_SYSTEM_PROMPT) // 4)
    if completion_tokens is None:
        completion_tokens = max(1, len(content) // 4)
    estimated_cost = _cost_usd(prompt_tokens, completion_tokens)
    return {
        **result,
        "mode": "live",
        "simulated": False,
        "latency_ms": latency_ms,
        "model": model_name,
        "estimated_cost_usd": (
            round(estimated_cost, 6) if estimated_cost is not None else None
        ),
    }
