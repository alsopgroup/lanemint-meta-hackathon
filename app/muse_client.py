"""Server-side Muse Spark client for the Muse Data Check web app.

Mirrors the OpenAI-compatible Meta API usage in ``src/runner.py`` at the
repo root, but asks the model for the demo's findings format instead of
benchmark answers. The API key is only ever read here, on the server.
"""

from __future__ import annotations

import json
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
# Meta API list pricing used only to estimate daily spend for the cap.
INPUT_USD_PER_TOKEN = 1.25 / 1_000_000
OUTPUT_USD_PER_TOKEN = 4.25 / 1_000_000

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


def _validate_csv(csv_text: str) -> tuple[str, int]:
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


def _max_tokens() -> int:
    raw = os.getenv("META_MAX_TOKENS")
    if not raw:
        return DEFAULT_MAX_TOKENS
    try:
        return max(256, min(int(raw), HARD_MAX_TOKENS))
    except ValueError:
        return DEFAULT_MAX_TOKENS


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
        raise MuseUpstreamError(f"Model did not return valid JSON: {exc}") from exc

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
    truncated, num_rows = _validate_csv(csv_text)

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MuseUpstreamError("openai package is not installed") from exc

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
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
        raise MuseUpstreamError(f"Muse request failed: {exc}") from exc
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
    estimated_cost = (
        prompt_tokens * INPUT_USD_PER_TOKEN
        + completion_tokens * OUTPUT_USD_PER_TOKEN
    )
    return {
        "findings": findings,
        "latency_ms": latency_ms,
        "model": model_name,
        "estimated_cost_usd": round(estimated_cost, 6),
    }
