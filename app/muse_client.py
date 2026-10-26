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
MAX_CSV_CHARS = 1_000_000
MAX_CSV_ROWS = 500

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


def live_config() -> dict:
    """Public, secret-free configuration for the frontend."""
    api_key = os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY")
    model = os.getenv("META_MODEL_NAME", "muse-spark-1.3")
    return {"live": bool(api_key), "mode": "live" if api_key else "simulated", "model": model}


def _count_data_rows(csv_text: str) -> int:
    lines = [line for line in csv_text.replace("\r", "").split("\n") if line.strip()]
    return max(0, len(lines) - 1)


def _truncate_csv(csv_text: str) -> str:
    lines = csv_text.replace("\r", "").split("\n")
    return "\n".join(lines[: MAX_CSV_ROWS + 1])


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
        rows: list[int] = []
        for value in item.get("rows") or []:
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


def check_csv(filename: str, csv_text: str, model: str | None = None) -> dict:
    """Call Muse Spark on a CSV and return validated findings.

    Raises MuseNotConfigured when no API key is set so the caller can
    tell the frontend to use its simulated fallback.
    """
    api_key = os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY")
    if not api_key:
        raise MuseNotConfigured("META_API_KEY is not set")
    if not csv_text or not csv_text.strip():
        raise MuseUpstreamError("CSV is empty")
    if len(csv_text) > MAX_CSV_CHARS:
        raise MuseUpstreamError("CSV is too large")

    base_url = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
    model_name = model or os.getenv("META_MODEL_NAME", "muse-spark-1.3")
    truncated = _truncate_csv(csv_text)
    num_rows = _count_data_rows(truncated)

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MuseUpstreamError("openai package is not installed") from exc

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
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
            max_tokens=16000,
        )
    except Exception as exc:  # upstream/network errors become a clean 502
        raise MuseUpstreamError(f"Muse request failed: {exc}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)

    content = ""
    if response.choices:
        content = response.choices[0].message.content or ""
    return {
        "findings": _parse_findings(content, num_rows),
        "latency_ms": latency_ms,
        "model": model_name,
    }
