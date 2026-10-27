"""Muse Data Check — FastAPI service for Railway.

Serves the static demo page and, when META_API_KEY is configured, a live
Muse Spark endpoint. Without a key the frontend uses simulated responses
(the prototype's default mode); the key never reaches the browser.

The live endpoint is abuse-guarded (see rate_limit.py): the model is
pinned server-side, request/CSV sizes are hard-capped, per-IP and daily
call/spend limits apply, and cross-origin browser calls are rejected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:  # Local development convenience; Railway uses real env vars.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from muse_client import (
    MuseBadRequest,
    MuseNotConfigured,
    MuseUpstreamError,
    check_csv,
    live_config,
    validate_csv,
)
from rate_limit import guard

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_BODY_BYTES = 80_000

app = FastAPI(title="Muse Data Check", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "mode": live_config()["mode"]}


@app.get("/api/config")
def config() -> dict:
    """Secret-free config so the UI can show live vs simulated mode."""
    return live_config()


def _origin_allowed(request: Request) -> bool:
    """Reject cross-origin browser calls to the paid endpoint.

    Same-origin fetches send an Origin whose host matches Host. Origins
    in ALLOWED_ORIGINS (comma-separated) are also accepted. Requests
    with no Origin header (curl, health tooling) are not blocked here —
    the rate/budget limits are the control for non-browser clients.
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    allowed = {
        item.strip().rstrip("/")
        for item in os.getenv("ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    }
    if origin.rstrip("/") in allowed:
        return True
    origin_host = urlparse(origin).netloc
    return bool(origin_host) and origin_host == (request.headers.get("host") or "")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


async def _read_json_payload(request: Request) -> tuple[dict | None, JSONResponse | None]:
    """Read the request body with a hard size bound, then parse JSON.

    Streaming the body (rather than trusting Content-Length) means
    chunked/header-less requests cannot bypass the cap and balloon
    memory. Returns (payload, None) or (None, error_response).
    """
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
        return None, JSONResponse({"error": "Request body too large"}, status_code=413)
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            return None, JSONResponse({"error": "Request body too large"}, status_code=413)
        chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JSONResponse({"error": "Request body must be JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return None, JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)
    return payload, None


@app.post("/api/check")
async def api_check(request: Request) -> JSONResponse:
    if not _origin_allowed(request):
        return JSONResponse({"error": "Cross-origin calls are not allowed"}, status_code=403)

    payload, error = await _read_json_payload(request)
    if error is not None:
        return error
    assert payload is not None

    filename = str(payload.get("filename") or "upload.csv")[:200]
    csv_text = payload.get("csv")
    if not isinstance(csv_text, str) or not csv_text.strip():
        return JSONResponse({"error": "Field 'csv' must be a non-empty string"}, status_code=400)
    # Validate the CSV shape before taking any live quota, so malformed
    # uploads rejected with 413 never consume rate/budget capacity.
    try:
        validate_csv(csv_text)
    except MuseBadRequest as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    # Any client-supplied "model" is ignored: the model is pinned by
    # META_MODEL_NAME on the server so callers cannot pick a pricier one.

    if not live_config()["live"]:
        return JSONResponse(
            {"error": "Live mode is not configured; use simulated responses", "mode": "simulated"},
            status_code=503,
        )

    allowed, retry_after, reason = guard.acquire(_client_ip(request))
    if not allowed:
        return JSONResponse(
            {"error": reason},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    try:
        # check_csv performs a blocking upstream request; run it off the
        # event loop so /healthz and static files stay responsive.
        result = await run_in_threadpool(check_csv, filename, csv_text)
    except MuseBadRequest as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    except MuseNotConfigured:
        return JSONResponse(
            {"error": "Live mode is not configured; use simulated responses", "mode": "simulated"},
            status_code=503,
        )
    except MuseUpstreamError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    finally:
        guard.release()
    guard.record_spend(float(result.pop("estimated_cost_usd", 0.0) or 0.0))
    return JSONResponse(result)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
