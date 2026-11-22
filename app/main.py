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
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

try:  # Local development convenience; Railway uses real env vars.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from leaderboard import load_leaderboard
from muse_client import (
    MAX_IMAGE_BYTES,
    MuseBadRequest,
    MuseNotConfigured,
    MuseUpstreamError,
    check_chart_image,
    check_csv,
    live_config,
    validate_csv,
    validate_image,
    worst_case_cost_usd,
    worst_case_image_cost_usd,
)
from rate_limit import guard
from sample_answers import match_sample
from sample_charts import list_sample_charts, match_sample_chart

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_BODY_BYTES = 80_000
# Base64 inflates bytes by ~4/3; allow a 5 MB image plus JSON overhead.
MAX_IMAGE_BODY_BYTES = (MAX_IMAGE_BYTES * 4 // 3) + 65_536

# Built-in allowed origins for the new lanemint.tech site, plus localhost for dev.
# ALLOWED_ORIGINS env var (comma-separated) adds extra origins on top of these.
DEFAULT_ALLOWED_ORIGINS = [
    "https://lanemint.tech",
    "https://www.lanemint.tech",
    "http://localhost:3000",
    "http://localhost:8000",
]

app = FastAPI(title="Muse Data Check", docs_url=None, redoc_url=None, openapi_url=None)

# CORS middleware for browser access from lanemint.tech and localhost (local dev).
# The main site's Node server proxies over Railway private networking, which sends
# no Origin header, so CORS is not applied to those requests.
def _get_allowed_origins() -> list[str]:
    """Built-in lanemint.tech origins plus ALLOWED_ORIGINS env var additions."""
    origins = DEFAULT_ALLOWED_ORIGINS.copy()
    extra = os.getenv("ALLOWED_ORIGINS", "").strip()
    if extra:
        origins.extend(item.strip() for item in extra.split(",") if item.strip())
    return origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "mode": live_config()["mode"]}


@app.get("/api/config")
def config() -> dict:
    """Secret-free config so the UI can show live vs simulated mode."""
    return live_config()


@app.post("/api/sample-match")
async def api_sample_match(request: Request) -> JSONResponse:
    """Recognise an upload that exactly matches a public sample.

    The CSV is normalised (BOM stripped, line endings normalised) and
    hashed server-side; the matching sample's public answer key is
    returned only on an exact hash match. Non-matching uploads get
    ``{"matched": false}`` and no key material. This is a read-only
    lookup: it never takes live-model quota.
    """
    if not _origin_allowed(request):
        return JSONResponse({"error": "Cross-origin calls are not allowed"}, status_code=403)
    payload, error = await _read_json_payload(request)
    if error is not None:
        return error
    assert payload is not None
    csv_text = payload.get("csv")
    if not isinstance(csv_text, str) or not csv_text.strip():
        return JSONResponse({"error": "Field 'csv' must be a non-empty string"}, status_code=400)
    matched = match_sample(csv_text)
    if matched is None:
        return JSONResponse({"matched": False})
    return JSONResponse({"matched": True, **matched})


def _origin_allowed(request: Request) -> bool:
    """Reject cross-origin browser calls to the paid endpoint.

    Built-in allowed origins (lanemint.tech, localhost) plus ALLOWED_ORIGINS
    env var additions are accepted. Same-origin fetches (Origin host matches
    Host) are also allowed. Requests with no Origin header (server-to-server
    proxy, curl, health tooling) pass through — the rate/budget limits are
    the control for non-browser clients.
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    # Normalize: strip trailing slash
    origin_normalized = origin.rstrip("/")
    # Check against built-in + env var allowed origins
    allowed = {o.rstrip("/") for o in _get_allowed_origins()}
    if origin_normalized in allowed:
        return True
    # Same-origin fallback
    origin_host = urlparse(origin).netloc
    return bool(origin_host) and origin_host == (request.headers.get("host") or "")


def _valid_ip(value: str) -> str:
    try:
        return str(ip_address(value))
    except ValueError:
        return ""


def _client_ip(request: Request) -> str:
    # Trust model on Railway: the edge proxy sets X-Real-IP to the
    # visitor address it observed, overwriting any client-supplied
    # value on the way through, so that header — not a forwarded-for
    # hop — identifies the visitor. The last X-Forwarded-For hop can
    # be Railway's own edge/internal address, which would bucket every
    # visitor together, so it is never used. Without X-Real-IP, fall
    # back to the first forwarded hop (the claimed original client),
    # then the TCP peer. Header values that are not valid IPs are
    # ignored. If the service were reached directly with no proxy,
    # forwarded headers would be forgeable — the global daily caps in
    # rate_limit.py are the backstop for that case.
    real_ip = _valid_ip((request.headers.get("x-real-ip") or "").strip())
    if real_ip:
        return real_ip
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [_valid_ip(hop.strip()) for hop in forwarded.split(",")]
    hops = [hop for hop in hops if hop]
    if hops:
        return hops[0]
    return request.client.host if request.client else "unknown"


async def _read_json_payload(
    request: Request, max_body_bytes: int = MAX_BODY_BYTES
) -> tuple[dict | None, JSONResponse | None]:
    """Read the request body with a hard size bound, then parse JSON.

    Streaming the body (rather than trusting Content-Length) means
    chunked/header-less requests cannot bypass the cap and balloon
    memory. Returns (payload, None) or (None, error_response).
    """
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_body_bytes:
        return None, JSONResponse({"error": "Request body too large"}, status_code=413)
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_body_bytes:
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

    # Reserve the worst-case cost up front: the reservation is held
    # against the daily budget for the whole call, so billed-but-failed
    # calls (e.g. malformed model output) cannot slip past the cap.
    # When pricing is not configured the cost is unknown (None): no
    # spend is reserved and the guard enforces call-count limits only.
    worst_case = worst_case_cost_usd(csv_text)
    reserve_usd = worst_case if worst_case is not None else 0.0
    allowed, retry_after, reason = guard.acquire(_client_ip(request), reserve_usd)
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
        guard.settle(reserve_usd, reserve_usd)
        return JSONResponse({"error": str(exc)}, status_code=413)
    except MuseNotConfigured:
        guard.settle(reserve_usd, reserve_usd)
        return JSONResponse(
            {"error": "Live mode is not configured; use simulated responses", "mode": "simulated"},
            status_code=503,
        )
    except MuseUpstreamError as exc:
        # The upstream may have billed this attempt; settle at the
        # reserved worst case rather than letting it vanish from the
        # daily spend counter. exc text is generic by construction.
        guard.settle(reserve_usd, reserve_usd)
        return JSONResponse({"error": str(exc)}, status_code=502)
    finally:
        guard.release()
    guard.settle(reserve_usd, float(result.pop("estimated_cost_usd", 0.0) or 0.0))
    return JSONResponse(result)


@app.get("/api/sample-charts")
def sample_charts() -> dict:
    """Bundled sample chart images the simulated mode can recognise."""
    return {"samples": list_sample_charts()}


@app.post("/api/check-image")
async def api_check_image(request: Request) -> JSONResponse:
    """Chart/table image check via Muse Spark image input (live), or
    the clearly-labelled simulated result for a bundled sample image
    when no key is configured. Same guards as /api/check: origin check,
    hard size caps, per-IP/daily caps via the shared guard, model pinned
    server-side, key never reaches the browser."""
    if not _origin_allowed(request):
        return JSONResponse({"error": "Cross-origin calls are not allowed"}, status_code=403)

    payload, error = await _read_json_payload(request, MAX_IMAGE_BODY_BYTES)
    if error is not None:
        return error
    assert payload is not None

    filename = str(payload.get("filename") or "chart.png")[:200]
    media_type = payload.get("media_type")
    encoded = payload.get("image_base64")
    if not isinstance(encoded, str) or not encoded.strip():
        return JSONResponse(
            {"error": "Field 'image_base64' must be a non-empty string"}, status_code=400
        )
    if not isinstance(media_type, str) or not media_type.strip():
        return JSONResponse(
            {"error": "Field 'media_type' must be a non-empty string"}, status_code=400
        )
    import base64
    import binascii

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return JSONResponse({"error": "image_base64 is not valid base64"}, status_code=400)
    # Validate before taking any live quota or simulated lookup.
    try:
        validate_image(image_bytes, media_type)
    except MuseBadRequest as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)

    if not live_config()["live"]:
        simulated = match_sample_chart(image_bytes)
        if simulated is not None:
            return JSONResponse(simulated)
        return JSONResponse(
            {
                "error": "Live mode is not configured; simulated image results "
                "are only available for the bundled sample charts",
                "mode": "simulated",
            },
            status_code=503,
        )

    worst_case = worst_case_image_cost_usd(len(image_bytes))
    reserve_usd = worst_case if worst_case is not None else 0.0
    allowed, retry_after, reason = guard.acquire(_client_ip(request), reserve_usd)
    if not allowed:
        return JSONResponse(
            {"error": reason},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    try:
        result = await run_in_threadpool(
            check_chart_image, filename, image_bytes, media_type
        )
    except MuseBadRequest as exc:
        guard.settle(reserve_usd, reserve_usd)
        return JSONResponse({"error": str(exc)}, status_code=413)
    except MuseNotConfigured:
        guard.settle(reserve_usd, reserve_usd)
        simulated = match_sample_chart(image_bytes)
        if simulated is not None:
            return JSONResponse(simulated)
        return JSONResponse(
            {"error": "Live mode is not configured; use simulated responses",
             "mode": "simulated"},
            status_code=503,
        )
    except MuseUpstreamError as exc:
        guard.settle(reserve_usd, reserve_usd)
        return JSONResponse({"error": str(exc)}, status_code=502)
    finally:
        guard.release()
    guard.settle(reserve_usd, float(result.pop("estimated_cost_usd", 0.0) or 0.0))
    return JSONResponse(result)


@app.get("/api/leaderboard")
def leaderboard() -> JSONResponse:
    """Held-out frontier leaderboard (data/leaderboard.json).

    Totals and Wilson intervals are recomputed server-side from the
    per-family counts in the data file, so the JSON file is the only
    place numbers are edited.
    """
    try:
        return JSONResponse(load_leaderboard())
    except ValueError:
        return JSONResponse(
            {"error": "Leaderboard data is unavailable"}, status_code=500)


@app.get("/", response_class=FileResponse)
def index():
    """Serve the demo page, or redirect when DEMO_REDIRECT_URL is set."""
    redirect_url = os.getenv("DEMO_REDIRECT_URL", "").strip()
    if redirect_url:
        return RedirectResponse(url=redirect_url, status_code=302)
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    # HOST env var: '::' for Railway (dual-stack IPv4/IPv6), '0.0.0.0' for local.
    # Default '::' works for Railway private networking and public access.
    host = os.getenv("HOST", "::")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
