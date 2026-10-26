"""Muse Data Check — FastAPI service for Railway.

Serves the static demo page and, when META_API_KEY is configured, a live
Muse Spark endpoint. Without a key the frontend uses simulated responses
(the prototype's default mode); the key never reaches the browser.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:  # Local development convenience; Railway uses real env vars.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from muse_client import MuseNotConfigured, MuseUpstreamError, check_csv, live_config

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Muse Data Check", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "mode": live_config()["mode"]}


@app.get("/api/config")
def config() -> dict:
    """Secret-free config so the UI can show live vs simulated mode."""
    return live_config()


@app.post("/api/check")
async def api_check(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Request body must be JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)

    filename = str(payload.get("filename") or "upload.csv")[:200]
    csv_text = payload.get("csv")
    if not isinstance(csv_text, str) or not csv_text.strip():
        return JSONResponse({"error": "Field 'csv' must be a non-empty string"}, status_code=400)
    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        return JSONResponse({"error": "Field 'model' must be a string"}, status_code=400)

    try:
        result = check_csv(filename, csv_text, model=model)
    except MuseNotConfigured:
        return JSONResponse(
            {"error": "Live mode is not configured; use simulated responses", "mode": "simulated"},
            status_code=503,
        )
    except MuseUpstreamError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return JSONResponse(result)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
