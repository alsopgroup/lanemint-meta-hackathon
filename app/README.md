# Muse Data Check — Web App

A web demo of the LaneMint × Meta benchmark: Muse Spark reads a CSV of
market data, flags duplicates, gaps, bad prices and unadjusted splits,
explains each one in plain English, and a deterministic scorer checks
its answer. Built from the Muse Data Check design prototype.

This directory is self-contained and is the **Root Directory of its own
Railway service** — it does not import anything from the repo root.

## Modes

- **Simulated (default):** No API key configured. The page replays the
  prototype's seeded sample findings and upload heuristics entirely in
  the browser, and shows an amber "simulated responses" indicator.
- **Live:** When `META_API_KEY` is set on the server, `POST /api/check`
  calls Muse Spark through the OpenAI-compatible Meta API
  (`META_BASE_URL`, default `https://api.meta.ai/v1`) and returns
  validated findings. The key is read only in `muse_client.py` and never
  reaches the browser (`GET /api/config` exposes only mode + model name).
  If the live call fails, the UI falls back to simulated responses.

## Run locally

```bash
cd app
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add META_API_KEY for live mode
uvicorn main:app --reload --port 8000
# open http://localhost:8000  ·  health: http://localhost:8000/healthz
```

## Tests

```bash
cd app
pip install -r requirements.txt pytest httpx
pytest tests/ -v
```

The live Muse call is stubbed in tests; no key or network is needed.

## Deploy on Railway

1. Create a **new service** in your Railway project from this repo.
2. Set the service **Root Directory** to `app`.
3. Railway detects `railway.toml` / `Dockerfile`; the service builds the
   image and starts `uvicorn main:app --host 0.0.0.0 --port $PORT`.
   Healthchecks hit `GET /healthz`.
4. Set service **Variables** (never commit these):
   - `META_API_KEY` — required for live mode; omit for simulated mode
   - `META_BASE_URL` — optional, default `https://api.meta.ai/v1`
   - `META_MODEL_NAME` — optional, default `muse-spark-1.3`
5. Deploy and confirm `https://<your-domain>/healthz` returns
   `{"status": "ok", "mode": "live"}` (or `"simulated"` without a key).

Alternatively, from this directory with the Railway CLI:
`railway up` (add `--new` / `--name` as needed for a fresh project).

## Layout

- `main.py` — FastAPI app: static page, `/healthz`, `/api/config`, `/api/check`
- `muse_client.py` — server-side Muse client + findings validation
- `static/` — page, LaneMint design tokens (`css/`), ported demo logic (`js/app.js`)
- `tests/test_api.py` — backend tests (health, config secrecy, fallback, live stub)
