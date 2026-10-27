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
  the browser. It is labelled as a demo everywhere — a Demo mode banner,
  "Simulated demo — no model called" in place of a model name, no
  measured-looking timings, a `[simulated]` stream log, and a
  "Demo pass" score — so it cannot be mistaken for a real model run.
- **Live:** When `META_API_KEY` is set on the server, `POST /api/check`
  calls Muse Spark through the OpenAI-compatible Meta API
  (`META_BASE_URL`, default `https://api.meta.ai/v1`) and returns
  validated findings. The key is read only in `muse_client.py` and never
  reaches the browser (`GET /api/config` exposes only mode + model name).
  If the live call fails, that run completes as clearly-labelled
  simulated demo results instead of hanging or faking a live run.

## Live endpoint abuse limits

`POST /api/check` spends real Meta credits, so it is guarded
(`rate_limit.py`, in-memory — resets on redeploy and per replica,
which fits a single Railway replica but is not a distributed limiter):

- The **model is pinned server-side** (`META_MODEL_NAME`); any model a
  client sends is ignored, so callers cannot pick a pricier model.
- Hard caps: 80 KB request body, 64,000-char CSV, 500 rows, 20 columns;
  `max_tokens` defaults to 4,096 and `META_MAX_TOKENS` is capped at
  8,192; upstream timeout 30s; at most 2 concurrent live calls.
- Per-IP limits (`X-Real-IP`, else first `X-Forwarded-For` hop):
  5/min, 30/hour. Global daily caps: 300 live calls and an estimated
  $5/day spend. Before each call the worst-case cost (full prompt +
  max output tokens at $1.25/$4.25 per 1M) is **reserved** against
  the budget and settled afterwards — successful calls settle at the
  actual token usage, failed calls settle at the reserved worst case,
  so billed-but-malformed responses cannot slip past the cap. The
  client uses `max_retries=0`, so one request is one billed attempt.
  Tune with `LIVE_RATE_PER_MINUTE`, `LIVE_RATE_PER_HOUR`,
  `LIVE_DAILY_MAX_CALLS`, `LIVE_DAILY_BUDGET_USD`. A value of `0`
  is a hard block (no live calls), not unlimited. Violations get
  `429` + `Retry-After`.
- **The daily counters reset on every redeploy** (and at UTC
  midnight): they live in process memory, so a redeploy restarts the
  $5/day budget from zero. Keep that in mind when redeploying often;
  the per-IP windows and the upstream account limits still apply.
- Cross-origin browser calls are rejected (`403`) unless the origin is
  same-host or listed in `ALLOWED_ORIGINS`. Non-browser clients can
  spoof `Origin`, so the rate/budget caps above are the real control.

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
3. In the service settings, set the **config file path to
   `/app/railway.toml`**. Without this, Railway silently ignores
   `app/railway.toml` and the `healthcheckPath = "/healthz"` healthcheck
   never applies — the service can look "up" on its port while Railway
   has no health signal.
4. Railway builds the `Dockerfile` and starts
   `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Set **Watch Paths** to `/app/**` so commits that only touch the
   repo root (`src/`, `tests/`, …) don't redeploy this service.
6. Set service **Variables** (never commit these):
   - `META_API_KEY` — required for live mode; omit for simulated mode
   - `META_BASE_URL` — optional, default `https://api.meta.ai/v1`
   - `META_MODEL_NAME` — optional, default `muse-spark-1.3`
   - `ALLOWED_ORIGINS=https://meta.lanemint.com` — the public origin
     allowed to call `/api/check` from a browser
   - Optional limit tuning: `LIVE_RATE_PER_MINUTE`,
     `LIVE_RATE_PER_HOUR`, `LIVE_DAILY_MAX_CALLS`,
     `LIVE_DAILY_BUDGET_USD`
7. Deploy, then verify in the deployment details/logs that the
   healthcheck on `/healthz` was actually applied (not just that the
   port answers), and confirm `https://<your-domain>/healthz` returns
   `{"status": "ok", "mode": "live"}` (or `"simulated"` without a key).

Alternatively, from this directory with the Railway CLI:
`railway up` (add `--new` / `--name` as needed for a fresh project).

## Layout

- `main.py` — FastAPI app: static page, `/healthz`, `/api/config`, `/api/check`
- `muse_client.py` — server-side Muse client + findings validation
- `rate_limit.py` — in-memory per-IP rate limit, daily call/spend budget, concurrency cap
- `static/` — page, LaneMint design tokens (`css/`), ported demo logic (`js/app.js`)
- `tests/test_api.py` — backend tests (health, config secrecy, fallback, live stub)
