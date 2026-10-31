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

## Chart image check (multimodal)

The **Chart image check** panel accepts a chart image or screenshot of
a price series (or a statement/table screenshot), PNG/JPEG/WebP/GIF up
to 5 MB. Muse Spark 1.3 accepts image input through the same Meta
Model API Chat Completions path as the CSV check (an `image_url`
content part with a base64 data URL in a `user` message — see Meta's
image-understanding docs); `check_chart_image` in `muse_client.py`
sends exactly that, with the model pinned server-side,
`reasoning_effort` low, and the shared token budget. It returns visible
problems (spikes, gaps, flat stretches, split-like level shifts) and
the extracted series as a table.

- **Live:** `POST /api/check-image` goes through the same guard as
  `/api/check` (origin check, hard size caps, per-IP and daily caps,
  worst-case reservation); the key never reaches the browser.
- **Simulated (no key):** the three bundled sample charts in
  `static/assets/sample-charts/` (generated from the public
  `samples/` data by `scripts/make_sample_charts.py`, manifest in
  `data/sample_charts.json`) are recognised by SHA-256 and return
  clearly labelled simulated results (`"simulated": true`). Any other
  image returns `503` with mode `simulated` — no findings are invented
  for images the demo has not seen.

## Leaderboard

The page's Leaderboard section shows held-out results for both
tiers: the **frontier** tier as a tools-off vs tools-on comparison
(96 tasks; tools take Muse Spark 1.3 from 50% to 99%, so that tier
saturates once tools are on) and the harder **`tools_frontier`** tier
with tools on at the default 12 turns (90 tasks; Spark 1.3 at
63/90 / 70%, ahead of Spark Open at 57% and Glimmer 30B base at 30%).
Approximate rates (e.g. Glimmer base tools-on at ~85%) are shown
exactly as labelled in the data — no exact count is invented for
them. The trace-LoRA run on `tools_frontier` is marked
`status: "partial"` in the data (0/36, not a complete run) and the
page renders it as "partial · 36 of 90 tasks" in place of a full-board
pass rate, so it cannot be read as a completed 0/90 result. All numbers live in the
single data file `data/leaderboard.json` — the frontend hard-codes
nothing. To publish new results, edit that file only: per-family pass
counts are summed server-side by `leaderboard.py`, which also
recomputes the pass rate and Wilson 95% interval and validates the
file's shape on load (`GET /api/leaderboard` returns 500 rather than
serving bad data). Demo mode works without `META_API_KEY`: without a
key the page runs its clearly labelled simulated mode and the
leaderboard still renders from the data file.
No run-cost or spend figures are shown anywhere in the public app;
the daily budget below is an internal server guard only and is never
returned to the browser.

## Live endpoint abuse limits

`POST /api/check` (and `POST /api/check-image`, which shares the same
guard, with its own larger body cap for base64 images) calls the live model, so it is guarded
(`rate_limit.py`, in-memory — resets on redeploy and per replica,
which fits a single Railway replica but is not a distributed limiter):

- The **model is pinned server-side** (`META_MODEL_NAME`); any model a
  client sends is ignored, so callers cannot pick a different model.
- Hard caps: 80 KB request body, 64,000-char CSV, 500 rows, 20 columns;
  `max_tokens` defaults to 8,192 and `META_MAX_TOKENS` is capped at
  16,384; upstream timeout 90s; at most 2 concurrent live calls.
  `META_REASONING_EFFORT` sets the `reasoning_effort` passed to the
  model (`low` by default, or `medium`/`high`; invalid or empty values
  fall back to `low`). Muse Spark is a reasoning model, so without an
  explicit effort it can spend the whole token budget on reasoning and
  return no answer. Set `META_REASONING_EFFORT=none` (or `off`) to omit
  the parameter entirely, e.g. for a non-reasoning model.
- Per-IP limits keyed on `X-Real-IP` (set by Railway's edge proxy to
  the visitor address it observed), falling back to the first
  `X-Forwarded-For` hop, then the TCP peer. The last forwarded hop
  is never used: it can be Railway's own edge address, which would
  bucket every visitor together. 5/min, 30/hour. Global daily caps: 300 live calls and a configurable
  daily budget (`LIVE_DAILY_BUDGET_USD`). Before each call the
  worst-case cost (full prompt + max output tokens) is **reserved**
  against the budget and settled afterwards — successful calls settle at the
  actual token usage, failed calls settle at the reserved worst case,
  so billed-but-malformed responses cannot slip past the cap. The
  client uses `max_retries=0`, so one request is one billed attempt.
  Tune with `LIVE_RATE_PER_MINUTE`, `LIVE_RATE_PER_HOUR`,
  `LIVE_DAILY_MAX_CALLS`, `LIVE_DAILY_BUDGET_USD`. A value of `0`
  is a hard block (no live calls), not unlimited. Violations get
  `429` + `Retry-After`.
- **Pricing is never hard-coded.** Spend estimates use only the
  optional server-side variables `META_INPUT_COST_PER_MTOK` and
  `META_OUTPUT_COST_PER_MTOK` (cost per million tokens, in the same
  currency as `LIVE_DAILY_BUDGET_USD`; set both from your provider's
  current pricing — no figures are recorded in this repo). If either
  variable is unset or invalid, cost is treated as **unknown**: no
  spend is estimated or reserved, the spend-based part of the budget
  guard simply does not engage, and the per-IP plus daily call-count
  limits (`LIVE_DAILY_MAX_CALLS`) still bound live usage. A budget
  explicitly set to `0` remains a hard block even then. To enforce a
  spend ceiling as well as a call-count ceiling, set both pricing
  variables (and `LIVE_DAILY_BUDGET_USD`) on the server.
- **The daily counters reset on every redeploy** (and at UTC
  midnight): they live in process memory, so a redeploy restarts the
  daily budget from zero. Keep that in mind when redeploying often;
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
   - `META_REASONING_EFFORT` — optional, default `low` (`low`/`medium`/`high`;
     `none`/`off` omits `reasoning_effort` entirely)
   - `META_MAX_TOKENS` — optional, default `8192`, hard-capped at `16384`
   - `ALLOWED_ORIGINS=https://meta.lanemint.com` — the public origin
     allowed to call `/api/check` from a browser
   - Optional limit tuning: `LIVE_RATE_PER_MINUTE`,
     `LIVE_RATE_PER_HOUR`, `LIVE_DAILY_MAX_CALLS`,
     `LIVE_DAILY_BUDGET_USD`
   - `META_INPUT_COST_PER_MTOK` and `META_OUTPUT_COST_PER_MTOK` —
     optional (set both, from your provider's current pricing, to
     enforce the spend-based daily budget; if unset, cost is unknown
     and only the call-count limits apply — see "Pricing is never
     hard-coded" above). Never commit real values.
7. Deploy, then verify in the deployment details/logs that the
   healthcheck on `/healthz` was actually applied (not just that the
   port answers), and confirm `https://<your-domain>/healthz` returns
   `{"status": "ok", "mode": "live"}` (or `"simulated"` without a key).

Alternatively, from this directory with the Railway CLI:
`railway up` (add `--new` / `--name` as needed for a fresh project).

## Layout

- `main.py` — FastAPI app: static page, `/healthz`, `/api/config`, `/api/check`, `/api/check-image`, `/api/sample-charts`, `/api/leaderboard`
- `leaderboard.py` + `data/leaderboard.json` — held-out frontier results (single data file, validated on load)
- `muse_client.py` — server-side Muse client + findings/image validation
- `sample_charts.py` + `data/sample_charts.json` — bundled sample chart images and their simulated results
- `rate_limit.py` — in-memory per-IP rate limit, daily call/spend budget, concurrency cap
- `static/` — page, LaneMint design tokens (`css/`), ported demo logic (`js/app.js`)
- `tests/test_api.py` — backend tests (health, config secrecy, fallback, live stub)
