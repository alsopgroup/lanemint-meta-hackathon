# syntax=docker/dockerfile:1

# --- Builder: install the app dependencies into an isolated venv ---
# This is the image Railway builds for the app service (Root Directory
# `app`, see railway.toml); it is also built by the root
# docker-compose.yml, whose build context stays this directory.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Runtime: venv + app code only, non-root ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Railway sets $PORT and also health-checks /healthz itself
# (railway.toml); this covers plain `docker run` / Compose use.
# Use 127.0.0.1 for the healthcheck (IPv4 loopback always works).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/healthz', timeout=4)"]

# No secrets are baked in: META_API_KEY is supplied at runtime only.
# HOST defaults to 0.0.0.0 (binds all IPv4 interfaces, including Railway private networking).
CMD ["sh", "-c", "uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
