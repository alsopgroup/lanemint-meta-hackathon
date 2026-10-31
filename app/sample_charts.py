"""Bundled sample chart images and their simulated results.

The PNGs in ``static/assets/sample-charts/`` are generated from the
repo's public ``samples/`` CSVs by ``scripts/make_sample_charts.py``,
which also writes ``data/sample_charts.json``: each entry carries the
PNG's SHA-256 and the findings planted in that public sample (from its
public answer key). In simulated mode (no API key) the server matches
an uploaded image by hash and returns that entry — clearly labelled
simulated. Unknown images get no simulated findings: the app never
pretends to have analysed an image it has not seen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "data" / "sample_charts.json"


def _load_entries() -> list[dict]:
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    samples = data.get("samples")
    return samples if isinstance(samples, list) else []


def list_sample_charts() -> list[dict]:
    """Public sample-chart metadata for the UI (no findings/series)."""
    return [
        {"id": e["id"], "file_name": e["file_name"], "url": e["url"]}
        for e in _load_entries()
        if e.get("id") and e.get("url")
    ]


def match_sample_chart(image_bytes: bytes) -> dict | None:
    """Simulated result for a bundled sample image, matched by SHA-256."""
    digest = hashlib.sha256(image_bytes).hexdigest()
    for entry in _load_entries():
        if entry.get("sha256") == digest:
            return {
                "mode": "simulated",
                "simulated": True,
                "sample_id": entry["id"],
                "summary": entry.get("summary", ""),
                "findings": entry.get("findings", []),
                "series": entry.get("series", []),
                "model": "Simulated demo — no model called",
                "latency_ms": 0,
                "estimated_cost_usd": None,
            }
    return None
