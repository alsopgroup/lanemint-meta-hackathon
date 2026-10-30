"""Recognised public samples for the Muse Data Check app.

Only the three public easy-tier samples in the repo's ``samples/``
folder are recognised. Their answer keys live server-side in
``data/sample_answers.json`` (task id, file name, SHA-256 of the
normalised CSV, and the key itself); no other answer keys exist in
this service, and a key is only returned when an uploaded CSV matches
a sample hash exactly.

Matching rule: strip a leading byte-order mark, normalise CRLF/CR
line endings to LF, then compare the SHA-256 hex digest of the UTF-8
encoding. Nothing else is normalised, so any edited value, added row
or reordered column is a different file and does not match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "sample_answers.json"


def normalize_csv_text(csv_text: str) -> str:
    """Strip a leading BOM and normalise line endings to LF."""
    if csv_text.startswith("\ufeff"):
        csv_text = csv_text[1:]
    return csv_text.replace("\r\n", "\n").replace("\r", "\n")


def normalized_sha256(csv_text: str) -> str:
    """SHA-256 hex digest of the normalised CSV text (UTF-8)."""
    return hashlib.sha256(normalize_csv_text(csv_text).encode("utf-8")).hexdigest()


def load_registry() -> list[dict]:
    """The recognised-sample entries, in registry order."""
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return list(data.get("samples", []))


def match_sample(csv_text: str) -> dict | None:
    """Registry entry whose hash matches ``csv_text``, or None.

    Returns a copy holding only the public fields for the matched
    sample; non-matching uploads get None and no key material at all.
    """
    digest = normalized_sha256(csv_text)
    for entry in load_registry():
        if entry.get("sha256") == digest:
            return {
                "task_id": entry["task_id"],
                "file_name": entry["file_name"],
                "sha256": entry["sha256"],
                "answer": entry["answer"],
            }
    return None
