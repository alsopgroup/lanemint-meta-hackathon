"""Recognised-sample tests for the Muse Data Check app.

The app only scores an upload when it exactly matches one of the
public easy-tier samples in the repo's ``samples/`` folder. These
tests keep the server-side registry (``app/data/sample_answers.json``)
in sync with ``samples/`` byte-for-byte (after BOM/line-ending
normalisation), and pin the disabled-Score contract in the frontend.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sample_answers
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
SAMPLES_DIR = REPO_ROOT / "samples"


def _sample_csv(task_id: str) -> str:
    return (SAMPLES_DIR / f"{task_id}.csv").read_bytes().decode("utf-8")


def _sample_answer(task_id: str) -> dict:
    return json.loads((SAMPLES_DIR / "answers" / f"{task_id}_answer.json").read_text())


def test_registry_matches_samples_folder_exactly():
    """Registry hashes/keys must equal samples/ — no drift, no extras."""
    entries = sample_answers.load_registry()
    csv_ids = sorted(p.stem for p in SAMPLES_DIR.glob("task_*.csv"))
    assert sorted(e["task_id"] for e in entries) == csv_ids
    assert len(entries) == len(csv_ids) == 3
    for entry in entries:
        task_id = entry["task_id"]
        assert entry["file_name"] == f"{task_id}.csv"
        assert entry["sha256"] == sample_answers.normalized_sha256(_sample_csv(task_id))
        assert entry["answer"] == _sample_answer(task_id)
        # Only the public easy-tier keys ship: problems + numeric fields.
        assert set(entry) == {"task_id", "file_name", "sha256", "answer"}
        assert "problems_found" in entry["answer"]


def test_normalisation_strips_bom_and_unifies_line_endings():
    csv_text = _sample_csv("task_001")
    crlf_bom = "\ufeff" + csv_text.replace("\n", "\r\n")
    cr_only = csv_text.replace("\n", "\r")
    assert sample_answers.normalized_sha256(crlf_bom) == \
        sample_answers.normalized_sha256(csv_text)
    assert sample_answers.normalized_sha256(cr_only) == \
        sample_answers.normalized_sha256(csv_text)
    # Any real edit is a different file.
    assert sample_answers.match_sample(csv_text + "2024-01-01,X,1,1,1,1,1\n") is None


def test_sample_match_endpoint_returns_key_only_on_exact_match():
    csv_text = _sample_csv("task_002")
    resp = client.post("/api/sample-match", json={"filename": "task_002.csv", "csv": csv_text})
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched"] is True
    assert body["task_id"] == "task_002"
    assert body["answer"] == _sample_answer("task_002")

    # Same bytes with a BOM and CRLF endings still match.
    variant = "\ufeff" + csv_text.replace("\n", "\r\n")
    resp = client.post("/api/sample-match", json={"filename": "x.csv", "csv": variant})
    assert resp.status_code == 200
    assert resp.json()["task_id"] == "task_002"


def test_sample_match_endpoint_exposes_nothing_for_other_files():
    resp = client.post(
        "/api/sample-match",
        json={"filename": "x.csv", "csv": "date,close\n2026-03-02,100.0\n"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"matched": False}
    assert "answer" not in resp.text
    assert "problems_found" not in resp.text


def test_sample_match_rejects_bad_payloads():
    assert client.post("/api/sample-match", json={"csv": ""}).status_code == 400
    assert client.post("/api/sample-match", json={"csv": 42}).status_code == 400
    assert client.post("/api/sample-match", json={}).status_code == 400


def test_after_view_removes_rows_the_score_counts_as_caught():
    """Recognised uploads: After view fixes align with score issues.

    Drives the real app.js under Node (see sample_after_view.test.mjs):
    the duplicate/bad-price rows a recognised sample's Score tab counts
    as caught must be marked removed in the After table, a missed
    finding must apply no fix, and built-in samples plus non-matched
    uploads must behave as before. Skipped without a Node runtime.
    """
    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("node runtime not available")
    harness = Path(__file__).resolve().parent / "sample_after_view.test.mjs"
    result = subprocess.run(  # noqa: S603 - fixed argv, repo-local script
        [node, str(harness)], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all assertions passed" in result.stdout


def test_frontend_disabled_score_contract():
    """Unscored uploads: Score step is disabled with an accessible reason."""
    js = client.get("/static/js/app.js").text
    assert "SCORE_DISABLED_REASON" in js
    assert "Scoring needs an answer key" in js
    assert "samples/ folder" in js
    assert "Findings still work on any file" in js
    assert 'aria-disabled="true"' in js
    assert 'aria-describedby="scoreDisabledNote"' in js
    assert 'id="scoreDisabledNote"' in js
    assert "title=" in js  # hover tooltip carries the same reason
    # The old land-on-an-empty-score-page flow is gone.
    assert "Uploaded files have no answer key, so they can't be scored" not in js
    # Matched uploads are scored through the normal score path.
    assert "matchedSample" in js
    assert "/api/sample-match" in js
    # After-view fix tags are aligned with the score-issue ids.
    assert "applySampleFixTags" in js
    assert "r.remove = issue.id" in js or "rowAt(n)" in js
    css = client.get("/static/css/app.css").text
    assert ".step-disabled" in css
