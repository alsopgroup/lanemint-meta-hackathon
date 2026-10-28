"""Tests for the held-out frontier leaderboard (app/leaderboard.py).

The numbers come from the single data file data/leaderboard.json;
these tests pin the published results, the recomputed totals and
Wilson intervals, and the loader's validation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from leaderboard import load_leaderboard, wilson_interval
from main import app

client = TestClient(app)


def _by_id(data):
    return {e["id"]: e for e in data["entries"]}


def test_published_results_and_recomputed_totals():
    data = load_leaderboard()
    entries = _by_id(data)
    assert list(entries) == [
        "muse-spark-1.3", "muse-spark-open", "muse-glimmer-30b-base",
        "muse-glimmer-30b-lora"]

    spark = entries["muse-spark-1.3"]
    assert {f: (v["passed"], v["total"])
            for f, v in spark["families"].items()} == {
        "F1": (13, 24), "F2": (17, 24), "F3": (14, 24), "F4": (4, 24)}
    assert spark["total"]["passed"] == 48
    assert spark["total"]["total"] == 96
    assert spark["total"]["pct"] == 50
    assert spark["total"]["ci95"] == [40, 60]

    opened = entries["muse-spark-open"]
    assert opened["total"]["passed"] == 2
    assert opened["total"]["total"] == 96
    # Headline percentages use whole-percent rounding for every model.
    assert opened["total"]["pct"] == 2
    assert opened["total"]["ci95"] == [1, 7]
    assert any("API errors" in n for n in opened["notes"])

    glimmer = entries["muse-glimmer-30b-base"]
    assert "scope" not in glimmer  # pooled over all held-out seeds now
    assert {f: (v["passed"], v["total"])
            for f, v in glimmer["families"].items()} == {
        "F1": (0, 24), "F2": (0, 24), "F3": (0, 24), "F4": (0, 24)}
    assert glimmer["total"]["passed"] == 0
    assert glimmer["total"]["total"] == 96
    assert glimmer["total"]["pct"] == 0
    assert glimmer["total"]["ci95"] == [0, 4]

    lora = entries["muse-glimmer-30b-lora"]
    assert lora["status"] == "coming_soon"
    assert lora["total"] is None
    assert lora["families"] is None


def test_benchmark_metadata():
    data = load_leaderboard()
    bench = data["benchmark"]
    assert bench["seeds"] == [42, 1001, 1002]
    assert bench["tasks"] == 96
    # Footnote: seed-42 F1 for Spark Open / Glimmer base predates the
    # current 4-window F1; everything else uses the 4-window version.
    notes = " ".join(bench.get("notes", []))
    assert "3-window" in notes and "4-window" in notes
    assert "Spark Open" in notes and "Glimmer 30B" in notes
    assert [f["id"] for f in bench["families"]] == ["F1", "F2", "F3", "F4"]
    assert [f["type"] for f in bench["families"]] == [
        "frontier_rolling_risk", "frontier_vendor_reconciliation",
        "frontier_factor_exposure", "frontier_portfolio_pipeline"]


def test_wilson_interval():
    assert wilson_interval(48, 96) == (40, 60)
    assert wilson_interval(2, 96) == (1, 7)
    assert wilson_interval(0, 32) == (0, 11)
    assert wilson_interval(96, 96) == (96, 100)


def _write(tmp_path, payload):
    path = tmp_path / "lb.json"
    path.write_text(json.dumps(payload))
    return path


def _valid_payload():
    return {
        "benchmark": {"families": [{"id": "F1"}, {"id": "F2"}]},
        "entries": [
            {"id": "a", "name": "A", "status": "complete",
             "families": {"F1": {"passed": 1, "total": 2},
                          "F2": {"passed": 0, "total": 2}}},
            {"id": "b", "name": "B", "status": "coming_soon"},
        ],
    }


def test_loader_recomputes_totals(tmp_path):
    data = load_leaderboard(_write(tmp_path, _valid_payload()))
    assert data["entries"][0]["total"]["passed"] == 1
    assert data["entries"][0]["total"]["total"] == 4


@pytest.mark.parametrize("mutate", [
    lambda p: p["entries"][0]["families"].update(
        {"F1": {"passed": 3, "total": 2}}),          # passed > total
    lambda p: p["entries"][0].update({"families": {"F1": {"passed": 1, "total": 2}}}),  # missing family
    lambda p: p["entries"].append(dict(p["entries"][0])),  # duplicate id
    lambda p: p["entries"][0].update({"status": "done"}),  # bad status
    lambda p: p.update({"entries": []}),                    # no entries
])
def test_loader_rejects_bad_data(tmp_path, mutate):
    payload = _valid_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        load_leaderboard(_write(tmp_path, payload))


def test_loader_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        load_leaderboard(tmp_path / "nope.json")


def test_leaderboard_endpoint():
    resp = client.get("/api/leaderboard")
    assert resp.status_code == 200
    body = resp.json()
    assert [e["id"] for e in body["entries"]] == [
        "muse-spark-1.3", "muse-spark-open", "muse-glimmer-30b-base",
        "muse-glimmer-30b-lora"]
    assert body["entries"][0]["total"] == {
        "passed": 48, "total": 96, "pct": 50, "ci95": [40, 60]}
    assert body["entries"][3]["status"] == "coming_soon"
    assert "api_key" not in resp.text.lower()


def test_index_has_leaderboard_and_mode_indicator():
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'id="leaderboard"' in resp.text
    assert 'id="leaderboardBody"' in resp.text
    # Visible live-vs-demo indicator next to the demo.
    assert 'id="modeDot"' in resp.text and 'id="modeText"' in resp.text
    assert 'id="demoBanner"' in resp.text


def test_index_has_no_old_benchmark_or_cost_figures():
    resp = client.get("/")
    assert resp.status_code == 200
    # The old seed-42 v4 easy/hard section is replaced by the frontier
    # leaderboard, and no spend/cost figures appear in the public app.
    assert 'id="benchmark"' not in resp.text
    assert "25 easy + 25 hard" not in resp.text
    assert "Total run cost" not in resp.text
    assert "$4.48" not in resp.text
