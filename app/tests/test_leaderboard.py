"""Tests for the held-out leaderboards (app/leaderboard.py).

The numbers come from the single data file data/leaderboard.json;
these tests pin the published results for both boards (frontier
tools off/on and tools_frontier at 12 turns), the recomputed totals
and Wilson intervals, and the loader's validation.
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


def _boards(data):
    return {b["id"]: b for b in data["boards"]}


def _by_id(board):
    return {e["id"]: e for e in board["entries"]}


def test_frontier_tools_board_published_results():
    boards = _boards(load_leaderboard())
    assert list(boards) == ["frontier-tools", "tools-frontier"]
    board = boards["frontier-tools"]
    assert board["tier"] == "frontier"
    assert board["tasks"] == 96
    entries = _by_id(board)
    assert list(entries) == [
        "muse-spark-1.3", "muse-spark-open", "muse-glimmer-30b-base",
        "muse-glimmer-30b-trace-lora", "muse-glimmer-30b-answer-lora"]

    spark = entries["muse-spark-1.3"]
    assert spark["tools_off"] == {
        "passed": 48, "total": 96, "pct": 50, "ci95": [40, 60],
        "approx": False, "display": None}
    assert spark["tools_on"]["passed"] == 95
    assert spark["tools_on"]["total"] == 96
    assert spark["tools_on"]["pct"] == 99

    opened = entries["muse-spark-open"]
    assert opened["tools_off"]["passed"] == 2
    assert opened["tools_off"]["pct"] == 2
    assert opened["tools_off"]["ci95"] == [1, 7]
    assert opened["tools_on"]["passed"] == 95
    assert opened["tools_on"]["pct"] == 99
    assert any("API errors" in n for n in opened["notes"])

    glimmer = entries["muse-glimmer-30b-base"]
    assert glimmer["tools_off"]["passed"] == 0
    assert glimmer["tools_off"]["pct"] == 0
    # Approximate rate only: no exact count is invented for it.
    assert glimmer["tools_on"]["passed"] is None
    assert glimmer["tools_on"]["total"] is None
    assert glimmer["tools_on"]["pct"] == 85
    assert glimmer["tools_on"]["approx"] is True
    assert glimmer["tools_on"]["display"] == "~85%"

    trace = entries["muse-glimmer-30b-trace-lora"]
    assert trace["tools_off"]["passed"] == 0
    assert trace["tools_on"]["pct"] == 15
    assert trace["tools_on"]["approx"] is True

    answer = entries["muse-glimmer-30b-answer-lora"]
    assert answer["tools_off"]["passed"] == 0
    assert answer["tools_on"]["passed"] == 0
    assert answer["tools_on"]["pct"] == 0


def test_tools_frontier_board_published_results():
    board = _boards(load_leaderboard())["tools-frontier"]
    assert board["tier"] == "tools_frontier"
    assert board["tasks"] == 90
    assert board["max_turns"] == 12
    assert [f["id"] for f in board["families"]] == [
        "T1", "T2", "T3", "T4", "T5"]
    assert [f["label"] for f in board["families"]] == [
        "Hedge", "FX min-var", "Risk parity", "Split/FIFO",
        "Constraint stack"]
    entries = _by_id(board)
    assert list(entries) == [
        "muse-spark-1.3", "muse-spark-open", "muse-glimmer-30b-base",
        "muse-glimmer-30b-answer-lora", "muse-glimmer-30b-trace-lora"]

    spark = entries["muse-spark-1.3"]
    assert {f: (v["passed"], v["total"])
            for f, v in spark["families"].items()} == {
        "T1": (18, 18), "T2": (10, 18), "T3": (11, 18),
        "T4": (18, 18), "T5": (6, 18)}
    assert spark["total"]["passed"] == 63
    assert spark["total"]["total"] == 90
    assert spark["total"]["pct"] == 70

    opened = entries["muse-spark-open"]
    assert opened["total"]["passed"] == 51
    assert opened["total"]["pct"] == 57

    glimmer = entries["muse-glimmer-30b-base"]
    assert glimmer["total"]["passed"] == 27
    assert glimmer["total"]["pct"] == 30

    answer = entries["muse-glimmer-30b-answer-lora"]
    assert answer["total"]["passed"] == 0
    assert answer["total"]["total"] == 90

    trace = entries["muse-glimmer-30b-trace-lora"]
    assert trace["scope"] == "partial run"
    assert trace["total"]["passed"] == 0
    assert trace["total"]["total"] == 36
    # Per-family cells are display-only: no denominator is published.
    assert all(v["display"] == "0" and v["passed"] is None
               for v in trace["families"].values())

    notes = " ".join(board["notes"])
    assert "25 of 27" in notes
    assert "fine-tuning Glimmer" in notes


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
    assert data["boards"][0]["entries"][0]["total"]["passed"] == 1
    assert data["boards"][0]["entries"][0]["total"]["total"] == 4


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
    assert [b["id"] for b in body["boards"]] == [
        "frontier-tools", "tools-frontier"]
    frontier, tools_frontier = body["boards"]
    assert frontier["entries"][0]["tools_off"] == {
        "passed": 48, "total": 96, "pct": 50, "ci95": [40, 60],
        "approx": False, "display": None}
    assert frontier["entries"][0]["tools_on"]["pct"] == 99
    assert tools_frontier["entries"][0]["total"] == {
        "passed": 63, "total": 90, "pct": 70, "ci95": [60, 78],
        "approx": False, "display": None}
    assert tools_frontier["max_turns"] == 12
    assert "api_key" not in resp.text.lower()


def test_index_has_leaderboard_and_mode_indicator():
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'id="leaderboard"' in resp.text
    assert 'id="leaderboardBody"' in resp.text
    # Visible live-vs-demo indicator next to the demo.
    assert 'id="modeDot"' in resp.text and 'id="modeText"' in resp.text
    assert 'id="demoBanner"' in resp.text
    # Both tiers are described in the leaderboard copy.
    assert "tools_frontier" in resp.text
    assert "12 turns" in resp.text


def test_index_has_no_old_benchmark_or_cost_figures():
    resp = client.get("/")
    assert resp.status_code == 200
    # The old seed-42 v4 easy/hard section is replaced by the frontier
    # leaderboard, and no spend/cost figures appear in the public app.
    assert 'id="benchmark"' not in resp.text
    assert "25 easy + 25 hard" not in resp.text
    assert "Total run cost" not in resp.text
    assert "$4.48" not in resp.text
