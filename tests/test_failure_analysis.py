"""Failure-analysis chart data must match the existing results file."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "failure_analysis", ROOT / "scripts" / "failure_analysis.py"
)
failure_analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(failure_analysis)


def test_tools_frontier_failure_split_matches_leaderboard():
    rows = {r["name"]: r for r in failure_analysis.load_failure_data()}

    spark = rows["Muse Spark 1.3"]
    # 63/90 pass at 12 turns; of the 27 misses, 25 are recorded in the
    # results data as turn-limit exhaustion, leaving 2 wrong answers.
    assert (spark["passed"], spark["total"]) == (63, 90)
    assert spark["passed"] / spark["total"] == 0.7
    assert spark["turn_limit"] == 25
    assert spark["wrong"] == 2
    assert spark["unrecorded"] == 0

    # No split is invented where the results data records none.
    for name in ("Muse Spark Open", "Muse Glimmer 30B (base)"):
        row = rows[name]
        assert row["turn_limit"] == 0 and row["wrong"] == 0
        assert row["unrecorded"] == row["total"] - row["passed"]

    # The partial trace-LoRA run stays partial (0/36, not 0/90).
    trace = rows["Glimmer + trace LoRA"]
    assert trace["status"] == "partial"
    assert (trace["passed"], trace["total"]) == (0, 36)


def test_chart_file_is_committed():
    chart = ROOT / "docs" / "assets" / "failure-analysis-tools-frontier.png"
    assert chart.exists() and chart.stat().st_size > 10_000
