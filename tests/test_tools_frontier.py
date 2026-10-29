"""Tools-frontier tier: offline checks from SCOPE.md section 6.

Check map:
1. The independent reference solver (src/tools_frontier_ref.py, which
   reads only task-directory files plus spec.md) scores 100% on the
   development seeds 7001-7010.
2. Trivial baselines (empty/zero answers) and every family's
   single-trap / naive wrong-method variants pass at most 5%.
3. Same seed -> byte-identical task files, tasks.json and answers.
4. Banned-feature lint over the generator source and generated outputs,
   plus the reserved-eval-seed refusal.
5. The reference computation fits the existing sandbox (when bubblewrap
   is available) and the tool runner handles a multi-file task.
"""

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import pytest

from src.scorer import TaskScorer
from src.tools_frontier_ref import solve_task
from src.tools_frontier_tier import (
    EVAL_SEEDS,
    TOOLS_FRONTIER_SETTINGS,
    ToolsFrontierTaskGenerator,
)

DEV_SEEDS = list(range(7001, 7011))
NAIVE_VARIANT = {
    "tools_frontier_convention_risk": "naive_simple_252",
    "tools_frontier_fx_calendar": "naive_ignore_fx_and_calendar",
    "tools_frontier_reconciliation": "vendor_a_always_no_corrections",
    "tools_frontier_split_holdings": "split_not_applied_to_holdings",
    "tools_frontier_portfolio_construction": "no_shrinkage",
}


def _generate(seed, tmp_root: Path, count=30) -> Path:
    out = tmp_root / f"seed{seed}"
    gen = ToolsFrontierTaskGenerator(seed=seed)
    gen.generate_all_tasks(out, num_tasks=count)
    return out


@pytest.fixture(scope="module")
def dev_sets(tmp_path_factory):
    """All ten development seeds, generated once for this module."""
    root = tmp_path_factory.mktemp("tools_frontier_dev")
    return {seed: _generate(seed, root) for seed in DEV_SEEDS}


def _tasks(root: Path):
    return json.loads((root / "tasks.json").read_text())


# ---------------------------------------------------------------- check 1

class TestReferenceSolver:
    def test_reference_scores_100_percent_on_dev_seeds(self, dev_sets):
        total = passed = 0
        for seed, root in dev_sets.items():
            scorer = TaskScorer(root / "answers")
            for task in _tasks(root):
                got = solve_task(task, root)
                result = scorer.score_task(task["id"], json.dumps(got))
                total += 1
                passed += bool(result["passed"] and result["score"] == 1.0)
                assert result["passed"] and result["score"] == 1.0, (
                    seed, task["id"], result["feedback"])
        assert total == 300 and passed == 300

    def test_thirty_tasks_six_per_family_per_seed(self, dev_sets):
        for root in dev_sets.values():
            tasks = _tasks(root)
            assert len(tasks) == 30
            counts = defaultdict(int)
            for task in tasks:
                counts[task["type"]] += 1
                task_dir = root / task["data_dir"]
                assert (task_dir / "spec.md").is_file()
                assert 3 <= len(task["files"]) <= 5
            assert set(counts.values()) == {6}

    def test_settings_knobs_present(self):
        assert set(TOOLS_FRONTIER_SETTINGS) == {
            "convention_risk", "fx_calendar", "reconciliation",
            "split_holdings", "portfolio_construction"}


# ---------------------------------------------------------------- check 2

def _zero_answer(expected):
    out = {}
    for key, value in expected.items():
        if key.startswith("_"):
            continue
        if key == "problems_found":
            out["problems"] = []
        elif isinstance(value, dict):
            out[key] = dict.fromkeys(value, 0)
        elif isinstance(value, int):
            out[key] = 0
        elif isinstance(value, float):
            out[key] = 0.0
        else:
            out[key] = ""
    return out


class TestBaselinesFail:
    def test_empty_and_zero_baselines_pass_at_most_5_percent(self, dev_sets):
        for root in dev_sets.values():
            scorer = TaskScorer(root / "answers")
            for task in _tasks(root):
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                for response in ("{}", json.dumps(_zero_answer(expected))):
                    result = scorer.score_task(task["id"], response)
                    assert not result["passed"], (task["id"], response[:40])

    def test_naive_and_single_trap_variants_fail_per_family(self, dev_sets):
        """Each family's stored wrong-method variants must fail >=95%.

        The variant answer is otherwise perfect: the reference answer
        with every core field replaced by the variant's value.  The
        generator only accepts a task when each variant misses at least
        one core field by more than 50x tolerance, so these must fail.
        """
        checked = defaultdict(int)
        failed = defaultdict(int)
        for root in dev_sets.values():
            scorer = TaskScorer(root / "answers")
            for task in _tasks(root):
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                core = expected["_tools_frontier"]["core"]
                for _name, variant in task["trap_variants"].items():
                    got = solve_task(task, root)
                    for field in core:
                        if field in variant:
                            got[field] = variant[field]
                    result = scorer.score_task(task["id"], json.dumps(got))
                    checked[task["type"]] += 1
                    failed[task["type"]] += not result["passed"]
        assert set(checked) == set(NAIVE_VARIANT)
        for family, n in checked.items():
            assert failed[family] / n >= 0.95, (family, failed[family], n)
            assert failed[family] == n, (family, failed[family], n)

    def test_one_wrong_core_field_never_passes(self, dev_sets):
        root = dev_sets[7001]
        scorer = TaskScorer(root / "answers")
        for task in _tasks(root):
            expected = json.loads(
                (root / "answers" / f"{task['id']}_answer.json").read_text())
            core = expected["_tools_frontier"]["core"]
            got = solve_task(task, root)
            field = core[0]
            if isinstance(expected[field], dict):
                key = next(iter(expected[field]))
                got[field] = dict(got[field])
                got[field][key] = got[field][key] + 1
            elif isinstance(expected[field], float):
                got[field] = expected[field] + max(0.05, abs(expected[field]) * 0.01)
            elif isinstance(expected[field], int):
                got[field] = expected[field] + 3
            else:
                got[field] = "1999-01-01"
            result = scorer.score_task(task["id"], json.dumps(got))
            assert not result["passed"], (task["id"], field)


# ---------------------------------------------------------------- check 3

class TestDeterminism:
    def test_same_seed_byte_identical(self, tmp_path):
        first = _generate(7003, tmp_path / "a", count=10)
        second = _generate(7003, tmp_path / "b", count=10)

        def digest(root):
            out = {}
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    out[str(path.relative_to(root))] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
            return out

        assert digest(first) == digest(second)


# ---------------------------------------------------------------- check 4

# Assembled so this test file itself does not trip a naive repo-wide grep
# for the banned feature names; the lint below scans the generator source
# and generated task outputs, which is what SCOPE.md section 6 requires.
_BANNED = ["divi" + "dend", "deli" + "st", "survivor" + "ship",
           "rena" + "me", "sta" + "le", "too-short", "too_short"]


class TestConstraintLint:
    def test_no_banned_features_in_generator_or_outputs(self, dev_sets):
        sources = [Path("src/tools_frontier_tier.py").read_text(),
                   Path("src/tools_frontier_ref.py").read_text()]
        for root in dev_sets.values():
            for path in root.rglob("*"):
                if path.is_file() and "answers" not in path.parts:
                    sources.append(path.read_text())
        blob = "\n".join(sources).lower()
        for word in _BANNED:
            assert word not in blob, word

    def test_no_equal_consecutive_closes_and_history_long_enough(self, dev_sets):
        import csv as csv_mod
        for root in dev_sets.values():
            for task in _tasks(root):
                task_dir = root / task["data_dir"]
                for name in task["files"]:
                    if not name.endswith("prices.csv") and name not in (
                            "us_prices.csv", "eu_prices.csv", "vendor_a.csv",
                            "vendor_b.csv"):
                        continue
                    rows = list(csv_mod.DictReader(open(task_dir / name)))
                    if "close" not in rows[0]:
                        continue
                    by_ticker = defaultdict(list)
                    for row in rows:
                        by_ticker[row["ticker"]].append(
                            (row["date"], float(row["close"])))
                    for ticker, pairs in by_ticker.items():
                        # Consecutive means consecutive trading dates in
                        # date order: one T1 ticker's rows are deliberately
                        # shuffled in the file (so sort first), and vendor
                        # files have one-vendor gap dates, across which a
                        # repeated value is not a consecutive repeat.
                        from datetime import datetime as _dt
                        ordered = sorted(pairs)
                        assert len(ordered) >= 100, (task["id"], name, ticker)
                        for (d1, c1), (d2, c2) in zip(
                                ordered, ordered[1:], strict=False):
                            day1 = _dt.strptime(d1, "%Y-%m-%d")
                            day2 = _dt.strptime(d2, "%Y-%m-%d")
                            weekdays_between = sum(
                                1 for k in range(1, (day2 - day1).days)
                                if (day1.fromordinal(day1.toordinal() + k)
                                    .weekday() < 5))
                            if d1 != d2 and weekdays_between == 0:
                                assert c1 != c2, (task["id"], name, ticker)

    def test_eval_seeds_blocked_without_eval_flag(self):
        assert EVAL_SEEDS == frozenset({42, 1001, 1002})
        for seed in (42, 1001, 1002):
            with pytest.raises(ValueError, match="reserved"):
                ToolsFrontierTaskGenerator(seed=seed)
            # Explicit evaluation opt-in is the only way through.
            ToolsFrontierTaskGenerator(seed=seed, allow_eval=True)
        ToolsFrontierTaskGenerator(seed=7001)

    def test_splits_only_as_explicit_table_events(self, dev_sets):
        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] == "tools_frontier_split_holdings":
                    assert "splits.csv" in task["files"]
                    expected = json.loads(
                        (root / "answers" / f"{task['id']}_answer.json").read_text())
                    assert expected["num_splits"] >= 1
                else:
                    assert "splits.csv" not in task["files"]


# ------------------------------------------------------- calibration fixes

class TestCalibrationFixes:
    """Regression tests for the 7101/7102 calibration findings."""

    def test_drawdown_sign_convention_stated_and_enforced(
            self, dev_sets, tmp_path):
        # Every memo that asks for a drawdown states the sign convention.
        root = dev_sets[7001]
        seen = set()
        for task in _tasks(root):
            spec = (root / task["data_dir"] / "spec.md").read_text()
            expected = json.loads(
                (root / "answers" / f"{task['id']}_answer.json").read_text())
            if any(k in expected for k in ("max_drawdown", "nav_max_drawdown")):
                assert "negative fraction" in spec, task["type"]
                seen.add(task["type"])
        assert len(seen) == 4  # T2 (NAV), T3, T4, T5
        # The scorer accepts only the stated convention: a positive
        # drawdown scores 0 even within tolerance of a near-zero key.
        answer = {
            "max_drawdown": -0.000001, "final_nav": 10.0,
            "_tools_frontier": {"family": "x", "core": ["max_drawdown"],
                                "rtol": 1e-6, "atol": 2e-6, "field_rtol": {}},
        }
        (tmp_path / "task_tf_9001_answer.json").write_text(json.dumps(answer))
        scorer = TaskScorer(tmp_path)
        good = scorer.score_task("task_tf_9001", json.dumps(
            {"max_drawdown": -0.000001, "final_nav": 10.0}))
        flipped = scorer.score_task("task_tf_9001", json.dumps(
            {"max_drawdown": 0.000001, "final_nav": 10.0}))
        assert good["passed"]
        assert flipped["component_scores"]["max_drawdown"] == 0.0
        assert not flipped["passed"]

    def test_memos_list_exact_allowed_codes(self, dev_sets):
        root = dev_sets[7001]
        by_type = {}
        for task in _tasks(root):
            by_type.setdefault(task["type"],
                               (root / task["data_dir"] / "spec.md").read_text())
        t3 = by_type["tools_frontier_reconciliation"]
        assert "may contain only: source_conflict, unit_change" in t3
        # The one-vendor date rule is stated precisely in the memo.
        assert "present in only one vendor is" in t3 and "NOT missing" in t3
        t4 = by_type["tools_frontier_split_holdings"]
        assert "may contain only: split, reverse_split" in t4
        assert "Split-table rows DO count as problems" in t4

    def test_t5_sector_caps_enforced_bind_and_cash_rules(self, dev_sets):
        total = binding = 0
        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_portfolio_construction":
                    continue
                total += 1
                task_dir = root / task["data_dir"]
                cfg = json.loads((task_dir / "constraints.json").read_text())
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                weights = expected["weights_first_rebalance"]
                for sector, cap in cfg["sector_caps"].items():
                    ssum = sum(w for t, w in weights.items()
                               if cfg["sectors"][t] == sector)
                    assert ssum <= cap + 2e-3, (task["id"], sector, ssum, cap)
                    if abs(ssum - cap) <= 2e-3:
                        binding += 1
                        break
                # Cash can never go negative, and every post-first
                # turnover is the actual one and within the cap.
                assert expected["cash_left"] >= 0
                turnovers = list(expected["turnover_by_rebalance"].values())
                assert all(t <= cfg["turnover_cap"] + 1e-6
                           for t in turnovers[1:]), task["id"]
        assert total == 60
        # Caps are constructed to bind at the first rebalance.
        assert binding >= 48, binding

    def test_rounded_and_unrounded_correct_values_both_pass(self, dev_sets):
        root = dev_sets[7001]
        scorer = TaskScorer(root / "answers")
        task = _tasks(root)[0]  # T1
        got = solve_task(task, root)
        assert scorer.score_task(task["id"], json.dumps(got))["passed"]
        # A correct value left unrounded sits within 5e-7 of the stored
        # 6-decimal key; it must pass just like the rounded value.
        field = "ann_vol_AST1"
        nudged = dict(got)
        nudged[field] = got[field] + 5e-7
        result = scorer.score_task(task["id"], json.dumps(nudged))
        assert result["component_scores"][field] == 1.0
        assert result["passed"]

    def test_t1_t2_hardening_fields_and_pitfalls(self, dev_sets):
        root = dev_sets[7001]
        t1 = next(t for t in _tasks(root)
                  if t["type"] == "tools_frontier_convention_risk")
        t2 = next(t for t in _tasks(root)
                  if t["type"] == "tools_frontier_fx_calendar")
        exp1 = json.loads((root / "answers" / f"{t1['id']}_answer.json").read_text())
        exp2 = json.loads((root / "answers" / f"{t2['id']}_answer.json").read_text())
        assert {"rolling2_max_beta", "downside_volatility",
                "log_return_correlation",
                "weekly_annualised_log_return"} <= set(exp1)
        assert {"effective_fx_rate_mid_probe", "max_nav_value",
                "max_nav_date", "nav_max_drawdown"} <= set(exp2)
        # Weekend prints exist in T1 prices and are excluded by calendar.
        import csv as csv_mod
        cal = {r["date"] for r in csv_mod.DictReader(
            open(root / t1["data_dir"] / "calendar.csv"))}
        price_dates = {r["date"] for r in csv_mod.DictReader(
            open(root / t1["data_dir"] / "prices.csv"))}
        assert price_dates - cal
        # T2 stamps include the after-midnight +01:00 form in some task.
        stamps = []
        for task in _tasks(root):
            if task["type"] == "tools_frontier_fx_calendar":
                stamps += [r["timestamp"] for r in csv_mod.DictReader(
                    open(root / task["data_dir"] / "fx.csv"))]
        assert any(s.endswith("T00:30:00+01:00") for s in stamps)


# ---------------------------------------------------------------- check 5

class TestSandboxAndRunnerFit:
    def test_reference_computation_inside_sandbox(self, dev_sets):
        from src.sandbox import run_python, sandbox_available
        if not sandbox_available():
            pytest.skip("bubblewrap sandbox unavailable on this host")
        root = dev_sets[7001]
        task = _tasks(root)[0]
        ref_source = Path("src/tools_frontier_ref.py").read_text()
        code = (ref_source
                + "\nimport json\n"
                + "print(json.dumps(solve_task_dir('/data')))\n")
        started = time.monotonic()
        result = run_python(code, root / task["data_dir"], timeout_seconds=30)
        elapsed = time.monotonic() - started
        assert not result["timed_out"] and result["exit"] == 0, result
        assert elapsed < 30
        assert len(result["stdout"]) <= 8000
        answer = json.loads(result["stdout"])
        expected = json.loads(
            (root / "answers" / f"{task['id']}_answer.json").read_text())
        assert answer["num_aligned_days"] == expected["num_aligned_days"]

    def test_tool_runner_handles_multi_file_task(self, dev_sets, tmp_path):
        from src.tool_runner import ToolTaskRunner
        root = dev_sets[7001]
        task = _tasks(root)[0]
        runner = ToolTaskRunner(dry_run=True, tasks_dir=root,
                                transcript_dir=tmp_path / "transcripts")
        data = runner.load_data(root, task["data_dir"])
        assert "===== FILE: spec.md =====" in data
        result = runner.run_task(task, data)
        assert result["task_id"] == task["id"]
        assert result["task_type"] == task["type"]
        preamble = runner._preamble(task, data)
        assert "/data/" in preamble and "spec.md" in preamble

    def test_plain_runner_loads_task_directory_and_token_cap(self, dev_sets, tmp_path):
        from src.runner import TaskRunner
        root = dev_sets[7001]
        runner = TaskRunner(dry_run=True)
        assert runner._max_tokens_for({"id": "task_tf_3001"}) == 32000
        results = runner.run_all_tasks(tasks_dir=root, output_dir=tmp_path / "out")
        assert len(results) == 30
        assert all(r["task_id"].startswith("task_tf_") for r in results)


# --------------------------------------------------------------------- CLI

class TestCli:
    def test_module_cli_and_generate_tasks_tier(self, tmp_path, monkeypatch):
        import sys

        from src import generate_tasks, tools_frontier_tier

        out = tmp_path / "tf"
        assert tools_frontier_tier.main(
            ["--seed", "7004", "--count", "5", "--output-dir", str(out)]) == 0
        assert len(json.loads((out / "tasks.json").read_text())) == 5

        out2 = tmp_path / "tf2"
        monkeypatch.setattr(sys, "argv", [
            "gen", "--tier", "tools_frontier", "--seed", "7004",
            "--count", "5", "--output-dir", str(out2)])
        generate_tasks.main()
        assert len(json.loads((out2 / "tasks.json").read_text())) == 5

        monkeypatch.setattr(sys, "argv", [
            "gen", "--tier", "tools_frontier", "--seed", "42",
            "--count", "5", "--output-dir", str(tmp_path / "no")])
        with pytest.raises(SystemExit) as excinfo:
            generate_tasks.main()
        assert excinfo.value.code == 1


# ------------------------------------------------------- re-pilot fixes

class TestRepilotFixes:
    """Round-3 fixes from the seed-7201 re-pilot (Spark 1.3 24/30,
    Spark Open 21/30, Glimmer 12/30, with T1/T3/T4 saturated)."""

    def test_t2_walkback_consecutive_late_and_allowed_codes(self, dev_sets):
        import csv as csv_mod
        from datetime import UTC
        from datetime import datetime as dt

        def read(task_dir, name):
            return list(csv_mod.DictReader(open(task_dir / name)))

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_fx_calendar":
                    continue
                task_dir = root / task["data_dir"]
                spec = (task_dir / "spec.md").read_text()
                # Item 2: exact allowed codes, and the two non-codes
                # (out-of-order rows, weekend prints) explicitly excluded.
                assert "may contain only: currency, timezone, calendar" in spec
                assert "out_of_order" in spec and "NOT flagged" in spec
                assert "weekend" in spec and "no category code" in spec
                # Item 1: the walk-back rule is stated explicitly.
                assert "most recent ON-TIME fix" in spec
                assert "a late fix is never used" in spec

                union = sorted(
                    {r["date"] for r in read(task_dir, "us_prices.csv")}
                    | {r["date"] for r in read(task_dir, "eu_prices.csv")})
                by_utc = {}
                for r in read(task_dir, "fx.csv"):
                    instant = dt.fromisoformat(
                        r["timestamp"].replace("Z", "+00:00")).astimezone(UTC)
                    by_utc[instant.strftime("%Y-%m-%d")] = (r, instant)
                # Walk the union dates: on-time flag, late runs, and the
                # walked-back rate (most recent on-time fix, converted).
                on_time, converted = {}, {}
                prev_on_time, run, max_run = None, 0, 0
                walked = {}
                for ds in union:
                    row, instant = by_utc[ds]
                    rate = float(row["rate"])
                    eff = rate if row["quote"] == "EURUSD" else 1 / rate
                    late = (instant.hour * 60 + instant.minute) > 16 * 60
                    on_time[ds] = not late
                    converted[ds] = eff
                    if late:
                        run += 1
                        max_run = max(max_run, run)
                        walked[ds] = prev_on_time
                    else:
                        run = 0
                        prev_on_time = eff
                # Every task exercises consecutive late fixes.
                assert max_run >= 2, task["id"]
                # The key's detail rate is the walked-back rate on a
                # date whose own fix AND the previous date's fix are
                # both late - not either late fix itself.
                import re as re_mod
                detail = re_mod.search(r"fx_detail_date: (\S+)", spec).group(1)
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                field = f"effective_fx_rate_{detail.replace('-', '')}"
                prev_ds = union[union.index(detail) - 1]
                assert not on_time[detail] and not on_time[prev_ds]
                assert abs(expected[field] - walked[detail]) < 1e-6
                # ... and it is not a copy of either late fix (the
                # reseeding gate separately enforces >50x-tolerance
                # separation from the wrong-method answers).
                assert abs(expected[field] - converted[detail]) > 1e-6
                assert abs(expected[field] - converted[prev_ds]) > 1e-6

    def test_t5_turnover_format_stated_and_keyed(self, dev_sets):
        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_portfolio_construction":
                    continue
                spec = (root / task["data_dir"] / "spec.md").read_text()
                # Item 3: the expected format is stated in the memo.
                assert "turnover_by_rebalance field is a JSON object" in spec
                assert '"1", "2", ... in rebalance order' in spec
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                turnovers = expected["turnover_by_rebalance"]
                assert list(turnovers) == [
                    str(i + 1) for i in range(expected["num_rebalances"])]
                assert all(isinstance(v, float) for v in turnovers.values())

    def test_t3_superseding_corrections_mixed_offsets(self, dev_sets):
        import csv as csv_mod
        from collections import Counter

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_reconciliation":
                    continue
                task_dir = root / task["data_dir"]
                spec = (task_dir / "spec.md").read_text()
                assert "supersedes every earlier" in spec
                assert "comparing the stamp strings as written" in \
                    " ".join(spec.split())
                corr = list(csv_mod.DictReader(
                    open(task_dir / "corrections.csv")))
                # At least one key carries two corrections (one
                # supersedes the other) in every task.
                counts = Counter((r["ticker"], r["date"]) for r in corr)
                assert max(counts.values()) >= 2, task["id"]
                # The superseded pair is stamped in +01:00 / -05:00
                # forms, so string order and instant order disagree.
                stamps = [r["as_of"] for r in corr]
                assert any(s.endswith("+01:00") for s in stamps)
                assert any(s.endswith("-05:00") for s in stamps)
                # At least one corrected key sits in Vendor B's cents
                # segment (correction closes remain dollars there).
                b_unit = {(r["ticker"], r["date"]): r["unit"]
                          for r in csv_mod.DictReader(
                              open(task_dir / "vendor_b.csv"))}
                assert any(b_unit[k] == "cents" for k in counts), task["id"]

    def test_t4_multiple_splits_and_noop_row(self, dev_sets):
        import csv as csv_mod

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_split_holdings":
                    continue
                task_dir = root / task["data_dir"]
                spec = (task_dir / "spec.md").read_text()
                assert "1-for-1" in spec and "no-op" in spec
                assert "not counted" in spec
                rows = list(csv_mod.DictReader(open(task_dir / "splits.csv")))
                real = [r for r in rows if r["ratio"] != "1-for-1"]
                # The no-op rows are present in the table but are not
                # events: num_splits counts only real rows.
                assert len(rows) - len(real) == 2, task["id"]
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                assert expected["num_splits"] == len(real)
                # One ticker splits twice: forward 2-for-1, then a
                # reverse 1-for-3 later in date order.
                first = sorted((r for r in real if r["ticker"] == "SPL1"),
                               key=lambda r: r["date"])
                assert [r["ratio"] for r in first] == ["2-for-1", "1-for-3"]

    def test_t1_second_benchmark_and_recent_window(self, dev_sets):
        import csv as csv_mod

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_convention_risk":
                    continue
                task_dir = root / task["data_dir"]
                spec = (task_dir / "spec.md").read_text()
                assert "benchmark=BMK2" in spec
                assert "window_returns=60" in spec
                tickers = {r["ticker"] for r in csv_mod.DictReader(
                    open(task_dir / "prices.csv"))}
                assert "BMK2" in tickers
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                assert {"rolling3_max_beta", "rolling3_max_beta_end_date",
                        "recent_vol_last_60"} <= set(expected)
                assert {"wrong_benchmark", "full_sample_instead_of_recent"} \
                    <= set(task["trap_variants"])


# ------------------------------------------------- computation hardening

class TestComputationHardening:
    """Round-4 hardening (seed 7301 pilot saturated at 30/30): scale,
    chained pipelines, cross-family composites and exact counts.  These
    tests pin the structural guarantees the generator constructs; the
    reference-solver test pins the numbers."""

    def test_t1_scale_and_composite_fields(self, dev_sets):
        import csv as csv_mod

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_convention_risk":
                    continue
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                # 8 assets, each with its own ann-vol convention field.
                assert sum(1 for k in expected
                           if k.startswith("ann_vol_")) == 8
                assert {"portfolio_ann_vol", "portfolio_max_drawdown",
                        "portfolio_max_drawdown_trough_date",
                        "portfolio_sharpe", "rolling_max_correlation",
                        "rolling_max_correlation_end_date",
                        "worst_day_return", "worst_day_date"} <= set(expected)
                assert expected["num_aligned_days"] == 260
                tickers = {r["ticker"] for r in csv_mod.DictReader(
                    open(root / task["data_dir"] / "prices.csv"))}
                assert {"BMK", "BMK2"} <= tickers and len(tickers) == 10
                assert "portfolio_from_price_levels" in task["trap_variants"]

    def test_t2_exact_fix_counts(self, dev_sets):
        import csv as csv_mod
        from datetime import UTC
        from datetime import datetime as dt

        def read(task_dir, name):
            return list(csv_mod.DictReader(open(task_dir / name)))

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_fx_calendar":
                    continue
                task_dir = root / task["data_dir"]
                union = sorted(
                    {r["date"] for r in read(task_dir, "us_prices.csv")}
                    | {r["date"] for r in read(task_dir, "eu_prices.csv")})
                by_utc, late = {}, 0
                inverted = weekend = 0
                for r in read(task_dir, "fx.csv"):
                    instant = dt.fromisoformat(
                        r["timestamp"].replace("Z", "+00:00")).astimezone(UTC)
                    ds = instant.strftime("%Y-%m-%d")
                    by_utc[ds] = instant
                    inverted += r["quote"] == "USDEUR"
                    weekend += ds not in union
                for ds in union:
                    inst = by_utc[ds]
                    late += (inst.hour * 60 + inst.minute) > 16 * 60
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                assert expected["num_late_fixes"] == late, task["id"]
                assert expected["num_inverted_fix_rows"] == inverted
                assert expected["num_weekend_prints"] == weekend
                assert {"nav_ann_vol", "nav_drawdown_trough_date",
                        "eu_sleeve_weight_last_probe"} <= set(expected)
                assert any(k.startswith("us_sleeve_usd_") for k in expected)
                assert 0.0 < expected["eu_sleeve_weight_last_probe"] < 1.0

    def test_t3_applicable_correction_counts(self, dev_sets):
        import csv as csv_mod
        import re as re_mod
        from datetime import UTC
        from datetime import datetime as dt

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_reconciliation":
                    continue
                task_dir = root / task["data_dir"]
                spec = (task_dir / "spec.md").read_text()
                # The nit fix: "applicable correction" is defined, and
                # post-report-time corrections are excluded entirely.
                assert "An APPLICABLE correction" in spec
                assert "applicable: it is excluded entirely" in spec
                report = re_mod.search(r"report_time: (\S+)", spec).group(1)
                report_inst = dt.fromisoformat(
                    report.replace("Z", "+00:00")).astimezone(UTC)
                corr = list(csv_mod.DictReader(
                    open(task_dir / "corrections.csv")))
                late = sum(
                    1 for r in corr
                    if dt.fromisoformat(r["as_of"].replace("Z", "+00:00"))
                    .astimezone(UTC) > report_inst)
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                assert expected["num_correction_rows_total"] == len(corr)
                assert expected["num_correction_rows_late"] == late
                assert expected["num_conflicted_keys"] == (
                    expected["resolved_by_correction"]
                    + expected["resolved_by_vendor_a"]
                    + expected["resolved_by_median"])
                # Second probe ticker and the resolved-series risk
                # composite are reported.
                assert sum(1 for k in expected
                           if k.startswith("resolved_close_REC2_")) == 2
                assert {"probe_ann_vol", "probe_ticker_max_drawdown",
                        "probe_ticker_drawdown_trough_date"} <= set(expected)

    def test_t4_rebalance_chain_and_cost_basis(self, dev_sets):
        import csv as csv_mod

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_split_holdings":
                    continue
                task_dir = root / task["data_dir"]
                spec = (task_dir / "spec.md").read_text()
                # The nit fix: the memo states the cashed-out
                # fraction's cost basis is removed from the lot.
                assert "REMOVED from the lot" in spec
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                assert expected["num_splits"] == 6
                # Terminal rebalance identity, recomputed from the
                # key's own end state and the last closes in the file.
                prices = list(csv_mod.DictReader(
                    open(task_dir / "prices.csv")))
                # Take each ticker's close on its maximum date (rows
                # are not guaranteed to be in date order).
                last_close, best_date = {}, {}
                for r in prices:
                    t = r["ticker"]
                    if r["date"] >= best_date.get(t, ""):
                        best_date[t] = r["date"]
                        last_close[t] = float(r["close"])
                tickers = sorted(last_close)
                end_mv = sum(expected[f"shares_{t}"] * last_close[t]
                             for t in tickers)
                total = end_mv + expected["cash_in_lieu"]
                # Terminal rebalance identity (memo rule 7): floor to
                # the asset's lot size, and a sub-minimum trade is not
                # made (target stays at the end share count).
                import re as re_mod

                lot_sizes = {t: int(v) for t, v in re_mod.findall(
                    r"(\w+)=(\d+)", re_mod.search(
                        r"lot size, in shares: ([^.]+)\.", spec).group(1))}
                min_trade = int(re_mod.search(
                    r"minimum\s+trade\s+size\s+is\s+(\d+)\s+shares",
                    spec).group(1))
                assert "n is the number of distinct tickers" in spec
                assert "20 consecutive daily returns (that is, 21" in spec
                assert "index[end]/index[end-20] - 1" in spec
                for t in tickers:
                    want = int((total / len(tickers)) // last_close[t])
                    want = int(want // lot_sizes[t]) * lot_sizes[t]
                    if 0 < abs(want - expected[f"shares_{t}"]) < min_trade:
                        want = expected[f"shares_{t}"]
                    assert expected["rebalance_target_shares"][t] == want, (
                        task["id"], t)
                    assert expected["rebalance_trade_shares"][t] == (
                        want - expected[f"shares_{t}"])
                    assert want % lot_sizes[t] == 0 or \
                        want == expected[f"shares_{t}"]
                leftover = total - sum(
                    expected["rebalance_target_shares"][t] * last_close[t]
                    for t in tickers)
                assert abs(expected["rebalance_leftover_cash"] - leftover) < 1e-4
                assert {"max_drawdown_trough_date", "twr_ann_vol",
                        "worst_rolling_20d_return",
                        "worst_rolling_20d_end_date",
                        "num_cash_in_lieu_events", "num_lots_end"} <= set(expected)
                assert expected["num_cash_in_lieu_events"] >= 1

    def test_t5_constraints_and_multiperiod_state(self, dev_sets):
        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_portfolio_construction":
                    continue
                task_dir = root / task["data_dir"]
                cfg = json.loads((task_dir / "constraints.json").read_text())
                assert len(cfg["lot_sizes"]) == 10
                assert cfg["min_trade_shares"] > 0
                assert len(set(cfg["sectors"].values())) == 3
                expected = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                assert {"weights_last_rebalance", "shares_last_rebalance",
                        "total_cost_paid",
                        "num_assets_traded_last_rebalance",
                        "nav_ann_vol"} <= set(expected)
                assert expected["num_rebalances"] == len(
                    cfg["rebalance_row_indices_1based"]) == 7
                for field in ("shares_first_rebalance",
                              "shares_last_rebalance"):
                    for t, s in expected[field].items():
                        # Every share count is a whole multiple of the
                        # asset's own lot size.  (The minimum-trade rule
                        # itself is pinned by the min_trade_size_ignored
                        # gate variant below: it changes the answer on
                        # every generated task, or the task is reseeded.)
                        assert s % cfg["lot_sizes"][t] == 0, (task["id"], t)
                assert "min_trade_size_ignored" in task["trap_variants"]
                for field in ("weights_first_rebalance",
                              "weights_last_rebalance"):
                    weights = expected[field]
                    assert all(w <= cfg["per_asset_cap"] + 1e-4
                               for w in weights.values())
                    for sector, cap in cfg["sector_caps"].items():
                        ssum = sum(w for t, w in weights.items()
                                   if cfg["sectors"][t] == sector)
                        assert ssum <= cap + 2e-3, (task["id"], sector)
                assert expected["total_cost_paid"] > 0
                assert 0 <= expected[
                    "num_assets_traded_last_rebalance"] <= 10


# ---------------------------------------------------------------- check 8

def _ipm_min_variance(cov, cap, sector_members, sector_caps):
    """Independent QP solver used only by the tests.

    Primal-dual interior-point method (Mehrotra predictor-corrector)
    for min w' Sigma w subject to sum(w)=1, 0<=w<=cap and the sector
    caps, written from scratch on NumPy: a different algorithm from
    the active-set solvers in the generator and the reference, so
    agreement is a genuine cross-check of optimality.
    """
    import numpy as np

    n = cov.shape[0]
    p_mat = 2 * cov
    rows, rhs = [], []
    for i in range(n):
        r = np.zeros(n)
        r[i] = 1.0
        rows.append(r)
        rhs.append(cap)
        r = np.zeros(n)
        r[i] = -1.0
        rows.append(r)
        rhs.append(0.0)
    for s_name, mem in sector_members.items():
        r = np.zeros(n)
        r[mem] = 1.0
        rows.append(r)
        rhs.append(sector_caps[s_name])
    g_mat, h_vec = np.array(rows), np.array(rhs)
    a_mat, b_vec = np.ones((1, n)), np.array([1.0])
    m = g_mat.shape[0]
    w = np.full(n, 1.0 / n)
    w = np.minimum(w, cap * 0.99)
    w = w / w.sum()
    s = np.maximum(h_vec - g_mat @ w, 1e-4)
    z = np.full(m, 1.0)
    y = np.zeros(1)

    def direction(rhs_c, theta, rd, rp, rc):
        kkt = np.zeros((n + 1, n + 1))
        kkt[:n, :n] = p_mat + g_mat.T @ np.diag(theta) @ g_mat
        kkt[:n, n:] = a_mat.T
        kkt[n:, :n] = a_mat
        rhs_v = np.concatenate([-rd - g_mat.T @ ((rhs_c + z * rc) / s), -rp])
        sol = np.linalg.solve(kkt, rhs_v)
        dw, dy = sol[:n], sol[n:]
        ds = -rc - g_mat @ dw
        dz = (rhs_c - z * ds) / s
        return dw, dy, ds, dz

    def step_len(v, dv):
        a = 1.0
        for x, dx in zip(v, dv, strict=True):
            if dx < 0:
                a = min(a, -x / dx)
        return a

    for _ in range(200):
        rd = p_mat @ w + a_mat.T @ y + g_mat.T @ z
        rp = a_mat @ w - b_vec
        rc = g_mat @ w + s - h_vec
        mu = float(s @ z) / m
        if max(np.max(np.abs(rd)), np.max(np.abs(rp)),
               np.max(np.abs(rc)), mu) < 5e-14:
            break
        theta = z / s
        dw, dy, ds, dz = direction(-(s * z), theta, rd, rp, rc)
        ap, ad = step_len(s, ds), step_len(z, dz)
        mu_aff = float((s + ap * ds) @ (z + ad * dz)) / m
        sigma = (mu_aff / mu) ** 3
        dw, dy, ds, dz = direction(
            -(s * z) - ds * dz + sigma * mu, theta, rd, rp, rc)
        ap, ad = step_len(s, ds) * 0.99, step_len(z, dz) * 0.99
        w += ap * dw
        y += ad * dy
        s += ap * ds
        z += ad * dz
    # Exact polish (interior-point crossover): take the active set the
    # IPM identified and solve that equality-constrained QP exactly
    # with one KKT linear solve.  Accept the polished point only if it
    # is feasible with correct-sign multipliers; otherwise keep the
    # IPM iterate.  This removes the IPM's slow tail on degenerate
    # problems while staying a different method from the production
    # active-set path.
    def polish(threshold):
        fixed = {}
        for i in range(n):
            if w[i] < threshold:
                fixed[i] = 0.0
            elif cap - w[i] < threshold:
                fixed[i] = cap
        active_sec = [s_name for s_name, mem in sector_members.items()
                      if sector_caps[s_name] - sum(w[i] for i in mem)
                      < threshold]
        free = [i for i in range(n) if i not in fixed]
        if not free:
            return None
        eqs = [(list(range(n)), 1.0)] + [
            (sector_members[s_name], sector_caps[s_name])
            for s_name in active_sec]
        mm = len(eqs)
        a_eq = np.zeros((mm, len(free)))
        b_eq = np.zeros(mm)
        for k, (mem, target) in enumerate(eqs):
            b_eq[k] = target - sum(fixed[i] for i in mem if i in fixed)
            for j, i in enumerate(free):
                if i in mem:
                    a_eq[k, j] = 1.0
        kkt = np.zeros((len(free) + mm, len(free) + mm))
        kkt[:len(free), :len(free)] = p_mat[np.ix_(free, free)]
        kkt[:len(free), len(free):] = a_eq.T
        kkt[len(free):, :len(free)] = a_eq
        fixed_idx = [i for i in range(n) if i in fixed]
        grad_fixed = (p_mat[np.ix_(free, fixed_idx)]
                      @ np.array([fixed[i] for i in fixed_idx])) \
            if fixed_idx else np.zeros(len(free))
        rhs = np.concatenate([-grad_fixed, b_eq])
        try:
            sol = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            sol = None
        if sol is not None:
            cand = np.zeros(n)
            for i, v in fixed.items():
                cand[i] = v
            for i, v in zip(free, sol[:len(free)], strict=True):
                cand[i] = v
            lam = sol[len(free):]
            grad = p_mat @ cand
            at_lam = np.zeros(n)
            for k, (mem, _t) in enumerate(eqs):
                for i in mem:
                    at_lam[i] += lam[k]
            ok = abs(float(cand.sum()) - 1.0) < 1e-9 \
                and float(cand.min()) >= -1e-9 \
                and float(cand.max()) <= cap + 1e-9 \
                and all(sum(cand[i] for i in mem)
                        <= sector_caps[s_name] + 1e-9
                        for s_name, mem in sector_members.items()) \
                and all(lam[1 + k] >= -1e-9
                        for k in range(len(active_sec))) \
                and all((grad[i] + at_lam[i] >= -1e-9) if v == 0.0
                        else (grad[i] + at_lam[i] <= 1e-9)
                        for i, v in fixed.items())
            if ok:
                return cand
        return None
    best = None
    for threshold in (1e-4, 1e-5, 1e-6, 1e-7):
        cand = polish(threshold)
        if cand is not None and (best is None
                                 or float(cand @ p_mat @ cand)
                                 < float(best @ p_mat @ best)):
            best = cand
    return best if best is not None else w


class TestOptimisationHardening:
    """Round-5 hardening (seed 7601 pilot: Spark 1.3 24/30 with T5 at
    0/6 and T1-T4 saturated): every family gains a field that needs a
    small constrained optimisation or a stateful multi-period
    simulation, the T5 optimiser is certified exact, and the tier's
    default turn budget rises to 16."""

    def test_t5_key_weights_match_independent_solver(self, dev_sets):
        import csv as csv_mod

        import numpy as np

        checked = 0
        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_portfolio_construction":
                    continue
                task_dir = root / task["data_dir"]
                cfg = json.loads((task_dir / "constraints.json").read_text())
                closes = {}
                for r in csv_mod.DictReader(open(task_dir / "prices.csv")):
                    closes.setdefault(r["ticker"], []).append(float(r["close"]))
                assets = sorted(closes)
                members = {}
                for t in assets:
                    members.setdefault(cfg["sectors"][t], []).append(
                        assets.index(t))
                rets_all = {
                    t: [b / a - 1 for a, b in zip(closes[t], closes[t][1:], strict=False)]
                    for t in assets}
                look = cfg["lookback_returns"]
                key = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                for field, idx1 in (
                        ("weights_first_rebalance",
                         cfg["rebalance_row_indices_1based"][0]),
                        ("weights_last_rebalance",
                         cfg["rebalance_row_indices_1based"][-1])):
                    end = idx1 - 1  # 1-based row -> 0-based index
                    rets = np.array(
                        [rets_all[t][end - look:end] for t in assets]).T
                    cov = np.cov(rets, rowvar=False, ddof=1)
                    delta = cfg["shrinkage_delta"]
                    cov = (1 - delta) * cov + delta * np.diag(np.diag(cov))
                    w_key = np.array([key[field][t] for t in assets])
                    # Primal feasibility of the stored key.
                    assert abs(float(w_key.sum()) - 1.0) < 1e-9
                    assert float(w_key.min()) >= -1e-12
                    assert float(w_key.max()) <= cfg["per_asset_cap"] + 1e-9
                    for s_name, mem in members.items():
                        assert sum(w_key[i] for i in mem) \
                            <= cfg["sector_caps"][s_name] + 1e-9
                    # The independent interior-point optimum agrees to
                    # 1e-7 (the old solver sat ~2.1e-5 away).
                    w_ipm = _ipm_min_variance(
                        cov, cfg["per_asset_cap"], members,
                        cfg["sector_caps"])
                    diff = float(np.max(np.abs(w_key - w_ipm)))
                    assert diff <= 1e-7, (task["id"], field, diff)
                    # KKT certificate: multipliers on the active set
                    # reproduce the gradient with the right signs.
                    grad = 2 * cov @ w_key
                    cols = [np.ones(len(assets))]
                    for s_name, mem in members.items():
                        if cfg["sector_caps"][s_name] - sum(
                                w_key[i] for i in mem) < 1e-7:
                            c = np.zeros(len(assets))
                            c[mem] = 1.0
                            cols.append(c)
                    for i in range(len(assets)):
                        if w_key[i] < 1e-7:
                            c = np.zeros(len(assets))
                            c[i] = -1.0
                            cols.append(c)
                        elif cfg["per_asset_cap"] - w_key[i] < 1e-7:
                            c = np.zeros(len(assets))
                            c[i] = 1.0
                            cols.append(c)
                    mat = np.array(cols).T
                    mult, *_ = np.linalg.lstsq(mat, -grad, rcond=None)
                    resid = float(np.max(np.abs(mat @ mult + grad)))
                    assert resid <= 1e-8, (task["id"], field, resid)
                    assert float(mult[1:].min(initial=0.0)) >= -1e-7
                    checked += 1
        assert checked == 120

    def test_t1_bounded_hedge_ratio(self, dev_sets):
        import csv as csv_mod
        import math as math_mod
        import re as re_mod

        binds = 0
        total = 0
        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_convention_risk":
                    continue
                task_dir = root / task["data_dir"]
                spec = (task_dir / "spec.md").read_text()
                mh = re_mod.search(
                    r"- hedge: asset=(\w+); benchmark=(\w+); "
                    r"window_returns=(\d+); return=(\w+); "
                    r"lower_bound=([\d.]+); upper_bound=([\d.]+)", spec)
                asset, bench = mh.group(1), mh.group(2)
                window = int(mh.group(3))
                lo, hi = float(mh.group(5)), float(mh.group(6))
                trading = {r["date"] for r in csv_mod.DictReader(
                    open(task_dir / "calendar.csv"))}
                series = {}
                for r in csv_mod.DictReader(open(task_dir / "prices.csv")):
                    if r["date"] in trading:
                        series.setdefault(r["ticker"], {})[r["date"]] = \
                            float(r["close"])
                def log_rets(t, _series=series):
                    ds = sorted(_series[t])
                    cs = [_series[t][d] for d in ds]
                    return [math_mod.log(b / a)
                            for a, b in zip(cs, cs[1:], strict=False)]
                ra = log_rets(asset)[-window:]
                rb = log_rets(bench)[-window:]
                ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
                cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True)) \
                    / (len(ra) - 1)
                var = sum((y - mb) ** 2 for y in rb) / (len(rb) - 1)
                unc = cov / var
                want = min(max(unc, lo), hi)
                key = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                assert abs(key["hedge_ratio"] - want) < 1e-6, task["id"]
                hedged = [a - want * b for a, b in zip(ra, rb, strict=True)]
                mh_ = sum(hedged) / len(hedged)
                sd = math_mod.sqrt(
                    sum((x - mh_) ** 2 for x in hedged) / (len(hedged) - 1))
                assert abs(key["hedged_ann_vol"]
                           - sd * math_mod.sqrt(252)) < 1e-6
                assert "hedge_ratio_unbounded" in task["trap_variants"]
                total += 1
                binds += want == hi and unc > hi
        assert total == 60
        # The construction picks the highest-beta asset, so the upper
        # bound genuinely binds on most tasks (else the gate variant
        # could not discriminate and the task would be reseeded).
        assert binds >= 40, binds

    def test_t2_monthly_hedge_resets(self, dev_sets):
        import csv as csv_mod

        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_fx_calendar":
                    continue
                task_dir = root / task["data_dir"]
                union = sorted(
                    {r["date"] for r in csv_mod.DictReader(
                        open(task_dir / "us_prices.csv"))}
                    | {r["date"] for r in csv_mod.DictReader(
                        open(task_dir / "eu_prices.csv"))})
                months = {d[:7] for d in union}
                key = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                assert key["num_hedge_resets"] == len(months), task["id"]
                assert {"hedge_pnl_total", "hedged_nav_ann_vol",
                        "hedged_nav_max_drawdown"} <= set(key)
                assert key["hedge_pnl_total"] != 0.0
                assert key["hedged_nav_max_drawdown"] <= 0.0
                assert "hedge_never_reset" in task["trap_variants"]
                spec = (task_dir / "spec.md").read_text()
                assert "Monthly FX hedge" in spec

    def test_t3_risk_parity_weights(self, dev_sets):
        import csv as csv_mod
        import math as math_mod
        import re as re_mod
        from datetime import UTC
        from datetime import datetime as dt

        def instant(stamp):
            return dt.fromisoformat(
                stamp.replace("Z", "+00:00")).astimezone(UTC)

        not_equal_weight = 0
        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_reconciliation":
                    continue
                task_dir = root / task["data_dir"]
                spec = (task_dir / "spec.md").read_text()
                rp_tickers = re_mod.search(
                    r"risk_parity_tickers: (.+)", spec).group(1).split(", ")
                assert len(rp_tickers) == 3
                report = instant(re_mod.search(
                    r"report_time: (\S+)", spec).group(1))
                # Compact independent resolution (memo rules 0-5).
                def first_wins(name, value_fn, _dir=task_dir):
                    out, status = {}, {}
                    for r in csv_mod.DictReader(open(_dir / name)):
                        k = (r["ticker"], r["date"])
                        if k not in out:
                            out[k] = value_fn(r)
                            status[k] = r.get("status")
                    return out, status
                a_val, a_status = first_wins(
                    "vendor_a.csv", lambda r: float(r["close"]))
                b_val, _ = first_wins(
                    "vendor_b.csv",
                    lambda r: float(r["close"]) / 100
                    if r["unit"] == "cents" else float(r["close"]))
                corr = {}
                for r in csv_mod.DictReader(open(task_dir / "corrections.csv")):
                    when = instant(r["as_of"])
                    k = (r["ticker"], r["date"])
                    if when <= report and (
                            k not in corr or when > corr[k][1]):
                        corr[k] = (float(r["close"]), when)
                resolved = {}
                for k in sorted(set(a_val) | set(b_val)):
                    av, bv = a_val.get(k), b_val.get(k)
                    if k in corr:
                        resolved[k] = corr[k][0]
                    elif av is not None and a_status.get(k) != "indicative":
                        resolved[k] = av
                    else:
                        avail = sorted(v for v in (av, bv)
                                       if v is not None)
                        resolved[k] = avail[len(avail) // 2] \
                            if len(avail) % 2 else (avail[0] + avail[-1]) / 2
                by = {t: {d: v for (tt, d), v in resolved.items() if tt == t}
                      for t in rp_tickers}
                days = sorted(set.intersection(
                    *[set(by[t]) for t in rp_tickers]))
                rets = {t: [math_mod.log(by[t][b] / by[t][a])
                            for a, b in zip(days, days[1:], strict=False)]
                        for t in rp_tickers}
                m = len(days) - 1
                means = {t: sum(rets[t]) / m for t in rp_tickers}
                cov = {(a, b): sum(
                    (rets[a][k] - means[a]) * (rets[b][k] - means[b])
                    for k in range(m)) / (m - 1)
                    for a in rp_tickers for b in rp_tickers}
                key = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                w = {t: key[f"risk_parity_weight_{t}"] for t in rp_tickers}
                assert abs(sum(w.values()) - 1.0) < 1e-5
                assert all(0.0 < x < 1.0 for x in w.values())
                # Equal risk contributions on the independent series.
                rc = {a: w[a] * sum(cov[(a, b)] * w[b] for b in rp_tickers)
                      for a in rp_tickers}
                assert max(rc.values()) - min(rc.values()) < 1e-6, (
                    task["id"], rc)
                not_equal_weight += any(
                    abs(x - 1.0 / 3.0) > 1e-3 for x in w.values())
                assert "risk_parity_equal_weights" in task["trap_variants"]
                assert "risk_parity_inverse_vol" in task["trap_variants"]
        # Risk parity is a real optimisation here, not 1/3 weights.
        assert not_equal_weight >= 50, not_equal_weight

    def test_t4_terminal_rebalance_variants(self, dev_sets):
        for root in dev_sets.values():
            for task in _tasks(root):
                if task["type"] != "tools_frontier_split_holdings":
                    continue
                assert "terminal_min_trade_ignored" in task["trap_variants"]
                assert "terminal_lots_ignored" in task["trap_variants"]
                # The gate only passes when the minimum-trade rule
                # actually bites: some asset's lot-floored target must
                # have been reverted to its end share count.
                key = json.loads(
                    (root / "answers" / f"{task['id']}_answer.json").read_text())
                variant = task["trap_variants"][
                    "terminal_min_trade_ignored"]
                assert variant["rebalance_trade_shares"] != \
                    key["rebalance_trade_shares"], task["id"]

    def test_tools_frontier_default_max_turns_16(self, monkeypatch, tmp_path):
        import sys

        from src import runner as runner_mod
        from src import tool_runner as tool_mod

        class FakeToolRunner:
            seen = {}

            def __init__(self, **kwargs):
                FakeToolRunner.seen.update(kwargs)

            def run_all_tasks(self, **kwargs):
                return []

        for tier, want in (("tools_frontier", 16), ("frontier", 12)):
            FakeToolRunner.seen = {}
            seen = FakeToolRunner.seen
            monkeypatch.setattr(tool_mod, "ToolTaskRunner", FakeToolRunner)
            tasks = tmp_path / f"tasks_{tier}"
            tasks.mkdir(exist_ok=True)
            monkeypatch.setattr(sys, "argv", [
                "runner", "--dry-run", "--tools", "python",
                "--tier", tier, "--tasks-dir", str(tasks),
                "--output-dir", str(tmp_path / f"out_{tier}")])
            assert runner_mod.main() == 0
            assert seen["max_turns"] == want, (tier, seen["max_turns"])
