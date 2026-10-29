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
                        by_ticker[row["ticker"]].append(float(row["close"]))
                    for ticker, closes in by_ticker.items():
                        assert len(closes) >= 100, (task["id"], name, ticker)
                        assert all(a != b for a, b in
                                   zip(closes, closes[1:], strict=False)), \
                            (task["id"], name, ticker)

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
