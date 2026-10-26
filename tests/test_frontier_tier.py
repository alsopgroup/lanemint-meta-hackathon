"""Frontier tier: settings, prompts, independent-solver proof, baselines."""

import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.frontier_tier import CODES_13, FRONTIER_FAMILY_SETTINGS, FrontierTaskGenerator
from src.scorer import TaskScorer, categories_for_items
from tests.frontier_solver import solve

SMALL = {
    "rolling_risk": {"rows": 110, "assets": 3, "window": 15},
    "vendor_reconciliation": {"rows": 90, "assets": 3, "conflicts": 6},
    "factor_exposure": {"rows": 120, "actions": 2},
    "portfolio_pipeline": {"rows": 110, "assets": 3,
                           "rebalance_every": 10, "cost_bps": 10},
}

OFF_LIMITS = ["delist", "survivorship", "rename", "dividend", "stale",
              "repeated_price", "too_short", "insufficient_history"]

FIELDS = {
    "frontier_rolling_risk": ["mean_betas", "peak_correlation",
                              "peak_correlation_asset", "peak_correlation_date",
                              "final_window_volatility", "num_assets",
                              "num_aligned_days"],
    "frontier_vendor_reconciliation": ["mismatch_count", "first_conflict_date",
                                       "reconciled_total_return", "num_tickers",
                                       "num_aligned_days"],
    "frontier_factor_exposure": ["alpha", "beta_market", "beta_sector",
                                 "r_squared", "num_return_days", "num_splits"],
    "frontier_portfolio_pipeline": ["num_aligned_days", "num_assets",
                                    "num_rebalances", "first_rebalance_value",
                                    "final_return", "max_drawdown"],
}


def _gen(seed, settings, count, tmp):
    gen = FrontierTaskGenerator(seed=seed, settings=settings)
    tasks = gen.generate_all_tasks(Path(tmp), num_tasks=count)
    return tasks, Path(tmp)


def _score_all(tasks, out):
    scorer = TaskScorer(out / "answers")
    results = []
    for task in tasks:
        got = solve(task, out / task["data_file"])
        results.append((task, scorer.score_task(task["id"], json.dumps(got))))
    return results


class TestSettings:
    def test_defaults_taken_from_plan(self):
        assert FRONTIER_FAMILY_SETTINGS["rolling_risk"] == \
            {"rows": 500, "assets": 4, "window": 60}
        assert 250 <= FRONTIER_FAMILY_SETTINGS["rolling_risk"]["rows"] <= 750
        assert FRONTIER_FAMILY_SETTINGS["vendor_reconciliation"]["conflicts"] == 12
        assert FRONTIER_FAMILY_SETTINGS["factor_exposure"]["actions"] == 3
        assert FRONTIER_FAMILY_SETTINGS["portfolio_pipeline"]["rebalance_every"] == 21

    def test_settings_override_reaches_prompt(self):
        gen = FrontierTaskGenerator(seed=1, settings=SMALL)
        task, _ = gen.generate_task_rolling_risk()
        assert "window of\n15" in task["prompt"] or "window of 15" in task["prompt"]


class TestGeneration:
    def test_reproducibility(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            t1, d1 = _gen(42, SMALL, 4, a)
            t2, d2 = _gen(42, SMALL, 4, b)
            assert t1 == t2
            for t in t1:
                assert (d1 / t["data_file"]).read_bytes() == \
                    (d2 / t["data_file"]).read_bytes()

    def test_task_ids_and_types(self):
        gen = FrontierTaskGenerator(seed=42, settings=SMALL)
        for maker in gen.task_generators():
            task, _ = maker()
            assert task["id"].startswith("task_f_")
            assert task["type"] in FIELDS

    def test_prompts_define_every_code_and_field(self):
        gen = FrontierTaskGenerator(seed=7, settings=SMALL)
        for maker in gen.task_generators():
            task, _ = maker()
            prompt = task["prompt"]
            for code in CODES_13:
                assert f"- {code}:" in prompt, (task["type"], code)
            for field in FIELDS[task["type"]]:
                assert field in prompt, (task["type"], field)
            assert "A-for-B" in prompt and "A new shares for B old shares" in prompt
            assert "problems" in prompt

    def test_f3_prompt_states_split_table_convention(self):
        gen = FrontierTaskGenerator(seed=7, settings=SMALL)
        task, payload = gen.generate_task_factor_exposure()
        assert "SPLIT_TABLE" in payload and "ratio" in task["prompt"]
        assert "multiply every AST close with date < D by B/A" in \
            " ".join(task["prompt"].split())

    def test_no_off_limits_or_hidden_splits(self):
        for seed in range(1, 31):
            gen = FrontierTaskGenerator(seed=seed, settings=SMALL)
            for maker in gen.task_generators():
                task, payload = maker()
                cats = categories_for_items(task["expected"]["problems_found"])
                for problem in task["expected"]["problems_found"]:
                    for word in OFF_LIMITS:
                        assert word not in problem.lower()
                if task["type"] == "frontier_factor_exposure":
                    if cats & {"split", "reverse_split"}:
                        assert "SPLIT_TABLE" in payload
                else:
                    # Splits appear ONLY as F3 split-table events; no
                    # hidden unadjusted-price traps elsewhere.
                    assert not cats & {"split", "reverse_split", "unit_change"}, \
                        (seed, task["type"], cats)


class TestCli:
    def test_generate_cli_frontier(self, monkeypatch, tmp_path):
        import sys

        from src import generate_tasks
        out = tmp_path / "frontier_out"
        monkeypatch.setattr(sys, "argv", ["gen", "--tier", "frontier",
                                          "--count", "4",
                                          "--output-dir", str(out)])
        generate_tasks.main()
        tasks = json.loads((out / "tasks.json").read_text())
        assert len(tasks) == 4
        assert all(t["id"].startswith("task_f_") for t in tasks)


class TestIndependentSolver:
    def test_solver_scores_100_percent_seeds_1_to_200(self):
        for seed in range(1, 201):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen(seed, SMALL, 4, tmp)
                for task, result in _score_all(tasks, out):
                    assert result["passed"] and result["score"] == 1.0, \
                        (seed, task["id"], result["feedback"])

    def test_solver_default_settings_seed42(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, out = _gen(42, None, 8, tmp)
            for task, result in _score_all(tasks, out):
                assert result["passed"] and result["score"] == 1.0, \
                    (task["id"], result["feedback"])


class TestScorerFrontierFields:
    def test_source_conflict_code(self):
        assert categories_for_items(["source_conflict_12_pairs"]) == {"source_conflict"}
        assert categories_for_items(["source conflict between vendors"]) == \
            {"source_conflict"}

    def test_checkpoint_score_reported(self, tmp_path):
        answer = {"num_aligned_days": 100, "num_assets": 3, "num_rebalances": 4,
                  "first_rebalance_value": 1.01, "final_return": 0.05,
                  "max_drawdown": -0.1, "problems_found": ["duplicate_row_x"]}
        (tmp_path / "task_f_2001_answer.json").write_text(json.dumps(answer))
        scorer = TaskScorer(tmp_path)
        good = scorer.score_task("task_f_2001", json.dumps(
            {**answer, "problems": ["duplicate"]}))
        assert good["passed"] and good["checkpoint_score"] == 1.0
        half = scorer.score_task("task_f_2001", json.dumps(
            {**answer, "final_return": 0.5, "max_drawdown": -0.9,
             "problems": ["duplicate"]}))
        assert half["checkpoint_score"] == pytest.approx(0.6)

    def test_date_and_dict_fields_exact_and_toleranced(self, tmp_path):
        answer = {"mean_betas": {"AST1": 1.0}, "peak_correlation": 0.5,
                  "peak_correlation_asset": "AST1",
                  "peak_correlation_date": "2024-01-31",
                  "final_window_volatility": 0.2, "num_assets": 1,
                  "num_aligned_days": 50, "problems_found": ["missing_day_x"]}
        (tmp_path / "task_f_2002_answer.json").write_text(json.dumps(answer))
        scorer = TaskScorer(tmp_path)
        good = {k: v for k, v in answer.items() if k != "problems_found"}
        good["problems"] = ["missing"]
        assert scorer.score_task("task_f_2002", json.dumps(good))["passed"]
        bad = {**good, "peak_correlation_date": "2024-02-01"}
        result = scorer.score_task("task_f_2002", json.dumps(bad))
        assert result["component_scores"]["peak_correlation_date"] == 0.0
        worse = {**good, "peak_correlation_date": "2024-02-01",
                 "peak_correlation_asset": "AST2"}
        assert not scorer.score_task("task_f_2002", json.dumps(worse))["passed"]


class TestGuessBaselines:
    """Fixed per-type guess and paste-all must stay near 0% on frontier."""

    def test_frontier_guess_baselines(self):
        by_type = defaultdict(list)
        for seed in range(1, 121):
            gen = FrontierTaskGenerator(seed=seed, settings=SMALL)
            for maker in gen.task_generators():
                task, _ = maker()
                by_type[task["type"]].append(frozenset(
                    categories_for_items(task["expected"]["problems_found"])))
        total = fixed = paste = 0
        for task_type, sets in by_type.items():
            most_common, count = Counter(sets).most_common(1)[0]
            assert count / len(sets) <= 0.15, (task_type, most_common, count)
            fixed += count
            paste += sum(1 for s in sets if s == frozenset(
                categories_for_items(CODES_13)))
            total += len(sets)
        assert fixed / total <= 0.15
        assert paste == 0
