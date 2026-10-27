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
                                       "reconciled_total_return",
                                       "reconciled_max_drawdown", "num_tickers",
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
        # Pilot-calibrated F1/F3 sizes (see FRONTIER_FAMILY_SETTINGS):
        # tractable enough that a strong model actually computes, with
        # difficulty in the data problems and rolling semantics.
        assert FRONTIER_FAMILY_SETTINGS["rolling_risk"] == \
            {"rows": 120, "assets": 2, "window": 10}
        assert FRONTIER_FAMILY_SETTINGS["vendor_reconciliation"]["conflicts"] == 12
        assert FRONTIER_FAMILY_SETTINGS["factor_exposure"] == \
            {"rows": 120, "actions": 1}
        assert FRONTIER_FAMILY_SETTINGS["portfolio_pipeline"]["rebalance_every"] == 21

    def test_f3_has_exactly_one_split_event(self):
        for seed in range(1, 41):
            gen = FrontierTaskGenerator(seed=seed)
            task, _ = gen.generate_task_factor_exposure()
            assert task["expected"]["num_splits"] == 1

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


def _score_makers_default(seed, count):
    """Generate ``count`` tasks at shipped defaults and score the solver."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "answers").mkdir()
        gen = FrontierTaskGenerator(seed=seed)  # shipped default settings
        gens = gen.task_generators()
        scorer = TaskScorer(out / "answers")
        results = []
        for i in range(count):
            task, payload = gens[i % len(gens)]()
            (out / "answers" / f"{task['id']}_answer.json").write_text(
                json.dumps(task["expected"]))
            data = out / "data.csv"
            data.write_text(payload)
            got = solve(task, data)
            results.append((task, scorer.score_task(
                task["id"], json.dumps(got))))
    return results


class TestIndependentSolver:
    def test_solver_scores_100_percent_seeds_1_to_400_default_settings(self):
        # Shipped default settings and the full default task count (25).
        for seed in range(1, 401):
            for task, result in _score_makers_default(seed, 25):
                assert result["passed"] and result["score"] == 1.0, \
                    (seed, task["id"], result["feedback"])
                assert result["analysis_flags"] == [], \
                    (seed, task["id"], result["analysis_flags"])

    def test_reference_answers_produce_no_no_attempt_flags_seeds_1_to_400(self):
        # The reference (expected) answers themselves must never look
        # like placeholders: 10,000 tasks at shipped defaults.
        from src.scorer import _is_no_attempt
        for seed in range(1, 401):
            gen = FrontierTaskGenerator(seed=seed)
            gens = gen.task_generators()
            for i in range(25):
                task, _ = gens[i % len(gens)]()
                assert not _is_no_attempt(task["expected"]), \
                    (seed, task["id"], task["expected"])

    @pytest.mark.parametrize(("seed", "count"), [(184, 3), (334, 7)])
    def test_injector_collision_regressions(self, seed, count):
        # Seed 184: a missing day deleted one copy of a duplicate.
        # Seed 334: a bad price sat on the second copy of a duplicate.
        for task, result in _score_makers_default(seed, count):
            assert result["passed"] and result["score"] == 1.0, \
                (seed, task["id"], result["feedback"])


class TestNoRepeatedPrices:
    """No repeated consecutive closes after cleaning, and consistent OHLC."""

    def test_cleaned_series_have_no_repeats_and_bars_are_consistent(self):
        from tests.frontier_solver import cleaned_frame, read_blocks
        for seed in range(1, 9):
            gen = FrontierTaskGenerator(seed=seed)  # shipped defaults
            gens = gen.task_generators()
            with tempfile.TemporaryDirectory() as tmp:
                data = Path(tmp) / "data.csv"
                for i in range(25):
                    task, payload = gens[i % len(gens)]()
                    data.write_text(payload)
                    frame = cleaned_frame(task, data)
                    for col in frame.columns:
                        vals = frame[col].dropna().tolist()
                        assert all(a != b for a, b in
                                   zip(vals, vals[1:], strict=False)), \
                            (seed, task["id"], col, "repeated close")
                    for block in read_blocks(data).values():
                        if "open" not in block.columns:
                            continue  # the split table has no bars
                        assert (block["low"] <= block["open"]).all()
                        assert (block["low"] <= block["close"]).all()
                        assert (block["high"] >= block["open"]).all()
                        assert (block["high"] >= block["close"]).all()

    def test_f3_unadjusted_prices_stay_above_five_dollars(self):
        from tests.frontier_solver import _bad_mask, read_blocks
        for seed in range(1, 21):
            gen = FrontierTaskGenerator(seed=seed)
            task, payload = gen.generate_task_factor_exposure()
            with tempfile.TemporaryDirectory() as tmp:
                data = Path(tmp) / "data.csv"
                data.write_text(payload)
                prices = read_blocks(data)["PRICES"]
                ast = prices[prices["ticker"] == "AST"].sort_values("date")
                closes = [float(c) for c in ast["close"]]
                bad = _bad_mask(closes)
                genuine = [c for c, b in zip(closes, bad, strict=True) if not b]
                assert min(genuine) >= 4.5, (seed, min(genuine))


class TestF2CentsConsistency:
    """No A/B pair may differ by exactly 1 cent; conflicts use integer cents."""

    def test_no_exactly_one_cent_pairs_seeds_1_to_200(self):
        from tests.frontier_solver import _frames, clean, read_blocks
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.csv"
            for seed in range(1, 201):
                gen = FrontierTaskGenerator(seed=seed)  # shipped defaults
                task, payload = gen.generate_task_vendor_reconciliation()
                data.write_text(payload)
                blocks = read_blocks(data)
                a_frames = _frames(blocks["VENDOR_A"])
                for ticker, b_group in blocks["VENDOR_B"].groupby("ticker"):
                    a = clean(a_frames[ticker])
                    b = clean(b_group.assign(close=b_group["close"] / 100.0))
                    for d in set(a.index) & set(b.index):
                        diff = abs(round(float(a[d]) * 100)
                                   - round(float(b[d]) * 100))
                        assert diff != 1, (seed, ticker, d, diff)

    @pytest.mark.parametrize(("seed", "task_id"), [
        (72, "task_f_2018"), (119, "task_f_2006"),
        (212, "task_f_2018"), (286, "task_f_2006")])
    def test_flagged_seed_keys_are_correct(self, seed, task_id):
        # The repeat nudge next to the shared missing day used to open an
        # exact 1-cent A/B gap that float comparison counted as a conflict.
        for task, result in _score_makers_default(seed, 25):
            if task["id"] == task_id:
                assert result["passed"] and result["score"] == 1.0, \
                    (seed, task_id, result["feedback"])
                return
        raise AssertionError(f"{task_id} not generated for seed {seed}")


class TestF2ShortcutFails:
    """First/last-close shortcuts score 0%: the drawdown core needs the path."""

    def test_endpoint_only_drawdown_never_passes(self):
        checked = 0
        for seed in range(1, 21):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen(seed, SMALL, 4, tmp)
                scorer = TaskScorer(out / "answers")
                for task in tasks:
                    if task["type"] != "frontier_vendor_reconciliation":
                        continue
                    key = json.loads((out / "answers" /
                                      f"{task['id']}_answer.json").read_text())
                    got = solve(task, out / task["data_file"])
                    # What a first/last-only reader can report for the
                    # drawdown: the two-point path [1, final value].
                    shortcut = min(0.0, got["reconciled_total_return"])
                    if abs(shortcut - key["reconciled_max_drawdown"]) <= 1e-3:
                        continue  # coincidentally right; not a shortcut win
                    got["reconciled_max_drawdown"] = round(shortcut, 6)
                    result = scorer.score_task(task["id"], json.dumps(got))
                    assert not result["passed"], (seed, task["id"])
                    checked += 1
        assert checked > 0


class TestCoreFieldRequired:
    """Correct codes with constant numbers must pass 0% (core is required)."""

    CORE = ("mean_betas", "reconciled_total_return", "reconciled_max_drawdown",
            "beta_market", "beta_sector", "final_return")

    def _core_values(self, expected):
        vals = []
        for field in self.CORE:
            if field in expected:
                v = expected[field]
                vals.extend(v.values() if isinstance(v, dict) else [v])
        return vals

    @pytest.mark.parametrize("constant", [0.0, 1.0, 0.123456])
    def test_constant_numbers_pass_zero(self, constant):
        checked = 0
        for seed in range(1, 21):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen(seed, SMALL, 4, tmp)
                scorer = TaskScorer(out / "answers")
                for task in tasks:
                    key = json.loads((out / "answers" /
                                      f"{task['id']}_answer.json").read_text())
                    # A constant that happens to equal the true core value
                    # is a correct answer for that field, not gaming.
                    if any(abs(constant - v) <= 1e-4
                           for v in self._core_values(key)):
                        continue
                    got = solve(task, out / task["data_file"])
                    for field, value in list(got.items()):
                        if isinstance(value, float):
                            got[field] = constant
                        elif isinstance(value, dict):
                            got[field] = dict.fromkeys(value, constant)
                    result = scorer.score_task(task["id"], json.dumps(got))
                    assert not result["passed"], \
                        (seed, task["id"], constant, result["score"])
                    checked += 1
        assert checked > 0


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

    def test_reconciled_max_drawdown_tolerance_is_1e_3(self, tmp_path):
        # The path statistic accumulates over the whole series, so it is
        # scored at 1e-3; other floats stay at 1e-4 and the endpoint-only
        # shortcut (off by far more) still fails the core check.
        answer = {"mismatch_count": 3, "first_conflict_date": "2024-02-01",
                  "reconciled_total_return": 0.1,
                  "reconciled_max_drawdown": -0.05, "num_tickers": 3,
                  "num_aligned_days": 100,
                  "problems_found": ["source_conflict_x"]}
        (tmp_path / "task_f_2003_answer.json").write_text(json.dumps(answer))
        scorer = TaskScorer(tmp_path)
        good = {k: v for k, v in answer.items() if k != "problems_found"}
        good["problems"] = ["source_conflict"]
        close = {**good, "reconciled_max_drawdown": -0.0495}  # 5e-4 off
        assert scorer.score_task("task_f_2003", json.dumps(close))["passed"]
        far = {**good, "reconciled_max_drawdown": -0.048}  # 2e-3 off
        result = scorer.score_task("task_f_2003", json.dumps(far))
        assert result["component_scores"]["reconciled_max_drawdown"] == 0.0
        assert not result["passed"]

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
        # Same method as the other tiers, at shipped defaults over the
        # full default task count (25 tasks per seed).
        by_type = defaultdict(list)
        for seed in range(1, 41):
            gen = FrontierTaskGenerator(seed=seed)
            gens = gen.task_generators()
            for i in range(25):
                task, _ = gens[i % len(gens)]()
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
        assert fixed / total <= 0.12
        assert paste == 0


class TestPromptSizeScript:
    def test_prompt_size_script_reports_max_per_family(self, tmp_path):
        import subprocess
        import sys
        tasks, out = _gen(42, SMALL, 4, tmp_path)
        repo = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [sys.executable, str(repo / "scripts" / "prompt_sizes.py"),
             "--tasks-dir", str(out)],
            capture_output=True, text=True, cwd=repo, check=True)
        for task_type in FIELDS:
            assert task_type in proc.stdout
        assert "overall max_chars" in proc.stdout
