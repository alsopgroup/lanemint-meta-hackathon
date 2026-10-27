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
    "frontier_rolling_risk": ["last_window_beta", "target_asset",
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
        assert "final 15 consecutive" in " ".join(task["prompt"].split())


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

    def test_f1_prompt_names_target_and_has_no_peak_search(self):
        gen = FrontierTaskGenerator(seed=7, settings=SMALL)
        task, _ = gen.generate_task_rolling_risk()
        assert "peak" not in task["prompt"].lower()
        assert f"target asset is {task['expected']['target_asset']}" in \
            " ".join(task["prompt"].split())
        assert task["expected"]["target_asset"] in ("AST1", "AST2", "AST3")
        assert set(task["expected"]) == {
            "last_window_beta", "target_asset", "final_window_volatility",
            "num_assets", "num_aligned_days", "problems_found"}
        flat = " ".join(task["prompt"].split())
        assert "Consider only the last window" in flat
        assert "not a full-sample regression" in flat

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


def _score_makers_default(seed, count, tamper=None):
    """Generate ``count`` tasks at shipped defaults and score the solver.

    ``tamper(task, got, data_path)`` may rewrite the solver's answer
    before scoring (gaming baselines); returning None leaves the answer
    unchanged and marks the task as not tampered. It runs after the
    answer key is written."""
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
            if tamper is not None:
                rewritten = tamper(task, got, data)
                if rewritten is not None:
                    got = rewritten
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

    CORE = ("last_window_beta", "reconciled_total_return",
            "reconciled_max_drawdown",
            "beta_market", "beta_sector", "final_return", "max_drawdown")

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

    def test_f3_beta_tolerance_is_1e_3(self, tmp_path):
        # Pilot near-misses landed 3e-4-1e-3 off the exact OLS betas, so
        # beta_market/beta_sector are scored at 1e-3; alpha stays at
        # 1e-4, r_squared is a core field at 1e-5 (log-returns
        # discriminator), and the betas remain core fields.
        answer = {"alpha": 0.0002, "beta_market": 1.22116,
                  "beta_sector": 0.41059, "r_squared": 0.83,
                  "num_return_days": 100, "num_splits": 1,
                  "problems_found": ["split_x"]}
        (tmp_path / "task_f_2004_answer.json").write_text(json.dumps(answer))
        scorer = TaskScorer(tmp_path)
        good = {k: v for k, v in answer.items() if k != "problems_found"}
        good["problems"] = ["split"]
        close = {**good, "beta_market": 1.22148,  # 3.2e-4 off
                 "beta_sector": 0.40963}          # 9.6e-4 off
        assert scorer.score_task("task_f_2004", json.dumps(close))["passed"]
        half_off = {**good, "beta_market": 1.22166}  # 5e-4 off
        assert scorer.score_task(
            "task_f_2004", json.dumps(half_off))["passed"]
        far = {**good, "beta_market": 1.22316}  # 2e-3 off
        result = scorer.score_task("task_f_2004", json.dumps(far))
        assert result["component_scores"]["beta_market"] == 0.0
        assert not result["passed"]

    def test_f1_fields_exact_and_toleranced(self, tmp_path):
        answer = {"last_window_beta": 1.0, "target_asset": "AST1",
                  "final_window_volatility": 0.2, "num_assets": 2,
                  "num_aligned_days": 50, "problems_found": ["missing_day_x"]}
        (tmp_path / "task_f_2002_answer.json").write_text(json.dumps(answer))
        scorer = TaskScorer(tmp_path)
        good = {k: v for k, v in answer.items() if k != "problems_found"}
        good["problems"] = ["missing"]
        assert scorer.score_task("task_f_2002", json.dumps(good))["passed"]
        # Ticker strings are exact: the wrong target scores 0 there.
        bad = {**good, "target_asset": "AST2"}
        result = scorer.score_task("task_f_2002", json.dumps(bad))
        assert result["component_scores"]["target_asset"] == 0.0
        # last_window_beta is an F1 core field: wrong beyond 1e-4 fails.
        off = {**good, "last_window_beta": 1.01}
        result = scorer.score_task("task_f_2002", json.dumps(off))
        assert result["component_scores"]["last_window_beta"] == 0.0
        assert not result["passed"]
        # final_window_volatility is also core (pilot: a wrong value
        # previously still passed on overall score alone).
        off_vol = {**good, "final_window_volatility": 0.9}
        result = scorer.score_task("task_f_2002", json.dumps(off_vol))
        assert result["component_scores"]["final_window_volatility"] == 0.0
        assert not result["passed"]

    def test_f4_max_drawdown_core_and_tolerance(self, tmp_path):
        # Frontier max_drawdown is core at 1e-3 (accumulated path
        # statistic, like reconciled_max_drawdown); 5e-4 off passes,
        # 2e-3 off scores 0 on the field and cannot pass.
        answer = {"num_aligned_days": 100, "num_assets": 4,
                  "num_rebalances": 4, "first_rebalance_value": 1.01,
                  "final_return": 0.05, "max_drawdown": -0.1,
                  "problems_found": ["duplicate_row_x"]}
        (tmp_path / "task_f_2005_answer.json").write_text(json.dumps(answer))
        scorer = TaskScorer(tmp_path)
        good = {k: v for k, v in answer.items() if k != "problems_found"}
        good["problems"] = ["duplicate"]
        assert scorer.score_task("task_f_2005", json.dumps(good))["passed"]
        close = {**good, "max_drawdown": -0.0995}  # 5e-4 off
        assert scorer.score_task("task_f_2005", json.dumps(close))["passed"]
        far = {**good, "max_drawdown": -0.098}  # 2e-3 off
        result = scorer.score_task("task_f_2005", json.dumps(far))
        assert result["component_scores"]["max_drawdown"] == 0.0
        assert not result["passed"]


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

    def _assert_baseline_fails(self, tamper):
        """Score solver answers rewritten by ``tamper`` (seeds 1-20 at
        shipped defaults); every tampered task must fail. ``tamper``
        returns None for tasks it does not apply to (including tasks
        whose true value is within tolerance of the tampered one, where
        the tampered answer is genuinely correct)."""
        checked = 0
        applied = []
        for seed in range(1, 21):
            applied.clear()  # task ids restart at task_f_2001 every seed

            def tracking(task, got, data, _tamper=tamper, _applied=applied):
                rewritten = _tamper(task, got, data)
                if rewritten is not None:
                    _applied.append(task["id"])
                return rewritten

            for task, result in _score_makers_default(seed, 25, tracking):
                if task["id"] in applied:
                    assert not result["passed"], \
                        (seed, task["id"], result["feedback"])
                    checked += 1
        assert checked > 0
        return checked

    def test_guess_beta_one_passes_zero(self):
        """Answering beta = 1.0 (everything else exactly right) must fail.

        F1's core is the target asset's last_window_beta (tolerance
        1e-4); F3's cores are beta_market/beta_sector (tolerance 1e-3).
        A task whose true beta happens to sit within tolerance of 1.0 is
        a correct answer, not a guess win, and is skipped."""
        def tamper(task, got, data):
            exp = task["expected"]
            if task["type"] == "frontier_rolling_risk":
                if abs(exp["last_window_beta"] - 1.0) <= 1e-4:
                    return None
                return {**got, "last_window_beta": 1.0}
            if task["type"] == "frontier_factor_exposure":
                if abs(exp["beta_market"] - 1.0) <= 1e-3 or \
                        abs(exp["beta_sector"] - 1.0) <= 1e-3:
                    return None
                return {**got, "beta_market": 1.0, "beta_sector": 1.0}
            return None

        self._assert_baseline_fails(tamper)

    def test_full_sample_beta_passes_zero(self):
        """F1: a full-sample beta (all returns) is not the last-window
        beta and must fail the core check."""
        def tamper(task, got, data):
            if task["type"] != "frontier_rolling_risk":
                return None
            from tests.frontier_solver import _aligned, _frames, read_blocks
            blocks = read_blocks(data)
            prices = _aligned(_frames(
                blocks[""] if "" in blocks else blocks["PRICES"]))
            rets = prices.pct_change().dropna()
            target = got["target_asset"]
            full = float(rets[target].cov(rets["IDX"])
                         / rets["IDX"].var(ddof=1))
            if abs(full - task["expected"]["last_window_beta"]) <= 1e-4:
                return None  # coincidentally the same; not a shortcut win
            return {**got, "last_window_beta": round(full, 6)}

        self._assert_baseline_fails(tamper)

    def test_log_return_answers_pass_zero(self):
        """F3: a regression run on log returns instead of simple returns
        must fail. Its betas can land within the 1e-3 tolerance, so the
        discriminator is r_squared, a core field at 1e-5: log returns
        shift R-squared by more than that whenever the betas are close
        (measured 0/400 across seeds 1-400)."""
        def tamper(task, got, data):
            if task["type"] != "frontier_factor_exposure":
                return None
            import re as _re

            import numpy as np
            import pandas as pd

            from tests.frontier_solver import _frames, clean, read_blocks
            blocks = read_blocks(data)
            frames = _frames(blocks["PRICES"])
            ast = clean(frames["AST"])
            for _, row in blocks["SPLIT_TABLE"].iterrows():
                m = _re.fullmatch(r"(\d+(?:\.\d+)?)-for-(\d+(?:\.\d+)?)",
                                  str(row["ratio"]))
                ast.loc[ast.index < str(row["date"])] *= \
                    (float(m.group(2)) / float(m.group(1)))
            aligned = pd.DataFrame(
                {"AST": ast, "MKT": clean(frames["MKT"]),
                 "SEC": clean(frames["SEC"])}).sort_index().dropna()
            rets = np.log(aligned / aligned.shift(1)).dropna()
            x = np.column_stack([np.ones(len(rets)), rets["MKT"],
                                 rets["SEC"]])
            coef, *_ = np.linalg.lstsq(x, rets["AST"].to_numpy(),
                                       rcond=None)
            resid = rets["AST"].to_numpy() - x @ coef
            ss_tot = float(((rets["AST"] - rets["AST"].mean()) ** 2).sum())
            return {**got, "alpha": round(float(coef[0]), 6),
                    "beta_market": round(float(coef[1]), 6),
                    "beta_sector": round(float(coef[2]), 6),
                    "r_squared": round(
                        1.0 - float((resid ** 2).sum()) / ss_tot, 6)}

        self._assert_baseline_fails(tamper)

    def test_wrong_drawdown_passes_zero(self):
        """F4: max_drawdown is a core field, so an otherwise-perfect
        answer with the drawdown wrong must fail (pilot passes had
        exactly this shape and previously reached 0.8 overall)."""
        def tamper(task, got, data):
            if task["type"] != "frontier_portfolio_pipeline":
                return None
            wrong = 0.0 if abs(task["expected"]["max_drawdown"]) > 1e-3 \
                else -0.9
            return {**got, "max_drawdown": wrong}

        self._assert_baseline_fails(tamper)


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
