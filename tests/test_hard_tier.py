"""Tests for the hard tier, including an independent-solver key check."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.hard_tier import HardTaskGenerator
from tests.independent_solver import solve

OFF_LIMITS = ["delisted", "survivorship", "ticker_rename", "rename", "dividend",
              "stale", "repeated_price", "too_short", "insufficient_history"]


def _gen_tasks(seed, count, tmp):
    gen = HardTaskGenerator(seed=seed)
    out = Path(tmp)
    tasks = gen.generate_all_tasks(out, num_tasks=count)
    return tasks, out


class TestHardTaskGenerator:
    def test_reproducibility(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            t1, d1 = _gen_tasks(42, 5, a)
            t2, d2 = _gen_tasks(42, 5, b)
            assert t1 == t2
            for t in t1:
                assert (d1 / t["data_file"]).read_bytes() == (d2 / t["data_file"]).read_bytes()
                k1 = json.loads((d1 / "answers" / f"{t['id']}_answer.json").read_text())
                k2 = json.loads((d2 / "answers" / f"{t['id']}_answer.json").read_text())
                assert k1 == k2

    def test_task_ids_use_h_prefix(self):
        gen = HardTaskGenerator(seed=42)
        task, _ = gen.generate_task_max_drawdown()
        assert task["id"].startswith("task_h_")

    def test_no_off_limits_problem_types(self):
        gen = HardTaskGenerator(seed=42)
        for maker in gen.task_generators():
            task, _ = maker()
            for problem in task["expected"]["problems_found"]:
                for word in OFF_LIMITS:
                    assert word not in problem.lower()

    def test_default_25_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, _ = _gen_tasks(42, 25, tmp)
        assert len(tasks) == 25
        types = [t["type"] for t in tasks]
        for expected_type in ["multi_asset_portfolio", "max_drawdown", "rolling_beta",
                              "dst_intraday", "single_asset_volatility"]:
            assert types.count(expected_type) == 5
        all_problems = []
        with tempfile.TemporaryDirectory() as tmp:
            tasks, out = _gen_tasks(42, 25, tmp)
            for t in tasks:
                key = json.loads((out / "answers" / f"{t['id']}_answer.json").read_text())
                all_problems.extend(key["problems_found"])
                assert 2 <= len(key["problems_found"]) <= 4
        joined = " ".join(all_problems).lower()
        for marker in ["calendar", "out_of_order", "reverse_split", "unit_change",
                       "currency", "adj_close", "timezone", "column_order"]:
            assert marker in joined, marker

    def test_problem_sets_vary_within_type(self):
        from collections import defaultdict
        seen = defaultdict(set)
        for seed in range(30):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen_tasks(seed, 25, tmp)
                for t in tasks:
                    key = json.loads((out / "answers" / f"{t['id']}_answer.json").read_text())
                    from src.scorer import categories_for_items
                    seen[t["type"]].add(frozenset(categories_for_items(key["problems_found"])))
        for task_type, sets in seen.items():
            assert len(sets) >= 3, (task_type, sets)

    def test_keys_match_data_across_seeds(self):
        from src.scorer import categories_for_items
        from tests.independent_solver import read_csv_any
        for seed in range(12):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen_tasks(seed, 25, tmp)
                for t in tasks:
                    key = json.loads((out / "answers" / f"{t['id']}_answer.json").read_text())
                    cats = categories_for_items(key["problems_found"])
                    raw = (out / t["data_file"]).read_text()
                    frame = read_csv_any(out / t["data_file"])
                    if "calendar" in cats:
                        per = [set(g["date"]) for _, g in frame.groupby("ticker")]
                        assert len(set().union(*per)) > len(set.intersection(*per))
                    if "column_order" in cats:
                        headers = [ln for ln in raw.splitlines()
                                   if set(ln.split(",")) <=
                                   {"date", "timestamp", "ticker", "open", "high", "low",
                                    "close", "adj_close", "volume", "currency", "fx_rate"}
                                   and ("date" in ln or "timestamp" in ln)]
                        assert len(headers) >= 2
                    if "currency" in cats:
                        assert set(frame["currency"]) - {"USD"}
                    if "adj_close" in cats:
                        sub = frame[frame["adj_close"].astype(str) != ""]
                        assert (sub["adj_close"].astype(float)
                                != sub["close"].astype(float)).any()
                    if "timezone" in cats:
                        stamps = frame["timestamp"].astype(str)
                        assert stamps.str.endswith("Z").any() and (
                            stamps.str.contains("+", regex=False).any()
                            or stamps.str.contains("-", regex=False).any())
                    if "out_of_order" in cats:
                        col = "timestamp" if "timestamp" in frame.columns else "date"
                        for _, g in frame.groupby("ticker"):
                            vals = list(g[col])
                            if vals != sorted(vals):
                                break
                        else:
                            raise AssertionError(f"out_of_order key but data sorted: {t['id']}")

    def test_no_stale_closes_seeds_1_to_200(self):
        from tests.independent_solver import read_csv_any
        for seed in range(1, 201):
            gen = HardTaskGenerator(seed=seed)
            dfs = gen.generate_clean_prices_multi_asset(
                ["A_US", "B_EU", "C"], datetime(2024, 1, 1), 60)
            for df in dfs.values():
                closes = df["close"].tolist()
                assert all(a != b for a, b in zip(closes, closes[1:], strict=False)), seed
            if seed % 40 == 0:  # spot-check injected task data too
                with tempfile.TemporaryDirectory() as tmp:
                    tasks, out = _gen_tasks(seed, 25, tmp)
                    for t in tasks:
                        frame = read_csv_any(out / t["data_file"])
                        if "date" not in frame.columns:
                            continue
                        for _, g in frame.groupby("ticker"):
                            g = g.sort_values("date")
                            closes = [float(c) for c in g["close"]]
                            assert all(a != b for a, b in
                                       zip(closes, closes[1:], strict=False)), (seed, t["id"])

    def test_ohlc_consistent(self):
        for seed in range(20):
            gen = HardTaskGenerator(seed=seed)
            dfs = gen.generate_clean_prices_multi_asset(["T"], datetime(2024, 1, 1), 60)
            df = dfs["T"]
            assert (df["low"] <= df["open"]).all() and (df["low"] <= df["close"]).all()
            assert (df["high"] >= df["open"]).all() and (df["high"] >= df["close"]).all()

    def test_prompts_state_rules_not_problems(self):
        gen = HardTaskGenerator(seed=7)
        for maker in gen.task_generators():
            task, _ = maker()
            prompt = task["prompt"]
            # fixed possible-issues list and general rules are present
            assert "may contain any" in prompt or "any\nsubset" in prompt or "subset" in prompt
            assert "ddof=1" in prompt
            assert "inner join" in prompt
            assert "America/New_York" in prompt
            assert "adj_close" in prompt
            assert "fx_rate" in prompt
            # the concrete injected problem strings must not appear
            for problem in task["expected"]["problems_found"]:
                assert problem not in prompt

    def test_intraday_prices_vary(self):
        gen = HardTaskGenerator(seed=42)
        _, intra = gen.generate_task_dst_intraday()
        assert intra["close"].nunique() > len(intra) // 2


class TestHandComputed:
    def _df(self, dates, closes):
        return pd.DataFrame({"date": pd.to_datetime(dates), "ticker": "T",
                             "open": closes, "high": closes, "low": closes,
                             "close": closes, "volume": 1000})

    def test_max_drawdown_hand(self):
        gen = HardTaskGenerator(seed=1)
        assert gen.compute_max_drawdown(pd.Series([100.0, 120.0, 60.0, 90.0])) == pytest.approx(-0.5)

    def test_inner_join_and_drift_hand(self):
        gen = HardTaskGenerator(seed=1)
        dates_a = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        dates_b = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        dfs = {"A": self._df(dates_a, [100.0, 110.0, 121.0, 121.0]),
               "B": self._df(dates_b, [200.0, 180.0, 198.0, 198.0])}
        ret, n = gen.compute_equal_weight_portfolio_return(dfs, rebalance_freq=20)
        # inner join keeps 01-02..01-04; A +10% then 0%, B -10% then +10%
        # holdings start .5/.5 on 01-02; after: .55/.45; then .55/.495 -> total 1.045
        assert n == 3
        assert ret == pytest.approx(0.045, abs=1e-9)

    def test_rebalance_is_not_daily_hand(self):
        gen = HardTaskGenerator(seed=1)
        frame = pd.DataFrame(
            {"A": [100.0, 200.0, 200.0, 200.0], "B": [100.0, 100.0, 200.0, 200.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]))
        # rebalance every 2 return observations:
        # start .5/.5; obs1: 1.0/.5 (total1.5); obs2: 1.0/1.0 -> rebalance to 1/1
        # obs3: no moves -> 2.0 total => return 1.0
        ret = gen._portfolio_from_prices(frame, rebalance_freq=2)
        assert ret == pytest.approx(1.0, abs=1e-9)
        # daily-rebalanced would give (1+.5)*(1+.5)*(1+0)-1 = 1.25, so this differs
        assert ret != pytest.approx(1.25)

    def test_beta_window_off_by_one_hand(self):
        gen = HardTaskGenerator(seed=1)
        idx = pd.Series([0.01, -0.02, 0.03, 0.005])
        asset = idx * 2.0
        # window=3 over 4 returns gives windows ending at obs 3 and 4 -> 2 betas of 2.0
        assert gen.compute_rolling_beta(asset, idx, window=3) == pytest.approx(2.0)

    def test_reverse_split_exact_multiple(self):
        gen = HardTaskGenerator(seed=3)
        dfs = gen.generate_clean_prices_multi_asset(["T"], datetime(2024, 1, 1), 60)
        df = dfs["T"]
        before = df.iloc[40]["close"]
        out, problems = gen.inject_reverse_split(df.copy(), multiple=3)
        assert out.iloc[40]["close"] == pytest.approx(before * 3)
        assert any("1-for-3" in p for p in problems)

    def test_out_of_order_label_counts_actual_swaps(self):
        gen = HardTaskGenerator(seed=3)
        dfs = gen.generate_clean_prices_multi_asset(["T"], datetime(2024, 1, 1), 60)
        _, problem = gen.inject_out_of_order_rows(dfs["T"], num_swaps=3)
        assert problem.endswith("3_swaps")


class TestIndependentSolver:
    def _check_dir(self, out, tasks):
        for task in tasks:
            key = json.loads((Path(out) / "answers" / f"{task['id']}_answer.json").read_text())
            got = solve(task, Path(out) / task["data_file"])
            for field, want in key.items():
                if field == "problems_found":
                    continue
                assert field in got, (task["id"], field)
                assert got[field] == pytest.approx(want, abs=1e-4), (task["id"], field, got, want)

    def test_seed42_all_25_and_20_other_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, out = _gen_tasks(42, 25, tmp)
            self._check_dir(out, tasks)
        for seed in range(1000, 1020):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen_tasks(seed, 5, tmp)
                self._check_dir(out, tasks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
