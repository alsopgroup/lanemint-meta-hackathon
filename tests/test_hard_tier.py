"""Tests for the hard tier, including an independent-solver key check."""

import json
import tempfile
from datetime import datetime, timedelta
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
                assert 1 <= len(key["problems_found"]) <= 4
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
                            rows = list(zip(g["date"], g["close"], strict=True))
                            for (d1, c1), (d2, c2) in zip(rows, rows[1:], strict=False):
                                assert d1 == d2 or float(c1) != float(c2), (seed, t["id"])

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


def _assert_no_stale(frame):
    if "timestamp" in frame.columns:
        for _, g in frame.groupby("ticker"):
            bars = sorted(
                (pd.Timestamp(s).tz_convert("America/New_York")
                 if pd.Timestamp(s).tzinfo else pd.Timestamp(s).tz_localize("UTC"),
                 float(c))
                for s, c in zip(g["timestamp"], g["close"], strict=True))
            for (t1, c1), (t2, c2) in zip(bars, bars[1:], strict=False):
                assert t1 == t2 or c1 != c2, "stale intraday close"
    else:
        for _, g in frame.groupby("ticker"):
            rows = sorted(zip(pd.to_datetime(g["date"]), g["close"], strict=True),
                          key=lambda r: r[0])  # stable: keeps file order within a date
            for (d1, c1), (d2, c2) in zip(rows, rows[1:], strict=False):
                assert d1 == d2 or float(c1) != float(c2), "stale daily close"


class TestFinalCsvScans:
    def test_hard_no_undeclared_jumps_and_no_stale_seeds_1_to_200(self):
        from tests.independent_solver import read_csv_any
        for seed in range(1, 201):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen_tasks(seed, 5, tmp)
                for t in tasks:
                    key = json.loads((out / "answers" / f"{t['id']}_answer.json").read_text())
                    frame = read_csv_any(out / t["data_file"])
                    _assert_no_stale(frame)
                    shifts = sum(1 for p in key["problems_found"]
                                 if p.startswith(("reverse_split", "unit_change")))
                    jumps = 0
                    col = "timestamp" if "timestamp" in frame.columns else "date"
                    for _, g in frame.groupby("ticker"):
                        if col == "date":
                            g = g.drop_duplicates(subset=["date"])
                            closes = [float(c) for c in
                                      g.sort_values("date")["close"]]
                        else:
                            closes = [float(c) for _, c in sorted(
                                (pd.Timestamp(s), c) for s, c in
                                zip(g["timestamp"], g["close"], strict=True))]
                        # de-duplicate identical instants for DST
                        if col == "timestamp":
                            seen, uniq = set(), []
                            for s, c in sorted((pd.Timestamp(s), float(c)) for s, c in
                                               zip(g["timestamp"], g["close"], strict=True)):
                                if s not in seen:
                                    seen.add(s)
                                    uniq.append(c)
                            closes = uniq
                        for a, b in zip(closes, closes[1:], strict=False):
                            if a > 0 and (b / a >= 2 or b / a <= 0.5):
                                jumps += 1
                    assert jumps <= shifts, (seed, t["id"], jumps, shifts, key)

    def test_easy_no_stale_final_csvs_seeds_1_to_200(self):
        from src.generate_tasks import TaskGenerator
        for seed in range(1, 201):
            with tempfile.TemporaryDirectory() as tmp:
                gen = TaskGenerator(seed=seed)
                tasks = gen.generate_all_tasks(Path(tmp), num_tasks=5)
                for t in tasks:
                    frame = pd.read_csv(Path(tmp) / t["data_file"])
                    _assert_no_stale(frame)


def _ny_instants(stamps):
    out = []
    for s in stamps:
        ts = pd.Timestamp(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        out.append(ts.tz_convert("America/New_York"))
    return out


def _sorted_closes(frame):
    """Per-ticker closes sorted by date/instant, exact duplicates dropped."""
    series = {}
    for ticker, g in frame.groupby("ticker"):
        if "timestamp" in frame.columns:
            pairs = sorted(zip(_ny_instants(g["timestamp"].astype(str)),
                               g["close"], strict=True), key=lambda p: p[0])
            seen, closes = set(), []
            for instant, close in pairs:
                if instant not in seen:
                    seen.add(instant)
                    closes.append(float(close))
            series[ticker] = closes
        else:
            g = g.assign(_k=pd.to_datetime(g["date"])).sort_values("_k")
            g = g.drop_duplicates(subset=["date"], keep="first")
            series[ticker] = [float(c) for c in g["close"]]
    return series


def _sustained_jump_ratios(closes):
    """Jump ratios at points where the new level is sustained (rule 6)."""
    ratios = []
    for i in range(1, len(closes) - 1):
        if closes[i - 1] > 0:
            ratio = closes[i] / closes[i - 1]
            if ratio > 1.8 and closes[i + 1] / closes[i - 1] > 1.8:
                ratios.append(ratio)
    return ratios


def _bars_per_ny_day(frame):
    per_day = {}
    for instant in _ny_instants(frame["timestamp"].astype(str)):
        per_day.setdefault(instant.date(), set()).add(instant)
    return sorted(len(v) for v in per_day.values())


def _has_missing_weekday(gen, ticker, dates):
    """A weekday inside the ticker's own range is absent (holidays excluded)."""
    days = {d.date() for d in pd.to_datetime(dates)}
    day = min(days) + timedelta(days=1)
    while day < max(days):
        if day.weekday() < 5 and day not in days:
            if ticker.endswith("_EU"):
                holiday = gen._is_eu_holiday(day)
            elif ticker.endswith("_US"):
                holiday = gen._is_us_holiday(day)
            else:
                holiday = False
            if not holiday:
                return True
        day += timedelta(days=1)
    return False


class TestKeyDetectability:
    """Every category in a key must be detectable in the final CSV."""

    def test_every_key_category_detectable_seeds_1_to_200(self):
        from src.scorer import categories_for_items
        from tests.independent_solver import read_csv_any
        calendar_gen = HardTaskGenerator(seed=0)
        for seed in range(1, 201):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen_tasks(seed, 5, tmp)
                for t in tasks:
                    key = json.loads(
                        (out / "answers" / f"{t['id']}_answer.json").read_text())
                    cats = categories_for_items(key["problems_found"])
                    frame = read_csv_any(out / t["data_file"])
                    where = (seed, t["id"], sorted(cats))
                    intraday = "timestamp" in frame.columns
                    if "duplicate" in cats:
                        cols = ["ticker", "timestamp" if intraday else "date"]
                        assert frame.duplicated(subset=cols).any(), where
                    if "missing" in cats:
                        if intraday:
                            # the missing bar really is absent: one New-York
                            # date has 4 distinct bar instants, the rest 5
                            assert _bars_per_ny_day(frame)[0] == 4, where
                        else:
                            assert any(
                                _has_missing_weekday(calendar_gen, ticker, g["date"])
                                for ticker, g in frame.groupby("ticker")), where
                    ratios = [r for closes in _sorted_closes(frame).values()
                              for r in _sustained_jump_ratios(closes)]
                    if "unit_change" in cats:
                        assert any(r > 50 for r in ratios), where
                    if "reverse_split" in cats:
                        assert any(1.8 < r <= 50 for r in ratios), where
                    for r in ratios:
                        # ~100x is a unit change; ~300x would mean a unit
                        # change and a reverse split share one row, and any
                        # sustained jump must be declared in the key
                        assert r <= 150, (where, r)
                        assert "unit_change" in cats if r > 50 \
                            else "reverse_split" in cats, (where, r)

    def test_known_collision_cases(self):
        from src.scorer import categories_for_items
        from tests.independent_solver import read_csv_any
        # Unit change and reverse split landed on the same row (~300x jump
        # labelled as both, readable as a unit change only).
        for seed in (54, 138):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen_tasks(seed, 25, tmp)
                t = next(t for t in tasks if t["id"] == "task_h_1020")
                key = json.loads(
                    (out / "answers" / "task_h_1020_answer.json").read_text())
                cats = categories_for_items(key["problems_found"])
                assert {"unit_change", "reverse_split"} <= cats
                frame = read_csv_any(out / t["data_file"])
                ratios = [r for closes in _sorted_closes(frame).values()
                          for r in _sustained_jump_ratios(closes)]
                assert any(r > 50 for r in ratios), seed
                assert any(1.8 < r <= 50 for r in ratios), seed
                assert all(r <= 150 for r in ratios), seed
        # The DST missing bar deleted the original of the repeated instant,
        # whose second-format copy restored it (nothing actually missing).
        for seed in (91, 98, 117):
            with tempfile.TemporaryDirectory() as tmp:
                tasks, out = _gen_tasks(seed, 25, tmp)
                t = next(t for t in tasks if t["id"] == "task_h_1004")
                key = json.loads(
                    (out / "answers" / "task_h_1004_answer.json").read_text())
                cats = categories_for_items(key["problems_found"])
                assert "missing" in cats
                frame = read_csv_any(out / t["data_file"])
                assert _bars_per_ny_day(frame)[0] == 4, seed


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
