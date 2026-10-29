"""Tools-frontier tier generator (families T1-T5).

Each task is a small directory of files (market data, auxiliary tables and
a ``spec.md`` methodology memo) rather than one CSV.  The memo states the
governing conventions in prose and in a convention table; different output
sections may use different conventions, and a later clause can govern one
section while a desk default governs another.  The prompt names only the
directory and the output schema, so the solver must read the memo.

Design rules for this tier:

* prices are generated so consecutive closes for a ticker are never equal;
* corporate actions appear only as rows of an explicit ``splits.csv``;
* every task is checked against wrong-method variants (see
  ``trap_variants`` in the task metadata).  When a variant would land
  within tolerance of any core field it is meant to change, the task is
  regenerated with fresh draws ("reseeded") until it discriminates;
* generation seeds 42, 1001 and 1002 are reserved for evaluation and are
  refused unless ``allow_eval=True`` (CLI: ``--eval``) is given.

The generator uses only the standard library so its output is byte-for-byte
deterministic for a given seed on any supported Python.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

EVAL_SEEDS = frozenset({42, 1001, 1002})

TOOLS_FRONTIER_SETTINGS = {
    "convention_risk": {"rows": 220, "assets": 3},
    "fx_calendar": {"rows": 180, "us_assets": 2, "eu_assets": 2},
    "reconciliation": {"rows": 180, "assets": 3},
    "split_holdings": {"rows": 180, "assets": 3},
    "portfolio_construction": {"rows": 160, "assets": 6, "lookback": 60,
                               "rebalance_every": 21, "first_rebalance": 80},
}

FAMILY_TYPES = {
    "convention_risk": "tools_frontier_convention_risk",
    "fx_calendar": "tools_frontier_fx_calendar",
    "reconciliation": "tools_frontier_reconciliation",
    "split_holdings": "tools_frontier_split_holdings",
    "portfolio_construction": "tools_frontier_portfolio_construction",
}

# Core-field scoring tolerances (see src/scorer.py schema handling).
# The memo instructs 6-decimal reporting, so the absolute tolerance is
# 2e-6: a value that is correct when rounded to 6 decimals and the same
# value left unrounded must both pass (their distance from the stored,
# 6-decimal key is at most 1e-6, and 2e-6 gives that rule a clean margin
# without opening a gap any wrong-method variant has been measured to fit
# through -- the discriminator gate below uses 50x this tolerance).
_RTOL = 1e-6
_ATOL = 2e-6


# --------------------------------------------------------------------------
# small numeric / calendar helpers (generator side)
# --------------------------------------------------------------------------

def _business_dates(rng: random.Random, n: int) -> list[datetime]:
    """[Function: _business_dates] Return ``n`` weekday dates."""
    start = datetime(2022, 1, 3) + timedelta(days=rng.randint(0, 400))
    while start.weekday() >= 5:
        start += timedelta(days=1)
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _dstr(day: datetime) -> str:
    return day.strftime("%Y-%m-%d")


def _closes(rng: random.Random, start: float, rets: list[float]) -> list[float]:
    """[Function: _closes] Price path whose rounded closes never repeat."""
    price, out, prev = float(start), [], None
    for r in rets:
        price = max(2.0, price * (1.0 + float(r)))
        close = round(price, 2)
        if prev is not None and close == prev:
            close = round(prev + 0.01, 2)
            price = float(close)
        out.append(close)
        prev = close
    return out


def _mean(xs):
    return sum(xs) / len(xs)


def _var(xs, ddof):
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - ddof)


def _std(xs, ddof):
    return math.sqrt(_var(xs, ddof))


def _cov(xs, ys, ddof):
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (len(xs) - ddof)


def _returns(closes, kind):
    if kind == "log":
        return [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False)]
    return [b / a - 1.0 for a, b in zip(closes, closes[1:], strict=False)]


def _ann_vol(closes, kind, periods, ddof):
    return _std(_returns(closes, kind), ddof) * math.sqrt(periods)


def _max_drawdown(values):
    """[Function: _max_drawdown] Return (drawdown, trough_index)."""
    peak, worst, trough = values[0], 0.0, 0
    for i, v in enumerate(values):
        peak = max(peak, v)
        dd = (v - peak) / peak if peak else 0.0
        if dd < worst:
            worst, trough = dd, i
    return worst, trough


def _csv(headers, rows) -> str:
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(r) for r in row))
    return "\n".join(lines) + "\n"


def _price_rows(ticker, dates, closes, rng):
    rows = []
    for day, close in zip(dates, closes, strict=True):
        open_ = round(close * rng.uniform(0.996, 1.004), 2)
        rows.append([_dstr(day), ticker, f"{open_:.2f}",
                     f"{max(open_, close) * rng.uniform(1.0, 1.008):.2f}",
                     f"{min(open_, close) * rng.uniform(0.992, 1.0):.2f}",
                     f"{close:.2f}", rng.randint(50_000, 9_000_000)])
    return rows


def _prompt(family_label: str, fields: list[str]) -> str:
    body = "\n".join(f"- {f}" for f in fields)
    return (
        f"You are given a {family_label} task as a directory of files.\n\n"
        "The files are mounted read-only in the data directory for this task "
        "(tool mode: /data/). Read spec.md first: it is the methodology memo "
        "for this task and it governs every computation, including which "
        "convention applies to which output section. The other files are the "
        "market and auxiliary data named in the memo. Do not assume a desk "
        "default overrides a section-specific clause in the memo.\n\n"
        "Return your answer as JSON with fields:\n"
        f"{body}\n"
        "Report decimal results rounded to 6 decimal places unless a field "
        "states otherwise; counts and share counts are exact integers; "
        "dates are YYYY-MM-DD."
    )


def _within(value, correct, rtol=_RTOL, atol=_ATOL):
    return abs(value - correct) <= max(atol, rtol * abs(correct))


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------

class ToolsFrontierTaskGenerator:
    """Generates tools-frontier tasks (T1-T5), 30 per seed by default."""

    def __init__(self, seed: int = 7001, settings: dict | None = None,
                 allow_eval: bool = False):
        if seed in EVAL_SEEDS and not allow_eval:
            raise ValueError(
                f"[ToolsFrontierTaskGenerator] seed {seed} is a reserved "
                "evaluation seed (42, 1001, 1002); generation for training "
                "or development is refused. Pass allow_eval=True / --eval "
                "only when intentionally building an evaluation set.")
        self.seed = seed
        self.rng = random.Random(seed)
        self.task_id = 3000
        self.settings = {k: dict(v) for k, v in TOOLS_FRONTIER_SETTINGS.items()}
        for family, overrides in (settings or {}).items():
            self.settings[family].update(overrides)

    # -- discriminator gate -------------------------------------------------

    def _discriminates(self, expected, core, variants) -> bool:
        """True when every wrong-method variant misses a core field.

        A variant is a full alternative answer.  It discriminates when at
        least one numeric or date core field differs from the correct
        value by more than 50x the scoring tolerance (numeric) or at all
        (dates / integers).
        """
        for variant in variants.values():
            differs = False
            for field in core:
                if field not in variant or field not in expected:
                    continue
                want, got = expected[field], variant[field]
                if isinstance(want, dict):
                    for key, sub in want.items():
                        if key in got and isinstance(sub, (int, float)) \
                                and not _within(float(got[key]), float(sub),
                                                50 * _RTOL, 50 * _ATOL):
                            differs = True
                    continue
                if isinstance(want, float) and isinstance(got, (int, float)):
                    if not _within(float(got), want, 50 * _RTOL, 50 * _ATOL):
                        differs = True
                elif got != want:
                    differs = True
            if not differs:
                return False
        return True

    def _build(self, builder, family: str):
        """Run ``builder(rng)`` until its trap variants discriminate."""
        for _attempt in range(200):
            files, expected, core, variants, problems = builder(self.rng)
            if self._discriminates(expected, core, variants):
                break
        else:  # pragma: no cover - defensive; settings make this unlikely
            raise RuntimeError(f"[ToolsFrontierTaskGenerator] {family}: "
                               "could not generate a discriminating task")
        self.task_id += 1
        task_id = f"task_tf_{self.task_id:03d}"
        expected = dict(expected)
        expected["_tools_frontier"] = {
            "family": FAMILY_TYPES[family],
            "core": core,
            "rtol": _RTOL,
            "atol": _ATOL,
            "field_rtol": {"weights_first_rebalance": 1e-4}
            if family == "portfolio_construction" else {},
        }
        if problems is not None:
            expected["problems_found"] = sorted(problems)
        task = {
            "id": task_id,
            "type": FAMILY_TYPES[family],
            "prompt": _prompt_for_family(family, expected),
            "data_file": f"{task_id}/spec.md",
            "data_dir": task_id,
            "files": sorted(files),
            "trap_variants": variants,
        }
        return task, files, expected

    # ------------------------------------------------------------------ T1
    def _t1_once(self, rng: random.Random):
        cfg = self.settings["convention_risk"]
        rows = cfg["rows"]
        dates = _business_dates(rng, rows)
        assets = [f"AST{i + 1}" for i in range(cfg["assets"])]
        bench = "BMK"
        bench_rets = [rng.gauss(0.0003, 0.010) for _ in range(rows - 1)]
        series = {bench: _closes(rng, rng.uniform(90, 130), [0.0] + bench_rets)}
        # A second benchmark, only partly correlated with the first:
        # one section below is governed by BMK2, not BMK, and using the
        # desk's first benchmark there is a scored error.
        bench2 = "BMK2"
        bench2_rets = [0.4 * r + rng.gauss(0, 0.009) for r in bench_rets]
        series[bench2] = _closes(rng, rng.uniform(90, 130), [0.0] + bench2_rets)
        asset_rets = {}
        for asset in assets:
            beta = rng.uniform(0.6, 1.5)
            rets = [beta * r + rng.gauss(0, 0.008) for r in bench_rets]
            asset_rets[asset] = rets
            series[asset] = _closes(rng, rng.uniform(40, 170), [0.0] + rets)

        vol_spec = {assets[0]: ("simple", 252, 1), assets[1]: ("log", 260, 1),
                    assets[2]: ("simple", 252, 0)}
        target, bench_name = assets[0], bench
        window, step = 20, 5

        def rolling_betas(rets_a, rets_b, win, stp):
            out = []
            for start in range(0, len(rets_a) - win + 1, stp):
                xa = rets_a[start:start + win]
                xb = rets_b[start:start + win]
                out.append((_cov(xa, xb, 1) / _var(xb, 1),
                            start + win - 1))  # index into returns
            return out

        simple = {t: _returns(c, "simple") for t, c in series.items()}
        rb = rolling_betas(simple[target], simple[bench_name], window, step)
        max_beta, max_end = max(rb, key=lambda p: p[0])
        end_date = _dstr(dates[max_end + 1])

        def weekly_closes(ticker):
            by_week = {}
            for day, close in zip(dates, series[ticker], strict=True):
                friday = day + timedelta(days=(4 - day.weekday()) % 7)
                by_week[friday] = close  # last close of the week wins
            keys = sorted(by_week)
            return [by_week[k] for k in keys]

        def weekly_te(kind, ddof, periods):
            ra = _returns(weekly_closes(assets[1]), kind)
            rbk = _returns(weekly_closes(bench_name), kind)
            diff = [a - b for a, b in zip(ra, rbk, strict=True)]
            return _std(diff, ddof) * math.sqrt(periods)

        def ewma_vol(ticker, lam, kind="simple"):
            rets = _returns(series[ticker], kind)
            var = rets[0] ** 2
            for r in rets[1:]:
                var = lam * var + (1 - lam) * r ** 2
            return math.sqrt(var * 252)

        rf_annual = 0.03
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1

        def sharpe(ticker, kind):
            rets = _returns(series[ticker], kind)
            excess = [r - rf_daily for r in rets]
            return _mean(excess) / _std(excess, 1) * math.sqrt(252)

        # Second rolling section: a different asset, window and step.
        rb2 = rolling_betas(simple[assets[1]], simple[bench_name], 15, 1)
        max_beta2, max_end2 = max(rb2, key=lambda p: p[0])
        # Third rolling section: another asset against the SECOND
        # benchmark (window 25, step 5).
        rb3 = rolling_betas(simple[assets[2]], simple[bench2], 25, 5)
        max_beta3, max_end3 = max(rb3, key=lambda p: p[0])
        # Recent-window volatility: only the last 60 log returns of the
        # first asset (after the calendar exclusion), not the full sample.
        recent_window = 60
        recent_rets = _returns(series[assets[0]], "log")[-recent_window:]
        recent_vol = _std(recent_rets, 1) * math.sqrt(252)
        recent_field = f"recent_vol_last_{recent_window}"
        downside_rets = simple[assets[2]]
        downside_vol = math.sqrt(
            _mean([min(r, 0.0) ** 2 for r in downside_rets])) * math.sqrt(252)
        log_a = _returns(series[assets[0]], "log")
        log_b = _returns(series[bench_name], "log")
        log_corr = _cov(log_a, log_b, 1) / (_std(log_a, 1) * _std(log_b, 1))
        weekly_a = _returns(weekly_closes(assets[0]), "log")
        weekly_ann = _mean(weekly_a) * 52

        # Pitfall rows that only appear on inspection: weekend prints for
        # the first asset (their dates are absent from calendar.csv and
        # the memo excludes them) and, below, one ticker's rows shuffled.
        weekend_points = []
        for idx in rng.sample(range(20, rows - 20), 3):
            saturday = dates[idx] + timedelta(days=1)
            while saturday.weekday() != 5:
                saturday += timedelta(days=1)
            weekend_points.append(
                (saturday, round(series[assets[0]][idx] * (1 + rng.uniform(-0.03, 0.03)), 2)))
        interleaved = sorted(
            [(d, c) for d, c in zip(dates, series[assets[0]], strict=True)]
            + weekend_points)
        weekend_ann_vol = _ann_vol([c for _d, c in interleaved], "simple", 252, 1)

        expected = {
            f"ann_vol_{a}": round(_ann_vol(series[a], *vol_spec[a]), 6)
            for a in assets
        }
        expected.update({
            "rolling_max_beta": round(max_beta, 6),
            "rolling_max_beta_end_date": end_date,
            "rolling2_max_beta": round(max_beta2, 6),
            "rolling2_max_beta_end_date": _dstr(dates[max_end2 + 1]),
            "weekly_tracking_error": round(weekly_te("log", 0, 52), 6),
            "ewma_vol_last": round(ewma_vol(assets[2], 0.94), 6),
            "sharpe_ratio": round(sharpe(assets[1], "simple"), 6),
            "downside_volatility": round(downside_vol, 6),
            "log_return_correlation": round(log_corr, 6),
            "weekly_annualised_log_return": round(weekly_ann, 6),
            "rolling3_max_beta": round(max_beta3, 6),
            "rolling3_max_beta_end_date": _dstr(dates[max_end3 + 1]),
            recent_field: round(recent_vol, 6),
            "num_aligned_days": rows,
        })
        core = [f"ann_vol_{a}" for a in assets] + [
            "rolling_max_beta", "rolling_max_beta_end_date",
            "rolling2_max_beta", "rolling2_max_beta_end_date",
            "weekly_tracking_error", "ewma_vol_last", "sharpe_ratio",
            "downside_volatility", "log_return_correlation",
            "weekly_annualised_log_return",
            "rolling3_max_beta", "rolling3_max_beta_end_date", recent_field]

        def _corr(kind):
            ra = _returns(series[assets[0]], kind)
            rbk = _returns(series[bench_name], kind)
            return _cov(ra, rbk, 1) / (_std(ra, 1) * _std(rbk, 1))

        # Section 10 against the first benchmark (the naive benchmark
        # choice), computed once for both naive-style answers below.
        rb3_vs_first = max(
            rolling_betas(simple[assets[2]], simple[bench_name], 25, 5),
            key=lambda p: p[0])
        naive = {
            f"ann_vol_{a}": round(_ann_vol(series[a], "simple", 252, 1), 6)
            for a in assets
        }
        naive.update({
            "rolling_max_beta": expected["rolling_max_beta"],
            "rolling_max_beta_end_date": end_date,
            "rolling2_max_beta": round(max(
                rolling_betas(simple[assets[1]], simple[bench_name], 15, 15),
                key=lambda p: p[0])[0], 6),
            "rolling2_max_beta_end_date": expected["rolling2_max_beta_end_date"],
            "weekly_tracking_error": round(weekly_te("simple", 1, 252), 6),
            "ewma_vol_last": round(_ann_vol(series[assets[2]], "simple", 252, 1), 6),
            "sharpe_ratio": round(sharpe(assets[1], "log"), 6),
            "downside_volatility": round(
                _ann_vol(series[assets[2]], "simple", 252, 1), 6),
            "log_return_correlation": round(_corr("simple"), 6),
            "weekly_annualised_log_return": round(
                _mean(_returns(weekly_closes(assets[0]), "simple")) * 52, 6),
            # Naive also reaches for the first benchmark in section 10
            # and uses the full sample in section 11.
            "rolling3_max_beta": round(rb3_vs_first[0], 6),
            "rolling3_max_beta_end_date": _dstr(dates[rb3_vs_first[1] + 1]),
            recent_field: round(_ann_vol(series[assets[0]], "simple", 252, 1), 6),
            "num_aligned_days": rows,
        })
        nonoverlap = rolling_betas(simple[target], simple[bench_name], window, window)
        no_beta, no_end = max(nonoverlap, key=lambda p: p[0])
        variant_nonoverlap = dict(expected)
        variant_nonoverlap.update({
            "rolling_max_beta": round(no_beta, 6),
            "rolling_max_beta_end_date": _dstr(dates[no_end + 1])})
        variant_weekend = dict(expected)
        variant_weekend[f"ann_vol_{assets[0]}"] = round(weekend_ann_vol, 6)
        # Wrong-benchmark trap: section 10 computed against BMK instead
        # of the BMK2 named in its convention-table row.
        variant_wrong_bench = dict(expected)
        variant_wrong_bench.update({
            "rolling3_max_beta": round(rb3_vs_first[0], 6),
            "rolling3_max_beta_end_date": _dstr(dates[rb3_vs_first[1] + 1])})
        # Full-sample trap: section 11 over every log return instead of
        # only the last 60.
        variant_full_sample = dict(expected)
        variant_full_sample[recent_field] = round(
            _ann_vol(series[assets[0]], "log", 252, 1), 6)
        variants = {"naive_simple_252": naive,
                    "nonoverlapping_rolling": variant_nonoverlap,
                    "weekend_prints_included": variant_weekend,
                    "wrong_benchmark": variant_wrong_bench,
                    "full_sample_instead_of_recent": variant_full_sample}

        rows_out = []
        for ticker in assets + [bench_name, bench2]:
            ticker_rows = _price_rows(ticker, dates, series[ticker], rng)
            if ticker == assets[0]:
                for day, close in weekend_points:
                    ticker_rows.append(_price_rows(ticker, [day], [close], rng)[0])
                ticker_rows.sort(key=lambda r: r[0])
            if ticker in (assets[1], bench2):
                # Row order in the file is not date order for these
                # tickers; the memo requires sorting before any use.
                rng.shuffle(ticker_rows)
            rows_out += ticker_rows
        spec = f"""# Methodology memo - convention-switch risk report (T1)

Desk default: simple returns, 252 periods per year, sample statistics
(ddof=1). A section clause below overrides the desk default for that
section only; honour the governing clause for each field.

## Convention table
- ann_vol[{assets[0]}]: return=simple; periods_per_year=252; ddof=1
- ann_vol[{assets[1]}]: return=log; periods_per_year=260; ddof=1
- ann_vol[{assets[2]}]: return=simple; periods_per_year=252; ddof=0 (population, risk section footnote)
- rolling_beta: asset={target}; benchmark={bench_name}; window={window}; step={step}; return=simple; ddof=1
- weekly_tracking_error: asset={assets[1]}; benchmark={bench_name}; resample=W-FRI last close; return=log; periods_per_year=52; ddof=0
- ewma_vol: asset={assets[2]}; return=simple; lambda=0.94; periods_per_year=252; seed_variance=first_return_squared
- sharpe: asset={assets[1]}; return=simple; annual_rf={rf_annual}; periods_per_year=252; ddof=1
- rolling_beta2: asset={assets[1]}; benchmark={bench_name}; window=15; step=1; return=simple; ddof=1
- downside_vol: asset={assets[2]}; return=simple; periods_per_year=252
- log_correlation: asset={assets[0]}; benchmark={bench_name}
- weekly_annualised_log_return: asset={assets[0]}; resample=W-FRI last close; periods_per_year=52
- rolling_beta3: asset={assets[2]}; benchmark={bench2}; window=25; step=5; return=simple; ddof=1
- recent_vol: asset={assets[0]}; window_returns={recent_window}; return=log; periods_per_year=252; ddof=1

## Data rules (read before computing)
- calendar.csv lists the trading dates. prices.csv can contain rows on
  dates that are not in calendar.csv (weekend prints); exclude every row
  whose date is not in calendar.csv before any computation.
- Row order inside prices.csv is not guaranteed to be date order.
  Sort each ticker by date before computing anything.

## Sections
1. Annualised volatility per asset per its convention-table row:
   std(returns, ddof) * sqrt(periods_per_year).
2. Rolling beta: overlapping windows of {window} consecutive simple
   returns advancing by {step} returns. rolling_max_beta is the largest
   window beta (cov/var, ddof=1); its end date is the date of the last
   return in that window (the second price date of the pair).
3. Weekly tracking error: resample each series to the last close of each
   week ending Friday, take log weekly returns, and report the population
   standard deviation (ddof=0) of asset-minus-benchmark weekly returns
   times sqrt(52).
4. EWMA volatility at the last date: variance starts at the square of the
   first simple return; each later return updates it as
   var = 0.94 * var + 0.06 * return^2. Report sqrt(var * 252).
5. Sharpe ratio: daily risk-free rate is (1 + annual_rf)^(1/252) - 1
   (the stated compounding); excess simple returns over that rate;
   mean(excess) / std(excess, ddof=1) * sqrt(252).
6. Rolling beta, second section: overlapping windows of 15 consecutive
   simple returns of {assets[1]} against {bench_name}, advancing by 1
   return. Report the largest window beta and its end date (same
   end-date rule as section 2). This section's window and step override
   section 2's.
7. Downside volatility of {assets[2]}: sqrt(mean over ALL simple returns
   of min(return, 0)^2) * sqrt(252). The denominator is the count of all
   returns, not the count of negative returns.
8. Log-return correlation between {assets[0]} and {bench_name} over the
   full sample of log returns (ddof cancels; report the correlation).
9. Weekly annualised log return of {assets[0]}: mean of the log weekly
   returns (W-FRI last close, as in section 3) times 52. This is a mean
   log return, not a compounded growth rate.
10. Rolling beta, third section: overlapping windows of 25 consecutive
   simple returns of {assets[2]} against {bench2} - this section's
   benchmark is {bench2}, NOT {bench_name} - advancing by 5 returns.
   Report the largest window beta and its end date (same end-date rule
   as section 2) as rolling3_max_beta and rolling3_max_beta_end_date.
11. Recent volatility of {assets[0]}: take only the LAST {recent_window}
   log returns of the (calendar-filtered, date-sorted) series - not the
   full sample - and report their sample standard deviation (ddof=1)
   times sqrt(252) as {recent_field}.

Prices are in prices.csv (date,ticker,open,high,low,close,volume);
after the data rules above are applied the series align on {rows}
business days.
"""
        cal_rows = [[_dstr(d), "true"] for d in dates]
        files = {"prices.csv": _csv(
            ["date", "ticker", "open", "high", "low", "close", "volume"], rows_out),
            "calendar.csv": _csv(["date", "is_business_day"], cal_rows),
            "spec.md": spec}
        return files, expected, core, variants, None

    # ------------------------------------------------------------------ T2
    def _t2_once(self, rng: random.Random):
        cfg = self.settings["fx_calendar"]
        rows = cfg["rows"]
        dates = _business_dates(rng, rows)
        us_assets = [f"US{i + 1}" for i in range(cfg["us_assets"])]
        eu_assets = [f"EU{i + 1}" for i in range(cfg["eu_assets"])]
        eu_holidays = sorted(rng.sample(dates[10:-10], 8))
        us_holidays = sorted(rng.sample(dates[10:-10], 3))
        us_dates = [d for d in dates if d not in us_holidays]
        eu_dates = [d for d in dates if d not in eu_holidays]

        def make_series(day_list, start_lo=40):
            rets = [rng.gauss(0.0004, 0.011) for _ in range(len(day_list) - 1)]
            return _closes(rng, rng.uniform(start_lo, 160), [0.0] + rets)

        us_series = {t: make_series(us_dates) for t in us_assets}
        eu_series = {t: make_series(eu_dates) for t in eu_assets}
        holdings = {t: rng.randrange(100, 900, 25) for t in us_assets + eu_assets}

        # FX fixes on the union calendar; two separate segments are
        # quoted inverted, flagged only by the quote column.
        union_dates = sorted(set(us_dates) | set(eu_dates))
        inv1 = (rng.randint(35, 55), 0)
        inv1 = (inv1[0], inv1[0] + rng.randint(15, 25))
        inv2 = (rng.randint(110, 130), 0)
        inv2 = (inv2[0], inv2[0] + rng.randint(15, 25))
        # A guaranteed run of consecutive late fixes (2-3 union dates):
        # every task exercises the walk-back rule below, not just some.
        late_start = rng.randint(30, 120)
        late_run = set(range(late_start, late_start + rng.choice([2, 3])))
        base_rate, fx_rows, fx_info = 1.08, [], {}
        for i, day in enumerate(union_dates):
            base_rate = max(0.7, base_rate * (1 + rng.gauss(0, 0.004)))
            inverted = (inv1[0] <= i < inv1[1]) or (inv2[0] <= i < inv2[1])
            quote = "USDEUR" if inverted else "EURUSD"
            rate = (1 / base_rate) if inverted else base_rate
            if i in late_run:
                form, hour_utc = "Z", 17  # after the 16:00 UTC cut-off
            elif i == 0:
                form, hour_utc = "Z", 7  # first fix is always on time
            else:
                form = rng.choice(["+00:00", "Z", "+01:00", "midnight+01"])
                hour_utc = rng.choice([7, 12, 15, 17])
            # Render the instant in the chosen offset form.  The
            # "midnight+01" form is stamped 00:30 local on the NEXT
            # calendar day, which is 23:30 UTC on the trade date itself:
            # its date string is not its UTC (trade) date.
            if form == "midnight+01":
                stamp = f"{_dstr(day + timedelta(days=1))}T00:30:00+01:00"
                utc_hour = 23
            elif form == "Z":
                stamp = f"{_dstr(day)}T{hour_utc:02d}:30:00Z"
                utc_hour = hour_utc
            elif form == "+01:00":
                stamp = f"{_dstr(day)}T{hour_utc + 1:02d}:30:00+01:00"
                utc_hour = hour_utc
            else:
                stamp = f"{_dstr(day)}T{hour_utc:02d}:30:00+00:00"
                utc_hour = hour_utc
            fx_rows.append([stamp, quote, f"{rate:.6f}"])
            # Answers must be computed from the rate as written to the
            # file (6 decimals), not the unrounded internal walk.
            # fx_info also keeps the stamp's own date string, for the
            # local-date trap variant below.
            fx_info[_dstr(day)] = (round(rate, 6), quote, utc_hour, stamp[:10])
        # Weekend FX prints: real rows in the file, on dates that are
        # not union trade dates.  The memo says to ignore them; using
        # one as a "previous fix" corrupts the next trade date.
        weekend_source = [d for d in union_dates if d.weekday() == 4][:4]
        for friday in weekend_source:
            saturday = friday + timedelta(days=1)
            fx_rows.append([f"{_dstr(saturday)}T10:00:00Z", "EURUSD", "9.999999"])
        # Row order in fx.csv carries no meaning (the memo says so):
        # fixes must be filed by their UTC instant, not by file position.
        rng.shuffle(fx_rows)

        cutoff = 16

        def effective_rates(fix_inversion=True, method="walkback",
                            key_by_stamp_date=False):
            """Effective USD-per-EUR rate per union date.

            ``walkback`` (the memo rule): a date uses its own fix only
            when that fix is on time (at or before the cut-off); a late
            fix is never used, and the date falls back to the most
            recent ON-TIME fix before it.  ``prev_date`` is the old trap:
            fall back to the previous date's fix even if it was late.
            ``same_day`` uses a late fix on its own date.  With
            ``key_by_stamp_date`` the fixes are (wrongly) filed under
            the calendar date written in their stamp instead of their
            UTC date and used at face value with no cut-off test.
            """
            if key_by_stamp_date:
                by_stamp = {}
                for day in union_dates:
                    rate, quote, _h, stamp_date = fx_info[_dstr(day)]
                    by_stamp[stamp_date] = rate if quote == "EURUSD" else (
                        1 / rate if fix_inversion else rate)
                filled, prev = {}, None
                for day in union_dates:
                    if _dstr(day) in by_stamp:
                        prev = by_stamp[_dstr(day)]
                    filled[_dstr(day)] = prev if prev is not None else 1.0
                return filled
            out, prev_any, prev_on_time = {}, None, None
            for day in union_dates:
                rate, quote, utc_hour, _sd = fx_info[_dstr(day)]
                eff = rate if quote == "EURUSD" else (1 / rate if fix_inversion else rate)
                on_time = utc_hour <= cutoff - 1  # stamps are at :30
                if method == "walkback":
                    chosen = eff if on_time else prev_on_time
                elif method == "prev_date":
                    chosen = eff if on_time else prev_any
                else:  # same_day
                    chosen = eff
                if on_time:
                    prev_on_time = eff
                prev_any = eff
                out[_dstr(day)] = chosen
            return out

        us_idx = {t: {_dstr(d): c for d, c in zip(us_dates, us_series[t], strict=True)}
                  for t in us_assets}
        eu_idx = {t: {_dstr(d): c for d, c in zip(eu_dates, eu_series[t], strict=True)}
                  for t in eu_assets}

        def eu_sleeve(ds, eff_map):
            return sum(holdings[t] * eu_idx[t][ds] * eff_map[ds] for t in eu_assets)

        def nav_series(eff, calendar="inner"):
            if calendar == "inner":
                days = sorted(set(us_dates) & set(eu_dates))
            else:
                days = union_dates
            navs, last_us, last_eu = {}, {}, {}
            for day in days:
                ds = _dstr(day)
                total = 0.0
                for t in us_assets:
                    if ds in us_idx[t]:
                        last_us[t] = us_idx[t][ds]
                    total += holdings[t] * last_us.get(t, 0.0)
                for t in eu_assets:
                    if ds in eu_idx[t]:
                        last_eu[t] = eu_idx[t][ds]
                    total += holdings[t] * last_eu.get(t, 0.0) * eff[ds]
                navs[ds] = total
            return navs

        eff = effective_rates()
        navs = nav_series(eff)
        inner_dates = sorted(navs)
        # The second date of the forced consecutive-late run: the memo
        # asks for the effective rate on this exact date, which pins
        # down the walk-back rule (own late fix and previous-date's
        # late fix are both wrong answers here).
        late_probe_date = _dstr(union_dates[late_start + 1])
        fx_detail_field = f"effective_fx_rate_{late_probe_date.replace('-', '')}"
        probe = [inner_dates[len(inner_dates) // 4], inner_dates[len(inner_dates) // 2],
                 inner_dates[(3 * len(inner_dates)) // 4]]
        first, last = inner_dates[0], inner_dates[-1]
        total_ret = navs[last] / navs[first] - 1
        const_eff = dict.fromkeys(eff, eff[first])
        navs_const = nav_series(const_eff)
        fx_contrib = total_ret - (navs_const[last] / navs_const[first] - 1)

        max_nav_date = max(inner_dates, key=lambda d: navs[d])
        nav_dd, _trough = _max_drawdown([navs[d] for d in inner_dates])
        eu_field = f"eu_sleeve_usd_{probe[-1].replace('-', '')}"
        expected = {f"nav_{d.replace('-', '')}": round(navs[d], 6) for d in probe}
        expected.update({
            "total_return_usd": round(total_ret, 6),
            "fx_pnl_contribution": round(fx_contrib, 6),
            "dates_dropped_by_alignment": len(union_dates) - len(inner_dates),
            "effective_fx_rate_mid_probe": round(eff[probe[1]], 6),
            fx_detail_field: round(eff[late_probe_date], 6),
            eu_field: round(eu_sleeve(probe[-1], eff), 6),
            "max_nav_value": round(navs[max_nav_date], 6),
            "max_nav_date": max_nav_date,
            "nav_max_drawdown": round(nav_dd, 6),
            "num_inner_days": len(inner_dates),
        })
        core = list(expected)[:3] + [
            "total_return_usd", "fx_pnl_contribution",
            "dates_dropped_by_alignment", "effective_fx_rate_mid_probe",
            fx_detail_field,
            eu_field, "max_nav_value", "max_nav_date", "nav_max_drawdown"]

        def variant_answer(eff_v, calendar="inner"):
            n = nav_series(eff_v, calendar)
            ks = sorted(n)
            out = {}
            for d in probe:
                out[f"nav_{d.replace('-', '')}"] = round(n.get(d, 0.0), 6)
            c = nav_series(dict.fromkeys(eff_v, eff_v[ks[0]]), calendar)
            v_max_date = max(ks, key=lambda d: n[d])
            v_dd, _t = _max_drawdown([n[d] for d in ks])
            out.update({
                "total_return_usd": round(n[ks[-1]] / n[ks[0]] - 1, 6),
                "fx_pnl_contribution": round(
                    (n[ks[-1]] / n[ks[0]] - 1) - (c[ks[-1]] / c[ks[0]] - 1), 6),
                "dates_dropped_by_alignment": len(union_dates) - len(ks),
                "effective_fx_rate_mid_probe": round(eff_v.get(probe[1], 0.0), 6),
                fx_detail_field: round(eff_v.get(late_probe_date, 0.0), 6),
                eu_field: round(eu_sleeve(probe[-1], eff_v), 6)
                if probe[-1] in eu_idx[eu_assets[0]] and probe[-1] in eff_v else 0.0,
                "max_nav_value": round(n[v_max_date], 6),
                "max_nav_date": v_max_date,
                "nav_max_drawdown": round(v_dd, 6),
                "num_inner_days": len(ks)})
            return out

        variants = {
            "naive_ignore_fx_and_calendar": variant_answer(
                dict.fromkeys(eff, 1.0), "union"),
            "no_inversion_fix": variant_answer(effective_rates(fix_inversion=False)),
            "same_day_fix_after_cutoff": variant_answer(
                effective_rates(method="same_day")),
            "previous_date_fix_even_if_late": variant_answer(
                effective_rates(method="prev_date")),
            "fx_filed_by_stamp_local_date": variant_answer(
                effective_rates(key_by_stamp_date=True)),
        }
        problems = ["currency_inverted_quote_segment", "timezone_mixed_utc_offsets",
                    "calendar_exchange_holidays"]

        us_rows, eu_rows = [], []
        for t in us_assets:
            us_rows += _price_rows(t, us_dates, us_series[t], rng)
        for t in eu_assets:
            eu_rows += _price_rows(t, eu_dates, eu_series[t], rng)
        cal_rows = [[ex, _dstr(d), "true"] for ex, ds in
                    (("EU", eu_holidays), ("US", us_holidays)) for d in ds] or \
            [["EU", _dstr(eu_holidays[0]), "true"]]
        cal_rows = [["EU", _dstr(d), "true"] for d in eu_holidays] + \
            [["US", _dstr(d), "true"] for d in us_holidays]
        spec = f"""# Methodology memo - multi-currency NAV (T2)

Portfolio holdings (shares) are on the holdings line below.
us_prices.csv is in USD
on the US calendar; eu_prices.csv is in EUR on the European calendar
(holidays are absent rows; see calendar.csv).

holdings: {json.dumps(holdings, sort_keys=True)}

## Convention table
- base_currency: USD
- fx_cutoff_utc_hour: {cutoff}
- portfolio_calendar: inner (dates present on both exchanges)
- nav_dates: {", ".join(probe)}
- fx_detail_date: {late_probe_date}

## Rules
1. Convert each European close to USD with the FX fix for the same trade
   date as of the 16:00 UTC cut-off. A fix's trade date is the UTC date
   of its timestamp: fx.csv timestamps use several UTC offset forms, and
   a stamp just after midnight in a +01:00 form (00:30 local) is 23:30
   UTC on the previous day, so its date string is not its trade date.
   Convert each timestamp to UTC first and file the fix under its UTC
   date. A fix is ON TIME when its UTC instant is at or before 16:00 UTC
   on its trade date, and LATE when it is after 16:00 UTC. The effective
   rate for a trade date is: its own fix, if that fix is on time;
   otherwise the most recent ON-TIME fix on the latest earlier union
   date that has one (walk back past any number of consecutive late
   fixes; a late fix is never used, for its own date or any later one).
   Row order in fx.csv is not chronological and carries no meaning.
   fx.csv also contains weekend prints on dates that are not union
   trade dates; they are not fixes for any trade date and must be
   ignored (in particular, never as a walked-back fix).
2. The quote column governs direction: EURUSD is USD per EUR (multiply),
   USDEUR is EUR per USD (divide by the rate to get USD). Two separate
   segments of fx.csv are quoted inverted; the quote column is the
   only flag, and it must be checked on every row.
3. NAV on a date is sum over holdings of shares * USD close, on the inner
   calendar. total_return_usd is last inner NAV / first inner NAV - 1.
4. fx_pnl_contribution is total_return_usd minus the total return of the
   same portfolio with every European conversion done at the first inner
   date's effective rate.
5. dates_dropped_by_alignment is the number of union dates (dates on
   either exchange) that the inner calendar drops.
6. effective_fx_rate_mid_probe is the effective USD-per-EUR rate actually
   used (after rules 1-2) on the middle nav_date.
   {fx_detail_field} is the effective rate applied on fx_detail_date
   under the same rules: that date's own fix arrived after the cut-off
   (as did the previous union date's), so the applied rate is the
   walked-back most recent on-time fix - neither that date's own fix
   nor the previous date's fix. The EU sleeve value on
   the last nav_date is the sum over European holdings of shares * EUR
   close * that date's effective rate. max_nav_value / max_nav_date are
   the largest inner-calendar NAV and its date. nav_max_drawdown is the
   minimum over inner dates of (NAV - running maximum NAV) / running
   maximum NAV: drawdowns are reported as a negative fraction (for
   example -0.083 means the NAV fell 8.3% from its running peak; 0.0 only
   if the NAV never falls). Do not report a drawdown as a positive number.

## Allowed problem codes (report exactly these, under exactly these rules)
The problems array may contain only: currency, timezone, calendar.
Report a code if and only if its rule is met:
- currency: at least one fx.csv row has quote USDEUR (the inverted
  quote segments).
- timezone: fx.csv timestamps use more than one UTC-offset form, or at
  least one stamp's local calendar date differs from its UTC date.
- calendar: the US and EU price files trade on different sets of dates.
The following are explicitly NOT flagged and must not be reported:
out_of_order (fx.csv row order is not chronological by design and
carries no meaning), and the weekend FX prints (they are ignored data
with no category code; there is no non-trading-print code in this
task's allowed list). Exchange holidays are absent rows and are the
calendar code above, not missing.

Report NAV on each nav_date, total_return_usd, fx_pnl_contribution,
dates_dropped_by_alignment, effective_fx_rate_mid_probe,
{fx_detail_field}, the EU sleeve
value, max_nav_value, max_nav_date, nav_max_drawdown and num_inner_days,
plus the problems array defined above.
"""
        files = {
            "us_prices.csv": _csv(["date", "ticker", "open", "high", "low", "close", "volume"], us_rows),
            "eu_prices.csv": _csv(["date", "ticker", "open", "high", "low", "close", "volume"], eu_rows),
            "fx.csv": _csv(["timestamp", "quote", "rate"], fx_rows),
            "calendar.csv": _csv(["exchange", "date", "is_holiday"], cal_rows),
            "spec.md": spec,
        }
        return files, expected, core, variants, problems

    # ------------------------------------------------------------------ T3
    def _t3_once(self, rng: random.Random):
        cfg = self.settings["reconciliation"]
        rows = cfg["rows"]
        dates = _business_dates(rng, rows)
        tickers = [f"REC{i + 1}" for i in range(cfg["assets"])]
        truth = {}
        for t in tickers:
            rets = [rng.gauss(0.0004, 0.012) for _ in range(rows - 1)]
            truth[t] = _closes(rng, rng.uniform(40, 170), [0.0] + rets)
        report_time = f"{_dstr(dates[-1])}T09:00:00Z"
        switch_idx = rows // 2

        a_rows, b_rows, corr_rows = [], [], []
        a_val, b_val, a_status = {}, {}, {}
        for t in tickers:
            a_drop = set(rng.sample(range(10, rows - 10), 6))
            b_drop = set(rng.sample(range(10, rows - 10), 6))
            indicative = set(rng.sample(range(rows), 22))
            for i, day in enumerate(dates):
                ds = _dstr(day)
                if i not in a_drop:
                    status = "indicative" if i in indicative else "firm"
                    a_val[(t, ds)] = truth[t][i]
                    a_status[(t, ds)] = status
                    a_rows.append([ds, t, f"{truth[t][i]:.2f}", status])
                if i not in b_drop:
                    unit = "dollars" if i < switch_idx else "cents"
                    val = truth[t][i] if unit == "dollars" else truth[t][i] * 100
                    b_val[(t, ds)] = truth[t][i]
                    b_rows.append([ds, t, f"{val:.2f}", unit])
        # Conflicting values between the vendors on chosen keys.
        conflict_keys = []
        for _ in range(26):
            t = rng.choice(tickers)
            i = rng.randint(5, rows - 6)
            ds = _dstr(dates[i])
            if (t, ds) in a_val and (t, ds) in b_val and (t, ds) not in conflict_keys:
                conflict_keys.append((t, ds))
                b_val[(t, ds)] = round(truth[t][i] * rng.choice([1.06, 0.93, 1.11]), 2)
        for row in b_rows:
            key = (row[1], row[0])
            if key in b_val:
                unit = "dollars" if dates.index(datetime.strptime(row[0], "%Y-%m-%d")) < switch_idx else "cents"
                row[2] = f"{b_val[key] if unit == 'dollars' else b_val[key] * 100:.2f}"
        # A conflict modification must never create an equal consecutive
        # close for a ticker in Vendor B's file order.
        prev_by_ticker: dict[str, float] = {}
        for row in b_rows:
            value = float(row[2])
            if prev_by_ticker.get(row[1]) == value:
                bump = 1.0 if row[3] == "cents" else 0.01
                row[2] = f"{value + bump:.2f}"
                b_val[(row[1], row[0])] = round(b_val[(row[1], row[0])] + 0.01, 2)
                value = float(row[2])
            prev_by_ticker[row[1]] = value
        # A few within-file duplicate rows in Vendor A (first copy governs).
        dup_keys = []
        for _ in range(4):
            row = list(rng.choice(a_rows))
            dup_close = round(float(row[2]) * 1.03, 2)
            # The duplicate copy must not equal a neighbouring date's
            # close for its ticker either (raw-file consecutive check).
            _idx = [_dstr(d) for d in dates].index(row[0])
            _neighbours = {truth[row[1]][j] for j in (_idx - 1, _idx + 1)
                           if 0 <= j < len(dates)}
            while dup_close in _neighbours:
                dup_close = round(dup_close * 1.01 + 0.11, 2)
            row[2] = f"{dup_close:.2f}"
            a_rows.append(row)
            dup_keys.append((row[1], row[0]))
        # Corrections: some effective at report time, some too late;
        # some keys carry two corrections (the later supersedes the
        # earlier); as_of stamps use mixed UTC offsets; at least one
        # corrected date sits in Vendor B's cents segment (correction
        # closes are in dollars regardless of Vendor B's unit that day).
        corr_entries: list[dict] = []

        def _as_of(day_str, utc_hour, form=None):
            """(display stamp in an offset form, normalised UTC)."""
            form = form or rng.choice(["Z", "+01:00", "-05:00"])
            if form == "+01:00":
                display = f"{day_str}T{utc_hour + 1:02d}:00:00+01:00"
            elif form == "-05:00" and utc_hour >= 5:
                display = f"{day_str}T{utc_hour - 5:02d}:00:00-05:00"
            else:
                display = f"{day_str}T{utc_hour:02d}:00:00Z"
            return display, f"{day_str}T{utc_hour:02d}:00:00Z"

        def _add_corr(t, ds, val, day_str, utc_hour, form=None):
            display, utc = _as_of(day_str, utc_hour, form)
            corr_entries.append({"key": (t, ds), "val": val,
                                 "display": display, "utc": utc})
            corr_rows.append([t, ds, f"{val:.2f}", display])

        # Correction keys are chosen so that every correction rule is
        # exercised: the first key belongs to the probe ticker (it
        # becomes a probe date below), and at least one key sits in
        # Vendor B's cents segment (correction closes stay in dollars).
        date_index = {_dstr(d): i for i, d in enumerate(dates)}
        cents_keys = [k for k in conflict_keys if date_index[k[1]] >= switch_idx]
        chosen: list = []
        probe_pool = [k for k in conflict_keys if k[0] == tickers[0]]
        if probe_pool:
            chosen.append(rng.choice(probe_pool))
        if cents_keys:
            cents_choice = rng.choice(cents_keys)
            if cents_choice not in chosen:
                chosen.append(cents_choice)
        pool = [k for k in conflict_keys if k not in chosen]
        target = min(8, len(conflict_keys))
        if len(pool) >= target - len(chosen):
            chosen += rng.sample(pool, target - len(chosen))
        corr_keys = chosen
        # Guarantee a conflicted key with Vendor A merely indicative and
        # no correction, so the median path is exercised on a conflict.
        if not any(k not in corr_keys and a_status.get(k) == "indicative"
                   for k in conflict_keys):
            cands = [k for k in conflict_keys if k not in corr_keys]
            if cands:
                t0, ds0 = rng.choice(cands)
                a_status[(t0, ds0)] = "indicative"
                for row in a_rows:
                    if row[0] == ds0 and row[1] == t0:
                        row[3] = "indicative"
                        break
        superseded = 0
        for idx, (t, ds) in enumerate(corr_keys):
            late = rng.random() < 0.3 and idx > 0  # first key never late
            val = round(truth[t][dates.index(datetime.strptime(ds, "%Y-%m-%d"))] * 1.01, 2)
            if not late and superseded < 3:
                # An earlier correction for the same key, superseded by
                # the later one below.  Both stamps fall on the same UTC
                # day in offset forms chosen so that string order and
                # instant order DISAGREE: the earlier instant (05:00Z)
                # displays as 06:00+01:00 while the later one (08:00Z)
                # displays as 03:00-05:00, so comparing the stamp
                # strings as written crowns the superseded value.
                _add_corr(t, ds, round(val * 0.97, 2),
                          _dstr(dates[-1]), 5, form="+01:00")
                superseded += 1
                _add_corr(t, ds, val, _dstr(dates[-1]), 8, form="-05:00")
            else:
                _add_corr(t, ds, val, _dstr(dates[-1]), 10 if late else 8)
        rng.shuffle(corr_rows)

        def corr_best(key, mode="latest"):
            """Winning correction value for a key, or None.

            ``latest``: latest as_of instant at or before report_time
            (the memo rule).  ``first``: earliest such correction (trap).
            ``lexicographic``: largest as_of string as written (trap:
            mixed offsets make string order differ from instant order).
            """
            entries = [e for e in corr_entries
                       if e["key"] == key and e["utc"] <= report_time]
            if not entries:
                return None
            if mode == "first":
                return min(entries, key=lambda e: e["utc"])["val"]
            if mode == "lexicographic":
                return max(entries, key=lambda e: e["display"])["val"]
            return max(entries, key=lambda e: e["utc"])["val"]

        corr = {}
        for t, ds in corr_keys:
            best = corr_best((t, ds))
            if best is not None:
                corr[(t, ds)] = (best, report_time)

        # Exact precedence for the correct answer, matching the memo.
        def resolve_correct():
            resolved, paths = {}, {"correction": 0, "vendor_a": 0, "median": 0}
            for key in sorted(set(a_val) | set(b_val)):
                av, bv = a_val.get(key), b_val.get(key)
                applicable = key in corr and corr[key][1] <= report_time
                conflicted = applicable or (
                    av is not None and bv is not None and abs(av - bv) > 0.01)
                if applicable:
                    resolved[key] = corr[key][0]
                    if conflicted:
                        paths["correction"] += 1
                elif av is not None and a_status.get(key) != "indicative":
                    resolved[key] = av
                    if conflicted:
                        paths["vendor_a"] += 1
                else:
                    avail = sorted(v for v in (av, bv) if v is not None)
                    resolved[key] = (avail[len(avail) // 2] if len(avail) % 2
                                     else (avail[0] + avail[-1]) / 2)
                    if conflicted:
                        paths["median"] += 1
            return resolved, paths

        resolved, paths = resolve_correct()

        def metrics(res):
            per = {t: [] for t in tickers}
            for (t, ds), v in sorted(res.items()):
                per[t].append((ds, v))
            common = sorted(set.intersection(*[ {d for d, _ in per[t]} for t in tickers])) \
                if tickers else []
            series = []
            by = {t: dict(per[t]) for t in tickers}
            for ds in common:
                series.append(_mean([by[t][ds] / by[t][common[0]] for t in tickers]))
            dd, trough = _max_drawdown(series)
            return series, common, dd, (common[trough] if common else None)

        series, common, dd, trough = metrics(resolved)
        probe_ticker = tickers[0]
        # Keep the corrected probe dates (the first carries a superseded
        # correction pair by construction); only top up if short.
        probe_dates = [ds for (t, ds) in corr_keys if t == probe_ticker][:2]
        for cand in (common[len(common) // 3], common[(2 * len(common)) // 3]):
            if len(probe_dates) >= 2:
                break
            if cand not in probe_dates:
                probe_dates.append(cand)
        expected = {
            "resolved_by_correction": paths["correction"],
            "resolved_by_vendor_a": paths["vendor_a"],
            "resolved_by_median": paths["median"],
            "reconciled_total_return": round(series[-1] - 1, 6),
            "max_drawdown": round(dd, 6),
            "max_drawdown_trough_date": trough,
            "num_resolved_days": len(common),
        }
        for d in probe_dates:
            expected[f"resolved_close_{probe_ticker}_{d.replace('-', '')}"] = \
                round(resolved[(probe_ticker, d)], 6)
        core = ["resolved_by_correction", "resolved_by_vendor_a",
                "resolved_by_median", "reconciled_total_return", "max_drawdown",
                "max_drawdown_trough_date"] + \
            [k for k in expected if k.startswith("resolved_close_")]

        def variant(mode):
            if mode == "vendor_a_always":
                res = {k: (a_val.get(k) if a_val.get(k) is not None else b_val.get(k))
                       for k in set(a_val) | set(b_val)}
                res = {k: v for k, v in res.items() if v is not None}
                p = {"correction": 0, "vendor_a": 0, "median": 0}
            elif mode == "no_corrections":
                res = {}
                p = {"correction": 0, "vendor_a": 0, "median": 0}
                for key in sorted(set(a_val) | set(b_val)):
                    av, bv = a_val.get(key), b_val.get(key)
                    conflicted = av is not None and bv is not None and abs(av - bv) > 0.01
                    if av is not None and a_status.get(key) != "indicative":
                        res[key] = av
                        if conflicted:
                            p["vendor_a"] += 1
                    else:
                        avail = sorted(v for v in (av, bv) if v is not None)
                        res[key] = (avail[len(avail) // 2] if len(avail) % 2
                                    else (avail[0] + avail[-1]) / 2)
                        if conflicted:
                            p["median"] += 1
            elif mode in ("first_correction_wins", "asof_string_compare"):
                # Correct precedence, but the wrong correction wins.
                res, paths_tmp = {}, {"correction": 0, "vendor_a": 0, "median": 0}
                sel = "first" if mode == "first_correction_wins" else "lexicographic"
                for key in sorted(set(a_val) | set(b_val)):
                    av, bv = a_val.get(key), b_val.get(key)
                    best = corr_best(key, sel)
                    conflicted = best is not None or (
                        av is not None and bv is not None and abs(av - bv) > 0.01)
                    if best is not None:
                        res[key] = best
                        if conflicted:
                            paths_tmp["correction"] += 1
                    elif av is not None and a_status.get(key) != "indicative":
                        res[key] = av
                        if conflicted:
                            paths_tmp["vendor_a"] += 1
                    else:
                        avail = sorted(v for v in (av, bv) if v is not None)
                        res[key] = (avail[len(avail) // 2] if len(avail) % 2
                                    else (avail[0] + avail[-1]) / 2)
                        if conflicted:
                            paths_tmp["median"] += 1
                s, c, d2, tr = metrics(res)
                out = {"resolved_by_correction": paths_tmp["correction"],
                       "resolved_by_vendor_a": paths_tmp["vendor_a"],
                       "resolved_by_median": paths_tmp["median"],
                       "reconciled_total_return": round(s[-1] - 1, 6),
                       "max_drawdown": round(d2, 6),
                       "max_drawdown_trough_date": tr,
                       "num_resolved_days": len(c)}
                for d in probe_dates:
                    out[f"resolved_close_{probe_ticker}_{d.replace('-', '')}"] = \
                        round(res.get((probe_ticker, d), 0.0), 6)
                return out
            else:  # ignore_indicative
                res = {}
                p = {"correction": 0, "vendor_a": 0, "median": 0}
                for key in sorted(set(a_val) | set(b_val)):
                    av, bv = a_val.get(key), b_val.get(key)
                    applicable = key in corr and corr[key][1] <= report_time
                    conflicted = applicable or (
                        av is not None and bv is not None and abs(av - bv) > 0.01)
                    if applicable:
                        res[key] = corr[key][0]
                        if conflicted:
                            p["correction"] += 1
                    elif av is not None:
                        res[key] = av
                        if conflicted:
                            p["vendor_a"] += 1
                    else:
                        res[key] = bv
                        if conflicted:
                            p["median"] += 1
            s, c, d2, tr = metrics(res)
            out = {"resolved_by_correction": p["correction"],
                   "resolved_by_vendor_a": p["vendor_a"],
                   "resolved_by_median": p["median"],
                   "reconciled_total_return": round(s[-1] - 1, 6),
                   "max_drawdown": round(d2, 6),
                   "max_drawdown_trough_date": tr,
                   "num_resolved_days": len(c)}
            for d in probe_dates:
                out[f"resolved_close_{probe_ticker}_{d.replace('-', '')}"] = \
                    round(res.get((probe_ticker, d), 0.0), 6)
            return out

        variants = {"vendor_a_always_no_corrections": variant("vendor_a_always"),
                    "no_corrections": variant("no_corrections"),
                    "ignore_indicative_flag": variant("ignore_indicative"),
                    "first_correction_wins": variant("first_correction_wins"),
                    "asof_string_compare": variant("asof_string_compare")}
        problems = ["source_conflict_vendor_disagreement", "unit_change_vendor_b_unit_column",
                    "duplicate_vendor_a_rows"]
        # The memo's missing rule, applied to the key exactly as stated:
        # a business date in a ticker's union range reported by neither
        # vendor.  A date reported by only one vendor is not missing.
        _all_keys = set(a_val) | set(b_val)
        for t in tickers:
            present = sorted(ds for (tt, ds) in _all_keys if tt == t)
            if any(_dstr(d) not in present
                   for d in dates
                   if present[0] <= _dstr(d) <= present[-1]):
                problems.append("missing_no_vendor_reports_date")
                break
        spec = f"""# Methodology memo - vendor reconciliation (T3)

report_time: {report_time}
probe_ticker: {probe_ticker}
probe_dates: {", ".join(probe_dates)}

Files: vendor_a.csv (date,ticker,close,status; closes in dollars),
vendor_b.csv (date,ticker,close,unit; the unit column states dollars or
cents for that row - one vendor switches units mid-file, stated in the
column), corrections.csv (ticker,date,close,as_of). Correction closes
are always in dollars, including for dates where Vendor B's unit is
cents. corrections.csv can contain several rows for the same
ticker-date (one supersedes another) and its rows are not in as_of
order; its as_of stamps use mixed UTC offsets.

## Resolution order (per ticker and date, over the union of vendor dates)
0. Within a vendor file, a repeated ticker-date row keeps its first
   occurrence in file order.
1. Convert Vendor B closes using its unit column (cents means /100).
2. A conflicted key is one with an applicable correction, or with both
   vendors present and converted closes differing by more than 0.01.
3. If corrections exist with as_of <= report_time, the one with the
   latest such as_of INSTANT wins and supersedes every earlier
   correction for that key (path: correction). Compare as_of instants
   in UTC: the stamps use mixed offset forms, so convert each stamp to
   UTC before comparing it with report_time or with other stamps;
   comparing the stamp strings as written gives the wrong winner on
   some keys. (report_time above is stated in UTC.)
4. Else, if Vendor A reports the date with status other than indicative,
   Vendor A wins (path: vendor_a).
5. Else the median of the available converted vendor closes wins (path:
   median). With two available values the median is their mean; with one
   it is that value.

## Drawdown sign convention
max_drawdown is the minimum over aligned dates of (path - running
maximum path) / running maximum path, reported as a negative fraction
(for example -0.083 means an 8.3% fall from the running peak; 0.0 only
if the path never falls). Do not report a drawdown as a positive number.

## Allowed problem codes (report exactly these, under exactly these rules)
The problems array may contain only: source_conflict, unit_change,
duplicate, missing. Report a code if and only if its rule is met:
- source_conflict: at least one conflicted key exists (rule 2 above,
  judged after unit conversion and first-occurrence de-duplication).
- unit_change: Vendor B's unit column takes more than one distinct
  value in the file.
- duplicate: at least one vendor file contains a repeated ticker-date
  row (judged on the raw file, before rule 0 is applied).
- missing: a business date inside a ticker's union date range (first to
  last date reported by either vendor for that ticker) has no row from
  EITHER vendor for that ticker. A date present in only one vendor is
  NOT missing: it is simply resolved from that vendor under the
  resolution order, and it must not be reported as missing.

Report the three path counts over conflicted keys only, the resolved
closes for the probe ticker on the probe dates, and, from the resolved
series aligned on dates present for every ticker, the equal-weight
buy-and-hold total return (mean of last/first close minus 1) and the max
drawdown of the mean normalised price path with its trough date, plus
the problems array defined above.
"""
        files = {
            "vendor_a.csv": _csv(["date", "ticker", "close", "status"], a_rows),
            "vendor_b.csv": _csv(["date", "ticker", "close", "unit"], b_rows),
            "corrections.csv": _csv(["ticker", "date", "close", "as_of"], corr_rows),
            "spec.md": spec,
        }
        return files, expected, core, variants, problems

    # ------------------------------------------------------------------ T4
    def _t4_once(self, rng: random.Random):
        cfg = self.settings["split_holdings"]
        rows = cfg["rows"]
        dates = _business_dates(rng, rows)
        date_strs = [_dstr(d) for d in dates]
        assets = [f"SPL{i + 1}" for i in range(cfg["assets"])]
        # Adjusted-basis paths, then un-adjust with explicit events.
        adj = {}
        for t in assets:
            rets = [rng.gauss(0.0004, 0.012) for _ in range(rows - 1)]
            adj[t] = _closes(rng, rng.uniform(30, 120), [0.0] + rets)
        events = []  # (ticker, date_str, A, B)
        # The first ticker splits twice: a forward split, then later a
        # reverse split on the same ticker (with cash-in-lieu).  The
        # split table additionally carries a 1-for-1 no-op row, added
        # below, which is not an event at all.
        ev1 = (assets[0], date_strs[rng.randint(45, 65)], 2, 1)
        ev2 = (assets[1], date_strs[rng.randint(100, 140)], 1, 3)
        ev3 = (assets[0], date_strs[rng.randint(125, 150)], 1, 3)
        events = [ev1, ev2, ev3]
        if rng.random() < 0.5:
            events.append((assets[2], date_strs[rng.randint(90, 120)], 3, 1))
        noop_row = (assets[2], date_strs[rng.randint(30, 160)], 1, 1)
        prices = {t: list(adj[t]) for t in assets}
        for t, ds, a, b in events:
            idx = date_strs.index(ds)
            for i in range(idx):
                prices[t][i] = round(prices[t][i] * (a / b), 2)
        # Guarantee no equal consecutive closes on the unadjusted series.
        for t in assets:
            for i in range(1, rows):
                if prices[t][i] == prices[t][i - 1]:
                    prices[t][i] = round(prices[t][i] + 0.01, 2)

        price_at = {t: dict(zip(date_strs, prices[t], strict=True)) for t in assets}

        # Trades: buys, a pre-split buy settling across the first ex-date,
        # and later sells. Settlement is booked date + 2 business days.
        def add_business(ds, n):
            return date_strs[min(date_strs.index(ds) + n, rows - 1)]

        trades = []  # (date, ticker, signed shares)
        trades.append((date_strs[5], assets[0], 120))
        trades.append((date_strs[6], assets[1], 150))
        trades.append((date_strs[7], assets[2], 90))
        cross_date = date_strs[date_strs.index(ev1[1]) - 2]
        trades.append((cross_date, assets[0], 37))
        # Partial sell between the first ticker's two splits (splits a
        # FIFO lot), a buy settling across its reverse split, and a
        # partial sell after it.
        trades.append((date_strs[90], assets[0], -40))
        cross2_date = date_strs[date_strs.index(ev3[1]) - 2]
        trades.append((cross2_date, assets[0], 25))
        trades.append((date_strs[165], assets[0], -30))
        trades.append((date_strs[120], assets[1], 60))
        trades.append((date_strs[150], assets[1], -50))
        trades.append((date_strs[160], assets[2], -30))
        trade_rows = [[ds, t, sh, f"{price_at[t][ds]:.2f}"]
                      for ds, t, sh in trades]

        def simulate(entitlement="settlement", apply_splits=True, lifo=False,
                     first_event_only=False):
            lots = {t: [] for t in assets}  # [shares, cost_per_share, settlement]
            cash_lieu, realised = 0.0, 0.0
            remaining_cost = 0.0
            daily_value, daily_flow = [], []
            trades_by_date = {}
            for ds, t, sh in trades:
                trades_by_date.setdefault(ds, []).append((t, sh))
            use_events = events
            if first_event_only:
                # Wrong-method variant: only each ticker's earliest event
                # is applied, later events on the same ticker are missed.
                use_events, seen = [], set()
                for t, ds, a, b in sorted(events, key=lambda e: e[1]):
                    if t not in seen:
                        seen.add(t)
                        use_events.append((t, ds, a, b))
            event_by_date = {}
            for t, ds, a, b in use_events:
                event_by_date.setdefault(ds, []).append((t, a, b))
            twr_index, prev_value = None, None
            for ds in date_strs:
                flow = 0.0
                if apply_splits and ds in event_by_date:
                    for t, a, b in event_by_date[ds]:
                        ratio = a / b
                        for lot in lots[t]:
                            eligible = (lot[2] < ds) if entitlement == "settlement" \
                                else True  # booked lots are already held
                            if not eligible:
                                continue
                            new_shares = lot[0] * ratio
                            whole = math.floor(new_shares + 1e-9)
                            frac = new_shares - whole
                            if frac > 1e-9:
                                cash_lieu += frac * price_at[t][ds]
                            lot[0] = whole
                            lot[1] = lot[1] / ratio
                        lots[t] = [lot for lot in lots[t] if lot[0] > 0]
                for t, sh in trades_by_date.get(ds, []):
                    px = price_at[t][ds]
                    flow += sh * px
                    if sh > 0:
                        lots[t].append([float(sh), px, add_business(ds, 2)])
                    else:
                        to_sell, order = -sh, (reversed(lots[t]) if lifo else lots[t])
                        order = list(order)
                        for lot in order:
                            if to_sell <= 0:
                                break
                            take = min(lot[0], to_sell)
                            realised += take * (px - lot[1])
                            lot[0] -= take
                            to_sell -= take
                        lots[t] = [lot for lot in lots[t] if lot[0] > 0]
                value = cash_lieu + sum(
                    sum(lot[0] for lot in lots[t]) * price_at[t][ds] for t in assets)
                daily_value.append(value)
                daily_flow.append(flow)
                if prev_value is not None and prev_value > 0:
                    r = (value - flow) / prev_value - 1
                    twr_index = (twr_index if twr_index is not None else 1.0) * (1 + r)
                elif value > 0 and twr_index is None:
                    twr_index = 1.0
                if value > 0:
                    prev_value = value
            # Recompute the TWR path cleanly from recorded values/flows.
            idx_series, prev, started = [], None, False
            for v, f in zip(daily_value, daily_flow, strict=True):
                if not started:
                    if v > 0:
                        idx_series.append(1.0)
                        prev, started = v, True
                    continue
                idx_series.append(idx_series[-1] * (1 + ((v - f) / prev - 1)) if prev else idx_series[-1])
                if v > 0:
                    prev = v
            dd, _ = _max_drawdown(idx_series) if idx_series else (0.0, 0)
            end_shares = {t: int(round(sum(lot[0] for lot in lots[t]))) for t in assets}
            remaining_cost = sum(lot[0] * lot[1] for t in assets for lot in lots[t])
            end_mv = sum(end_shares[t] * prices[t][-1] for t in assets)
            return {
                **{f"shares_{t}": end_shares[t] for t in assets},
                "realised_pnl_fifo": round(realised, 6),
                "unrealised_pnl": round(end_mv - remaining_cost, 6),
                "cash_in_lieu": round(cash_lieu, 6),
                "twr_total_return": round((idx_series[-1] - 1) if idx_series else 0.0, 6),
                "max_drawdown": round(dd, 6),
                "num_trades": len(trades),
                "num_splits": len(events),
            }

        expected = simulate()
        core = [f"shares_{t}" for t in assets] + [
            "realised_pnl_fifo", "unrealised_pnl", "cash_in_lieu",
            "twr_total_return", "max_drawdown"]
        variants = {
            "split_not_applied_to_holdings": simulate(apply_splits=False),
            "settlement_ignored_booked_only": simulate(entitlement="booked"),
            "lifo_instead_of_fifo": simulate(lifo=True),
            "second_split_ignored": simulate(first_event_only=True),
        }
        problems = []
        for _t, ds, a, b in events:
            problems.append(f"{'split' if a > b else 'reverse_split'}_{ds}_{a}-for-{b}")
        rows_out = []
        for t in assets:
            rows_out += _price_rows(t, dates, prices[t], rng)
        # The 1-for-1 no-op row is written to the table like any other
        # row (and the table is shuffled: row order carries no meaning),
        # but per the memo it is not an event at all.
        split_rows = [[ds, t, f"{a}-for-{b}"] for t, ds, a, b in events]
        split_rows.append([noop_row[1], noop_row[0],
                           f"{noop_row[2]}-for-{noop_row[3]}"])
        rng.shuffle(split_rows)
        spec = """# Methodology memo - split-table holdings (T4)

Files: prices.csv (unadjusted closes), splits.csv (date,ticker,ratio
with ratio A-for-B meaning A new shares for B old shares; the date is
the ex-date), trades.csv (date,ticker,shares,price; shares signed,
executions at that price). Positions are in shares as booked.

## Rules
0. The split table may list more than one event for the same ticker
   (for example a forward split followed later by a reverse split);
   every real event applies, in date order, to the lots held at the
   time, and fractional cash-in-lieu is computed at each event. The
   table's row order carries no meaning: sort by date. A row whose
   ratio is 1-for-1 (A == B) is a no-op corporate-action placeholder:
   it is not an event, it must not adjust any lot, it is not counted
   in num_splits, and it does not count for either problem code below.
1. A lot is a buy trade. Its settlement date is the booked date plus 2
   business days on the prices calendar. A split applies at the ex-date
   open only to lots whose settlement date is before the ex-date: their
   shares multiply by A/B and cost per share divides by A/B. A trade
   booked before the ex-date but settling on or after it does not
   receive that split.
2. After a split, fractional shares are cashed out: each adjusted lot
   keeps floor(shares) and the fractional part is paid in cash at the
   ex-date close (cash-in-lieu, held as cash thereafter).
3. Sells consume lots first-in first-out at the trade price; realised
   P&L is proceeds minus the FIFO cost consumed. Cash-in-lieu is not
   part of realised P&L.
4. Unrealised P&L is end market value of remaining shares (last close)
   minus their remaining cost basis; cash is excluded.
5. Time-weighted return: build a daily index starting at 1.0 on the
   first date with positive holdings value (share value plus
   cash-in-lieu, after that date's trades). Each later date's return is
   (value - net trade flow that date) / previous positive value - 1,
   where a buy is a positive flow of shares*price and a sell negative.
   Modified Dietz is not used. Max drawdown is on this index, reported
   as a negative fraction (for example -0.083 means the index fell 8.3%
   from its running peak; 0.0 only if it never falls). Do not report a
   drawdown as a positive number.

## Allowed problem codes (report exactly these, under exactly these rules)
The problems array may contain only: split, reverse_split.
Split-table rows DO count as problems, as follows:
- split: splits.csv contains at least one row whose ratio A-for-B has
  A > B (a forward split).
- reverse_split: splits.csv contains at least one row whose ratio
  A-for-B has A < B.
- A 1-for-1 row (A == B) counts for neither code (rule 0 above).
Report both codes when the table contains both kinds of row, exactly
one when it contains only one kind, and neither if the table were empty.
No other category code may be reported for this task.

Report end share counts per asset, realised_pnl_fifo, unrealised_pnl,
cash_in_lieu, twr_total_return, max_drawdown, num_trades, num_splits and
the problems array defined above.
"""
        files = {
            "prices.csv": _csv(["date", "ticker", "open", "high", "low", "close", "volume"], rows_out),
            "splits.csv": _csv(["date", "ticker", "ratio"], split_rows),
            "trades.csv": _csv(["date", "ticker", "shares", "price"], trade_rows),
            "spec.md": spec,
        }
        return files, expected, core, variants, problems

    # ------------------------------------------------------------------ T5
    def _t5_once(self, rng: random.Random):
        cfg = self.settings["portfolio_construction"]
        rows, n = cfg["rows"], cfg["assets"]
        dates = _business_dates(rng, rows)
        date_strs = [_dstr(d) for d in dates]
        assets = [f"OPT{i + 1}" for i in range(n)]
        sectors = {assets[i]: ("Alpha" if i < n // 2 else "Beta") for i in range(n)}
        series = {}
        for t in assets:
            rets = [rng.gauss(0.0004, rng.uniform(0.008, 0.016)) for _ in range(rows - 1)]
            series[t] = _closes(rng, rng.uniform(30, 160), [0.0] + rets)
        lookback, delta = cfg["lookback"], 0.2
        capital, buffer_pct, lot = 1_000_000.0, 0.02, 10
        cap_asset = 0.30
        # Sector caps are set per task below (after a first look at the
        # unconstrained optimum) so that the dominant sector's cap
        # actually binds at the first rebalance.
        sector_caps = {"Alpha": 0.70, "Beta": 0.70}
        turnover_cap, cost_bps = 0.25, 5.0
        first_idx = cfg["first_rebalance"]  # 0-based row index
        rebalance_idx = list(range(first_idx, rows - 5, cfg["rebalance_every"]))

        import numpy as np

        sector_members: dict[str, list[int]] = {}
        for i, t in enumerate(assets):
            sector_members.setdefault(sectors[t], []).append(i)

        def _solve_eq_qp(cov, fixed, equalities):
            """Equality QP over free assets; returns (weights, lambdas).

            ``equalities`` is a list of (member indices, target) rows that
            include the fixed assets' contributions on the right side.
            """
            free = [i for i in range(n) if i not in fixed]
            w = np.zeros(n)
            for i, v in fixed.items():
                w[i] = v
            if not free:
                return w, np.zeros(len(equalities))
            m = len(equalities)
            a_mat = np.zeros((m, len(free)))
            b_vec = np.zeros(m)
            for k, (members, target) in enumerate(equalities):
                b_vec[k] = target - sum(fixed[i] for i in members if i in fixed)
                for j, i in enumerate(free):
                    if i in members:
                        a_mat[k, j] = 1.0
            kkt = np.zeros((len(free) + m, len(free) + m))
            kkt[:len(free), :len(free)] = 2 * cov[np.ix_(free, free)]
            kkt[:len(free), len(free):] = a_mat.T
            kkt[len(free):, :len(free)] = a_mat
            sol = np.linalg.solve(
                kkt, np.concatenate([np.zeros(len(free)), b_vec]))
            for i, v in zip(free, sol[:len(free)], strict=True):
                w[i] = v
            return w, sol[len(free):]

        def _min_var_weights(cov, capped, enforce_sector=True):
            """Exact min-variance weights: sum=1, 0<=w<=cap, sector caps.

            Active-set solve over asset bounds and, when a sector cap is
            violated, that sector's sum as an added equality; fixed
            assets and active sector equalities are released when their
            KKT multipliers have the wrong sign.  The shrunk covariance
            is strictly positive definite, so the optimum is unique.
            """
            cap = cap_asset if capped else 1.0
            active_sectors: list[str] = []
            w = np.full(n, 1.0 / n)
            for _outer in range(20):
                fixed: dict[int, float] = {}
                lambdas = np.zeros(1 + len(active_sectors))
                for _inner in range(100):
                    equalities = [(list(range(n)), 1.0)] + [
                        (sector_members[s], sector_caps[s]) for s in active_sectors]
                    w, lambdas = _solve_eq_qp(cov, fixed, equalities)
                    free = [i for i in range(n) if i not in fixed]
                    below = [(w[i], i) for i in free if w[i] < -1e-10]
                    above = [(w[i], i) for i in free if w[i] > cap + 1e-10]
                    if below:
                        fixed[min(below)[1]] = 0.0
                        continue
                    if above:
                        fixed[max(above)[1]] = cap
                        continue
                    # KKT release check for fixed assets: stationarity
                    # is grad + A^T lambda (+ bound multiplier) = 0.
                    grad = 2 * cov @ w
                    at_lam = np.zeros(n)
                    all_eq = [(list(range(n)), 1.0)] + [
                        (sector_members[s], sector_caps[s]) for s in active_sectors]
                    for k, (members, _t) in enumerate(all_eq):
                        for i in members:
                            at_lam[i] += lambdas[k]
                    release, worst = None, 1e-9
                    for i, v in fixed.items():
                        resid = grad[i] + at_lam[i]
                        violation = -resid if v == 0.0 else resid
                        if violation > worst:
                            release, worst = i, violation
                    if release is None:
                        break
                    del fixed[release]
                if not (capped and enforce_sector):
                    return w
                sector_sum = {s: float(sum(w[i] for i in mem))
                              for s, mem in sector_members.items()}
                violated = [s for s in sector_sum
                            if sector_sum[s] > sector_caps[s] + 1e-9
                            and s not in active_sectors]
                if violated:
                    active_sectors.append(
                        max(violated, key=lambda s: sector_sum[s] - sector_caps[s]))
                    continue
                # An active sector whose multiplier is negative is not
                # really binding; release the worst such sector.
                release_sector, worst = None, -1e-9
                for k, s in enumerate(active_sectors):
                    if lambdas[1 + k] < worst:
                        release_sector, worst = s, lambdas[1 + k]
                if release_sector is None:
                    return w
                active_sectors.remove(release_sector)
            return w  # pragma: no cover - active set converges in practice

        _weight_cache: dict = {}

        def estimate_weights(end_idx, shrink=True, capped=True, enforce_sector=True):
            key = (end_idx, shrink, capped, enforce_sector)
            if key in _weight_cache:
                return _weight_cache[key]
            rets = np.array([_returns(series[t], "simple")[end_idx - lookback:end_idx]
                             for t in assets]).T
            cov = np.cov(rets, rowvar=False, ddof=1)
            if shrink:
                cov = (1 - delta) * cov + delta * np.diag(np.diag(cov))
            w = _min_var_weights(cov, capped, enforce_sector)
            _weight_cache[key] = w
            return w

        # Set the dominant sector's cap just below its unconstrained
        # first-rebalance share, so the sector cap binds there (the memo
        # and constraints file state the resulting cap).
        _w_pre = estimate_weights(first_idx, enforce_sector=False)
        _sector_sums = {}
        for _t, _wi in zip(assets, _w_pre, strict=True):
            _sector_sums[sectors[_t]] = _sector_sums.get(sectors[_t], 0.0) + float(_wi)
        _dominant = max(_sector_sums, key=_sector_sums.get)
        sector_caps[_dominant] = round(
            min(max(_sector_sums[_dominant] - 0.06, 0.25), 0.65), 4)
        if sector_caps[_dominant] >= _sector_sums[_dominant]:
            sector_caps[_dominant] = round(_sector_sums[_dominant] - 0.03, 4)
        _weight_cache.clear()

        def allocate(desired_values, nav, prices_now):
            """Floor to lots, then greedy leftover-cash allocation.

            Greedy buys one lot at a time for the first asset (largest
            residual, ties by ticker) whose residual is positive and
            whose lot cost fits in the remaining cash.  Remaining cash
            here is the whole cash after flooring, so greedy allocation
            may spend into the cash buffer; it stops when no positive-
            residual asset has an affordable lot.
            """
            shares = [int(math.floor(dv / (p * lot)) * lot)
                      for dv, p in zip(desired_values, prices_now, strict=True)]
            cash = nav - sum(s * p for s, p in zip(shares, prices_now, strict=True))
            while True:
                residuals = [dv - s * p for dv, s, p in
                             zip(desired_values, shares, prices_now, strict=True)]
                order = sorted(range(n), key=lambda i: (-residuals[i], assets[i]))
                bought = False
                for i in order:
                    cost = lot * prices_now[i]
                    if residuals[i] > 1e-9 and cash >= cost - 1e-9:
                        shares[i] += lot
                        cash -= cost
                        bought = True
                        break
                if not bought:
                    return shares, cash

        def simulate(shrink=True, capped=True, use_turnover_cap=True, apply_cost=True):
            shares = [0] * n
            cash = capital
            nav_path = []
            weights_first, shares_first, turnovers = None, None, {}
            reb_set = set(rebalance_idx)
            for idx, _ds in enumerate(date_strs):
                prices_now = [series[t][idx] for t in assets]
                if idx in reb_set:
                    nav_pre = cash + sum(s * p for s, p in zip(shares, prices_now, strict=True))
                    w = estimate_weights(idx, shrink, capped)
                    desired = [wi * nav_pre * (1 - buffer_pct) for wi in w]
                    new_shares, new_cash = allocate(desired, nav_pre, prices_now)
                    traded = sum(abs(ns - s) * p for ns, s, p in
                                 zip(new_shares, shares, prices_now, strict=True))
                    # Turnover is defined on the actual post-rounding
                    # share changes, and the cap is checked on that
                    # actual turnover (rebalances after the first only):
                    # scale every share change toward zero by
                    # cap/actual, truncated to whole lots, with no
                    # further greedy allocation afterwards.
                    if any(shares) and use_turnover_cap \
                            and traded / nav_pre > turnover_cap:
                        factor = turnover_cap / (traded / nav_pre)
                        new_shares = [
                            s + (1 if ns > s else -1)
                            * int(math.floor(abs(ns - s) * factor / lot)) * lot
                            for ns, s in zip(new_shares, shares, strict=True)]
                        new_cash = nav_pre - sum(
                            s * p for s, p in zip(new_shares, prices_now, strict=True))
                        traded = sum(abs(ns - s) * p for ns, s, p in
                                     zip(new_shares, shares, prices_now, strict=True))
                    turnover = traded / nav_pre if nav_pre else 0.0
                    turnovers[str(len(turnovers) + 1)] = round(turnover, 6)
                    if apply_cost:
                        # Cost is deducted from cash after the final
                        # share counts are set.  If that would take cash
                        # below zero, return lots (largest excess of
                        # allocated over desired value, ties by ticker)
                        # and recompute traded notional and cost, until
                        # cash after cost is >= 0.  Cash never goes
                        # negative.
                        cost = traded * cost_bps / 10000
                        while new_cash - cost < 0 and any(s > 0 for s in new_shares):
                            excess = [ns * p - dv for ns, dv, p in
                                      zip(new_shares, desired, prices_now, strict=True)]
                            pick = min(range(n), key=lambda i: (-excess[i], assets[i]))
                            if new_shares[pick] <= 0:
                                pick = max(range(n), key=lambda i: new_shares[i])
                            new_shares[pick] -= lot
                            new_cash += lot * prices_now[pick]
                            traded = sum(abs(ns - s) * p for ns, s, p in
                                         zip(new_shares, shares, prices_now, strict=True))
                            cost = traded * cost_bps / 10000
                            turnovers[str(len(turnovers))] = round(
                                traded / nav_pre, 6) if nav_pre else 0.0
                        new_cash -= cost
                    if weights_first is None:
                        weights_first = {t: round(float(wi), 6) for t, wi in zip(assets, w, strict=True)}
                        shares_first = {t: int(s) for t, s in zip(assets, new_shares, strict=True)}
                    shares, cash = new_shares, new_cash
                nav_path.append(cash + sum(s * p for s, p in zip(shares, prices_now, strict=True)))
            dd, _ = _max_drawdown(nav_path)
            return {
                "weights_first_rebalance": weights_first,
                "shares_first_rebalance": shares_first,
                "turnover_by_rebalance": turnovers,
                "num_rebalances": len(rebalance_idx),
                "final_nav": round(nav_path[-1], 6),
                "max_drawdown": round(dd, 6),
                "cash_left": round(cash, 6),
            }

        expected = simulate()
        core = ["weights_first_rebalance", "shares_first_rebalance",
                "final_nav", "max_drawdown", "cash_left"]
        variants = {
            "no_shrinkage": simulate(shrink=False),
            "no_asset_cap": simulate(capped=False),
            "no_turnover_cap_or_cost": simulate(use_turnover_cap=False, apply_cost=False),
        }
        rows_out = []
        for t in assets:
            rows_out += _price_rows(t, dates, series[t], rng)
        constraints = {
            "capital": capital, "cash_buffer_pct": buffer_pct, "lot_size": lot,
            "per_asset_cap": cap_asset, "sector_caps": sector_caps,
            "sectors": sectors, "turnover_cap": turnover_cap, "cost_bps": cost_bps,
            "lookback_returns": lookback, "shrinkage_delta": delta,
            "rebalance_row_indices_1based": [i + 1 for i in rebalance_idx],
        }
        spec = """# Methodology memo - constrained portfolio construction (T5)

Files: prices.csv and constraints.json (capital, cash buffer, lot size,
per-asset and sector caps, turnover cap, cost in basis points, lookback,
shrinkage delta, sectors and the 1-based rebalance row indices).

## Pipeline (at each rebalance row, in exactly this order)
1. Estimate the covariance from the last lookback_returns simple daily
   returns ending at that row (sample covariance, ddof=1), then shrink
   toward its diagonal: Sigma = (1-delta)*Sigma + delta*diag(Sigma).
2. Weights minimise w' Sigma w subject to ALL of: sum(w)=1, long-only
   (w >= 0), the per-asset cap (w_i <= per_asset_cap for every asset)
   and the sector caps (for each sector, the sum of weights of its
   assets <= that sector's cap in constraints.json). Both caps are hard
   constraints, not targets. The shrunk covariance is strictly positive
   definite, so the optimum is unique.
3. Trade to the weights. Desired value per asset is
   weight * NAV_pre * (1 - cash_buffer_pct), where NAV_pre is cash plus
   holdings at that row's closes. Convert desired values to shares:
   (a) floor each desired value to a whole lot multiple;
   (b) greedy allocation: repeatedly take the asset with the largest
   residual (desired value minus allocated value; ties by ticker name,
   alphabetically first), and if its residual is positive and one lot
   costs no more than the remaining cash, buy one lot for it; repeat
   until no asset qualifies. "Remaining cash" in this step is the whole
   cash left after flooring (NAV_pre minus allocated value), so greedy
   allocation MAY spend into the cash buffer: the final cash after this
   step can be below cash_buffer_pct * NAV_pre, down to less than the
   cheapest lot. Greedy allocation never makes cash negative, because a
   lot is bought only when the cash before the purchase covers it.
4. Turnover is defined on the ACTUAL post-rounding share changes of
   step 3: turnover = (sum over assets of |new shares - old shares| *
   price) / NAV_pre. At the first rebalance the old share counts are
   zero, the turnover cap does not limit the allocation, and the
   reported turnover is this actual invested fraction. At every later
   rebalance the cap is checked on this actual turnover: if it exceeds
   turnover_cap, scale every asset's share change toward zero by the
   factor turnover_cap / actual_turnover, truncate each scaled change
   to a whole number of lots toward zero, and take those as the final
   share counts (no further greedy allocation after scaling). The
   reported turnover is then recomputed from these final share counts,
   so the reported turnover is always the actual one and, after the
   first rebalance, never exceeds the cap.
5. Transaction cost, in this order: traded notional is computed from
   the final share counts of step 4; cost = cost_bps/10000 * traded
   notional; cash after trading is NAV_pre minus final holdings value
   minus cost. If that cash would be negative, return lots before
   deducting: repeatedly remove one lot from the asset whose allocated
   value most exceeds its step-3 desired value (ties by ticker name,
   alphabetically first; recompute traded notional, turnover and cost
   after each removal) until cash after cost is >= 0, then deduct the
   cost. Cash can therefore never be negative at any rebalance, and
   cash_left is the cash after the last rebalance's cost deduction.
   Between rebalances shares and cash are constant; NAV is
   shares*close plus cash daily. max_drawdown is over this daily NAV
   path, reported as a negative fraction (for example -0.083 means the
   NAV fell 8.3% from its running peak; 0.0 only if it never falls).
   Do not report a drawdown as a positive number.

Report weights and share counts at the first rebalance, turnover at each
rebalance (the actual turnover of step 4/5), num_rebalances, final_nav,
max_drawdown of the daily NAV path and cash_left. The
turnover_by_rebalance field is a JSON object whose keys are the strings
"1", "2", ... in rebalance order (the first rebalance is "1") and whose
values are the actual turnover fractions for those rebalances, each
rounded to 6 decimals: for example {"1": 0.982143, "2": 0.169956}.
weights_first_rebalance and shares_first_rebalance are likewise JSON
objects keyed by ticker.
"""
        files = {
            "prices.csv": _csv(["date", "ticker", "open", "high", "low", "close", "volume"], rows_out),
            "constraints.json": json.dumps(constraints, indent=2, sort_keys=True) + "\n",
            "spec.md": spec,
        }
        return files, expected, core, variants, None

    # ------------------------------------------------------------------ API
    def task_generators(self):
        return [
            ("convention_risk", self._t1_once),
            ("fx_calendar", self._t2_once),
            ("reconciliation", self._t3_once),
            ("split_holdings", self._t4_once),
            ("portfolio_construction", self._t5_once),
        ]

    def generate_all_tasks(self, output_dir, num_tasks: int = 30):
        """[Function: generate_all_tasks] Write a full task set to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "answers").mkdir(exist_ok=True)
        families = self.task_generators()
        tasks = []
        for i in range(num_tasks):
            family, builder = families[i % len(families)]
            task, files, expected = self._build(builder, family)
            task_dir = output_dir / task["id"]
            task_dir.mkdir(exist_ok=True)
            for name, text in files.items():
                (task_dir / name).write_text(text)
            with open(output_dir / "answers" / f"{task['id']}_answer.json", "w") as fh:
                json.dump(expected, fh, indent=2)
            tasks.append(task)
        with open(output_dir / "tasks.json", "w") as fh:
            json.dump(tasks, fh, indent=2)
        print(f"[generate_all_tasks] Generated {len(tasks)} tools-frontier "
              f"tasks in {output_dir}")
        return tasks


def _prompt_for_family(family: str, expected: dict) -> str:
    labels = {
        "convention_risk": "convention-switch risk report",
        "fx_calendar": "multi-currency NAV",
        "reconciliation": "vendor reconciliation",
        "split_holdings": "split-table holdings",
        "portfolio_construction": "constrained portfolio construction",
    }
    fields = []
    for key in expected:
        if key.startswith("_"):
            continue
        if key == "problems_found":
            fields.append("problems: JSON array of category codes that apply")
        elif isinstance(expected[key], dict):
            fields.append(f"{key}: JSON object as described in spec.md")
        elif isinstance(expected[key], int):
            fields.append(f"{key}: integer")
        elif isinstance(expected[key], str):
            fields.append(f"{key}: YYYY-MM-DD date string")
        else:
            fields.append(f"{key}: number (6 decimals)")
    return _prompt(labels[family], fields)


def main(argv=None):
    """[Function: main] CLI for tools-frontier task generation."""
    parser = argparse.ArgumentParser(description="Generate tools-frontier tasks")
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--count", type=int, default=30,
                        help="Number of tasks (default 30 = 6 per family)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("tasks_tools_frontier"))
    parser.add_argument("--eval", action="store_true", dest="allow_eval",
                        help="Allow a reserved evaluation seed (42/1001/1002). "
                             "Never use this for training or development data.")
    args = parser.parse_args(argv)
    generator = ToolsFrontierTaskGenerator(seed=args.seed, allow_eval=args.allow_eval)
    generator.generate_all_tasks(args.output_dir, num_tasks=args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
