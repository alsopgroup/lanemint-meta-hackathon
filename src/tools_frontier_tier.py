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
_RTOL = 1e-6
_ATOL = 1e-8


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

        expected = {
            f"ann_vol_{a}": round(_ann_vol(series[a], *vol_spec[a]), 6)
            for a in assets
        }
        expected.update({
            "rolling_max_beta": round(max_beta, 6),
            "rolling_max_beta_end_date": end_date,
            "weekly_tracking_error": round(weekly_te("log", 0, 52), 6),
            "ewma_vol_last": round(ewma_vol(assets[2], 0.94), 6),
            "sharpe_ratio": round(sharpe(assets[1], "simple"), 6),
            "num_aligned_days": rows,
            "num_return_days": rows - 1,
        })
        core = [f"ann_vol_{a}" for a in assets] + [
            "rolling_max_beta", "rolling_max_beta_end_date",
            "weekly_tracking_error", "ewma_vol_last", "sharpe_ratio"]

        naive = {
            f"ann_vol_{a}": round(_ann_vol(series[a], "simple", 252, 1), 6)
            for a in assets
        }
        naive.update({
            "rolling_max_beta": expected["rolling_max_beta"],
            "rolling_max_beta_end_date": end_date,
            "weekly_tracking_error": round(weekly_te("simple", 1, 252), 6),
            "ewma_vol_last": round(_ann_vol(series[assets[2]], "simple", 252, 1), 6),
            "sharpe_ratio": round(sharpe(assets[1], "log"), 6),
            "num_aligned_days": rows, "num_return_days": rows - 1,
        })
        nonoverlap = rolling_betas(simple[target], simple[bench_name], window, window)
        no_beta, no_end = max(nonoverlap, key=lambda p: p[0])
        variant_nonoverlap = dict(expected)
        variant_nonoverlap.update({
            "rolling_max_beta": round(no_beta, 6),
            "rolling_max_beta_end_date": _dstr(dates[no_end + 1])})
        variants = {"naive_simple_252": naive,
                    "nonoverlapping_rolling": variant_nonoverlap}

        rows_out = []
        for ticker in assets + [bench_name]:
            rows_out += _price_rows(ticker, dates, series[ticker], rng)
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

Prices are in prices.csv (date,ticker,open,high,low,close,volume) and are
already clean and aligned on {rows} business days.
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

        # FX fixes on the union calendar; a contiguous segment is quoted
        # inverted, flagged only by the quote column.
        union_dates = sorted(set(us_dates) | set(eu_dates))
        inv_start = rng.randint(50, 70)
        inv_end = inv_start + rng.randint(20, 35)
        base_rate, fx_rows, fx_info = 1.08, [], {}
        for i, day in enumerate(union_dates):
            base_rate = max(0.7, base_rate * (1 + rng.gauss(0, 0.004)))
            inverted = inv_start <= i < inv_end
            quote = "USDEUR" if inverted else "EURUSD"
            rate = (1 / base_rate) if inverted else base_rate
            form = rng.choice(["+00:00", "Z", "+01:00"])
            hour_utc = rng.choice([7, 12, 15, 17])
            # Render the instant in the chosen offset form.
            if form == "Z":
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
            fx_info[_dstr(day)] = (round(rate, 6), quote, utc_hour)

        cutoff = 16

        def effective_rates(use_prev_after_cutoff=True, fix_inversion=True):
            out, prev = {}, None
            for day in union_dates:
                rate, quote, utc_hour = fx_info[_dstr(day)]
                eff = rate if quote == "EURUSD" else (1 / rate if fix_inversion else rate)
                chosen = eff
                if use_prev_after_cutoff and utc_hour > cutoff and prev is not None:
                    chosen = prev
                out[_dstr(day)] = chosen
                prev = eff
            return out

        def nav_series(eff, calendar="inner"):
            if calendar == "inner":
                days = sorted(set(us_dates) & set(eu_dates))
            else:
                days = union_dates
            us_idx = {t: {_dstr(d): c for d, c in zip(us_dates, us_series[t], strict=True)}
                      for t in us_assets}
            eu_idx = {t: {_dstr(d): c for d, c in zip(eu_dates, eu_series[t], strict=True)}
                      for t in eu_assets}
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
        probe = [inner_dates[len(inner_dates) // 4], inner_dates[len(inner_dates) // 2],
                 inner_dates[(3 * len(inner_dates)) // 4]]
        first, last = inner_dates[0], inner_dates[-1]
        total_ret = navs[last] / navs[first] - 1
        const_eff = dict.fromkeys(eff, eff[first])
        navs_const = nav_series(const_eff)
        fx_contrib = total_ret - (navs_const[last] / navs_const[first] - 1)

        expected = {f"nav_{d.replace('-', '')}": round(navs[d], 6) for d in probe}
        expected.update({
            "total_return_usd": round(total_ret, 6),
            "fx_pnl_contribution": round(fx_contrib, 6),
            "dates_dropped_by_alignment": len(union_dates) - len(inner_dates),
            "num_inner_days": len(inner_dates),
        })
        core = list(expected)[:3] + ["total_return_usd", "fx_pnl_contribution",
                                     "dates_dropped_by_alignment"]

        def variant_answer(eff_v, calendar="inner"):
            n = nav_series(eff_v, calendar)
            ks = sorted(n)
            out = {}
            for d in probe:
                out[f"nav_{d.replace('-', '')}"] = round(n.get(d, 0.0), 6)
            c = nav_series(dict.fromkeys(eff_v, eff_v[ks[0]]), calendar)
            out.update({
                "total_return_usd": round(n[ks[-1]] / n[ks[0]] - 1, 6),
                "fx_pnl_contribution": round(
                    (n[ks[-1]] / n[ks[0]] - 1) - (c[ks[-1]] / c[ks[0]] - 1), 6),
                "dates_dropped_by_alignment": len(union_dates) - len(ks),
                "num_inner_days": len(ks)})
            return out

        variants = {
            "naive_ignore_fx_and_calendar": variant_answer(
                dict.fromkeys(eff, 1.0), "union"),
            "no_inversion_fix": variant_answer(effective_rates(fix_inversion=False)),
            "same_day_fix_after_cutoff": variant_answer(
                effective_rates(use_prev_after_cutoff=False)),
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

## Rules
1. Convert each European close to USD with the FX fix for the same trade
   date as of the 16:00 UTC cut-off. fx.csv timestamps use several UTC
   offset forms; convert each timestamp to UTC first. If the fix instant
   is after the cut-off, use the previous union date's fix instead.
2. The quote column governs direction: EURUSD is USD per EUR (multiply),
   USDEUR is EUR per USD (divide by the rate to get USD). A contiguous
   segment of fx.csv is quoted inverted; the quote column is the only flag.
3. NAV on a date is sum over holdings of shares * USD close, on the inner
   calendar. total_return_usd is last inner NAV / first inner NAV - 1.
4. fx_pnl_contribution is total_return_usd minus the total return of the
   same portfolio with every European conversion done at the first inner
   date's effective rate.
5. dates_dropped_by_alignment is the number of union dates (dates on
   either exchange) that the inner calendar drops.

Report NAV on each nav_date, total_return_usd, fx_pnl_contribution,
dates_dropped_by_alignment and num_inner_days. Also report a problems
array with the category codes that apply, from: currency, timezone,
calendar, duplicate, missing, bad_price, split, reverse_split,
unit_change, out_of_order, adj_close, column_order, source_conflict.
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
            row[2] = f"{float(row[2]) * 1.03:.2f}"
            a_rows.append(row)
            dup_keys.append((row[1], row[0]))
        # Corrections: some effective at report time, some too late.
        corr = {}
        corr_keys = rng.sample(conflict_keys, 8) if len(conflict_keys) >= 8 else conflict_keys
        for t, ds in corr_keys:
            late = rng.random() < 0.3
            as_of = f"{_dstr(dates[-1])}T{10 if late else 8:02d}:00:00Z"
            val = round(truth[t][dates.index(datetime.strptime(ds, "%Y-%m-%d"))] * 1.01, 2)
            corr_rows.append([t, ds, f"{val:.2f}", as_of])
            corr[(t, ds)] = (val, as_of)

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
        probe_dates = [ds for (t, ds) in corr_keys if t == probe_ticker][:2]
        if len(probe_dates) < 2:
            probe_dates = [common[len(common) // 3], common[(2 * len(common)) // 3]]
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
                    "ignore_indicative_flag": variant("ignore_indicative")}
        problems = ["source_conflict_vendor_disagreement", "unit_change_vendor_b_unit_column",
                    "duplicate_vendor_a_rows"]
        spec = f"""# Methodology memo - vendor reconciliation (T3)

report_time: {report_time}
probe_ticker: {probe_ticker}
probe_dates: {", ".join(probe_dates)}

Files: vendor_a.csv (date,ticker,close,status; closes in dollars),
vendor_b.csv (date,ticker,close,unit; the unit column states dollars or
cents for that row - one vendor switches units mid-file, stated in the
column), corrections.csv (ticker,date,close,as_of).

## Resolution order (per ticker and date, over the union of vendor dates)
0. Within a vendor file, a repeated ticker-date row keeps its first
   occurrence in file order.
1. Convert Vendor B closes using its unit column (cents means /100).
2. A conflicted key is one with an applicable correction, or with both
   vendors present and converted closes differing by more than 0.01.
3. If a correction exists with as_of <= report_time, the correction with
   the latest such as_of wins (path: correction).
4. Else, if Vendor A reports the date with status other than indicative,
   Vendor A wins (path: vendor_a).
5. Else the median of the available converted vendor closes wins (path:
   median). With two available values the median is their mean; with one
   it is that value.

Report the three path counts over conflicted keys only, the resolved
closes for the probe ticker on the probe dates, and, from the resolved
series aligned on dates present for every ticker, the equal-weight
buy-and-hold total return (mean of last/first close minus 1) and the max
drawdown of the mean normalised price path with its trough date. Also
report a problems array from the standard category list.
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
        ev1 = (assets[0], date_strs[rng.randint(50, 80)], 2, 1)
        ev2 = (assets[1], date_strs[rng.randint(100, 140)], 1, 3)
        events = [ev1, ev2]
        if rng.random() < 0.5:
            events.append((assets[2], date_strs[rng.randint(90, 150)], 3, 1))
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
        trades.append((date_strs[90], assets[0], -40))
        trades.append((date_strs[120], assets[1], 60))
        trades.append((date_strs[150], assets[1], -50))
        trades.append((date_strs[160], assets[2], -30))
        trade_rows = [[ds, t, sh, f"{price_at[t][ds]:.2f}"]
                      for ds, t, sh in trades]

        def simulate(entitlement="settlement", apply_splits=True, lifo=False):
            lots = {t: [] for t in assets}  # [shares, cost_per_share, settlement]
            cash_lieu, realised = 0.0, 0.0
            remaining_cost = 0.0
            daily_value, daily_flow = [], []
            trades_by_date = {}
            for ds, t, sh in trades:
                trades_by_date.setdefault(ds, []).append((t, sh))
            event_by_date = {}
            for t, ds, a, b in events:
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
        }
        problems = []
        for _t, ds, a, b in events:
            problems.append(f"{'split' if a > b else 'reverse_split'}_{ds}_{a}-for-{b}")
        rows_out = []
        for t in assets:
            rows_out += _price_rows(t, dates, prices[t], rng)
        split_rows = [[ds, t, f"{a}-for-{b}"] for t, ds, a, b in events]
        spec = """# Methodology memo - split-table holdings (T4)

Files: prices.csv (unadjusted closes), splits.csv (date,ticker,ratio
with ratio A-for-B meaning A new shares for B old shares; the date is
the ex-date), trades.csv (date,ticker,shares,price; shares signed,
executions at that price). Positions are in shares as booked.

## Rules
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
   Modified Dietz is not used. Max drawdown is on this index.

Report end share counts per asset, realised_pnl_fifo, unrealised_pnl,
cash_in_lieu, twr_total_return, max_drawdown, num_trades, num_splits and
a problems array from the standard category list.
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
        cap_asset, cap_sector = 0.30, 0.70
        turnover_cap, cost_bps = 0.25, 5.0
        first_idx = cfg["first_rebalance"]  # 0-based row index
        rebalance_idx = list(range(first_idx, rows - 5, cfg["rebalance_every"]))

        import numpy as np

        def _min_var_weights(cov, capped):
            """Exact min-variance weights under sum=1 and 0<=w<=cap.

            Active-set solve: fix the most violated bound, re-solve the
            equality QP for the free assets, then release any fixed asset
            whose KKT multiplier has the wrong sign.  The shrunk
            covariance is strictly positive definite, so this terminates
            at the unique optimum.  Sector caps are feasibility-checked
            afterwards (they do not bind at the generated optima).
            """
            cap = cap_asset if capped else 1.0
            fixed: dict[int, float] = {}
            w = np.full(n, 1.0 / n)
            for _ in range(100):
                free = [i for i in range(n) if i not in fixed]
                w = np.zeros(n)
                for i, v in fixed.items():
                    w[i] = v
                rhs = 1 - sum(fixed.values())
                if free:
                    kkt = np.zeros((len(free) + 1, len(free) + 1))
                    kkt[:len(free), :len(free)] = 2 * cov[np.ix_(free, free)]
                    kkt[:len(free), -1] = 1
                    kkt[-1, :len(free)] = 1
                    sol = np.linalg.solve(
                        kkt, np.array([0.0] * len(free) + [rhs]))
                    for i, v in zip(free, sol[:len(free)], strict=True):
                        w[i] = v
                    below = [(w[i], i) for i in free if w[i] < -1e-10]
                    above = [(w[i], i) for i in free if w[i] > cap + 1e-10]
                    if below:
                        fixed[min(below)[1]] = 0.0
                        continue
                    if above:
                        fixed[max(above)[1]] = cap
                        continue
                grad = 2 * cov @ w
                mu = _mean([grad[i] for i in free]) if free else _mean(list(grad))
                release, worst = None, 1e-9
                for i, v in fixed.items():
                    violation = (mu - grad[i]) if v == 0.0 else (grad[i] - mu)
                    if violation > worst:
                        release, worst = i, violation
                if release is None:
                    return w
                del fixed[release]
            return w  # pragma: no cover - active set converges in practice

        _weight_cache: dict = {}

        def estimate_weights(end_idx, shrink=True, capped=True):
            key = (end_idx, shrink, capped)
            if key in _weight_cache:
                return _weight_cache[key]
            rets = np.array([_returns(series[t], "simple")[end_idx - lookback:end_idx]
                             for t in assets]).T
            cov = np.cov(rets, rowvar=False, ddof=1)
            if shrink:
                cov = (1 - delta) * cov + delta * np.diag(np.diag(cov))
            w = _min_var_weights(cov, capped)
            if capped:
                sec = {}
                for t, wi in zip(assets, w, strict=True):
                    sec[sectors[t]] = sec.get(sectors[t], 0.0) + float(wi)
                if any(v > cap_sector + 1e-9 for v in sec.values()):
                    # Sector cap binds: fall back to clipping the sector
                    # and renormalising the remainder (deterministic; the
                    # generated draws make this path vanishingly rare).
                    w = np.minimum(w, cap_sector)
                    w = w / w.sum()
            _weight_cache[key] = w
            return w

        def allocate(desired_values, nav, prices_now):
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
                    if cash >= cost - 1e-9:
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
                    turnover_raw = sum(abs(dv - s * p) for dv, s, p in
                                       zip(desired, shares, prices_now, strict=True)) / nav_pre \
                        if any(shares) else sum(desired) / nav_pre
                    # First allocation from cash: turnover is the invested
                    # fraction; later turnovers use traded notional / NAV.
                    if any(shares):
                        traded_target = desired
                        if use_turnover_cap and turnover_raw > turnover_cap:
                            scale = turnover_cap / turnover_raw
                            traded_target = [s * p + (dv - s * p) * scale for dv, s, p in
                                             zip(desired, shares, prices_now, strict=True)]
                        new_shares, new_cash = allocate(traded_target, nav_pre, prices_now)
                    else:
                        new_shares, new_cash = allocate(desired, nav_pre, prices_now)
                    traded = sum(abs(ns - s) * p for ns, s, p in
                                 zip(new_shares, shares, prices_now, strict=True))
                    turnover = traded / nav_pre if nav_pre else 0.0
                    turnovers[str(len(turnovers) + 1)] = round(turnover, 6)
                    if apply_cost:
                        new_cash -= traded * cost_bps / 10000
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
            "per_asset_cap": cap_asset, "sector_caps": {"Alpha": cap_sector, "Beta": cap_sector},
            "sectors": sectors, "turnover_cap": turnover_cap, "cost_bps": cost_bps,
            "lookback_returns": lookback, "shrinkage_delta": delta,
            "rebalance_row_indices_1based": [i + 1 for i in rebalance_idx],
        }
        spec = """# Methodology memo - constrained portfolio construction (T5)

Files: prices.csv and constraints.json (capital, cash buffer, lot size,
per-asset and sector caps, turnover cap, cost in basis points, lookback,
shrinkage delta, sectors and the 1-based rebalance row indices).

## Pipeline (at each rebalance row, in this order)
1. Estimate the covariance from the last lookback_returns simple daily
   returns ending at that row (sample covariance, ddof=1), then shrink
   toward its diagonal: Sigma = (1-delta)*Sigma + delta*diag(Sigma).
2. Weights minimise w' Sigma w subject to sum(w)=1, long-only,
   per-asset cap and sector caps. The shrunk covariance is strictly
   positive definite, so the optimum is unique.
3. Trade to the weights: desired value per asset is weight * NAV *
   (1 - cash_buffer_pct), where NAV is cash plus holdings at that row's
   closes. Convert to round lots by flooring to a lot multiple, then
   greedily spend leftover cash one lot at a time on the asset with the
   largest residual (desired minus allocated value), ties by ticker,
   repeating until no lot is affordable.
4. Turnover is traded notional (sum of |change in shares| * price)
   divided by pre-trade NAV. At rebalances after the first, if turnover
   to the full target would exceed the turnover cap, scale every
   asset's desired value from its current value toward the target by
   cap/turnover first, then apply step 3 to the scaled values.
5. Deduct cost_bps of traded notional from cash. Between rebalances
   shares and cash are constant; NAV is shares*close plus cash daily.

Report weights and share counts at the first rebalance, turnover at each
rebalance, num_rebalances, final_nav, max_drawdown of the daily NAV path
and cash_left.
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
