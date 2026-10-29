"""Independent reference solver for the tools-frontier tier.

This module is deliberately independent of ``src/tools_frontier_tier.py``:
it imports nothing from the generator and derives every answer only from
the files inside one task directory (market data, auxiliary tables,
holdings/constraints JSON and the ``spec.md`` methodology memo).  Memo
parameters are parsed from the memo text, so a memo change changes the
computed answer.

Only the standard library plus NumPy (for the T5 covariance/QP step) is
used.  Public entry points:

* ``solve_task_dir(task_dir)`` -> answer dict (without scorer metadata)
* ``solve_task(task, tasks_dir)`` -> answer dict for a tasks.json entry
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------

def _read_csv(task_dir: Path, name: str) -> list[dict]:
    with open(task_dir / name, newline="") as fh:
        return list(csv.DictReader(fh))


def _spec(task_dir: Path) -> str:
    return (task_dir / "spec.md").read_text()


def _kv(spec: str, key: str) -> str:
    m = re.search(rf"^- {re.escape(key)}:\s*(.+)$", spec, re.MULTILINE)
    if not m:
        raise ValueError(f"[solve_task_dir] memo is missing '{key}'")
    return m.group(1).strip()


def _price_map(task_dir: Path, name="prices.csv"):
    """ticker -> (ordered date strings, closes aligned to those dates)."""
    rows = _read_csv(task_dir, name)
    by_ticker: dict[str, dict[str, float]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], {})[row["date"]] = float(row["close"])
    out = {}
    for ticker, mapping in by_ticker.items():
        dates = sorted(mapping)
        out[ticker] = (dates, [mapping[d] for d in dates])
    return out


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs, ddof):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def _returns(closes, kind):
    if kind == "log":
        return [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False)]
    return [b / a - 1.0 for a, b in zip(closes, closes[1:], strict=False)]


def _drawdown(values):
    peak, worst, trough = values[0], 0.0, 0
    for i, v in enumerate(values):
        peak = max(peak, v)
        if peak and (v - peak) / peak < worst:
            worst, trough = (v - peak) / peak, i
    return worst, trough


# --------------------------------------------------------------------------
# T1 - convention-switch risk report
# --------------------------------------------------------------------------

def _solve_t1(task_dir: Path) -> dict:
    spec = _spec(task_dir)
    series = _price_map(task_dir)
    # calendar.csv lists the trading dates; rows on other dates (weekend
    # prints) are excluded before any computation.  _price_map already
    # returns each ticker sorted by date.
    if (task_dir / "calendar.csv").exists():
        trading = {r["date"] for r in _read_csv(task_dir, "calendar.csv")}
        for ticker, (dates, closes) in series.items():
            kept = [(d, c) for d, c in zip(dates, closes, strict=True) if d in trading]
            series[ticker] = ([d for d, _c in kept], [c for _d, c in kept])
    tickers = sorted(series)
    answer = {}
    weekly_cfg, ewma_cfg, sharpe_cfg = None, None, None
    rolling_cfg = None
    for line in spec.splitlines():
        m = re.match(r"- ann_vol\[(\w+)\]: return=(\w+); periods_per_year=(\d+); ddof=(\d)", line)
        if m:
            ticker, kind, periods, ddof = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            rets = _returns(series[ticker][1], kind)
            answer[f"ann_vol_{ticker}"] = round(_std(rets, ddof) * math.sqrt(periods), 6)
        m = re.match(r"- rolling_beta: asset=(\w+); benchmark=(\w+); window=(\d+); step=(\d+)", line)
        if m:
            rolling_cfg = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
        m = re.match(r"- weekly_tracking_error: asset=(\w+); benchmark=(\w+); .*return=(\w+); periods_per_year=(\d+); ddof=(\d)", line)
        if m:
            weekly_cfg = (m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)))
        m = re.match(r"- ewma_vol: asset=(\w+); return=(\w+); lambda=([\d.]+); periods_per_year=(\d+)", line)
        if m:
            ewma_cfg = (m.group(1), m.group(2), float(m.group(3)), int(m.group(4)))
        m = re.match(r"- sharpe: asset=(\w+); return=(\w+); annual_rf=([\d.]+); periods_per_year=(\d+); ddof=(\d)", line)
        if m:
            sharpe_cfg = (m.group(1), m.group(2), float(m.group(3)), int(m.group(4)), int(m.group(5)))

    asset, bench, window, step = rolling_cfg
    ra = _returns(series[asset][1], "simple")
    rb = _returns(series[bench][1], "simple")

    def beta(xa, xb):
        mx, my = _mean(xa), _mean(xb)
        cov = sum((x - mx) * (y - my) for x, y in zip(xa, xb, strict=True)) / (len(xa) - 1)
        var = sum((y - my) ** 2 for y in xb) / (len(xb) - 1)
        return cov / var

    best, best_end = None, None
    for start in range(0, len(ra) - window + 1, step):
        value = beta(ra[start:start + window], rb[start:start + window])
        if best is None or value > best:
            best, best_end = value, start + window - 1
    answer["rolling_max_beta"] = round(best, 6)
    answer["rolling_max_beta_end_date"] = series[asset][0][best_end + 1]

    # Second rolling section, downside volatility, log correlation and
    # weekly annualised log return (memo convention table).
    m2 = re.search(r"- rolling_beta2: asset=(\w+); benchmark=(\w+); window=(\d+); step=(\d+)", spec)
    asset2, bench2, win2, step2 = m2.group(1), m2.group(2), int(m2.group(3)), int(m2.group(4))
    ra2 = _returns(series[asset2][1], "simple")
    rb2 = _returns(series[bench2][1], "simple")
    best2, best2_end = None, None
    for start in range(0, len(ra2) - win2 + 1, step2):
        value = beta(ra2[start:start + win2], rb2[start:start + win2])
        if best2 is None or value > best2:
            best2, best2_end = value, start + win2 - 1
    answer["rolling2_max_beta"] = round(best2, 6)
    answer["rolling2_max_beta_end_date"] = series[asset2][0][best2_end + 1]

    # Third rolling section, against the benchmark named in its own
    # convention-table row (BMK2, not the first benchmark).
    m3 = re.search(r"- rolling_beta3: asset=(\w+); benchmark=(\w+); window=(\d+); step=(\d+)", spec)
    asset3, bench3, win3, step3 = m3.group(1), m3.group(2), int(m3.group(3)), int(m3.group(4))
    ra3 = _returns(series[asset3][1], "simple")
    rb3 = _returns(series[bench3][1], "simple")
    best3, best3_end = None, None
    for start in range(0, len(ra3) - win3 + 1, step3):
        value = beta(ra3[start:start + win3], rb3[start:start + win3])
        if best3 is None or value > best3:
            best3, best3_end = value, start + win3 - 1
    answer["rolling3_max_beta"] = round(best3, 6)
    answer["rolling3_max_beta_end_date"] = series[asset3][0][best3_end + 1]

    # Recent-window volatility: only the last window_returns returns of
    # the calendar-filtered series, under this row's own convention.
    mr = re.search(r"- recent_vol: asset=(\w+); window_returns=(\d+); return=(\w+); "
                   r"periods_per_year=(\d+); ddof=(\d+)", spec)
    recent = _returns(series[mr.group(1)][1], mr.group(3))[-int(mr.group(2)):]
    answer[f"recent_vol_last_{mr.group(2)}"] = round(
        _std(recent, int(mr.group(5))) * math.sqrt(int(mr.group(4))), 6)

    md = re.search(r"- downside_vol: asset=(\w+); return=(\w+); periods_per_year=(\d+)", spec)
    d_rets = _returns(series[md.group(1)][1], md.group(2))
    answer["downside_volatility"] = round(
        math.sqrt(_mean([min(r, 0.0) ** 2 for r in d_rets])) * math.sqrt(int(md.group(3))), 6)

    mc = re.search(r"- log_correlation: asset=(\w+); benchmark=(\w+)", spec)
    la = _returns(series[mc.group(1)][1], "log")
    lb = _returns(series[mc.group(2)][1], "log")
    mla, mlb = _mean(la), _mean(lb)
    cov = sum((x - mla) * (y - mlb) for x, y in zip(la, lb, strict=True)) / (len(la) - 1)
    answer["log_return_correlation"] = round(
        cov / (_std(la, 1) * _std(lb, 1)), 6)

    asset, bench, kind, periods, ddof = weekly_cfg

    def weekly(ticker):
        by_week = {}
        for ds, close in zip(*series[ticker], strict=True):
            day = datetime.strptime(ds, "%Y-%m-%d")
            friday = day + timedelta(days=(4 - day.weekday()) % 7)
            by_week[friday] = close
        return [by_week[k] for k in sorted(by_week)]

    diff = [a - b for a, b in zip(_returns(weekly(asset), kind),
                                  _returns(weekly(bench), kind), strict=True)]
    answer["weekly_tracking_error"] = round(_std(diff, ddof) * math.sqrt(periods), 6)

    mw = re.search(r"- weekly_annualised_log_return: asset=(\w+);", spec)
    answer["weekly_annualised_log_return"] = round(
        _mean(_returns(weekly(mw.group(1)), "log")) * 52, 6)

    asset, kind, lam, periods = ewma_cfg
    rets = _returns(series[asset][1], kind)
    var = rets[0] ** 2
    for r in rets[1:]:
        var = lam * var + (1 - lam) * r ** 2
    answer["ewma_vol_last"] = round(math.sqrt(var * periods), 6)

    asset, kind, rf_annual, periods, ddof = sharpe_cfg
    rf_daily = (1 + rf_annual) ** (1 / periods) - 1
    excess = [r - rf_daily for r in _returns(series[asset][1], kind)]
    answer["sharpe_ratio"] = round(_mean(excess) / _std(excess, ddof) * math.sqrt(periods), 6)

    any_ticker = tickers[0]
    answer["num_aligned_days"] = len(series[any_ticker][0])
    return answer


# --------------------------------------------------------------------------
# T2 - multi-file FX + calendar join
# --------------------------------------------------------------------------

def _solve_t2(task_dir: Path) -> dict:
    spec = _spec(task_dir)
    cutoff = int(re.search(r"fx_cutoff_utc_hour: (\d+)", spec).group(1))
    probe = re.search(r"nav_dates: (.+)", spec).group(1).split(", ")
    detail_date = re.search(r"fx_detail_date: (\S+)", spec).group(1)
    holdings = {k: float(v) for k, v in json.loads(
        re.search(r"^holdings: (\{.*\})$", spec, re.MULTILINE).group(1)).items()}
    us = _price_map(task_dir, "us_prices.csv")
    eu = _price_map(task_dir, "eu_prices.csv")
    us_dates = set(next(iter(us.values()))[0])
    eu_dates = set(next(iter(eu.values()))[0])
    union_dates = sorted(us_dates | eu_dates)
    inner_dates = sorted(us_dates & eu_dates)

    fx_rows = _read_csv(task_dir, "fx.csv")

    def to_utc(stamp):
        # A fix's trade date is its UTC date: a 00:30 +01:00 stamp is
        # 23:30 UTC on the previous day.  Weekend prints land on UTC
        # dates outside the union calendar and are never looked up.
        instant = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return instant.astimezone(UTC)

    # Memo rule: a date uses its own fix only when the fix instant is at
    # or before the cut-off; otherwise walk back to the most recent
    # ON-TIME fix (a late fix is never used).  Row order is meaningless.
    effective, prev_on_time = {}, None
    fx_by_utc_date = {}
    for row in fx_rows:
        fx_by_utc_date[to_utc(row["timestamp"]).strftime("%Y-%m-%d")] = row
    for ds in union_dates:
        row = fx_by_utc_date[ds]
        rate = float(row["rate"])
        eff = rate if row["quote"] == "EURUSD" else 1 / rate
        instant = to_utc(row["timestamp"])
        on_time = (instant.hour * 60 + instant.minute) <= cutoff * 60
        effective[ds] = eff if on_time else prev_on_time
        if on_time:
            prev_on_time = eff

    def nav(eff_map, ds):
        total = 0.0
        for ticker, (_dates, closes) in us.items():
            total += holdings[ticker] * dict(zip(_dates, closes, strict=True))[ds]
        for ticker, (_dates, closes) in eu.items():
            total += holdings[ticker] * dict(zip(_dates, closes, strict=True))[ds] * eff_map[ds]
        return total

    navs = {ds: nav(effective, ds) for ds in inner_dates}
    const = dict.fromkeys(effective, effective[inner_dates[0]])
    navs_const = {ds: nav(const, ds) for ds in inner_dates}
    total = navs[inner_dates[-1]] / navs[inner_dates[0]] - 1
    max_date = max(inner_dates, key=lambda d: navs[d])
    dd, _trough = _drawdown([navs[d] for d in inner_dates])
    eu_last = sum(
        holdings[t] * dict(zip(*eu[t], strict=True))[probe[-1]] * effective[probe[-1]]
        for t in eu)
    answer = {f"nav_{d.replace('-', '')}": round(navs[d], 6) for d in probe}
    answer.update({
        "total_return_usd": round(total, 6),
        "fx_pnl_contribution": round(
            total - (navs_const[inner_dates[-1]] / navs_const[inner_dates[0]] - 1), 6),
        "dates_dropped_by_alignment": len(union_dates) - len(inner_dates),
        "effective_fx_rate_mid_probe": round(effective[probe[1]], 6),
        f"effective_fx_rate_{detail_date.replace('-', '')}": round(
            effective[detail_date], 6),
        f"eu_sleeve_usd_{probe[-1].replace('-', '')}": round(eu_last, 6),
        "max_nav_value": round(navs[max_date], 6),
        "max_nav_date": max_date,
        "nav_max_drawdown": round(dd, 6),
        "num_inner_days": len(inner_dates),
        "problems": _t2_problems(task_dir, fx_rows, us_dates, eu_dates),
    })
    return answer


def _t2_problems(task_dir, fx_rows, us_dates, eu_dates):
    problems = []
    if any(r["quote"] != "EURUSD" for r in fx_rows):
        problems.append("currency")
    forms = {"Z" if r["timestamp"].endswith("Z") else r["timestamp"][-6:] for r in fx_rows}
    if len(forms) > 1:
        problems.append("timezone")
    if us_dates != eu_dates:
        problems.append("calendar")
    return sorted(problems)


# --------------------------------------------------------------------------
# T3 - conflicting-duplicate reconciliation
# --------------------------------------------------------------------------

def _solve_t3(task_dir: Path) -> dict:
    spec = _spec(task_dir)
    report_time = re.search(r"report_time: (\S+)", spec).group(1)
    probe_ticker = re.search(r"probe_ticker: (\S+)", spec).group(1)
    probe_dates = re.search(r"probe_dates: (.+)", spec).group(1).split(", ")

    def first_wins(rows, value_fn):
        out, status = {}, {}
        for row in rows:
            key = (row["ticker"], row["date"])
            if key in out:
                continue
            out[key] = value_fn(row)
            status[key] = row.get("status")
        return out, status

    a_val, a_status = first_wins(_read_csv(task_dir, "vendor_a.csv"),
                                 lambda r: float(r["close"]))
    b_val, _ = first_wins(
        _read_csv(task_dir, "vendor_b.csv"),
        lambda r: float(r["close"]) / 100 if r["unit"] == "cents" else float(r["close"]))
    # Corrections: compare as_of instants in UTC (mixed offset forms),
    # keep only those at or before report_time, latest instant wins and
    # supersedes earlier corrections for the same key.
    from datetime import UTC as _UTC

    def _instant(stamp):
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(_UTC)

    report_instant = _instant(report_time)
    corr = {}
    for row in _read_csv(task_dir, "corrections.csv"):
        key = (row["ticker"], row["date"])
        when = _instant(row["as_of"])
        if when > report_instant:
            continue
        if key not in corr or when > corr[key][1]:
            corr[key] = (float(row["close"]), when)

    resolved, paths = {}, {"correction": 0, "vendor_a": 0, "median": 0}
    for key in sorted(set(a_val) | set(b_val)):
        av, bv = a_val.get(key), b_val.get(key)
        applicable = key in corr  # already filtered to <= report_time
        conflicted = applicable or (av is not None and bv is not None and abs(av - bv) > 0.01)
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
            resolved[key] = avail[len(avail) // 2] if len(avail) % 2 else (avail[0] + avail[-1]) / 2
            if conflicted:
                paths["median"] += 1

    tickers = sorted({t for t, _ in resolved})
    by = {t: {d: v for (tt, d), v in resolved.items() if tt == t} for t in tickers}
    common = sorted(set.intersection(*[set(by[t]) for t in tickers]))
    path = [_mean([by[t][d] / by[t][common[0]] for t in tickers]) for d in common]
    dd, trough = _drawdown(path)
    answer = {
        "resolved_by_correction": paths["correction"],
        "resolved_by_vendor_a": paths["vendor_a"],
        "resolved_by_median": paths["median"],
        "reconciled_total_return": round(path[-1] - 1, 6),
        "max_drawdown": round(dd, 6),
        "max_drawdown_trough_date": common[trough],
        "num_resolved_days": len(common),
        "problems": _t3_problems(task_dir, resolved, a_val, b_val),
    }
    for d in probe_dates:
        answer[f"resolved_close_{probe_ticker}_{d.replace('-', '')}"] = \
            round(resolved[(probe_ticker, d)], 6)
    return answer


def _t3_problems(task_dir, resolved, a_val, b_val):
    problems = []
    if any(k in b_val and abs(v - b_val[k]) > 0.01 or
           (k in a_val and k in b_val and abs(a_val[k] - b_val[k]) > 0.01)
           for k, v in resolved.items()):
        problems.append("source_conflict")
    if len({r["unit"] for r in _read_csv(task_dir, "vendor_b.csv")}) > 1:
        problems.append("unit_change")
    for name in ("vendor_a.csv", "vendor_b.csv"):
        keys = [(r["ticker"], r["date"]) for r in _read_csv(task_dir, name)]
        if len(keys) != len(set(keys)):
            problems.append("duplicate")
            break
    # Memo missing rule: a weekday inside a ticker's union date range
    # reported by neither vendor.  One-vendor dates are not missing.
    present_by = {}
    for _ticker, ds in resolved:
        present_by.setdefault(_ticker, set()).add(ds)
    for _ticker, present in present_by.items():
        day = datetime.strptime(min(present), "%Y-%m-%d")
        last = datetime.strptime(max(present), "%Y-%m-%d")
        while day <= last:
            if day.weekday() < 5 and day.strftime("%Y-%m-%d") not in present:
                problems.append("missing")
                break
            day += timedelta(days=1)
    return sorted(set(problems))


# --------------------------------------------------------------------------
# T4 - split-table corporate actions on holdings
# --------------------------------------------------------------------------

def _solve_t4(task_dir: Path) -> dict:
    prices = _price_map(task_dir)
    assets = sorted(prices)
    dates = prices[assets[0]][0]
    price_at = {t: dict(zip(*prices[t], strict=True)) for t in assets}
    events = {}
    problems = []
    for row in _read_csv(task_dir, "splits.csv"):
        a, b = (float(x) for x in row["ratio"].split("-for-"))
        # Memo: a 1-for-1 row is a no-op - ignore it entirely (not an
        # event, not counted in num_splits, not a problem).
        if a == b:
            continue
        events.setdefault(row["date"], []).append((row["ticker"], a, b))
        problems.append("split" if a > b else "reverse_split")
    trades_by_date: dict[str, list] = {}
    n_trades = 0
    for row in _read_csv(task_dir, "trades.csv"):
        trades_by_date.setdefault(row["date"], []).append(
            (row["ticker"], int(row["shares"]), float(row["price"])))
        n_trades += 1

    lots = {t: [] for t in assets}
    cash_lieu, realised = 0.0, 0.0
    values, flows = [], []
    for ds in dates:
        for ticker, a, b in events.get(ds, []):
            ratio = a / b
            for lot in lots[ticker]:
                if lot[2] >= ds:  # settlement must be before the ex-date
                    continue
                new_shares = lot[0] * ratio
                whole = math.floor(new_shares + 1e-9)
                cash_lieu += (new_shares - whole) * price_at[ticker][ds]
                lot[0], lot[1] = float(whole), lot[1] / ratio
            lots[ticker] = [lot for lot in lots[ticker] if lot[0] > 0]
        flow = 0.0
        for ticker, shares, px in trades_by_date.get(ds, []):
            flow += shares * px
            if shares > 0:
                settle_idx = min(dates.index(ds) + 2, len(dates) - 1)
                lots[ticker].append([float(shares), px, dates[settle_idx]])
            else:
                to_sell = -shares
                for lot in lots[ticker]:
                    if to_sell <= 0:
                        break
                    take = min(lot[0], to_sell)
                    realised += take * (px - lot[1])
                    lot[0] -= take
                    to_sell -= take
                lots[ticker] = [lot for lot in lots[ticker] if lot[0] > 0]
        flows.append(flow)
        values.append(cash_lieu + sum(
            sum(lot[0] for lot in lots[t]) * price_at[t][ds] for t in assets))

    index, prev, started = [], None, False
    for v, f in zip(values, flows, strict=True):
        if not started:
            if v > 0:
                index.append(1.0)
                prev, started = v, True
            continue
        index.append(index[-1] * (1 + ((v - f) / prev - 1)))
        if v > 0:
            prev = v
    dd, _ = _drawdown(index)
    end_shares = {t: int(round(sum(lot[0] for lot in lots[t]))) for t in assets}
    remaining_cost = sum(lot[0] * lot[1] for t in assets for lot in lots[t])
    end_mv = sum(end_shares[t] * price_at[t][dates[-1]] for t in assets)
    answer = {
        **{f"shares_{t}": end_shares[t] for t in assets},
        "realised_pnl_fifo": round(realised, 6),
        "unrealised_pnl": round(end_mv - remaining_cost, 6),
        "cash_in_lieu": round(cash_lieu, 6),
        "twr_total_return": round(index[-1] - 1, 6),
        "max_drawdown": round(dd, 6),
        "num_trades": n_trades,
        "num_splits": sum(len(v) for v in events.values()),
        "problems": sorted(set(problems)),
    }
    return answer


# --------------------------------------------------------------------------
# T5 - constrained portfolio construction pipeline
# --------------------------------------------------------------------------

def _solve_t5(task_dir: Path) -> dict:
    import numpy as np

    cfg = json.loads((task_dir / "constraints.json").read_text())
    prices = _price_map(task_dir)
    assets = sorted(prices)
    dates = prices[assets[0]][0]
    closes = {t: prices[t][1] for t in assets}
    rebalance_idx = [i - 1 for i in cfg["rebalance_row_indices_1based"]]
    lookback, delta = cfg["lookback_returns"], cfg["shrinkage_delta"]
    lot, buffer_pct = cfg["lot_size"], cfg["cash_buffer_pct"]

    sector_members: dict[str, list[int]] = {}
    for i, t in enumerate(assets):
        sector_members.setdefault(cfg["sectors"][t], []).append(i)

    def weights_at(end_idx):
        rets = np.array([_returns(closes[t], "simple")[end_idx - lookback:end_idx]
                         for t in assets]).T
        cov = np.cov(rets, rowvar=False, ddof=1)
        cov = (1 - delta) * cov + delta * np.diag(np.diag(cov))
        n, cap = len(assets), cfg["per_asset_cap"]
        # Active-set solve for min w' Sigma w subject to sum(w)=1,
        # 0<=w<=per-asset cap and every sector cap in constraints.json:
        # sector sums enter as added equalities whenever violated, and
        # fixed assets / active sectors are released on wrong-sign KKT
        # multipliers.
        active_sectors: list[str] = []
        w = np.full(n, 1.0 / n)
        for _outer in range(20):
            fixed: dict[int, float] = {}
            lambdas = np.zeros(1 + len(active_sectors))
            for _inner in range(100):
                equalities = [(list(range(n)), 1.0)] + [
                    (sector_members[s], cfg["sector_caps"][s]) for s in active_sectors]
                free = [i for i in range(n) if i not in fixed]
                w = np.zeros(n)
                for i, v in fixed.items():
                    w[i] = v
                if free:
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
                    lambdas = sol[len(free):]
                below = [(w[i], i) for i in range(n)
                         if i not in fixed and w[i] < -1e-10]
                above = [(w[i], i) for i in range(n)
                         if i not in fixed and w[i] > cap + 1e-10]
                if below:
                    fixed[min(below)[1]] = 0.0
                    continue
                if above:
                    fixed[max(above)[1]] = cap
                    continue
                grad = 2 * cov @ w
                at_lam = np.zeros(n)
                for k, (members, _t) in enumerate(equalities):
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
            sector_sum = {s: float(sum(w[i] for i in mem))
                          for s, mem in sector_members.items()}
            violated = [s for s in sector_sum
                        if sector_sum[s] > cfg["sector_caps"][s] + 1e-9
                        and s not in active_sectors]
            if violated:
                active_sectors.append(
                    max(violated, key=lambda s: sector_sum[s] - cfg["sector_caps"][s]))
                continue
            release_sector, worst = None, -1e-9
            for k, s in enumerate(active_sectors):
                if lambdas[1 + k] < worst:
                    release_sector, worst = s, lambdas[1 + k]
            if release_sector is None:
                return w
            active_sectors.remove(release_sector)
        return w

    def allocate(desired, nav, px):
        # Floor to lots, then greedy: largest positive residual first
        # (ties by ticker), one lot at a time while cash covers it.
        # Remaining cash includes the buffer, so this may spend into it.
        shares = [int(math.floor(d / (p * lot)) * lot) for d, p in zip(desired, px, strict=True)]
        cash = nav - sum(s * p for s, p in zip(shares, px, strict=True))
        while True:
            residuals = [d - s * p for d, s, p in zip(desired, shares, px, strict=True)]
            for i in sorted(range(len(assets)), key=lambda i: (-residuals[i], assets[i])):
                if residuals[i] > 1e-9 and cash >= lot * px[i] - 1e-9:
                    shares[i] += lot
                    cash -= lot * px[i]
                    break
            else:
                return shares, cash

    shares, cash = [0] * len(assets), float(cfg["capital"])
    weights_first = shares_first = None
    turnovers, nav_path = {}, []
    for idx, _ds in enumerate(dates):
        px = [closes[t][idx] for t in assets]
        if idx in rebalance_idx:
            nav_pre = cash + sum(s * p for s, p in zip(shares, px, strict=True))
            w = weights_at(idx)
            desired = [float(wi) * nav_pre * (1 - buffer_pct) for wi in w]
            new_shares, new_cash = allocate(desired, nav_pre, px)
            traded = sum(abs(ns - s) * p for ns, s, p in zip(new_shares, shares, px, strict=True))
            # Cap is checked on the actual post-rounding turnover
            # (later rebalances only): scale share changes toward zero
            # by cap/actual, truncated to whole lots; no re-allocation.
            if any(shares) and traded / nav_pre > cfg["turnover_cap"]:
                factor = cfg["turnover_cap"] / (traded / nav_pre)
                new_shares = [
                    s + (1 if ns > s else -1)
                    * int(math.floor(abs(ns - s) * factor / lot)) * lot
                    for ns, s in zip(new_shares, shares, strict=True)]
                new_cash = nav_pre - sum(
                    s * p for s, p in zip(new_shares, px, strict=True))
                traded = sum(abs(ns - s) * p for ns, s, p in zip(new_shares, shares, px, strict=True))
            turnovers[str(len(turnovers) + 1)] = round(traded / nav_pre, 6)
            # Cost last: if cash after cost would be negative, return
            # lots (largest allocated-minus-desired excess, ties by
            # ticker) and recompute, until cash after cost is >= 0.
            cost = traded * cfg["cost_bps"] / 10000
            while new_cash - cost < 0 and any(s > 0 for s in new_shares):
                excess = [ns * p - d for ns, d, p in zip(new_shares, desired, px, strict=True)]
                pick = min(range(len(assets)), key=lambda i: (-excess[i], assets[i]))
                if new_shares[pick] <= 0:
                    pick = max(range(len(assets)), key=lambda i: new_shares[i])
                new_shares[pick] -= lot
                new_cash += lot * px[pick]
                traded = sum(abs(ns - s) * p for ns, s, p in zip(new_shares, shares, px, strict=True))
                cost = traded * cfg["cost_bps"] / 10000
                turnovers[str(len(turnovers))] = round(traded / nav_pre, 6)
            new_cash -= cost
            if weights_first is None:
                weights_first = {t: round(float(wi), 6) for t, wi in zip(assets, w, strict=True)}
                shares_first = {t: int(s) for t, s in zip(assets, new_shares, strict=True)}
            shares, cash = new_shares, new_cash
        nav_path.append(cash + sum(s * p for s, p in zip(shares, px, strict=True)))
    dd, _ = _drawdown(nav_path)
    return {
        "weights_first_rebalance": weights_first,
        "shares_first_rebalance": shares_first,
        "turnover_by_rebalance": turnovers,
        "num_rebalances": len(rebalance_idx),
        "final_nav": round(nav_path[-1], 6),
        "max_drawdown": round(dd, 6),
        "cash_left": round(cash, 6),
    }


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

_SOLVERS = {
    "tools_frontier_convention_risk": _solve_t1,
    "tools_frontier_fx_calendar": _solve_t2,
    "tools_frontier_reconciliation": _solve_t3,
    "tools_frontier_split_holdings": _solve_t4,
    "tools_frontier_portfolio_construction": _solve_t5,
}


def _family_of(task_dir: Path) -> str:
    spec = _spec(task_dir)
    m = re.search(r"\(T(\d)\)", spec)
    return {1: "tools_frontier_convention_risk", 2: "tools_frontier_fx_calendar",
            3: "tools_frontier_reconciliation", 4: "tools_frontier_split_holdings",
            5: "tools_frontier_portfolio_construction"}[int(m.group(1))]


def solve_task_dir(task_dir) -> dict:
    """[solve_task_dir] Solve one task directory from its files only."""
    task_dir = Path(task_dir)
    return _SOLVERS[_family_of(task_dir)](task_dir)


def solve_task(task: dict, tasks_dir) -> dict:
    """[solve_task] Solve a tasks.json entry located under ``tasks_dir``."""
    solver = _SOLVERS[task["type"]]
    return solver(Path(tasks_dir) / task["data_dir"])
