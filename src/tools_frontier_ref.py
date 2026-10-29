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
from datetime import datetime, timedelta
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
    answer["num_return_days"] = answer["num_aligned_days"] - 1
    return answer


# --------------------------------------------------------------------------
# T2 - multi-file FX + calendar join
# --------------------------------------------------------------------------

def _solve_t2(task_dir: Path) -> dict:
    spec = _spec(task_dir)
    cutoff = int(re.search(r"fx_cutoff_utc_hour: (\d+)", spec).group(1))
    probe = re.search(r"nav_dates: (.+)", spec).group(1).split(", ")
    holdings = {k: float(v) for k, v in json.loads(
        re.search(r"^holdings: (\{.*\})$", spec, re.MULTILINE).group(1)).items()}
    us = _price_map(task_dir, "us_prices.csv")
    eu = _price_map(task_dir, "eu_prices.csv")
    us_dates = set(next(iter(us.values()))[0])
    eu_dates = set(next(iter(eu.values()))[0])
    union_dates = sorted(us_dates | eu_dates)
    inner_dates = sorted(us_dates & eu_dates)

    fx_rows = _read_csv(task_dir, "fx.csv")

    def to_utc_hour(stamp):
        # Forms seen in the file: ...Z, ...+00:00, ...+01:00.
        if stamp.endswith("Z"):
            return int(stamp[11:13])
        offset = int(stamp[-5:-3]) * (1 if stamp[-6] == "+" else -1)
        return int(stamp[11:13]) - offset

    effective, prev = {}, None
    fx_by_date = {row["timestamp"][:10]: row for row in fx_rows}
    for ds in union_dates:
        row = fx_by_date[ds]
        rate = float(row["rate"])
        eff = rate if row["quote"] == "EURUSD" else 1 / rate
        effective[ds] = prev if (to_utc_hour(row["timestamp"]) > cutoff and prev is not None) else eff
        prev = eff

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
    answer = {f"nav_{d.replace('-', '')}": round(navs[d], 6) for d in probe}
    answer.update({
        "total_return_usd": round(total, 6),
        "fx_pnl_contribution": round(
            total - (navs_const[inner_dates[-1]] / navs_const[inner_dates[0]] - 1), 6),
        "dates_dropped_by_alignment": len(union_dates) - len(inner_dates),
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
    corr = {}
    for row in _read_csv(task_dir, "corrections.csv"):
        key = (row["ticker"], row["date"])
        if key not in corr or row["as_of"] > corr[key][1]:
            corr[key] = (float(row["close"]), row["as_of"])

    resolved, paths = {}, {"correction": 0, "vendor_a": 0, "median": 0}
    for key in sorted(set(a_val) | set(b_val)):
        av, bv = a_val.get(key), b_val.get(key)
        applicable = key in corr and corr[key][1] <= report_time
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

    def weights_at(end_idx):
        rets = np.array([_returns(closes[t], "simple")[end_idx - lookback:end_idx]
                         for t in assets]).T
        cov = np.cov(rets, rowvar=False, ddof=1)
        cov = (1 - delta) * cov + delta * np.diag(np.diag(cov))
        n, cap = len(assets), cfg["per_asset_cap"]
        # Active-set solve for min w' Sigma w, sum(w)=1, 0<=w<=cap:
        # fix the most violated bound, re-solve the equality QP for the
        # free assets, release fixed assets with wrong-sign multipliers.
        fixed: dict[int, float] = {}
        w = np.full(n, 1.0 / n)
        for _ in range(100):
            free = [i for i in range(n) if i not in fixed]
            w = np.zeros(n)
            for i, v in fixed.items():
                w[i] = v
            if free:
                kkt = np.zeros((len(free) + 1, len(free) + 1))
                kkt[:len(free), :len(free)] = 2 * cov[np.ix_(free, free)]
                kkt[:len(free), -1] = 1
                kkt[-1, :len(free)] = 1
                sol = np.linalg.solve(kkt, [0.0] * len(free) + [1 - sum(fixed.values())])
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
                break
            del fixed[release]
        return w

    def allocate(desired, nav, px):
        shares = [int(math.floor(d / (p * lot)) * lot) for d, p in zip(desired, px, strict=True)]
        cash = nav - sum(s * p for s, p in zip(shares, px, strict=True))
        while True:
            residuals = [d - s * p for d, s, p in zip(desired, shares, px, strict=True)]
            for i in sorted(range(len(assets)), key=lambda i: (-residuals[i], assets[i])):
                if cash >= lot * px[i] - 1e-9:
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
            if any(shares):
                raw = sum(abs(d - s * p) for d, s, p in zip(desired, shares, px, strict=True)) / nav_pre
                if raw > cfg["turnover_cap"]:
                    scale = cfg["turnover_cap"] / raw
                    desired = [s * p + (d - s * p) * scale
                               for d, s, p in zip(desired, shares, px, strict=True)]
            new_shares, new_cash = allocate(desired, nav_pre, px)
            traded = sum(abs(ns - s) * p for ns, s, p in zip(new_shares, shares, px, strict=True))
            turnovers[str(len(turnovers) + 1)] = round(traded / nav_pre, 6)
            new_cash -= traded * cfg["cost_bps"] / 10000
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
