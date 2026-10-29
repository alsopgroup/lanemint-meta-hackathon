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

    # Composite equal-weight portfolio: daily return is the mean of the
    # member assets' simple returns; compound an index from 1.0.
    members = re.search(r"- portfolio: assets=([\w,]+);", spec).group(1).split(",")
    member_rets = {t: _returns(series[t][1], "simple") for t in members}
    prets = [_mean([member_rets[t][i] for t in members])
             for i in range(len(member_rets[members[0]]))]
    pindex = [1.0]
    for r in prets:
        pindex.append(pindex[-1] * (1 + r))
    pdd, ptrough = _drawdown(pindex)
    pexc = [r - rf_daily for r in prets]
    answer["portfolio_ann_vol"] = round(_std(prets, 1) * math.sqrt(periods), 6)
    answer["portfolio_max_drawdown"] = round(pdd, 6)
    answer["portfolio_max_drawdown_trough_date"] = series[members[0]][0][ptrough]
    answer["portfolio_sharpe"] = round(
        _mean(pexc) / _std(pexc, 1) * math.sqrt(periods), 6)

    # Rolling correlation section (log returns, memo window/step).
    mrc = re.search(r"- rolling_correlation: asset=(\w+); benchmark=(\w+); "
                    r"window=(\d+); step=(\d+); return=(\w+)", spec)
    ca = _returns(series[mrc.group(1)][1], mrc.group(5))
    cb = _returns(series[mrc.group(2)][1], mrc.group(5))
    win, stp = int(mrc.group(3)), int(mrc.group(4))
    best_c, best_c_end = None, None
    for start in range(0, len(ca) - win + 1, stp):
        xa, xb = ca[start:start + win], cb[start:start + win]
        mx, my = _mean(xa), _mean(xb)
        cov = sum((x - mx) * (y - my) for x, y in zip(xa, xb, strict=True)) / (win - 1)
        value = cov / (_std(xa, 1) * _std(xb, 1))
        if best_c is None or value > best_c:
            best_c, best_c_end = value, start + win - 1
    answer["rolling_max_correlation"] = round(best_c, 6)
    answer["rolling_max_correlation_end_date"] = series[mrc.group(1)][0][best_c_end + 1]

    # Worst single day of the named asset.
    mwd = re.search(r"- worst_day: asset=(\w+); return=(\w+)", spec)
    wrets = _returns(series[mwd.group(1)][1], mwd.group(2))
    wi = min(range(len(wrets)), key=lambda i: wrets[i])
    answer["worst_day_return"] = round(wrets[wi], 6)
    answer["worst_day_date"] = series[mwd.group(1)][0][wi + 1]

    # Bounded minimum-tracking-error hedge (memo section 15): the
    # unconstrained minimiser is cov/var over the last window_returns
    # returns; the reported ratio clips it into the stated bounds.
    mh = re.search(r"- hedge: asset=(\w+); benchmark=(\w+); window_returns=(\d+); "
                   r"return=(\w+); lower_bound=([\d.]+); upper_bound=([\d.]+); "
                   r"periods_per_year=(\d+); ddof=(\d+)", spec)
    ha = _returns(series[mh.group(1)][1], mh.group(4))[-int(mh.group(3)):]
    hb = _returns(series[mh.group(2)][1], mh.group(4))[-int(mh.group(3)):]
    mha, mhb = _mean(ha), _mean(hb)
    cov_h = sum((x - mha) * (y - mhb)
                for x, y in zip(ha, hb, strict=True)) / (len(ha) - 1)
    var_h = sum((y - mhb) ** 2 for y in hb) / (len(hb) - 1)
    h = min(max(cov_h / var_h, float(mh.group(5))), float(mh.group(6)))
    answer["hedge_ratio"] = round(h, 6)
    hedged = [a - h * b for a, b in zip(ha, hb, strict=True)]
    answer["hedged_ann_vol"] = round(
        _std(hedged, int(mh.group(8))) * math.sqrt(int(mh.group(7))), 6)

    # Stateful hedged-strategy simulation (memo section 16): rolling
    # re-estimation of the bounded hedge ratio every rebalance_step
    # indices, short resized in whole hedge lots, costs charged to
    # cash, NAV carried daily over asset + short + cash.
    ms = re.search(
        r"- hedge_strategy: asset=(\w+); benchmark=(\w+); capital=(\d+); "
        r"window_returns=(\d+); first_rebalance_index=(\d+); "
        r"rebalance_step=(\d+); hedge_lot=(\d+); cost_bps=([\d.]+); "
        r"lower_bound=([\d.]+); upper_bound=([\d.]+)", spec)
    s_asset, s_bench = ms.group(1), ms.group(2)
    capital = float(ms.group(3))
    s_window, s_first = int(ms.group(4)), int(ms.group(5))
    s_step, s_lot = int(ms.group(6)), int(ms.group(7))
    s_cost = float(ms.group(8))
    s_lo, s_hi = float(ms.group(9)), float(ms.group(10))
    closes_a, closes_b = series[s_asset][1], series[s_bench][1]
    log_a = _returns(closes_a, "log")
    log_b = _returns(closes_b, "log")

    def _cov_var(xa, xb):
        mx, my = _mean(xa), _mean(xb)
        cov = sum((x - mx) * (y - my) for x, y in zip(xa, xb, strict=True)) \
            / (len(xa) - 1)
        var = sum((y - my) ** 2 for y in xb) / (len(xb) - 1)
        return cov / var

    asset_shares = int(capital // closes_a[0])
    cash = capital - asset_shares * closes_a[0]
    short, total_cost, ratios, navs = 0.0, 0.0, [], []
    for idx in range(len(closes_a)):
        if idx >= s_first and (idx - s_first) % s_step == 0:
            ratio = _cov_var(log_a[idx - s_window:idx],
                             log_b[idx - s_window:idx])
            ratio = min(max(ratio, s_lo), s_hi)
            ratios.append(ratio)
            desired = float(int(
                ratio * asset_shares * closes_a[idx] / closes_b[idx]
                // s_lot) * s_lot)
            trade = desired - short
            cost = abs(trade) * closes_b[idx] * s_cost / 10000
            cash += trade * closes_b[idx] - cost
            total_cost += cost
            short = desired
        navs.append(cash + asset_shares * closes_a[idx] - short * closes_b[idx])
    s_dd, _s_tr = _drawdown(navs)
    s_log_rets = [math.log(b / a) for a, b in zip(navs, navs[1:], strict=False)]
    answer.update({
        "hedge_strategy_final_nav": round(navs[-1], 6),
        "hedge_strategy_ann_vol": round(_std(s_log_rets, 1) * math.sqrt(252), 6),
        "hedge_strategy_max_drawdown": round(s_dd, 6),
        "hedge_strategy_total_cost": round(total_cost, 6),
        "num_hedge_rebalances": len(ratios),
        "hedge_strategy_final_short_shares": int(short),
        "hedge_strategy_last_ratio": round(ratios[-1], 6),
        "hedge_strategy_cash_end": round(cash, 6),
    })

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
    num_late = 0
    for ds in union_dates:
        row = fx_by_utc_date[ds]
        rate = float(row["rate"])
        eff = rate if row["quote"] == "EURUSD" else 1 / rate
        instant = to_utc(row["timestamp"])
        on_time = (instant.hour * 60 + instant.minute) <= cutoff * 60
        effective[ds] = eff if on_time else prev_on_time
        if on_time:
            prev_on_time = eff
        else:
            num_late += 1

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
    dd, trough_idx = _drawdown([navs[d] for d in inner_dates])
    eu_last = sum(
        holdings[t] * dict(zip(*eu[t], strict=True))[probe[-1]] * effective[probe[-1]]
        for t in eu)
    us_last = sum(
        holdings[t] * dict(zip(*us[t], strict=True))[probe[-1]] for t in us)
    nav_log_rets = [math.log(navs[b] / navs[a])
                    for a, b in zip(inner_dates, inner_dates[1:], strict=False)]
    # Monthly FX hedge on the EU sleeve (memo rule 8): on the first
    # union date of each month, sell forward hedge_ratio times the EU
    # sleeve's EUR value (most recent closes) at that date's effective
    # rate; settle at the next reset (or the last union date); hold
    # the USD cash P&L uninvested.
    hedge_ratio = float(re.search(
        r"- fx_hedge: .*hedge_ratio=([\d.]+)", spec).group(1))
    eu_maps = {t: dict(zip(*eu[t], strict=True)) for t in eu}
    eu_filled, last_eu = {}, {}
    for ds in union_dates:
        for t in eu:
            if ds in eu_maps[t]:
                last_eu[t] = eu_maps[t][ds]
        eu_filled[ds] = sum(holdings[t] * last_eu.get(t, 0.0) for t in eu)
    resets, seen_months = [], set()
    for ds in union_dates:
        if ds[:7] not in seen_months:
            seen_months.add(ds[:7])
            resets.append(ds)
    settlements = []
    for k, r_ds in enumerate(resets):
        settle = resets[k + 1] if k + 1 < len(resets) else union_dates[-1]
        settlements.append((settle, hedge_ratio * eu_filled[r_ds]
                            * (effective[r_ds] - effective[settle])))
    hedged_series = [navs[ds] + sum(p for d2, p in settlements if d2 <= ds)
                     for ds in inner_dates]
    h_dd, _h_trough = _drawdown(hedged_series)
    h_log = [math.log(b / a)
             for a, b in zip(hedged_series, hedged_series[1:], strict=False)]

    # Monthly rebalanced sleeve strategy (memo rule 9): rolling
    # bounded two-asset minimum-variance sleeve weight from trailing
    # sleeve returns (EU measured in USD at effective rates), traded
    # in lots with minimum trade size and costs against cash.
    mst = re.search(
        r"- strategy: capital=(\d+); window_returns=(\d+); "
        r"eu_weight_bounds=([\d.]+),([\d.]+); cash_buffer_pct=([\d.]+); "
        r"min_trade_shares=(\d+); cost_bps=([\d.]+); lots: (.+)$",
        spec, re.MULTILINE)
    st_capital, st_window = float(mst.group(1)), int(mst.group(2))
    st_lo, st_hi = float(mst.group(3)), float(mst.group(4))
    st_buffer = float(mst.group(5))
    st_min_trade, st_cost_bps = int(mst.group(6)), float(mst.group(7))
    st_lots = {t: int(v) for t, v in re.findall(r"(\w+)=(\d+)", mst.group(8))}
    us_tickers, eu_tickers = sorted(us), sorted(eu)
    st_assets = us_tickers + eu_tickers
    us_map = {t: dict(zip(*us[t], strict=True)) for t in us_tickers}
    eu_map = {t: dict(zip(*eu[t], strict=True)) for t in eu_tickers}

    def st_price(ticker, ds):
        if ticker in us_map:
            return us_map[ticker][ds]
        return eu_map[ticker][ds] * effective[ds]

    sleeve_us, sleeve_eu = [], []
    for k in range(len(inner_dates) - 1):
        d0, d1 = inner_dates[k], inner_dates[k + 1]
        sleeve_us.append(_mean([us_map[t][d1] / us_map[t][d0] - 1
                                for t in us_tickers]))
        sleeve_eu.append(_mean([
            (eu_map[t][d1] * effective[d1]) / (eu_map[t][d0] * effective[d0]) - 1
            for t in eu_tickers]))

    def _var_of(xs):
        m = _mean(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    st_shares = dict.fromkeys(st_assets, 0)
    st_cash, st_total_cost, st_weights = st_capital, 0.0, []
    st_nav_path = []
    st_reb = {i for i, ds in enumerate(inner_dates)
              if i >= st_window
              and (i == 0 or ds[:7] != inner_dates[i - 1][:7])}
    for idx, ds in enumerate(inner_dates):
        if idx in st_reb:
            nav_pre = st_cash + sum(st_shares[t] * st_price(t, ds)
                                    for t in st_assets)
            xa = sleeve_us[idx - st_window:idx]
            xb = sleeve_eu[idx - st_window:idx]
            var_us, var_eu = _var_of(xa), _var_of(xb)
            mx, my = _mean(xa), _mean(xb)
            cov_se = sum((x - mx) * (y - my)
                         for x, y in zip(xa, xb, strict=True)) / (len(xa) - 1)
            denom = var_us + var_eu - 2 * cov_se
            w_unc = (var_us - cov_se) / denom if denom else 0.5
            w_eu = min(max(w_unc, st_lo), st_hi)
            st_weights.append(w_eu)
            desired = {}
            for t in us_tickers:
                desired[t] = (1 - w_eu) / len(us_tickers) * nav_pre * (1 - st_buffer)
            for t in eu_tickers:
                desired[t] = w_eu / len(eu_tickers) * nav_pre * (1 - st_buffer)
            new_shares = {t: int(desired[t] / st_price(t, ds) // st_lots[t])
                          * st_lots[t] for t in st_assets}
            cash_left = nav_pre - sum(new_shares[t] * st_price(t, ds)
                                      for t in st_assets)
            while True:
                residuals = {t: desired[t] - new_shares[t] * st_price(t, ds)
                             for t in st_assets}
                bought = False
                for t in sorted(st_assets, key=lambda t: (-residuals[t], t)):
                    lot_cost = st_lots[t] * st_price(t, ds)
                    if residuals[t] > 1e-9 and cash_left >= lot_cost - 1e-9:
                        new_shares[t] += st_lots[t]
                        cash_left -= lot_cost
                        bought = True
                        break
                if not bought:
                    break
            new_shares = {
                t: st_shares[t]
                if 0 < abs(new_shares[t] - st_shares[t]) < st_min_trade
                else new_shares[t] for t in st_assets}
            traded = sum(abs(new_shares[t] - st_shares[t]) * st_price(t, ds)
                         for t in st_assets)
            cash_left = nav_pre - sum(new_shares[t] * st_price(t, ds)
                                      for t in st_assets)
            cost = traded * st_cost_bps / 10000
            while cash_left - cost < 0 and any(new_shares[t] > 0
                                               for t in st_assets):
                excess = {t: new_shares[t] * st_price(t, ds) - desired[t]
                          for t in st_assets}
                pick = min(st_assets, key=lambda t: (-excess[t], t))
                if new_shares[pick] <= 0:
                    pick = max(st_assets, key=lambda t: new_shares[t])
                new_shares[pick] -= st_lots[pick]
                cash_left += st_lots[pick] * st_price(pick, ds)
                traded = sum(abs(new_shares[t] - st_shares[t]) * st_price(t, ds)
                             for t in st_assets)
                cost = traded * st_cost_bps / 10000
            st_cash = cash_left - cost
            st_total_cost += cost
            st_shares = new_shares
        st_nav_path.append(st_cash + sum(st_shares[t] * st_price(t, ds)
                                         for t in st_assets))
    st_dd, _st_tr = _drawdown(st_nav_path)
    st_log = [math.log(b / a)
              for a, b in zip(st_nav_path, st_nav_path[1:], strict=False)]
    st_last = inner_dates[-1]
    strategy_fields = {
        "strategy_final_nav": round(st_nav_path[-1], 6),
        "strategy_ann_vol": round(_std(st_log, 1) * math.sqrt(252), 6),
        "strategy_max_drawdown": round(st_dd, 6),
        "strategy_total_cost": round(st_total_cost, 6),
        "num_strategy_rebalances": len(st_weights),
        "strategy_final_shares": {t: int(st_shares[t]) for t in st_assets},
        "strategy_last_eu_weight": round(st_weights[-1], 6),
        "strategy_cash_end": round(st_cash, 6),
        "strategy_eu_weight_final": round(
            sum(st_shares[t] * st_price(t, st_last) for t in eu_tickers)
            / st_nav_path[-1], 6),
    }
    union_set = set(union_dates)
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
        f"us_sleeve_usd_{probe[-1].replace('-', '')}": round(us_last, 6),
        "eu_sleeve_weight_last_probe": round(eu_last / navs[probe[-1]], 6),
        "max_nav_value": round(navs[max_date], 6),
        "max_nav_date": max_date,
        "nav_max_drawdown": round(dd, 6),
        "nav_drawdown_trough_date": inner_dates[trough_idx],
        "nav_ann_vol": round(_std(nav_log_rets, 1) * math.sqrt(252), 6),
        "num_late_fixes": num_late,
        "num_inverted_fix_rows": sum(1 for r in fx_rows if r["quote"] != "EURUSD"),
        "num_weekend_prints": sum(
            1 for r in fx_rows
            if to_utc(r["timestamp"]).strftime("%Y-%m-%d") not in union_set),
        "hedge_pnl_total": round(sum(p for _d, p in settlements), 6),
        "num_hedge_resets": len(resets),
        "hedged_nav_ann_vol": round(_std(h_log, 1) * math.sqrt(252), 6),
        "hedged_nav_max_drawdown": round(h_dd, 6),
        **strategy_fields,
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
    probe_ticker_2 = re.search(r"probe_ticker_2: (\S+)", spec).group(1)
    probe_dates_2 = re.search(r"probe_dates_2: (.+)", spec).group(1).split(", ")
    rp_tickers = re.search(
        r"risk_parity_tickers: (.+)", spec).group(1).split(", ")

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
    corr_rows = _read_csv(task_dir, "corrections.csv")
    # "Applicable" (memo rule 1) = as_of instant at or before
    # report_time; later rows are excluded entirely and counted here.
    n_corr_late = 0
    for row in corr_rows:
        key = (row["ticker"], row["date"])
        when = _instant(row["as_of"])
        if when > report_instant:
            n_corr_late += 1
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

    def _parity_weights(cov):
        """Risk-parity weights for a 3x3 covariance (Newton, memo)."""

        def solve3(mat, vec):
            a = [row[:] + [vec[i]] for i, row in enumerate(mat)]
            for col in range(3):
                piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
                a[col], a[piv] = a[piv], a[col]
                for r in range(3):
                    if r != col:
                        f = a[r][col] / a[col][col]
                        for cc in range(col, 4):
                            a[r][cc] -= f * a[col][cc]
            return [a[i][3] / a[i][i] for i in range(3)]

        def obj(w):
            return 0.5 * sum(w[i] * cov[i][j] * w[j]
                             for i in range(3) for j in range(3)) \
                - sum(math.log(x) for x in w)

        w = [1.0 / math.sqrt(cov[i][i]) for i in range(3)]
        for _it in range(200):
            sw = [sum(cov[i][j] * w[j] for j in range(3)) for i in range(3)]
            grad = [sw[i] - 1.0 / w[i] for i in range(3)]
            if max(abs(g) for g in grad) < 1e-13:
                break
            hess = [[cov[i][j] + (1.0 / w[i] ** 2 if i == j else 0.0)
                     for j in range(3)] for i in range(3)]
            step = solve3(hess, [-g for g in grad])
            t_step, f0 = 1.0, obj(w)
            for _ls in range(80):
                cand = [w[i] + t_step * step[i] for i in range(3)]
                if all(x > 0 for x in cand) and obj(cand) <= f0:
                    break
                t_step *= 0.5
            w = [w[i] + t_step * step[i] for i in range(3)]
        total = sum(w)
        return [x / total for x in w]

    # Risk-parity weights (memo Derived fields): unique positive
    # weights summing to 1 with w_i (Sigma w)_i equal across the three
    # tickers, Sigma the ddof=1 log-return covariance on the dates
    # present for all three.  Solved via the strictly convex
    # reformulation min 0.5 w' Sigma w - sum(ln w_i) (Newton +
    # backtracking), whose minimiser has exactly equal contributions.
    rp_days = sorted(set.intersection(*[set(by[t]) for t in rp_tickers]))
    rp_rets = {t: _returns([by[t][d] for d in rp_days], "log")
               for t in rp_tickers}
    n_rp = len(rp_days) - 1
    rp_mean = {t: _mean(rp_rets[t]) for t in rp_tickers}
    rp_cov = [[sum((rp_rets[a][k] - rp_mean[a]) * (rp_rets[b][k] - rp_mean[b])
                   for k in range(n_rp)) / (n_rp - 1)
               for b in rp_tickers] for a in rp_tickers]

    rp_weights = dict(zip(
        rp_tickers, _parity_weights(rp_cov), strict=True))

    # Rolling risk-parity strategy (memo Derived fields): re-solve
    # the parity weights on trailing 60-return windows and trade
    # them in lots with minimum trade size and costs at resolved
    # closes, carrying shares and cash across rebalances.
    mrs = re.search(
        r"rp_strategy: capital=(\d+); window_returns=(\d+); "
        r"first_rebalance_index=(\d+); rebalance_step=(\d+); "
        r"cash_buffer_pct=([\d.]+); min_trade_shares=(\d+); "
        r"cost_bps=([\d.]+); lots: (.+)$", spec, re.MULTILINE)
    rs_capital, rs_window = float(mrs.group(1)), int(mrs.group(2))
    rs_first, rs_step = int(mrs.group(3)), int(mrs.group(4))
    rs_buffer = float(mrs.group(5))
    rs_min_trade, rs_cost_bps = int(mrs.group(6)), float(mrs.group(7))
    rs_lots = {t: int(v) for t, v in re.findall(r"(\w+)=(\d+)", mrs.group(8))}
    rs_days = rp_days
    rs_closes = [[by[t][d] for d in rs_days] for t in rp_tickers]

    def rs_window_weights(idx):
        rets = []
        for c in rs_closes:
            seg = c[idx - rs_window:idx + 1]
            rets.append([math.log(b / a)
                         for a, b in zip(seg, seg[1:], strict=False)])
        m = len(rets[0])
        means = [_mean(r) for r in rets]
        cov = [[sum((rets[i][k] - means[i]) * (rets[j][k] - means[j])
                    for k in range(m)) / (m - 1)
                for j in range(3)] for i in range(3)]
        return _parity_weights(cov)

    rs_shares = [0, 0, 0]
    rs_cash, rs_total_cost, rs_last_w = rs_capital, 0.0, None
    rs_nav_path, rs_n_reb = [], 0
    for idx in range(len(rs_days)):
        px = [rs_closes[i][idx] for i in range(3)]
        if idx >= rs_first and (idx - rs_first) % rs_step == 0:
            nav_pre = rs_cash + sum(s * p for s, p in
                                    zip(rs_shares, px, strict=True))
            w = rs_window_weights(idx)
            rs_last_w, rs_n_reb = w, rs_n_reb + 1
            desired = [wi * nav_pre * (1 - rs_buffer) for wi in w]
            lot_of = [rs_lots[t] for t in rp_tickers]
            new_shares = [int(desired[i] / px[i] // lot_of[i]) * lot_of[i]
                          for i in range(3)]
            cash_left = nav_pre - sum(s * p for s, p in
                                      zip(new_shares, px, strict=True))
            while True:
                residuals = [desired[i] - new_shares[i] * px[i]
                             for i in range(3)]
                bought = False
                for i in sorted(range(3), key=lambda i: (
                        -residuals[i], rp_tickers[i])):
                    lot_cost = lot_of[i] * px[i]
                    if residuals[i] > 1e-9 and cash_left >= lot_cost - 1e-9:
                        new_shares[i] += lot_of[i]
                        cash_left -= lot_cost
                        bought = True
                        break
                if not bought:
                    break
            new_shares = [
                rs_shares[i] if 0 < abs(new_shares[i] - rs_shares[i])
                < rs_min_trade else new_shares[i] for i in range(3)]
            traded = sum(abs(ns - s) * p for ns, s, p in
                         zip(new_shares, rs_shares, px, strict=True))
            cash_left = nav_pre - sum(s * p for s, p in
                                      zip(new_shares, px, strict=True))
            cost = traded * rs_cost_bps / 10000
            while cash_left - cost < 0 and any(s > 0 for s in new_shares):
                excess = [new_shares[i] * px[i] - desired[i] for i in range(3)]
                pick = min(range(3), key=lambda i: (-excess[i], rp_tickers[i]))
                if new_shares[pick] <= 0:
                    pick = max(range(3), key=lambda i: new_shares[i])
                new_shares[pick] -= lot_of[pick]
                cash_left += lot_of[pick] * px[pick]
                traded = sum(abs(ns - s) * p for ns, s, p in
                             zip(new_shares, rs_shares, px, strict=True))
                cost = traded * rs_cost_bps / 10000
            rs_cash = cash_left - cost
            rs_total_cost += cost
            rs_shares = new_shares
        rs_nav_path.append(rs_cash + sum(s * p for s, p in
                                         zip(rs_shares, px, strict=True)))
    rs_dd, _rs_tr = _drawdown(rs_nav_path)
    rs_log = [math.log(b / a)
              for a, b in zip(rs_nav_path, rs_nav_path[1:], strict=False)]
    rp_strategy_fields = {
        "rp_strategy_final_nav": round(rs_nav_path[-1], 6),
        "rp_strategy_ann_vol": round(_std(rs_log, 1) * math.sqrt(252), 6),
        "rp_strategy_max_drawdown": round(rs_dd, 6),
        "rp_strategy_total_cost": round(rs_total_cost, 6),
        "num_rp_rebalances": rs_n_reb,
        "rp_final_shares": {t: int(s) for t, s in
                            zip(rp_tickers, rs_shares, strict=True)},
        "rp_last_weights": {t: round(float(wi), 10) for t, wi in
                            zip(rp_tickers, rs_last_w, strict=True)},
        "rp_strategy_cash_end": round(rs_cash, 6),
    }
    common = sorted(set.intersection(*[set(by[t]) for t in tickers]))
    path = [_mean([by[t][d] / by[t][common[0]] for t in tickers]) for d in common]
    dd, trough = _drawdown(path)
    # Probe-ticker risk on its own resolved series (date order).
    p_dates = sorted(by[probe_ticker])
    p_closes = [by[probe_ticker][d] for d in p_dates]
    p_dd, p_trough = _drawdown(p_closes)
    answer = {
        "resolved_by_correction": paths["correction"],
        "resolved_by_vendor_a": paths["vendor_a"],
        "resolved_by_median": paths["median"],
        "num_conflicted_keys": sum(paths.values()),
        "reconciled_total_return": round(path[-1] - 1, 6),
        "max_drawdown": round(dd, 6),
        "max_drawdown_trough_date": common[trough],
        "probe_ann_vol": round(
            _std(_returns(p_closes, "log"), 1) * math.sqrt(252), 6),
        "probe_ticker_max_drawdown": round(p_dd, 6),
        "probe_ticker_drawdown_trough_date": p_dates[p_trough],
        "num_correction_rows_total": len(corr_rows),
        "num_correction_rows_late": n_corr_late,
        **{f"risk_parity_weight_{t}": round(rp_weights[t], 6)
           for t in rp_tickers},
        **rp_strategy_fields,
        "num_resolved_days": len(common),
        "problems": _t3_problems(task_dir, resolved, a_val, b_val),
    }
    for d in probe_dates:
        answer[f"resolved_close_{probe_ticker}_{d.replace('-', '')}"] = \
            round(resolved[(probe_ticker, d)], 6)
    for d in probe_dates_2:
        answer[f"resolved_close_{probe_ticker_2}_{d.replace('-', '')}"] = \
            round(resolved[(probe_ticker_2, d)], 6)
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

    # Periodic-rebalance parameters (memo rule 9) and the terminal
    # lot sizes / minimum trade size (memo rule 7).
    spec_t4 = _spec(task_dir)
    mrb = re.search(r"Periodic rebalances: dates are ([0-9, -]+); "
                    r"cost_bps=(\d+)", spec_t4)
    rebalance_dates = {d.strip() for d in mrb.group(1).split(",")}
    rebalance_cost_bps = float(mrb.group(2))
    lot_sizes_t4 = {t: int(v) for t, v in re.findall(
        r"(\w+)=(\d+)", re.search(
            r"lot size, in shares: ([^.]+)\.", spec_t4).group(1))}
    min_trade_t4 = int(re.search(
        r"minimum\s+trade\s+size\s+is\s+(\d+)\s+shares", spec_t4).group(1))

    lots = {t: [] for t in assets}
    cash_lieu, realised = 0.0, 0.0
    cash_balance = 0.0
    reb_cost_total, reb_trade_count, reb_count = 0.0, 0, 0
    lieu_events = 0
    values, flows = [], []
    for ds in dates:
        for ticker, a, b in events.get(ds, []):
            ratio = a / b
            for lot in lots[ticker]:
                if lot[2] >= ds:  # settlement must be before the ex-date
                    continue
                new_shares = lot[0] * ratio
                whole = math.floor(new_shares + 1e-9)
                # The fraction's basis leaves with the cash: per-share
                # cost divides by the ratio (memo rule 2).
                cash_lieu += (new_shares - whole) * price_at[ticker][ds]
                cash_balance += (new_shares - whole) * price_at[ticker][ds]
                if new_shares - whole > 1e-9:
                    lieu_events += 1
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
        if ds in rebalance_dates:
            # Periodic equal-weight rebalance (memo rule 9): targets
            # from the cash balance plus holdings value, floored to
            # lots with the minimum trade size, cost safeguard on the
            # leftover, sells FIFO before buys, cost deducted last.
            reb_count += 1
            current = {t: int(round(sum(lot[0] for lot in lots[t])))
                       for t in assets}
            total_v = cash_balance + sum(current[t] * price_at[t][ds]
                                         for t in assets)
            target_dollar = total_v / len(assets)
            target = {t: int(math.floor(target_dollar / price_at[t][ds]))
                      for t in assets}
            target = {t: int(target[t] // lot_sizes_t4[t]) * lot_sizes_t4[t]
                      for t in assets}
            target = {t: current[t]
                      if 0 < abs(target[t] - current[t]) < min_trade_t4
                      else target[t] for t in assets}

            traded_rb = sum(abs(target[t] - current[t]) * price_at[t][ds]
                            for t in assets)
            cost_rb = traded_rb * rebalance_cost_bps / 10000
            leftover_rb = total_v - sum(target[t] * price_at[t][ds]
                                        for t in assets)
            while leftover_rb < cost_rb and any(
                    target[t] > current[t] for t in assets):
                buys = [t for t in assets if target[t] > current[t]]
                pick = min(buys, key=lambda t: (
                    -(target[t] * price_at[t][ds] - target_dollar), t))
                target[pick] = max(current[pick],
                                   target[pick] - lot_sizes_t4[pick])
                traded_rb = sum(abs(target[t] - current[t]) * price_at[t][ds]
                                for t in assets)
                cost_rb = traded_rb * rebalance_cost_bps / 10000
                leftover_rb = total_v - sum(target[t] * price_at[t][ds]
                                            for t in assets)
            for t in assets:
                if target[t] < current[t]:
                    px = price_at[t][ds]
                    to_sell = current[t] - target[t]
                    for lot in lots[t]:
                        if to_sell <= 0:
                            break
                        take = min(lot[0], to_sell)
                        realised += take * (px - lot[1])
                        lot[0] -= take
                        to_sell -= take
                    lots[t] = [lot for lot in lots[t] if lot[0] > 0]
                    cash_balance += (current[t] - target[t]) * px
                    reb_trade_count += 1
            for t in assets:
                if target[t] > current[t]:
                    diff = target[t] - current[t]
                    settle_idx = min(dates.index(ds) + 2, len(dates) - 1)
                    lots[t].append([float(diff), price_at[t][ds],
                                    dates[settle_idx]])
                    cash_balance -= diff * price_at[t][ds]
                    reb_trade_count += 1
            cash_balance -= cost_rb
            reb_cost_total += cost_rb
        flows.append(flow)
        values.append(cash_balance + sum(
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
    dd, dd_idx = _drawdown(index)
    idx_dates = dates[len(dates) - len(index):]
    end_shares = {t: int(round(sum(lot[0] for lot in lots[t]))) for t in assets}
    remaining_cost = sum(lot[0] * lot[1] for t in assets for lot in lots[t])
    end_mv = sum(end_shares[t] * price_at[t][dates[-1]] for t in assets)
    # Downstream chain: risk on the TWR index, then the terminal
    # equal-weight rebalance on the end state (memo rules 6-8).
    idx_rets = _returns(index, "simple")
    worst_roll, worst_roll_end = None, None
    for j in range(20, len(index)):
        r = index[j] / index[j - 20] - 1
        if worst_roll is None or r < worst_roll:
            worst_roll, worst_roll_end = r, j
    total_value = end_mv + cash_balance
    # Terminal rebalance (memo rule 7): n is the number of distinct
    # tickers in prices.csv; targets are floored to whole multiples of
    # each asset's lot size, and a trade smaller than the minimum
    # trade size is not made (target stays at the end share count).
    n_assets = len(assets)
    spec = _spec(task_dir)
    lot_sizes = {t: int(v) for t, v in re.findall(
        r"(\w+)=(\d+)", re.search(
            r"lot size, in shares: ([^.]+)\.", spec).group(1))}
    term_min_trade = int(re.search(
        r"minimum\s+trade\s+size\s+is\s+(\d+)\s+shares", spec).group(1))
    target_shares = {t: int(math.floor(
        (total_value / n_assets) / price_at[t][dates[-1]])) for t in assets}
    target_shares = {
        t: int(target_shares[t] // lot_sizes[t]) * lot_sizes[t]
        for t in assets}
    target_shares = {
        t: end_shares[t]
        if 0 < abs(target_shares[t] - end_shares[t]) < term_min_trade
        else target_shares[t] for t in assets}
    answer = {
        **{f"shares_{t}": end_shares[t] for t in assets},
        "realised_pnl_fifo": round(realised, 6),
        "unrealised_pnl": round(end_mv - remaining_cost, 6),
        "cash_in_lieu": round(cash_lieu, 6),
        "twr_total_return": round(index[-1] - 1, 6),
        "max_drawdown": round(dd, 6),
        "max_drawdown_trough_date": idx_dates[dd_idx],
        "twr_ann_vol": round(_std(idx_rets, 1) * math.sqrt(252), 6)
        if len(idx_rets) > 1 else 0.0,
        "worst_rolling_20d_return": round(worst_roll, 6)
        if worst_roll is not None else 0.0,
        "worst_rolling_20d_end_date": idx_dates[worst_roll_end]
        if worst_roll_end is not None else dates[-1],
        "num_cash_in_lieu_events": lieu_events,
        "num_lots_end": sum(len(lots[t]) for t in assets),
        "cash_end": round(cash_balance, 6),
        "rebalance_total_cost": round(reb_cost_total, 6),
        "num_periodic_rebalances": reb_count,
        "num_rebalance_trades": reb_trade_count,
        "rebalance_target_shares": target_shares,
        "rebalance_trade_shares": {
            t: target_shares[t] - end_shares[t] for t in assets},
        "rebalance_leftover_cash": round(
            total_value - sum(target_shares[t] * price_at[t][dates[-1]]
                              for t in assets), 6),
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
    lot_sizes, min_trade = cfg["lot_sizes"], cfg["min_trade_shares"]
    buffer_pct = cfg["cash_buffer_pct"]

    sector_members: dict[str, list[int]] = {}
    for i, t in enumerate(assets):
        sector_members.setdefault(cfg["sectors"][t], []).append(i)

    def weights_at(end_idx):
        rets = np.array([_returns(closes[t], "simple")[end_idx - lookback:end_idx]
                         for t in assets]).T
        cov = np.cov(rets, rowvar=False, ddof=1)
        cov = (1 - delta) * cov + delta * np.diag(np.diag(cov))
        n, cap = len(assets), cfg["per_asset_cap"]
        caps_s = cfg["sector_caps"]
        sector_of = [cfg["sectors"][t] for t in assets]

        def solve_eq(fixed, equalities):
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
            # Stationarity over the free block includes the fixed
            # assets' gradient contribution 2*Sigma_f,fixed w_fixed;
            # omitting it solves the wrong QP whenever anything is
            # fixed (the old solver's ~2e-5 optimality gap).
            fixed_idx = [i for i in range(n) if i in fixed]
            grad_fixed = (2 * cov[np.ix_(free, fixed_idx)]
                          @ np.array([fixed[i] for i in fixed_idx])) \
                if fixed_idx else np.zeros(len(free))
            rhs = np.concatenate([-grad_fixed, b_vec])
            try:
                sol = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
            for i, v in zip(free, sol[:len(free)], strict=True):
                w[i] = v
            return w, sol[len(free):]

        # Primal active-set method for min w' Sigma w subject to
        # sum(w)=1, 0<=w<=cap and the sector caps: from a feasible
        # start, solve the working-set equality QP exactly, step as far
        # as feasibility allows, and drop the worst wrong-sign KKT
        # multiplier at the working-set optimum.  Ends only at a full
        # KKT point - the unique optimum, exact to linear-algebra
        # precision (no gradient-tolerance early stop).
        w = np.zeros(n)
        remaining = 1.0
        sector_left = dict(caps_s)
        for i in range(n):
            take = max(0.0, min(cap, remaining,
                                sector_left.get(sector_of[i], 1.0)))
            w[i] = take
            remaining -= take
            if sector_of[i] in sector_left:
                sector_left[sector_of[i]] -= take
        fixed = {i: float(w[i]) for i in range(n)
                 if w[i] <= 1e-13 or w[i] >= cap - 1e-13}
        active = sorted(
            s for s in caps_s
            if sum(w[i] for i in sector_members[s]) >= caps_s[s] - 1e-12)
        for _step in range(1000):
            equalities = [(list(range(n)), 1.0)] + [
                (sector_members[s], caps_s[s]) for s in active]
            w_star, lambdas = solve_eq(fixed, equalities)
            step = w_star - w
            if float(np.max(np.abs(step))) <= 1e-13:
                grad = 2 * cov @ w
                at_lam = np.zeros(n)
                for k, (members, _t) in enumerate(equalities):
                    for i in members:
                        at_lam[i] += lambdas[k]
                drop, worst = None, 1e-11
                for i, v in fixed.items():
                    resid = grad[i] + at_lam[i]
                    violation = -resid if v == 0.0 else resid
                    if violation > worst:
                        drop, worst = ("asset", i), violation
                for k, s in enumerate(active):
                    if -lambdas[1 + k] > worst:
                        drop, worst = ("sector", s), -lambdas[1 + k]
                if drop is None:
                    return w
                if drop[0] == "asset":
                    del fixed[drop[1]]
                else:
                    active.remove(drop[1])
                continue
            alpha, blocker = 1.0, None
            for i in range(n):
                if i in fixed:
                    continue
                if step[i] > 1e-14:
                    cand = (cap - w[i]) / step[i]
                    if cand < alpha:
                        alpha, blocker = cand, ("asset", i, cap)
                elif step[i] < -1e-14:
                    cand = w[i] / -step[i]
                    if cand < alpha:
                        alpha, blocker = cand, ("asset", i, 0.0)
            for s in caps_s:
                if s in active:
                    continue
                members = sector_members[s]
                slope = float(sum(step[i] for i in members))
                if slope > 1e-14:
                    slack = caps_s[s] - float(sum(w[i] for i in members))
                    cand = slack / slope
                    if cand < alpha:
                        alpha, blocker = cand, ("sector", s, caps_s[s])
            w = w + alpha * step
            w[np.abs(w) < 1e-15] = 0.0
            if blocker is not None and alpha < 1.0:
                if blocker[0] == "asset":
                    fixed[blocker[1]] = float(blocker[2])
                    w[blocker[1]] = blocker[2]
                else:
                    active.append(blocker[1])
                    active.sort()
        return w

    def allocate(desired, nav, px):
        # Floor to each asset's own lot size, then greedy: largest
        # positive residual first (ties by ticker), one lot at a time
        # while cash covers it.  Remaining cash includes the buffer,
        # so this may spend into it.
        shares = [int(math.floor(d / (p * lot_sizes[t])) * lot_sizes[t])
                  for d, p, t in zip(desired, px, assets, strict=True)]
        cash = nav - sum(s * p for s, p in zip(shares, px, strict=True))
        while True:
            residuals = [d - s * p for d, s, p in zip(desired, shares, px, strict=True)]
            for i in sorted(range(len(assets)), key=lambda i: (-residuals[i], assets[i])):
                if residuals[i] > 1e-9 and cash >= lot_sizes[assets[i]] * px[i] - 1e-9:
                    shares[i] += lot_sizes[assets[i]]
                    cash -= lot_sizes[assets[i]] * px[i]
                    break
            else:
                return shares, cash

    shares, cash = [0] * len(assets), float(cfg["capital"])
    weights_first = shares_first = None
    weights_last = shares_last = None
    total_cost, traded_last = 0.0, 0
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
                    * int(math.floor(abs(ns - s) * factor / lot_sizes[t])) * lot_sizes[t]
                    for ns, s, t in zip(new_shares, shares, assets, strict=True)]
                new_cash = nav_pre - sum(
                    s * p for s, p in zip(new_shares, px, strict=True))
                traded = sum(abs(ns - s) * p for ns, s, p in zip(new_shares, shares, px, strict=True))
            # Minimum trade size (memo step 5): a nonzero change
            # smaller than min_trade_shares is not traded at all.
            new_shares = [
                s if 0 < abs(ns - s) < min_trade else ns
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
                new_shares[pick] -= lot_sizes[assets[pick]]
                new_cash += lot_sizes[assets[pick]] * px[pick]
                traded = sum(abs(ns - s) * p for ns, s, p in zip(new_shares, shares, px, strict=True))
                cost = traded * cfg["cost_bps"] / 10000
                turnovers[str(len(turnovers))] = round(traded / nav_pre, 6)
            new_cash -= cost
            total_cost += cost
            if weights_first is None:
                weights_first = {t: round(float(wi), 6) for t, wi in zip(assets, w, strict=True)}
                shares_first = {t: int(s) for t, s in zip(assets, new_shares, strict=True)}
            weights_last = {t: round(float(wi), 6) for t, wi in zip(assets, w, strict=True)}
            shares_last = {t: int(s) for t, s in zip(assets, new_shares, strict=True)}
            traded_last = sum(
                1 for ns, s in zip(new_shares, shares, strict=True) if ns != s)
            shares, cash = new_shares, new_cash
        nav_path.append(cash + sum(s * p for s, p in zip(shares, px, strict=True)))
    dd, _ = _drawdown(nav_path)
    return {
        "weights_first_rebalance": weights_first,
        "shares_first_rebalance": shares_first,
        "weights_last_rebalance": weights_last,
        "shares_last_rebalance": shares_last,
        "turnover_by_rebalance": turnovers,
        "num_rebalances": len(rebalance_idx),
        "total_cost_paid": round(total_cost, 6),
        "num_assets_traded_last_rebalance": traded_last,
        "nav_ann_vol": round(_std(_returns(nav_path, "simple"), 1) * math.sqrt(252), 6),
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
