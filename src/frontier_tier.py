"""Frontier tier task generator (families F1-F4).

Settings per family live in FRONTIER_FAMILY_SETTINGS (rows, assets,
window / conflicts / actions / rebalance steps, cost) and can be
overridden per family via ``FrontierTaskGenerator(settings=...)``.

Design constraints (see README): no dividends, delistings/survivorship,
ticker renames, stale or repeated prices, or too-short histories. Splits
appear ONLY as explicit split-table events in F3; F1/F2/F4 price series
contain no split or unit-change level shifts at all. Every rule the
reference key uses is stated in the prompt text.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FRONTIER_FAMILY_SETTINGS = {
    # F1: long-history multi-asset rolling risk (plan: 250-750 rows,
    # 3-6 assets plus an index, 60-day window).
    "rolling_risk": {"rows": 500, "assets": 4, "window": 60},
    # F2: two-vendor reconciliation (plan knobs: conflicts, rows,
    # unit difference between files -> Vendor B quotes cents).
    "vendor_reconciliation": {"rows": 250, "assets": 3, "conflicts": 12},
    # F3: split-adjusted two-factor exposures (plan knobs: rows,
    # number of corporate actions; factors fixed at market + sector).
    "factor_exposure": {"rows": 500, "actions": 3},
    # F4: chained portfolio pipeline (plan knobs: assets, rebalances,
    # cost bps). Steps = rebalance interval in return observations.
    "portfolio_pipeline": {"rows": 300, "assets": 4,
                           "rebalance_every": 21, "cost_bps": 10},
}

CODES_13 = ["duplicate", "missing", "bad_price", "split", "reverse_split",
            "unit_change", "out_of_order", "timezone", "currency", "adj_close",
            "column_order", "calendar", "source_conflict"]

ISSUES = """Problem category definitions (fixed list; report a code only when its definition is met):
- duplicate: within one CSV block, a row has the same ticker and the same date as an earlier row in that block.
- missing: a Monday-Friday date inside a ticker's date range has no row; for vendor tasks, a date counts as present when either vendor reports it, and it is missing only when neither does.
- bad_price: a single close deviates by more than 50% from both the previous and the next close (in date order for that ticker, after duplicate removal) and then reverts; a one-day spike, not a sustained level shift.
- split: the split table lists at least one forward split event. Split ratios are written A-for-B, meaning A new shares for B old shares, so A>B is a forward split.
- reverse_split: the split table lists at least one reverse split event (A-for-B with A<B).
- unit_change: a sustained jump to about 100x the previous close that holds because prices switch units. Frontier price files contain no unit changes except the stated Vendor B cents convention in vendor tasks, which is not a unit_change.
- out_of_order: within a ticker and one CSV block, rows are not in ascending date order in the file as given.
- timezone: timestamp strings use more than one UTC-offset form. Frontier tasks use daily dates only, so this never applies.
- currency: a currency column shows prices quoted in a currency other than USD. Frontier files have no currency column, so this never applies.
- adj_close: an adj_close column is present and disagrees with close. Frontier files have no adj_close column, so this never applies.
- column_order: a second header row appears partway through a block with columns in a different order. Frontier blocks each have exactly one header, so this never applies.
- calendar: tickers trade on different sets of dates even though each is complete on its own calendar. All frontier tickers share the Monday-Friday calendar, so differing date sets imply missing, not calendar.
- source_conflict: in a vendor task, the two vendors report closes for the same ticker and date that differ by more than 0.01 after Vendor B is converted from cents to dollars and after each vendor is cleaned (duplicates and bad ticks removed)."""

RULES = """General consistency rules (apply in this order, whichever apply):
1. Parsing: the data file is described in the File layout section of each task. A line containing only an upper-case block name (for example VENDOR_A) starts that block; the next line is the block's CSV header. Blocks without a name line consist of a single header and its rows.
2. Ordering and duplicates: within a block, sort records by ticker, then by date. Sorting is stable. A row is a duplicate if it has the same ticker and the same date as an earlier row in the same block; drop every such duplicate, keeping the first occurrence in file order.
3. Bad ticks: after duplicate removal, for each ticker separately, a close is a bad tick when it deviates by more than 50% from both the previous close and the next close in that ticker's date-ordered series (neighbours taken from the series before any bad tick is removed). Remove every bad tick in one pass. First and last rows can never be bad ticks.
4. Prices: always use the close column after the adjustments stated for the task. Do not round intermediate values.
5. Returns are simple returns: r_t = P_t / P_{t-1} - 1, on consecutive observations of the cleaned, aligned series being analysed. Where a standard deviation, variance, covariance or correlation is used it is the sample statistic (ddof=1). Annualised volatility is std(daily returns, ddof=1) * sqrt(252).
6. Alignment across tickers is an inner join on date: keep only dates present for every ticker after cleaning. Never forward-fill.
7. Rounding: report decimal results rounded to 6 decimal places; counts are exact integers; dates are YYYY-MM-DD."""


def _prompt(layout: str, analytic: str, fields: str) -> str:
    return (
        "You are given market data. The data may contain quality issues.\n\n"
        f"{ISSUES}\n\n{RULES}\n\nFile layout:\n{layout}\n\n"
        f"Computation:\n{analytic}\n\n"
        "Return your answer as JSON with fields:\n"
        "- problems: JSON array of category codes from the fixed list above, "
        "with exactly one code per issue type present and no others "
        "(for example [\"duplicate\", \"missing\"])\n"
        f"{fields}"
    )


def _business_dates(n: int) -> list[datetime]:
    start = datetime(2021, 1, 4) + timedelta(days=random.randint(0, 500))
    while start.weekday() >= 5:
        start += timedelta(days=1)
    dates, day = [], start
    while len(dates) < n:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return dates


def _closes(start: float, rets) -> list[float]:
    price, out, prev = float(start), [], None
    for r in rets:
        price = max(1.0, price * (1.0 + float(r)))
        close = round(price, 2)
        if prev is not None and close == prev:
            close = round(prev + 0.01, 2)
            price = float(close)
        out.append(close)
        prev = close
    return out


def _frame(ticker: str, dates, closes) -> pd.DataFrame:
    rows = []
    for date, close in zip(dates, closes, strict=True):
        rows.append({
            "date": date.strftime("%Y-%m-%d"), "ticker": ticker,
            "open": round(close * random.uniform(0.995, 1.005), 2),
            "high": round(close * random.uniform(1.0, 1.01), 2),
            "low": round(close * random.uniform(0.99, 1.0), 2),
            "close": float(close),
            "volume": random.randint(100_000, 10_000_000),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reference
def clean_series(df: pd.DataFrame) -> pd.Series:
    """Prompt rules 2-3 for one ticker: dedupe, drop bad ticks. -> closes by date."""
    df = df.copy()
    df = df.sort_values("date", kind="stable").drop_duplicates(
        subset=["date"], keep="first")
    closes = [float(c) for c in df["close"]]
    keep = [True] * len(closes)
    for i in range(1, len(closes) - 1):
        prev, cur, nxt = closes[i - 1], closes[i], closes[i + 1]
        if prev > 0 and nxt > 0 and (cur / prev < 0.5 or cur / prev > 1.5) \
                and (cur / nxt < 0.5 or cur / nxt > 1.5):
            keep[i] = False
    dates = list(df["date"])
    return pd.Series([c for c, k in zip(closes, keep, strict=True) if k],
                     index=[d for d, k in zip(dates, keep, strict=True) if k],
                     dtype=float)


def aligned_prices(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = {t: clean_series(df) for t, df in frames.items()}
    return pd.DataFrame(series).sort_index().dropna()


_EXTRAS_DECK = [
    [], ["duplicate"], ["missing"], ["bad_price"], ["out_of_order"],
    ["duplicate", "missing"], ["bad_price", "out_of_order"],
    ["missing", "out_of_order"], ["duplicate", "bad_price"],
    ["duplicate", "out_of_order"], ["missing", "bad_price"],
    ["duplicate", "missing", "bad_price"],
    ["missing", "bad_price", "out_of_order"],
    ["duplicate", "missing", "out_of_order"],
    ["duplicate", "bad_price", "out_of_order"],
    ["duplicate", "missing", "bad_price", "out_of_order"],
]


class FrontierTaskGenerator:
    """Generates frontier-tier tasks (F1-F4) with per-family settings."""

    def __init__(self, seed: int = 42, settings: dict | None = None):
        random.seed(seed)
        self.task_id = 2000
        self.settings = {k: dict(v) for k, v in FRONTIER_FAMILY_SETTINGS.items()}
        for family, overrides in (settings or {}).items():
            self.settings[family].update(overrides)
        self._decks: dict[str, list] = {}

    # ------------------------------------------------------------ injectors
    def _deck(self, kind: str, items: list) -> list:
        if kind not in self._decks:
            deck = list(items)
            random.shuffle(deck)
            self._decks[kind] = [deck, 0]
        deck, idx = self._decks[kind]
        self._decks[kind][1] = idx + 1
        return list(deck[idx % len(deck)])

    @staticmethod
    def _inject(df: pd.DataFrame, kind: str):
        df = df.copy().reset_index(drop=True)
        if kind == "duplicate":
            idx = random.randint(5, len(df) - 5)
            out = pd.concat([df.iloc[:idx + 1], df.iloc[idx:idx + 1],
                             df.iloc[idx + 1:]], ignore_index=True)
            return out, f"duplicate_row_{df.iloc[idx]['date']}"
        if kind == "missing":
            idx = random.randint(5, len(df) - 5)
            label = df.iloc[idx]["date"]
            return pd.concat([df.iloc[:idx], df.iloc[idx + 1:]],
                             ignore_index=True), f"missing_day_{label}"
        if kind == "bad_price":
            idx = random.randint(5, len(df) - 5)
            mult = random.choice([0.1, 10.0])
            df.loc[idx, "close"] = round(float(df.loc[idx, "close"]) * mult, 2)
            df.loc[idx, "high"] = max(float(df.loc[idx, "high"]),
                                      float(df.loc[idx, "close"]))
            df.loc[idx, "low"] = min(float(df.loc[idx, "low"]),
                                     float(df.loc[idx, "close"]))
            return df, f"bad_price_{df.iloc[idx]['date']}_{mult}"
        raise ValueError(kind)

    def _apply_extras(self, frames: dict[str, pd.DataFrame], extras: list[str],
                      tickers: list[str]):
        """Apply bundle extras to random tickers; returns problem strings."""
        problems, working = [], {t: f.copy() for t, f in frames.items()}
        for kind in extras:
            ticker = random.choice(tickers)
            if kind == "out_of_order":
                continue  # applied on the combined frame at the end
            working[ticker], p = self._inject(working[ticker], kind)
            problems.append(p)
        return working, problems

    @staticmethod
    def _shuffle(frames: dict[str, pd.DataFrame], extras: list[str]):
        combined = pd.concat(frames.values(), ignore_index=True)
        if "out_of_order" in extras:
            tickers = list(frames)
            for _ in range(3):
                ticker = random.choice(tickers)
                pos = combined.index[combined["ticker"] == ticker].tolist()
                i, j = random.choice(pos), random.choice(pos)
                if i != j:
                    a, b = combined.loc[i].copy(), combined.loc[j].copy()
                    combined.loc[i], combined.loc[j] = b, a
            return combined, ["out_of_order_rows_3_swaps"]
        return combined, []

    # ------------------------------------------------------------------- F1
    def generate_task_rolling_risk(self):
        self.task_id += 1
        cfg = self.settings["rolling_risk"]
        rows, n_assets, window = cfg["rows"], cfg["assets"], cfg["window"]
        dates = _business_dates(rows)
        assets = [f"AST{i + 1}" for i in range(n_assets)]
        idx_rets = [random.gauss(0.0003, 0.010) for _ in range(rows - 1)]
        frames = {"IDX": _frame("IDX", dates, _closes(
            random.uniform(80, 120), [0.0] + idx_rets))}
        for asset in assets:
            beta = random.uniform(0.6, 1.6)
            rets = [beta * r + random.gauss(0, 0.008) for r in idx_rets]
            frames[asset] = _frame(asset, dates, _closes(
                random.uniform(40, 180), [0.0] + rets))
        extras = self._deck("f1_extras", _EXTRAS_DECK)
        frames, problems = self._apply_extras(frames, extras, assets + ["IDX"])
        combined, more = self._shuffle(frames, extras)
        problems += more

        prices = aligned_prices(frames)
        rets = prices.pct_change().dropna()
        betas, best = {}, None
        for asset in assets:
            vals = []
            for end in range(window, len(rets) + 1):
                w = rets.iloc[end - window:end]
                var = w["IDX"].var(ddof=1)
                vals.append(w[asset].cov(w["IDX"]) / var if var > 0 else 0.0)
                corr = w[asset].corr(w["IDX"])
                cand = (float(corr), str(rets.index[end - 1]), asset)
                if best is None or cand[0] > best[0]:
                    best = cand
            betas[asset] = round(float(np.mean(vals)), 6)
        ew = rets[assets].mean(axis=1)
        expected = {
            "mean_betas": betas,
            "peak_correlation": round(best[0], 6),
            "peak_correlation_asset": best[2],
            "peak_correlation_date": best[1],
            "final_window_volatility": round(
                float(ew.iloc[-window:].std(ddof=1) * 252 ** 0.5), 6),
            "num_assets": n_assets, "num_aligned_days": len(prices),
            "problems_found": sorted(problems),
        }
        layout = ("One CSV block with header date,ticker,open,high,low,close,volume. "
                  f"Tickers: {', '.join(assets)} and the index IDX.")
        analytic = f"""Clean every ticker with the general rules and align on the
inner join. Compute simple daily returns. For each asset and each window of
{window} consecutive return observations (windows end at return observations
{window}, {window + 1}, ..., N), beta = cov(asset, IDX, ddof=1) / var(IDX,
ddof=1) and correlation = the sample correlation of asset and IDX returns in
that window. mean_betas maps each asset ticker to the arithmetic mean of its
window betas. The peak correlation is the largest window correlation over all
assets and windows; report its value, the asset, and the window-end date (the
date of the last return in that window; break ties by earliest date, then by
asset ticker order as listed). The equal-weight portfolio return on a day is
the mean of the assets' returns that day (IDX excluded);
final_window_volatility is the annualised volatility of the last {window}
equal-weight portfolio returns."""
        task = {"id": f"task_f_{self.task_id:03d}", "type": "frontier_rolling_risk",
                "prompt": _prompt(layout, analytic,
                                  "- mean_betas: JSON object mapping each asset ticker to its mean beta (6 decimals)\n"
                                  "- peak_correlation: (6 decimals)\n"
                                  "- peak_correlation_asset: ticker string\n"
                                  "- peak_correlation_date: YYYY-MM-DD\n"
                                  "- final_window_volatility: (6 decimals)\n"
                                  "- num_assets: number of asset tickers (IDX excluded)\n"
                                  "- num_aligned_days: dates kept by the inner join"),
                "data_file": f"task_f_{self.task_id:03d}.csv", "expected": expected}
        return task, combined.to_csv(index=False)

    # ------------------------------------------------------------------- F2
    def generate_task_vendor_reconciliation(self):
        self.task_id += 1
        cfg = self.settings["vendor_reconciliation"]
        rows, n_assets, n_conf = cfg["rows"], cfg["assets"], cfg["conflicts"]
        dates = _business_dates(rows)
        tickers = [f"VEN{i + 1}" for i in range(n_assets)]
        base = {}
        for ticker in tickers:
            rets = [random.gauss(0.0004, 0.012) for _ in range(rows - 1)]
            base[ticker] = _frame(ticker, dates, _closes(
                random.uniform(40, 180), [0.0] + rets))
        extras = self._deck("f2_extras", _EXTRAS_DECK)
        problems = []
        if "missing" in extras:
            # Missing means neither vendor reports the date: delete it
            # from the shared base before the two blocks diverge.
            t = random.choice(tickers)
            base[t], p = self._inject(base[t], "missing")
            problems.append(p)
        # Vendor A gaps that Vendor B fills (not missing, not conflicts).
        # Dropped before any injection, so a gap can never delete an
        # injected bad tick or one copy of a duplicate pair.
        a_frames = {t: f.copy() for t, f in base.items()}
        for ticker in tickers:
            candidates = list(a_frames[ticker]["date"].unique())
            gap_dates = random.sample(candidates[10:-10], 3)
            a_frames[ticker] = a_frames[ticker][
                ~a_frames[ticker]["date"].isin(gap_dates)].reset_index(drop=True)
        a_frames, more = self._apply_extras(
            a_frames,
            [e for e in extras if e not in ("out_of_order", "missing")], tickers)
        problems += more
        b_frames = {}
        for ticker in tickers:
            b = base[ticker].copy()
            b["close"] = (b["close"] * 100).round(2)
            b_frames[ticker] = b
        # Conflicts: change Vendor B closes on dates Vendor A still reports.
        conflict_dates = []
        for _ in range(n_conf):
            ticker = random.choice(tickers)
            date = random.choice(a_frames[ticker]["date"].tolist())
            mask = (b_frames[ticker]["date"] == date)
            if mask.any():
                factor = random.choice([0.93, 1.06, 1.11, 0.88])
                b_frames[ticker].loc[mask, "close"] = (
                    b_frames[ticker].loc[mask, "close"] * factor).round(2)
                conflict_dates.append(date)
        a_combined, more = self._shuffle(a_frames, extras)
        problems += more
        b_combined = pd.concat(b_frames.values(), ignore_index=True)
        payload = "VENDOR_A\n" + a_combined.to_csv(index=False) + \
            "VENDOR_B\n" + b_combined.to_csv(index=False)

        trusted_frames = {}
        for ticker in tickers:
            a = clean_series(a_frames[ticker])
            b = clean_series(b_frames[ticker].assign(
                close=b_frames[ticker]["close"] / 100.0))
            dates_all = sorted(set(a.index) | set(b.index))
            trusted_frames[ticker] = pd.Series(
                [a[d] if d in a.index else b[d] for d in dates_all],
                index=dates_all, dtype=float)
        conflicts = []
        for ticker in tickers:
            a = clean_series(a_frames[ticker])
            b = clean_series(b_frames[ticker].assign(
                close=b_frames[ticker]["close"] / 100.0))
            for d in sorted(set(a.index) & set(b.index)):
                if abs(float(a[d]) - float(b[d])) > 0.01:
                    conflicts.append((d, ticker))
        aligned = pd.DataFrame(trusted_frames).sort_index().dropna()
        rel = aligned.iloc[-1] / aligned.iloc[0]
        problems = [f"source_conflict_{len(conflicts)}_pairs"] + problems \
            if conflicts else problems
        expected = {
            "mismatch_count": len(conflicts),
            "first_conflict_date": min(d for d, _ in conflicts) if conflicts else None,
            "reconciled_total_return": round(float(rel.mean() - 1.0), 6),
            "num_tickers": n_assets, "num_aligned_days": len(aligned),
            "problems_found": sorted(problems),
        }
        layout = ("Two CSV blocks. The first starts with a line VENDOR_A, the "
                  "second with a line VENDOR_B. Each block then has header "
                  "date,ticker,open,high,low,close,volume and one row per "
                  f"ticker and date. Tickers: {', '.join(tickers)}. Vendor A "
                  "quotes close in dollars. Vendor B quotes close in cents "
                  "(divide Vendor B closes by 100 before any comparison or use).")
        analytic = """Clean each vendor block separately with the general rules
(duplicates and bad ticks are judged within a block), converting Vendor B
closes from cents to dollars first. A source_conflict is a (ticker, date)
reported by both cleaned vendors whose closes differ by more than 0.01;
mismatch_count is the number of such pairs and first_conflict_date is the
earliest conflicting date (report null when there are none). Build the
trusted close series per ticker: use Vendor A's cleaned close when Vendor A
reports the date, otherwise Vendor B's converted close. Align the trusted
series on the inner join. The reconciled portfolio holds equal dollar
amounts of each ticker from the first aligned date (buy and hold, no
rebalancing): its value on a date is the mean over tickers of
close / first aligned close, and reconciled_total_return is the final value
minus 1."""
        task = {"id": f"task_f_{self.task_id:03d}",
                "type": "frontier_vendor_reconciliation",
                "prompt": _prompt(layout, analytic,
                                  "- mismatch_count: integer\n"
                                  "- first_conflict_date: YYYY-MM-DD or null\n"
                                  "- reconciled_total_return: (6 decimals)\n"
                                  "- num_tickers: number of tickers\n"
                                  "- num_aligned_days: dates kept by the inner join"),
                "data_file": f"task_f_{self.task_id:03d}.csv", "expected": expected}
        return task, payload

    # ------------------------------------------------------------------- F3
    def generate_task_factor_exposure(self):
        self.task_id += 1
        cfg = self.settings["factor_exposure"]
        rows, n_actions = cfg["rows"], cfg["actions"]
        dates = _business_dates(rows)
        mkt = [random.gauss(0.0003, 0.010) for _ in range(rows - 1)]
        sec = [0.4 * m + random.gauss(0, 0.007) for m in mkt]
        ast = [0.0002 + 1.15 * m + 0.55 * s + random.gauss(0, 0.006)
               for m, s in zip(mkt, sec, strict=True)]
        frames = {
            "AST": _frame("AST", dates, _closes(random.uniform(40, 120), [0.0] + ast)),
            "MKT": _frame("MKT", dates, _closes(random.uniform(80, 140), [0.0] + mkt)),
            "SEC": _frame("SEC", dates, _closes(random.uniform(60, 110), [0.0] + sec)),
        }
        extras = self._deck("f3_extras", _EXTRAS_DECK)
        frames, problems = self._apply_extras(
            frames, [e for e in extras if e not in ("out_of_order", "bad_price")],
            ["AST"])
        # Explicit split-table events; raw AST prices are un-adjusted.
        # Picks are chosen before the bad tick is injected, and the bad
        # tick is kept away from split dates: un-adjusting rescales its
        # neighbours, and a spike next to a split boundary could stop
        # deviating from both of them.
        raw = frames["AST"].copy().reset_index(drop=True)
        picks = sorted(random.sample(range(40, len(raw) - 40), n_actions))
        while any(b - a < 30 for a, b in zip(picks, picks[1:], strict=False)):
            picks = sorted(random.sample(range(40, len(raw) - 40), n_actions))
        if "bad_price" in extras:
            idx = random.randint(5, len(raw) - 5)
            while any(abs(idx - pos) <= 1 for pos in picks):
                idx = random.randint(5, len(raw) - 5)
            mult = random.choice([0.1, 10.0])
            raw.loc[idx, "close"] = round(float(raw.loc[idx, "close"]) * mult, 2)
            problems.append(f"bad_price_{raw.loc[idx, 'date']}_{mult}")
        events = []
        for pos in picks:
            a, b = random.choice([(2, 1), (3, 1), (1, 3), (1, 10)])
            events.append((str(raw.iloc[pos]["date"]), a, b))
            problems.append(
                f"{'split' if a > b else 'reverse_split'}_{raw.iloc[pos]['date']}_{a}-for-{b}")
            raw.loc[:pos - 1, "close"] = (
                raw.loc[:pos - 1, "close"] * (a / b)).round(2)
        prev, prev_date = None, None
        for i in range(len(raw)):  # never stale after un-adjusting
            close = float(raw.loc[i, "close"])
            if prev is not None and raw.loc[i, "date"] != prev_date \
                    and close == prev:
                raw.loc[i, "close"] = round(prev + 0.01, 2)
            prev, prev_date = float(raw.loc[i, "close"]), raw.loc[i, "date"]
        frames["AST"] = raw
        combined, more = self._shuffle(frames, extras)
        problems += more
        split_csv = "date,ticker,ratio\n" + "\n".join(
            f"{d},AST,{a}-for-{b}" for d, a, b in events) + "\n"
        payload = "PRICES\n" + combined.to_csv(index=False) + \
            "SPLIT_TABLE\n" + split_csv

        # Apply the split table exactly as the prompt states: back-adjust
        # every cleaned AST close before each event date by B/A
        # (cumulative over events).
        closes = clean_series(raw.copy())
        for d, a, b in events:
            closes.loc[closes.index < d] = closes.loc[closes.index < d] * (b / a)
        aligned = pd.DataFrame({"AST": closes,
                                "MKT": clean_series(frames["MKT"]),
                                "SEC": clean_series(frames["SEC"])}
                               ).sort_index().dropna()
        rets = aligned.pct_change().dropna()
        x = np.column_stack([np.ones(len(rets)), rets["MKT"], rets["SEC"]])
        coef, *_ = np.linalg.lstsq(x, rets["AST"].to_numpy(), rcond=None)
        fitted = x @ coef
        ss_res = float(((rets["AST"].to_numpy() - fitted) ** 2).sum())
        ss_tot = float(((rets["AST"].to_numpy() - rets["AST"].mean()) ** 2).sum())
        expected = {
            "alpha": round(float(coef[0]), 6),
            "beta_market": round(float(coef[1]), 6),
            "beta_sector": round(float(coef[2]), 6),
            "r_squared": round(1.0 - ss_res / ss_tot, 6) if ss_tot > 0 else 0.0,
            "num_return_days": len(rets), "num_splits": len(events),
            "problems_found": sorted(problems),
        }
        layout = ("Two blocks. The PRICES block has header "
                  "date,ticker,open,high,low,close,volume with tickers AST, "
                  "MKT and SEC. The SPLIT_TABLE block has header "
                  "date,ticker,ratio; each row is one split event for AST on "
                  "that date, with ratio written A-for-B (A new shares for B "
                  "old shares). AST closes in PRICES are unadjusted: on a "
                  "split date the price basis changes.")
        analytic = """Apply the general rules to the PRICES block. Then
back-adjust AST closes with the split table: for each event with ratio
A-for-B on date D, multiply every AST close with date < D by B/A (events
apply cumulatively; the close on D itself is already on the new basis).
MKT and SEC are never adjusted. Compute simple daily returns on the aligned
(inner join) adjusted AST, MKT and SEC closes and run the ordinary least
squares regression AST_return = alpha + beta_market * MKT_return +
beta_sector * SEC_return + error over all aligned return days. Report the
coefficients and the regression R-squared (1 - SS_res/SS_tot). num_splits is
the number of split-table events."""
        task = {"id": f"task_f_{self.task_id:03d}", "type": "frontier_factor_exposure",
                "prompt": _prompt(layout, analytic,
                                  "- alpha: (6 decimals)\n"
                                  "- beta_market: (6 decimals)\n"
                                  "- beta_sector: (6 decimals)\n"
                                  "- r_squared: (6 decimals)\n"
                                  "- num_return_days: integer\n"
                                  "- num_splits: number of split-table events"),
                "data_file": f"task_f_{self.task_id:03d}.csv", "expected": expected}
        return task, payload

    # ------------------------------------------------------------------- F4
    def generate_task_portfolio_pipeline(self):
        self.task_id += 1
        cfg = self.settings["portfolio_pipeline"]
        rows, n_assets = cfg["rows"], cfg["assets"]
        every, bps = cfg["rebalance_every"], cfg["cost_bps"]
        dates = _business_dates(rows)
        assets = [f"PF{i + 1}" for i in range(n_assets)]
        frames = {}
        for asset in assets:
            rets = [random.gauss(0.0004, 0.013) for _ in range(rows - 1)]
            frames[asset] = _frame(asset, dates, _closes(
                random.uniform(40, 180), [0.0] + rets))
        extras = self._deck("f4_extras", _EXTRAS_DECK)
        frames, problems = self._apply_extras(frames, extras, assets)
        combined, more = self._shuffle(frames, extras)
        problems += more

        prices = aligned_prices(frames)
        vals = prices.to_numpy(dtype=float)
        n = prices.shape[1]
        holdings = [1.0 / n] * n
        values = [1.0]
        first_rebalance_value, n_rebalances = None, 0
        for step in range(1, len(vals)):
            holdings = [h * vals[step][a] / vals[step - 1][a]
                        for a, h in enumerate(holdings)]
            total = sum(holdings)
            if step % every == 0 and step < len(vals) - 1:
                total *= (1.0 - bps / 10000.0)
                holdings = [total / n] * n
                n_rebalances += 1
                if first_rebalance_value is None:
                    first_rebalance_value = total
            values.append(total)
        series = pd.Series(values, dtype=float)
        peak = series.cummax()
        expected = {
            "num_aligned_days": len(prices), "num_assets": n_assets,
            "num_rebalances": n_rebalances,
            "first_rebalance_value": round(float(first_rebalance_value), 6),
            "final_return": round(float(values[-1] - 1.0), 6),
            "max_drawdown": round(float(((series - peak) / peak).min()), 6),
            "problems_found": sorted(problems),
        }
        layout = ("One CSV block with header date,ticker,open,high,low,close,volume. "
                  f"Tickers: {', '.join(assets)}.")
        analytic = f"""Clean every ticker with the general rules and align on
the inner join. Portfolio pipeline: value starts at 1.0 on the first aligned
date with equal dollar holdings in each asset (no return on that first
date). Each following date, holdings drift with that date's asset returns
(holding value times close ratio). After applying returns on return
observations {every}, {2 * every}, ... (the move into the 2nd aligned date
is observation 1), but not on the final return observation, rebalance:
reset every holding to an equal share of the current total, then deduct the
transaction cost of {bps} basis points of the total value (multiply the
total by 1 - {bps}/10000) and scale holdings to the reduced total.
first_rebalance_value is the total value immediately after the first
rebalance and its cost deduction. final_return is the final total value
minus 1. max_drawdown is the minimum over all dates (including the first,
value 1.0) of (value - running maximum value) / running maximum value,
using the after-cost value on rebalance dates. num_rebalances counts the
rebalances performed."""
        task = {"id": f"task_f_{self.task_id:03d}",
                "type": "frontier_portfolio_pipeline",
                "prompt": _prompt(layout, analytic,
                                  "- num_aligned_days: dates kept by the inner join\n"
                                  "- num_assets: number of tickers\n"
                                  "- num_rebalances: integer\n"
                                  "- first_rebalance_value: (6 decimals)\n"
                                  "- final_return: (6 decimals)\n"
                                  "- max_drawdown: (negative decimal, 6 decimals)"),
                "data_file": f"task_f_{self.task_id:03d}.csv", "expected": expected}
        return task, combined.to_csv(index=False)

    # ------------------------------------------------------------- generation
    def task_generators(self):
        return [self.generate_task_rolling_risk,
                self.generate_task_vendor_reconciliation,
                self.generate_task_factor_exposure,
                self.generate_task_portfolio_pipeline]

    def generate_all_tasks(self, output_dir, num_tasks=20):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "answers").mkdir(exist_ok=True)
        tasks, gens = [], self.task_generators()
        for i in range(num_tasks):
            task, payload = gens[i % len(gens)]()
            (output_dir / task["data_file"]).write_text(payload)
            tasks.append({k: v for k, v in task.items() if k != "expected"})
            with open(output_dir / "answers" / f"{task['id']}_answer.json", "w") as f:
                json.dump(task["expected"], f, indent=2)
        with open(output_dir / "tasks.json", "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"Generated {len(tasks)} frontier-tier tasks in {output_dir}")
        return tasks
