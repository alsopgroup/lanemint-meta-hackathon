"""
Hard tier task generator.

Every hard prompt carries the same fixed list of possible issues and the same
general consistency rules, so a prompt never reveals which issues a given
file actually contains.  Each prompt then adds the exact analytic definition
for its task type.  Reference keys are computed from those stated rules only.
"""

import io
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
PRICE_COLS = ["open", "high", "low", "close"]

POSSIBLE_ISSUES = """Problem categories (fixed list): duplicate, missing, bad_price, \
split, reverse_split, unit_change, out_of_order, timezone, currency, adj_close, \
column_order, calendar. A file may contain any subset of these, including none \
of some."""

GENERAL_RULES = """General consistency rules (apply in this order, whichever apply):
1. Parsing: the file is CSV. If a row's fields are themselves column names, that
   row is a new header; rows after it follow that header's column order until
   any later header row. Map every data row by its current header names.
2. Ordering: sort records ascending by ticker (if present), then by date or
   timestamp. Sorting is stable. Exact duplicate records for the same
   ticker and date/timestamp are dropped, keeping the first.
3. Price field: always use the "close" column after the adjustments below.
   Ignore any "adj_close" column entirely; it may be partly adjusted and can
   disagree with "close".
4. Currency: if a "currency" column is present, a row's prices are quoted in
   that currency and its "fx_rate" column is US dollars per 1 unit of that
   currency on that row's date (1.0 for USD rows). Convert before any other
   price adjustment: USD_close = close * fx_rate.
5. Timestamps: if a "timestamp" column is present, timestamps may mix UTC and
   offset forms. Convert every timestamp to America/New_York. If the same
   instant occurs more than once, keep only the first occurrence in original
   file order. The daily close for an America/New_York calendar date is the
   close of the last remaining bar on that date.
6. Sustained price-level shifts (per ticker, on sorted converted closes):
   compare each close with the previous close. "Sustained" means the next
   available close divided by the close before the shift is also > 1.8; if
   it is not, make no adjustment for that shift. If the ratio is > 50 and
   sustained above 50, prices from that row onward switch to cents: divide
   them by 100. Otherwise, if the ratio is > 1.8 and sustained, it is a
   reverse split: divide prices from that row onward by whichever of the
   candidate ratios 3 or 10 is closest to the observed ratio in relative
   terms (|ratio - candidate| / candidate). Do not round adjusted prices.
7. Returns are simple returns: r_t = P_t / P_{t-1} - 1, on consecutive
   observations of the aligned series being analysed. Where a standard
   deviation, variance or covariance is used it is the sample statistic
   (ddof=1). Annualised volatility, where asked, is std(daily returns,
   ddof=1) * sqrt(252).
8. Alignment across tickers is an inner join on date: keep only dates present
   for every ticker after cleaning. Never forward-fill.
9. Rounding: report decimal results rounded to 6 decimal places; counts are
   exact integers. Do not round intermediate values."""


def _fmt_prompt(analytic: str, fields: str) -> str:
    return (
        "You are given market data in CSV format. The data may contain quality issues.\n\n"
        f"{POSSIBLE_ISSUES}\n\n{GENERAL_RULES}\n\n"
        f"Computation:\n{analytic}\n\n"
        "Return your answer as JSON with fields:\n"
        "- problems: JSON array of category codes from the fixed list above, "
        "with exactly one code per issue type present and no others "
        "(for example [\"unit_change\", \"out_of_order\"])\n"
        f"{fields}"
    )


class HardTaskGenerator:
    """Generates hard-tier tasks with complex, interacting data quality issues."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.task_id = 1000

    # ------------------------------------------------------------------ data
    def _is_us_holiday(self, date):
        return (date.month, date.day) in {(1, 1), (7, 4), (12, 25), (11, 28)}

    def _is_eu_holiday(self, date):
        return (date.month, date.day) in {(1, 1), (5, 1), (12, 25), (12, 26), (8, 15)}

    def generate_clean_prices_multi_asset(self, tickers, start_date, days):
        dfs = {}
        for ticker in tickers:
            dates = [start_date + timedelta(days=i) for i in range(days * 2)]
            if ticker.endswith("_EU"):
                dates = [d for d in dates if d.weekday() < 5 and not self._is_eu_holiday(d)]
            elif ticker.endswith("_US"):
                dates = [d for d in dates if d.weekday() < 5 and not self._is_us_holiday(d)]
            else:
                dates = [d for d in dates if d.weekday() < 5]
            dates = dates[:days]
            price = random.uniform(50, 150)
            rows = []
            prev_close = None
            for date in dates:
                price = max(0.01, price * (1 + random.gauss(0.0005, 0.02)))
                close = round(price, 2)
                if prev_close is not None and close == prev_close:
                    close = round(prev_close + 0.01, 2)  # never a stale close
                    price = float(close)
                open_price = close * random.uniform(0.99, 1.01)
                high = max(open_price, float(close)) * random.uniform(1.00, 1.02)
                low = min(open_price, float(close)) * random.uniform(0.98, 1.00)
                rows.append({
                    "date": date, "ticker": ticker,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": close,
                    "volume": random.randint(100_000, 10_000_000),
                })
                prev_close = close
            dfs[ticker] = pd.DataFrame(rows)
        return dfs

    # -------------------------------------------------------------- injectors
    def inject_out_of_order_rows(self, df, num_swaps=3):
        df = df.copy().reset_index(drop=True)
        done = 0
        attempts = 0
        while done < num_swaps and attempts < 100 and len(df) >= 4:
            attempts += 1
            i = random.randint(1, len(df) - 2)
            j = random.randint(1, len(df) - 2)
            if i == j:
                continue
            a, b = df.iloc[i].copy(), df.iloc[j].copy()
            df.iloc[i], df.iloc[j] = b, a
            done += 1
        return df, f"out_of_order_rows_{done}_swaps"

    @staticmethod
    def _row_label(df, idx):
        row = df.iloc[idx]
        if "date" in df.columns:
            return row["date"].strftime("%Y-%m-%d")
        return str(row["timestamp"])

    def inject_duplicate_row(self, df):
        df = df.copy().reset_index(drop=True)
        idx = random.randint(5, len(df) - 5)
        out = pd.concat([df.iloc[:idx + 1], df.iloc[idx:idx + 1], df.iloc[idx + 1:]],
                        ignore_index=True)
        return out, f"duplicate_row_{self._row_label(df, idx)}"

    def inject_missing_row(self, df):
        df = df.copy().reset_index(drop=True)
        idx = random.randint(5, len(df) - 5)
        label = self._row_label(df, idx)
        out = pd.concat([df.iloc[:idx], df.iloc[idx + 1:]], ignore_index=True)
        kind = "day" if "date" in df.columns else "bar"
        return out, f"missing_{kind}_{label}"

    @staticmethod
    def dedupe_stale_closes(df, order_col="date"):
        """Nudge closes (by +0.01) that repeat the previous distinct date/instant.

        Runs after injection, so split/unit/intraday scaling cannot leave
        identical consecutive closes. Rows sharing the same date/instant
        (intended duplicates) are exempt.
        """
        df = df.copy()
        if "close" not in df.columns:
            return df
        groups = df.groupby("ticker", sort=False) if "ticker" in df.columns else [("", df)]
        for _, group in groups:
            if order_col == "timestamp":
                keys = [pd.Timestamp(v) for v in group[order_col]]
                if keys and keys[0].tzinfo is None:
                    keys = [k.tz_localize("UTC") for k in keys]
                keys = [k.tz_convert(NY) for k in keys]
            else:
                keys = list(group[order_col])
            order = sorted(range(len(group)), key=lambda i: keys[i])
            prev_close, prev_key = None, None
            for pos in order:
                idx = group.index[pos]
                key = keys[pos]
                close = round(float(df.loc[idx, "close"]), 2)
                if prev_close is not None and key != prev_key and close == prev_close:
                    close = round(prev_close + 0.01, 2)
                    df.loc[idx, "close"] = close
                    if "high" in df.columns:
                        df.loc[idx, "high"] = max(float(df.loc[idx, "high"]), close)
                    if "low" in df.columns:
                        df.loc[idx, "low"] = min(float(df.loc[idx, "low"]), close)
                prev_close, prev_key = close, key
        return df

    def inject_unit_change(self, df, change_point):
        df = df.copy().reset_index(drop=True)
        df.loc[change_point:, PRICE_COLS] = (df.loc[change_point:, PRICE_COLS] * 100).round(2)
        return df, f"unit_change_{self._row_label(df, change_point)}_dollars_to_cents"

    def inject_reverse_split(self, df, multiple=3, add_adj_close=False):
        """Exact integer reverse split: post-split prices x multiple, volume / multiple."""
        df = df.copy().reset_index(drop=True)
        idx = random.randint(20, len(df) - 20)
        df.loc[idx:, PRICE_COLS] = (df.loc[idx:, PRICE_COLS] * multiple).round(2)
        df.loc[idx:, "volume"] = (df.loc[idx:, "volume"] // multiple).astype(int)
        date_str = self._row_label(df, idx)
        problems = [f"reverse_split_{date_str}_1-for-{multiple}"]
        if add_adj_close:
            df, p = self.add_adj_close_conflict(df, multiple=multiple)
            problems.append(p)
        return df, problems

    def add_adj_close_conflict(self, df, multiple=3):
        """Add a partly-adjusted adj_close that disagrees with close."""
        df = df.copy().reset_index(drop=True)
        df["adj_close"] = df["close"].copy()
        half = len(df) // 2
        df.loc[:half, "adj_close"] = (df.loc[:half, "close"] * multiple).round(2)
        date_str = df.iloc[half]["date"].strftime("%Y-%m-%d") if "date" in df.columns else "intraday"
        return df, f"adj_close_conflict_{date_str}"

    def add_currency_columns(self, df, currency, base_fx=1.08):
        """Add currency/fx_rate; fx_rate is USD per 1 unit, same for a given date."""
        df = df.copy()
        rate = base_fx
        rates = {}
        for date in df["date"]:
            if date not in rates:
                rate = max(0.5, rate * random.uniform(0.995, 1.005))
                rates[date] = 1.0 if currency == "USD" else round(rate, 6)
        df["currency"] = currency
        df["fx_rate"] = df["date"].map(rates)
        return df

    def inject_mixed_timezone_intraday(self, df, repeat_instant=True):
        """Expand daily rows to intraday bars in mixed UTC / New-York forms.

        Bars run 01:00-20:00 New-York time across the spring-forward week, so
        the 20:00 bar's UTC date is the next calendar day, and the UTC offset
        changes mid-series. Prices and volumes vary bar to bar. Optionally one
        instant is repeated in a second textual form (same values).
        """
        rows = []
        hours = [1, 9, 12, 16, 20]
        factors = [0.990, 0.995, 1.004, 1.000, 1.002]
        for _, row in df.iterrows():
            day = row["date"]
            for hour, factor in zip(hours, factors, strict=True):
                close = round(row["close"] * factor, 2)
                local = datetime(day.year, day.month, day.day, hour, tzinfo=NY)
                if (len(rows) // len(hours)) % 2 == 0:
                    stamp = local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    stamp = local.isoformat()
                rows.append({
                    "timestamp": stamp, "ticker": row["ticker"],
                    "open": round(close * 0.999, 2), "high": round(close * 1.001, 2),
                    "low": round(close * 0.998, 2), "close": close,
                    "volume": int(row["volume"] * random.uniform(0.15, 0.35)),
                })
        out = pd.DataFrame(rows)
        if repeat_instant:
            src = out.iloc[10].copy()
            instant = pd.Timestamp(src["timestamp"])
            if str(src["timestamp"]).endswith("Z"):
                src["timestamp"] = instant.tz_convert(NY).isoformat()
            else:
                src["timestamp"] = instant.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            out = pd.concat([out, src.to_frame().T], ignore_index=True)
        for col in [*PRICE_COLS, "volume"]:
            out[col] = pd.to_numeric(out[col])
        return out, "timezone_mixed_offsets_repeated_instant"

    def make_column_order_csv(self, df, split_point):
        """Serialise as two concatenated exports with different column orders."""
        df = df.reset_index(drop=True)
        first = df.iloc[:split_point]
        second = df.iloc[split_point:]
        order2 = ["ticker", "date", "close", "open", "high", "low", "volume"]
        order2 = [c for c in order2 if c in df.columns] + \
            [c for c in df.columns if c not in order2]
        buf = io.StringIO()
        first.to_csv(buf, index=False)
        second.to_csv(buf, index=False, columns=order2)
        return buf.getvalue(), f"column_order_change_at_row_{split_point}"

    # ------------------------------------------------------- reference rules
    @staticmethod
    def _adjust_shifts(df):
        """Apply rule 6 to a sorted single-ticker frame (close already in USD)."""
        df = df.reset_index(drop=True).copy()
        closes = df["close"].astype(float).tolist()
        i = 1
        while i < len(closes):
            prev, cur = closes[i - 1], closes[i]
            if prev > 0 and cur / prev > 1.8 and i + 1 < len(closes):
                sustained = closes[i + 1] / prev
                if sustained <= 1.8:
                    factor = None
                elif cur / prev > 50 and sustained > 50:
                    factor = 100.0
                elif sustained > 1.8:
                    ratio = cur / prev
                    factor = float(min((3, 10), key=lambda c: abs(ratio - c) / c))
                else:
                    factor = None
                if factor and factor != 1.0:
                    for j in range(i, len(closes)):
                        closes[j] /= factor
                    i += 1
                    continue
            i += 1
        df["close"] = closes
        return df

    def _clean_series(self, df):
        df = df.copy().sort_values("date").reset_index(drop=True)
        df = df.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
        if "currency" in df.columns:
            df["close"] = df["close"].astype(float) * df["fx_rate"].astype(float)
        return self._adjust_shifts(df)

    def compute_max_drawdown(self, prices):
        prices = pd.Series(prices, dtype=float)
        return float(((prices - prices.cummax()) / prices.cummax()).min())

    def compute_rolling_beta(self, asset_returns, index_returns, window=20):
        aligned = pd.DataFrame({"asset": asset_returns, "index": index_returns}).dropna()
        if len(aligned) <= window:
            return 0.0
        betas = []
        for end in range(window, len(aligned) + 1):
            w = aligned.iloc[end - window:end]
            var = w["index"].var(ddof=1)
            if var > 0:
                betas.append(w["asset"].cov(w["index"]) / var)
        return float(sum(betas) / len(betas)) if betas else 0.0

    def compute_equal_weight_portfolio_return(self, asset_dfs, rebalance_freq=20):
        prices = {}
        for ticker, df in asset_dfs.items():
            clean = self._clean_series(df)
            prices[ticker] = clean.set_index("date")["close"]
        frame = pd.DataFrame(prices).sort_index().dropna()  # inner join, no fill
        return self._portfolio_from_prices(frame, rebalance_freq), len(frame)

    @staticmethod
    def _portfolio_from_prices(frame, rebalance_freq=20):
        n = frame.shape[1]
        holdings = [1.0 / n] * n
        values = frame.to_numpy(dtype=float)
        for step in range(1, len(values)):
            for a in range(n):
                holdings[a] *= values[step][a] / values[step - 1][a]
            if step % rebalance_freq == 0:
                holdings = [sum(holdings) / n] * n
        return float(sum(holdings) - 1.0)

    # ------------------------------------------------------------------ tasks
    def _start(self):
        return datetime(2024, 1, 1) + timedelta(days=random.randint(0, 60))

    _BUNDLES = {
        "portfolio": [["reverse_split"], ["out_of_order"], ["currency"], ["unit_change"],
                      ["adj_close", "out_of_order"], ["reverse_split", "currency"],
                      ["out_of_order", "currency", "adj_close"], ["reverse_split", "adj_close"],
                      ["column_order", "out_of_order"], ["unit_change", "out_of_order"],
                      ["reverse_split", "out_of_order", "currency", "adj_close"],
                      ["column_order", "currency"]],
        "max_drawdown": [["unit_change"], ["reverse_split", "adj_close"], ["out_of_order"],
                         ["column_order"], ["missing"],
                         ["unit_change", "out_of_order"], ["duplicate", "column_order"],
                         ["reverse_split", "out_of_order"], ["adj_close", "column_order"],
                         ["unit_change", "missing", "out_of_order", "column_order"],
                         ["reverse_split", "unit_change"], ["duplicate", "out_of_order"]],
        "beta": [["currency"], ["reverse_split"], ["adj_close", "out_of_order"],
                 ["currency", "reverse_split", "adj_close"], ["unit_change", "out_of_order"],
                 ["column_order", "currency"], ["missing", "reverse_split"],
                 ["duplicate", "adj_close"],
                 ["currency", "out_of_order", "unit_change", "adj_close"],
                 ["reverse_split", "column_order"]],
        "dst": [["timezone"], ["timezone", "out_of_order"], ["timezone", "column_order"],
                ["timezone", "adj_close"], ["timezone", "currency"], ["timezone", "unit_change"],
                ["timezone", "duplicate"], ["timezone", "missing"],
                ["timezone", "out_of_order", "column_order", "adj_close"],
                ["timezone", "reverse_split"]],
        "volatility": [["unit_change"], ["reverse_split", "adj_close"], ["out_of_order"],
                       ["column_order", "missing"], ["duplicate"], ["adj_close", "column_order"],
                       ["unit_change", "out_of_order", "column_order"],
                       ["reverse_split", "out_of_order"], ["missing", "adj_close"],
                       ["unit_change", "duplicate"], ["reverse_split", "unit_change"]],
    }

    def _apply_series_bundle(self, df, bundle):
        """Apply a bundle to one series; shifts always precede shuffling."""
        problems = []
        if "duplicate" in bundle:
            df, p = self.inject_duplicate_row(df)
            problems.append(p)
        if "missing" in bundle:
            df, p = self.inject_missing_row(df)
            problems.append(p)
        if "unit_change" in bundle:
            df, p = self.inject_unit_change(df, random.randint(30, len(df) - 30))
            problems.append(p)
        if "reverse_split" in bundle:
            df, ps = self.inject_reverse_split(
                df, multiple=random.choice([3, 10]), add_adj_close="adj_close" in bundle)
            problems.extend(ps)
        elif "adj_close" in bundle:
            df, p = self.add_adj_close_conflict(df)
            problems.append(p)
        if "out_of_order" in bundle:
            df, p = self.inject_out_of_order_rows(df, 4)
            problems.append(p)
        return df, problems

    def _bundle(self, kind):
        # Cycle through a seeded shuffle of the bundle deck so a run of tasks
        # of one type covers the deck instead of possibly repeating one bundle.
        if not hasattr(self, "_decks"):
            self._decks = {}
        if kind not in self._decks:
            deck = list(self._BUNDLES[kind])
            random.shuffle(deck)
            self._decks[kind] = [deck, 0]
        deck, idx = self._decks[kind]
        self._decks[kind][1] = idx + 1
        return list(deck[idx % len(deck)])

    def generate_task_multi_asset_alignment(self):
        self.task_id += 1
        tickers = ["AAA_US", "BBB_EU", "CCC_US"]
        dfs = self.generate_clean_prices_multi_asset(tickers, self._start(), random.randint(80, 110))
        problems = []
        date_sets = [set(d["date"]) for d in dfs.values()]
        if len(set().union(*date_sets)) > len(set.intersection(*date_sets)):
            problems.append("calendar_misalignment_US_EU_holidays")
        bundle = self._bundle("portfolio")
        while problems and len(bundle) > 3:  # keep total key size <= 4 with calendar
            bundle = self._bundle("portfolio")
        if "currency" in bundle:
            t = random.choice(tickers)
            dfs[t] = self.add_currency_columns(dfs[t], "EUR")
            for other in tickers:
                if other != t:
                    dfs[other] = self.add_currency_columns(dfs[other], "USD")
            problems.append(f"currency_mismatch_{t}_EUR")
        # Price-level shifts must be injected before any row shuffling, so a
        # swapped row can never land on the wrong side of a split.
        if "unit_change" in bundle:
            t = random.choice(tickers)
            dfs[t], p = self.inject_unit_change(
                dfs[t], random.randint(30, len(dfs[t]) - 30))
            problems.append(p)
        if "reverse_split" in bundle:
            t = random.choice(tickers)
            dfs[t], ps = self.inject_reverse_split(dfs[t], multiple=random.choice([3, 10]))
            problems.extend(ps)
        if "adj_close" in bundle:
            t = random.choice(tickers)
            dfs[t], p = self.add_adj_close_conflict(dfs[t])
            problems.append(p)
        if "out_of_order" in bundle:
            t = random.choice(tickers)
            dfs[t], p = self.inject_out_of_order_rows(dfs[t], 3)
            problems.append(p)
        for t in tickers:
            dfs[t] = self.dedupe_stale_closes(dfs[t])
        combined = pd.concat(dfs.values(), ignore_index=True)
        if "column_order" in bundle:
            combined, p = self.make_column_order_csv(
                combined, random.randint(30, len(combined) - 30))
            problems.append(p)
        ret, n_days = self.compute_equal_weight_portfolio_return(dfs)
        expected = {"portfolio_return": round(ret, 6), "num_assets": len(tickers),
                    "num_aligned_days": n_days, "problems_found": sorted(problems)}
        analytic = """Tickers in the file: AAA_US, BBB_EU, CCC_US. Build one
close-price series per ticker, apply the general rules, align them, and
compute an equal-weight portfolio return: portfolio value starts at 1.0 on
the first aligned date with 1/3 of value held in each asset (no return is
earned on that first date). Holdings then drift with each asset's returns.
After applying returns on every 20th return observation (observations 20,
40, ... counting the first return, i.e. the move into the 2nd aligned date,
as observation 1), reset every holding to one third of the then-current
total value; rebalancing does not change that day's return.
portfolio_return = final portfolio value - 1.0."""
        task = {"id": f"task_h_{self.task_id:03d}", "type": "multi_asset_portfolio",
                "prompt": _fmt_prompt(analytic, "- portfolio_return: (6 decimals)\n"
                                      "- num_assets: number of tickers\n"
                                      "- num_aligned_days: dates kept by the inner join"),
                "data_file": f"task_h_{self.task_id:03d}.csv", "expected": expected}
        return task, combined

    def generate_task_max_drawdown(self):
        self.task_id += 1
        ticker = random.choice(["TECH_A", "ENERGY_B", "FINANCE_C"])
        df = self.generate_clean_prices_multi_asset([ticker], self._start(),
                                                    random.randint(100, 140))[ticker]
        bundle = self._bundle("max_drawdown")
        df, problems = self._apply_series_bundle(df, bundle)
        df = self.dedupe_stale_closes(df)
        payload = df
        if "column_order" in bundle:
            payload, p = self.make_column_order_csv(df, random.randint(30, len(df) - 30))
            problems.append(p)
        clean = self._clean_series(df)
        expected = {"max_drawdown": round(self.compute_max_drawdown(clean["close"]), 6),
                    "num_clean_days": len(clean), "problems_found": sorted(problems)}
        analytic = f"""Use the single ticker {ticker}. After the general rules, let
cummax_t be the maximum cleaned close up to and including t. The drawdown at
t is (close_t - cummax_t) / cummax_t, and max_drawdown is the minimum (most
negative) drawdown. num_clean_days is the number of data rows after parsing
and cleaning."""
        task = {"id": f"task_h_{self.task_id:03d}", "type": "max_drawdown",
                "prompt": _fmt_prompt(analytic, "- max_drawdown: (negative decimal, 6 decimals)\n"
                                      "- num_clean_days: integer"),
                "data_file": f"task_h_{self.task_id:03d}.csv", "expected": expected}
        return task, payload

    def generate_task_beta(self):
        self.task_id += 1
        tickers = ["ACME", "IDX"]
        dfs = self.generate_clean_prices_multi_asset(tickers, self._start(), random.randint(90, 120))
        asset, bench = dfs["ACME"], dfs["IDX"]
        problems = []
        bundle = self._bundle("beta")
        if "currency" in bundle:
            asset = self.add_currency_columns(asset, "EUR")
            bench = self.add_currency_columns(bench, "USD")
            problems.append("currency_mismatch_ACME_EUR")
        if "duplicate" in bundle:
            asset, p = self.inject_duplicate_row(asset)
            problems.append(p)
        if "missing" in bundle:
            bench, p = self.inject_missing_row(bench)
            problems.append(p)
        # Shifts before shuffling.
        if "unit_change" in bundle:
            asset, p = self.inject_unit_change(asset, random.randint(30, len(asset) - 30))
            problems.append(p)
        if "reverse_split" in bundle:
            asset, ps = self.inject_reverse_split(
                asset, multiple=random.choice([3, 10]),
                add_adj_close="adj_close" in bundle)
            problems.extend(ps)
        elif "adj_close" in bundle:
            asset, p = self.add_adj_close_conflict(asset)
            problems.append(p)
        if "out_of_order" in bundle:
            if random.choice([True, False]):
                asset, p = self.inject_out_of_order_rows(asset, 3)
            else:
                bench, p = self.inject_out_of_order_rows(bench, 3)
            problems.append(p)
        asset = self.dedupe_stale_closes(asset)
        bench = self.dedupe_stale_closes(bench)
        combined = pd.concat([asset, bench], ignore_index=True)
        if "column_order" in bundle:
            combined, p = self.make_column_order_csv(
                combined, random.randint(30, len(combined) - 30))
            problems.append(p)
        clean_a = self._clean_series(asset)
        clean_b = self._clean_series(bench)
        frame = pd.DataFrame({"asset": clean_a.set_index("date")["close"],
                              "index": clean_b.set_index("date")["close"]}).sort_index().dropna()
        rets = frame.pct_change().dropna()
        beta = self.compute_rolling_beta(rets["asset"], rets["index"], 20)
        expected = {"mean_beta": round(beta, 6), "num_aligned_days": len(frame),
                    "num_return_days": len(rets), "problems_found": sorted(problems)}
        analytic = """ACME is the asset and IDX is the benchmark. After the general
rules, align the two cleaned close series, compute simple daily returns for
each, and drop the first, undefined return. For each window of 20 consecutive
return observations, ending at return observations 20, 21, ..., N, compute
beta = cov(asset returns, benchmark returns, ddof=1) / var(benchmark returns,
ddof=1). mean_beta is the arithmetic mean of those window betas."""
        task = {"id": f"task_h_{self.task_id:03d}", "type": "rolling_beta",
                "prompt": _fmt_prompt(analytic, "- mean_beta: (6 decimals)\n"
                                      "- num_aligned_days: integer\n- num_return_days: integer"),
                "data_file": f"task_h_{self.task_id:03d}.csv", "expected": expected}
        return task, combined

    def generate_task_dst_intraday(self):
        self.task_id += 1
        # 2024-03-04 is the Monday before the 2024 US spring-forward.
        start = datetime(2024, 3, 4) + timedelta(days=random.randint(0, 3))
        df = self.generate_clean_prices_multi_asset(["INTR"], start, 12)["INTR"]
        bundle = self._bundle("dst")
        intra, p = self.inject_mixed_timezone_intraday(df, repeat_instant=True)
        problems = [p]
        if "duplicate" in bundle:
            intra, p = self.inject_duplicate_row(intra)
            problems.append(p)
        if "missing" in bundle:
            intra, p = self.inject_missing_row(intra)
            problems.append(p)
        if "currency" in bundle:
            rate, rates = 1.08, {}
            for stamp in intra["timestamp"]:
                day = pd.Timestamp(stamp).tz_convert(NY).date() \
                    if pd.Timestamp(stamp).tzinfo else pd.Timestamp(stamp).date()
                if day not in rates:
                    rate = max(0.5, rate * random.uniform(0.995, 1.005))
                    rates[day] = round(rate, 6)
            intra["currency"] = "EUR"
            intra["fx_rate"] = [rates[pd.Timestamp(s).tz_convert(NY).date()
                                    if pd.Timestamp(s).tzinfo else pd.Timestamp(s).date()]
                                for s in intra["timestamp"]]
            problems.append("currency_mismatch_INTR_EUR")
        # Shifts before shuffling.
        if "unit_change" in bundle:
            intra, p = self.inject_unit_change(intra, random.randint(20, len(intra) - 20))
            problems.append(p)
        if "reverse_split" in bundle:
            intra, ps = self.inject_reverse_split(
                intra, multiple=random.choice([3, 10]), add_adj_close="adj_close" in bundle)
            problems.extend(ps)
        elif "adj_close" in bundle:
            intra, p = self.add_adj_close_conflict(intra)
            problems.append(p)
        if "out_of_order" in bundle:
            intra, p = self.inject_out_of_order_rows(intra, 4)
            problems.append(p)
        intra = self.dedupe_stale_closes(intra, order_col="timestamp")
        payload = intra
        if "column_order" in bundle:
            payload, p = self.make_column_order_csv(intra, random.randint(20, len(intra) - 20))
            problems.append(p)
        daily = self._daily_closes_from_intraday(intra)
        rets = daily.pct_change().dropna()
        expected = {"mean_return": round(float(rets.mean()), 6),
                    "std_return": round(float(rets.std(ddof=1)), 6),
                    "annualized_volatility": round(float(rets.std(ddof=1) * 252 ** 0.5), 6),
                    "num_clean_days": len(daily), "problems_found": sorted(problems)}
        analytic = """The file holds intraday bars for INTR. Apply the general rules
to obtain one daily close per America/New_York calendar date, then compute
simple daily returns between consecutive daily closes. Report their mean,
their sample standard deviation (ddof=1), and the annualised volatility
std(daily returns, ddof=1) * sqrt(252). num_clean_days is the number of
daily closes."""
        task = {"id": f"task_h_{self.task_id:03d}", "type": "dst_intraday",
                "prompt": _fmt_prompt(analytic, "- mean_return: (6 decimals)\n"
                                      "- std_return: (6 decimals)\n"
                                      "- annualized_volatility: (6 decimals)\n"
                                      "- num_clean_days: integer"),
                "data_file": f"task_h_{self.task_id:03d}.csv", "expected": expected}
        return task, payload

    @classmethod
    def _daily_closes_from_intraday(cls, intra):
        seen = set()
        records = []
        has_fx = "currency" in intra.columns
        for _, row in intra.iterrows():
            instant = pd.Timestamp(row["timestamp"])
            if instant.tzinfo is None:
                instant = instant.tz_localize("UTC")
            instant = instant.tz_convert(NY)
            key = instant.isoformat()
            if key in seen:
                continue
            seen.add(key)
            close = float(row["close"]) * (float(row["fx_rate"]) if has_fx else 1.0)
            records.append((instant, close))
        records.sort(key=lambda r: r[0])
        adjusted = cls._adjust_shifts(pd.DataFrame(
            {"close": [c for _, c in records]}))["close"].tolist()
        daily = {}
        for (instant, _), close in zip(records, adjusted, strict=True):
            daily[instant.date()] = close
        return pd.Series([daily[d] for d in sorted(daily)], dtype=float)

    def generate_task_single_asset_volatility(self):
        self.task_id += 1
        ticker = random.choice(["TECH_A", "ENERGY_B", "FINANCE_C"])
        df = self.generate_clean_prices_multi_asset([ticker], self._start(),
                                                    random.randint(90, 120))[ticker]
        bundle = self._bundle("volatility")
        df, problems = self._apply_series_bundle(df, bundle)
        df = self.dedupe_stale_closes(df)
        payload = df
        if "column_order" in bundle:
            payload, p = self.make_column_order_csv(df, random.randint(30, len(df) - 30))
            problems.append(p)
        clean = self._clean_series(df)
        rets = clean["close"].pct_change().dropna()
        expected = {"annualized_volatility": round(float(rets.std(ddof=1) * 252 ** 0.5), 6),
                    "num_clean_days": len(clean), "problems_found": sorted(problems)}
        analytic = f"""Use the single ticker {ticker}. After the general rules, compute
simple daily returns between consecutive cleaned closes and report the
annualised volatility std(daily returns, ddof=1) * sqrt(252).
num_clean_days is the number of data rows after parsing and cleaning."""
        task = {"id": f"task_h_{self.task_id:03d}", "type": "single_asset_volatility",
                "prompt": _fmt_prompt(analytic, "- annualized_volatility: (6 decimals)\n"
                                      "- num_clean_days: integer"),
                "data_file": f"task_h_{self.task_id:03d}.csv", "expected": expected}
        return task, payload

    # ------------------------------------------------------------- generation
    def task_generators(self):
        return [self.generate_task_multi_asset_alignment, self.generate_task_max_drawdown,
                self.generate_task_beta, self.generate_task_dst_intraday,
                self.generate_task_single_asset_volatility]

    def generate_all_tasks(self, output_dir, num_tasks=25):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        answers_dir = output_dir / "answers"
        answers_dir.mkdir(exist_ok=True)
        tasks = []
        gens = self.task_generators()
        for i in range(num_tasks):
            task, payload = gens[i % len(gens)]()
            csv_path = output_dir / task["data_file"]
            if isinstance(payload, str):
                csv_path.write_text(payload)
            else:
                payload.to_csv(csv_path, index=False)
            tasks.append({k: v for k, v in task.items() if k != "expected"})
            with open(answers_dir / f"{task['id']}_answer.json", "w") as f:
                json.dump(task["expected"], f, indent=2)
        with open(output_dir / "tasks.json", "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"Generated {len(tasks)} hard-tier tasks in {output_dir}")
        return tasks
