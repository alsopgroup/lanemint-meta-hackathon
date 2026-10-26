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

POSSIBLE_ISSUES = """Possible issue categories in this benchmark (a file may contain any \
subset of these, including none of some): duplicate, missing, bad_price, split, \
reverse_split, unit_change, out_of_order, timezone, currency, adj_close, \
column_order, calendar. Report problems_found using these category codes where \
they apply, each followed by a short detail (for example "unit_change: ...")."""

GENERAL_RULES = """General consistency rules (apply in this order, whichever apply):
1. Parsing: the file is CSV. If a row's fields are themselves column names, that
   row is a new header; rows after it follow that header's column order until
   any later header row. Map every data row by its current header names.
2. Ordering: sort records ascending by ticker (if present), then by date or
   timestamp. Sorting is stable.
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
   compare each close with the previous close. If the ratio is > 50, prices
   from that row onward switch to cents: divide them by 100. Otherwise, if
   the ratio is > 1.8, it is a reverse split whose ratio is the nearest
   integer to that ratio: divide prices from that row onward by that integer.
   "Sustained" means the next available close divided by the close before the
   shift is also > 50 (unit change) or > 1.8 (reverse split). Do not round
   adjusted prices.
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
        "- problems_found: list of category codes with short details\n"
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
            for date in dates:
                price = max(0.01, price * (1 + random.gauss(0.0005, 0.02)))
                rows.append({
                    "date": date, "ticker": ticker,
                    "open": round(price * random.uniform(0.99, 1.01), 2),
                    "high": round(price * random.uniform(1.00, 1.02), 2),
                    "low": round(price * random.uniform(0.98, 1.00), 2),
                    "close": round(price, 2),
                    "volume": random.randint(100_000, 10_000_000),
                })
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

    def inject_unit_change(self, df, change_point):
        df = df.copy().reset_index(drop=True)
        df.loc[change_point:, PRICE_COLS] = df.loc[change_point:, PRICE_COLS] * 100
        date_str = df.iloc[change_point]["date"].strftime("%Y-%m-%d")
        return df, f"unit_change_{date_str}_dollars_to_cents"

    def inject_reverse_split(self, df, multiple=3, add_adj_close=False):
        """Exact integer reverse split: post-split prices x multiple, volume / multiple."""
        df = df.copy().reset_index(drop=True)
        idx = random.randint(20, len(df) - 20)
        df.loc[idx:, PRICE_COLS] = df.loc[idx:, PRICE_COLS] * multiple
        df.loc[idx:, "volume"] = (df.loc[idx:, "volume"] // multiple).astype(int)
        if add_adj_close:
            df["adj_close"] = df["close"].copy()
            half = idx // 2
            # Partly adjusted and therefore conflicting: only the first half
            # of the pre-split history is already on the post-split basis.
            df.loc[:half, "adj_close"] = df.loc[:half, "close"] * multiple
        date_str = df.iloc[idx]["date"].strftime("%Y-%m-%d")
        problems = [f"reverse_split_{date_str}_1-for-{multiple}"]
        if add_adj_close:
            problems.append(f"adj_close_conflict_{date_str}")
        return df, problems

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

    def inject_mixed_timezone_intraday(self, df):
        """Expand daily rows to intraday bars in mixed UTC / New-York forms.

        Prices vary bar to bar. One instant is repeated in a second textual
        form (same values); no price is repeated across distinct bars.
        """
        rows = []
        for _, row in df.iterrows():
            day = row["date"]
            closes = [row["close"] * 0.995, row["close"] * 1.004, row["close"]]
            for hour, close in zip([9, 12, 16], closes, strict=True):
                local = datetime(day.year, day.month, day.day, hour, tzinfo=NY)
                if (len(rows) // 3) % 2 == 0:
                    stamp = local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    stamp = local.isoformat()
                rows.append({
                    "timestamp": stamp, "ticker": row["ticker"],
                    "open": round(close * 0.999, 2), "high": round(close * 1.001, 2),
                    "low": round(close * 0.998, 2), "close": round(close, 2),
                    "volume": int(row["volume"] // 3),
                })
        out = pd.DataFrame(rows)
        # Repeat one instant in the other textual form, later in file order.
        src = out.iloc[10].copy()
        instant = pd.Timestamp(src["timestamp"])
        if str(src["timestamp"]).endswith("Z"):
            src["timestamp"] = instant.tz_convert(NY).isoformat()
        else:
            src["timestamp"] = instant.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        out = pd.concat([out, src.to_frame().T], ignore_index=True)
        return out, "timezone_dst_mixed_offsets_duplicate_instant"

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
                if cur / prev > 50 and sustained > 50:
                    factor = 100.0
                elif sustained > 1.8:
                    factor = float(round(cur / prev))
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

    def generate_task_multi_asset_alignment(self):
        self.task_id += 1
        tickers = ["AAA_US", "BBB_EU", "CCC_US"]
        dfs = self.generate_clean_prices_multi_asset(tickers, self._start(), random.randint(80, 110))
        problems = ["calendar_misalignment_US_EU_holidays"]
        t1 = random.choice(tickers)
        dfs[t1], p = self.inject_out_of_order_rows(dfs[t1], 3)
        problems.append(p)
        t2 = random.choice([t for t in tickers if t != t1])
        dfs[t2], ps = self.inject_reverse_split(dfs[t2], multiple=3)
        problems.extend(ps)
        combined = pd.concat(dfs.values(), ignore_index=True)
        ret, n_days = self.compute_equal_weight_portfolio_return(dfs)
        expected = {"portfolio_return": round(ret, 6), "num_assets": len(tickers),
                    "num_aligned_days": n_days, "problems_found": sorted(problems)}
        analytic = """Tickers in the file: AAA_US, BBB_EU, CCC_US (BBB_EU trades on a
European calendar, the others on a US calendar; that difference is part of
the data). Build one close-price series per ticker, apply the general rules,
align them, and compute an equal-weight portfolio return: portfolio value
starts at 1.0 on the first aligned date with 1/3 of value held in each asset
(no return is earned on that first date). Holdings then drift with each
asset's returns. After applying returns on every 20th return observation
(observations 20, 40, ... counting the first return, i.e. the move into the
2nd aligned date, as observation 1), reset every holding to one third of the
then-current total value; rebalancing does not change that day's return.
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
        problems = []
        df, p = self.inject_unit_change(df, random.randint(30, len(df) - 30))
        problems.append(p)
        df, p = self.inject_out_of_order_rows(df, 4)
        problems.append(p)
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
        asset = self.add_currency_columns(dfs["ACME"], "EUR")
        bench = self.add_currency_columns(dfs["IDX"], "USD")
        asset["adj_close"] = asset["close"]
        asset, ps = self.inject_reverse_split(asset, multiple=10, add_adj_close=True)
        bench["adj_close"] = bench["close"]
        combined = pd.concat([asset, bench], ignore_index=True)
        problems = ["currency_mismatch_ACME_EUR", *ps]
        clean_a = self._clean_series(asset)
        clean_b = self._clean_series(bench)
        frame = pd.DataFrame({"asset": clean_a.set_index("date")["close"],
                              "index": clean_b.set_index("date")["close"]}).sort_index().dropna()
        rets = frame.pct_change().dropna()
        beta = self.compute_rolling_beta(rets["asset"], rets["index"], 20)
        expected = {"mean_beta": round(beta, 6), "num_aligned_days": len(frame),
                    "num_return_days": len(rets), "problems_found": sorted(problems)}
        analytic = """ACME is the asset and IDX is the benchmark. After the general
rules (note in particular the currency conversion and which price column to
use), align the two cleaned close series, compute simple daily returns for
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
        intra, p = self.inject_mixed_timezone_intraday(df)
        problems = [p]
        intra, p = self.inject_out_of_order_rows(intra, 4)
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
        return task, intra

    @staticmethod
    def _daily_closes_from_intraday(intra):
        seen = set()
        records = []
        for _, row in intra.iterrows():
            instant = pd.Timestamp(row["timestamp"])
            if instant.tzinfo is None:
                instant = instant.tz_localize("UTC")
            instant = instant.tz_convert(NY)
            key = instant.isoformat()
            if key in seen:
                continue
            seen.add(key)
            records.append((instant, float(row["close"])))
        records.sort(key=lambda r: r[0])
        daily = {}
        for instant, close in records:
            daily[instant.date()] = close
        return pd.Series([daily[d] for d in sorted(daily)], dtype=float)

    def generate_task_volatility_column_order(self):
        self.task_id += 1
        ticker = random.choice(["TECH_A", "ENERGY_B", "FINANCE_C"])
        df = self.generate_clean_prices_multi_asset([ticker], self._start(),
                                                    random.randint(90, 120))[ticker]
        df, ps = self.inject_reverse_split(df, multiple=3, add_adj_close=True)
        problems = list(ps)
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
        task = {"id": f"task_h_{self.task_id:03d}", "type": "volatility_column_order",
                "prompt": _fmt_prompt(analytic, "- annualized_volatility: (6 decimals)\n"
                                      "- num_clean_days: integer"),
                "data_file": f"task_h_{self.task_id:03d}.csv", "expected": expected}
        return task, payload

    # ------------------------------------------------------------- generation
    def task_generators(self):
        return [self.generate_task_multi_asset_alignment, self.generate_task_max_drawdown,
                self.generate_task_beta, self.generate_task_dst_intraday,
                self.generate_task_volatility_column_order]

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
