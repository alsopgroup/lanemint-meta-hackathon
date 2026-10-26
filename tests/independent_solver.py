"""Independent solver for hard-tier tasks.

Written separately from src/hard_tier.py: it reads only a task's prompt-side
information (its type) and its CSV file, and applies the rules stated in the
hard-tier prompts. It shares no code with the reference cleaner.
"""

import csv
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
KNOWN_COLS = {"date", "timestamp", "ticker", "open", "high", "low", "close",
              "adj_close", "volume", "currency", "fx_rate"}


def read_csv_any(path):
    header = None
    records = []
    with open(path, newline="") as f:
        for raw in csv.reader(f):
            if not raw or all(not c.strip() for c in raw):
                continue
            cells = [c.strip() for c in raw]
            if set(cells) <= KNOWN_COLS and ("date" in cells or "timestamp" in cells):
                header = cells
                continue
            records.append(dict(zip(header, cells, strict=False)))
    return pd.DataFrame(records)


def clean_close_series(frame):
    """Sort, convert currency, then fix sustained level shifts. Returns closes by date."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    closes = [float(v) for v in frame["close"]]
    if "currency" in frame.columns:
        closes = [c * float(r) for c, r in zip(closes, frame["fx_rate"], strict=True)]
    i = 1
    while i < len(closes):
        if closes[i - 1] > 0:
            ratio = closes[i] / closes[i - 1]
            if ratio > 1.8 and i + 1 < len(closes) and closes[i + 1] / closes[i - 1] > 1.8:
                if ratio > 50:
                    divisor = 100.0
                else:
                    divisor = float(min((3, 10), key=lambda c: abs(ratio - c) / c))
                if divisor != 1.0:
                    closes = closes[:i] + [c / divisor for c in closes[i:]]
        i += 1
    return pd.Series(closes, index=frame["date"].values)


def solve(task, csv_path):
    kind = task["type"]
    if kind == "dst_intraday":
        frame = read_csv_any(csv_path)
        seen, bars = set(), []
        for stamp, close in zip(frame["timestamp"], frame["close"], strict=True):
            instant = pd.Timestamp(stamp)
            if instant.tzinfo is None:
                instant = instant.tz_localize("UTC")
            instant = instant.tz_convert(NY)
            if instant in seen:
                continue
            seen.add(instant)
            bars.append((instant, float(close)))
        bars.sort(key=lambda b: b[0])
        daily = {}
        for instant, close in bars:
            daily[instant.date()] = close
        closes = pd.Series([daily[d] for d in sorted(daily)], dtype=float)
        rets = closes.pct_change().dropna()
        return {"mean_return": round(float(rets.mean()), 6),
                "std_return": round(float(rets.std(ddof=1)), 6),
                "annualized_volatility": round(float(rets.std(ddof=1) * 252 ** 0.5), 6),
                "num_clean_days": len(closes)}

    frame = read_csv_any(csv_path)
    if kind == "max_drawdown":
        closes = clean_close_series(frame)
        peak = closes.cummax()
        return {"max_drawdown": round(float(((closes - peak) / peak).min()), 6),
                "num_clean_days": len(closes)}

    if kind == "single_asset_volatility":
        closes = clean_close_series(frame)
        rets = closes.pct_change().dropna()
        return {"annualized_volatility": round(float(rets.std(ddof=1) * 252 ** 0.5), 6),
                "num_clean_days": len(closes)}

    series = {}
    for ticker, group in frame.groupby("ticker"):
        series[ticker] = clean_close_series(group)
    prices = pd.DataFrame(series).sort_index().dropna()

    if kind == "multi_asset_portfolio":
        n = prices.shape[1]
        holdings = [1.0 / n] * n
        vals = prices.to_numpy(dtype=float)
        for step in range(1, len(vals)):
            holdings = [h * vals[step][a] / vals[step - 1][a] for a, h in enumerate(holdings)]
            if step % 20 == 0:
                holdings = [sum(holdings) / n] * n
        return {"portfolio_return": round(float(sum(holdings) - 1.0), 6),
                "num_assets": n, "num_aligned_days": len(prices)}

    if kind == "rolling_beta":
        rets = prices.pct_change().dropna()
        asset, bench = rets["ACME"], rets["IDX"]
        betas = []
        for end in range(20, len(rets) + 1):
            a = asset.iloc[end - 20:end]
            b = bench.iloc[end - 20:end]
            betas.append(a.cov(b) / b.var(ddof=1))
        return {"mean_beta": round(float(sum(betas) / len(betas)), 6),
                "num_aligned_days": len(prices), "num_return_days": len(rets)}

    raise ValueError(f"unknown hard task type {kind}")
