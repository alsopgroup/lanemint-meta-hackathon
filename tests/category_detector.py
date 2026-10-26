"""Independent rule-based problem-category detector.

Written from the category definitions stated in the task prompts (both
tiers), not from the generator code. Given a task CSV it returns the set
of category codes whose definitions the data meets, so tests can check
that every answer key agrees with what a careful reader would report.
"""

import re
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tests.independent_solver import read_csv_any

NY = ZoneInfo("America/New_York")

# Market holiday calendars (month, day) referenced by the definitions:
# a ticker complete on its own calendar skips weekends and these days.
US_HOLIDAYS = {(1, 1), (7, 4), (12, 25), (11, 28)}
EU_HOLIDAYS = {(1, 1), (5, 1), (12, 25), (12, 26), (8, 15)}
KNOWN_COLS = {"date", "timestamp", "ticker", "open", "high", "low", "close",
              "adj_close", "volume", "currency", "fx_rate"}


def _offset_form(stamp):
    m = re.search(r"(Z|[+-]\d{2}:\d{2})$", stamp.strip())
    return m.group(1) if m else ""


def _instants(stamps):
    out = []
    for s in stamps:
        ts = pd.Timestamp(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        out.append(ts.tz_convert(NY))
    return out


def _holidays_for(ticker):
    if ticker.endswith("_EU"):
        return EU_HOLIDAYS
    if ticker.endswith("_US"):
        return US_HOLIDAYS
    return set()


def _ticker_has_missing_day(ticker, dates):
    """A weekday inside the ticker's own range is absent (holidays excluded)."""
    days = {d.date() for d in pd.to_datetime(dates)}
    if len(days) < 2:
        return False
    holidays = _holidays_for(ticker)
    day = min(days) + timedelta(days=1)
    while day < max(days):
        if day.weekday() < 5 and (day.month, day.day) not in holidays \
                and day not in days:
            return True
        day += timedelta(days=1)
    return False


def _shift_categories(closes):
    """bad_price / split / reverse_split / unit_change from a close series."""
    found = set()
    closes = [float(c) for c in closes]
    # One-day spikes first: >50% away from both neighbours, then reverting.
    masked = list(closes)
    for i in range(1, len(closes) - 1):
        prev, cur, nxt = closes[i - 1], closes[i], closes[i + 1]
        if prev > 0 and nxt > 0 and \
                (cur / prev < 0.5 or cur / prev > 1.5) and \
                (cur / nxt < 0.5 or cur / nxt > 1.5):
            found.add("bad_price")
            masked[i] = masked[i - 1]
    # Sustained level shifts on the spike-masked series.
    i = 1
    while i + 1 < len(masked):
        prev = masked[i - 1]
        if prev <= 0:
            i += 1
            continue
        ratio = masked[i] / prev
        # "Sustained" means the new level holds, not just for the next row
        # (a duplicated bad tick produces two identical spikes in a row).
        following = [masked[i + 1] / prev]
        if i + 2 < len(masked):
            following.append(masked[i + 2] / prev)
        sustained = min(following)
        if ratio > 1.8 and sustained > 1.8:
            if ratio > 50:
                found.add("unit_change")
                factor = 100.0
            else:
                found.add("reverse_split")
                factor = float(min((3, 10), key=lambda c: abs(ratio - c) / c))
            masked = masked[:i] + [c / factor for c in masked[i:]]
        elif 0.4 <= ratio <= 0.6 and all(0.4 <= v <= 0.6 for v in following):
            found.add("split")
            masked = masked[:i] + [c * 2.0 for c in masked[i:]]
        i += 1
    return found


def _series_by_ticker(frame, intraday):
    """(key, close) pairs per ticker, sorted by date/instant, FX-converted."""
    has_fx = "currency" in frame.columns
    series = {}
    for ticker, g in frame.groupby("ticker", sort=False):
        if intraday:
            pairs = sorted(zip(_instants(g["timestamp"].astype(str)),
                               g["close"], g.get("fx_rate", [1.0] * len(g)),
                               strict=True), key=lambda p: p[0])
            series[ticker] = [(instant, float(close) * (float(fx) if has_fx else 1.0))
                              for instant, close, fx in pairs]
        else:
            g = g.assign(_k=pd.to_datetime(g["date"])).sort_values(
                "_k", kind="stable")
            fx = g["fx_rate"] if has_fx else [1.0] * len(g)
            series[ticker] = [(d, float(c) * float(f)) for d, c, f in
                              zip(g["_k"], g["close"], fx, strict=True)]
    return series


def _first_per_key(pairs):
    seen, out = set(), []
    for key, close in pairs:
        if key not in seen:
            seen.add(key)
            out.append((key, close))
    return out


def detect_categories(csv_path):
    """Return the category codes whose definitions the CSV data meets."""
    raw = Path(csv_path).read_text()
    frame = read_csv_any(csv_path)
    found = set()

    headers = []
    for ln in raw.splitlines():
        cells = [c.strip() for c in ln.split(",")]
        if set(cells) <= KNOWN_COLS and ("date" in cells or "timestamp" in cells):
            headers.append(tuple(cells))
    if len(headers) >= 2 and len(set(headers)) >= 2:
        found.add("column_order")

    if "adj_close" in frame.columns:
        for adj, close in zip(frame["adj_close"], frame["close"], strict=True):
            if str(adj) not in ("", "nan") and abs(float(adj) - float(close)) > 1e-9:
                found.add("adj_close")
                break

    if "currency" in frame.columns and \
            set(frame["currency"].astype(str)) - {"USD"}:
        found.add("currency")

    intraday = "timestamp" in frame.columns
    if intraday:
        stamps = frame["timestamp"].astype(str)
        if len({_offset_form(s) for s in stamps}) > 1:
            found.add("timezone")
        if frame.duplicated(subset=["ticker", "timestamp"]).any():
            found.add("duplicate")
        per_day = {}
        for instant in _instants(stamps):
            per_day.setdefault(instant.date(), set()).add(instant)
        counts = [len(v) for v in per_day.values()]
        if counts and min(counts) < max(counts):
            found.add("missing")
        for _, g in frame.groupby("ticker", sort=False):
            seq = _instants(g["timestamp"].astype(str))
            if any(b < a for a, b in zip(seq, seq[1:], strict=False)):
                found.add("out_of_order")
                break
    else:
        if frame.duplicated(subset=["ticker", "date"]).any():
            found.add("duplicate")
        date_sets = []
        for ticker, g in frame.groupby("ticker", sort=False):
            if _ticker_has_missing_day(ticker, g["date"]):
                found.add("missing")
            date_sets.append(set(pd.to_datetime(g["date"])))
            seq = list(pd.to_datetime(g["date"]))
            if any(b < a for a, b in zip(seq, seq[1:], strict=False)):
                found.add("out_of_order")
        if len(date_sets) > 1 and "missing" not in found and \
                len(set().union(*date_sets)) > len(set.intersection(*date_sets)):
            found.add("calendar")

    for pairs in _series_by_ticker(frame, intraday).values():
        # Level shifts are judged on the first-row-per-date series. A bad
        # tick sitting on one copy of a duplicated date is additionally
        # visible in the raw rows, so bad_price is also checked there.
        found |= _shift_categories([c for _, c in _first_per_key(pairs)])
        found |= _shift_categories([c for _, c in pairs]) & {"bad_price"}
    return found
