"""Independent solver for frontier-tier tasks.

Written separately from src/frontier_tier.py: it reads only a task's
prompt text (for the stated window / rebalance / cost parameters), its
type, and its data file, and applies the rules stated in the frontier
prompts. It shares no code with the generator.
"""

import io
import re

import numpy as np
import pandas as pd

_MARKERS = {"VENDOR_A", "VENDOR_B", "PRICES", "SPLIT_TABLE"}


def read_blocks(path):
    """Split a frontier data file into {block_name: DataFrame}."""
    lines = open(path).read().splitlines()
    blocks, name, buf = {}, "", []
    for line in lines:
        if line.strip() in _MARKERS:
            if buf:
                blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))
            name, buf = line.strip(), []
        else:
            buf.append(line)
    if buf:
        blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))
    return blocks


def _bad_mask(closes):
    bad = [False] * len(closes)
    for i in range(1, len(closes) - 1):
        p, c, n = closes[i - 1], closes[i], closes[i + 1]
        if p > 0 and n > 0 and (c / p < 0.5 or c / p > 1.5) \
                and (c / n < 0.5 or c / n > 1.5):
            bad[i] = True
    return bad


def clean(df):
    """One ticker: sort, dedupe (keep first in file order), drop bad ticks."""
    ordered = df.assign(_o=range(len(df))).sort_values(
        ["date", "_o"], kind="stable").drop_duplicates(subset=["date"], keep="first")
    closes = [float(c) for c in ordered["close"]]
    bad = _bad_mask(closes)
    return pd.Series([c for c, b in zip(closes, bad, strict=True) if not b],
                     index=[str(d) for d, b in
                            zip(ordered["date"], bad, strict=True) if not b],
                     dtype=float)


def _detect(frame):
    """duplicate / missing / bad_price / out_of_order in one daily block."""
    found = set()
    if frame.duplicated(subset=["ticker", "date"]).any():
        found.add("duplicate")
    for _, g in frame.groupby("ticker", sort=False):
        dates = [str(d) for d in g["date"]]
        if dates != sorted(dates):
            found.add("out_of_order")
        uniq = sorted(set(dates))
        days = pd.bdate_range(uniq[0], uniq[-1]).strftime("%Y-%m-%d")
        if set(days) - set(uniq):
            found.add("missing")
        ordered = g.assign(_o=range(len(g))).sort_values(
            ["date", "_o"], kind="stable").drop_duplicates(subset=["date"])
        if any(_bad_mask([float(c) for c in ordered["close"]])):
            found.add("bad_price")
    return found


def _aligned(frames):
    return pd.DataFrame({t: clean(g) for t, g in frames.items()}
                        ).sort_index().dropna()


def _frames(block):
    frames = {}
    for ticker, group in block.groupby("ticker", sort=False):
        frames[ticker] = group
    return frames


def cleaned_frame(task, csv_path):
    """The cleaned, adjusted, aligned close frame a task's math uses.

    F1/F4: cleaned closes aligned on the inner join. F2: trusted series
    aligned. F3: split-adjusted AST with MKT/SEC, aligned. Tests scan
    this frame for repeated consecutive closes.
    """
    kind = task["type"]
    blocks = read_blocks(csv_path)
    if kind in ("frontier_rolling_risk", "frontier_portfolio_pipeline"):
        block = blocks[""] if "" in blocks else blocks["PRICES"]
        return _aligned(_frames(block))
    if kind == "frontier_vendor_reconciliation":
        a_frames, b_frames = _frames(blocks["VENDOR_A"]), _frames(blocks["VENDOR_B"])
        trusted = {}
        for ticker in sorted(set(a_frames) | set(b_frames)):
            a = clean(a_frames[ticker]) if ticker in a_frames \
                else pd.Series(dtype=float)
            b = clean(b_frames[ticker].assign(
                close=b_frames[ticker]["close"] / 100.0)) \
                if ticker in b_frames else pd.Series(dtype=float)
            union = sorted(set(a.index) | set(b.index))
            trusted[ticker] = pd.Series(
                [a[d] if d in a.index else b[d] for d in union],
                index=union, dtype=float)
        return pd.DataFrame(trusted).sort_index().dropna()
    if kind == "frontier_factor_exposure":
        frames = _frames(blocks["PRICES"])
        ast = clean(frames["AST"])
        for _, row in blocks["SPLIT_TABLE"].iterrows():
            m = re.fullmatch(r"(\d+(?:\.\d+)?)-for-(\d+(?:\.\d+)?)",
                             str(row["ratio"]))
            ast.loc[ast.index < str(row["date"])] = \
                ast.loc[ast.index < str(row["date"])] * \
                (float(m.group(2)) / float(m.group(1)))
        return pd.DataFrame({"AST": ast, "MKT": clean(frames["MKT"]),
                             "SEC": clean(frames["SEC"])}).sort_index().dropna()
    raise ValueError(f"unknown frontier task type {kind}")


def solve(task, csv_path):
    kind = task["type"]
    prompt = task.get("prompt", "")
    blocks = read_blocks(csv_path)

    if kind == "frontier_rolling_risk":
        block = blocks[""] if "" in blocks else blocks["PRICES"]
        problems = _detect(block)
        prices = _aligned(_frames(block))
        assets = sorted(t for t in prices.columns if t != "IDX")
        window = int(re.search(r"window of\s+(\d+)", prompt).group(1))
        rets = prices.pct_change().dropna()
        cov = rets.rolling(window).cov(rets["IDX"])
        var = rets["IDX"].rolling(window).var(ddof=1)
        betas = {a: round(float((cov[a] / var).mean()), 6) for a in assets}
        # First maximum in (window-end date, asset order) order is the
        # prompt's tie break: earliest date, then asset ticker order.
        corr = rets.rolling(window).corr(rets["IDX"])[assets].dropna()
        pos = int(np.argmax(corr.to_numpy()))
        wi, ai = np.unravel_index(pos, corr.shape)
        ew = rets[assets].mean(axis=1)
        return {"problems": sorted(problems), "mean_betas": betas,
                "peak_correlation": round(float(corr.to_numpy()[wi, ai]), 6),
                "peak_correlation_asset": assets[ai],
                "peak_correlation_date": str(corr.index[wi]),
                "final_window_volatility": round(
                    float(ew.iloc[-window:].std(ddof=1) * np.sqrt(252)), 6),
                "num_assets": len(assets), "num_aligned_days": len(prices)}

    if kind == "frontier_vendor_reconciliation":
        a_block, b_block = blocks["VENDOR_A"], blocks["VENDOR_B"]
        # Missing is judged on the union of both vendors (added below),
        # never on one block alone.
        problems = (_detect(a_block) | _detect(b_block)) - {"missing"}
        a_frames, b_frames = _frames(a_block), _frames(b_block)
        trusted, conflicts = {}, []
        for ticker in sorted(set(a_frames) | set(b_frames)):
            a = clean(a_frames[ticker]) if ticker in a_frames else pd.Series(dtype=float)
            b_raw = b_frames[ticker].assign(
                close=b_frames[ticker]["close"] / 100.0) \
                if ticker in b_frames else None
            b = clean(b_raw) if b_raw is not None else pd.Series(dtype=float)
            union = sorted(set(a.index) | set(b.index))
            days = pd.bdate_range(union[0], union[-1]).strftime("%Y-%m-%d") \
                if union else []
            if set(days) - set(union):
                problems.add("missing")
            trusted[ticker] = pd.Series(
                [a[d] if d in a.index else b[d] for d in union],
                index=union, dtype=float)
            for d in sorted(set(a.index) & set(b.index)):
                if abs(float(a[d]) - float(b[d])) > 0.01:
                    conflicts.append(d)
        if conflicts:
            problems.add("source_conflict")
        aligned = pd.DataFrame(trusted).sort_index().dropna()
        rel = aligned.iloc[-1] / aligned.iloc[0]
        return {"problems": sorted(problems),
                "mismatch_count": len(conflicts),
                "first_conflict_date": min(conflicts) if conflicts else None,
                "reconciled_total_return": round(float(rel.mean() - 1.0), 6),
                "num_tickers": len(trusted), "num_aligned_days": len(aligned)}

    if kind == "frontier_factor_exposure":
        block, table = blocks["PRICES"], blocks["SPLIT_TABLE"]
        problems = _detect(block)
        events = []
        for _, row in table.iterrows():
            m = re.fullmatch(r"(\d+(?:\.\d+)?)-for-(\d+(?:\.\d+)?)", str(row["ratio"]))
            a, b = float(m.group(1)), float(m.group(2))
            events.append((str(row["date"]), a, b))
            problems.add("split" if a > b else "reverse_split")
        frames = _frames(block)
        ast = clean(frames["AST"])
        for d, a, b in events:
            ast.loc[ast.index < d] = ast.loc[ast.index < d] * (b / a)
        aligned = pd.DataFrame({"AST": ast, "MKT": clean(frames["MKT"]),
                                "SEC": clean(frames["SEC"])}).sort_index().dropna()
        rets = aligned.pct_change().dropna()
        x = np.column_stack([np.ones(len(rets)), rets["MKT"], rets["SEC"]])
        coef, *_ = np.linalg.lstsq(x, rets["AST"].to_numpy(), rcond=None)
        resid = rets["AST"].to_numpy() - x @ coef
        ss_tot = float(((rets["AST"] - rets["AST"].mean()) ** 2).sum())
        return {"problems": sorted(problems),
                "alpha": round(float(coef[0]), 6),
                "beta_market": round(float(coef[1]), 6),
                "beta_sector": round(float(coef[2]), 6),
                "r_squared": round(1.0 - float((resid ** 2).sum()) / ss_tot, 6),
                "num_return_days": len(rets), "num_splits": len(events)}

    if kind == "frontier_portfolio_pipeline":
        block = blocks[""] if "" in blocks else blocks["PRICES"]
        problems = _detect(block)
        prices = _aligned(_frames(block))
        every = int(re.search(r"observations\s+(\d+),", prompt).group(1))
        bps = float(re.search(r"cost of\s+(\d+(?:\.\d+)?)\s+basis points", prompt).group(1))
        vals = prices.to_numpy(dtype=float)
        n = prices.shape[1]
        holdings = np.full(n, 1.0 / n)
        values, first, count = [1.0], None, 0
        for step in range(1, len(vals)):
            holdings = holdings * vals[step] / vals[step - 1]
            total = float(holdings.sum())
            if step % every == 0 and step < len(vals) - 1:
                total *= 1.0 - bps / 10000.0
                holdings = np.full(n, total / n)
                count += 1
                if first is None:
                    first = total
            values.append(total)
        series = pd.Series(values, dtype=float)
        drawdown = (series - series.cummax()) / series.cummax()
        return {"problems": sorted(problems),
                "num_aligned_days": len(prices), "num_assets": n,
                "num_rebalances": count,
                "first_rebalance_value": round(float(first), 6),
                "final_return": round(float(values[-1] - 1.0), 6),
                "max_drawdown": round(float(drawdown.min()), 6)}

    raise ValueError(f"unknown frontier task type {kind}")
