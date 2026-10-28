"""Deterministic worked-solution traces for benchmark tasks.

A trace is the step-by-step computation behind a task's answer key:
the cleaning log (rows read, duplicates and bad ticks dropped per
ticker), the aligned counts, the per-window / per-step intermediates
and the final values. It is recomputed from the task's own data file
with the rules stated in the prompts (the same computations the
generators use to build the key), so a trace only ever contains its
own task's numbers. Frontier families F1-F4 get full intermediates;
other task types get a compact parse-and-report trace.

Fireworks SFT accepts thinking traces as ``reasoning_content`` on an
assistant message (docs.fireworks.ai/fine-tuning/supervised-fine-tuning),
so the training rows keep the final JSON in ``content`` and put the
steps in ``reasoning_content``.
"""

import io
import json
import re

import numpy as np
import pandas as pd

_MARKERS = {"VENDOR_A", "VENDOR_B", "PRICES", "SPLIT_TABLE"}

_encoding = None


def count_tokens(text: str) -> int:
    """Token count for cap/validation purposes: tiktoken cl100k_base
    when available, else a labelled chars-per-token approximation."""
    global _encoding
    try:
        import tiktoken
        if _encoding is None:
            _encoding = tiktoken.get_encoding("cl100k_base")
        return len(_encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _read_blocks(text: str) -> dict:
    blocks, name, buf = {}, "", []
    for line in text.splitlines():
        if line.strip() in _MARKERS:
            if buf:
                blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))
            name, buf = line.strip(), []
        else:
            buf.append(line)
    if buf:
        blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))
    return blocks


def _bad_mask(closes) -> list:
    bad = [False] * len(closes)
    for i in range(1, len(closes) - 1):
        p, c, n = closes[i - 1], closes[i], closes[i + 1]
        if p > 0 and n > 0 and (c / p < 0.5 or c / p > 1.5) \
                and (c / n < 0.5 or c / n > 1.5):
            bad[i] = True
    return bad


def _clean(df):
    """Clean one ticker's frame; return (close Series, log dict)."""
    ordered = df.assign(_o=range(len(df))).sort_values(
        ["date", "_o"], kind="stable")
    deduped = ordered.drop_duplicates(subset=["date"], keep="first")
    closes = [float(c) for c in deduped["close"]]
    bad = _bad_mask(closes)
    series = pd.Series(
        [c for c, b in zip(closes, bad, strict=True) if not b],
        index=[str(d) for d, b in
               zip(deduped["date"], bad, strict=True) if not b],
        dtype=float)
    log = {"rows": len(df), "duplicates_dropped": len(df) - len(deduped),
           "bad_ticks_dropped": sum(bad), "kept": len(series)}
    return series, log


def _detect(frame) -> set:
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


def _frames(block) -> dict:
    frames = {}
    for ticker, group in block.groupby("ticker", sort=False):
        frames[ticker] = group
    return frames


def _aligned(frames, logs=None):
    cleaned = {}
    for ticker, frame in frames.items():
        series, log = _clean(frame)
        cleaned[ticker] = series
        if logs is not None:
            logs[ticker] = log
    return pd.DataFrame(cleaned).sort_index().dropna()


def _f(x) -> str:
    return f"{float(x):.6f}"


def _cap_steps(steps: str, max_tokens: int) -> str:
    """Shrink the steps to the token cap; the final JSON is separate
    and is never truncated."""
    if max_tokens is None or count_tokens(steps) <= max_tokens:
        return steps
    marker = "\n[steps truncated to fit the trace token cap]"
    while count_tokens(steps) > max_tokens and len(steps) > len(marker) + 8:
        steps = steps[: max(0, int(len(steps) * 0.8))]
        steps = steps.rsplit("\n", 1)[0] + marker
    while count_tokens(steps) > max_tokens and steps:
        steps = steps[: int(len(steps) * 0.8)]
    return steps


def _trace_f1(task, data, expected):
    blocks = _read_blocks(data)
    block = blocks[""] if "" in blocks else blocks["PRICES"]
    logs: dict = {}
    prices = _aligned(_frames(block), logs)
    assets = sorted(t for t in prices.columns if t != "IDX")
    window = int(re.search(r"windows of\s+(\d+) consecutive",
                           task["prompt"]).group(1))
    target = expected.get("target_asset") or re.search(
        r"target asset is (\w+)", task["prompt"]).group(1)
    n_win = sum(1 for k in expected if re.fullmatch(r"window_beta_\d+", k))
    rets = prices.pct_change().dropna()
    win_blocks = [rets.iloc[-(n_win - i) * window:-(n_win - i - 1) * window
                            or None] for i in range(n_win)]
    ew = rets[assets].mean(axis=1)
    lines = ["Step 1 - parse and clean each ticker "
             "(sort by date, drop duplicate dates keeping the first, "
             "drop one-day bad ticks):"]
    for ticker in sorted(logs):
        log = logs[ticker]
        lines.append(
            f"  {ticker}: {log['rows']} rows, "
            f"{log['duplicates_dropped']} duplicate(s) dropped, "
            f"{log['bad_ticks_dropped']} bad tick(s) dropped, "
            f"{log['kept']} closes kept")
    lines.append(
        f"Step 2 - align on the inner join: {len(prices)} aligned days "
        f"({prices.index[0]} to {prices.index[-1]}), "
        f"{len(rets)} simple returns per ticker")
    lines.append(
        f"Step 3 - problems detected: "
        f"{', '.join(sorted(_detect(block))) or 'none'}")
    lines.append(
        f"Step 4 - target asset {target}: beta vs IDX in the last "
        f"{n_win} non-overlapping {window}-return windows "
        f"(cov/var, ddof=1, simple returns):")
    answer = {"problems": sorted(_detect(block))}
    for i, blk in enumerate(win_blocks, 1):
        cov = float(blk[target].cov(blk["IDX"]))
        var = float(blk["IDX"].var(ddof=1))
        beta = cov / var
        answer[f"window_beta_{i}"] = round(beta, 6)
        lines.append(
            f"  window {i} (returns {len(rets) - (n_win - i + 1) * window + 1}"
            f" to {len(rets) - (n_win - i) * window}): cov={_f(cov)}, "
            f"var={_f(var)}, beta={_f(beta)}")
    vol = float(ew.iloc[-window:].std(ddof=1) * np.sqrt(252))
    answer.update({
        "target_asset": target,
        "final_window_volatility": round(vol, 6),
        "num_assets": len(assets), "num_aligned_days": len(prices)})
    lines.append(
        f"Step 5 - equal-weight portfolio (mean of {assets} returns): "
        f"final-window volatility = std(last {window} EW returns, "
        f"ddof=1) * sqrt(252) = {_f(vol)}")
    return lines, answer


def _trace_f2(task, data, expected):
    blocks = _read_blocks(data)
    a_block, b_block = blocks["VENDOR_A"], blocks["VENDOR_B"]
    problems = (_detect(a_block) | _detect(b_block)) - {"missing"}
    a_frames, b_frames = _frames(a_block), _frames(b_block)
    lines = ["Step 1 - parse both vendor blocks; convert Vendor B from "
             "cents to dollars (divide by 100); clean each series "
             "(dedupe keep-first, drop bad ticks):"]
    trusted, conflicts = {}, []
    for ticker in sorted(set(a_frames) | set(b_frames)):
        a, log_a = _clean(a_frames[ticker]) if ticker in a_frames \
            else (pd.Series(dtype=float),
                  {"rows": 0, "duplicates_dropped": 0,
                   "bad_ticks_dropped": 0, "kept": 0})
        b, log_b = (pd.Series(dtype=float),
                    {"rows": 0, "duplicates_dropped": 0,
                     "bad_ticks_dropped": 0, "kept": 0})
        if ticker in b_frames:
            b, log_b = _clean(b_frames[ticker].assign(
                close=b_frames[ticker]["close"] / 100.0))
        union = sorted(set(a.index) | set(b.index))
        days = pd.bdate_range(union[0], union[-1]).strftime("%Y-%m-%d") \
            if union else []
        if set(days) - set(union):
            problems.add("missing")
        trusted[ticker] = pd.Series(
            [a[d] if d in a.index else b[d] for d in union],
            index=union, dtype=float)
        pair_conflicts = [
            d for d in sorted(set(a.index) & set(b.index))
            if abs(round(float(a[d]) * 100)
                   - round(float(b[d]) * 100)) > 1]
        conflicts.extend(pair_conflicts)
        lines.append(
            f"  {ticker}: A kept {log_a['kept']}/{log_a['rows']}, "
            f"B kept {log_b['kept']}/{log_b['rows']} closes, "
            f"{len(pair_conflicts)} conflicting date(s)")
    if conflicts:
        problems.add("source_conflict")
    aligned = pd.DataFrame(trusted).sort_index().dropna()
    rel = aligned.iloc[-1] / aligned.iloc[0]
    value = (aligned / aligned.iloc[0]).mean(axis=1)
    drawdown = (value - value.cummax()) / value.cummax()
    total_return = float(rel.mean() - 1.0)
    max_dd = float(drawdown.min())
    lines.append(
        f"Step 2 - trusted series (Vendor A where present, B fills "
        f"gaps): {len(aligned)} aligned days ({aligned.index[0]} to "
        f"{aligned.index[-1]}); per-ticker total returns: "
        + ", ".join(f"{t}={_f(rel[t] - 1.0)}" for t in aligned.columns))
    lines.append(
        f"Step 3 - conflicts: {len(conflicts)} mismatching close(s) "
        f"(> $0.01 after cent conversion), first on "
        f"{min(conflicts) if conflicts else 'n/a'}")
    lines.append(
        f"Step 4 - equal-weight portfolio anchored at 1.0: final value "
        f"{_f(value.iloc[-1])}, reconciled_total_return="
        f"{_f(total_return)}; worst drawdown {_f(max_dd)} on "
        f"{drawdown.idxmin()} over every daily value")
    lines.append(
        f"Step 5 - problems detected: {', '.join(sorted(problems))}")
    answer = {"problems": sorted(problems),
              "mismatch_count": len(conflicts),
              "first_conflict_date": min(conflicts) if conflicts else None,
              "reconciled_total_return": round(total_return, 6),
              "reconciled_max_drawdown": round(max_dd, 6),
              "num_tickers": len(trusted),
              "num_aligned_days": len(aligned)}
    return lines, answer


def _trace_f3(task, data, expected):
    blocks = _read_blocks(data)
    block, table = blocks["PRICES"], blocks["SPLIT_TABLE"]
    problems = _detect(block)
    frames = _frames(block)
    events = []
    lines = ["Step 1 - read the split table and back-adjust AST closes "
             "(multiply every close before the event date by B/A for an "
             "A-for-B split):"]
    for _, row in table.iterrows():
        m = re.fullmatch(r"(\d+(?:\.\d+)?)-for-(\d+(?:\.\d+)?)",
                         str(row["ratio"]))
        a, b = float(m.group(1)), float(m.group(2))
        events.append((str(row["date"]), a, b))
        problems.add("split" if a > b else "reverse_split")
        lines.append(
            f"  {row['date']}: {row['ratio']} -> factor {_f(b / a)} "
            f"applied to AST closes before {row['date']}")
    ast = _clean(frames["AST"])[0]
    for d, a, b in events:
        ast.loc[ast.index < d] = ast.loc[ast.index < d] * (b / a)
    aligned = pd.DataFrame({"AST": ast, "MKT": _clean(frames["MKT"])[0],
                            "SEC": _clean(frames["SEC"])[0]}
                           ).sort_index().dropna()
    rets = aligned.pct_change().dropna()
    x = np.column_stack([np.ones(len(rets)), rets["MKT"], rets["SEC"]])
    coef, *_ = np.linalg.lstsq(x, rets["AST"].to_numpy(), rcond=None)
    resid = rets["AST"].to_numpy() - x @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((rets["AST"] - rets["AST"].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    lines.append(
        f"Step 2 - clean and align AST/MKT/SEC: {len(aligned)} aligned "
        f"days, {len(rets)} simple returns; problems detected: "
        f"{', '.join(sorted(problems))}")
    lines.append(
        f"Step 3 - OLS of AST returns on a constant, MKT and SEC "
        f"returns: alpha={_f(coef[0])}, beta_market={_f(coef[1])}, "
        f"beta_sector={_f(coef[2])}")
    lines.append(
        f"Step 4 - fit: SS_res={_f(ss_res)}, SS_tot={_f(ss_tot)}, "
        f"r_squared = 1 - SS_res/SS_tot = {_f(r2)}; "
        f"residual_std={_f(float(np.std(resid, ddof=1)))}")
    answer = {"problems": sorted(problems),
              "alpha": round(float(coef[0]), 6),
              "beta_market": round(float(coef[1]), 6),
              "beta_sector": round(float(coef[2]), 6),
              "r_squared": round(r2, 6),
              "num_return_days": len(rets), "num_splits": len(events)}
    if "residual_std" in expected:
        answer["residual_std"] = round(float(np.std(resid, ddof=1)), 6)
    return lines, answer


def _trace_f4(task, data, expected):
    blocks = _read_blocks(data)
    block = blocks[""] if "" in blocks else blocks["PRICES"]
    logs: dict = {}
    prices = _aligned(_frames(block), logs)
    every = int(re.search(r"observations\s+(\d+),", task["prompt"]).group(1))
    bps = float(re.search(r"cost of\s+(\d+(?:\.\d+)?)\s+basis points",
                          task["prompt"]).group(1))
    vals = prices.to_numpy(dtype=float)
    n = prices.shape[1]
    holdings = np.full(n, 1.0 / n)
    values, first, count = [1.0], None, 0
    lines = ["Step 1 - parse and clean each ticker "
             "(dedupe keep-first, drop bad ticks):"]
    for ticker in sorted(logs):
        log = logs[ticker]
        lines.append(
            f"  {ticker}: {log['rows']} rows, "
            f"{log['duplicates_dropped']} duplicate(s), "
            f"{log['bad_ticks_dropped']} bad tick(s) dropped, "
            f"{log['kept']} closes kept")
    lines.append(
        f"Step 2 - align on the inner join: {len(prices)} aligned days, "
        f"{n} assets, equal-weight start at 1.0; drift daily, "
        f"rebalance every {every} observations at {bps:g} bps cost")
    for step in range(1, len(vals)):
        holdings = holdings * vals[step] / vals[step - 1]
        total = float(holdings.sum())
        if step % every == 0 and step < len(vals) - 1:
            after = total * (1.0 - bps / 10000.0)
            lines.append(
                f"  step {step} ({prices.index[step]}): value before "
                f"cost {_f(total)}, after cost {_f(after)}; reset to "
                f"equal weights")
            total = after
            holdings = np.full(n, total / n)
            count += 1
            if first is None:
                first = total
        values.append(total)
    series = pd.Series(values, dtype=float)
    drawdown = (series - series.cummax()) / series.cummax()
    lines.append(
        f"Step 3 - daily path: {count} rebalance(s), first rebalance "
        f"value {_f(first)}, final value {_f(values[-1])}, "
        f"final_return={_f(values[-1] - 1.0)}")
    lines.append(
        f"Step 4 - max drawdown over every daily value: "
        f"{_f(float(drawdown.min()))}; problems detected: "
        f"{', '.join(sorted(_detect(block))) or 'none'}")
    answer = {"problems": sorted(_detect(block)),
              "num_aligned_days": len(prices), "num_assets": n,
              "num_rebalances": count,
              "first_rebalance_value": round(float(first), 6),
              "final_return": round(float(values[-1] - 1.0), 6),
              "max_drawdown": round(float(drawdown.min()), 6)}
    return lines, answer


_BUILDERS = {
    "frontier_rolling_risk": _trace_f1,
    "frontier_vendor_reconciliation": _trace_f2,
    "frontier_factor_exposure": _trace_f3,
    "frontier_portfolio_pipeline": _trace_f4,
}


def _answer_from_expected(expected: dict) -> dict:
    """Prompt-format answer (category codes) built from a stored key."""
    from src.scorer import categories_for_key_items
    answer = {k: v for k, v in expected.items() if k != "problems_found"}
    return {"problems": sorted(categories_for_key_items(
        expected.get("problems_found", []))), **answer}


def build_trace_parts(task: dict, data: str, expected: dict,
                      max_tokens: int | None = 4096):
    """Return (steps_text, answer_dict) for one task.

    Step values are recomputed from ``data``; the answer dict uses the
    same numbers, so its final values equal the key whenever the key
    was built with the prompt's rules. The steps (never the answer)
    are shortened when they exceed ``max_tokens``.
    """
    builder = _BUILDERS.get(task.get("type"))
    if builder is not None:
        lines, answer = builder(task, data, expected)
    else:
        rows = len([ln for ln in data.splitlines() if ln.strip()])
        answer = _answer_from_expected(expected)
        lines = [
            f"Step 1 - parse the data file: {rows} non-empty lines.",
            "Step 2 - apply the cleaning and computation rules stated "
            "in the task to the parsed series.",
            f"Step 3 - problems detected: "
            f"{', '.join(answer['problems']) or 'none'}.",
        ]
    steps = _cap_steps("\n".join(lines), max_tokens)
    return steps, answer


def build_trace(task: dict, data: str, expected: dict,
                max_tokens: int | None = 4096) -> str:
    """Full trace text: steps, then the final answer as JSON."""
    steps, answer = build_trace_parts(task, data, expected, max_tokens)
    return f"{steps}\nFinal answer:\n{json.dumps(answer)}"
