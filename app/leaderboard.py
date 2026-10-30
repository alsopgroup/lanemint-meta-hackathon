"""Held-out leaderboard data for the demo page.

The numbers live in exactly one place — ``data/leaderboard.json`` in
this directory — so updating the leaderboard means editing that file
(and nothing in the frontend or this loader). The loader validates
the file's shape, recomputes every total from the per-family counts
where exact counts are given, and attaches a pass percentage and a
95% Wilson score interval, so the displayed numbers are always
internally consistent.

Data shape: the file holds one or more ``boards`` (for example the
frontier tools-off/tools-on comparison and the ``tools_frontier``
per-family board). Each board has its own entries; an entry carries
exact ``passed``/``total`` counts, an ``approx_pct`` where only an
approximate rate was reported, or a ``display`` string for partial
runs where no denominator is published per family. Approximate and
display-only figures are passed through labelled as such — the loader
never invents exact counts for them. The legacy single-board shape
(``benchmark`` + ``entries``) is still accepted and wrapped as one
board.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "leaderboard.json"
_Z = 1.96  # 95% Wilson score interval


def wilson_interval(passed: int, total: int) -> tuple[int, int]:
    """95% Wilson score interval for a pass rate, in whole percent."""
    if total <= 0:
        return (0, 0)
    p = passed / total
    denom = 1 + _Z**2 / total
    centre = (p + _Z**2 / (2 * total)) / denom
    half = _Z * math.sqrt(p * (1 - p) / total
                          + _Z**2 / (4 * total**2)) / denom
    return (max(0, round((centre - half) * 100)),
            min(100, round((centre + half) * 100)))


def _check_count(value: object, where: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object with passed/total")
    passed, total = value.get("passed"), value.get("total")
    if not isinstance(passed, int) or not isinstance(total, int) \
            or total <= 0 or not 0 <= passed <= total:
        raise ValueError(
            f"{where}: need integers with 0 <= passed <= total, "
            f"got passed={passed!r} total={total!r}")
    return passed, total


def _summarise(passed: int, total: int) -> dict:
    low, high = wilson_interval(passed, total)
    # Headline percentages are whole-percent for every model, so the
    # displayed rates are directly comparable (e.g. 2/96 shows as 2%).
    return {"passed": passed, "total": total,
            "pct": round(100 * passed / total) if total else 0,
            "ci95": [low, high], "approx": False, "display": None}


def _summarise_result(value: object, where: str) -> dict:
    """Summarise an exact count, an approximate rate, or a display cell."""
    if isinstance(value, dict) and "approx_pct" in value:
        approx = value["approx_pct"]
        if not isinstance(approx, (int, float)) or not 0 <= approx <= 100:
            raise ValueError(f"{where}: approx_pct must be within 0-100")
        pct = round(approx) if float(approx).is_integer() else approx
        return {"passed": None, "total": None, "pct": pct,
                "ci95": None, "approx": True, "display": f"~{pct}%"}
    if isinstance(value, dict) and "display" in value \
            and "passed" not in value:
        display = value["display"]
        if not isinstance(display, str) or not display:
            raise ValueError(f"{where}: display must be a non-empty string")
        return {"passed": None, "total": None, "pct": None,
                "ci95": None, "approx": False, "display": display}
    passed, total = _check_count(value, where)
    return _summarise(passed, total)


def _load_entries(raw_entries: object, family_ids: list[str],
                  board_id: str) -> list[dict]:
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"board {board_id!r}: 'entries' must be a "
                         "non-empty list")
    entries = []
    seen_ids = set()
    for entry in raw_entries:
        if not isinstance(entry, dict) or not entry.get("id") \
                or not entry.get("name"):
            raise ValueError("each leaderboard entry needs id and name")
        if entry["id"] in seen_ids:
            raise ValueError(f"duplicate leaderboard entry id "
                             f"{entry['id']!r}")
        seen_ids.add(entry["id"])
        status = entry.get("status")
        if status not in ("complete", "coming_soon"):
            raise ValueError(
                f"entry {entry['id']!r}: status must be 'complete' or "
                f"'coming_soon'")
        out = {"id": entry["id"], "name": entry["name"],
               "provider": entry.get("provider", ""),
               "status": status, "notes": list(entry.get("notes", []))}
        if entry.get("scope"):
            out["scope"] = entry["scope"]
        if status == "coming_soon":
            out["families"] = None
            out["total"] = None
            entries.append(out)
            continue
        for key in ("tools_off", "tools_on"):
            if entry.get(key) is not None:
                out[key] = _summarise_result(
                    entry[key], f"entry {entry['id']!r} {key}")
        per_family = entry.get("families")
        if per_family is not None:
            if not family_ids or not isinstance(per_family, dict) \
                    or set(per_family) != set(family_ids):
                raise ValueError(
                    f"entry {entry['id']!r}: families must cover "
                    f"exactly {family_ids}")
            fams, passed_sum, total_sum, all_exact = {}, 0, 0, True
            for fid in family_ids:
                fams[fid] = _summarise_result(
                    per_family[fid],
                    f"entry {entry['id']!r} family {fid}")
                if fams[fid]["passed"] is None:
                    all_exact = False
                else:
                    passed_sum += fams[fid]["passed"]
                    total_sum += fams[fid]["total"]
            out["families"] = fams
            if entry.get("total") is not None:
                passed, total = _check_count(
                    entry["total"], f"entry {entry['id']!r} total")
                out["total"] = _summarise(passed, total)
            elif all_exact:
                out["total"] = _summarise(passed_sum, total_sum)
            else:
                raise ValueError(
                    f"entry {entry['id']!r}: display-only families need "
                    f"an explicit 'total'")
        elif entry.get("total") is not None:
            passed, total = _check_count(entry["total"],
                                         f"entry {entry['id']!r} total")
            out["families"] = None
            out["total"] = _summarise(passed, total)
        elif "tools_off" in out or "tools_on" in out:
            out["families"] = None
            out["total"] = None
        else:
            raise ValueError(
                f"entry {entry['id']!r}: complete entries need "
                f"'families', 'total', or tools_off/tools_on counts")
        entries.append(out)
    return entries


def _load_board(raw_board: object, index: int) -> dict:
    if not isinstance(raw_board, dict):
        raise ValueError(f"board {index}: expected an object")
    board_id = raw_board.get("id") or f"board-{index}"
    families = raw_board.get("families", [])
    if not isinstance(families, list):
        raise ValueError(f"board {board_id!r}: families must be a list")
    family_ids = [f.get("id") for f in families if isinstance(f, dict)]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError(f"board {board_id!r}: families must list "
                         "unique family ids")
    if families and not family_ids:
        raise ValueError(f"board {board_id!r}: families must list ids")
    board = {"id": board_id,
             "title": raw_board.get("title", "Leaderboard"),
             "tier": raw_board.get("tier"),
             "tasks": raw_board.get("tasks"),
             "max_turns": raw_board.get("max_turns"),
             "headline": raw_board.get("headline"),
             "note": raw_board.get("note"),
             "notes": list(raw_board.get("notes", [])),
             "families": families,
             "entries": _load_entries(raw_board.get("entries"),
                                      family_ids, board_id)}
    # Legacy single-board files nested their metadata under "benchmark".
    benchmark = raw_board.get("benchmark")
    if isinstance(benchmark, dict):
        board["benchmark"] = benchmark
        if benchmark.get("families") and not families:
            board["families"] = benchmark["families"]
    return board


def load_leaderboard(path: Path | None = None) -> dict:
    """Load, validate, and enrich the leaderboard JSON file."""
    path = Path(path) if path is not None else DATA_PATH
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read leaderboard data {path}: {exc}") \
            from exc
    if not isinstance(raw, dict):
        raise ValueError("leaderboard data must be an object")
    if isinstance(raw.get("boards"), list) and raw["boards"]:
        boards = [_load_board(b, i) for i, b in enumerate(raw["boards"])]
        board_ids = [b["id"] for b in boards]
        if len(board_ids) != len(set(board_ids)):
            raise ValueError("board ids must be unique")
    elif isinstance(raw.get("entries"), list) and raw["entries"]:
        # Legacy shape: {"benchmark": {...families...}, "entries": [...]}
        benchmark = raw.get("benchmark", {})
        legacy = {"id": benchmark.get("tier", "leaderboard"),
                  "title": raw.get("title", "Leaderboard"),
                  "tier": benchmark.get("tier"),
                  "tasks": benchmark.get("tasks"),
                  "note": benchmark.get("note"),
                  "notes": list(benchmark.get("notes", [])),
                  "families": benchmark.get("families", []),
                  "benchmark": benchmark,
                  "entries": raw["entries"]}
        boards = [_load_board(legacy, 0)]
    else:
        raise ValueError("leaderboard data must have a non-empty "
                         "'boards' list (or legacy 'entries')")
    return {"title": raw.get("title", "Leaderboard"),
            "updated": raw.get("updated"),
            "boards": boards}
