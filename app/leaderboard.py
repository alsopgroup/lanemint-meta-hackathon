"""Held-out frontier leaderboard data for the demo page.

The numbers live in exactly one place — ``data/leaderboard.json`` in
this directory — so updating the leaderboard means editing that file
(and nothing in the frontend or this loader). The loader validates
the file's shape, recomputes every total from the per-family counts,
and attaches a pass percentage and a 95% Wilson score interval, so
the displayed numbers are always internally consistent.
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
    return {"passed": passed, "total": total,
            "pct": round(100 * passed / total, 1) if total else 0.0,
            "ci95": [low, high]}


def load_leaderboard(path: Path | None = None) -> dict:
    """Load, validate, and enrich the leaderboard JSON file."""
    path = Path(path) if path is not None else DATA_PATH
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read leaderboard data {path}: {exc}") \
            from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list) \
            or not raw["entries"]:
        raise ValueError("leaderboard data must have a non-empty "
                         "'entries' list")
    families = raw.get("benchmark", {}).get("families", [])
    family_ids = [f.get("id") for f in families if isinstance(f, dict)]
    if len(family_ids) != len(set(family_ids)) or not family_ids:
        raise ValueError("benchmark.families must list unique family ids")

    entries = []
    seen_ids = set()
    for entry in raw["entries"]:
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
        per_family = entry.get("families")
        if per_family is not None:
            if not isinstance(per_family, dict) \
                    or set(per_family) != set(family_ids):
                raise ValueError(
                    f"entry {entry['id']!r}: families must cover "
                    f"exactly {family_ids}")
            fams, passed_sum, total_sum = {}, 0, 0
            for fid in family_ids:
                passed, total = _check_count(
                    per_family[fid], f"entry {entry['id']!r} family {fid}")
                fams[fid] = _summarise(passed, total)
                passed_sum, total_sum = passed_sum + passed, total_sum + total
            out["families"] = fams
            out["total"] = _summarise(passed_sum, total_sum)
        elif entry.get("total") is not None:
            passed, total = _check_count(entry["total"],
                                         f"entry {entry['id']!r} total")
            out["families"] = None
            out["total"] = _summarise(passed, total)
        else:
            raise ValueError(
                f"entry {entry['id']!r}: complete entries need "
                f"'families' or 'total'")
        entries.append(out)

    return {"title": raw.get("title", "Leaderboard"),
            "updated": raw.get("updated"),
            "benchmark": raw.get("benchmark", {}),
            "entries": entries}
