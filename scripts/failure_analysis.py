"""Generate the failure-analysis chart from existing results data only.

Reads the single source of truth, ``app/data/leaderboard.json`` (the
``tools_frontier`` board: tools on, 12 turns, 90 held-out tasks), and
renders stacked pass / miss bars to
``docs/assets/failure-analysis-tools-frontier.png``.

Honesty rule: a turn-limit vs wrong-answer split is drawn only where the
results data actually records one. Today that is Muse Spark 1.3, whose
board note records "25 of 27" misses as turn-limit exhaustion (parsed
from the note and checked against the pass counts). For every other
model the miss segment is shown as "breakdown not recorded" — no split
is invented. Nothing here reruns a model or calls any API.

Usage:
    pip install matplotlib
    python scripts/failure_analysis.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "app" / "data" / "leaderboard.json"
OUTPUT = ROOT / "docs" / "assets" / "failure-analysis-tools-frontier.png"

# The one documented failure split, e.g. "(25 of 27)".
_TURN_LIMIT_RE = re.compile(r"\((\d+) of (\d+)\) are running out of turns")


def load_failure_data() -> list[dict]:
    """Per-model pass / turn-limit / wrong-answer / unrecorded counts."""
    data = json.loads(LEADERBOARD.read_text())
    board = next(b for b in data["boards"] if b["id"] == "tools-frontier")
    turn_limit: dict[str, tuple[int, int]] = {}
    for note in board.get("notes", []):
        match = _TURN_LIMIT_RE.search(note)
        if match and "Spark 1.3" in note:
            turn_limit["Muse Spark 1.3"] = (int(match.group(1)), int(match.group(2)))

    rows: list[dict] = []
    for entry in board["entries"]:
        if entry.get("total"):
            passed, total = entry["total"]["passed"], entry["total"]["total"]
        else:
            families = entry.get("families", {})
            passed = sum(f.get("passed", 0) for f in families.values())
            total = sum(f.get("total", 0) for f in families.values())
        missed = total - passed
        tl, wrong, unrecorded = 0, 0, missed
        if entry["name"] in turn_limit:
            tl, recorded_missed = turn_limit[entry["name"]]
            if recorded_missed != missed:
                raise SystemExit(
                    f"[failure_analysis] note says {recorded_missed} misses for "
                    f"{entry['name']} but counts give {missed}; refusing to chart"
                )
            wrong, unrecorded = missed - tl, 0
        rows.append(
            {
                "name": entry["name"],
                "status": entry.get("status", "complete"),
                "passed": passed,
                "total": total,
                "turn_limit": tl,
                "wrong": wrong,
                "unrecorded": unrecorded,
            }
        )
    return rows


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "[failure_analysis] matplotlib is required: pip install matplotlib"
        ) from None

    rows = load_failure_data()
    for row in rows:
        print(f"[failure_analysis] {row}")

    names = [
        f"{r['name']} ({r['passed']}/{r['total']})"
        + (" · partial" if r["status"] == "partial" else "")
        for r in rows
    ]
    # Bottom-to-top in data order; reverse so the strongest model is on top.
    order = list(reversed(rows))
    labels = list(reversed(names))
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.barh(y, [r["passed"] for r in order], color="#009d5e", label="Pass")
    left = [r["passed"] for r in order]
    ax.barh(
        y,
        [r["turn_limit"] for r in order],
        left=left,
        color="#e8a33d",
        label="Miss — turn-limit exhaustion (recorded)",
    )
    left = [a + r["turn_limit"] for a, r in zip(left, order, strict=True)]
    ax.barh(
        y,
        [r["wrong"] for r in order],
        left=left,
        color="#d64545",
        label="Miss — wrong answer (recorded)",
    )
    left = [a + r["wrong"] for a, r in zip(left, order, strict=True)]
    ax.barh(
        y,
        [r["unrecorded"] for r in order],
        left=left,
        color="#9aa0a6",
        hatch="//",
        edgecolor="#5f6368",
        label="Miss — failure breakdown not recorded",
    )
    for i, r in enumerate(order):
        ax.text(
            r["passed"] / 2,
            i,
            str(r["passed"]),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
        if r["turn_limit"]:
            ax.text(
                r["passed"] + r["turn_limit"] / 2,
                i,
                str(r["turn_limit"]),
                ha="center",
                va="center",
                fontsize=9,
            )
        if r["wrong"]:
            ax.text(
                r["passed"] + r["turn_limit"] + r["wrong"] / 2,
                i,
                str(r["wrong"]),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    ax.set_yticks(list(y), labels)
    ax.set_xlabel("Held-out tasks (tools_frontier, tools on, 12 turns)")
    ax.set_title(
        "Failure analysis: turn efficiency, not wrong answers, is the frontier\n"
        "Muse Spark 1.3 misses 27 of 90 — 25 recorded as turn-limit exhaustion, 2 wrong answers",
        fontsize=11,
    )
    ax.set_xlim(0, 90)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    print(f"[failure_analysis] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
