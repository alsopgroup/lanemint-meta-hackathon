"""Generate the app's bundled sample chart images from ``samples/``.

Plots the close series of each public sample CSV in ``samples/`` to
``app/static/assets/sample-charts/`` and writes the matching simulated
findings manifest (``app/data/sample_charts.json``) that the app uses
in simulated/offline mode. Findings are derived from the public answer
keys in ``samples/answers/`` — nothing is invented, and no held-out
eval data (seeds 42/1001/1002) is used; the samples use seed 31.

The app matches an uploaded/simulated image by the SHA-256 of its
bytes, so regenerating the PNGs also regenerates the manifest; commit
both together.

Usage:
    pip install matplotlib pandas
    python scripts/make_sample_charts.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT_DIR = ROOT / "app" / "static" / "assets" / "sample-charts"
MANIFEST = ROOT / "app" / "data" / "sample_charts.json"

_KIND_TO_TYPE = {
    "bad_price": "prices",
    "duplicate_row": "dups",
    "missing_day": "gaps",
    "stock_split": "splits",
}


def _finding(label: str) -> dict:
    """One simulated finding from a public answer-key problem label."""
    # Labels look like bad_price_2024-03-15_10.0 / stock_split_2024-05-17_2.0
    kind = next(k for k in _KIND_TO_TYPE if label.startswith(k))
    parts = label.split("_")
    date = parts[2] if len(parts) > 2 else ""
    detail = parts[3] if len(parts) > 3 else ""
    if kind == "bad_price":
        title, explain = (
            "One-day price spike",
            f"The close on {date} spikes to about {detail}× its neighbours "
            "for a single day and then reverts — a bad tick, not a market move.",
        )
    elif kind == "duplicate_row":
        title, explain = (
            "Duplicate row",
            f"The bar for {date} appears twice, so that session would be "
            "counted twice in any return or volume calculation. Keep one copy.",
        )
    elif kind == "missing_day":
        title, explain = (
            "Missing day",
            f"{date} is a weekday with no bar, so a return across this break "
            "covers more than one session. Flag it; do not silently fill it.",
        )
    else:  # stock_split
        title, explain = (
            "Unadjusted split",
            f"From {date} prices sit at about 1/{detail} of their previous "
            "level and stay there, with volume scaled up — an unadjusted "
            "split. Earlier prices should be adjusted onto the new basis.",
        )
    return {"type": _KIND_TO_TYPE[kind], "title": title, "explain": explain}


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        raise SystemExit(
            "[make_sample_charts] matplotlib and pandas are required: "
            "pip install matplotlib pandas"
        ) from None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for csv_path in sorted(SAMPLES.glob("task_*.csv")):
        task_id = csv_path.stem
        answer = json.loads(
            (SAMPLES / "answers" / f"{task_id}_answer.json").read_text()
        )
        frame = pd.read_csv(csv_path)
        png_path = OUT_DIR / f"chart_{task_id}.png"
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(frame["date"], frame["close"], color="#009d5e", lw=1.6)
        ax.set_title(f"{frame['ticker'].iloc[0]} daily close — {task_id} (public sample)")
        ax.set_ylabel("close")
        # Show at most ~8 date labels so the axis stays legible.
        step = max(1, len(frame) // 8)
        ax.set_xticks(range(0, len(frame), step))
        ax.set_xticklabels(frame["date"].iloc[::step], rotation=30, ha="left", fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)

        series = [
            {"date": str(d), "value": round(float(v), 2)}
            for d, v in zip(frame["date"], frame["close"], strict=True)
        ]
        entries.append(
            {
                "id": task_id,
                "file_name": png_path.name,
                "url": f"/static/assets/sample-charts/{png_path.name}",
                "sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
                "source_csv": f"samples/{csv_path.name}",
                "summary": (
                    f"Simulated result for the bundled public sample {task_id}: "
                    "the visible problems planted in that sample's price series."
                ),
                "findings": [_finding(x) for x in answer.get("problems_found", [])],
                "series": series,
            }
        )
        print(f"[make_sample_charts] {png_path.name} ({len(series)} points)")

    MANIFEST.write_text(json.dumps({"samples": entries}, indent=2) + "\n")
    print(f"[make_sample_charts] wrote {MANIFEST}")


if __name__ == "__main__":
    main()
