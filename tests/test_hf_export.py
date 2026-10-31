"""HF export: builds a valid public dataset folder from NEW seeds only."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_export(output_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_hf_dataset.py"),
            "--output-dir",
            str(output_dir),
            "--count-easy",
            "2",
            "--count-hard",
            "2",
            "--count-frontier",
            "2",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )


def test_export_builds_dataset_folder(tmp_path):
    out = tmp_path / "hf_dataset"
    result = _run_export(out)
    assert result.returncode == 0, result.stdout + result.stderr

    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: other")
    for tag in ("meta-ai", "muse-spark", "finance", "benchmark"):
        assert tag in card

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seeds"] == [5001, 5002]
    assert manifest["counts_per_tier"] == {"easy": 4, "hard": 4, "frontier": 4}
    assert manifest["total_tasks"] == 12
    # Held-out eval seeds appear only in the exclusion record.
    assert manifest["excluded_seeds"]["held_out_eval"] == [42, 1001, 1002]

    for tier in ("easy", "hard", "frontier"):
        for seed in (5001, 5002):
            tasks = out / "data" / tier / f"seed_{seed}" / "tasks.json"
            assert tasks.exists()
            listed = json.loads(tasks.read_text())
            answers = out / "answers" / tier / f"seed_{seed}"
            assert len(list(listed)) == 2
            assert len(list(answers.glob("*_answer.json"))) == 2
    # No file path or content names a held-out eval seed set.
    assert not list(out.rglob("*seed_42*"))
    assert not list(out.rglob("*seed_1001*"))
    assert not list(out.rglob("*seed_1002*"))


def test_export_refuses_held_out_and_training_seeds(tmp_path):
    for seeds in ("42", "1001", "5001,1002", "2020"):
        result = _run_export(tmp_path / "x", "--seeds", seeds)
        assert result.returncode != 0
        assert "refused" in (result.stdout + result.stderr)
