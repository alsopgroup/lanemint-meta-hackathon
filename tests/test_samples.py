"""Regeneration and upload-fit checks for the judge samples in ``samples/``.

The samples are generated, not hand-written: regenerating the easy tier
with the documented seed must reproduce every generated file in
``samples/`` byte for byte, so the shipped CSVs and answer keys cannot
drift from ``src.generate_tasks``. (``samples/README.md`` is the one
file in that folder the generator does not emit.)
"""

import importlib.util
import json
from pathlib import Path

from src.generate_tasks import TaskGenerator
from src.scorer import categories_for_key_items

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SAMPLE_SEED = 31
SAMPLE_COUNT = 3
RESERVED_EVAL_SEEDS = {42, 1001, 1002}
EXPECTED_CATEGORIES = {"duplicate", "missing", "bad_price", "split"}


def _load_app_csv_validator():
    """Load the demo app's CSV validator without importing the app package."""
    path = Path(__file__).resolve().parent.parent / "app" / "muse_client.py"
    spec = importlib.util.spec_from_file_location("app_muse_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_seed_is_not_a_reserved_eval_seed():
    assert SAMPLE_SEED not in RESERVED_EVAL_SEEDS


def test_samples_match_generator_output_byte_for_byte(tmp_path):
    """samples/ must equal fresh generator output for the chosen seed."""
    regenerated = tmp_path / "samples"
    TaskGenerator(seed=SAMPLE_SEED).generate_all_tasks(
        regenerated, num_tasks=SAMPLE_COUNT)

    generated_files = sorted(
        p.relative_to(regenerated) for p in regenerated.rglob("*") if p.is_file())
    assert generated_files, "generator produced no files"

    shipped_files = sorted(
        p.relative_to(SAMPLES_DIR) for p in SAMPLES_DIR.rglob("*") if p.is_file())
    # The only extra shipped file is the hand-written README.
    assert shipped_files == sorted([*generated_files, Path("README.md")])

    for relative in generated_files:
        assert (SAMPLES_DIR / relative).read_bytes() == \
            (regenerated / relative).read_bytes(), \
            f"samples/{relative} has drifted from the generator; regenerate with: " \
            f"python -m src.generate_tasks --tier easy --seed {SAMPLE_SEED} " \
            f"--count {SAMPLE_COUNT} --output-dir samples"


def test_samples_cover_all_easy_issue_categories():
    """Together the samples cover duplicates, gaps, bad prices and splits."""
    covered = set()
    answers = sorted((SAMPLES_DIR / "answers").glob("*_answer.json"))
    assert len(answers) == SAMPLE_COUNT
    for answer_path in answers:
        answer = json.loads(answer_path.read_text())
        assert answer["problems_found"], f"{answer_path.name} plants no problems"
        covered |= categories_for_key_items(answer["problems_found"])
    assert EXPECTED_CATEGORIES <= covered


def test_each_sample_csv_has_a_matching_answer_key():
    tasks = json.loads((SAMPLES_DIR / "tasks.json").read_text())
    assert len(tasks) == SAMPLE_COUNT
    for task in tasks:
        assert (SAMPLES_DIR / task["data_file"]).is_file()
        assert (SAMPLES_DIR / "answers" / f"{task['id']}_answer.json").is_file()


def test_samples_fit_app_upload_limits():
    """Every sample CSV must be accepted by the app's upload path.

    Server side: ``validate_csv`` in ``app/muse_client.py`` caps
    characters, rows and columns. Client side: ``parseCSV`` in
    ``app/static/js/app.js`` reads the header plus the first 500 data
    rows, so the header must be the generator's column set.
    """
    muse_client = _load_app_csv_validator()
    tasks = json.loads((SAMPLES_DIR / "tasks.json").read_text())
    for task in tasks:
        csv_text = (SAMPLES_DIR / task["data_file"]).read_text()
        header = csv_text.splitlines()[0]
        assert header == "date,ticker,open,high,low,close,volume"
        # Raises MuseBadRequest if any server-side cap is exceeded.
        _csv, row_count = muse_client.validate_csv(csv_text)
        assert 0 < row_count <= muse_client.MAX_CSV_ROWS
