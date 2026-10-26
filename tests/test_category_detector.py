"""The category detector (written from the prompt definitions) must
reproduce every key's problem set exactly, on both tiers."""

import json
import tempfile
from pathlib import Path

from src.generate_tasks import TaskGenerator
from src.hard_tier import HardTaskGenerator
from src.scorer import categories_for_key_items
from tests.category_detector import detect_categories


def _check(gen, tmp, tasks):
    out = Path(tmp)
    for t in tasks:
        key = json.loads((out / "answers" / f"{t['id']}_answer.json").read_text())
        want = categories_for_key_items(key["problems_found"])
        got = detect_categories(out / t["data_file"])
        assert got == want, (t["id"], sorted(got), sorted(want),
                             key["problems_found"])


class TestDetectorMatchesKeys:
    def test_hard_seeds_1_to_200(self):
        for seed in range(1, 201):
            with tempfile.TemporaryDirectory() as tmp:
                tasks = HardTaskGenerator(seed=seed).generate_all_tasks(
                    Path(tmp), num_tasks=5)
                _check(None, tmp, tasks)

    def test_easy_seeds_1_to_200(self):
        for seed in range(1, 201):
            with tempfile.TemporaryDirectory() as tmp:
                tasks = TaskGenerator(seed=seed).generate_all_tasks(
                    Path(tmp), num_tasks=5)
                _check(None, tmp, tasks)


class TestDefinitionsInPrompts:
    def test_hard_prompts_define_every_code(self):
        gen = HardTaskGenerator(seed=7)
        codes = ["duplicate", "missing", "bad_price", "split", "reverse_split",
                 "unit_change", "out_of_order", "timezone", "currency",
                 "adj_close", "column_order", "calendar"]
        for maker in gen.task_generators():
            task, _ = maker()
            for code in codes:
                assert f"- {code}:" in task["prompt"], (task["type"], code)
            # calendar and missing must be distinguishable
            assert "each ticker is complete on its own calendar" in task["prompt"]
            assert "inside its own date range" in task["prompt"]

    def test_easy_prompts_define_every_code(self):
        gen = TaskGenerator(seed=42)
        for maker in (gen.generate_task_return_calculation,
                      gen.generate_task_backtest,
                      gen.generate_task_volatility):
            task, _ = maker()
            for code in ("duplicate", "missing", "bad_price", "split"):
                assert f"- {code}:" in task["prompt"], code
