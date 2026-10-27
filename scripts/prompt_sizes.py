#!/usr/bin/env python3
"""Report per-task prompt sizes for a generated task set.

The size reported is the full user message the runner sends: the task
prompt plus "\\n\\nData:\\n" plus the data file contents. Sizes are in
characters, and in tokens when a tokenizer is available offline
(tiktoken with a cached encoding); no network calls are made.

Usage:
    python3 scripts/prompt_sizes.py [--tasks-dir tasks_frontier]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _tokenizer():
    """Return (token-count function, method label).

    Prefers tiktoken's cl100k_base encoding (an approximation for Muse
    models) when it is importable and its encoding is cached/available;
    otherwise falls back to a regex approximation so tokens are always
    reported. No paid calls are made either way.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda text: len(enc.encode(text))), "cl100k_base"
    except Exception:
        import re
        pieces = re.compile(r"\w+|[^\w\s]")
        return (lambda text: len(pieces.findall(text))), \
            "regex approximation (tiktoken unavailable)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks_frontier"),
                        help="Generated task directory (default: tasks_frontier)")
    args = parser.parse_args()

    tasks = json.loads((args.tasks_dir / "tasks.json").read_text())
    count_tokens, method = _tokenizer()
    by_type = defaultdict(list)
    for task in tasks:
        data = (args.tasks_dir / task["data_file"]).read_text()
        full = f"{task['prompt']}\n\nData:\n{data}"
        by_type[task["type"]].append(
            (task["id"], len(full), count_tokens(full)))

    print(f"token method: {method}")
    print(f"{'type':38} {'n':>3} {'max_chars':>10} {'mean_chars':>11} "
          f"{'max_tokens':>12}")
    overall_max = 0
    for task_type, rows in sorted(by_type.items()):
        max_id, max_chars, max_tokens = max(rows, key=lambda r: r[1])
        overall_max = max(overall_max, max_chars)
        mean_chars = sum(r[1] for r in rows) / len(rows)
        print(f"{task_type:38} {len(rows):>3} {max_chars:>10} "
              f"{mean_chars:>11.0f} {max_tokens:>12}  max={max_id}")
    print(f"overall max_chars: {overall_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
