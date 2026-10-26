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
    """Return a token-count function, or None when unavailable offline."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks_frontier"),
                        help="Generated task directory (default: tasks_frontier)")
    args = parser.parse_args()

    tasks = json.loads((args.tasks_dir / "tasks.json").read_text())
    count_tokens = _tokenizer()
    by_type = defaultdict(list)
    for task in tasks:
        data = (args.tasks_dir / task["data_file"]).read_text()
        full = f"{task['prompt']}\n\nData:\n{data}"
        by_type[task["type"]].append(
            (task["id"], len(full),
             count_tokens(full) if count_tokens else None))

    tok_note = "tokens (cl100k_base)" if count_tokens else "tokens n/a offline"
    print(f"{'type':38} {'n':>3} {'max_chars':>10} {'mean_chars':>11} "
          f"{'max_tokens':>12}  ({tok_note})")
    overall_max = 0
    for task_type, rows in sorted(by_type.items()):
        max_id, max_chars, max_tokens = max(rows, key=lambda r: r[1])
        overall_max = max(overall_max, max_chars)
        mean_chars = sum(r[1] for r in rows) / len(rows)
        tok = str(max_tokens) if max_tokens is not None else "n/a"
        print(f"{task_type:38} {len(rows):>3} {max_chars:>10} "
              f"{mean_chars:>11.0f} {tok:>12}  max={max_id}")
    print(f"overall max_chars: {overall_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
