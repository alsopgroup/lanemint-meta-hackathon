"""
Prepare Fireworks SFT data from benchmark tasks.

Seed-disjoint by construction: every task carries the seed that
generated it (in-process generation, or a tasks directory declared as
``PATH@SEED``), the train/validation split is by whole seed, and the
locked eval seeds (default 42, 1001, 1002) are refused as dataset
seeds. An optional ``--eval-dirs`` SHA-256 check refuses to write
anything if a training or validation CSV or answer is byte-equivalent to an eval
task's. All randomness comes from a local ``random.Random``.

Each run writes ``train.jsonl``, ``validation.jsonl``,
``validation_task_ids.json`` and a ``manifest.json`` (generator
commit, seeds, per-family counts, token stats, output SHA-256s).
With ``--target trace`` the assistant message gains a
``reasoning_content`` worked solution from ``src.trace`` (Fireworks
SFT supports thinking traces in that field); ``content`` stays the
final answer JSON.
"""

import hashlib
import json
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from src.trace import build_trace_parts, count_tokens

DEFAULT_EXCLUDE_SEEDS = (42, 1001, 1002)
SYSTEM_MESSAGE = (
    'You are a financial data analyst. Analyze the provided data and '
    'return results as valid JSON only.')


def parse_seed_list(text: str) -> list[int]:
    """'2001-2004,2007' -> [2001, 2002, 2003, 2004, 2007] (sorted, unique)."""
    seeds = set()
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(part))
    return sorted(seeds)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_answer_hash(expected: dict) -> str:
    return _sha256_bytes(
        json.dumps(expected, sort_keys=True).encode('utf-8'))


def _generator_commit() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent, timeout=10)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


class FineTunePrep:
    """Prepares seed-disjoint fine-tuning data from tasks."""

    def __init__(self, rng_seed: int = 0):
        """Local RNG only; the global random state is never touched."""
        self.rng = random.Random(rng_seed)

    # ------------------------------------------------------------ loading
    def load_tasks_with_answers(self, tasks_dir: Path) -> list[dict]:
        """Load tasks with their expected answers and data payloads."""
        tasks_dir = Path(tasks_dir)
        with open(tasks_dir / 'tasks.json') as f:
            tasks = json.load(f)
        answers_dir = tasks_dir / 'answers'
        for task in tasks:
            with open(answers_dir / f"{task['id']}_answer.json") as f:
                task['expected'] = json.load(f)
            with open(tasks_dir / task['data_file']) as f:
                task['data'] = f.read()
        return tasks

    @staticmethod
    def _infer_tier(task: dict) -> str:
        """Tier of a directory-loaded task (for --mix tier groups)."""
        task_id = str(task.get('id', ''))
        if task_id.startswith('task_f_') or \
                str(task.get('type', '')).startswith('frontier_'):
            return 'frontier'
        if task_id.startswith('task_h_'):
            return 'hard'
        return 'easy'

    def load_records(self, tasks_dir: Path, seed: int | None,
                     tier: str | None = None) -> list[dict]:
        """Load one tasks directory as dataset records tagged by seed."""
        records = []
        for task in self.load_tasks_with_answers(tasks_dir):
            records.append({
                'task': task, 'seed': seed,
                'tier': tier or self._infer_tier(task),
                'type': task['type'],
                'csv_sha256': _sha256_bytes(task['data'].encode('utf-8')),
                'answer_sha256': _canonical_answer_hash(task['expected']),
            })
        return records

    def generate_records(self, tier: str, seeds: list[int], count: int,
                         tmp_root: Path) -> list[dict]:
        """Generate tasks in-process, one fresh generator per seed."""
        records = []
        for seed in seeds:
            if tier == 'frontier':
                from src.frontier_tier import FrontierTaskGenerator
                gen = FrontierTaskGenerator(seed=seed)
            elif tier == 'hard':
                from src.hard_tier import HardTaskGenerator
                gen = HardTaskGenerator(seed=seed)
            else:
                from src.generate_tasks import TaskGenerator
                gen = TaskGenerator(seed=seed)
            out = Path(tmp_root) / f"{tier}_{seed}"
            gen.generate_all_tasks(out, num_tasks=count)
            records.extend(self.load_records(out, seed=seed, tier=tier))
        return records

    # ------------------------------------------------------------ answers
    @staticmethod
    def target_answer(expected: dict) -> dict:
        """Convert a stored key to the answer format the prompts ask for.

        Stored keys use ``problems_found`` with descriptive strings; prompts
        ask for a JSON ``problems`` array of category codes.
        """
        from src.scorer import categories_for_key_items
        answer = {k: v for k, v in expected.items() if k != 'problems_found'}
        answer = {'problems': sorted(categories_for_key_items(
            expected.get('problems_found', []))), **answer}
        return answer

    def format_for_training(self, task: dict, target: str = 'answer',
                            trace_max_tokens: int = 4096) -> dict:
        """Format a single task as a Fireworks SFT training example."""
        assistant = {'role': 'assistant',
                     'content': json.dumps(
                         self.target_answer(task['expected']))}
        if target == 'trace':
            steps, _ = build_trace_parts(
                task, task['data'], task['expected'],
                max_tokens=trace_max_tokens)
            assistant['reasoning_content'] = steps
        return {'messages': [
            {'role': 'system', 'content': SYSTEM_MESSAGE},
            {'role': 'user',
             'content': f"{task['prompt']}\n\nData:\n{task['data']}"},
            assistant]}

    # ------------------------------------------------------------ grouping
    @staticmethod
    def _row_tokens(example: dict) -> int:
        text = "\n".join(
            str(m.get('content', '')) + str(m.get('reasoning_content', ''))
            for m in example['messages'])
        return count_tokens(text)

    def _apply_mix(self, records: list[dict],
                   mix: dict[str, float]) -> list[dict]:
        """Downsample records to the requested group proportions.

        A group key names a task type if any record has that type,
        else a tier. Unlisted groups share the remaining proportion
        equally. Sampling uses the local RNG and never upsamples.
        """
        if not mix:
            return records

        def group_of(rec):
            if rec['type'] in mix:
                return rec['type']
            if rec['tier'] in mix:
                return rec['tier']
            return f"other:{rec['tier'] or rec['type']}"

        groups: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            groups[group_of(rec)].append(rec)
        listed = {g: p for g, p in mix.items() if g in groups}
        rest = max(0.0, 1.0 - sum(listed.values()))
        unlisted = [g for g in groups if g not in listed]
        props = dict(listed)
        for g in unlisted:
            props[g] = rest / len(unlisted) if unlisted else 0.0
        total_prop = sum(props.values())
        if total_prop <= 0:
            return records
        props = {g: p / total_prop for g, p in props.items()}
        total = min(int(len(groups[g]) / props[g]) if props[g] > 0
                    else 0 for g in groups)
        kept = []
        for g, recs in groups.items():
            n = min(len(recs), int(total * props[g]))
            if n >= len(recs):
                kept.extend(recs)
            elif n > 0:
                idx = sorted(self.rng.sample(range(len(recs)), n))
                kept.extend(recs[i] for i in idx)
        order = {id(r): i for i, r in enumerate(records)}
        kept.sort(key=lambda r: order[id(r)])
        return kept

    # ------------------------------------------------------------ pipeline
    def prepare(self, output_dir: Path, tasks_dirs=(), tier: str = None,
                seeds=(), count: int = 25, val_seeds=(),
                exclude_seeds=DEFAULT_EXCLUDE_SEEDS, eval_dirs=(),
                families=None, mix=None, target: str = 'answer',
                trace_max_tokens: int = 4096, train_ratio: float = 0.8,
                allow_eval_seeds: bool = False):
        """Build the dataset; raise ValueError before writing anything
        when a seed rule or the eval hash check is violated."""
        output_dir = Path(output_dir)
        val_seeds = set(val_seeds)
        excluded = set(exclude_seeds)
        missing_locked = sorted(set(DEFAULT_EXCLUDE_SEEDS) - excluded)
        if missing_locked and not allow_eval_seeds:
            raise ValueError(
                f"refusing to build dataset: --exclude-seeds "
                f"{sorted(excluded)} would unlock locked eval seed(s) "
                f"{missing_locked}; pass --allow-eval-seeds to "
                f"override the lock explicitly")
        if allow_eval_seeds and missing_locked:
            print("WARNING: --allow-eval-seeds in effect; the locked "
                  "eval-seed protection (42, 1001, 1002) is "
                  "overridden for this run")

        with tempfile.TemporaryDirectory() as tmp:
            records: list[dict] = []
            for path, seed in tasks_dirs:
                records.extend(self.load_records(path, seed=seed))
            if seeds:
                records.extend(self.generate_records(
                    tier or 'frontier', list(seeds), count, Path(tmp)))

        # ---- refusal checks (nothing has been written yet) ----
        dataset_seeds = {r['seed'] for r in records
                         if r['seed'] is not None}
        bad_excluded = sorted(dataset_seeds & excluded)
        if bad_excluded:
            raise ValueError(
                f"refusing to build dataset: seed(s) {bad_excluded} are "
                f"locked eval seeds (excluded: {sorted(excluded)}); "
                f"training or validating on them would contaminate "
                f"the held-out evaluation")
        undeclared = [r for r in records if r['seed'] is None]
        if undeclared:
            raise ValueError(
                f"refusing to build dataset: {len(undeclared)} task(s) "
                f"have no declared generation seed; declare each "
                f"tasks directory as PATH@SEED (or generate with "
                f"--seeds) so the locked eval-seed refusal can see "
                f"them")
        train_records = [r for r in records if r['seed'] not in val_seeds]
        val_records = [r for r in records if r['seed'] in val_seeds]
        if not val_records and val_seeds:
            raise ValueError(
                f"validation seeds {sorted(val_seeds)} produced no "
                f"tasks; check --val-seeds against --seeds/--tasks-dir")

        if families:
            keep = set(families)
            train_records = [r for r in train_records
                             if r['type'] in keep]
            val_records = [r for r in val_records if r['type'] in keep]

        if eval_dirs:
            eval_csv, eval_ans = set(), set()
            for path in eval_dirs:
                for rec in self.load_records(path, seed=None):
                    eval_csv.add(rec['csv_sha256'])
                    eval_ans.add(rec['answer_sha256'])

            def _collides(rec):
                return rec['csv_sha256'] in eval_csv \
                    or rec['answer_sha256'] in eval_ans

            train_hits = [r for r in train_records if _collides(r)]
            val_hits = [r for r in val_records if _collides(r)]
            collisions = train_hits + val_hits
            if collisions:
                sample = ", ".join(
                    f"seed {r['seed']}:{r['task']['id']}"
                    for r in collisions[:5])
                raise ValueError(
                    f"refusing to write: {len(collisions)} dataset "
                    f"task(s) match an eval task by SHA-256 (CSV or "
                    f"answer; {len(train_hits)} in training, "
                    f"{len(val_hits)} in validation), e.g. {sample}; "
                    f"training or validating on held-out eval tasks "
                    f"is not allowed")
        else:
            eval_csv = eval_ans = set()

        if mix:
            train_records = self._apply_mix(train_records, mix)
            val_records = self._apply_mix(val_records, mix)

        # ---- format, measure, then write ----
        def build(recs):
            rows = [self.format_for_training(
                r['task'], target=target,
                trace_max_tokens=trace_max_tokens) for r in recs]
            tokens = [self._row_tokens(row) for row in rows]
            return rows, tokens

        train_rows, train_tokens = build(train_records)
        val_rows, val_tokens = build(val_records)

        output_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, rows in (('train.jsonl', train_rows),
                           ('validation.jsonl', val_rows)):
            path = output_dir / name
            blob = "".join(json.dumps(r) + "\n" for r in rows) \
                .encode('utf-8')
            path.write_bytes(blob)
            files[name] = {'sha256': _sha256_bytes(blob),
                           'bytes': len(blob), 'rows': len(rows)}
        val_ids = [f"seed{r['seed']}:{r['task']['id']}"
                   if r['seed'] is not None else r['task']['id']
                   for r in val_records]
        (output_dir / 'validation_task_ids.json').write_text(
            json.dumps(val_ids, indent=2))

        def stats(tokens):
            return {'rows': len(tokens),
                    'max_tokens': max(tokens) if tokens else 0,
                    'mean_tokens': round(sum(tokens) / len(tokens), 1)
                    if tokens else 0.0}

        def by_family(recs):
            return dict(sorted(Counter(r['type'] for r in recs).items()))

        manifest = {
            'generator_commit': _generator_commit(),
            'target': target,
            'trace_max_tokens': trace_max_tokens if target == 'trace'
            else None,
            'tier': tier,
            'train_seeds': sorted({r['seed'] for r in train_records
                                   if r['seed'] is not None}),
            'val_seeds': sorted({r['seed'] for r in val_records
                                 if r['seed'] is not None}),
            'excluded_seeds': sorted(excluded),
            'families': sorted(families) if families else None,
            'mix': mix,
            'counts': {
                'train': {'total': len(train_records),
                          'by_family': by_family(train_records)},
                'validation': {'total': len(val_records),
                               'by_family': by_family(val_records)}},
            'token_stats': {'train': stats(train_tokens),
                            'validation': stats(val_tokens),
                            'tokenizer': 'cl100k_base if available, '
                                         'else chars/4 approximation'},
            'eval_check': {
                'eval_dirs': [str(p) for p in eval_dirs],
                'eval_csv_hashes': len(eval_csv),
                'eval_answer_hashes': len(eval_ans),
                'train_collisions': 0,
                'validation_collisions': 0},
            'files': files,
        }
        (output_dir / 'manifest.json').write_text(
            json.dumps(manifest, indent=2))
        return {'train_file': str(output_dir / 'train.jsonl'),
                'validation_file': str(output_dir / 'validation.jsonl'),
                'manifest_file': str(output_dir / 'manifest.json'),
                'num_train': len(train_records),
                'num_validation': len(val_records),
                'manifest': manifest}

    # ------------------------------------------------------------ legacy
    def split_and_prepare(self, tasks_dir: Path, output_dir: Path,
                          train_ratio: float = 0.8, seed: int = None):
        """Single-directory entry point.

        The directory's generation seed must be declared via ``seed``
        (the CLI equivalent of ``--tasks-dir PATH@SEED``); an
        undeclared directory is refused, because its seed is unknown
        and locked eval-seed tasks could otherwise pass the exclusion
        check unseen. ``train_ratio`` is accepted for compatibility
        but the split is by seed (all of this directory's tasks land
        in training unless their seed is a validation seed in
        :meth:`prepare` terms; here they are training records).
        """
        if seed is None:
            raise ValueError(
                "split_and_prepare requires the directory's "
                "generation seed (seed=<seed>, CLI: --tasks-dir "
                "PATH@SEED); undeclared directories would bypass "
                "the locked eval-seed refusal")
        return self.prepare(output_dir,
                            tasks_dirs=[(Path(tasks_dir), seed)],
                            train_ratio=train_ratio)


def _parse_mix(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    mix = {}
    for part in text.split(','):
        key, _, value = part.partition('=')
        mix[key.strip()] = float(value)
    return mix


def _parse_tasks_dir(values) -> list:
    parsed = []
    for value in values or []:
        path, at, seed = str(value).rpartition('@')
        if at and seed.lstrip('-').isdigit():
            parsed.append((Path(path), int(seed)))
        else:
            parsed.append((Path(value), None))
    return parsed


def main():
    """CLI for preparing Fireworks SFT data."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare seed-disjoint Fireworks SFT data')
    parser.add_argument('--tasks-dir', action='append', default=None,
                        help='Tasks directory (repeatable); the '
                             'generation seed must be declared as '
                             '@SEED, e.g. tasks_frontier@2001, so '
                             'undeclared directories cannot bypass the '
                             'locked-seed refusal')
    parser.add_argument('--tier', choices=['easy', 'hard', 'frontier'],
                        default=None,
                        help='Tier to generate in-process with --seeds '
                             '(default: frontier)')
    parser.add_argument('--seeds', default=None,
                        help='Generation seeds, e.g. 2001-2040 (with '
                             '--tier); generated fresh per seed')
    parser.add_argument('--count', type=int, default=25,
                        help='Tasks generated per seed (default: 25)')
    parser.add_argument('--val-seeds', default='',
                        help='Seeds held out for validation, e.g. '
                             '2037-2040; split is by whole seed')
    parser.add_argument('--exclude-seeds', default='42,1001,1002',
                        help='Locked eval seeds refused as dataset '
                             'seeds (default: 42,1001,1002); dropping '
                             'a locked seed from this list requires '
                             '--allow-eval-seeds')
    parser.add_argument('--allow-eval-seeds', action='store_true',
                        help='Explicitly override the locked eval-seed '
                             'protection (42, 1001, 1002); prints a '
                             'warning. Never use for real training runs')
    parser.add_argument('--eval-dirs', action='append', default=None,
                        help='Eval tasks directory (repeatable); any '
                             'training/validation CSV/answer SHA-256 match refuses '
                             'the run before anything is written')
    parser.add_argument('--families', default=None,
                        help='Comma-separated task types to keep')
    parser.add_argument('--mix', default=None,
                        help='Group proportions, e.g. hard=0.15,'
                             'frontier=0.85 (type or tier keys)')
    parser.add_argument('--target', choices=['answer', 'trace'],
                        default='answer',
                        help="Assistant target: final JSON only, or "
                             "with a reasoning_content worked trace")
    parser.add_argument('--trace-max-tokens', type=int, default=4096,
                        help='Token cap for trace reasoning_content '
                             '(default: 4096)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('finetune_data'),
                        help='Output directory (default: finetune_data/)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Accepted for compatibility; the split '
                             'is by seed, not ratio')

    args = parser.parse_args()
    tasks_dirs = _parse_tasks_dir(args.tasks_dir)
    if args.tasks_dir is not None:
        undeclared = [str(p) for p, seed in tasks_dirs if seed is None]
        if undeclared:
            print(f"Error: --tasks-dir {undeclared} must declare the "
                  f"generation seed as PATH@SEED (e.g. "
                  f"tasks_frontier@2001); undeclared directories "
                  f"would bypass the locked eval-seed refusal")
            return 1
    if not tasks_dirs and not args.seeds:
        print("Error: no dataset source: pass --seeds with --tier to "
              "generate tasks, or --tasks-dir PATH@SEED with a "
              "declared generation seed. There is no implicit tasks/ "
              "default: generate_tasks builds it at seed 42, a locked "
              "eval seed, so training on it sight-unseen is refused")
        return 1
    for path, _seed in tasks_dirs:
        if not path.exists():
            print(f"Error: Tasks directory not found: {path}")
            print("Run: python -m src.generate_tasks first")
            return 1

    prep = FineTunePrep()
    try:
        result = prep.prepare(
            args.output_dir,
            tasks_dirs=tasks_dirs,
            tier=args.tier or ('frontier' if args.seeds else None),
            seeds=parse_seed_list(args.seeds) if args.seeds else (),
            count=args.count,
            val_seeds=parse_seed_list(args.val_seeds),
            exclude_seeds=parse_seed_list(args.exclude_seeds),
            eval_dirs=[Path(p) for p in args.eval_dirs or []],
            families=args.families.split(',') if args.families else None,
            mix=_parse_mix(args.mix),
            target=args.target,
            trace_max_tokens=args.trace_max_tokens,
            train_ratio=args.train_ratio,
            allow_eval_seeds=args.allow_eval_seeds)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    manifest = result['manifest']
    print("\n✓ Fireworks SFT data prepared")
    print(f"  {result['num_train']} training examples "
          f"(max {manifest['token_stats']['train']['max_tokens']} tokens)")
    print(f"  {result['num_validation']} validation examples "
          f"(max {manifest['token_stats']['validation']['max_tokens']}"
          f" tokens)")
    print(f"  Manifest: {result['manifest_file']}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
