"""Fireworks fine-tuning pipeline: seed-disjoint prep, eval hash
check, traces, and Fireworks SFT validation. No network, no Fireworks.
"""

import json
import re

import pytest

from src.finetune_prep import FineTunePrep, parse_seed_list
from src.frontier_tier import FrontierTaskGenerator
from src.trace import build_trace, build_trace_parts, count_tokens


def _prepare(tmp_path, name='out', **kwargs):
    kwargs.setdefault('tier', 'frontier')
    return FineTunePrep().prepare(tmp_path / name, **kwargs)


class TestSeedSplit:
    def test_split_is_by_seed_and_deterministic(self, tmp_path):
        kwargs = {'seeds': [11, 12, 13], 'count': 8, 'val_seeds': [13]}
        first = _prepare(tmp_path, 'a', **kwargs)
        _prepare(tmp_path, 'b', **kwargs)
        for name in ('train.jsonl', 'validation.jsonl', 'manifest.json'):
            assert (tmp_path / 'a' / name).read_bytes() == \
                (tmp_path / 'b' / name).read_bytes()
        manifest = json.loads((tmp_path / 'a' / 'manifest.json')
                              .read_text())
        assert manifest['train_seeds'] == [11, 12]
        assert manifest['val_seeds'] == [13]
        assert manifest['excluded_seeds'] == [42, 1001, 1002]
        assert first['num_train'] == 16
        assert first['num_validation'] == 8
        assert manifest['counts']['train']['total'] == 16
        assert manifest['files']['train.jsonl']['rows'] == 16
        assert len(manifest['files']['train.jsonl']['sha256']) == 64

    def test_exclude_seed_refusal_writes_nothing(self, tmp_path):
        out = tmp_path / 'out'
        with pytest.raises(ValueError, match='locked eval seeds'):
            FineTunePrep().prepare(
                out, tier='frontier', seeds=[2001, 42], count=4)
        assert not out.exists()

    def test_eval_seed_as_validation_also_refused(self, tmp_path):
        with pytest.raises(ValueError, match='locked eval seeds'):
            FineTunePrep().prepare(
                tmp_path / 'out', tier='frontier', seeds=[2001, 1001],
                count=4, val_seeds=[1001])

    def test_tasks_dir_without_seed_declaration_refused(self, tmp_path,
                                                        monkeypatch,
                                                        capsys):
        # An undeclared tasks directory cannot bypass the locked-seed
        # refusal: the CLI requires PATH@SEED.
        import sys

        from src.finetune_prep import main
        tasks_dir = tmp_path / 'tasks78'
        FrontierTaskGenerator(seed=78).generate_all_tasks(tasks_dir,
                                                          num_tasks=4)
        monkeypatch.setattr(sys, 'argv', [
            'prep', '--tasks-dir', str(tasks_dir),
            '--output-dir', str(tmp_path / 'out')])
        assert main() == 1
        assert '@SEED' in capsys.readouterr().out
        assert not (tmp_path / 'out').exists()

    def test_empty_exclude_seeds_refused_without_override(self, tmp_path):
        # --exclude-seeds '' must not silently disable the lock.
        with pytest.raises(ValueError, match='allow-eval-seeds'):
            FineTunePrep().prepare(
                tmp_path / 'out', tier='frontier', seeds=[2001],
                count=4, exclude_seeds=[])
        assert not (tmp_path / 'out').exists()

    def test_allow_eval_seeds_override_warns_and_proceeds(self, tmp_path,
                                                          capsys):
        result = FineTunePrep().prepare(
            tmp_path / 'out', tier='frontier', seeds=[42], count=4,
            exclude_seeds=[], allow_eval_seeds=True)
        assert result['num_train'] == 4
        out = capsys.readouterr().out
        assert 'WARNING' in out and '--allow-eval-seeds' in out

    def test_no_args_refused_even_with_default_tasks_dir(self, tmp_path,
                                                         monkeypatch):
        # The implicit tasks/ default was generated at locked seed 42
        # by generate_tasks; finetune_prep with no arguments must
        # refuse rather than train on it sight-unseen.
        import sys

        from src.finetune_prep import main
        from src.generate_tasks import TaskGenerator
        TaskGenerator(seed=42).generate_all_tasks(tmp_path / 'tasks',
                                                  num_tasks=8)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, 'argv', ['prep'])
        assert main() == 1
        assert not (tmp_path / 'finetune_data').exists()

    def test_split_and_prepare_requires_declared_seed(self, tmp_path):
        from src.generate_tasks import TaskGenerator
        tasks_dir = tmp_path / 'tasks3001'
        TaskGenerator(seed=3001).generate_all_tasks(tasks_dir,
                                                    num_tasks=8)
        with pytest.raises(ValueError, match='generation seed'):
            FineTunePrep().split_and_prepare(tasks_dir, tmp_path / 'no')
        assert not (tmp_path / 'no').exists()
        result = FineTunePrep().split_and_prepare(
            tasks_dir, tmp_path / 'yes', seed=3001)
        assert result['num_train'] == 8

    def test_allow_eval_seeds_silent_when_lock_intact(self, tmp_path,
                                                      capsys):
        # The override warning only fires when a locked seed is
        # actually dropped from the exclusion list.
        FineTunePrep().prepare(tmp_path / 'out', tier='frontier',
                               seeds=[2001], count=4,
                               allow_eval_seeds=True)
        assert 'WARNING' not in capsys.readouterr().out

    def test_parse_seed_list(self):
        assert parse_seed_list('2001-2003,2007') == [2001, 2002, 2003,
                                                     2007]

    def test_families_filter(self, tmp_path):
        result = _prepare(tmp_path, seeds=[11], count=8,
                          families=['frontier_factor_exposure'])
        assert result['num_train'] == 2
        manifest = result['manifest']
        assert manifest['counts']['train']['by_family'] == \
            {'frontier_factor_exposure': 2}


    def test_mix_applies_to_directory_loaded_tiers(self, tmp_path):
        # Directory-loaded tasks have no explicit tier; it is inferred
        # from the task id/type so tier-named mix groups still apply.
        from src.generate_tasks import TaskGenerator
        easy_dir = tmp_path / 'easy77'
        TaskGenerator(seed=77).generate_all_tasks(easy_dir, num_tasks=8)
        frontier_dir = tmp_path / 'frontier78'
        FrontierTaskGenerator(seed=78).generate_all_tasks(
            frontier_dir, num_tasks=8)
        result = FineTunePrep().prepare(
            tmp_path / 'out',
            tasks_dirs=[(easy_dir, 201), (frontier_dir, 202)],
            mix={'easy': 0.25, 'frontier': 0.75})
        by_family = result['manifest']['counts']['train']['by_family']
        easy_rows = sum(n for t, n in by_family.items()
                        if not t.startswith('frontier_'))
        frontier_rows = sum(n for t, n in by_family.items()
                            if t.startswith('frontier_'))
        assert (easy_rows, frontier_rows) == (2, 7)


class TestEvalHashCheck:
    def test_hash_collision_refusal(self, tmp_path):
        eval_dir = tmp_path / 'eval99'
        FrontierTaskGenerator(seed=99).generate_all_tasks(eval_dir,
                                                          num_tasks=8)
        out = tmp_path / 'out'
        with pytest.raises(ValueError, match='SHA-256'):
            FineTunePrep().prepare(
                out, tasks_dirs=[(eval_dir, 99)], eval_dirs=[eval_dir])
        assert not out.exists()

    def test_validation_row_collision_refused(self, tmp_path):
        # The hash check covers validation rows too: training tasks
        # are disjoint here, the validation seed's tasks are the eval
        # tasks themselves.
        eval_dir = tmp_path / 'eval98'
        FrontierTaskGenerator(seed=98).generate_all_tasks(eval_dir,
                                                          num_tasks=8)
        train_dir = tmp_path / 'train97'
        FrontierTaskGenerator(seed=97).generate_all_tasks(train_dir,
                                                          num_tasks=8)
        out = tmp_path / 'out'
        with pytest.raises(ValueError, match='in validation'):
            FineTunePrep().prepare(
                out, tasks_dirs=[(train_dir, 97), (eval_dir, 98)],
                val_seeds=[98], eval_dirs=[eval_dir])
        assert not out.exists()

    def test_disjoint_eval_dirs_pass(self, tmp_path):
        eval_dir = tmp_path / 'eval98'
        FrontierTaskGenerator(seed=98).generate_all_tasks(eval_dir,
                                                          num_tasks=8)
        result = _prepare(tmp_path, seeds=[97], count=8,
                          eval_dirs=[eval_dir])
        assert result['num_train'] == 8
        assert result['manifest']['eval_check']['train_collisions'] == 0
        assert result['manifest']['eval_check']['eval_csv_hashes'] == 8


class TestTraceTarget:
    def test_trace_rows_use_reasoning_content(self, tmp_path):
        _prepare(tmp_path, seeds=[11], count=4, target='trace')
        row = json.loads((tmp_path / 'out' / 'train.jsonl')
                         .read_text().splitlines()[0])
        assistant = row['messages'][2]
        assert assistant['role'] == 'assistant'
        assert assistant['reasoning_content'].strip()
        answer = json.loads(assistant['content'])
        assert 'window_beta_1' in answer  # first task of the cycle is F1

    def test_trace_token_cap(self):
        gen = FrontierTaskGenerator(seed=5)
        task, payload = gen.generate_task_portfolio_pipeline()
        steps, answer = build_trace_parts(
            task, payload, task['expected'], max_tokens=120)
        assert count_tokens(steps) <= 120
        assert answer['final_return'] == \
            task['expected']['final_return']


class TestTraceCorrectness:
    """For every frontier family, seeds 1-20: the trace's final numbers
    equal the key, and the trace contains no other task's values."""

    def test_traces_match_keys_seeds_1_to_20(self):
        from src.scorer import categories_for_key_items
        traces = {}
        for seed in range(1, 21):
            gen = FrontierTaskGenerator(seed=seed)
            for maker in gen.task_generators():
                task, payload = maker()
                trace = build_trace(task, payload, task['expected'])
                answer = json.loads(trace.rsplit('Final answer:\n', 1)[1])
                expected = task['expected']
                assert set(answer['problems']) == set(
                    categories_for_key_items(expected['problems_found'])), \
                    (seed, task['id'])
                for field, value in expected.items():
                    if field == 'problems_found':
                        continue
                    got = answer[field]
                    if isinstance(value, float):
                        assert got == pytest.approx(value, abs=1e-6), \
                            (seed, task['id'], field)
                    else:
                        assert got == value, (seed, task['id'], field)
                if task['type'] == 'frontier_rolling_risk':
                    traces[seed] = (trace, answer)

    def test_trace_contains_no_other_tasks_values(self):
        built = []
        for seed in (31, 32):
            gen = FrontierTaskGenerator(seed=seed)
            task, payload = gen.generate_task_rolling_risk()
            built.append((build_trace(task, payload, task['expected']),
                          task['expected']))
        (trace_a, exp_a), (trace_b, exp_b) = built
        assert json.dumps(exp_b, sort_keys=True) not in trace_a
        assert json.dumps(exp_a, sort_keys=True) not in trace_b
        beta_b = f"{exp_b['window_beta_1']:.6f}"
        if abs(exp_a['window_beta_1'] - exp_b['window_beta_1']) > 1e-3:
            assert beta_b not in trace_a


class TestTracePrecision:
    def test_printed_cov_var_reproduce_window_betas(self):
        # Intermediates are printed at enough precision to reproduce
        # the final values; the final values themselves are unchanged.
        gen = FrontierTaskGenerator(seed=7)
        task, payload = gen.generate_task_rolling_risk()
        steps, answer = build_trace_parts(
            task, payload, task['expected'], max_tokens=None)
        pairs = re.findall(
            r"cov=([-\d.]+), var=([-\d.]+), beta=([-\d.]+)", steps)
        assert len(pairs) == 4
        # Fixed-point intermediates: no scientific notation anywhere.
        assert not re.search(r"\d[eE][+-]\d", steps)
        for i, (cov, var, beta) in enumerate(pairs, 1):
            expected = task['expected'][f'window_beta_{i}']
            assert float(cov) / float(var) == pytest.approx(
                expected, abs=1e-6)
            assert float(beta) == pytest.approx(expected, abs=1e-6)
            assert answer[f'window_beta_{i}'] == pytest.approx(
                expected, abs=1e-6)


class TestFireworksValidation:
    def _rows(self, n=3, content='{"ok": true}'):
        return [{'messages': [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'task'},
            {'role': 'assistant', 'content': content}]}
            for _ in range(n)]

    def test_valid_rows_pass(self):
        from src.finetune import validate_rows
        assert validate_rows(self._rows(), 32768) == []

    def test_row_count_rule(self):
        from src.finetune import validate_rows
        errors = validate_rows(self._rows(2), 32768)
        assert any('3 to 3000000' in e for e in errors)

    def test_role_and_assistant_rules(self):
        from src.finetune import validate_rows
        bad_role = self._rows()
        bad_role[0]['messages'][1]['role'] = 'human'
        assert any('invalid role' in e
                   for e in validate_rows(bad_role, 32768))
        empty = self._rows()
        empty[1]['messages'][2]['content'] = '  '
        assert any('non-empty assistant' in e
                   for e in validate_rows(empty, 32768))

    def test_token_length_check_fails_on_overflow(self):
        from src.finetune import validate_rows
        rows = self._rows(content='word ' * 500)
        errors = validate_rows(rows, max_context_length=50)
        assert errors and all('max-context-length' in e for e in errors)
        assert validate_rows(rows, max_context_length=32768) == []

    def test_main_prints_plan_but_executes_nothing(self, tmp_path,
                                                  monkeypatch, capsys):
        import sys

        from src.finetune import main
        train = tmp_path / 'train.jsonl'
        val = tmp_path / 'validation.jsonl'
        for path in (train, val):
            path.write_text("".join(json.dumps(r) + "\n"
                                    for r in self._rows()))
        monkeypatch.setattr(sys, 'argv', [
            'finetune', '--train-file', str(train),
            '--validation-file', str(val), '--dry-run'])
        assert main() == 0
        out = capsys.readouterr().out
        assert 'NOT executed' in out
        assert 'firectl dataset create glimmer-sft-answer' in out
        assert '--base-model accounts/fireworks/models/' \
               'muse-glimmer-30b' in out
        assert '--lora-rank 16' in out and '--epochs 2' in out
        assert '--max-context-length 32768' in out
        assert '--evaluation-dataset glimmer-sft-answer-eval' in out
        assert '--output-model muse-glimmer-30b-lora-answer' in out

    def test_trace_target_gets_distinct_names(self, tmp_path,
                                              monkeypatch, capsys):
        import sys

        from src.finetune import main
        train = tmp_path / 'train.jsonl'
        val = tmp_path / 'validation.jsonl'
        for path in (train, val):
            path.write_text("".join(json.dumps(r) + "\n"
                                    for r in self._rows()))
        monkeypatch.setattr(sys, 'argv', [
            'finetune', '--train-file', str(train),
            '--validation-file', str(val), '--target', 'trace'])
        assert main() == 0
        out = capsys.readouterr().out
        assert 'firectl dataset create glimmer-sft-trace' in out
        assert '--evaluation-dataset glimmer-sft-trace-eval' in out
        assert '--output-model muse-glimmer-30b-lora-trace' in out

    def test_target_manifest_mismatch_refused(self, tmp_path,
                                              monkeypatch):
        import sys

        from src.finetune import main
        train = tmp_path / 'train.jsonl'
        val = tmp_path / 'validation.jsonl'
        for path in (train, val):
            path.write_text("".join(json.dumps(r) + "\n"
                                    for r in self._rows()))
        (tmp_path / 'manifest.json').write_text(
            json.dumps({'target': 'trace'}))
        monkeypatch.setattr(sys, 'argv', [
            'finetune', '--train-file', str(train),
            '--validation-file', str(val), '--target', 'answer'])
        assert main() == 1
        monkeypatch.setattr(sys, 'argv', [
            'finetune', '--train-file', str(train),
            '--validation-file', str(val), '--target', 'trace'])
        assert main() == 0

    def test_main_rejects_overlong_rows(self, tmp_path, monkeypatch):
        import sys

        from src.finetune import main
        train = tmp_path / 'train.jsonl'
        train.write_text("".join(
            json.dumps(r) + "\n"
            for r in self._rows(content='word ' * 500)))
        monkeypatch.setattr(sys, 'argv', [
            'finetune', '--train-file', str(train),
            '--validation-file', str(tmp_path / 'missing.jsonl'),
            '--max-context-length', '50'])
        assert main() == 1
