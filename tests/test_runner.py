"""Runner token-cap defaults: hard tier 32000, easy tier 16000."""

from pathlib import Path

from src.runner import TaskRunner


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs

        class _Msg:
            content = '{"problems": []}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _runner_with_fake(monkeypatch, **runner_kwargs):
    monkeypatch.delenv('META_MAX_TOKENS', raising=False)
    runner = TaskRunner(dry_run=True, **runner_kwargs)
    runner.dry_run = False
    fake = _FakeCompletions()
    runner.client = type('C', (), {'chat': type('Chat', (), {
        'completions': fake})()})()
    return runner, fake


def test_hard_tier_defaults_to_32000(monkeypatch):
    runner, fake = _runner_with_fake(monkeypatch)
    runner.run_task({'id': 'task_h_1003', 'type': 'rolling_beta', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 32000


def test_frontier_tier_defaults_to_32000(monkeypatch):
    runner, fake = _runner_with_fake(monkeypatch)
    runner.run_task({'id': 'task_f_2001', 'type': 'frontier_rolling_risk',
                     'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 32000


def test_tier_flag_selects_tasks_dir(monkeypatch, tmp_path):
    import sys

    from src import runner as runner_mod
    seen = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def run_all_tasks(self, tasks_dir, output_dir, limit=None, resume=False,
                          per_family=None):
            seen['tasks_dir'] = tasks_dir
            seen['per_family'] = per_family
            return []

    monkeypatch.setattr(runner_mod, 'TaskRunner', FakeRunner)
    monkeypatch.setattr(sys, 'argv', ['runner', '--tier', 'frontier',
                                      '--output-dir', str(tmp_path)])
    (tmp_path.parent / 'tasks_frontier').mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path.parent)
    assert runner_mod.main() == 0
    assert seen['tasks_dir'] == Path('tasks_frontier')


def test_easy_tier_defaults_to_16000(monkeypatch):
    runner, fake = _runner_with_fake(monkeypatch)
    runner.run_task({'id': 'task_003', 'type': 'return_calculation', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 16000


def test_run_task_logs_finish_reason_and_usage(monkeypatch):
    from types import SimpleNamespace

    runner, _fake = _runner_with_fake(monkeypatch)
    usage = SimpleNamespace(prompt_tokens=111, completion_tokens=222,
                            total_tokens=333)
    choice = SimpleNamespace(
        finish_reason='length',
        message=SimpleNamespace(content='{"problems": []}'))
    runner.client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(choices=[choice], usage=usage))))
    result = runner.run_task({'id': 'task_f_2001',
                              'type': 'frontier_rolling_risk', 'prompt': 'p'}, 'd')
    assert result['finish_reason'] == 'length'
    assert result['prompt_tokens'] == 111
    assert result['completion_tokens'] == 222
    assert result['total_tokens'] == 333


def test_client_never_retries_and_has_timeout(monkeypatch):
    import openai

    from src.runner import TaskRunner
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, 'OpenAI', FakeOpenAI)
    TaskRunner(api_key='test-key')
    assert captured['max_retries'] == 0
    assert captured['timeout'] == 900.0


class _StubClient:
    """Fake dry-run client; fails on the Nth call when fail_at is set."""

    def __init__(self, fail_at=None, exc=None):
        self.calls = 0
        self.fail_at = fail_at
        self.exc = exc or ConnectionError('sensitive request detail')

    def chat_completions_create(self, messages, **kwargs):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise self.exc
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"problems": []}'}}]}


class FakeApiError(Exception):
    status_code = 429


def _tasks_dir(tmp_path):
    import json as _json
    tmp_path.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(1, 4):
        (tmp_path / f'data_{i}.csv').write_text('date,close\n2024-01-01,1\n')
        tasks.append({'id': f'task_{i:03d}', 'type': 'return_calculation',
                      'prompt': 'p', 'data_file': f'data_{i}.csv'})
    (tmp_path / 'tasks.json').write_text(_json.dumps(tasks))
    return tmp_path


def _typed_tasks_dir(tmp_path, types=("fam_a", "fam_b")):
    """Tasks dir with interleaved types: id i has type types[i % len]."""
    import json as _json
    tmp_path.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(1, 7):
        (tmp_path / f'data_{i}.csv').write_text('date,close\n2024-01-01,1\n')
        tasks.append({'id': f'task_{i:03d}', 'type': types[i % len(types)],
                      'prompt': 'p', 'data_file': f'data_{i}.csv'})
    (tmp_path / 'tasks.json').write_text(_json.dumps(tasks))
    return tmp_path


def _runner_with_client(client, model_name=None, reasoning_effort=None,
                        max_tokens=None):
    from src.runner import TaskRunner
    runner = TaskRunner(dry_run=True, model_name=model_name,
                        reasoning_effort=reasoning_effort,
                        max_tokens=max_tokens)
    runner.client = client
    runner.dry_run = True
    return runner


def test_mid_run_failure_keeps_results_and_records_no_error_text(tmp_path):
    import json as _json
    tasks_dir = _tasks_dir(tmp_path / 'tasks')
    tasks_dir.mkdir(exist_ok=True)
    out = tmp_path / 'out'
    runner = _runner_with_client(_StubClient(fail_at=2))

    results = runner.run_all_tasks(tasks_dir, out)  # must not raise
    assert len(results) == 3
    assert results[1]['error'] == {'type': 'ConnectionError', 'status': None}
    assert results[1]['response'] == ''
    # results.json holds everything; results.jsonl streamed all 3 entries
    saved = _json.loads((out / 'results.json').read_text())
    assert [r['task_id'] for r in saved] == ['task_001', 'task_002', 'task_003']
    lines = (out / 'results.jsonl').read_text().strip().splitlines()
    assert len(lines) == 3
    assert _json.loads(lines[1])['error']['type'] == 'ConnectionError'
    # never the error text/body
    assert 'sensitive request detail' not in (out / 'results.json').read_text()
    assert 'sensitive request detail' not in (out / 'results.jsonl').read_text()


def test_error_status_code_recorded(tmp_path):
    tasks_dir = _tasks_dir(tmp_path / 'tasks')
    tasks_dir.mkdir(exist_ok=True)
    runner = _runner_with_client(
        _StubClient(fail_at=1, exc=FakeApiError('nope')))
    results = runner.run_all_tasks(tasks_dir, tmp_path / 'out')
    assert results[0]['error'] == {'type': 'FakeApiError', 'status': 429}


def test_resume_skips_successes_and_retries_errors(tmp_path):
    tasks_dir = _tasks_dir(tmp_path / 'tasks')
    tasks_dir.mkdir(exist_ok=True)
    out = tmp_path / 'out'
    first = _runner_with_client(_StubClient(fail_at=2))
    first.run_all_tasks(tasks_dir, out)

    second_client = _StubClient()
    second = _runner_with_client(second_client)
    results = second.run_all_tasks(tasks_dir, out, resume=True)
    assert second_client.calls == 1  # only the errored task was retried
    assert all('error' not in r for r in results)
    assert [r['task_id'] for r in results] == ['task_001', 'task_002', 'task_003']

    third_client = _StubClient()
    third = _runner_with_client(third_client)
    third.run_all_tasks(tasks_dir, out, resume=True)
    assert third_client.calls == 0  # everything already in the output


def test_resume_repairs_partial_last_line_with_exactly_one_call(tmp_path):
    import json as _json
    tasks_dir = _tasks_dir(tmp_path / 'tasks')
    out = tmp_path / 'out'
    _runner_with_client(_StubClient()).run_all_tasks(tasks_dir, out)

    # Simulate a crash mid-write: truncate the last JSONL record.
    jsonl = out / 'results.jsonl'
    data = jsonl.read_bytes()
    last_start = data.rfind(b'\n', 0, len(data) - 1) + 1
    jsonl.write_bytes(data[:last_start] + data[last_start:last_start + 10])
    assert not jsonl.read_bytes().endswith(b'\n')

    client = _StubClient()
    results = _runner_with_client(client).run_all_tasks(
        tasks_dir, out, resume=True)
    assert client.calls == 1  # only the task whose record was cut
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 3
    parsed = [_json.loads(line) for line in lines]  # every line is whole
    assert [p['task_id'] for p in parsed] == ['task_001', 'task_002', 'task_003']
    assert all('error' not in r for r in results)


def test_without_resume_starts_fresh(tmp_path):
    import json as _json
    tasks_dir = _tasks_dir(tmp_path / 'tasks')
    out = tmp_path / 'out'
    _runner_with_client(_StubClient()).run_all_tasks(tasks_dir, out)

    class AlwaysFail:
        def __init__(self):
            self.calls = 0

        def chat_completions_create(self, messages, **kwargs):
            self.calls += 1
            raise ConnectionError('down')

    client = AlwaysFail()
    results = _runner_with_client(client).run_all_tasks(tasks_dir, out)
    assert client.calls == 3  # nothing skipped, nothing appended to stale
    lines = (out / 'results.jsonl').read_text().strip().splitlines()
    assert len(lines) == 3
    assert all('error' in r for r in results)
    saved = _json.loads((out / 'results.json').read_text())
    assert all('error' in r for r in saved)  # stale successes replaced


def test_resume_refuses_different_model(tmp_path):
    import pytest

    tasks_dir = _tasks_dir(tmp_path / 'tasks')
    out = tmp_path / 'out'
    results = _runner_with_client(
        _StubClient(), model_name='model-a').run_all_tasks(tasks_dir, out)
    assert results[0]['model'] == 'model-a'
    assert results[0]['max_tokens'] == 16000

    client = _StubClient()
    with pytest.raises(ValueError, match='--resume refused'):
        _runner_with_client(client, model_name='model-b').run_all_tasks(
            tasks_dir, out, resume=True)
    assert client.calls == 0


def test_reasoning_effort_sent_and_recorded(monkeypatch):
    monkeypatch.delenv('META_REASONING_EFFORT', raising=False)
    runner, fake = _runner_with_fake(monkeypatch, reasoning_effort='high')
    result = runner.run_task({'id': 'task_f_2001',
                              'type': 'frontier_rolling_risk', 'prompt': 'p'}, 'd')
    assert fake.kwargs['reasoning_effort'] == 'high'
    assert result['reasoning_effort'] == 'high'


def test_reasoning_effort_omitted_by_default(monkeypatch):
    monkeypatch.delenv('META_REASONING_EFFORT', raising=False)
    runner, fake = _runner_with_fake(monkeypatch)
    result = runner.run_task({'id': 'task_f_2001',
                              'type': 'frontier_rolling_risk', 'prompt': 'p'}, 'd')
    assert 'reasoning_effort' not in fake.kwargs
    assert result['reasoning_effort'] is None


def test_resume_refuses_reasoning_effort_mismatch(tmp_path):
    import pytest

    tasks_dir = _tasks_dir(tmp_path / 'tasks')
    out = tmp_path / 'out'
    _runner_with_client(_StubClient(),
                        reasoning_effort='low').run_all_tasks(tasks_dir, out)
    client = _StubClient()
    with pytest.raises(ValueError, match='--resume refused'):
        _runner_with_client(client, reasoning_effort='high').run_all_tasks(
            tasks_dir, out, resume=True)
    assert client.calls == 0


def test_resume_refuses_max_tokens_mismatch(tmp_path):
    import pytest

    tasks_dir = _tasks_dir(tmp_path / 'tasks')
    out = tmp_path / 'out'
    _runner_with_client(_StubClient(), max_tokens=32000).run_all_tasks(
        tasks_dir, out)
    # Same cap (e.g. re-running default effort at 65536 end-to-end):
    # a cap recorded as 65536 resumes only under 65536.
    client = _StubClient()
    with pytest.raises(ValueError, match='--resume refused.*max_tokens'):
        _runner_with_client(client, max_tokens=65536).run_all_tasks(
            tasks_dir, out, resume=True)
    assert client.calls == 0
    # And the matching cap resumes without any new calls.
    client2 = _StubClient()
    _runner_with_client(client2, max_tokens=32000).run_all_tasks(
        tasks_dir, out, resume=True)
    assert client2.calls == 0


def test_per_family_selects_first_n_of_each_type_reproducibly(tmp_path):
    from collections import Counter
    tasks_dir = _typed_tasks_dir(tmp_path / 'tasks')
    runner = _runner_with_client(_StubClient())
    results = runner.run_all_tasks(tasks_dir, tmp_path / 'out1',
                                   per_family=2)
    assert [r['task_id'] for r in results] == \
        ['task_001', 'task_002', 'task_003', 'task_004']
    assert Counter(r['task_type'] for r in results) == \
        {'fam_a': 2, 'fam_b': 2}
    # Same selection again: deterministic, file order.
    again = _runner_with_client(_StubClient()).run_all_tasks(
        tasks_dir, tmp_path / 'out2', per_family=2)
    assert [r['task_id'] for r in again] == [r['task_id'] for r in results]


def test_per_family_default_runs_everything(tmp_path):
    tasks_dir = _typed_tasks_dir(tmp_path / 'tasks')
    results = _runner_with_client(_StubClient()).run_all_tasks(
        tasks_dir, tmp_path / 'out')
    assert len(results) == 6  # default behaviour unchanged


def test_env_overrides_tier_default(monkeypatch):
    runner, fake = _runner_with_fake(monkeypatch)
    monkeypatch.setenv('META_MAX_TOKENS', '20000')
    runner.run_task({'id': 'task_h_1003', 'type': 'rolling_beta', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 20000


def test_env_example_leaves_token_defaults_untouched(monkeypatch):
    """Loading .env.example (cp .env.example .env) must not re-impose 16000."""
    from dotenv import dotenv_values
    env_file = Path(__file__).resolve().parent.parent / '.env.example'
    values = dotenv_values(env_file)
    assert values.get('META_MAX_TOKENS') is None, \
        '.env.example must not set META_MAX_TOKENS (it would override the tier defaults)'
    runner, fake = _runner_with_fake(monkeypatch)
    for key, value in values.items():
        if value is not None:
            monkeypatch.setenv(key, value)
    runner.run_task({'id': 'task_h_1003', 'type': 'rolling_beta', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 32000
    runner.run_task({'id': 'task_003', 'type': 'return_calculation', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 16000


def test_explicit_max_tokens_wins(monkeypatch):
    runner, fake = _runner_with_fake(monkeypatch, max_tokens=45000)
    monkeypatch.setenv('META_MAX_TOKENS', '20000')
    runner.run_task({'id': 'task_003', 'type': 'return_calculation', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 45000
