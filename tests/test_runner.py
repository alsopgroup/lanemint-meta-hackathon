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

        def run_all_tasks(self, tasks_dir, output_dir, limit=None):
            seen['tasks_dir'] = tasks_dir
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
    assert captured['timeout'] == 300.0


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
