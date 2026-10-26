"""Runner token-cap defaults: hard tier 32000, easy tier 16000."""

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


def test_easy_tier_defaults_to_16000(monkeypatch):
    runner, fake = _runner_with_fake(monkeypatch)
    runner.run_task({'id': 'task_003', 'type': 'return_calculation', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 16000


def test_env_overrides_tier_default(monkeypatch):
    runner, fake = _runner_with_fake(monkeypatch)
    monkeypatch.setenv('META_MAX_TOKENS', '20000')
    runner.run_task({'id': 'task_h_1003', 'type': 'rolling_beta', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 20000


def test_explicit_max_tokens_wins(monkeypatch):
    runner, fake = _runner_with_fake(monkeypatch, max_tokens=45000)
    monkeypatch.setenv('META_MAX_TOKENS', '20000')
    runner.run_task({'id': 'task_003', 'type': 'return_calculation', 'prompt': 'p'}, 'data')
    assert fake.kwargs['max_tokens'] == 45000
