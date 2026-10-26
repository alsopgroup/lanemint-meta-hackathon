"""Tests for the Muse Data Check web service (app/).

Run from the app/ directory:  pytest tests/ -v
Uses FastAPI's TestClient; the live Muse call is monkeypatched, so no
API key or network is needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import muse_client
import pytest
from fastapi.testclient import TestClient
from main import app
from rate_limit import guard

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_guard():
    guard.reset()
    yield
    guard.reset()

CSV = "date,close\n2026-03-02,100.0\n2026-03-03,101.0\n"


def test_healthz(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "mode": "simulated"}


def test_config_never_exposes_key(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "secret-test-key")
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"live": True, "mode": "live", "model": "muse-spark-1.3"}
    assert "secret-test-key" not in resp.text


def test_index_serves_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Muse data check" in resp.text
    assert "Run a check" in resp.text


def test_check_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    resp = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert resp.status_code == 503
    assert resp.json()["mode"] == "simulated"


def test_check_rejects_bad_payloads():
    assert client.post("/api/check", json={"csv": ""}).status_code == 400
    assert client.post("/api/check", json={"filename": "x.csv"}).status_code == 400
    assert client.post("/api/check", json={"csv": 42}).status_code == 400
    assert client.post("/api/check", content="not json").status_code == 400


def test_check_live_success(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")

    def fake_create(self, **kwargs):  # noqa: ANN001, ANN202 - test double
        # The browser may send any model name; the server must pin its own.
        assert kwargs["model"] == "muse-spark-1.3"
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_tokens"] == muse_client.DEFAULT_MAX_TOKENS
        message = type(
            "M", (), {"content": '{"findings": [{"type": "dups", "title": '
                                '"Duplicate bar", "rows": [1, 2], '
                                '"explain": "Rows 1 and 2 are identical."}]}'}
        )()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()

    _stub_openai(monkeypatch, fake_create)
    resp = client.post(
        "/api/check",
        json={"filename": "x.csv", "csv": CSV, "model": "bogus-expensive-model"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "muse-spark-1.3"
    assert body["findings"] == [
        {"type": "dups", "title": "Duplicate bar", "rows": [1, 2],
         "explain": "Rows 1 and 2 are identical."}
    ]
    assert isinstance(body["latency_ms"], int)


class _Chat:
    def __init__(self, create):  # noqa: ANN001
        self.completions = self
        self._create = create

    def create(self, **kwargs):  # noqa: ANN003, ANN202
        return self._create(None, **kwargs)


def _stub_openai(monkeypatch, fake_create):  # noqa: ANN001
    import openai

    class FakeOpenAI:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.chat = _Chat(fake_create)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)


def _live_ok_create(self, **kwargs):  # noqa: ANN001, ANN202
    message = type(
        "M", (), {"content": '{"findings": [{"type": "dups", "title": "Dup", '
                            '"rows": [1], "explain": "e"}]}'}
    )()
    return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def test_check_oversized_csv_413():
    big = "date,close\n" + ("2026-01-05,1\n" * 10)
    big = big + ("x" * (muse_client.MAX_CSV_CHARS + 1))
    resp = client.post("/api/check", json={"filename": "x.csv", "csv": big})
    assert resp.status_code == 413


def test_check_too_many_rows_413(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    rows = "date,close\n" + "".join(f"2026-01-{(i % 28) + 1:02d},1\n" for i in range(501))
    resp = client.post("/api/check", json={"filename": "x.csv", "csv": rows})
    assert resp.status_code == 413


def test_check_cross_origin_403(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    resp = client.post(
        "/api/check", json={"filename": "x.csv", "csv": CSV},
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 403
    same_origin = client.post(
        "/api/check", json={"filename": "x.csv", "csv": CSV},
        headers={"Origin": "http://testserver"},
    )
    assert same_origin.status_code == 503  # origin ok; falls back (no key)


def test_check_rate_limited_per_ip(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "1")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    _stub_openai(monkeypatch, _live_ok_create)
    first = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    second = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1


def test_check_daily_budget_cap(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "100")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    monkeypatch.setenv("LIVE_DAILY_BUDGET_USD", "0.000001")
    _stub_openai(monkeypatch, _live_ok_create)
    first = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    second = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert first.status_code == 200
    assert second.status_code == 429
    assert "budget" in second.json()["error"]


def test_parse_findings_scalar_rows():
    content = (
        '{"findings": ['
        '{"type": "dups", "title": "A", "rows": 2, "explain": "e"},'
        '{"type": "gaps", "title": "B", "rows": "3", "explain": "e"},'
        '{"type": "prices", "title": "C", "rows": "bad", "explain": "e"}]}'
    )
    findings = muse_client._parse_findings(content, num_rows=5)
    assert findings[0]["rows"] == [2]
    assert findings[1]["rows"] == [3]
    assert findings[2]["rows"] == []


def test_parse_findings_validates_and_clamps():
    content = (
        '{"findings": ['
        '{"type": "dups", "title": "Dup", "rows": [1, 99, "x"], "explain": "e"},'
        '{"type": "nope", "title": "Bad type", "rows": [1], "explain": "e"},'
        '{"type": "gaps", "title": "Gap", "rows": [2], "explain": "e"}]}'
    )
    findings = muse_client._parse_findings(content, num_rows=3)
    assert [f["type"] for f in findings] == ["dups", "gaps"]
    assert findings[0]["rows"] == [1]


def test_parse_findings_rejects_non_json():
    with pytest.raises(muse_client.MuseUpstreamError):
        muse_client._parse_findings("not json at all", num_rows=3)


def test_check_csv_requires_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(muse_client.MuseNotConfigured):
        muse_client.check_csv("x.csv", CSV)
