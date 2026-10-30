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


def test_config_simulated_without_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    resp = client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json() == {"live": False, "mode": "simulated",
                           "model": "muse-spark-1.3"}


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
    # One request = one billed attempt: the SDK must not retry.
    assert _stub_openai.last_client_kwargs.get("max_retries") == 0
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
            _stub_openai.last_client_kwargs = kwargs
            self.chat = _Chat(fake_create)

    _stub_openai.last_client_kwargs = {}
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


def _live_full_usage_create(self, **kwargs):  # noqa: ANN001, ANN202
    """Live stub whose reported usage costs ~one worst-case reserve."""
    message = type(
        "M", (), {"content": '{"findings": [{"type": "dups", "title": "Dup", '
                            '"rows": [1], "explain": "e"}]}'}
    )()
    usage = type("U", (), {"prompt_tokens": 400, "completion_tokens": 4096})()
    return type("R", (), {"choices": [type("C", (), {"message": message})()],
                          "usage": usage})()


def _set_synthetic_pricing(monkeypatch):
    """Synthetic test-only pricing (not a real provider price)."""
    monkeypatch.setenv("META_INPUT_COST_PER_MTOK", "1000000")
    monkeypatch.setenv("META_OUTPUT_COST_PER_MTOK", "1000000")


def _unset_pricing(monkeypatch):
    monkeypatch.delenv("META_INPUT_COST_PER_MTOK", raising=False)
    monkeypatch.delenv("META_OUTPUT_COST_PER_MTOK", raising=False)


def test_pricing_unset_means_cost_unknown(monkeypatch):
    _unset_pricing(monkeypatch)
    # No pricing is embedded in the module, and cost is not guessed.
    assert not hasattr(muse_client, "INPUT_USD_PER_TOKEN")
    assert not hasattr(muse_client, "OUTPUT_USD_PER_TOKEN")
    assert muse_client.worst_case_cost_usd(CSV) is None


@pytest.mark.parametrize("raw", ["not-a-number", "", "-1", "nan", "inf"])
def test_pricing_invalid_means_cost_unknown(monkeypatch, raw):
    monkeypatch.setenv("META_INPUT_COST_PER_MTOK", raw)
    monkeypatch.setenv("META_OUTPUT_COST_PER_MTOK", "1000000")
    assert muse_client.worst_case_cost_usd(CSV) is None


def test_pricing_partially_set_means_cost_unknown(monkeypatch):
    monkeypatch.setenv("META_INPUT_COST_PER_MTOK", "1000000")
    monkeypatch.delenv("META_OUTPUT_COST_PER_MTOK", raising=False)
    assert muse_client.worst_case_cost_usd(CSV) is None


def test_pricing_read_from_environment(monkeypatch):
    _set_synthetic_pricing(monkeypatch)
    # Synthetic pricing is one unit per token, so the worst case is
    # simply the estimated input tokens plus the output cap.
    input_tokens = max(
        1, (len(muse_client._SYSTEM_PROMPT) + len(CSV) + 64) // 4)
    assert muse_client.worst_case_cost_usd(CSV) == (
        input_tokens + muse_client.DEFAULT_MAX_TOKENS)


def test_check_csv_cost_unknown_without_pricing(monkeypatch):
    _unset_pricing(monkeypatch)
    monkeypatch.setenv("META_API_KEY", "test-key")
    _stub_openai(monkeypatch, _live_full_usage_create)
    result = muse_client.check_csv("x.csv", CSV)
    assert result["estimated_cost_usd"] is None


def test_check_csv_cost_from_env_pricing(monkeypatch):
    _set_synthetic_pricing(monkeypatch)
    monkeypatch.setenv("META_API_KEY", "test-key")
    _stub_openai(monkeypatch, _live_full_usage_create)
    result = muse_client.check_csv("x.csv", CSV)
    # Stub reports 400 prompt + 4096 completion tokens, at the
    # synthetic one-unit-per-token test pricing.
    assert result["estimated_cost_usd"] == 400 + 4096


def test_live_without_pricing_enforces_call_count_only(monkeypatch):
    _unset_pricing(monkeypatch)
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "100")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    monkeypatch.setenv("LIVE_DAILY_MAX_CALLS", "1")
    _stub_openai(monkeypatch, _live_ok_create)
    first = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    second = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert first.status_code == 200
    assert second.status_code == 429
    assert "daily live-call cap" in second.json()["error"]


def test_check_daily_budget_cap(monkeypatch):
    _set_synthetic_pricing(monkeypatch)
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "100")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    # Budget fits one reserved worst case (settled at ~full usage)
    # but not a second reservation on top of that spend.
    budget = (muse_client.worst_case_cost_usd(CSV) or 0) * 1.5
    monkeypatch.setenv("LIVE_DAILY_BUDGET_USD", str(budget))
    _stub_openai(monkeypatch, _live_full_usage_create)
    first = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    second = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert first.status_code == 200
    assert second.status_code == 429
    assert "budget" in second.json()["error"]


def test_malformed_responses_still_hit_budget(monkeypatch):
    """Billed-but-unparseable responses must count against the budget."""
    _set_synthetic_pricing(monkeypatch)
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "100")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    budget = (muse_client.worst_case_cost_usd(CSV) or 0) * 1.5
    monkeypatch.setenv("LIVE_DAILY_BUDGET_USD", str(budget))

    def garbage_create(self, **kwargs):  # noqa: ANN001, ANN202
        message = type("M", (), {"content": "definitely not json"})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

    _stub_openai(monkeypatch, garbage_create)
    first = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    second = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert first.status_code == 502
    # The failed call settled at its reserved worst case, so the next
    # reservation no longer fits the budget.
    assert second.status_code == 429
    assert "budget" in second.json()["error"]


def test_upstream_error_text_never_reaches_browser(monkeypatch, capsys):
    monkeypatch.setenv("META_API_KEY", "test-key")

    class FakeAPIError(RuntimeError):
        status_code = 401

    def exploding_create(self, **kwargs):  # noqa: ANN001, ANN202
        raise FakeAPIError("upstream echoed header authorization: Bearer FAKEKEY123")

    _stub_openai(monkeypatch, exploding_create)
    resp = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert resp.status_code == 502
    assert resp.json()["error"] == "Muse request failed"
    assert "FAKEKEY123" not in resp.text
    logged = capsys.readouterr().out
    assert "FAKEKEY123" not in logged
    assert "FakeAPIError status=401" in logged
    assert "Bearer" not in logged


def test_parse_error_log_has_no_response_body(capsys):
    secret = "FAKEKEY123 echoed in the body"
    with pytest.raises(muse_client.MuseUpstreamError):
        muse_client._parse_findings(secret, num_rows=3)
    logged = capsys.readouterr().out
    assert "FAKEKEY123" not in logged
    assert "JSONDecodeError" in logged


def test_zero_limit_is_a_hard_block(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "0")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    _stub_openai(monkeypatch, _live_ok_create)
    resp = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert resp.status_code == 429
    assert "rate limit" in resp.json()["error"]


def test_zero_budget_is_a_hard_block(monkeypatch):
    # Even with pricing unset (cost unknown), a 0 budget blocks.
    _unset_pricing(monkeypatch)
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "100")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    monkeypatch.setenv("LIVE_DAILY_BUDGET_USD", "0")
    _stub_openai(monkeypatch, _live_ok_create)
    resp = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert resp.status_code == 429
    assert "budget" in resp.json()["error"]


def test_client_ip_prefers_railway_x_real_ip(monkeypatch):
    # Railway's edge sets X-Real-IP to the observed visitor, so it
    # keys the bucket: forged X-Forwarded-For values must not mint a
    # fresh bucket, and the last forwarded hop (possibly Railway's
    # own edge address) must not merge all visitors into one bucket.
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "1")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    _stub_openai(monkeypatch, _live_ok_create)
    first = client.post(
        "/api/check", json={"filename": "x.csv", "csv": CSV},
        headers={"X-Real-IP": "203.0.113.7",
                 "X-Forwarded-For": "198.51.100.1, 192.0.2.10"},
    )
    second = client.post(
        "/api/check", json={"filename": "x.csv", "csv": CSV},
        headers={"X-Real-IP": "203.0.113.7",
                 "X-Forwarded-For": "198.51.100.2, 192.0.2.11"},
    )
    third = client.post(
        "/api/check", json={"filename": "x.csv", "csv": CSV},
        headers={"X-Real-IP": "203.0.113.8",
                 "X-Forwarded-For": "198.51.100.1, 192.0.2.10"},
    )
    assert first.status_code == 200
    assert second.status_code == 429  # same X-Real-IP despite forged XFF
    assert third.status_code == 200  # different visitor, fresh bucket


def test_client_ip_forwarded_for_fallback(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "1")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    _stub_openai(monkeypatch, _live_ok_create)
    first = client.post(
        "/api/check", json={"filename": "x.csv", "csv": CSV},
        headers={"X-Forwarded-For": "198.51.100.1, 192.0.2.10"},
    )
    second = client.post(
        "/api/check", json={"filename": "x.csv", "csv": CSV},
        headers={"X-Forwarded-For": "198.51.100.1, 192.0.2.11"},
    )
    assert first.status_code == 200
    assert second.status_code == 429  # same first hop, no X-Real-IP sent


def test_check_chunked_oversized_body_413():
    # A generator body makes httpx use chunked encoding (no
    # Content-Length), which must not bypass the body size cap.
    def oversized():
        yield b'{"filename": "x.csv", "csv": "date,close\n'
        yield b"1" * (80_000 + 1)
        yield b'"}'

    resp = client.post(
        "/api/check", content=oversized(),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413


def test_invalid_csv_does_not_consume_quota(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_RATE_PER_MINUTE", "1")
    monkeypatch.setenv("LIVE_RATE_PER_HOUR", "100")
    _stub_openai(monkeypatch, _live_ok_create)
    rows = "date,close\n" + "".join(
        f"2026-01-{(i % 28) + 1:02d},1\n" for i in range(501))
    bad = client.post("/api/check", json={"filename": "x.csv", "csv": rows})
    assert bad.status_code == 413
    # The 413 above must not have taken the single per-minute slot.
    ok = client.post("/api/check", json={"filename": "x.csv", "csv": CSV})
    assert ok.status_code == 200


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
