"""Tests for the chart-image check (app/tests/test_image_check.py).

Live Muse calls are stubbed exactly like test_api.py: no API key,
network, or paid call is involved. Simulated results come only from
the bundled sample charts generated from the public samples/ data.
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import muse_client
import pytest
from fastapi.testclient import TestClient
from main import app
from rate_limit import guard

client = TestClient(app)
SAMPLE_PNG = (
    Path(__file__).resolve().parent.parent
    / "static" / "assets" / "sample-charts" / "chart_task_001.png"
)


@pytest.fixture(autouse=True)
def _reset_guard():
    guard.reset()
    yield
    guard.reset()


def _payload(data: bytes, media_type: str = "image/png", name: str = "chart.png"):
    return {
        "filename": name,
        "media_type": media_type,
        "image_base64": base64.b64encode(data).decode(),
    }


def test_sample_charts_listed():
    resp = client.get("/api/sample-charts")
    assert resp.status_code == 200
    ids = {s["id"] for s in resp.json()["samples"]}
    assert ids == {"task_001", "task_002", "task_003"}


def test_simulated_sample_chart_without_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    resp = client.post("/api/check-image", json=_payload(SAMPLE_PNG.read_bytes()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "simulated" and body["simulated"] is True
    assert body["sample_id"] == "task_001"
    # Findings come from the public sample's answer key (all three kinds).
    assert {f["type"] for f in body["findings"]} == {"prices", "dups", "gaps"}
    assert body["series"] and body["series"][0]["value"] > 0
    assert "secret" not in resp.text


def test_simulated_unknown_image_is_not_faked(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    # A valid PNG that is not a bundled sample: no simulated findings.
    unknown = SAMPLE_PNG.read_bytes() + b"\x00"
    resp = client.post("/api/check-image", json=_payload(unknown))
    assert resp.status_code == 503
    assert resp.json()["mode"] == "simulated"
    assert "findings" not in resp.json()


def test_image_rejects_bad_payloads(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    assert client.post("/api/check-image", json={"image_base64": "x"}).status_code == 400
    bad_b64 = {"filename": "x.png", "media_type": "image/png", "image_base64": "!!!"}
    assert client.post("/api/check-image", json=bad_b64).status_code == 400
    not_png = _payload(b"not an image at all")
    assert client.post("/api/check-image", json=not_png).status_code == 413
    wrong_type = _payload(SAMPLE_PNG.read_bytes(), media_type="text/plain")
    assert client.post("/api/check-image", json=wrong_type).status_code == 413


def test_image_cross_origin_403(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    resp = client.post(
        "/api/check-image",
        json=_payload(SAMPLE_PNG.read_bytes()),
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 403


_IMAGE_JSON = (
    '{"summary": "Series with a spike.", "findings": [{"type": "prices", '
    '"title": "One-day spike", "explain": "Price spikes for one day."}], '
    '"series": [{"date": "2024-01-02", "value": 101.5}, '
    '{"date": "bad", "value": "oops"}]}'
)


def test_live_image_check_sends_image_part(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    captured: dict = {}

    def fake_create(self, **kwargs):  # noqa: ANN001, ANN202 - test double
        captured.update(kwargs)
        message = type("M", (), {"content": _IMAGE_JSON})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()
        return type("R", (), {"choices": [choice], "usage": usage})()

    import openai

    class FakeOpenAI:
        def __init__(self, **kwargs):  # noqa: ANN003
            captured["client_kwargs"] = kwargs
            self.chat = type(
                "Chat", (), {"completions": type("Comp", (), {"create": fake_create})()}
            )()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    resp = client.post(
        "/api/check-image",
        json={**_payload(SAMPLE_PNG.read_bytes()), "model": "bogus-model"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "live" and body["simulated"] is False
    assert body["model"] == "muse-spark-1.3"
    assert body["findings"][0]["type"] == "prices"
    # The unreadable series point is dropped, not invented.
    assert body["series"] == [{"date": "2024-01-02", "value": 101.5}]
    # Request shape: model pinned, reasoning low, image in a user message.
    assert captured["model"] == "muse-spark-1.3"
    assert captured["reasoning_effort"] == "low"
    assert captured["max_tokens"] == muse_client.DEFAULT_MAX_TOKENS
    user = captured["messages"][1]
    assert user["role"] == "user"
    parts = {p["type"]: p for p in user["content"]}
    assert parts["image_url"]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "test-key" not in resp.text


def test_image_reserve_includes_image_payload(monkeypatch):
    """The spend reserve must not be a text-only estimate (Bugbot).

    With synthetic one-unit-per-token pricing, the reserved worst case
    equals the estimated prompt tokens plus the output cap, so the
    image-inclusive parts are directly assertable.
    """
    monkeypatch.setenv("META_INPUT_COST_PER_MTOK", "1000000")
    monkeypatch.setenv("META_OUTPUT_COST_PER_MTOK", "1000000")

    text_only = muse_client._cost_usd(
        max(1, len(muse_client._IMAGE_SYSTEM_PROMPT) // 4),
        muse_client._max_tokens(),
    )
    size = len(SAMPLE_PNG.read_bytes())
    reserved = muse_client.worst_case_image_cost_usd(size)
    assert reserved is not None and text_only is not None
    assert reserved > text_only
    # At least the fixed per-image allowance is charged on top of text.
    assert reserved - text_only >= muse_client.FIXED_IMAGE_TOKEN_ALLOWANCE
    # The estimate grows with image size and is capped at the upload limit.
    assert muse_client._estimated_image_prompt_tokens(0) < (
        muse_client._estimated_image_prompt_tokens(size)
    )
    assert muse_client._estimated_image_prompt_tokens(
        muse_client.MAX_IMAGE_BYTES * 10
    ) == muse_client._estimated_image_prompt_tokens(muse_client.MAX_IMAGE_BYTES)
    # No size supplied: reserve assumes the largest accepted upload.
    assert muse_client.worst_case_image_cost_usd() == (
        muse_client.worst_case_image_cost_usd(muse_client.MAX_IMAGE_BYTES)
    )
    assert muse_client.worst_case_image_cost_usd() >= reserved


def test_image_usage_fallback_includes_image_payload(monkeypatch):
    """When the API omits usage, settled cost uses the image estimate."""
    monkeypatch.setenv("META_INPUT_COST_PER_MTOK", "1000000")
    monkeypatch.setenv("META_OUTPUT_COST_PER_MTOK", "1000000")
    monkeypatch.setenv("META_API_KEY", "test-key")

    def fake_create(self, **kwargs):  # noqa: ANN001, ANN202 - no usage attr
        message = type("M", (), {"content": _IMAGE_JSON})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice]})()

    import openai

    class FakeOpenAI:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.chat = type(
                "Chat", (), {"completions": type("Comp", (), {"create": fake_create})()}
            )()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    image_bytes = SAMPLE_PNG.read_bytes()
    result = muse_client.check_chart_image("chart.png", image_bytes, "image/png")
    expected_prompt = muse_client._estimated_image_prompt_tokens(len(image_bytes))
    expected_completion = max(1, len(_IMAGE_JSON) // 4)
    assert result["estimated_cost_usd"] == expected_prompt + expected_completion
    assert result["estimated_cost_usd"] > muse_client.FIXED_IMAGE_TOKEN_ALLOWANCE


def test_validate_image_magic_bytes():
    assert muse_client.validate_image(SAMPLE_PNG.read_bytes(), "image/png") == "image/png"
    with pytest.raises(muse_client.MuseBadRequest):
        muse_client.validate_image(b"\xff\xd8\xff", "image/png")
