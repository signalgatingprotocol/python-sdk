"""Tests for the Jev-backed risk classifier. No network: all HTTP mocked."""

import io
import json
import urllib.error

import pytest

from signal_gating.security.jev import JevRiskClassifier, classify_with_jev
from signal_gating.security.risk import RiskLevel


def _response(payload: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode())


def _jev_answer(choice: str, confidence: float) -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "risk": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": {choice: confidence},
            }
        },
    }


@pytest.fixture
def mock_urlopen(monkeypatch):
    calls = {}

    class FakeResponse:
        def __init__(self, payload: bytes):
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        calls["timeout"] = timeout
        calls["url"] = request.full_url
        calls["auth"] = request.headers.get("Authorization")
        behavior = calls.get("behavior", "ok")
        if behavior == "ok":
            return FakeResponse(_response(_jev_answer("send", 0.92)).read())
        if behavior == "low_confidence":
            return FakeResponse(_response(_jev_answer("read", 0.4)).read())
        if behavior == "bad_choice":
            return FakeResponse(_response(_jev_answer("nuke", 0.99)).read())
        if behavior == "garbage":
            return FakeResponse(b"not json")
        if behavior == "http_error":
            raise urllib.error.HTTPError(
                calls["url"], 500, "server error", {}, io.BytesIO()
            )
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def test_classify_send(mock_urlopen):
    classifier = JevRiskClassifier(api_key="test-key")
    risk = classifier.classify("send_email", "Send an email via SES")
    assert risk.level is RiskLevel.SEND
    assert risk.confidence == pytest.approx(0.92)
    assert mock_urlopen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert mock_urlopen["auth"] == "Bearer test-key"


def test_no_api_key_is_unknown():
    classifier = JevRiskClassifier(api_key=None)
    # Ensure env var does not leak in
    import os

    monkeypatch_key = os.environ.pop("TYPESAFE_API_KEY", None)
    try:
        risk = classifier.classify("anything")
        assert risk.level is RiskLevel.UNKNOWN
    finally:
        if monkeypatch_key is not None:
            os.environ["TYPESAFE_API_KEY"] = monkeypatch_key


def test_low_confidence_is_unknown(mock_urlopen):
    mock_urlopen["behavior"] = "low_confidence"
    risk = JevRiskClassifier(api_key="k").classify("read_file", "read a file")
    assert risk.level is RiskLevel.UNKNOWN


def test_unknown_choice_is_unknown(mock_urlopen):
    mock_urlopen["behavior"] = "bad_choice"
    risk = JevRiskClassifier(api_key="k").classify("x")
    assert risk.level is RiskLevel.UNKNOWN


def test_garbage_response_is_unknown(mock_urlopen):
    mock_urlopen["behavior"] = "garbage"
    risk = JevRiskClassifier(api_key="k").classify("x")
    assert risk.level is RiskLevel.UNKNOWN


def test_http_error_is_unknown(mock_urlopen):
    mock_urlopen["behavior"] = "http_error"
    risk = JevRiskClassifier(api_key="k").classify("x")
    assert risk.level is RiskLevel.UNKNOWN


def test_timeout_is_unknown(monkeypatch):
    def boom(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    risk = JevRiskClassifier(api_key="k", timeout=1.0).classify("x")
    assert risk.level is RiskLevel.UNKNOWN


def test_fallback_to_heuristic(mock_urlopen):
    mock_urlopen["behavior"] = "http_error"
    classifier = JevRiskClassifier(api_key="k", fallback_to_heuristic=True)
    risk = classifier.classify("delete_database", "drop all tables")
    assert risk.level is RiskLevel.DESTRUCTIVE  # heuristic still works


def test_convenience_wrapper(mock_urlopen):
    risk = classify_with_jev("send_email", "send mail", api_key="k")
    assert risk.level is RiskLevel.SEND


def test_all_four_levels_map(mock_urlopen, monkeypatch):
    import urllib.request

    for choice, expected in [
        ("read", RiskLevel.READ),
        ("draft", RiskLevel.DRAFT),
        ("send", RiskLevel.SEND),
        ("destructive", RiskLevel.DESTRUCTIVE),
    ]:
        payload = _jev_answer(choice, 0.95)

        class FakeResponse:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda req, timeout=None: FakeResponse()
        )
        risk = JevRiskClassifier(api_key="k").classify("tool")
        assert risk.level is expected, choice
