"""Tests for per-tool risk classification."""

from signal_gating.security.risk import (
    RiskLevel,
    ToolRisk,
    classify_tool,
    describe_policy,
    effective_level,
)


def test_effective_level_unknown_is_destructive():
    assert effective_level(RiskLevel.UNKNOWN) is RiskLevel.DESTRUCTIVE
    assert effective_level(RiskLevel.READ) is RiskLevel.READ


def test_classify_destructive_verbs():
    r = classify_tool("delete_user", "Permanently delete a user account")
    assert r.level is RiskLevel.DESTRUCTIVE


def test_classify_code_execution():
    r = classify_tool("run_shell", "Execute a shell command", source="subprocess.run(cmd)")
    assert r.level is RiskLevel.DESTRUCTIVE


def test_classify_send():
    r = classify_tool("send_email", "Send an email via SMTP")
    assert r.level is RiskLevel.SEND


def test_classify_draft():
    r = classify_tool("write_draft", "Write a draft document to the staging area")
    assert r.level is RiskLevel.DRAFT


def test_classify_read():
    r = classify_tool("read_file", "Read a file from disk")
    assert r.level is RiskLevel.READ


def test_classify_unknown_when_no_signal():
    r = classify_tool("blorpt", "Does something unspecified")
    assert r.level is RiskLevel.UNKNOWN
    assert r.confidence == 0.0


def test_gates_at():
    r = ToolRisk(RiskLevel.SEND, ("outbound",), 0.9)
    assert r.gates_at(RiskLevel.READ) is True
    assert r.gates_at(RiskLevel.SEND) is False
    # UNKNOWN always gates below DESTRUCTIVE ceiling
    u = ToolRisk(RiskLevel.UNKNOWN, (), 0.0)
    assert u.gates_at(RiskLevel.SEND) is True


def test_describe_policy():
    text = describe_policy(RiskLevel.READ)
    assert "READ" in text
    assert "UNKNOWN" in text
