"""Tests for the fail-closed policy engine."""

import json

from signal_gating.security.enforcement import Decision, PolicyEngine
from signal_gating.security.risk import RiskLevel, ToolRisk


def _engine(**kwargs):
    return PolicyEngine(max_auto_approve=RiskLevel.READ, **kwargs)


def test_unknown_tool_denied():
    e = _engine()
    assert e.evaluate("never_registered") is Decision.DENY


def test_read_auto_approved():
    e = _engine()
    e.register("read_file", ToolRisk(RiskLevel.READ, ("read",), 0.9))
    assert e.evaluate("read_file") is Decision.ALLOW


def test_send_requires_approval_without_approver():
    e = _engine()
    e.register("send_email", ToolRisk(RiskLevel.SEND, ("outbound",), 0.9))
    assert e.evaluate("send_email") is Decision.REQUIRE_APPROVAL


def test_send_denied_when_approver_rejects():
    e = _engine(approver=lambda req: False)
    e.register("send_email", ToolRisk(RiskLevel.SEND, ("outbound",), 0.9))
    assert e.evaluate("send_email") is Decision.DENY


def test_send_allowed_when_approver_accepts():
    e = _engine(approver=lambda req: True)
    e.register("send_email", ToolRisk(RiskLevel.SEND, ("outbound",), 0.9))
    assert e.evaluate("send_email") is Decision.ALLOW


def test_dry_run_records_would_decisions(tmp_path):
    audit = tmp_path / "audit.jsonl"
    e = _engine(audit_path=audit, dry_run=True)
    e.register("read_file", ToolRisk(RiskLevel.READ, (), 1.0))
    assert e.evaluate("read_file") is Decision.WOULD_ALLOW
    assert e.evaluate("mystery_tool") is Decision.WOULD_DENY
    lines = audit.read_text().strip().split("\n")
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["decision"] == "would_allow"
    assert "timestamp" in rec


def test_audit_trail_written(tmp_path):
    audit = tmp_path / "audit.jsonl"
    e = _engine(audit_path=audit)
    e.evaluate("nope")
    rec = json.loads(audit.read_text().strip())
    assert rec["tool_name"] == "nope"
    assert rec["decision"] == "deny"
    assert rec["risk_level"] == "UNKNOWN"


def test_denial_rate():
    e = _engine()
    e.register("read_file", ToolRisk(RiskLevel.READ, (), 1.0))
    e.evaluate("read_file")
    e.evaluate("unknown_1")
    e.evaluate("unknown_2")
    assert e.denial_rate() == 2 / 3


def test_as_gate_blocks_tool_call_signal():
    import asyncio

    from signal_gating.agent import ToolCallSignal

    e = _engine()
    e.register("read_file", ToolRisk(RiskLevel.READ, (), 1.0))
    gate = e.as_gate("worker")

    allowed = asyncio.run(gate.process(ToolCallSignal(tool_name="read_file")))
    assert allowed is not None
    blocked = asyncio.run(gate.process(ToolCallSignal(tool_name="delete_db")))
    assert blocked is None


def test_secure_agent_example_runs():
    import subprocess
    import sys
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "secure_agent.py"
    result = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert "read_config -> allow" in result.stdout
    assert "rm_rf_everything -> deny" in result.stdout


def test_tool_decorator_accepts_risk():
    from signal_gating.agent import Agent

    agent = Agent("risky")

    @agent.tool(risk=RiskLevel.SEND)
    def send_alert(message: str) -> str:
        return f"sent: {message}"

    spec = agent.get_tool("send_alert")
    assert spec is not None
    assert spec.risk is RiskLevel.SEND


def test_tool_decorator_defaults_unknown():
    from signal_gating.agent import Agent

    agent = Agent("mystery")

    @agent.tool()
    def do_something(x: int) -> int:
        return x

    spec = agent.get_tool("do_something")
    assert spec is not None
    assert spec.risk is RiskLevel.UNKNOWN
