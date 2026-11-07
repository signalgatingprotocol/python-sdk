"""Tests for the fail-closed policy engine."""

import asyncio
import json

import pytest

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


class _NonBooleanApproval:
    def __bool__(self):
        raise AssertionError("approval results must not be coerced")

    def __repr__(self):
        raise AssertionError("approval results must not be disclosed")


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param("deny", id="string"),
        pytest.param(1, id="integer"),
        pytest.param({"approved": False, "detail": "private-result"}, id="mapping"),
        pytest.param(None, id="none"),
        pytest.param(_NonBooleanApproval(), id="custom-object"),
    ],
)
def test_non_boolean_approval_is_denied_and_audited(tmp_path, returned):
    audit = tmp_path / "audit.jsonl"
    requests = []

    def approve(request):
        requests.append(request)
        return returned

    engine = _engine(approver=approve, audit_path=audit)
    engine.register("reviewed_tool", ToolRisk(RiskLevel.SEND))

    assert engine.evaluate("reviewed_tool", {"value": "original"}, "worker") is Decision.DENY
    assert len(requests) == 1
    assert len(engine.records) == 1
    record = engine.records[0]
    assert record.tool_name == "reviewed_tool"
    assert record.agent_name == "worker"
    assert record.risk_level == "SEND"
    assert record.decision == "deny"
    assert type(returned).__name__ in record.reason
    assert "return True or False" in record.reason
    persisted = audit.read_text()
    assert "private-result" not in persisted
    assert [json.loads(line) for line in persisted.splitlines()] == [record.to_dict()]


@pytest.mark.parametrize("policy_required", [False, True], ids=["risk-ceiling", "yaml-rule"])
@pytest.mark.parametrize(
    "returned",
    [
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param("deny", id="string"),
        pytest.param(1, id="integer"),
        pytest.param({"approved": False}, id="mapping"),
    ],
)
async def test_tool_execution_requires_boolean_approval(policy_required, returned):
    from signal_gating.agent import ToolCallSignal
    from signal_gating.core import Agent, Mesh
    from signal_gating.policy.loader import engine_from_policy, load_policy_from_dict

    risk = RiskLevel.READ if policy_required else RiskLevel.SEND
    if policy_required:
        policy = load_policy_from_dict({"rules": [{"require_approval": True}]})
        engine = engine_from_policy(policy)
    else:
        engine = _engine()
    requests = []
    executed = []

    def approve(request):
        requests.append(request)
        return returned

    engine.approver = approve
    engine.register("reviewed_tool", ToolRisk(risk))
    worker = Agent("worker", gates=[engine.as_gate("worker")])

    @worker.tool(risk=risk)
    def reviewed_tool(value: str) -> None:
        executed.append(value)

    async with Mesh([worker]) as mesh:
        await mesh.inject(
            worker, ToolCallSignal(tool_name="reviewed_tool", arguments={"value": "original"})
        )

    assert len(requests) == 1
    assert requests[0].arguments == {"value": "original"}
    assert requests[0].agent_name == "worker"
    assert executed == (["original"] if returned is True else [])
    assert len(engine.records) == 1
    assert engine.records[0].decision == ("allow" if returned is True else "deny")


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


@pytest.mark.parametrize(
    ("recipient", "approved"),
    [("trusted@example.com", True), ("attacker@example.com", False)],
)
async def test_mesh_tool_approval_uses_call_arguments(recipient, approved):
    from signal_gating.core import Agent, Mesh

    requests = []
    executed = []

    def approve(request):
        requests.append(request)
        return request.arguments.get("to") == "trusted@example.com"

    engine = _engine(approver=approve)
    engine.register("send_email", ToolRisk(RiskLevel.SEND))
    worker = Agent("worker", gates=[engine.as_gate("worker")])

    @worker.tool(risk=RiskLevel.SEND)
    async def send_email(to: str, payload: dict) -> dict:
        executed.append(to)
        return payload

    payload = {"body": "status update", "tags": ["reviewed"]}
    async with Mesh([worker]) as mesh:
        if approved:
            result = await mesh.call_tool(
                worker, "send_email", to=recipient, payload=payload, timeout=1
            )
            assert result == payload
        else:
            with pytest.raises(asyncio.TimeoutError):
                await mesh.call_tool(
                    worker, "send_email", to=recipient, payload=payload, timeout=0.05
                )

    assert len(requests) == 1
    assert requests[0].arguments == {"to": recipient, "payload": payload}
    assert requests[0].agent_name == "worker"
    assert executed == ([recipient] if approved else [])
    assert engine.records[-1].decision == ("allow" if approved else "deny")


async def test_as_gate_approval_arguments_are_detached_mutable_containers():
    from signal_gating.agent import ToolCallSignal

    original = {"payload": {"recipients": ["trusted@example.com"]}}
    snapshots = []

    def approve(request):
        assert request.arguments == original
        snapshots.append(request.arguments)
        request.arguments["payload"]["recipients"].append("approval-note")
        request.arguments["payload"]["reviewed"] = True
        return True

    engine = _engine(approver=approve)
    engine.register("send_email", ToolRisk(RiskLevel.SEND))
    gate = engine.as_gate("worker")
    signal = ToolCallSignal(tool_name="send_email", arguments=original)

    assert await gate.process(signal) is signal
    assert await gate.process(signal) is signal
    assert signal.arguments == original
    assert snapshots[0] is not snapshots[1]
    assert snapshots[0]["payload"] is not snapshots[1]["payload"]
    assert snapshots[0]["payload"]["recipients"] == [
        "trusted@example.com", "approval-note"
    ]


async def test_as_gate_approval_mutation_preserves_mutable_argument_leaves():
    from signal_gating.agent import ToolCallSignal

    def approve(request):
        request.arguments["payload"]["data"][0] = ord("X")
        return True

    engine = _engine(approver=approve)
    engine.register("send_email", ToolRisk(RiskLevel.SEND))
    signal = ToolCallSignal(
        tool_name="send_email", arguments={"payload": {"data": bytearray(b"safe")}}
    )

    assert await engine.as_gate("worker").process(signal) is signal
    assert signal.arguments["payload"]["data"] == bytearray(b"safe")


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        (RiskLevel.READ, Decision.ALLOW),
        (None, Decision.DENY),
        (RiskLevel.SEND, Decision.REQUIRE_APPROVAL),
    ],
)
async def test_as_gate_does_not_copy_opaque_arguments_without_approval(risk, expected):
    from threading import Lock

    from signal_gating.agent import ToolCallSignal

    engine = _engine()
    if risk is not None:
        engine.register("local_tool", ToolRisk(risk))
    signal = ToolCallSignal(tool_name="local_tool", arguments={"lock": Lock()})

    result = await engine.as_gate("worker").process(signal)

    assert result is (signal if expected is Decision.ALLOW else None)
    assert len(engine.records) == 1
    assert engine.records[0].decision == expected.value
    assert engine.records[0].agent_name == "worker"


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
