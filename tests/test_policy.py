"""Tests for policy-as-code loading and evaluation."""

import json

import pytest

from signal_gating.policy.loader import (
    PolicyRule,
    engine_from_policy,
    load_policy,
    load_policy_from_dict,
)
from signal_gating.security.enforcement import Decision
from signal_gating.security.risk import RiskLevel, ToolRisk


def _policy_dict():
    return {
        "version": 1,
        "name": "test",
        "default": "deny",
        "max_auto_approve": "READ",
        "rules": [
            {"match": {"tools": ["read_*"]}, "allow": True},
            {"match": {"risk": ["SEND", "DESTRUCTIVE"]}, "require_approval": True},
        ],
    }


def test_load_policy_from_dict():
    p = load_policy_from_dict(_policy_dict())
    assert p.name == "test"
    assert p.version == 1
    assert p.default == "deny"
    assert p.max_auto_approve is RiskLevel.READ
    assert len(p.rules) == 2


def test_first_match_wins():
    p = load_policy_from_dict(_policy_dict())
    assert p.decide("read_file", risk=RiskLevel.READ) is Decision.ALLOW
    assert p.decide("send_email", risk=RiskLevel.SEND) is Decision.REQUIRE_APPROVAL
    assert p.decide("mystery", risk=RiskLevel.UNKNOWN) is Decision.DENY


def test_default_deny():
    p = load_policy_from_dict(_policy_dict())
    assert p.decide("anything_else") is Decision.DENY


def test_default_allow_warns():
    d = _policy_dict()
    d["default"] = "allow"
    with pytest.warns(UserWarning, match="fail-open"):
        load_policy_from_dict(d)


def test_load_policy_from_yaml(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "version: 1\nname: yaml-test\ndefault: deny\n"
        "max_auto_approve: DRAFT\nrules: []\n"
    )
    p = load_policy(path)
    assert p.name == "yaml-test"
    assert p.max_auto_approve is RiskLevel.DRAFT


def test_engine_from_policy():
    p = load_policy_from_dict(_policy_dict())
    engine = engine_from_policy(
        p, tool_risks={"read_file": ToolRisk(RiskLevel.READ, (), 1.0)}
    )
    assert engine.evaluate("read_file") is Decision.ALLOW
    # Policy has no allow rule for this tool: default deny short-circuits
    assert engine.evaluate("write_file") is Decision.DENY


def test_engine_from_policy_dry_run():
    p = load_policy_from_dict(_policy_dict())
    engine = engine_from_policy(
        p,
        tool_risks={"read_file": ToolRisk(RiskLevel.READ, (), 1.0)},
        dry_run=True,
    )
    d = engine.evaluate("read_file")
    assert d in (Decision.ALLOW, Decision.WOULD_ALLOW)


@pytest.fixture
def approval_policy(tmp_path):
    path = tmp_path / "approval.yaml"
    path.write_text(
        "version: 2\nname: review-required\ndefault: deny\n"
        "max_auto_approve: READ\nrules:\n"
        "  - match:\n      tools: ['*']\n    require_approval: true\n",
        encoding="utf-8",
    )
    return load_policy(path)


@pytest.mark.parametrize("risk_level", [RiskLevel.READ, RiskLevel.SEND])
@pytest.mark.parametrize(
    ("approved", "expected"), [(True, Decision.ALLOW), (False, Decision.DENY)]
)
def test_policy_required_approval_invokes_callback(
    approval_policy, tmp_path, risk_level, approved, expected
):
    audit = tmp_path / "audit.jsonl"
    risk = ToolRisk(risk_level)
    engine = engine_from_policy(
        approval_policy, tool_risks={"reviewed_tool": risk}, audit_path=audit
    )
    requests = []
    arguments = {"payload": {"items": ["original"]}}

    def approve(request):
        assert request.arguments == arguments
        requests.append(request)
        request.arguments["payload"]["items"].append("approval-note")
        return approved

    engine.approver = approve
    assert engine.evaluate("reviewed_tool", arguments, "worker") is expected
    assert len(requests) == 1
    assert requests[0].tool_name == "reviewed_tool"
    assert requests[0].agent_name == "worker"
    assert requests[0].risk is risk
    assert "policy 'review-required' v2 requires approval" in requests[0].reason
    assert arguments == {"payload": {"items": ["original"]}}
    assert len(engine.records) == 1
    record = engine.records[0]
    assert record.decision == expected.value
    assert record.risk_level == risk_level.name
    assert record.agent_name == "worker"
    assert [json.loads(line) for line in audit.read_text().splitlines()] == [record.to_dict()]


@pytest.mark.parametrize("dry_run", [False, True])
def test_policy_required_approval_stays_pending_without_live_approver(approval_policy, dry_run):
    engine = engine_from_policy(
        approval_policy, tool_risks={"reviewed_tool": ToolRisk(RiskLevel.READ)},
        dry_run=dry_run,
    )
    requests = []
    if dry_run:
        engine.approver = lambda request: requests.append(request) or True

    assert engine.evaluate("reviewed_tool") is Decision.REQUIRE_APPROVAL
    assert requests == []
    assert len(engine.records) == 1
    assert engine.records[0].decision == Decision.REQUIRE_APPROVAL.value


def test_policy_required_approval_never_authorizes_unregistered_tool(approval_policy):
    engine = engine_from_policy(approval_policy)
    requests = []
    engine.approver = lambda request: requests.append(request) or True

    assert engine.evaluate("unregistered_tool") is Decision.DENY
    assert requests == []
    assert len(engine.records) == 1
    assert engine.records[0].risk_level == "UNKNOWN"
    assert engine.records[0].decision == Decision.DENY.value


def test_policy_deny_precedes_approval(approval_policy):
    approval_policy.rules.insert(0, PolicyRule(action="deny"))
    engine = engine_from_policy(
        approval_policy, tool_risks={"reviewed_tool": ToolRisk(RiskLevel.READ)}
    )
    requests = []
    engine.approver = lambda request: requests.append(request) or True

    assert engine.evaluate("reviewed_tool") is Decision.DENY
    assert requests == []
    assert len(engine.records) == 1
    assert engine.records[0].decision == Decision.DENY.value


@pytest.mark.parametrize("risk_level", [RiskLevel.READ, RiskLevel.SEND])
@pytest.mark.parametrize("approved", [False, True])
async def test_policy_required_approval_gates_tool_execution(approval_policy, risk_level, approved):
    from signal_gating.agent import ToolCallSignal
    from signal_gating.core import Agent, Mesh

    engine = engine_from_policy(
        approval_policy, tool_risks={"reviewed_tool": ToolRisk(risk_level)}
    )
    requests = []
    executed = []

    def approve(request):
        requests.append(request)
        return request.arguments["value"] == "approved"

    engine.approver = approve
    worker = Agent("worker", gates=[engine.as_gate("worker")])

    @worker.tool(risk=risk_level)
    async def reviewed_tool(value: str) -> None:
        executed.append(value)

    value = "approved" if approved else "rejected"
    async with Mesh([worker]) as mesh:
        await mesh.inject(
            worker, ToolCallSignal(tool_name="reviewed_tool", arguments={"value": value})
        )

    assert len(requests) == 1
    assert requests[0].agent_name == "worker"
    assert requests[0].arguments == {"value": value}
    assert executed == ([value] if approved else [])
    assert len(engine.records) == 1
    assert engine.records[0].decision == (Decision.ALLOW if approved else Decision.DENY).value
