"""Tests for policy-as-code loading and evaluation."""

import pytest

from signal_gating.policy.loader import (
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
