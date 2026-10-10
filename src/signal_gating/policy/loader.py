"""YAML policy loading and compilation to a PolicyEngine."""

from __future__ import annotations

import fnmatch
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from signal_gating.security.enforcement import Decision, PolicyEngine
from signal_gating.security.risk import RiskLevel, ToolRisk


@dataclass(slots=True, frozen=True)
class PolicyRule:
    """One ordered rule: the first matching rule wins."""

    tool_patterns: tuple[str, ...] = ()
    agent_patterns: tuple[str, ...] = ()
    risk_levels: tuple[RiskLevel, ...] = ()
    action: str = "deny"  # allow | deny | require_approval
    approver_name: str = ""


@dataclass(slots=True)
class Policy:
    """A compiled, versioned policy."""

    name: str
    version: int
    default: str  # allow | deny
    max_auto_approve: RiskLevel
    rules: list[PolicyRule] = field(default_factory=list)

    def matches(self, rule: PolicyRule, tool_name: str, agent_name: str,
                risk: RiskLevel | None) -> bool:
        if rule.tool_patterns and not any(
            fnmatch.fnmatch(tool_name, p) for p in rule.tool_patterns
        ):
            return False
        if rule.agent_patterns and not any(
            fnmatch.fnmatch(agent_name, p) for p in rule.agent_patterns
        ):
            return False
        if rule.risk_levels and (risk is None or risk not in rule.risk_levels):
            return False
        return True

    def decide(
        self, tool_name: str, agent_name: str = "",
        risk: RiskLevel | None = None,
    ) -> Decision:
        """First matching rule wins; otherwise the default applies."""
        for rule in self.rules:
            if self.matches(rule, tool_name, agent_name, risk):
                if rule.action == "allow":
                    return Decision.ALLOW
                if rule.action == "require_approval":
                    return Decision.REQUIRE_APPROVAL
                return Decision.DENY
        return Decision.ALLOW if self.default == "allow" else Decision.DENY


def _parse_risk_levels(raw: Any) -> tuple[RiskLevel, ...]:
    if not raw:
        return ()
    names = [raw] if isinstance(raw, str) else list(raw)
    return tuple(RiskLevel[n.upper()] for n in names)


def load_policy_from_dict(data: dict[str, Any]) -> Policy:
    version = int(data.get("version", 1))
    name = str(data.get("name", "unnamed"))
    default = str(data.get("default", "deny")).lower()
    if default not in ("allow", "deny"):
        raise ValueError(f"policy default must be allow or deny, got {default!r}")
    if default == "allow":
        warnings.warn(
            "Policy default is 'allow': every unmatched tool call is permitted. "
            "This is fail-open. Prefer default: deny.",
            UserWarning,
            stacklevel=3,
        )
    max_auto = RiskLevel[str(data.get("max_auto_approve", "READ")).upper()]

    rules: list[PolicyRule] = []
    for raw_rule in data.get("rules", []) or []:
        match = raw_rule.get("match", {}) or {}
        tools = match.get("tools", []) or []
        agents = match.get("agents", []) or []
        action = "deny"
        approver_name = ""
        if raw_rule.get("allow") is True:
            action = "allow"
        elif raw_rule.get("require_approval") is True:
            action = "require_approval"
            approver_name = str(raw_rule.get("approver", ""))
        rules.append(
            PolicyRule(
                tool_patterns=tuple(tools if isinstance(tools, list) else [tools]),
                agent_patterns=tuple(agents if isinstance(agents, list) else [agents]),
                risk_levels=_parse_risk_levels(match.get("risk")),
                action=action,
                approver_name=approver_name,
            )
        )
    return Policy(
        name=name, version=version, default=default,
        max_auto_approve=max_auto, rules=rules,
    )


def load_policy(path: str | Path) -> Policy:
    """Load a policy from a YAML file."""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Policy file {path} must contain a YAML mapping")
    return load_policy_from_dict(data)


def engine_from_policy(
    policy: Policy,
    tool_risks: dict[str, ToolRisk] | None = None,
    audit_path: str | Path | None = None,
    dry_run: bool = False,
) -> PolicyEngine:
    """Build a PolicyEngine wired to a declarative policy.

    The policy's rule order is preserved by wrapping decide() as the
    approver-adjacent check: rule matches for allow/deny short-circuit,
    require_approval delegates to the engine's approver (or denies).
    """
    engine = PolicyEngine(
        max_auto_approve=policy.max_auto_approve,
        audit_path=audit_path,
        dry_run=dry_run,
    )
    for tool_name, risk in (tool_risks or {}).items():
        engine.register(tool_name, risk)

    base_evaluate = engine.evaluate

    def evaluate_with_policy(
        tool_name: str,
        arguments: dict[str, object] | None = None,
        agent_name: str = "",
    ) -> Decision:
        risk = engine._registry.get(tool_name)
        level = risk.level if risk else None
        verdict = policy.decide(tool_name, agent_name, level)
        if verdict is Decision.ALLOW:
            return base_evaluate(tool_name, arguments, agent_name)
        # Policy deny / require_approval short-circuits before approver.
        if verdict is Decision.REQUIRE_APPROVAL and engine.approver is None:
            return engine._record(
                tool_name, agent_name, level or RiskLevel.UNKNOWN,
                Decision.REQUIRE_APPROVAL,
                f"policy {policy.name!r} v{policy.version} requires approval",
            )
        return engine._record(
            tool_name, agent_name, level or RiskLevel.UNKNOWN,
            Decision.DENY if verdict is Decision.DENY else Decision.REQUIRE_APPROVAL,
            f"policy {policy.name!r} v{policy.version} rule matched",
        )

    engine.evaluate = evaluate_with_policy  # type: ignore[method-assign]
    return engine
