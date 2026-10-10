"""The policy engine: default-deny evaluation with audit trails.

Evaluation order for every tool call:

1. If the tool is unknown to the registry, deny. Unknown tools never execute.
2. Map the tool's risk level through ``effective_level`` (UNKNOWN becomes
   DESTRUCTIVE).
3. If the effective level is at or below ``max_auto_approve``, allow.
4. Otherwise, require approval: invoke the approver callback. If no approver
   is configured, deny (fail closed, never fail open).

Every decision is appended to an audit trail (JSONL) with timestamp, tool,
risk, decision, and reason. Dry-run mode evaluates without enforcing:
decisions are recorded as ``would_allow`` / ``would_deny`` so policies can
be validated against production traffic before they bite.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from signal_gating.security.risk import RiskLevel, ToolRisk, effective_level


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    WOULD_ALLOW = "would_allow"
    WOULD_DENY = "would_deny"


@dataclass(slots=True, frozen=True)
class ApprovalRequest:
    """What an approver sees when a tool call needs a human (or policy)."""

    tool_name: str
    arguments: dict[str, object]
    risk: ToolRisk
    agent_name: str
    reason: str


Approver = Callable[[ApprovalRequest], bool]


@dataclass(slots=True)
class AuditRecord:
    timestamp: float
    tool_name: str
    agent_name: str
    risk_level: str
    decision: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "agent_name": self.agent_name,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "reason": self.reason,
        }


class PolicyEngine:
    """Fail-closed tool-call authorization."""

    def __init__(
        self,
        max_auto_approve: RiskLevel = RiskLevel.READ,
        approver: Approver | None = None,
        audit_path: str | Path | None = None,
        dry_run: bool = False,
    ) -> None:
        self.max_auto_approve = max_auto_approve
        self.approver = approver
        self.dry_run = dry_run
        self._audit_path = Path(audit_path) if audit_path else None
        self._registry: dict[str, ToolRisk] = {}
        self._records: list[AuditRecord] = []

    def register(self, tool_name: str, risk: ToolRisk) -> None:
        """Register a tool's risk classification with the engine."""
        self._registry[tool_name] = risk

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, object] | None = None,
        agent_name: str = "",
    ) -> Decision:
        """Evaluate one tool call. Never raises; denial is a value."""
        risk = self._registry.get(tool_name)
        if risk is None:
            return self._record(
                tool_name, agent_name, RiskLevel.UNKNOWN,
                Decision.DENY, "unknown tool: default deny",
            )

        level = effective_level(risk.level)
        if level <= effective_level(self.max_auto_approve):
            return self._record(
                tool_name, agent_name, risk.level,
                Decision.ALLOW,
                f"risk {risk.level.name} within auto-approve ceiling "
                f"{self.max_auto_approve.name}",
            )

        if self.approver is not None and not self.dry_run:
            approved = self.approver(
                ApprovalRequest(
                    tool_name=tool_name,
                    arguments=arguments or {},
                    risk=risk,
                    agent_name=agent_name,
                    reason=(
                        f"risk {risk.level.name} exceeds auto-approve ceiling "
                        f"{self.max_auto_approve.name}"
                    ),
                )
            )
            if approved:
                return self._record(
                    tool_name, agent_name, risk.level,
                    Decision.ALLOW, "approved by approver callback",
                )
            return self._record(
                tool_name, agent_name, risk.level,
                Decision.DENY, "rejected by approver callback",
            )

        # No approver, or dry run: fail closed.
        return self._record(
            tool_name, agent_name, risk.level,
            Decision.REQUIRE_APPROVAL,
            f"risk {risk.level.name} exceeds auto-approve ceiling "
            f"{self.max_auto_approve.name}; no approver configured",
        )

    def _record(
        self,
        tool_name: str,
        agent_name: str,
        level: RiskLevel,
        decision: Decision,
        reason: str,
    ) -> Decision:
        if self.dry_run and decision in (Decision.ALLOW, Decision.DENY):
            decision = (
                Decision.WOULD_ALLOW
                if decision is Decision.ALLOW
                else Decision.WOULD_DENY
            )
        record = AuditRecord(
            timestamp=time.time(),
            tool_name=tool_name,
            agent_name=agent_name,
            risk_level=level.name,
            decision=decision.value,
            reason=reason,
        )
        self._records.append(record)
        if self._audit_path is not None:
            with self._audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        return decision

    @property
    def records(self) -> list[AuditRecord]:
        return list(self._records)

    def denial_rate(self) -> float:
        """Fraction of evaluated calls that were denied or gated."""
        if not self._records:
            return 0.0
        gated = {
            Decision.DENY.value,
            Decision.REQUIRE_APPROVAL.value,
            Decision.WOULD_DENY.value,
        }
        n = sum(1 for r in self._records if r.decision in gated)
        return n / len(self._records)

    def as_gate(self, agent_name: str = ""):
        """Return an SGP ``Gate`` enforcing this engine on tool calls.

        Attach it to an agent's gate list and every inbound
        ``ToolCallSignal`` is evaluated against the policy. Non-tool
        signals pass through untouched. Composes with every other gate::

            engine = PolicyEngine(max_auto_approve=RiskLevel.READ)
            worker = Agent("worker", gates=[engine.as_gate("worker")])
        """
        from signal_gating.agent import ToolCallSignal
        from signal_gating.gate import Gate

        engine = self

        def _check(signal) -> bool:
            if isinstance(signal, ToolCallSignal):
                decision = engine.evaluate(signal.tool_name, agent_name=agent_name)
                return decision in (Decision.ALLOW, Decision.WOULD_ALLOW)
            return True

        return Gate.filter(_check, name="policy_enforcement")


def evaluate(
    engine: PolicyEngine,
    tool_name: str,
    arguments: dict[str, object] | None = None,
    agent_name: str = "",
) -> Decision:
    """Convenience wrapper around ``PolicyEngine.evaluate``."""
    return engine.evaluate(tool_name, arguments=arguments, agent_name=agent_name)
