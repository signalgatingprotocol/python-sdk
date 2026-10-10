"""Fail-closed authorization for agentic AI.

Every framework in 2026 ships permissive tool defaults. MCP delegates
authorization to transport-level OAuth scopes and defines no per-tool ACL.
This package is the layer they left out:

- ``risk``: per-tool risk classification (READ / DRAFT / SEND / DESTRUCTIVE).
  Unknown tools are treated as DESTRUCTIVE for gating, never allowed through.
- ``audit``: MCP annotation distrust. Servers self-declare readOnlyHint and
  destructiveHint; this module verifies the claim against observed behavior.
- ``enforcement``: the policy engine. Default-deny evaluation, dry-run mode,
  append-only audit trail.
- ``redteam``: documented attack scenarios (prompt injection via tool output,
  MCP privilege escalation, data exfiltration) with executable proofs that
  the enforcement layer blocks each one.
"""

from signal_gating.security.audit import (
    AuditFinding,
    AuditReport,
    DeclaredHints,
    ObservedBehavior,
    audit_tool,
    scan_source,
)
from signal_gating.security.enforcement import (
    ApprovalRequest,
    Decision,
    PolicyEngine,
    evaluate,
)
from signal_gating.security.redteam import (
    AttackScenario,
    ScenarioResult,
    all_scenarios,
    run_all,
    run_scenario,
)
from signal_gating.security.risk import (
    RiskLevel,
    ToolRisk,
    classify_tool,
    describe_policy,
    effective_level,
)

__all__ = [
    "ApprovalRequest",
    "AttackScenario",
    "AuditFinding",
    "AuditReport",
    "Decision",
    "DeclaredHints",
    "ObservedBehavior",
    "PolicyEngine",
    "RiskLevel",
    "ScenarioResult",
    "ToolRisk",
    "all_scenarios",
    "audit_tool",
    "classify_tool",
    "describe_policy",
    "effective_level",
    "evaluate",
    "run_all",
    "run_scenario",
    "scan_source",
]
