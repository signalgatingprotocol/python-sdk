"""Per-tool risk classification.

The four-class taxonomy the community converged on in 2026:

- READ: no side effects. Reading files, querying state, pure computation.
- DRAFT: reversible side effects. Writing drafts, staging changes, temp files.
- SEND: external effects. Network calls, messages sent, issues filed.
- DESTRUCTIVE: irreversible or high-blast-radius. Deletes, money movement,
  privilege changes, mass writes, code execution.

UNKNOWN is the fifth level and the most important one: any tool whose risk
cannot be determined is treated as DESTRUCTIVE for gating purposes.
Unclassified tools never slip through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum


class RiskLevel(IntEnum):
    """Ordered risk levels. Higher value means higher risk."""

    READ = 1
    DRAFT = 2
    SEND = 3
    DESTRUCTIVE = 4
    UNKNOWN = 5

    def __str__(self) -> str:
        return self.name


def effective_level(level: RiskLevel) -> RiskLevel:
    """The level used for gating decisions.

    UNKNOWN collapses to DESTRUCTIVE: when in doubt, require the strongest
    protection. This is the fail-closed rule.
    """
    if level is RiskLevel.UNKNOWN:
        return RiskLevel.DESTRUCTIVE
    return level


@dataclass(slots=True, frozen=True)
class ToolRisk:
    """A risk classification for one tool."""

    level: RiskLevel
    reasons: tuple[str, ...] = ()
    confidence: float = 1.0

    def gates_at(self, max_auto: RiskLevel) -> bool:
        """True if this tool requires approval under a max-auto policy."""
        return effective_level(self.level) > effective_level(max_auto)


# Heuristic signals, ordered by strength. Each entry is
# (compiled pattern, risk level, reason).
_READ_SIGNALS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), reason)
    for p, reason in [
        (r"\b(get|fetch|read|load|list|search|query|describe|show|view|lookup|check|validate|parse|inspect)\b", "read-like verb"),
        (r"\b(readonly|read.only|immutable|pure)\b", "declared read-only"),
    ]
)

_DESTRUCTIVE_SIGNALS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), reason)
    for p, reason in [
        (r"\b(delete|remove|destroy|drop|purge|wipe|truncate|kill|terminate)\b", "destructive verb"),
        (r"\b(exec|execute|eval|system|popen|subprocess|shell|spawn)\b", "code execution"),
        (r"\b(chmod|chown|sudo|su\b|setuid|privilege|escalat)\b", "privilege operation"),
        (r"\b(payment|charge|transfer|withdraw|refund|invoice|billing)\b", "money movement"),
        (r"\b(rm\s+-rf|format|mkfs|dd\s+if=)\b", "destructive shell idiom"),
    ]
)

_SEND_SIGNALS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), reason)
    for p, reason in [
        (r"\b(send|post|publish|notify|email|sms|tweet|webhook|broadcast)\b", "outbound communication verb"),
        (r"\b(requests?\.(post|put|delete)|httpx|urllib|fetch\(|axios)\b", "network call"),
        (r"\b(slack|discord|telegram|twilio|ses|sns)\b", "messaging integration"),
    ]
)

_DRAFT_SIGNALS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), reason)
    for p, reason in [
        (r"\b(write|save|store|create|update|draft|stage|append|insert)\b", "write-like verb"),
        (r"\b(temp|tmp|cache|staging|preview)\b", "reversible target"),
    ]
)


def classify_tool(
    name: str,
    description: str = "",
    source: str = "",
) -> ToolRisk:
    """Heuristically classify a tool's risk level.

    This is a first-pass classifier, not a proof. Anything it cannot
    confidently place lands at UNKNOWN, which gates as DESTRUCTIVE.
    Explicit ``risk=`` declarations on registration always win over
    this heuristic.
    """
    haystack = f"{name} {description} {source}"
    reasons: list[str] = []

    for pattern, reason in _DESTRUCTIVE_SIGNALS:
        if pattern.search(haystack):
            reasons.append(reason)
            return ToolRisk(RiskLevel.DESTRUCTIVE, tuple(reasons), 0.85)

    for pattern, reason in _SEND_SIGNALS:
        if pattern.search(haystack):
            reasons.append(reason)
            return ToolRisk(RiskLevel.SEND, tuple(reasons), 0.8)

    for pattern, reason in _DRAFT_SIGNALS:
        if pattern.search(haystack):
            reasons.append(reason)
            return ToolRisk(RiskLevel.DRAFT, tuple(reasons), 0.75)

    for pattern, reason in _READ_SIGNALS:
        if pattern.search(haystack):
            reasons.append(reason)
            return ToolRisk(RiskLevel.READ, tuple(reasons), 0.7)

    return ToolRisk(
        RiskLevel.UNKNOWN,
        ("no heuristic signal matched",),
        0.0,
    )


def describe_policy(max_auto: RiskLevel) -> str:
    """Human-readable summary of a max-auto-approve policy."""
    levels = [lvl for lvl in RiskLevel if lvl is not RiskLevel.UNKNOWN]
    auto = [lvl.name for lvl in levels if lvl <= max_auto]
    gated = [lvl.name for lvl in levels if lvl > max_auto]
    return (
        f"auto-approve: {', '.join(auto) or 'nothing'}; "
        f"require approval: {', '.join(gated)}, UNKNOWN (treated as DESTRUCTIVE)"
    )
