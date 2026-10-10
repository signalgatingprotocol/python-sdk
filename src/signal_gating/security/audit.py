"""MCP annotation distrust.

MCP servers self-declare tool annotations: readOnlyHint, destructiveHint,
idempotentHint, openWorldHint. The Agentjacking attack class (June 2026)
demonstrated that these annotations are not a reliable trust signal:
a compromised or malicious server simply lies.

This module verifies the claim. It compares what a tool *declares* against
what it *observably does* (static source scan plus runtime behavior record)
and reports every mismatch as a finding. A tool that declares readOnlyHint
but opens a socket fails the audit.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class DeclaredHints:
    """What an MCP server (or tool author) claims about a tool."""

    read_only_hint: bool = False
    destructive_hint: bool = False
    idempotent_hint: bool = False
    open_world_hint: bool = False


@dataclass(slots=True, frozen=True)
class ObservedBehavior:
    """What a tool was observed to do.

    Populated by static source scanning (``scan_source``) and/or by the
    runtime behavior recorder. Either source alone is enough to raise
    a finding; both together raise confidence.
    """

    reads_filesystem: bool = False
    writes_filesystem: bool = False
    network_calls: bool = False
    subprocess_calls: bool = False
    eval_or_exec: bool = False
    privilege_ops: bool = False
    sources: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class AuditFinding:
    """One mismatch between declaration and observation."""

    code: str
    severity: str  # "critical", "high", "medium"
    message: str


@dataclass(slots=True)
class AuditReport:
    """The full result of auditing one tool."""

    tool_name: str
    declared: DeclaredHints
    observed: ObservedBehavior
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def trusted(self) -> bool:
        """A tool is trusted only if it passes audit with no criticals."""
        return self.passed


# Static source patterns: (pattern, behavior flag, description)
_SOURCE_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = tuple(
    (re.compile(p), flag, desc)
    for p, flag, desc in [
        (r"\bopen\s*\(.{0,80}['\"]w['\"]", "writes_filesystem", "open() in write mode"),
        (r"\.\s*write\s*\(", "writes_filesystem", ".write() call"),
        (r"\bos\s*\.\s*(remove|unlink|rmdir|makedirs|rename|replace)\b", "writes_filesystem", "os filesystem mutation"),
        (r"\bpathlib\b.{0,40}\.(write_text|write_bytes|unlink|mkdir|rename)\b", "writes_filesystem", "pathlib mutation"),
        (r"\bshutil\s*\.\s*(rmtree|move|copy)", "writes_filesystem", "shutil mutation"),
        (r"\b(socket|requests|httpx|urllib|aiohttp|websocket)\b", "network_calls", "network library usage"),
        (r"\b(subprocess|os\s*\.\s*system|os\s*\.\s*popen|Popen)\b", "subprocess_calls", "subprocess invocation"),
        (r"\b(eval|exec)\s*\(", "eval_or_exec", "eval/exec call"),
        (r"\b(chmod|chown|setuid|sudo)\b", "privilege_ops", "privilege operation"),
        (r"['\"](UPDATE|DELETE\s+FROM|INSERT\s+INTO|DROP\s+TABLE|ALTER\s+TABLE|TRUNCATE)\b", "writes_filesystem", "SQL mutation statement"),
        (r"\bopen\s*\(", "reads_filesystem", "open() call"),
    ]
)


def scan_source(source: str) -> ObservedBehavior:
    """Statically scan Python source for observable side-effect signals."""
    flags: dict[str, bool] = {}
    matched: list[str] = []
    for pattern, flag, desc in _SOURCE_PATTERNS:
        if pattern.search(source):
            flags[flag] = True
            matched.append(desc)
    # A write-mode open() also reads; keep both flags honest.
    return ObservedBehavior(
        reads_filesystem=flags.get("reads_filesystem", False),
        writes_filesystem=flags.get("writes_filesystem", False),
        network_calls=flags.get("network_calls", False),
        subprocess_calls=flags.get("subprocess_calls", False),
        eval_or_exec=flags.get("eval_or_exec", False),
        privilege_ops=flags.get("privilege_ops", False),
        sources=tuple(matched),
    )


def audit_tool(
    tool_name: str,
    declared: DeclaredHints,
    observed: ObservedBehavior,
) -> AuditReport:
    """Compare declaration against observation. Distrust by default."""
    findings: list[AuditFinding] = []

    side_effects = (
        observed.writes_filesystem
        or observed.network_calls
        or observed.subprocess_calls
        or observed.eval_or_exec
        or observed.privilege_ops
    )

    if declared.read_only_hint and side_effects:
        performed = [s for s in observed.sources]
        findings.append(
            AuditFinding(
                code="READONLY_LIE",
                severity="critical",
                message=(
                    f"Tool {tool_name!r} declares readOnlyHint but performs "
                    f"side effects: {', '.join(performed) or 'observed at runtime'}. "
                    "Treat the declaration as hostile."
                ),
            )
        )

    if declared.destructive_hint is False and (
        observed.writes_filesystem or observed.privilege_ops
    ):
        findings.append(
            AuditFinding(
                code="DESTRUCTIVE_UNDISCLOSED",
                severity="high",
                message=(
                    f"Tool {tool_name!r} omits destructiveHint but mutates "
                    "state. destructiveHint=false is not evidence of safety."
                ),
            )
        )

    if observed.eval_or_exec:
        findings.append(
            AuditFinding(
                code="CODE_EXECUTION",
                severity="critical",
                message=(
                    f"Tool {tool_name!r} executes dynamically generated code. "
                    "No annotation combination makes this safe to auto-approve."
                ),
            )
        )

    if observed.subprocess_calls and not declared.open_world_hint:
        findings.append(
            AuditFinding(
                code="HIDDEN_SUBPROCESS",
                severity="high",
                message=(
                    f"Tool {tool_name!r} spawns subprocesses without "
                    "openWorldHint. The declared interaction surface is false."
                ),
            )
        )

    return AuditReport(
        tool_name=tool_name,
        declared=declared,
        observed=observed,
        findings=findings,
    )
