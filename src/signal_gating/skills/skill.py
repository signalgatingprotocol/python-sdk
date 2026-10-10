"""Agent Skills spec implementation for SGP.

Spec (agentskills.io): a skill is a directory containing SKILL.md with
YAML frontmatter (``name`` and ``description`` required), plus optional
``scripts/``, ``references/``, ``assets/`` subdirectories. Progressive
disclosure: agents load metadata at startup, full instructions on use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from signal_gating.security.risk import RiskLevel, ToolRisk, classify_tool

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass(slots=True)
class Skill:
    """An installed skill: metadata plus local paths."""

    name: str
    description: str
    path: Path
    scripts: list[Path] = field(default_factory=list)
    references: list[Path] = field(default_factory=list)

    @property
    def instructions(self) -> str:
        return (self.path / "SKILL.md").read_text(encoding="utf-8")


def parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    """Split SKILL.md into (frontmatter dict, body). Minimal YAML subset."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("SKILL.md must start with --- frontmatter ---")
    raw, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("'\"")
    if "name" not in meta or "description" not in meta:
        raise ValueError("SKILL.md frontmatter requires 'name' and 'description'")
    return meta, body


def load_skill(path: str | Path) -> Skill:
    """Load one skill directory."""
    path = Path(path)
    skill_md = path / "SKILL.md"
    if not skill_md.is_file():
        raise ValueError(f"{path} is not a skill: SKILL.md missing")
    meta, _ = parse_skill_md(skill_md.read_text(encoding="utf-8"))
    scripts_dir = path / "scripts"
    refs_dir = path / "references"
    return Skill(
        name=meta["name"],
        description=meta["description"],
        path=path,
        scripts=sorted(scripts_dir.glob("*.py")) if scripts_dir.is_dir() else [],
        references=sorted(refs_dir.glob("*.md")) if refs_dir.is_dir() else [],
    )


def load_skills_dir(root: str | Path) -> list[Skill]:
    """Load every skill under a skills root (e.g. .agents/skills/)."""
    root = Path(root)
    if not root.is_dir():
        return []
    skills: list[Skill] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "SKILL.md").is_file():
            skills.append(load_skill(child))
    return skills


def export_skill(
    name: str,
    description: str,
    tools: list[dict[str, Any]],
    dest: str | Path,
) -> Path:
    """Write a skill directory from an SGP tool surface.

    ``tools`` entries: {"name", "description", "risk": RiskLevel (optional)}.
    Every exported tool is risk-classified; the SKILL.md documents the
    gating contract so installing agents know what they are getting.
    """
    dest = Path(dest) / name.replace(" ", "-").lower()
    (dest / "scripts").mkdir(parents=True, exist_ok=True)
    (dest / "references").mkdir(parents=True, exist_ok=True)

    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        "---",
        "",
        f"# {name}",
        "",
        description,
        "",
        "## Gated tools",
        "",
        "Every tool below is risk-classified. Install this skill behind a",
        "fail-closed policy engine: READ tools auto-approve, everything else",
        "requires approval.",
        "",
    ]
    for tool in tools:
        risk = tool.get("risk")
        if risk is None:
            classified = classify_tool(tool["name"], tool.get("description", ""))
            level = classified.level
        elif isinstance(risk, RiskLevel):
            level = risk
        else:
            level = RiskLevel[str(risk).upper()]
        lines.append(f"- `{tool['name']}` ({level.name}): {tool.get('description', '')}")
    lines += [
        "",
        "## References",
        "",
        "See `references/policy.md` for the recommended default-deny policy.",
    ]
    (dest / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    (dest / "references" / "policy.md").write_text(
        "# Recommended policy\n\n"
        "```yaml\n"
        "version: 1\n"
        "name: skill-default\n"
        "default: deny\n"
        "max_auto_approve: READ\n"
        "```\n",
        encoding="utf-8",
    )
    return dest


CONTROL_SKILL_MD = """---
name: sgp-control
description: Fail-closed tool authorization for AI agents. Use when an agent needs to call tools with risk-proportional gating.
---

# sgp-control

The control plane for agentic tool use. Drop this skill into `.agents/skills/`
and wire its policy engine in front of every tool call.

## The contract

1. Every tool is classified: READ, DRAFT, SEND, DESTRUCTIVE, or UNKNOWN.
2. UNKNOWN is treated as DESTRUCTIVE. Unclassified tools never execute.
3. READ tools auto-approve. Everything else requires approval or denies.
4. Every decision is audit-logged. Denial is a value, never an exception.

## Quick start

```python
from signal_gating.security import PolicyEngine, RiskLevel, ToolRisk

engine = PolicyEngine(max_auto_approve=RiskLevel.READ)
engine.register("read_file", ToolRisk(RiskLevel.READ, ("read-like verb",), 0.9))

decision = engine.evaluate("read_file", agent_name="assistant")
assert decision.value == "allow"

decision = engine.evaluate("delete_database", agent_name="assistant")
assert decision.value == "deny"  # unknown tool: default deny
```

## Policy as code

```yaml
version: 1
name: production
default: deny
max_auto_approve: READ
rules:
  - match: {tools: ["read_*"]}
    allow: true
  - match: {risk: ["SEND", "DESTRUCTIVE"]}
    require_approval: true
```
"""
