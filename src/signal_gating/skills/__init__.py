"""Agent Skills compatibility.

The Agent Skills open standard (agentskills.io, 40+ tools, Linux Foundation
governance pending) is the converged distribution format for agent
capabilities: a directory with a ``SKILL.md`` (name + description
frontmatter) plus optional ``scripts/``, ``references/``, ``assets/``.

This package makes SGP the control plane for skills:

- ``export_skill``: turn an SGP agent's tool surface into a skill directory
  other agents can install.
- ``load_skill`` / ``load_skills_dir``: read installed skills and register
  their scripts as SGP tools, each classified with a risk level.
- ``CONTROL_SKILL_MD``: the canonical ``sgp-control`` skill text, so any
  skills-compatible agent can adopt fail-closed tool gating by installing
  one directory.
"""

from signal_gating.skills.skill import (
    CONTROL_SKILL_MD,
    Skill,
    export_skill,
    load_skill,
    load_skills_dir,
    parse_skill_md,
)

__all__ = [
    "CONTROL_SKILL_MD",
    "Skill",
    "export_skill",
    "load_skill",
    "load_skills_dir",
    "parse_skill_md",
]
