"""Tests for Agent Skills compatibility."""

import pytest

from signal_gating.security.risk import RiskLevel
from signal_gating.skills.skill import (
    CONTROL_SKILL_MD,
    export_skill,
    load_skill,
    load_skills_dir,
    parse_skill_md,
)


def test_parse_skill_md():
    meta, body = parse_skill_md(CONTROL_SKILL_MD)
    assert meta["name"] == "sgp-control"
    assert "description" in meta
    assert "control plane" in body


def test_parse_skill_md_requires_frontmatter():
    with pytest.raises(ValueError):
        parse_skill_md("no frontmatter here")


def test_export_and_load_skill(tmp_path):
    dest = export_skill(
        name="test-skill",
        description="A test skill",
        tools=[
            {"name": "read_file", "description": "Read a file", "risk": RiskLevel.READ},
            {"name": "mystery", "description": "Does something"},
        ],
        dest=tmp_path,
    )
    skill = load_skill(dest)
    assert skill.name == "test-skill"
    assert "READ" in skill.instructions
    assert (dest / "references" / "policy.md").is_file()


def test_load_skills_dir(tmp_path):
    export_skill("skill-a", "First", [], tmp_path)
    export_skill("skill-b", "Second", [], tmp_path)
    (tmp_path / "not-a-skill").mkdir()
    skills = load_skills_dir(tmp_path)
    assert {s.name for s in skills} == {"skill-a", "skill-b"}


def test_load_skills_dir_missing():
    assert load_skills_dir("/nonexistent/path") == []
