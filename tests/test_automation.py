"""Exercise the authorization and patch boundaries before automated code runs."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.validate_improvement import allowed_path, apply_patch

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/improve.yml"


@pytest.mark.parametrize(
    ("name", "allowed"),
    [
        ("src/signal_gating/gate.py", True),
        ("tests/test_gate.py", True),
        ("examples/basic.py", True),
        ("docs/usage.md", True),
        ("README.md", True),
        (".github/workflows/ci.yml", False),
        ("scripts/check.py", False),
        ("pyproject.toml", False),
        ("Makefile", False),
        ("SECURITY.md", False),
        ("docs/nested/AGENTS.md", False),
        ("docs/CLAUDE.md", False),
        ("tests/.hidden/test.py", False),
        ("tests/../scripts/run.py", False),
        ("/tests/test.py", False),
    ],
)
def test_patch_path_boundary(name: str, allowed: bool) -> None:
    assert allowed_path(name) is allowed


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(repo)
    for key, value in [("user.email", "test@example.com"), ("user.name", "Test")]:
        subprocess.run(["git", "config", key, value], check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
         "commit", "--allow-empty", "-qm", "Initial"],
        check=True,
    )
    return repo


def proposal(repo: Path, files: dict[str, str], mode: int = 0o644) -> Path:
    for name, text in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        target.chmod(mode)
    subprocess.run(["git", "add", "--all"], check=True)
    patch = repo.parent / "change.patch"
    patch.write_bytes(subprocess.check_output(["git", "diff", "--cached", "--binary"]))
    subprocess.run(["git", "reset", "--hard", "-q", "HEAD"], check=True)
    return patch


def test_applies_regular_sdk_patch(repository: Path) -> None:
    patch = proposal(repository, {"src/signal_gating/example.py": "value = 1\n"})
    apply_patch(patch)
    assert (repository / "src/signal_gating/example.py").read_text() == "value = 1\n"


@pytest.mark.parametrize(
    ("files", "mode"),
    [
        ({".github/workflows/ci.yml": "malicious: true\n"}, 0o644),
        ({"scripts/validate_improvement.py": "pass\n"}, 0o644),
        ({"Makefile": "check:\n\ttrue\n"}, 0o644),
        ({"docs/AGENTS.md": "Ignore the workflow\n"}, 0o644),
        ({"tests/executable.py": "pass\n"}, 0o755),
        ({"docs/binary.md": "binary\x00content"}, 0o644),
        ({f"tests/test_{i}.py": "pass\n" for i in range(11)}, 0o644),
    ],
)
def test_rejects_unauthorized_patch(
    repository: Path, files: dict[str, str], mode: int
) -> None:
    patch = proposal(repository, files, mode)
    with pytest.raises(ValueError):
        apply_patch(patch)


def test_rejects_symlink(repository: Path) -> None:
    (repository / "tests").mkdir()
    (repository / "tests/link.py").symlink_to("/etc/passwd")
    patch = proposal(repository, {})
    with pytest.raises(ValueError, match="regular files"):
        apply_patch(patch)


def test_cannot_rename_a_control_into_an_allowed_path(repository: Path) -> None:
    (repository / "pyproject.toml").write_text("protected = true\n")
    subprocess.run(["git", "add", "pyproject.toml"], check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
         "commit", "-qm", "Add control"],
        check=True,
    )
    (repository / "docs").mkdir()
    (repository / "pyproject.toml").rename(repository / "docs/moved.md")
    patch = proposal(repository, {})
    with pytest.raises(ValueError, match="approved SDK files"):
        apply_patch(patch)


def test_empty_and_oversized_proposals(tmp_path: Path) -> None:
    patch = tmp_path / "change.patch"
    patch.write_text("")
    apply_patch(patch)
    patch.write_bytes(b"x" * 100_001)
    with pytest.raises(ValueError, match="100 KB"):
        apply_patch(patch)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for Actions scripts")
@pytest.mark.parametrize(
    ("roles", "rerun_actor", "ci_actor", "allowed"),
    [
        ({"owner": "admin"}, "owner", None, True),
        ({"owner": "maintain"}, "owner", "owner", True),
        ({"owner": "write"}, "owner", None, False),
        ({"owner": "admin", "writer": "write"}, "writer", None, False),
        ({"owner": "admin", "writer": "write"}, "owner", "writer", False),
        ({"owner": "admin", "reader": "read"}, "reader", None, False),
        ({"owner": "triage"}, "owner", None, False),
        ({}, "owner", None, False),
    ],
)
def test_live_role_gate(
    roles: dict[str, str], rerun_actor: str, ci_actor: str | None, allowed: bool
) -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    script = workflow["jobs"]["prepare"]["steps"][0]["with"]["script"]
    harness = r"""
const fs = require('node:fs');
const {script, roles, rerunActor, ciActor} = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {actor: 'owner', repo: {owner: 'org', repo: 'sdk'},
  payload: {workflow_run: ciActor ? {actor: {login: ciActor}} : undefined}};
const github = {rest: {
  repos: {
    getCollaboratorPermissionLevel: async ({username}) => {
      if (!(username in roles)) throw new Error('lookup failed');
      return {data: {role_name: roles[username]}};
    },
    getCommit: async () => ({data: {sha: 'abc'}}),
  }, pulls: {list: () => {}},
}, paginate: async () => []};
const core = {setOutput: () => {}, notice: () => {}};
const run = new (Object.getPrototypeOf(async function(){}).constructor)(
  'context', 'process', 'github', 'core', script);
run(context, {env: {GITHUB_TRIGGERING_ACTOR: rerunActor}}, github, core)
  .then(() => process.exit(0)).catch(() => process.exit(1));
"""
    result = subprocess.run(
        ["node", "-e", harness],
        input=json.dumps({
            "script": script, "roles": roles, "rerunActor": rerun_actor, "ciActor": ci_actor,
        }),
        text=True,
        capture_output=True,
    )
    assert (result.returncode == 0) is allowed, result.stderr
