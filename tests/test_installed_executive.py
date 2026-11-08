"""Exercise a real wheel from another cwd, without optional model dependencies."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import venv
from pathlib import Path

import pytest

from signal_gating import __version__

ROOT = Path(__file__).resolve().parents[1]
LEGACY_EXPORTS = ["Agent", "Gate", "Mesh", "MeshEvent", "Receipt", "Signal", "TrajectoryRecorder"]


def test_source_distribution_excludes_private_continuation_records(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", ".gitignore"):
        shutil.copy2(ROOT / name, project / name)
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
    private = project / ".superpowers" / "sdd"
    private.mkdir(parents=True)
    (private / ".gitignore").write_text("*\n")
    (private / "private-continuation.txt").write_text("synthetic private continuation marker\n")
    output = tmp_path / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "sdist", "-d", str(output)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        names = archive.getnames()
    assert any(name.endswith("/src/signal_gating/executive/cli.py") for name in names)
    assert not any("/.superpowers/" in name for name in names)


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    root = tmp_path_factory.mktemp("executive-wheel")
    wheels = root / "wheels"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheels),
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    environment = root / "venv"
    venv.create(environment, with_pip=True)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(next(wheels.glob("*.whl")))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    # Reuse the runner's declared dependencies without a network install. Nested
    # venv system_site_packages would inherit the base interpreter, not the runner.
    # Add only its site-packages directory; its editable .pth files are not run.
    location = subprocess.run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    (Path(location.stdout.strip()) / "test-runtime-dependencies.pth").write_text(
        sysconfig.get_path("purelib") + "\n"
    )
    return python


def invoke(python, cwd, *arguments, module=False):
    command = (
        [str(python), "-I", "-m", "signal_gating.executive"]
        if module
        else [
            str(
                python.parent
                / ("signal-gating-executive.exe" if os.name == "nt" else "signal-gating-executive")
            )
        ]
    )
    # No checkout path injection; stdin EOF closes the foreground owner session.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return subprocess.run(
        [*command, *map(str, arguments)],
        cwd=cwd,
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_wheel_has_console_entrypoint_and_dependency_light_compatible_import(installed, tmp_path):
    code = """
import importlib.abc, importlib.metadata, json, pathlib, sys
class NoOpenAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'openai':
            raise AssertionError('base wheel imported optional OpenAI')
sys.meta_path.insert(0, NoOpenAI())
import signal_gating, signal_gating.core as core
from signal_gating.executive import export_contract_schemas
from signal_gating.executive.cli import main
assert pathlib.Path(signal_gating.__file__).is_relative_to(pathlib.Path(sys.prefix))
dist = importlib.metadata.distribution('signal-gating')
entries = [e.value for e in dist.entry_points
           if e.group == 'console_scripts' and e.name == 'signal-gating-executive']
print(json.dumps({'entries': entries, 'legacy': core.__all__,
                  'schemas': len(export_contract_schemas()['schemas']),
                  'version': signal_gating.__version__}))
"""
    result = subprocess.run(
        [str(installed), "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["entries"] == ["signal_gating.executive.cli:main"]
    assert report["legacy"] == LEGACY_EXPORTS
    assert report["schemas"] == 69
    assert report["version"] == __version__
    assert not result.stderr


def test_installed_demo_completes_and_restart_inspection_is_readonly(installed, tmp_path):
    demo = tmp_path / "demo"
    completed = invoke(installed, tmp_path, "demo", "--store", demo)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)["evaluation"]
    assert report["metrics"]["forbidden_dispatches"] == 0
    assert report["passed"] is True
    assert report["scope"] == "synthetic_fixture"
    inspected = invoke(
        installed, tmp_path, "inspect", "--store", demo / "store", "--json", module=True
    )
    assert inspected.returncode == 0, inspected.stdout + inspected.stderr
    state = json.loads(inspected.stdout)
    assert state["goals"][0]["status"] == "completed"
    assert state["summary"]["completed_goal_ids"] == [state["goals"][0]["id"]]
    again = invoke(installed, tmp_path, "inspect", "--store", demo / "store", "--json")
    assert again.returncode == 0, again.stdout + again.stderr
    assert json.loads(again.stdout)["sequence"] == state["sequence"]
    assert not completed.stderr + inspected.stderr + again.stderr


def test_installed_config_goal_loading_restart_and_export(installed, tmp_path):
    config = tmp_path / "config.json"
    goal = tmp_path / "goal.json"
    config.write_bytes((ROOT / "examples/executive/local-config.json").read_bytes())
    goal.write_bytes((ROOT / "examples/executive/goal.json").read_bytes())
    store = tmp_path / "store"
    other = tmp_path / "elsewhere"
    other.mkdir()
    run = invoke(
        installed, other, "run", "--store", store, "--config", config, "--goal", goal, "--batch"
    )
    assert run.returncode == 3, run.stdout + run.stderr
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    assert rows[0]["receipt"]["disposition"] == "accept"
    final = rows[-1]
    goal_id = json.loads(goal.read_text())["id"]
    assert final["goals"][0]["id"] == goal_id
    assert final["tasks"][0]["wake_condition"]["reason_code"] == "fixture_owner_needed"
    resume = invoke(
        installed,
        other,
        "run",
        "--store",
        store,
        "--config",
        config,
        "--resume",
        goal_id,
        "--batch",
    )
    assert resume.returncode == 3, resume.stdout + resume.stderr
    loaded = json.loads(resume.stdout.splitlines()[0])
    assert loaded["status"] == "loaded" and loaded["receipt"] is None
    assert loaded["commitments"] == final["commitments"]
    output = tmp_path / "observation.json"
    exported = invoke(
        installed, other, "export", "--store", store, "--goal", goal_id, "--output", output
    )
    assert exported.returncode == 0, exported.stdout + exported.stderr
    assert json.loads(output.read_text())["authority"] == "observation_only"
    assert not run.stderr + resume.stderr + exported.stderr


@pytest.mark.parametrize("live", [False, True])
def test_installed_live_diagnostics_precede_provider_or_store_creation(installed, tmp_path, live):
    value = json.loads((ROOT / "examples/executive/local-config.json").read_text())
    value["workers"][0].update(
        kind="openai", model="operator-model", api_key_env="SGP_INSTALLED_TEST_UNUSED_KEY"
    )
    config = tmp_path / "live.json"
    config.write_text(json.dumps(value))
    goal = tmp_path / "goal.json"
    goal.write_bytes((ROOT / "examples/executive/goal.json").read_bytes())
    store = tmp_path / "store"
    flags = ["--live"] if live else []
    result = invoke(
        installed,
        tmp_path,
        "run",
        "--store",
        store,
        "--config",
        config,
        "--goal",
        goal,
        "--batch",
        *flags,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    error = json.loads(result.stdout)
    assert error["reason_code"] == "startup_error"
    expected_detail = "maximum_charge_microunits" if live else "--live"
    assert expected_detail in error["detail"]
    assert not store.exists()
    assert not result.stderr
