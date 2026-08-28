from __future__ import annotations

import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

from signal_gating.demo import TriageResult

EXPECTED_OUTPUT = """signals_in=12
signals_admitted=3
handler_load_reduction=75.0%
priority_drops=6
duplicate_drops=3
critical_incidents_preserved=3/3
result=PASS
"""


def test_demo_module_runs_the_measurable_proof_from_any_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "signal_gating.demo"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == EXPECTED_OUTPUT
    assert completed.stderr == ""


def test_distribution_exposes_demo_console_command() -> None:
    matching = [
        entry_point
        for entry_point in distribution("signal-gating").entry_points
        if entry_point.group == "console_scripts" and entry_point.name == "signal-gating-demo"
    ]

    assert [entry_point.value for entry_point in matching] == ["signal_gating.demo:main"]


def test_demo_pass_requires_the_exact_expected_fixture() -> None:
    valid = TriageResult(
        total_alerts=12,
        admitted_alerts=3,
        priority_drops=6,
        duplicate_drops=3,
        unique_critical_incidents=3,
        critical_incidents_preserved=3,
    )
    lost_incident = TriageResult(
        total_alerts=12,
        admitted_alerts=2,
        priority_drops=6,
        duplicate_drops=3,
        unique_critical_incidents=3,
        critical_incidents_preserved=2,
    )

    assert getattr(valid, "matches_expected_fixture", False)
    assert not getattr(lost_incident, "matches_expected_fixture", False)
