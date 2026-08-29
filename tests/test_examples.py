from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "filename",
    ["agent_team.py", "scripted_workflow.py"],
)
def test_workflow_examples_do_not_run_when_loaded_as_modules(
    filename: str,
    tmp_path: Path,
) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / filename
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "runpy.run_path(sys.argv[1], run_name='example_under_test')"
            ),
            str(example),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
