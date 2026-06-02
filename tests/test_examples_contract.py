"""Contracts for examples promoted in docs."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_financial_physics_mesh_example_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "financial_physics_mesh.py")],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "accepted_decisions=2" in completed.stdout
    assert "all_receipts_verify=True" in completed.stdout
    assert "ACCEPTED BUY AAPL" in completed.stdout
