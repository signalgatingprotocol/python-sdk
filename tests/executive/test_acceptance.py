"""Nine-step synthetic release acceptance through the real local coordinator."""

import json
import os
import subprocess
import sys
from pathlib import Path

from signal_gating.executive import (
    ActionRecord,
    ArtifactRef,
    Commitment,
    ContextPacket,
    DecisionReceipt,
    ExecutiveSignal,
    Goal,
    Grant,
    TaskFrame,
)
from signal_gating.executive.demo import run_demo
from signal_gating.executive.reducers import rebuild
from signal_gating.executive.store import Store


async def test_nine_step_release_and_measured_receipts(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    result = await run_demo(tmp_path / "demo")
    assert result.passed, result.failures
    assert result.completed
    assert result.scope == "synthetic_fixture"
    assert result.metrics["stale_proposals_rejected"] == 1
    assert result.metrics["forbidden_dispatches"] == 0
    assert result.metrics["resumed_commitments"] == 1
    assert result.metrics["coalesced_events"] == 2
    assert result.metrics["compatibility_checks_passed"] == 1
    assert result.metrics["unrelated_artifacts_retained"] == 1
    assert result.metrics["reconciled_effects"] == 1
    assert result.metrics["interrupted_frames_retained"] == 1
    with Store.open_readonly(tmp_path / "demo" / "store") as store:
        state = store.view()
        assert rebuild(store.events()) == state
        assert len([r for r in state.records if isinstance(r, Commitment)]) == 1
        assert [r.status for r in state.records if isinstance(r, Goal)] == ["completed"]
        assert [r.status for r in state.records if isinstance(r, TaskFrame)] == ["completed"]
        actions = [r for r in state.records if isinstance(r, ActionRecord)]
        assert {a.capability_id for a in actions} == {
            "write_text",
            "run_check",
            "fixture_prepare_review",
        }
        assert all(a.status == "succeeded" and a.dispatch_attempts == 1 for a in actions)
        assert any(isinstance(r, ArtifactRef) and not r.shared_with for r in state.records)
        revisions = [
            raw["revision"]
            for e in store.events()
            for raw in e.payload.get("records", ())
            if raw.get("kind") == "goal"
        ]
        assert {1, 2, 3, 4, 5} <= set(revisions)
        old = next(
            r for r in state.records if isinstance(r, ContextPacket) and r.goal.revision == 4
        )
        assert any(ref.id == old.goal.id and ref.revision == 4 for ref in old.mandatory_read_set)
        assert any(
            isinstance(r, DecisionReceipt)
            and r.subject_id == old.episode_id
            and r.disposition == "stale"
            for r in state.records
        )
        assert not any(action.episode_id == old.episode_id for action in actions)
        logs = [
            r for r in state.records if isinstance(r, ExecutiveSignal) and r.signal_kind == "log"
        ]
        assert len(logs) == len({log.producer_id for log in logs}) == 3
        goal = next(r for r in state.records if isinstance(r, Goal))
        assert len(goal.constraints) == 1 and goal.constraints[0].hard
        assert goal.constraints[0].enforcement_id == "fixture-python310-v1"
        assert all(
            "deploy" not in grant.capabilities
            for grant in state.records
            if isinstance(grant, Grant)
        )
        assert any(e.event_type == "goal.paused" for e in store.events())
        assert (
            sum(e.payload.get("reason_code") == "kernel_configuration" for e in store.events()) == 2
        )


def test_cli_demo_runs_outside_checkout(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "signal_gating.executive",
            "demo",
            "--store",
            str(tmp_path / "demo"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    output = json.loads(result.stdout)
    assert output["evaluation"]["passed"]
    assert output["evaluation"]["scope"] == "synthetic_fixture"
    assert output["evaluation"]["metrics"]["forbidden_dispatches"] == 0
    assert not result.stderr


def test_static_310_checker_rejects_incompatible_project_then_checks_repair(tmp_path):
    from signal_gating.executive.demo import (
        COMPATIBILITY_CHECK,
        INCOMPATIBLE_SOURCE,
        RELEASE_SOURCE,
        REQUIREMENT,
    )

    (tmp_path / "release.py").write_text(INCOMPATIBLE_SOURCE)
    (tmp_path / "requires-python.txt").write_text(REQUIREMENT)
    command = [sys.executable, "-I", "-S", "-c", COMPATIBILITY_CHECK]
    incompatible = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert incompatible.returncode != 0
    assert "SyntaxError" in incompatible.stderr
    assert "PASS" not in incompatible.stdout
    (tmp_path / "release.py").write_text(RELEASE_SOURCE)
    compatible = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert compatible.returncode == 0
    assert "static Python 3.10 grammar and current-interpreter test PASS" in compatible.stdout
    assert not compatible.stderr
