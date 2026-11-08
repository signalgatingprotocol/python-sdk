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
    commit = Store.commit
    observed_admissions = []

    def checked_commit(store, batch):
        for event in batch.events:
            for raw in event.payload.get("records", ()):
                if raw.get("kind") == "context_packet":
                    refs = raw["manifest"].get("observed_refs", ())
                    bound = {(ref.id, ref.revision) for ref in batch.expected_revisions}
                    assert all((ref["id"], ref["revision"]) in bound for ref in refs)
                    if refs:
                        observed_admissions.append(raw["id"])
        return commit(store, batch)

    monkeypatch.setattr(Store, "commit", checked_commit)
    result = await run_demo(tmp_path / "demo")
    assert result.passed, result.failures
    assert observed_admissions
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
    assert result.metrics["resumed_context_supplied"] == 1
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
        resumed_at = next(e.sequence for e in store.events() if e.event_type == "goal.resumed")
        admission = next(
            e
            for e in store.events()
            if e.sequence > resumed_at
            and any(raw.get("kind") == "context_packet" for raw in e.payload.get("records", ()))
        )
        resumed_id = next(
            raw["id"] for raw in admission.payload["records"] if raw.get("kind") == "context_packet"
        )
        resumed = next(
            r for r in state.records if isinstance(r, ContextPacket) and r.id == resumed_id
        )
        before = rebuild(tuple(e for e in store.events() if e.sequence < admission.sequence))
        prior = {r.id: r for r in before.records}
        rendered = json.loads(resumed.rendered_input)["untrusted_evidence"]
        useful = next(
            a
            for a in rendered["artifacts"]
            if a["text"] == ("Unrelated fixture arithmetic test: 2+2 == 4 PASS\n")
        )
        retained = prior[useful["ref"]["id"]]
        assert useful["ref"]["revision"] == retained.revision
        assert rendered["verification_status"]
        gap = next(
            status
            for status in rendered["verification_status"]
            if status["criterion"]["id"] == resumed.goal.criteria[0].id
        )
        finding = prior[gap["ref"]["id"]]
        assert finding.revision == gap["ref"]["revision"]
        assert finding.disposition == gap["disposition"] == "insufficient_evidence"
        assert finding.reason_code == gap["reason_code"] == "registered_check_unproven"
        assert gap["criterion"]["revision"] == resumed.goal.criteria[0].revision == 2
        assert any(c.id == old.task.commitment_id for c in resumed.commitments)
        assert resumed.goal.constraints[0].hard
        from signal_gating.executive.authority import reference
        from signal_gating.executive.demo import _Worker

        principal_id = json.loads(resumed.rendered_input)["trusted_authority"]["principal"]["id"]
        assert principal_id in retained.shared_with
        assert reference(retained) in resumed.mandatory_read_set
        assert reference(finding) not in resumed.mandatory_read_set
        assert reference(finding) in resumed.manifest.observed_refs
        assert gap["semantics"] == "observation_as_of_admission"
        current_finding = next(r for r in state.records if r.id == finding.id)
        assert current_finding.revision > finding.revision
        assert current_finding.disposition == "accept"
        assert current_finding.reason_code == "registered_check_passed"
        check = next(a for a in actions if a.capability_id == "run_check")
        assert not any(ref.id == finding.id for ref in check.read_set)
        assert any(
            ref.id == gap["criterion"]["id"] and ref.revision == 2
            for ref in resumed.mandatory_read_set
        )
        assert not any(
            isinstance(r, ActionRecord)
            and r.capability_id == "run_check"
            and r.status == "succeeded"
            for r in before.records
        )
        # Each omission independently prevents repair, even with the phase selected.
        for missing in ("useful_artifact", "compatibility_gap"):
            data = json.loads(resumed.rendered_input)
            evidence = data["untrusted_evidence"]
            if missing == "useful_artifact":
                evidence["artifacts"] = [
                    a for a in evidence["artifacts"] if a["ref"]["id"] != retained.id
                ]
            else:
                evidence["verification_status"] = [
                    status
                    for status in evidence["verification_status"]
                    if status["criterion"]["id"] != resumed.goal.criteria[0].id
                ]
            worker = _Worker()
            worker.phase = "repair"
            blocked = await worker.run_episode(
                resumed.model_copy(update={"rendered_input": json.dumps(data)})
            )
            assert blocked.stop_reason == "blocked" and not blocked.artifact_drafts
            assert blocked.blocker.reason_code == "resume_context_incomplete"
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


async def test_repair_phase_alone_cannot_supply_missing_resume_context():
    from signal_gating.executive.demo import _Worker

    from .factories import packet

    worker = _Worker()
    worker.phase = "repair"
    result = await worker.run_episode(packet())
    assert result.stop_reason == "blocked"
    assert not result.artifact_drafts
    assert result.blocker.reason_code == "resume_context_incomplete"
