"""SIGKILL only test-owned children, at real durable coordinator boundaries.

The child uses actual Kernel/Dispatcher/LocalWorkspace. Wrappers below are test
barriers, never production crash hooks. No child runs a command or spawns a
process of its own; kill/wait/communicate always drain that exact owned child.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from signal_gating.executive import (
    ActionIntent,
    ActionObservation,
    ActionRecord,
    ArtifactDraft,
    ArtifactRef,
    CapabilitySpec,
    Commitment,
    CreateGoal,
    DecideAction,
    DecisionReceipt,
    EpisodeResult,
    Goal,
    Limits,
    PauseGoal,
    Reservation,
    Reverify,
    Scope,
    SetGrant,
    ShareArtifact,
    TaskFrame,
    Usage,
    WorkspaceConfig,
)
from signal_gating.executive.actions import CapabilityRegistry
from signal_gating.executive.reducers import rebuild
from signal_gating.executive.store import Store

from .factories import GOAL_ID, grant, owner
from .test_kernel import FakeClock, blocked, configuration, new_goal, open_kernel, records

BOUNDARIES = (
    "before_authorization",
    "after_authorization",
    "after_dispatch",
    "after_effect",
    "before_observation",
)
TEXT = "Actual fixture write, once.\n"


class FaultWorker:
    def __init__(self, *, durable=False):
        self.phase = "intent" if durable else "draft"
        self.durable = durable

    async def run_episode(self, packet):
        if self.phase == "draft":
            self.phase = "intent"
            return blocked(packet).model_copy(
                update={
                    "artifact_drafts": (
                        ArtifactDraft(key="result", text=TEXT, media_type="text/plain"),
                    ),
                }
            )
        if self.phase == "intent":
            self.phase = "done"
            if self.durable:
                capability, arguments = "fixture.durable", {}
            else:
                visible = json.loads(packet.rendered_input)["untrusted_evidence"]["artifacts"]
                selected = next(a["ref"] for a in visible if a["text"] == TEXT)
                capability, arguments = (
                    "write_text",
                    {
                        "path": "result.txt",
                        "expected_sha256": "missing",
                        "content_ref": selected,
                    },
                )
            return EpisodeResult(
                stop_reason="proposed",
                usage=Usage.zero(),
                proposals=(
                    ActionIntent(
                        episode_id=packet.episode_id,
                        fencing_token=packet.fencing_token,
                        read_set=packet.mandatory_read_set,
                        capability_id=capability,
                        arguments=arguments,
                        scope=Scope(
                            owner_id=owner().id,
                            goal_id=GOAL_ID,
                            resource_paths=(".",) if self.durable else ("result.txt",),
                        ),
                    ),
                ),
            )
        return blocked(packet)


class DurableEffect:
    def __init__(self, root):
        self.root = root

    async def execute(self, action):
        with (self.root / "invocation-count.txt").open("a") as handle:
            handle.write("invocation\n")
            handle.flush()
            os.fsync(handle.fileno())
        # This fixture honors its durable key contract beyond the advertised
        # minimum window. Invocation counting still detects any unsafe retry.
        effect = self.root / ("effect-" + action.idempotency_key + ".txt")
        if not effect.exists():
            with effect.open("x") as handle:
                handle.write("effect\n")
                handle.flush()
                os.fsync(handle.fileno())
        return ActionObservation(
            action_id=action.id,
            expected_revision=action.revision,
            outcome="succeeded",
            usage=Usage.zero(),
        )

    async def reconcile(self, action):
        raise AssertionError("Unconfigured lookup must never run")

    async def cancel(self, action):
        return ActionObservation(
            action_id=action.id, expected_revision=action.revision, outcome="outcome_unknown"
        )


def setup(root, *, durable=False):
    registry = CapabilityRegistry()
    if durable:
        registry.register(
            CapabilitySpec(
                id="fixture.durable",
                binding_id="ae" * 16,
                binding_version="fixture-v1",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                effect_class="consequential",
                idempotency="durable",
                idempotency_scope="fixture-root",
                idempotency_retention_seconds=1,
                reconciliation="none",
                usage_accounting="known",
            ),
            DurableEffect(root),
            resource_scope=lambda intent: Scope(
                owner_id=owner().id, goal_id=GOAL_ID, resource_paths=(".",)
            ),
            quote=Limits(
                input_tokens=0, output_tokens=0, calls=1, elapsed_seconds=1, cost_microunits=0
            ),
            quote_source="synthetic_fixture",
            quote_version="v1",
        )
    config = configuration(
        workspace=WorkspaceConfig(root=root / "workspace", writable_paths=(".",))
    )
    return config, registry


def barrier(root, boundary, kernel):
    state = kernel.store.view()
    action = records(kernel, ActionRecord)[0]
    with (root / "ready.tmp").open("x") as handle:
        json.dump(
            {
                "boundary": boundary,
                "action_id": action.id,
                "status": action.status,
                "sequence": state.sequence,
                "pid": os.getpid(),
            },
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(root / "ready.tmp", root / "ready.json")
    while True:
        time.sleep(0.05)


async def child(root, boundary, durable=False):
    (root / "workspace").mkdir(parents=True)
    config, registry = setup(root, durable=durable)
    worker = FaultWorker(durable=durable)
    async with open_kernel(root / "store", worker, config=config, registry=registry) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        await kernel.submit(
            SetGrant(
                grant=grant(
                    subject_id=kernel.worker_principals["fixture"].id,
                    capabilities=("fixture.durable",) if durable else ("write_text",),
                    scope=Scope(owner_id=owner().id, goal_id=GOAL_ID, resource_paths=(".",)),
                )
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        if not durable:
            artifact = next(r for r in records(kernel, ArtifactRef) if r.media_type == "text/plain")
            assert not artifact.shared_with
            await kernel.submit(
                ShareArtifact(
                    artifact_id=artifact.id,
                    expected_revision=artifact.revision,
                    principal_ids=(kernel.worker_principals["fixture"].id,),
                ),
                principal=owner(),
            )
            await kernel.submit(Reverify(goal_id=GOAL_ID), principal=owner())
            await kernel.run_until_idle()
        action = records(kernel, ActionRecord)[0]
        assert action.status == "awaiting_review"
        await kernel.submit(
            DecideAction(
                action_id=action.id,
                expected_revision=action.revision,
                decision="approve",
                arguments_digest=action.arguments_digest,
                scope=action.scope,
                constraint_revision=action.constraint_revision,
            ),
            principal=owner(),
        )
        original_commit = kernel.dispatcher._commit

        def commit(selected, disposition, reason, **kwargs):
            if boundary == "before_authorization" and reason == "action_authorized":
                barrier(root, boundary, kernel)
            result = original_commit(selected, disposition, reason, **kwargs)
            if (boundary == "after_authorization" and reason == "action_authorized") or (
                boundary == "after_dispatch" and reason == "action_dispatching"
            ):
                barrier(root, boundary, kernel)
            return result

        kernel.dispatcher._commit = commit
        if boundary == "after_effect":
            original_replace = os.replace

            def replace(source, destination, **kwargs):
                result = original_replace(source, destination, **kwargs)
                if destination == "result.txt":
                    os.fsync(kwargs["dst_dir_fd"])
                    barrier(root, boundary, kernel)
                return result

            os.replace = replace
        if boundary == "before_observation":

            def observation(*args, **kwargs):
                barrier(root, boundary, kernel)

            kernel.dispatcher.record_observation = observation
        await kernel.run_until_idle()
        raise AssertionError("Boundary was never reached")


def kill_at(root, boundary, *, durable=False):
    checkout = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(checkout / "src"), str(checkout))))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.executive.test_faults",
            str(root),
            boundary,
            str(int(durable)),
        ],
        cwd=root.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not (root / "ready.json").exists():
            if process.poll() is not None:
                output, error = process.communicate(timeout=5)
                pytest.fail(f"Owned child exited before barrier: {output} {error}")
            if time.monotonic() >= deadline:
                pytest.fail("Owned child did not reach its durable barrier")
            time.sleep(0.02)
        marker = json.loads((root / "ready.json").read_text())
        assert marker["pid"] == process.pid and marker["boundary"] == boundary
        with Store.open_readonly(root / "store") as store:
            assert store.view() == rebuild(store.events())
            assert store.view().sequence == marker["sequence"]
        process.kill()
        output, error = process.communicate(timeout=5)
        assert process.returncode == -signal.SIGKILL
        assert output == error == ""
        return marker
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        assert process.poll() is not None


def effect_identity(path):
    info = path.stat()
    # Lookups may update atime. An effect must preserve file identity/content time.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@pytest.mark.parametrize("boundary", BOUNDARIES)
async def test_killed_process_recovers_without_duplicate_effect(tmp_path, boundary):
    root = tmp_path / "case"
    marker = kill_at(root, boundary)
    expected_status = "authorized" if boundary in BOUNDARIES[:2] else "dispatching"
    assert marker["status"] == expected_status
    file = root / "workspace" / "result.txt"
    effect = boundary in {"after_effect", "before_observation"}
    assert file.exists() == effect
    identity = effect_identity(file) if effect else None
    config, registry = setup(root)
    worker = FaultWorker()
    worker.phase = "done"
    async with open_kernel(root / "store", worker, config=config, registry=registry) as kernel:
        action = records(kernel, ActionRecord)[0]
        task_id = records(kernel, TaskFrame)[0].id
        commitment_id = records(kernel, Commitment)[0].id
        if boundary in BOUNDARIES[:2]:
            assert action.dispatch_attempts == 0
            await kernel.dispatcher.dispatch_one(action.id)
            assert file.read_text() == TEXT
            assert records(kernel, ActionRecord)[0].status == "succeeded"
            assert records(kernel, ActionRecord)[0].dispatch_attempts == 1
        else:
            assert action.status == "outcome_unknown"
            assert any(r.status == "uncertain" for r in records(kernel, Reservation))
            denial = await kernel.dispatcher.dispatch_one(action.id)
            assert denial.reason_code == "action_not_dispatchable"
            await kernel.dispatcher.reconcile_one(action.id)
            resolved = records(kernel, ActionRecord)[0]
            assert resolved.status == (
                "succeeded" if boundary == "before_observation" else "needs_human"
            )
            assert resolved.dispatch_attempts == 1
            assert file.exists() == effect
            if effect:
                assert effect_identity(file) == identity
        assert (
            len(records(kernel, ActionRecord))
            == len(records(kernel, TaskFrame))
            == len(records(kernel, Commitment))
            == 1
        )
        assert records(kernel, TaskFrame)[0].id == task_id
        assert records(kernel, Commitment)[0].id == commitment_id
        assert kernel.store.view() == rebuild(kernel.store.events())
        goal = records(kernel, Goal)[0]
        await kernel.submit(
            PauseGoal(goal_id=goal.id, expected_revision=goal.revision), principal=owner()
        )
    async with open_kernel(root / "store", worker, config=config) as kernel:
        await kernel.run_until_idle()
        assert records(kernel, Goal)[0].status == "paused"
        assert records(kernel, Commitment)[0].id == commitment_id
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_completed_effect_missing_receipt_after_kill_requires_human(tmp_path):
    root = tmp_path / "missing"
    kill_at(root, "before_observation")
    with Store.open_readonly(root / "store") as store:
        artifacts = [
            r
            for r in store.view().records
            if isinstance(r, ArtifactRef)
            and r.media_type == "application/vnd.sgp.workspace-receipt-v1+json"
        ]
        assert len(artifacts) == 2
        for artifact in artifacts:
            (root / "store" / artifact.storage_location).unlink()
    config, registry = setup(root)
    worker = FaultWorker()
    worker.phase = "done"
    before = effect_identity(root / "workspace" / "result.txt")
    async with open_kernel(root / "store", worker, config=config, registry=registry) as kernel:
        action = records(kernel, ActionRecord)[0]
        await kernel.dispatcher.reconcile_one(action.id)
        assert records(kernel, ActionRecord)[0].status == "needs_human"
        assert records(kernel, ActionRecord)[0].dispatch_attempts == 1
        assert any(r.status == "uncertain" for r in records(kernel, Reservation))
        assert effect_identity(root / "workspace" / "result.txt") == before
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_expired_idempotency_window_never_repeats_unknown_effect(tmp_path):
    root = tmp_path / "expired"
    marker = kill_at(root, "before_observation", durable=True)
    effect = root / ("effect-" + marker["action_id"] + ".txt")
    assert effect.read_text() == "effect\n"
    config, registry = setup(root, durable=True)
    clock = FakeClock()
    clock.advance(2)  # The registered durable retention was one second.
    worker = FaultWorker(durable=True)
    worker.phase = "done"
    async with open_kernel(
        root / "store", worker, config=config, registry=registry, clock=clock
    ) as kernel:
        kernel.dispatcher.max_retries = 1  # Prove expiry, not the default zero-retry setting.
        action = records(kernel, ActionRecord)[0]
        assert action.status == "outcome_unknown"
        assert action.dispatched_at is not None
        await kernel.dispatcher.reconcile_one(action.id)
        assert records(kernel, ActionRecord)[0].status == "needs_human"
        assert records(kernel, ActionRecord)[0].dispatch_attempts == 1
        assert effect.read_text() == "effect\n"
        assert (root / "invocation-count.txt").read_text() == "invocation\n"
        assert not any(
            r.reason_code == "durable_idempotent_retry"
            for r in records(
                kernel,
                DecisionReceipt,
            )
        )
        assert kernel.store.view() == rebuild(kernel.store.events())


if __name__ == "__main__":
    asyncio.run(child(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3]))))
