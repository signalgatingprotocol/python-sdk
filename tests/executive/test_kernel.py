"""End-to-end coordinator contracts using only controlled local workers."""

import asyncio
from datetime import timedelta

import pytest

from signal_gating.executive import (
    ArtifactDraft,
    ArtifactRef,
    Budget,
    CompletionClaim,
    CreateGoal,
    Episode,
    EpisodeResult,
    Goal,
    Kernel,
    KernelConfig,
    PauseGoal,
    Ref,
    Reservation,
    ResumeGoal,
    ReviseGoal,
    ShareArtifact,
    TaskFrame,
    Usage,
    WakeCondition,
    WorkerConfig,
)
from signal_gating.executive.actions import CapabilityRegistry
from signal_gating.executive.messages import Reverify
from signal_gating.executive.reducers import rebuild

from .factories import NOW, criterion, goal, owner
from .test_workers import ControlledWorker


class FakeClock:
    def __init__(self):
        self.current = NOW
        self.waiters = []

    def now(self):
        return self.current

    def monotonic(self):
        return (self.current - NOW).total_seconds()

    async def wait_until(self, at):
        if self.current >= at:
            return
        ready = asyncio.get_running_loop().create_future()
        self.waiters.append((at, ready))
        try:
            await ready
        finally:
            self.waiters.remove((at, ready))

    def advance(self, seconds):
        self.current += timedelta(seconds=seconds)
        for at, ready in self.waiters:
            if self.current >= at and not ready.done():
                ready.set_result(None)


def usage(**updates):
    return Usage.zero().model_copy(update={"calls": 1, **updates})


class FunctionWorker:
    def __init__(self, function):
        self.function, self.packets = function, []

    async def run_episode(self, packet):
        self.packets.append(packet)
        return self.function(packet)


def configuration(**updates):
    return KernelConfig(
        owner=owner(),
        bootstrap_owner=True,
        workers=(
            WorkerConfig(
                id="fixture", kind="fixture", context_token_limit=32768, max_output_tokens=128
            ),
        ),
        **updates,
    )


def open_kernel(path, worker, *, config=None, clock=None, registry=None, verifiers=None):
    return Kernel.open(
        path,
        config=config or configuration(),
        workers={"fixture": worker},
        capabilities=registry or CapabilityRegistry(),
        verifiers=verifiers or {},
        clock=clock or FakeClock(),
    )


def new_goal(**updates):
    from signal_gating.executive import Limits

    return goal(
        **{
            "limits": Limits(
                input_tokens=100000,
                output_tokens=10000,
                calls=20,
                elapsed_seconds=1000,
                cost_microunits=0,
            ),
            "criteria": (
                criterion(
                    verifier_id="artifact-integrity-v1", acceptance_rule="all_exist_and_intact"
                ),
            ),
            **updates,
        }
    )


def records(kernel, kind):
    return [r for r in kernel.inspect().state.records if isinstance(r, kind)]


def blocked(packet):
    return EpisodeResult(
        stop_reason="blocked",
        blocker=WakeCondition(kind="human_decision", reason_code="owner_needed"),
        usage=usage(),
    )


async def test_creation_duplicate_preserves_original_ack_and_single_budget(tmp_path):
    async with open_kernel(tmp_path, FunctionWorker(blocked)) as kernel:
        command = CreateGoal(goal=new_goal())
        receipt = await kernel.submit(command, principal=owner())
        sequence = kernel.inspect().state.sequence
        duplicate = await kernel.submit(command, principal=owner())
        assert receipt.disposition == "accept"
        assert duplicate == receipt
        assert kernel.inspect().state.sequence == sequence
        assert (
            len(records(kernel, Goal))
            == len(records(kernel, Budget))
            == len(records(kernel, TaskFrame))
            == 1
        )
        assert records(kernel, Goal)[0].budget_id == records(kernel, Budget)[0].id
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_empty_results_block_after_two_new_failures_without_idle_spin(tmp_path):
    worker = FunctionWorker(lambda p: EpisodeResult(stop_reason="proposed", usage=usage()))
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        summary = await asyncio.wait_for(kernel.run_until_idle(), 5)
        assert len(worker.packets) == 1
        kernel.clock.advance(60)
        summary = await kernel.run_until_idle()
        assert len(worker.packets) == 2
        assert summary.blocked_task_ids
        sequence = kernel.store.view().sequence
        await kernel.run_until_idle()
        assert kernel.store.view().sequence == sequence
        assert records(kernel, TaskFrame)[0].failure_count == 2


async def test_pause_ack_fences_cancellation_resistant_worker_and_drains_late_usage(tmp_path):
    worker = ControlledWorker()
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        running = asyncio.create_task(kernel.run_until_idle())
        await asyncio.wait_for(worker.started.wait(), 5)
        current = records(kernel, Goal)[0]
        receipt = await kernel.submit(
            PauseGoal(goal_id=current.id, expected_revision=current.revision), principal=owner()
        )
        assert receipt.disposition == "accept"
        assert records(kernel, Goal)[0].status == "paused"
        await asyncio.wait_for(worker.cancelled.wait(), 5)
        await asyncio.wait_for(running, 5)
        assert records(kernel, Reservation)[0].status == "uncertain"
        worker.release.set_result(
            EpisodeResult(stop_reason="proposed", usage=usage(cost_microunits=3))
        )
        await worker.done.wait()
        await kernel.runner.late_usage_ready.wait()
        await kernel.run_until_idle()
        reservation = records(kernel, Reservation)[0]
        assert (reservation.observed_usage or reservation.usage).cost_microunits == 3
        assert records(kernel, Goal)[0].status == "paused"
        assert all(e.status == "fenced" for e in records(kernel, Episode))
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_new_draft_publish_share_current_refs_and_verify(tmp_path):
    calls = 0

    def produce(packet):
        nonlocal calls
        calls += 1
        if calls == 1:
            artifacts = (Ref(id="draft:result", revision=1, kind="artifact"),)
            drafts = (ArtifactDraft(key="result", media_type="text/plain", text="new work"),)
        else:
            artifacts = tuple(
                Ref.model_validate(r["ref"])
                for r in __import__("json").loads(packet.rendered_input)["untrusted_evidence"][
                    "artifacts"
                ]
            )
            drafts = ()
        return EpisodeResult(
            stop_reason="proposed",
            artifact_drafts=drafts,
            usage=usage(),
            proposals=(
                CompletionClaim(
                    episode_id=packet.episode_id,
                    fencing_token=packet.fencing_token,
                    read_set=packet.mandatory_read_set,
                    criterion_ids=(packet.goal.criteria[0].id,),
                    artifacts=artifacts,
                ),
            ),
        )

    async with open_kernel(
        tmp_path, FunctionWorker(produce), config=configuration(verification_repair_rounds=0)
    ) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        await asyncio.wait_for(kernel.run_until_idle(), 5)
        draft = next(r for r in records(kernel, ArtifactRef) if r.media_type == "text/plain")
        assert draft.shared_with == ()
        assert records(kernel, Goal)[0].status == "active"
        await kernel.submit(
            ShareArtifact(
                artifact_id=draft.id,
                expected_revision=draft.revision,
                principal_ids=(
                    kernel.verifier_principal.id,
                    kernel.worker_principals["fixture"].id,
                ),
            ),
            principal=owner(),
        )
        await kernel.submit(Reverify(goal_id=records(kernel, Goal)[0].id), principal=owner())
        summary = await asyncio.wait_for(kernel.run_until_idle(), 5)
        assert summary.completed_goal_ids
        assert calls == 2
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_exhaustion_blocks_without_consuming_scheduler_selection(tmp_path):
    from signal_gating.executive import Limits

    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        g = new_goal().model_copy(update={"limits": Limits(calls=0, cost_microunits=0)})
        await kernel.submit(CreateGoal(goal=g), principal=owner())
        summary = await kernel.run_until_idle()
        assert summary.blocked_task_ids
        assert not worker.packets
        assert not records(kernel, Episode)
        assert not records(kernel, Reservation)
        assert not [r for r in kernel.store.view().records if r.kind == "scheduler"]


async def test_controlled_deadline_retains_hold_and_rejects_late_proposal(tmp_path):
    clock = FakeClock()
    worker = ControlledWorker()
    async with open_kernel(tmp_path, worker, clock=clock) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        running = asyncio.create_task(kernel.run_until_idle())
        await asyncio.wait_for(worker.started.wait(), 5)
        clock.advance(120)
        await asyncio.wait_for(running, 5)
        assert records(kernel, Episode)[0].status == "expired"
        assert records(kernel, Reservation)[0].status == "uncertain"
        worker.release.set_result(EpisodeResult(stop_reason="proposed", usage=usage()))
        await worker.done.wait()
        await kernel.runner.late_usage_ready.wait()
        await kernel.run_until_idle()
        assert records(kernel, Episode)[0].status == "expired"
        assert len(records(kernel, Episode)) == 1
        assert records(kernel, Reservation)[0].status == "settled"


async def test_concurrent_episodes_reserve_before_worker_and_pause_independently(tmp_path):
    from uuid import uuid4

    started = asyncio.Queue()
    releases = {}

    class Worker:
        async def run_episode(self, packet):
            assert records(kernel, Reservation)
            assert records(kernel, Episode)
            future = asyncio.get_running_loop().create_future()
            releases[packet.goal.id] = future
            await started.put(packet)
            await future
            return blocked(packet)

    async with open_kernel(tmp_path, Worker()) as kernel:
        goals = (new_goal(), new_goal(id=uuid4().hex, criteria=()))
        for g in goals:
            await kernel.submit(CreateGoal(goal=g), principal=owner())
        running = asyncio.create_task(kernel.run_until_idle())
        first = await asyncio.wait_for(started.get(), 5)
        second = await asyncio.wait_for(started.get(), 5)
        assert first.goal.id != second.goal.id
        assert len([e for e in records(kernel, Episode) if e.status == "running"]) == 2
        await kernel.submit(
            PauseGoal(goal_id=first.goal.id, expected_revision=first.goal.revision),
            principal=owner(),
        )
        releases[second.goal.id].set_result(None)
        await asyncio.wait_for(running, 5)
        assert next(g for g in records(kernel, Goal) if g.id == first.goal.id).status == "paused"
        assert next(g for g in records(kernel, Goal) if g.id == second.goal.id).status == "active"


async def test_eof_stops_owned_calls_without_waiting_for_reasoning_lease(tmp_path):
    from signal_gating.executive import AuthenticatedCommand

    worker = ControlledWorker()
    async with open_kernel(tmp_path, worker) as kernel:

        async def commands():
            yield AuthenticatedCommand(command=CreateGoal(goal=new_goal()), principal=owner())
            await worker.started.wait()

        serving = asyncio.create_task(kernel.serve(commands()))
        await asyncio.wait_for(worker.started.wait(), 5)
        try:
            summary = await asyncio.wait_for(asyncio.shield(serving), 1)
            assert summary.stop_reason == "eof"
            assert records(kernel, Reservation)[0].status == "uncertain"
        finally:
            await kernel.stop()
            worker.release.set_result(EpisodeResult(stop_reason="proposed", usage=usage()))
            await worker.done.wait()
            await serving


async def test_worker_draft_later_write_delayed_approval_check_and_shared_proof(tmp_path):
    import hashlib
    import sys

    from signal_gating.executive import (
        ActionIntent,
        ActionRecord,
        ArgvCheck,
        DecideAction,
        Scope,
        SetGrant,
        WorkspaceConfig,
    )
    from signal_gating.executive.authority import reference

    from .factories import grant

    workspace = tmp_path / "work"
    workspace.mkdir()
    clock = FakeClock()
    calls = 0
    text = "authored in a worker draft\n"

    def produce(packet):
        nonlocal calls
        calls += 1
        common = dict(
            episode_id=packet.episode_id,
            fencing_token=packet.fencing_token,
            read_set=packet.mandatory_read_set,
        )
        if calls == 1:
            return EpisodeResult(
                stop_reason="blocked",
                blocker=WakeCondition(kind="human_decision", reason_code="share_draft"),
                artifact_drafts=(ArtifactDraft(key="work", media_type="text/plain", text=text),),
                usage=usage(),
            )
        if calls == 2:
            selected = Ref.model_validate(
                __import__("json").loads(packet.rendered_input)["untrusted_evidence"]["artifacts"][
                    0
                ]["ref"]
            )
            intent = ActionIntent(
                **common,
                capability_id="write_text",
                arguments={
                    "path": "result.txt",
                    "expected_sha256": "missing",
                    "content_ref": selected.model_dump(mode="json"),
                },
                scope=Scope(
                    owner_id=packet.goal.owner_id,
                    goal_id=packet.goal.id,
                    resource_paths=("result.txt",),
                ),
            )
        elif calls == 3:
            intent = ActionIntent(
                **common,
                capability_id="run_check",
                arguments={
                    "command_id": "check",
                    "criterion_ref": reference(packet.goal.criteria[0]).model_dump(mode="json"),
                    "expected_sha256": {"result.txt": hashlib.sha256(text.encode()).hexdigest()},
                },
                scope=Scope(
                    owner_id=packet.goal.owner_id, goal_id=packet.goal.id, resource_paths=(".",)
                ),
            )
        else:
            intent = CompletionClaim(**common, criterion_ids=(packet.goal.criteria[0].id,))
        return EpisodeResult(stop_reason="proposed", proposals=(intent,), usage=usage())

    config = configuration(
        workspace=WorkspaceConfig(
            root=workspace,
            writable_paths=(".",),
            named_checks=(
                ArgvCheck(
                    id="check",
                    argv=(
                        sys.executable,
                        "-c",
                        "from pathlib import Path; "
                        "assert Path('result.txt').read_text() == 'authored in a worker draft\\n'",
                    ),
                    maximum_charge_microunits=0,
                ),
            ),
        ),
        verifier_ids=("local-check-v1",),
        verification_repair_rounds=0,
    )
    g = new_goal(
        criteria=(
            criterion(
                verifier_id="local-check-v1",
                acceptance_rule="exit_zero",
                check_id="check",
                workspace_inputs=("result.txt",),
            ),
        )
    )
    async with open_kernel(
        tmp_path / "store", FunctionWorker(produce), clock=clock, config=config
    ) as kernel:
        await kernel.submit(CreateGoal(goal=g), principal=owner())
        await kernel.submit(
            SetGrant(
                grant=grant(
                    subject_id=kernel.worker_principals["fixture"].id,
                    capabilities=("write_text", "run_check"),
                    scope=Scope(owner_id=owner().id, goal_id=g.id, resource_paths=(".",)),
                )
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        draft = next(r for r in records(kernel, ArtifactRef) if r.media_type == "text/plain")
        await kernel.submit(
            ShareArtifact(
                artifact_id=draft.id,
                expected_revision=draft.revision,
                principal_ids=(
                    kernel.worker_principals["fixture"].id,
                    kernel.verifier_principal.id,
                ),
            ),
            principal=owner(),
        )
        await kernel.submit(Reverify(goal_id=g.id), principal=owner())
        summary = await kernel.run_until_idle()
        assert len(summary.pending_review_ids) == 1, ";".join(
            str(getattr(r, "reason_code", getattr(r, "status", "")))
            for r in kernel.store.view().records
            if r.kind in {"action", "receipt"}
        )
        assert calls == 2
        write = records(kernel, ActionRecord)[0]
        assert write.status == "awaiting_review"
        assert not (workspace / "result.txt").exists()
        clock.advance(121)
        receipt = await kernel.submit(
            DecideAction(
                action_id=write.id,
                expected_revision=write.revision,
                decision="approve",
                arguments_digest=write.arguments_digest,
                scope=write.scope,
                constraint_revision=write.constraint_revision,
            ),
            principal=owner(),
        )
        assert receipt.disposition == "accept"
        summary = await kernel.run_until_idle()
        assert (workspace / "result.txt").read_text() == text
        assert len(summary.pending_review_ids) == 1
        check = next(a for a in records(kernel, ActionRecord) if a.capability_id == "run_check")
        await kernel.submit(
            DecideAction(
                action_id=check.id,
                expected_revision=check.revision,
                decision="approve",
                arguments_digest=check.arguments_digest,
                scope=check.scope,
                constraint_revision=check.constraint_revision,
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        assert (
            next(a for a in records(kernel, ActionRecord) if a.id == check.id).status == "succeeded"
        )
        assert records(kernel, Goal)[0].status != "completed"
        for artifact in records(kernel, ArtifactRef):
            if (
                artifact.media_type != "application/vnd.sgp.executive-context+json"
                and kernel.verifier_principal.id not in artifact.shared_with
            ):
                await kernel.submit(
                    ShareArtifact(
                        artifact_id=artifact.id,
                        expected_revision=artifact.revision,
                        principal_ids=(kernel.verifier_principal.id,),
                    ),
                    principal=owner(),
                )
        await kernel.submit(Reverify(goal_id=g.id), principal=owner())
        summary = await kernel.run_until_idle()
        assert summary.completed_goal_ids == (g.id,), ";".join(
            str(getattr(r, "reason_code", getattr(r, "status", "")))
            for r in kernel.store.view().records
            if r.kind in {"verification", "receipt", "task"}
        )
        assert calls == 5
        assert kernel.store.view() == rebuild(kernel.store.events())
        bindings = dict(kernel.registry.specs)
    async with open_kernel(
        tmp_path / "store", FunctionWorker(produce), clock=clock, config=config
    ) as restarted:
        assert dict(restarted.registry.specs) == bindings
        assert (await restarted.run_until_idle()).completed_goal_ids == (g.id,), ";".join(
            str(getattr(r, "reason_code", getattr(r, "status", "")))
            for r in restarted.store.view().records
            if r.kind in {"verification", "receipt", "task"}
        )

        from signal_gating.executive import Commitment

        from .test_verification import change

        commitment = records(restarted, Commitment)[0]
        change(restarted.store, commitment, deadline=NOW + timedelta(days=1))
        assert not (await restarted.run_until_idle()).completed_goal_ids


@pytest.mark.parametrize("reason", ["exhausted", "needs_evidence"])
async def test_progression_waits_instead_of_repeating_episode(tmp_path, reason):
    from signal_gating.executive import EvidenceRequest

    def produce(packet):
        proposals = (
            ()
            if reason == "exhausted"
            else (
                EvidenceRequest(
                    episode_id=packet.episode_id,
                    fencing_token=packet.fencing_token,
                    read_set=packet.mandatory_read_set,
                    query="Owner evidence needed",
                ),
            )
        )
        return EpisodeResult(stop_reason=reason, proposals=proposals, usage=usage())

    worker = FunctionWorker(produce)
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        await kernel.run_until_idle()
        assert len(worker.packets) == 1
        condition = records(kernel, TaskFrame)[0].wake_condition
        assert condition.kind == ("budget" if reason == "exhausted" else "new_evidence")
        sequence = kernel.store.view().sequence
        await kernel.run_until_idle()
        assert kernel.store.view().sequence == sequence


async def test_live_correction_fences_old_call_and_resumes_current_goal(tmp_path):
    first = ControlledWorker()
    second = asyncio.Event()
    packets = []

    class Worker:
        async def run_episode(self, packet):
            packets.append(packet)
            if len(packets) == 1:
                return await first.run_episode(packet)
            second.set()
            return blocked(packet)

    async with open_kernel(tmp_path, Worker()) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        running = asyncio.create_task(kernel.run_until_idle())
        await first.started.wait()
        current = records(kernel, Goal)[0]
        await kernel.submit(
            ReviseGoal(
                goal_id=current.id,
                expected_revision=current.revision,
                outcome="Corrected owner outcome",
            ),
            principal=owner(),
        )
        await asyncio.wait_for(second.wait(), 5)
        await running
        assert packets[1].goal.outcome == "Corrected owner outcome"
        assert packets[1].fencing_token > packets[0].fencing_token
        first.release.set_result(EpisodeResult(stop_reason="proposed", usage=usage()))
        await first.done.wait()
        await kernel.runner.late_usage_ready.wait()
        await kernel.run_until_idle()
        assert records(kernel, Episode)[0].status in {"accepted", "fenced"}
        assert len(packets) == 2


async def test_real_resolve_usage_and_late_measured_overage_preserved(tmp_path):
    from signal_gating.executive import ResolveUsage

    worker = ControlledWorker()
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        running = asyncio.create_task(kernel.run_until_idle())
        await worker.started.wait()
        current = records(kernel, Goal)[0]
        await kernel.submit(
            PauseGoal(goal_id=current.id, expected_revision=current.revision), principal=owner()
        )
        await running
        reservation = records(kernel, Reservation)[0]
        ack = await kernel.submit(
            ResolveUsage(
                reservation_id=reservation.id,
                expected_revision=reservation.revision,
                accept_reserved_maximum=True,
                explanation="Accept configured maximum",
            ),
            principal=owner(),
        )
        assert ack.disposition == "accept"
        assert records(kernel, Reservation)[0].status == "settled"
        worker.release.set_result(
            EpisodeResult(stop_reason="proposed", usage=usage(cost_microunits=5))
        )
        await worker.done.wait()
        await kernel.runner.late_usage_ready.wait()
        await kernel.run_until_idle()
        reservation = records(kernel, Reservation)[0]
        assert reservation.accounting_conflict
        assert reservation.observed_usage.cost_microunits == 5
        assert records(kernel, Budget)[0].accounting_uncertain
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_owner_criterion_approval_triggers_verification_without_extra_command(tmp_path):
    from signal_gating.executive import ApproveCriterion
    from signal_gating.executive.authority import reference
    from signal_gating.executive.verification import VerificationResult

    g = new_goal(
        criteria=(
            criterion(
                evaluator_type="human",
                verifier_id="owner-approval-v1",
                acceptance_rule="explicit_owner_approval",
            ),
        )
    )
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=g), principal=owner())
        artifact = kernel.artifacts.put(
            b"owner selected", principal=owner(), goal_id=g.id, media_type="text/plain"
        )
        await kernel.submit(
            ShareArtifact(artifact_id=artifact.id, principal_ids=(kernel.verifier_principal.id,)),
            principal=owner(),
        )
        artifact = next(r for r in records(kernel, ArtifactRef) if r.id == artifact.id)
        current = records(kernel, Goal)[0]
        await kernel.submit(
            ApproveCriterion(
                goal_id=g.id,
                expected_revision=current.revision,
                criterion=reference(g.criteria[0]),
                artifacts=(reference(artifact),),
                explanation="Reviewed exact content",
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        assert records(kernel, VerificationResult)[0].disposition == "accept"


async def test_injected_local_verifier_is_bounded_and_fenced_on_stop(tmp_path):
    from signal_gating.executive.verification import ArtifactVerifier

    started, cancelled, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release = asyncio.get_running_loop().create_future()

    class Verifier:
        async def verify(self, selected, state):
            started.set()
            try:
                try:
                    await asyncio.shield(release)
                except asyncio.CancelledError:
                    cancelled.set()
                    await release
                return ArtifactVerifier(
                    kernel.artifacts, principal=kernel.verifier_principal
                )._result(selected, state, "accept", "trusted-local-check")
            finally:
                finished.set()

    def claim(packet):
        return EpisodeResult(
            stop_reason="proposed",
            usage=usage(),
            proposals=(
                CompletionClaim(
                    episode_id=packet.episode_id,
                    fencing_token=packet.fencing_token,
                    read_set=packet.mandatory_read_set,
                    criterion_ids=(packet.goal.criteria[0].id,),
                ),
            ),
        )

    async with open_kernel(
        tmp_path, FunctionWorker(claim), verifiers={"trusted-local-v1": Verifier()}
    ) as kernel:
        await kernel.submit(
            CreateGoal(goal=new_goal(criteria=(criterion(verifier_id="trusted-local-v1"),))),
            principal=owner(),
        )
        running = asyncio.create_task(kernel.run_until_idle())
        await started.wait()
        try:
            await kernel.stop()
            await asyncio.wait_for(asyncio.shield(running), 1)
            assert not records(kernel, Goal)[0].status == "completed"
        finally:
            release.set_result(None)
            await finished.wait()
            await running


async def test_serve_waits_for_fake_timer_then_resumes_without_polling(tmp_path):
    from signal_gating.executive import AuthenticatedCommand

    clock = FakeClock()
    armed = asyncio.Queue()
    original_wait = clock.wait_until

    async def timer(at):
        await armed.put(at)
        await original_wait(at)

    clock.wait_until = timer
    second, eof = asyncio.Event(), asyncio.Event()
    count = 0

    def produce(packet):
        nonlocal count
        count += 1
        if count == 1:
            return EpisodeResult(
                stop_reason="blocked",
                blocker=WakeCondition(
                    kind="time", at=NOW + timedelta(seconds=5), reason_code="scheduled"
                ),
                usage=usage(),
            )
        second.set()
        return blocked(packet)

    async with open_kernel(tmp_path, FunctionWorker(produce), clock=clock) as kernel:

        async def commands():
            yield AuthenticatedCommand(command=CreateGoal(goal=new_goal()), principal=owner())
            await eof.wait()

        serving = asyncio.create_task(kernel.serve(commands()))
        while await armed.get() != NOW + timedelta(seconds=5):
            pass
        sequence = kernel.store.view().sequence
        assert count == 1
        assert kernel.store.view().sequence == sequence
        clock.advance(5)
        await asyncio.wait_for(second.wait(), 5)
        eof.set()
        assert (await serving).stop_reason == "eof"
        assert count == 2


async def test_owner_resume_reactivates_only_eligible_work(tmp_path):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        current = records(kernel, Goal)[0]
        await kernel.submit(
            PauseGoal(goal_id=current.id, expected_revision=current.revision), principal=owner()
        )
        await kernel.run_until_idle()
        assert not worker.packets
        current = records(kernel, Goal)[0]
        await kernel.submit(
            ResumeGoal(goal_id=current.id, expected_revision=current.revision), principal=owner()
        )
        await kernel.run_until_idle()
        assert len(worker.packets) == 1


async def test_unknown_effect_requires_human_and_never_blindly_retries(tmp_path):
    from signal_gating.executive import (
        ActionIntent,
        ActionObservation,
        ActionRecord,
        CapabilitySpec,
        Limits,
        Scope,
        SetGrant,
    )

    from .factories import grant, signal

    registry = CapabilityRegistry()

    class Adapter:
        calls = 0

        async def execute(self, action):
            self.calls += 1
            return ActionObservation(
                action_id=action.id,
                expected_revision=action.revision,
                outcome="outcome_unknown",
                usage=Usage(calls=1),
            )

        async def reconcile(self, action):
            raise AssertionError("No trusted lookup exists")

        async def cancel(self, action):
            raise AssertionError("Cancellation does not prove absence")

    adapter = Adapter()
    registry.register(
        CapabilitySpec(
            id="effect",
            binding_version="1",
            description="Controlled uncertain effect",
            input_schema={"type": "object"},
            effect_class="consequential",
        ),
        adapter,
        resource_scope=lambda _: Scope(resource_paths=(".",)),
        quote=Limits(
            input_tokens=0, output_tokens=0, calls=1, elapsed_seconds=30, cost_microunits=0
        ),
        quote_source="trusted-fixture",
        quote_version="1",
        require_approval=False,
    )

    def produce(packet):
        return EpisodeResult(
            stop_reason="proposed",
            usage=usage(),
            proposals=(
                ActionIntent(
                    episode_id=packet.episode_id,
                    fencing_token=packet.fencing_token,
                    read_set=packet.mandatory_read_set,
                    capability_id="effect",
                    arguments={},
                    scope=Scope(
                        owner_id=packet.goal.owner_id, goal_id=packet.goal.id, resource_paths=(".",)
                    ),
                ),
            ),
        )

    async with open_kernel(tmp_path, FunctionWorker(produce), registry=registry) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        await kernel.submit(
            SetGrant(
                grant=grant(
                    subject_id=kernel.worker_principals["fixture"].id,
                    capabilities=("effect",),
                    scope=Scope(owner_id=owner().id, goal_id=new_goal().id, resource_paths=(".",)),
                )
            ),
            principal=owner(),
        )
        summary = await asyncio.wait_for(kernel.run_until_idle(), 5)
        assert adapter.calls == 1
        assert len(summary.unknown_action_ids) == 1
        assert records(kernel, ActionRecord)[0].status == "needs_human"
        assert any(r.status == "uncertain" for r in records(kernel, Reservation))
        assert (await kernel.ingest(signal(), principal=owner())).disposition == "admit"
        await kernel.run_until_idle()
        assert adapter.calls == 1
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_bounded_signal_queue_backpressure_is_not_durable_ack(tmp_path):
    from uuid import uuid4

    from .factories import signal

    async with open_kernel(
        tmp_path, FunctionWorker(blocked), config=configuration(max_queued_ordinary_items=1)
    ) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        first = signal()
        second = signal(id=uuid4().hex)
        assert (await kernel.ingest(first, principal=owner())).disposition == "admit"
        sequence = kernel.store.view().sequence
        rejected = await kernel.ingest(second, principal=owner())
        assert rejected.reason_code == "ordinary_queue_full"
        assert kernel.store.view().sequence == sequence
        await kernel.run_until_idle()
        assert (await kernel.ingest(second, principal=owner())).disposition == "admit"


async def test_conflicting_bundle_rejects_draft_and_releases_stage_pins(tmp_path):
    from signal_gating.executive import EvidenceRequest

    def conflicting(packet):
        common = dict(
            episode_id=packet.episode_id,
            fencing_token=packet.fencing_token,
            read_set=packet.mandatory_read_set,
        )
        return EpisodeResult(
            stop_reason="proposed",
            usage=usage(),
            artifact_drafts=(
                ArtifactDraft(key="rejected", media_type="text/plain", text="must not publish"),
            ),
            proposals=(
                CompletionClaim(**common, criterion_ids=(packet.goal.criteria[0].id,)),
                EvidenceRequest(**common, query="Conflicting wait"),
            ),
        )

    async with open_kernel(tmp_path, FunctionWorker(conflicting)) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        await kernel.run_until_idle()
        assert records(kernel, TaskFrame)[0].failure_reason == "conflicting_lifecycle"
        assert not [a for a in records(kernel, ArtifactRef) if a.media_type == "text/plain"]
        assert not kernel.artifacts._staged
        assert kernel.artifacts.collect_orphans() == 1


async def test_configured_provider_quote_bounds_deadline_before_fake_call(tmp_path):
    import json

    from signal_gating.executive import Limits, UTF8ByteCounter
    from signal_gating.executive.adapters.openai_worker import OpenAIWorker

    from .test_openai_worker import FakeClient

    class Client(FakeClient):
        async def create(self, **kwargs):
            episode = records(kernel, Episode)[0]
            reservation = records(kernel, Reservation)[0]
            assert episode.deadline == NOW + timedelta(seconds=30)
            assert reservation.subject_id == episode.id
            assert reservation.status == "reserved"
            assert not kernel.store._connection.in_transaction
            return await super().create(**kwargs)

    client = Client(
        json.dumps(
            {
                "stop_reason": "blocked",
                "blocker": {"kind": "human_decision", "reason_code": "owner_needed"},
            }
        )
    )
    worker = OpenAIWorker(
        client,
        model="fixture-model",
        counter=UTF8ByteCounter(),
        capacity=32768,
        quote=Limits(
            input_tokens=32768, output_tokens=2048, calls=1, elapsed_seconds=30, cost_microunits=3
        ),
        quote_source="test-operator",
        quote_version="1",
    )
    config = KernelConfig(
        owner=owner(),
        bootstrap_owner=True,
        workers=(
            WorkerConfig(
                id="fixture",
                kind="openai",
                model="fixture-model",
                api_key_env="UNUSED_TEST_KEY",
                context_token_limit=32768,
            ),
        ),
    )
    async with open_kernel(tmp_path, worker, config=config) as kernel:
        await kernel.submit(
            CreateGoal(
                goal=new_goal(
                    limits=Limits(
                        input_tokens=100000,
                        output_tokens=10000,
                        calls=5,
                        elapsed_seconds=1000,
                        cost_microunits=10,
                    )
                )
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        assert client.calls == 1
        assert records(kernel, Reservation)[0].status == "uncertain"
        assert records(kernel, Budget)[0].reserved.cost_microunits == 3
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_owner_budget_revision_updates_the_enforced_ledger_atomically(tmp_path):
    from signal_gating.executive import Limits

    async with open_kernel(tmp_path, FunctionWorker(blocked)) as kernel:
        await kernel.submit(
            CreateGoal(goal=new_goal(limits=Limits(calls=0, cost_microunits=0))), principal=owner()
        )
        await kernel.run_until_idle()
        current = records(kernel, Goal)[0]
        limits = new_goal().limits
        ack = await kernel.submit(
            ReviseGoal(goal_id=current.id, expected_revision=current.revision, limits=limits),
            principal=owner(),
        )
        assert ack.disposition == "accept"
        assert records(kernel, Budget)[0].limits == limits
        await kernel.run_until_idle()
        assert len(records(kernel, Episode)) == 1
        current = records(kernel, Goal)[0]
        await kernel.submit(
            ReviseGoal(
                goal_id=current.id,
                expected_revision=current.revision,
                limits=Limits(calls=0, cost_microunits=0),
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        assert len(records(kernel, Episode)) == 1
        assert records(kernel, Budget)[0].actual.calls == 1
        assert kernel.store.view() == rebuild(kernel.store.events())


def active_wait_barrier(kernel, active):
    """Observe a settled active wait, after admission notifications were drained."""
    waiting = asyncio.Event()
    original_wait = kernel._wait_change

    async def observe():
        if active.is_set() and not kernel._changed.is_set():
            waiting.set()
        await original_wait()

    kernel._wait_change = observe
    return waiting


@pytest.mark.parametrize("boundary", ["task_time", "belief_valid_from"])
async def test_timer_wakes_eligible_task_while_independent_worker_active(tmp_path, boundary):
    from uuid import uuid4

    from signal_gating.executive import Belief, Scope
    from signal_gating.executive.authority import reference

    clock = FakeClock()
    due = NOW + timedelta(seconds=5)
    delayed_goal = new_goal(criteria=())
    independent_goal = new_goal(id=uuid4().hex, criteria=())
    independent_started, delayed_started = asyncio.Event(), asyncio.Event()
    release = asyncio.get_running_loop().create_future()
    delayed_calls = 0

    class Worker:
        async def run_episode(self, packet):
            nonlocal delayed_calls
            if packet.goal.id == independent_goal.id:
                independent_started.set()
                await release
                return blocked(packet)
            delayed_calls += 1
            if boundary == "task_time" and delayed_calls == 1:
                return EpisodeResult(
                    stop_reason="blocked",
                    blocker=WakeCondition(kind="time", at=due, reason_code="scheduled"),
                    usage=usage(),
                )
            delayed_started.set()
            return blocked(packet)

    async with open_kernel(tmp_path, Worker(), clock=clock) as kernel:
        await kernel.submit(CreateGoal(goal=delayed_goal), principal=owner())
        if boundary == "belief_valid_from":
            task = records(kernel, TaskFrame)[0]
            belief = Belief(
                claim="Future supported prerequisite",
                scope=Scope(owner_id=owner().id, goal_id=delayed_goal.id),
                epistemic_status="supported",
                valid_from=due,
            )
            kernel.store.commit(
                kernel._batch(
                    (
                        belief,
                        task.model_copy(
                            update={
                                "revision": task.revision + 1,
                                "dependencies": (reference(belief),),
                            }
                        ),
                    ),
                    kernel.store.view(),
                )
            )
        await kernel.run_until_idle()
        assert not delayed_started.is_set()
        await kernel.submit(CreateGoal(goal=independent_goal), principal=owner())
        waiting = active_wait_barrier(kernel, independent_started)
        running = asyncio.create_task(kernel.run_until_idle())
        await asyncio.wait_for(independent_started.wait(), 1)
        await asyncio.wait_for(waiting.wait(), 1)
        try:
            clock.advance(5)
            await asyncio.wait_for(delayed_started.wait(), 1)
            independent = next(
                t for t in records(kernel, TaskFrame) if t.goal_id == independent_goal.id
            )
            assert independent.status == "running"
            assert not release.done()
        finally:
            release.set_result(None)
            await asyncio.wait_for(running, 1)
        assert not clock.waiters
        assert kernel.store.view() == rebuild(kernel.store.events())


async def test_timer_reverifies_expired_completion_while_independent_worker_active(tmp_path):
    import json
    from uuid import uuid4

    from signal_gating.executive import Belief, Commitment, Scope
    from signal_gating.executive.authority import reference

    clock = FakeClock()
    due = NOW + timedelta(seconds=5)
    completed_goal = new_goal()
    independent_goal = new_goal(id=uuid4().hex, criteria=())
    independent_started, rechecked = asyncio.Event(), asyncio.Event()
    release = asyncio.get_running_loop().create_future()

    class Worker:
        async def run_episode(self, packet):
            if packet.goal.id == independent_goal.id:
                independent_started.set()
                await release
                return blocked(packet)
            evidence = tuple(
                Ref.model_validate(a["ref"])
                for a in json.loads(packet.rendered_input)["untrusted_evidence"]["artifacts"]
            )
            return EpisodeResult(
                stop_reason="proposed",
                usage=usage(),
                proposals=(
                    CompletionClaim(
                        episode_id=packet.episode_id,
                        fencing_token=packet.fencing_token,
                        read_set=packet.mandatory_read_set,
                        criterion_ids=(packet.goal.criteria[0].id,),
                        artifacts=evidence,
                    ),
                ),
            )

    async with open_kernel(tmp_path, Worker(), clock=clock) as kernel:
        await kernel.submit(CreateGoal(goal=completed_goal), principal=owner())
        belief = Belief(
            claim="Expiring required-work support",
            scope=Scope(owner_id=owner().id, goal_id=completed_goal.id),
            epistemic_status="supported",
            valid_until=due,
        )
        commitment = records(kernel, Commitment)[0]
        kernel.store.commit(
            kernel._batch(
                (
                    belief,
                    commitment.model_copy(
                        update={
                            "revision": commitment.revision + 1,
                            "dependencies": (reference(belief),),
                        }
                    ),
                ),
                kernel.store.view(),
            )
        )
        artifact = kernel.artifacts.put(
            b"actual completion evidence",
            principal=owner(),
            goal_id=completed_goal.id,
            media_type="text/plain",
        )
        await kernel.submit(
            ShareArtifact(
                artifact_id=artifact.id,
                principal_ids=(
                    kernel.worker_principals["fixture"].id,
                    kernel.verifier_principal.id,
                ),
            ),
            principal=owner(),
        )
        assert (await kernel.run_until_idle()).completed_goal_ids == (completed_goal.id,)
        original_call = kernel._verification_call

        async def observed_call(subject_id, call):
            result = await original_call(subject_id, call)
            if clock.now() >= due:
                rechecked.set()
            return result

        kernel._verification_call = observed_call
        await kernel.submit(CreateGoal(goal=independent_goal), principal=owner())
        waiting = active_wait_barrier(kernel, independent_started)
        running = asyncio.create_task(kernel.run_until_idle())
        await asyncio.wait_for(independent_started.wait(), 1)
        await asyncio.wait_for(waiting.wait(), 1)
        try:
            clock.advance(5)
            await asyncio.wait_for(rechecked.wait(), 1)
            current = next(g for g in records(kernel, Goal) if g.id == completed_goal.id)
            assert current.status == "active"
            dependent = next(
                t for t in records(kernel, TaskFrame) if t.goal_id == completed_goal.id
            )
            assert dependent.status == "blocked"
            assert dependent.wake_condition.reason_code == "completion_prerequisites_unsettled"
            independent = next(
                t for t in records(kernel, TaskFrame) if t.goal_id == independent_goal.id
            )
            assert independent.status == "running"
            assert not release.done()
        finally:
            release.set_result(None)
            await asyncio.wait_for(running, 1)
        assert not clock.waiters
        assert kernel.store.view() == rebuild(kernel.store.events())
