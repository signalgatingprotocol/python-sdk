"""Cross-component regressions for the final executive integration review."""

from datetime import timedelta
from uuid import uuid4

import pytest

from signal_gating.executive import (
    ArtifactDraft,
    Budget,
    CommitBatch,
    Commitment,
    CreateGoal,
    DecisionReceipt,
    EpisodeResult,
    EvidenceRequest,
    ExecutiveSignal,
    Grant,
    JournalEvent,
    Ref,
    Reservation,
    Scope,
    SetGrant,
    ShareArtifact,
    TaskDecomposition,
    TaskFrame,
)
from signal_gating.executive.reducers import rebuild

from .factories import NOW, owner
from .test_kernel import FunctionWorker, blocked, new_goal, open_kernel, records, usage


def configure(kernel, *writes):
    state = kernel.store.view()
    kernel.store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=state.sequence + 1,
                    event_type="state.changed",
                    producer_id="trusted-test-configuration",
                    principal_id=owner().id,
                    subject=Ref(id=writes[0].id),
                    payload={"records": [r.model_dump(mode="json") for r in writes]},
                ),
            ),
            expected_revisions=tuple(Ref(id=r.id, revision=r.revision) for r in state.records),
        )
    )


def commitment_budget(kernel, calls):
    parent = records(kernel, Budget)[0]
    commitment = records(kernel, Commitment)[0]
    child = parent.model_copy(
        update={
            "id": uuid4().hex,
            "parent_id": parent.id,
            "limits": parent.limits.model_copy(update={"calls": calls}),
        }
    )
    configure(
        kernel,
        child,
        commitment.model_copy(update={"revision": commitment.revision + 1, "budget_id": child.id}),
    )
    return child


@pytest.mark.parametrize("calls", [0, 1])
async def test_episode_enforces_zero_and_exhausted_commitment_budget(tmp_path, calls):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        child = commitment_budget(kernel, calls)
        await kernel.run_until_idle()
        assert len(worker.packets) == calls
        if calls:
            task = records(kernel, TaskFrame)[0]
            configure(
                kernel,
                task.model_copy(
                    update={
                        "revision": task.revision + 1,
                        "status": "ready",
                        "wake_condition": None,
                    }
                ),
            )
            await kernel.run_until_idle()
            assert len(worker.packets) == 1
            assert worker.packets[0].remaining_limits.calls == 1
        assert all(r.budget_id == child.id for r in records(kernel, Reservation))
        assert all(b.actual.calls == calls for b in records(kernel, Budget))
        assert records(kernel, TaskFrame)[0].wake_condition.reason_code == "budget_exhausted"


async def test_decomposition_inherits_episode_budget_and_shared_ancestor_accounting(tmp_path):
    def produce(packet):
        if worker.packets.index(packet):
            return blocked(packet)
        commitment = Commitment(
            goal_id=packet.goal.id,
            responsible_principal_id=packet.commitments[0].responsible_principal_id,
        )
        task = TaskFrame(goal_id=packet.goal.id, commitment_id=commitment.id, subgoal="child")
        return EpisodeResult(
            stop_reason="blocked",
            blocker=blocked(packet).blocker,
            proposals=(
                TaskDecomposition(
                    episode_id=packet.episode_id,
                    fencing_token=packet.fencing_token,
                    read_set=packet.mandatory_read_set,
                    commitments=(commitment,),
                    tasks=(task,),
                ),
            ),
            usage=usage(),
        )

    worker = FunctionWorker(produce)
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        child = commitment_budget(kernel, 2)
        await kernel.run_until_idle()
        assert len(worker.packets) == 2
        assert len(records(kernel, Commitment)) == 2
        assert all(c.budget_id == child.id for c in records(kernel, Commitment))
        assert all(r.budget_id == child.id for r in records(kernel, Reservation))
        assert all(b.actual.calls == 2 for b in records(kernel, Budget))


@pytest.mark.parametrize("kind", ["revoked", "expired", "other_principal"])
async def test_fresh_episode_ignores_unusable_historical_grants(tmp_path, kind):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        g = new_goal()
        await kernel.submit(CreateGoal(goal=g), principal=owner())
        grant = Grant(
            issuer_id=owner().id,
            subject_id=kernel.verifier_principal.id
            if kind == "other_principal"
            else kernel.worker_principals["fixture"].id,
            capabilities=("unused",),
            scope=Scope(owner_id=owner().id, goal_id=g.id),
            revoked=kind != "expired",
            expires_at=NOW - timedelta(seconds=1) if kind == "expired" else None,
        )
        await kernel.submit(SetGrant(grant=grant), principal=owner())
        await kernel.run_until_idle()
        assert len(worker.packets) == 1
        assert grant.id not in {r.id for r in worker.packets[0].mandatory_read_set}


@pytest.mark.parametrize("change", ["revocation", "expiry"])
async def test_inflight_grant_revocation_still_fences_episode(tmp_path, change):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        g = new_goal()
        await kernel.submit(CreateGoal(goal=g), principal=owner())
        grant = Grant(
            issuer_id=owner().id,
            subject_id=kernel.worker_principals["fixture"].id,
            capabilities=("unused",),
            scope=Scope(owner_id=owner().id, goal_id=g.id),
            expires_at=NOW + timedelta(seconds=5),
        )
        await kernel.submit(SetGrant(grant=grant), principal=owner())
        kernel._observe_events()
        ((episode, packet),) = kernel._admit()
        assert kernel._is_current(episode, packet)
        if change == "revocation":
            configure(kernel, grant.model_copy(update={"revision": 2, "revoked": True}))
        else:
            kernel.clock.advance(6)
        assert not kernel._is_current(episode, packet)
        await kernel._episode(episode, packet)
        assert not worker.packets


@pytest.mark.parametrize("aggregate", [False, True])
async def test_local_draft_rejection_preserves_known_usage_and_releases_pins(tmp_path, aggregate):
    drafts = (ArtifactDraft(key="a", text="a" * (5 if aggregate else 9), media_type="text/plain"),)
    if aggregate:
        drafts += (ArtifactDraft(key="b", text="b" * 5, media_type="text/plain"),)
    worker = FunctionWorker(lambda p: blocked(p).model_copy(update={"artifact_drafts": drafts}))
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        # Context is already staged during admission; these bounds apply to result drafts.
        original = kernel.artifacts.stage_drafts

        def stage(*args, **kwargs):
            kernel.artifacts.max_artifact_bytes = 8 if args[0] == drafts else 8388608
            kernel.artifacts.max_draft_bytes = 8 if args[0] == drafts else 8388608
            return original(*args, **kwargs)

        kernel.artifacts.stage_drafts = stage
        await kernel.run_until_idle()
        (reservation,) = records(kernel, Reservation)
        assert reservation.usage == usage()
        assert reservation.accounting_conflict is None
        assert reservation.observed_usage is None
        assert reservation.revision == 2
        assert reservation.status == "settled"
        assert not kernel.artifacts._staged
        assert records(kernel, TaskFrame)[0].failure_reason == (
            "drafts_too_large" if aggregate else "artifact_too_large"
        )
        assert all(b.actual.calls == 1 and b.reserved.calls == 0 for b in records(kernel, Budget))


async def test_evidence_request_current_revision_task_lifecycle_and_restart(tmp_path):
    def produce(packet):
        if len(worker.packets) > 1:
            return blocked(packet)
        ref = next(r for r in packet.mandatory_read_set if r.id == artifact.id)
        return EpisodeResult(
            stop_reason="needs_evidence",
            proposals=(
                EvidenceRequest(
                    episode_id=packet.episode_id,
                    fencing_token=packet.fencing_token,
                    read_set=packet.mandatory_read_set,
                    refs=(ref,),
                    query="Need evidence",
                ),
            ),
            usage=usage(),
        )

    worker = FunctionWorker(produce)
    async with open_kernel(tmp_path, worker) as kernel:
        g = new_goal()
        await kernel.submit(CreateGoal(goal=g), principal=owner())
        artifact = kernel.artifacts.put(
            b"evidence", principal=owner(), goal_id=g.id, media_type="text/plain"
        )
        await kernel.submit(
            ShareArtifact(
                artifact_id=artifact.id,
                expected_revision=1,
                principal_ids=(kernel.worker_principals["fixture"].id,),
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        (request,) = [
            r for r in records(kernel, ExecutiveSignal) if r.signal_kind == "evidence_request"
        ]
        origin = records(kernel, TaskFrame)[0]
        # An unrelated task must not inherit the historical request.
        unrelated = origin.model_copy(
            update={"id": uuid4().hex, "revision": 1, "status": "ready", "wake_condition": None}
        )
        configure(kernel, unrelated)
        await kernel.submit(
            ShareArtifact(
                artifact_id=artifact.id,
                expected_revision=2,
                principal_ids=(kernel.verifier_principal.id,),
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        assert len(worker.packets) == 2
        assert worker.packets[-1].task.id == unrelated.id
        assert not [r for r in records(kernel, DecisionReceipt) if r.subject_id == request.id]
        await kernel.ingest(
            ExecutiveSignal(
                signal_kind="evidence",
                principal_id=owner().id,
                producer_id="owner",
                scope=Scope(owner_id=owner().id, goal_id=g.id),
                artifacts=(Ref(id=artifact.id, revision=3),),
            ),
            principal=owner(),
        )
        await kernel.run_until_idle()
        assert len(worker.packets) == 3
        assert worker.packets[-1].task.id == origin.id
        assert Ref(id=artifact.id, revision=3) in worker.packets[-1].mandatory_read_set
        assert any(
            r.subject_id == request.id and r.reason_code == "evidence_request_superseded"
            for r in records(kernel, DecisionReceipt)
        )
        assert next(r for r in records(kernel, ExecutiveSignal) if r.id == request.id) == request
        assert kernel.store.view() == rebuild(kernel.store.events())
    async with open_kernel(tmp_path, worker) as kernel:
        current = next(r for r in records(kernel, TaskFrame) if r.id == origin.id)
        configure(
            kernel,
            current.model_copy(
                update={"revision": current.revision + 1, "status": "ready", "wake_condition": None}
            ),
        )
        await kernel.run_until_idle()
        assert len(worker.packets) == 4
        assert len([r for r in records(kernel, DecisionReceipt) if r.subject_id == request.id]) == 1


@pytest.mark.parametrize("ceiling", [2, 3])
async def test_public_parent_children_share_budget_and_exhaust_ancestor(tmp_path, ceiling):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        parent = new_goal()
        parent = parent.model_copy(
            update={"limits": parent.limits.model_copy(update={"calls": ceiling})}
        )
        await kernel.submit(CreateGoal(goal=parent), principal=owner())
        children = [
            new_goal(
                id=uuid4().hex,
                parent_id=parent.id,
                criteria=tuple(c.model_copy(update={"id": uuid4().hex}) for c in parent.criteria),
            )
            for _ in range(2)
        ]
        for child in children:
            assert (
                await kernel.submit(CreateGoal(goal=child), principal=owner())
            ).disposition == "accept"
        await kernel.run_until_idle()
        assert len(worker.packets) == ceiling
        budgets = records(kernel, Budget)
        assert next(b for b in budgets if b.scope.goal_id == parent.id).actual.calls == ceiling
        assert sum(b.actual.calls for b in budgets if b.scope.goal_id != parent.id) == sum(
            p.goal.id != parent.id for p in worker.packets
        )
        assert all(
            t.wake_condition.reason_code != "budget_scope_mismatch"
            for t in records(kernel, TaskFrame)
        )


@pytest.mark.parametrize("scope", ["foreign_owner", "foreign_goal", "detached"])
async def test_episode_rejects_foreign_or_detached_commitment_budget(tmp_path, scope):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        g = new_goal()
        await kernel.submit(CreateGoal(goal=g), principal=owner())
        child = commitment_budget(kernel, 2)
        update = {"revision": child.revision + 1}
        if scope == "detached":
            update["parent_id"] = None
        else:
            foreign = new_goal(
                id=uuid4().hex,
                criteria=tuple(c.model_copy(update={"id": uuid4().hex}) for c in g.criteria),
            )
            configure(kernel, foreign)
            update["scope"] = Scope(
                owner_id=kernel.verifier_principal.id if scope == "foreign_owner" else owner().id,
                goal_id=foreign.id if scope == "foreign_goal" else g.id,
            )
        configure(kernel, child.model_copy(update=update))
        await kernel.run_until_idle()
        assert not worker.packets
        assert not records(kernel, Reservation)
        assert records(kernel, TaskFrame)[0].wake_condition.reason_code == "budget_scope_mismatch"


async def test_commitment_budget_binding_is_rechecked_before_invocation(tmp_path):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        child = commitment_budget(kernel, 2)
        kernel._observe_events()
        ((episode, packet),) = kernel._admit()
        assert kernel._is_current(episode, packet)
        commitment = records(kernel, Commitment)[0]
        configure(
            kernel,
            commitment.model_copy(
                update={"revision": commitment.revision + 1, "budget_id": child.parent_id}
            ),
        )
        assert not kernel._is_current(episode, packet)
        await kernel._episode(episode, packet)
        assert not worker.packets


@pytest.mark.parametrize("alteration", ["sibling", "foreign_owner", "missing_parent"])
async def test_descendant_budget_requires_actual_owned_goal_ancestry(tmp_path, alteration):
    from signal_gating.executive import BudgetError, Goal
    from signal_gating.executive.budgets import BudgetPolicy

    from .test_budgets import maxima

    async with open_kernel(tmp_path, FunctionWorker(blocked)) as kernel:
        parent = new_goal()
        await kernel.submit(CreateGoal(goal=parent), principal=owner())
        child = new_goal(
            id=uuid4().hex,
            parent_id=parent.id,
            criteria=tuple(c.model_copy(update={"id": uuid4().hex}) for c in parent.criteria),
        )
        await kernel.submit(CreateGoal(goal=child), principal=owner())
        state = kernel.store.view()
        current = next(r for r in state.records if isinstance(r, Goal) and r.id == child.id)
        altered = current.model_copy(
            update={
                "owner_id": uuid4().hex if alteration == "foreign_owner" else current.owner_id,
                "parent_id": uuid4().hex
                if alteration == "missing_parent"
                else None
                if alteration == "sibling"
                else current.parent_id,
            }
        )
        invalid = state.model_copy(
            update={"records": tuple(altered if r.id == current.id else r for r in state.records)}
        )
        with pytest.raises(BudgetError) as error:
            BudgetPolicy().reserve(
                invalid, current.budget_id, maxima(calls=1), uuid4().hex, trusted_maximum_quote=True
            )
        assert error.value.code == "budget_scope_mismatch"


async def test_descendant_budget_reservation_fences_goal_ancestry_changes(tmp_path):
    from signal_gating.executive import ConflictError, Goal
    from signal_gating.executive.budgets import BudgetPolicy

    from .test_budgets import maxima

    async with open_kernel(tmp_path, FunctionWorker(blocked)) as kernel:
        parent = new_goal()
        await kernel.submit(CreateGoal(goal=parent), principal=owner())
        child = new_goal(
            id=uuid4().hex,
            parent_id=parent.id,
            criteria=tuple(c.model_copy(update={"id": uuid4().hex}) for c in parent.criteria),
        )
        await kernel.submit(CreateGoal(goal=child), principal=owner())
        current = next(g for g in records(kernel, Goal) if g.id == child.id)
        batch = BudgetPolicy().reserve(
            kernel.store.view(),
            current.budget_id,
            maxima(calls=1),
            uuid4().hex,
            trusted_maximum_quote=True,
        )
        configure(
            kernel, current.model_copy(update={"revision": current.revision + 1, "parent_id": None})
        )
        with pytest.raises(ConflictError):
            kernel.store.commit(batch)
        assert not records(kernel, Reservation)


@pytest.mark.parametrize("ancestor", [False, True])
async def test_current_commitment_and_ancestor_ceiling_rechecked_before_invocation(
    tmp_path, ancestor
):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        child = commitment_budget(kernel, 2)
        kernel._observe_events()
        ((episode, packet),) = kernel._admit()
        assert kernel._is_current(episode, packet)
        budget = next(
            b
            for b in records(kernel, Budget)
            if b.id == (child.parent_id if ancestor else child.id)
        )
        configure(
            kernel,
            budget.model_copy(
                update={
                    "revision": budget.revision + 1,
                    "limits": budget.limits.model_copy(update={"calls": 0}),
                }
            ),
        )
        assert not kernel._is_current(episode, packet)
        await kernel._episode(episode, packet)
        assert not worker.packets
