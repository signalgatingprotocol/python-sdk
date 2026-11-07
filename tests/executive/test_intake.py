"""Atomic intake receipts, canonical identities and bounded admission."""

from datetime import timedelta

import pytest

from signal_gating.executive import (
    CommitBatch,
    ExecutiveSignal,
    JournalEvent,
    PauseGoal,
    Ref,
    ReviseGoal,
    Scope,
)
from signal_gating.executive.intake import Intake
from signal_gating.executive.store import Store

from .factories import GOAL_ID, NOW, OWNER_ID, goal, owner, signal


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "store") as current:
        event = JournalEvent(
            sequence=1,
            event_type="state.changed",
            producer_id="bootstrap",
            principal_id=OWNER_ID,
            subject=Ref(id=OWNER_ID),
            payload={"records": [r.model_dump(mode="json") for r in (owner(), goal())]},
        )
        current.commit(CommitBatch(events=(event,)))
        yield current


def accept(store, item, **options):
    intake = Intake(store, **options)
    result = intake.accept(item, principal=owner(), state=store.view())
    store.commit(result)
    return result.response_receipt


def test_exact_duplicate_returns_original_receipt_after_reopen(store):
    command = ReviseGoal(
        goal_id=GOAL_ID, expected_revision=1, outcome="Corrected", external_id="correction-1"
    )
    original = accept(store, command)
    intake = Intake(store)
    duplicate = intake.accept(
        command.model_copy(update={"id": "d" * 32, "created_at": NOW}),
        principal=owner(),
        state=store.view(),
    )
    assert duplicate.response_receipt == original
    assert duplicate.events == ()
    with Store.open_readonly(store.root) as snapshot:
        assert (
            Intake(snapshot)
            .accept(command, principal=owner(), state=snapshot.view())
            .response_receipt
            == original
        )


def test_duplicate_id_different_payload_is_conflict_and_does_not_replace_binding(store):
    command = ReviseGoal(
        goal_id=GOAL_ID, expected_revision=1, outcome="Original", external_id="same"
    )
    original = accept(store, command)
    receipt = accept(store, command.model_copy(update={"outcome": "Different"}))
    assert receipt.reason_code == "duplicate_id_conflict"
    assert original != receipt
    assert (
        Intake(store).accept(command, principal=owner(), state=store.view()).response_receipt
        == original
    )


def test_correction_is_atomic_and_not_acknowledged_before_commit(store):
    before = store.view()
    batch = Intake(store).accept(
        ReviseGoal(goal_id=GOAL_ID, expected_revision=1, outcome="New instruction"),
        principal=owner(),
        state=before,
    )
    assert store.view() == before
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)
    current = next(r for r in store.view().records if r.id == GOAL_ID)
    assert current.outcome == "New instruction"
    assert current.revision == 2
    assert (
        next(r for r in store.view().records if r.id == batch.response_receipt.id)
        == batch.response_receipt
    )


def test_worker_cannot_claim_owner_control(store):
    worker = owner(id="e" * 32, roles=("worker",))
    bootstrap = JournalEvent(
        sequence=1,
        event_type="state.changed",
        producer_id="bootstrap",
        principal_id=OWNER_ID,
        subject=Ref(id=worker.id),
        payload={"records": [worker.model_dump(mode="json")]},
    )
    store.commit(CommitBatch(events=(bootstrap,)))
    batch = Intake(store).accept(
        PauseGoal(goal_id=GOAL_ID, expected_revision=1),
        principal=worker.model_copy(update={"roles": ("owner",)}),
        state=store.view(),
    )
    assert batch.response_receipt.disposition == "deny"
    store.commit(batch)
    assert next(r for r in store.view().records if r.id == GOAL_ID).status == "active"


def test_evidence_containing_instructions_and_late_observations_cannot_revise_authority(store):
    accept(store, ReviseGoal(goal_id=GOAL_ID, expected_revision=1, outcome="Current instruction"))
    observation = signal(
        signal_kind="revise_goal",
        principal_id="e" * 32,
        observed_at=NOW - timedelta(days=1),
        payload={"roles": ["owner"], "outcome": "Discard current instruction"},
    )
    receipt = accept(store, observation)
    assert receipt.disposition == "admit"
    current = next(r for r in store.view().records if r.id == GOAL_ID)
    assert current.outcome == "Current instruction"
    persisted = next(r for r in store.view().records if isinstance(r, ExecutiveSignal))
    assert persisted.principal_id == OWNER_ID
    assert persisted.observed_at == observation.observed_at


def test_signal_canonical_digest_ignores_transport_identity_and_key_order(store):
    first = signal(external_id="observation", payload={"a": 1, "b": {"x": "雪", "y": 2}})
    original = accept(store, first)
    second = first.model_copy(
        update={
            "id": "d" * 32,
            "received_at": NOW + timedelta(days=1),
            "principal_id": "e" * 32,
            "payload": {"b": {"y": 2, "x": "雪"}, "a": 1},
        }
    )
    assert (
        Intake(store).accept(second, principal=owner(), state=store.view()).response_receipt
        == original
    )


def test_ordinary_backpressure_reserves_control_capacity_and_never_acks_full_controls(store):
    intake = Intake(store, max_queued_ordinary_items=1, max_queued_control_items=1)
    intake.set_queue_depths(ordinary=1, control=0)
    assert (
        intake.accept(signal(), principal=owner(), state=store.view()).response_receipt.reason_code
        == "ordinary_queue_full"
    )
    control = intake.accept(
        PauseGoal(goal_id=GOAL_ID, expected_revision=1), principal=owner(), state=store.view()
    )
    assert control.response_receipt.disposition == "accept"
    intake.set_queue_depths(ordinary=1, control=1)
    blocked = intake.accept(
        PauseGoal(goal_id=GOAL_ID, expected_revision=1), principal=owner(), state=store.view()
    )
    assert blocked.response_receipt.reason_code == "control_queue_full"
    assert not blocked.inbox_acknowledgments
    assert not blocked.events
    with pytest.raises(ValueError):
        intake.set_queue_depths(ordinary=-1, control=True)


def test_duplicates_precede_capacity_checks(store):
    item = signal(external_id="dup")
    original = accept(store, item)
    intake = Intake(store, max_queued_ordinary_items=1)
    intake.set_queue_depths(ordinary=1, control=0)
    assert intake.accept(item, principal=owner(), state=store.view()).response_receipt == original


def test_oversize_unicode_is_bounded_and_payload_is_never_persisted(store):
    secret = "PRIVATE雪" * 3000
    batch = Intake(store, max_signal_bytes=512).accept(
        signal(payload={"private": secret}), principal=owner(), state=store.view()
    )
    assert batch.response_receipt.reason_code == "payload_too_large"
    assert len(batch.model_dump_json().encode()) < 4096
    assert "PRIVATE" not in batch.model_dump_json()
    store.commit(batch)
    assert not any(isinstance(r, ExecutiveSignal) for r in store.view().records)


def test_stale_revision_returns_receipt_and_no_correction(store):
    accept(store, ReviseGoal(goal_id=GOAL_ID, expected_revision=1, outcome="Current"))
    result = accept(store, ReviseGoal(goal_id=GOAL_ID, expected_revision=1, outcome="Stale"))
    assert result.disposition == "stale"
    assert next(r for r in store.view().records if r.id == GOAL_ID).outcome == "Current"


def test_retry_cannot_change_authenticated_origin_to_read_a_receipt(store):
    command = ReviseGoal(
        goal_id=GOAL_ID, expected_revision=1, outcome="Private", external_id="owner-only"
    )
    original = accept(store, command)
    reply = Intake(store).accept(
        command, principal=owner(authenticated_origin="other-adapter"), state=store.view()
    )
    assert reply.response_receipt != original
    assert reply.response_receipt.disposition == "deny"


def test_set_grant_updates_current_revision_and_revocation(store):
    from signal_gating.executive import SetGrant

    from .factories import grant

    permission = grant()
    assert accept(store, SetGrant(grant=permission)).disposition == "accept"
    revoked = permission.model_copy(
        update={"revision": 2, "revoked": True, "revocation_revision": 1}
    )
    assert accept(store, SetGrant(grant=revoked, expected_revision=1)).disposition == "accept"
    assert next(r for r in store.view().records if r.id == permission.id) == revoked


def test_usage_resolver_is_authorized_before_callback_and_must_commit_events(store):
    from signal_gating.executive import Budget, Limits, Reservation, ResolveUsage

    budget = Budget(
        id="a" * 32, scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID), limits=Limits(calls=3)
    )
    reservation = Reservation(
        id="b" * 32,
        budget_id=budget.id,
        subject_id=GOAL_ID,
        maximum=Limits(calls=1),
        status="uncertain",
    )
    seed = JournalEvent(
        sequence=1,
        event_type="state.changed",
        producer_id="fixture",
        principal_id=OWNER_ID,
        subject=Ref(id=budget.id),
        payload={"records": [r.model_dump(mode="json") for r in (budget, reservation)]},
    )
    store.commit(CommitBatch(events=(seed,)))
    command = ResolveUsage(
        reservation_id=reservation.id,
        expected_revision=1,
        accept_reserved_maximum=True,
        explanation="Use reserved maximum",
    )
    assert accept(store, command).reason_code == "unsupported_operation"
    assert not store.inbox_acknowledgments()
    calls = []

    def resolver(item, state):
        calls.append((item, state))
        # Task6 owns budget arithmetic. This fixture proves plumbing only.
        event = JournalEvent(
            sequence=1,
            event_type="audit.recorded",
            producer_id="budget-test",
            principal_id=OWNER_ID,
            subject=Ref(id=reservation.id),
            payload={"reason_code": "resolver_called", "details": {}},
        )
        return CommitBatch(
            events=(event,), expected_revisions=(Ref(id=reservation.id, revision=1),)
        )

    result = Intake(store, usage_resolver=resolver).accept(
        command, principal=owner(), state=store.view()
    )
    assert result.response_receipt.disposition == "accept"
    assert len(calls) == 1
    assert len(result.events) == 2
    store.commit(result)
    calls.clear()
    denied = Intake(store, usage_resolver=resolver).accept(
        command, principal=owner(id="f" * 32, roles=("worker",)), state=store.view()
    )
    assert denied.response_receipt.disposition == "deny"
    assert calls == []


def test_conflicting_rules_block_work_and_require_explicit_complete_revision_resolution(store):
    from signal_gating.executive import (
        Commitment,
        Constraint,
        PolicyDecision,
        ResolveConflict,
        ResumeGoal,
    )
    from signal_gating.executive.authority import Authority, RuleRegistry

    from .factories import COMMITMENT_ID, frame

    commitment = Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID)
    task = frame()
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=task.id),
                    payload={"records": [r.model_dump(mode="json") for r in (commitment, task)]},
                ),
            )
        )
    )
    constraint = Constraint(
        description="Trusted conflict", enforcement="rule", enforcement_id="conflict"
    )
    other = Constraint(description="Preserve me", hard=False)
    rules = RuleRegistry(
        {
            "conflict": lambda _: PolicyDecision(
                gate="execution", disposition="deny", reason_code="constraint_conflict"
            )
        }
    )
    authority = Authority(rules=rules)
    receipt = accept(
        store,
        ReviseGoal(goal_id=GOAL_ID, expected_revision=1, constraints=(constraint, other)),
        authority=authority,
    )
    assert receipt.reason_code == "constraint_conflict"
    current = next(r for r in store.view().records if r.id == GOAL_ID)
    assert current.status == "blocked"
    blocked_task = next(r for r in store.view().records if r.id == task.id)
    assert blocked_task.status == "blocked"
    assert blocked_task.wake_condition.reason_code == "constraint_conflict"
    assert Ref(id=constraint.id, revision=1) in receipt.state_revisions
    assert (
        accept(
            store,
            ReviseGoal(goal_id=GOAL_ID, expected_revision=current.revision, constraints=()),
            authority=authority,
        ).reason_code
        == "constraint_conflict_unresolved"
    )
    assert (
        accept(
            store,
            ResumeGoal(goal_id=GOAL_ID, expected_revision=current.revision),
            authority=authority,
        ).disposition
        != "accept"
    )
    resolution = ResolveConflict(
        goal_id=GOAL_ID,
        expected_goal_revision=current.revision,
        conflicting_revisions=(Ref(id=constraint.id, revision=1),),
        resolved_constraints=(),
        explanation="Remove named conflicting restriction",
    )
    assert accept(store, resolution, authority=authority).disposition == "accept"
    resolved = next(r for r in store.view().records if r.id == GOAL_ID)
    assert resolved.status == "active"
    assert resolved.constraints == (other,)
    resumed_task = next(r for r in store.view().records if r.id == task.id)
    assert resumed_task.wake_condition is None
    assert resumed_task.status == "suspended"


def test_create_goal_has_initial_work_and_replays_after_reopen(store):
    from signal_gating.executive import CreateGoal, TaskFrame

    target = goal(id="f" * 32, criteria=())
    receipt = accept(store, CreateGoal(goal=target))
    assert receipt.disposition == "accept"
    tasks = [r for r in store.view().records if isinstance(r, TaskFrame) and r.goal_id == target.id]
    assert len(tasks) == 1 and tasks[0].status == "ready"
    with Store.open_readonly(store.root) as snapshot:
        assert snapshot.view() == store.view()


def test_pause_receipt_and_cascade_are_one_atomic_batch(store):
    import sqlite3

    from signal_gating.executive import ConflictError

    before = store.view()
    batch = Intake(store).accept(
        PauseGoal(goal_id=GOAL_ID, expected_revision=1, external_id="stop"),
        principal=owner(),
        state=before,
    )
    with sqlite3.connect(store.root / "executive.sqlite3") as db:
        db.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON records "
            "WHEN NEW.kind = 'receipt' "
            "BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END"
        )
    with pytest.raises(ConflictError):
        store.commit(batch)
    assert store.view() == before
    assert store.inbox_acknowledgments() == ()


def test_oversized_producer_rejection_metadata_is_bounded(store):
    batch = Intake(store, max_signal_bytes=512).accept(
        signal(producer_id="PRIVATE" * 1000), principal=owner(), state=store.view()
    )
    assert batch.response_receipt.reason_code == "payload_too_large"
    assert len(batch.model_dump_json().encode()) < 4096
    assert "PRIVATE" not in batch.model_dump_json()


def test_invalid_unicode_is_a_bounded_rejection(store):
    result = Intake(store).accept(
        signal(payload={"text": "\ud800"}), principal=owner(), state=store.view()
    )
    assert result.response_receipt.reason_code == "invalid_payload_encoding"
    store.commit(result)


def test_owner_action_attestation_does_not_retry_an_effect(store):
    from signal_gating.executive import Commitment, ResolveAction

    from .factories import COMMITMENT_ID, action, frame

    commitment = Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID)
    issued = action(status="outcome_unknown")
    seed = (commitment, frame(), issued)
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=issued.id),
                    payload={"records": [r.model_dump(mode="json") for r in seed]},
                ),
            )
        )
    )
    result = accept(
        store,
        ResolveAction(
            action_id=issued.id,
            expected_revision=1,
            outcome="failed",
            evidence=(Ref(id=GOAL_ID, kind="goal"),),
            explanation="Owner checked external system",
        ),
    )
    assert result.disposition == "accept"
    current = next(r for r in store.view().records if r.id == issued.id)
    assert current.status == "failed"
    assert current.external_receipt.startswith("human-attestation:")
    assert not store.outbox()


def test_conflict_dto_rejects_ambiguous_or_unnamed_resolution():
    from pydantic import ValidationError

    from signal_gating.executive import ResolveConflict

    with pytest.raises(ValidationError):
        ResolveConflict(belief_ids=("a" * 32,), explanation="No revision")
    with pytest.raises(ValidationError):
        ResolveConflict(
            goal_id=GOAL_ID,
            expected_goal_revision=1,
            conflicting_revisions=(Ref(id="a" * 32),),
            resolved_constraints=(),
            belief_ids=("a" * 32,),
            explanation="Mixed mode",
        )


def test_signal_sequence_only_retries_are_durable_and_conflicts_reject(store):
    item = signal(external_id=None, producer_sequence=7)
    original = accept(store, item)
    assert (
        Intake(store)
        .accept(item.model_copy(update={"id": "d" * 32}), principal=owner(), state=store.view())
        .response_receipt
        == original
    )
    conflict = accept(
        store, item.model_copy(update={"external_id": "different", "payload": {"new": True}})
    )
    assert conflict.reason_code == "duplicate_sequence_conflict"


def test_evidence_attachment_respects_current_goal_and_artifact_acl(store):
    from signal_gating.executive import ArtifactRef, AttachEvidence

    artifact = ArtifactRef(
        digest="a" * 64,
        storage_location="artifacts/" + "a" * 64,
        media_type="text/plain",
        origin="fixture",
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
        size_bytes=0,
        valid=False,
    )
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=artifact.id),
                    payload={"records": [artifact.model_dump(mode="json")]},
                ),
            )
        )
    )
    result = accept(
        store, AttachEvidence(goal_id=GOAL_ID, evidence=(Ref(id=artifact.id, kind="artifact"),))
    )
    assert result.reason_code == "evidence_access_denied"


def test_oversized_causal_metadata_is_not_echoed_in_rejection(store):
    item = signal(causal_ids=("a" * 32,) * 1000)
    batch = Intake(store, max_signal_bytes=512).accept(item, principal=owner(), state=store.view())
    assert batch.response_receipt.reason_code == "payload_too_large"
    assert len(batch.model_dump_json().encode()) < 4096


def test_durable_denial_and_oversize_retry_return_original_receipt(store):
    command = ReviseGoal(
        goal_id=GOAL_ID, expected_revision=9, external_id="stale", outcome="No change"
    )
    original = accept(store, command)
    assert original.disposition == "stale"
    assert (
        Intake(store).accept(command, principal=owner(), state=store.view()).response_receipt
        == original
    )
    huge = signal(producer_id="PRIVATE" * 2000, external_id="SECRET" * 2000)
    rejected = accept(store, huge, max_signal_bytes=512)
    duplicate = Intake(store, max_signal_bytes=512).accept(
        huge, principal=owner(), state=store.view()
    )
    assert duplicate.response_receipt == rejected
    assert duplicate.events == ()
    assert len(duplicate.model_dump_json().encode()) < 4096
    assert all(
        "PRIVATE" not in ack.model_dump_json() and "SECRET" not in ack.model_dump_json()
        for ack in store.inbox_acknowledgments()
    )


def test_correction_handles_pending_work_without_invalid_transition(store):
    from signal_gating.executive import Commitment

    from .factories import COMMITMENT_ID, frame

    commitment = Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID)
    task = frame(status="pending")
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=task.id),
                    payload={"records": [r.model_dump(mode="json") for r in (commitment, task)]},
                ),
            )
        )
    )
    result = accept(
        store, ReviseGoal(goal_id=GOAL_ID, expected_revision=1, outcome="Correct pending work")
    )
    assert result.disposition == "accept"
    assert next(r for r in store.view().records if r.id == task.id).status == "pending"


def test_explicit_belief_resolution_is_revision_checked(store):
    from signal_gating.executive import Belief, ResolveConflict

    beliefs = tuple(
        Belief(
            claim=claim,
            scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
            epistemic_status="disputed",
        )
        for claim in ("A", "B")
    )
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=beliefs[0].id),
                    payload={"records": [r.model_dump(mode="json") for r in beliefs]},
                ),
            )
        )
    )
    command = ResolveConflict(
        belief_ids=tuple(b.id for b in beliefs),
        accepted_belief_id=beliefs[0].id,
        conflicting_revisions=tuple(Ref(id=b.id, kind="belief") for b in beliefs),
        explanation="Owner inspected conflicting claims",
    )
    assert accept(store, command).disposition == "accept"
    current = {r.id: r for r in store.view().records}
    assert current[beliefs[0].id].epistemic_status == "supported"
    assert current[beliefs[1].id].epistemic_status == "retracted"
    assert (
        accept(store, command.model_copy(update={"id": "d" * 32})).reason_code == "stale_conflict"
    )


def test_owner_approval_is_bound_to_arguments_scope_and_goal_revision(store):
    from signal_gating.executive import Commitment, DecideAction

    from .factories import COMMITMENT_ID, action, frame

    commitment = Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID)
    pending = action(status="awaiting_review")
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=pending.id),
                    payload={
                        "records": [
                            r.model_dump(mode="json") for r in (commitment, frame(), pending)
                        ]
                    },
                ),
            )
        )
    )
    command = DecideAction(
        action_id=pending.id,
        expected_revision=1,
        decision="approve",
        arguments_digest=pending.arguments_digest,
        scope=pending.scope,
        constraint_revision=1,
    )
    assert (
        accept(
            store, command.model_copy(update={"id": "f" * 32, "arguments_digest": "f" * 64})
        ).reason_code
        == "stale_action_approval"
    )
    assert accept(store, command).disposition == "accept"
    assert next(r for r in store.view().records if r.id == pending.id).status == "authorized"
    assert not store.outbox()


def test_constraint_verdict_is_evaluated_once_before_mutating_goal_status(store):
    from signal_gating.executive import Constraint, PolicyDecision
    from signal_gating.executive.authority import Authority, RuleRegistry

    calls = []

    def rule(inputs):
        calls.append(inputs["goal"].status)
        return PolicyDecision(
            gate="execution",
            disposition="deny" if inputs["goal"].status == "active" else "authorize",
            reason_code="constraint_conflict" if inputs["goal"].status == "active" else "ok",
        )

    constraint = Constraint(
        description="Depends on current state", enforcement="rule", enforcement_id="rule"
    )
    receipt = accept(
        store,
        ReviseGoal(goal_id=GOAL_ID, expected_revision=1, constraints=(constraint,)),
        authority=Authority(rules=RuleRegistry({"rule": rule})),
    )
    assert receipt.reason_code == "constraint_conflict"
    assert calls == ["active"]


def test_revised_conflicting_constraint_can_be_removed_with_replayable_history(store):
    from signal_gating.executive import Constraint, PolicyDecision, ResolveConflict
    from signal_gating.executive.authority import Authority, RuleRegistry
    from signal_gating.executive.reducers import rebuild

    constraint = Constraint(
        description="Conflicting rule", enforcement="rule", enforcement_id="conflict"
    )
    authority = Authority(
        rules=RuleRegistry(
            {
                "conflict": lambda _: PolicyDecision(
                    gate="execution", disposition="deny", reason_code="constraint_conflict"
                )
            }
        )
    )
    first = accept(
        store,
        ReviseGoal(goal_id=GOAL_ID, expected_revision=1, constraints=(constraint,)),
        authority=authority,
    )
    assert first.reason_code == "constraint_conflict"
    blocked = next(r for r in store.view().records if r.id == GOAL_ID)
    revised = constraint.model_copy(update={"revision": 2, "description": "Still conflicting"})
    second = accept(
        store,
        ResolveConflict(
            goal_id=GOAL_ID,
            expected_goal_revision=blocked.revision,
            conflicting_revisions=(Ref(id=constraint.id),),
            resolved_constraints=(revised,),
            explanation="Revise the named conflict",
        ),
        authority=authority,
    )
    assert second.reason_code == "constraint_conflict"
    blocked = next(r for r in store.view().records if r.id == GOAL_ID)
    resolved = accept(
        store,
        ResolveConflict(
            goal_id=GOAL_ID,
            expected_goal_revision=blocked.revision,
            conflicting_revisions=(Ref(id=constraint.id, revision=2),),
            resolved_constraints=(),
            explanation="Remove revised conflict",
        ),
        authority=authority,
    )
    assert resolved.disposition == "accept"
    current = {r.id: r for r in store.view().records}
    assert current[GOAL_ID].status == "active"
    assert current[GOAL_ID].constraints == ()
    assert current[constraint.id] == revised
    assert Ref(id=constraint.id, revision=2) in second.state_revisions
    assert rebuild(store.events()) == store.view()
    with Store.open_readonly(store.root) as snapshot:
        assert snapshot.view() == store.view()


def test_paused_goal_accepts_conflicting_correction_and_resolution_without_resuming(store):
    from signal_gating.executive import (
        Commitment,
        Constraint,
        PolicyDecision,
        ResolveConflict,
        ResumeGoal,
    )
    from signal_gating.executive.authority import Authority, RuleRegistry
    from signal_gating.executive.reducers import rebuild

    from .factories import COMMITMENT_ID, frame

    commitment = Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID)
    task = frame()
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=task.id),
                    payload={"records": [r.model_dump(mode="json") for r in (commitment, task)]},
                ),
            )
        )
    )
    assert accept(store, PauseGoal(goal_id=GOAL_ID, expected_revision=1)).disposition == "accept"
    paused = {r.id: r for r in store.view().records}
    constraint = Constraint(
        description="New conflict while paused", enforcement="rule", enforcement_id="conflict"
    )
    authority = Authority(
        rules=RuleRegistry(
            {
                "conflict": lambda _: PolicyDecision(
                    gate="execution", disposition="deny", reason_code="constraint_conflict"
                )
            }
        )
    )
    receipt = accept(
        store,
        ReviseGoal(
            goal_id=GOAL_ID, expected_revision=paused[GOAL_ID].revision, constraints=(constraint,)
        ),
        authority=authority,
    )
    assert receipt.reason_code == "constraint_conflict"
    corrected = {r.id: r for r in store.view().records}
    assert corrected[GOAL_ID].status == "paused"
    assert Ref(id=constraint.id) in receipt.state_revisions
    assert corrected[task.id].status not in ("ready", "running")
    assert corrected[task.id].fencing_token > paused[task.id].fencing_token
    assert corrected[task.id].wake_condition.reason_code == "constraint_conflict"
    assert (
        accept(
            store,
            ResumeGoal(goal_id=GOAL_ID, expected_revision=corrected[GOAL_ID].revision),
            authority=authority,
        ).reason_code
        == "constraint_conflict_unresolved"
    )
    resolved = accept(
        store,
        ResolveConflict(
            goal_id=GOAL_ID,
            expected_goal_revision=corrected[GOAL_ID].revision,
            conflicting_revisions=(Ref(id=constraint.id),),
            resolved_constraints=(),
            explanation="Remove named conflict without resuming",
        ),
        authority=authority,
    )
    assert resolved.disposition == "accept"
    current = {r.id: r for r in store.view().records}
    assert current[GOAL_ID].status == "paused"
    assert current[task.id].status == "suspended"
    assert current[task.id].wake_condition is None
    assert rebuild(store.events()) == store.view()
    with Store.open_readonly(store.root) as snapshot:
        assert snapshot.view() == store.view()
    assert (
        accept(
            store,
            ResumeGoal(goal_id=GOAL_ID, expected_revision=current[GOAL_ID].revision),
            authority=authority,
        ).disposition
        == "accept"
    )
    assert next(r for r in store.view().records if r.id == task.id).status == "ready"
