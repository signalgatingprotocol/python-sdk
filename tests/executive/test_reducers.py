"""Deterministic lifecycle, dependency, and replay guarantees."""

from itertools import count
from pathlib import Path

import pytest

from signal_gating.executive import (
    CommitBatch,
    Commitment,
    ConflictError,
    Episode,
    JournalEvent,
    Ref,
    StateView,
    VerificationResult,
    WakeCondition,
)
from signal_gating.executive.reducers import (
    InvalidTransition,
    rebuild,
    reduce_event,
    validate_transition,
)
from signal_gating.executive.store import Store

from .factories import (
    COMMITMENT_ID,
    GOAL_ID,
    NOW,
    OWNER_ID,
    TASK_ID,
    action,
    criterion,
    frame,
    goal,
    owner,
)

IDS = count(100)


def event(event_type="state.changed", *, sequence=1, subject=GOAL_ID, payload=None):
    return JournalEvent(
        id=f"{next(IDS):032x}",
        created_at=NOW,
        sequence=sequence,
        event_type=event_type,
        producer_id="fixture",
        principal_id=OWNER_ID,
        subject=Ref(
            id=subject, revision=max(1, sequence - 1) if event_type.startswith("goal.") else 1
        ),
        payload=payload or {},
    )


def changed(*records, sequence=1):
    return event(
        sequence=sequence, payload={"records": [r.model_dump(mode="json") for r in records]}
    )


def commitment(**updates):
    return Commitment.model_validate(
        {
            "id": COMMITMENT_ID,
            "created_at": NOW,
            "goal_id": GOAL_ID,
            "responsible_principal_id": OWNER_ID,
            **updates,
        }
    )


def initial(*extra, task=None, **goal_updates):
    return changed(owner(), goal(**goal_updates), commitment(), task or frame(), *extra)


def get(state, identifier):
    return next(record for record in state.records if record.id == identifier)


@pytest.mark.parametrize(
    ("record", "target"),
    [
        (frame(status="completed"), "running"),
        (frame(status="failed"), "ready"),
        (frame(status="ready"), "completed"),
        (goal(status="completed"), "active"),
        (action(status="dispatching"), "cancelled"),
        (action(status="outcome_unknown"), "authorized"),
        (action(status="succeeded"), "dispatching"),
    ],
)
def test_rejects_illegal_transitions(record, target):
    with pytest.raises(InvalidTransition):
        validate_transition(record, target)


@pytest.mark.parametrize(
    ("record", "target"),
    [
        (frame(status="pending"), "ready"),
        (frame(), "running"),
        (frame(status="running"), "verifying"),
        (frame(status="verifying"), "completed"),
        (frame(status="blocked"), "ready"),
        (frame(status="suspended"), "ready"),
        (action(), "authorized"),
        (action(status="awaiting_review"), "denied"),
        (action(status="dispatching"), "outcome_unknown"),
        (action(status="reconciling"), "needs_human"),
        (action(status="needs_human"), "succeeded"),
    ],
)
def test_accepts_lifecycle_edges(record, target):
    validate_transition(record, target)


def test_pause_fences_running_task_and_preserves_inflight_effect():
    running = frame(
        status="running",
        fencing_token=7,
        pending_action_ids=(action().id,),
        resume_brief="Keep draft",
    )
    before = reduce_event(StateView(), initial(action(status="dispatching"), task=running))
    after = reduce_event(before, event("goal.paused", sequence=2))
    assert get(after, GOAL_ID).status == "paused"
    task = get(after, TASK_ID)
    assert (task.status, task.fencing_token, task.revision) == ("suspended", 8, 2)
    assert task.pending_action_ids == running.pending_action_ids
    assert task.resume_brief == "Keep draft"
    assert get(after, action().id).status == "dispatching"
    assert get(before, TASK_ID).status == "running"


def test_resume_preserves_blockers_and_pending_effects():
    blocker = WakeCondition(kind="human_decision", reason_code="review")
    blocked = frame(id="a" * 32, status="blocked", wake_condition=blocker)
    inflight = frame(status="running", pending_action_ids=(action().id,))
    events = [
        initial(blocked, action(status="dispatching"), task=inflight),
        event("goal.paused", sequence=2),
        event("goal.resumed", sequence=3),
    ]
    state = rebuild(events)
    assert get(state, GOAL_ID).status == "active"
    assert get(state, TASK_ID).status == "blocked"
    assert get(state, TASK_ID).wake_condition.reason_code == "pending_actions"
    assert get(state, blocked.id).status == "blocked"
    assert get(state, blocked.id).wake_condition == blocker


def test_resume_ready_work_only_after_settled_dependencies():
    dependency = commitment(id="a" * 32, required=False, status="completed")
    task = frame(status="running", dependencies=(Ref(id=dependency.id),))
    state = rebuild(
        [
            initial(dependency, task=task),
            event("goal.paused", sequence=2),
            event("goal.resumed", sequence=3),
        ]
    )
    assert get(state, TASK_ID).status == "ready"
    assert get(state, TASK_ID).fencing_token == 1


def test_cancel_fences_tasks_cancels_undispatched_and_retains_uncertain_actions():
    actions = tuple(
        action(id=f"{n:032x}", status=status)
        for n, status in enumerate(
            (
                "proposed",
                "awaiting_review",
                "authorized",
                "dispatching",
                "outcome_unknown",
                "reconciling",
                "needs_human",
                "succeeded",
            ),
            200,
        )
    )
    state = rebuild(
        [
            initial(*actions, task=frame(status="running", fencing_token=4)),
            event("goal.cancelled", sequence=2),
        ]
    )
    assert get(state, GOAL_ID).status == "cancelled"
    assert get(state, TASK_ID).status == "cancelled"
    assert get(state, TASK_ID).fencing_token == 5
    assert get(state, COMMITMENT_ID).status == "cancelled"
    assert [get(state, a.id).status for a in actions] == [
        "cancelled",
        "cancelled",
        "cancelled",
        "dispatching",
        "outcome_unknown",
        "reconciling",
        "needs_human",
        "succeeded",
    ]


def test_goal_controls_fence_active_episodes():
    episode = Episode(
        id="5" * 32,
        created_at=NOW,
        task_id=TASK_ID,
        fencing_token=1,
        context_packet_id="c" * 32,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        allowed_proposal_types=(),
        deadline=NOW,
    )
    # A packet is a persisted snapshot. It does not create graph edges back to its episode.
    from .factories import packet

    state = rebuild(
        [
            initial(packet(), episode, task=frame(status="running", fencing_token=1)),
            event("goal.paused", sequence=2),
        ]
    )
    assert get(state, episode.id).status == "fenced"


@pytest.mark.parametrize(
    "broken",
    [
        frame(dependencies=(Ref(id="f" * 32),)),
        frame(dependencies=(Ref(id=TASK_ID),)),
        frame(commitment_id="f" * 32),
        frame(goal_id="f" * 32),
    ],
)
def test_missing_references_and_cycles_reject_atomically(broken):
    state = StateView()
    with pytest.raises(InvalidTransition):
        reduce_event(state, initial(task=broken))
    assert state == StateView()


def test_multinode_dependency_cycle_is_rejected_atomically():
    first = frame(dependencies=(Ref(id="a" * 32),))
    second = frame(id="a" * 32, dependencies=(Ref(id=TASK_ID),))
    with pytest.raises(InvalidTransition):
        rebuild([initial(second, task=first)])


def test_completion_requires_required_commitment_but_ignores_optional_task():
    optional = commitment(id="a" * 32, required=False, status="blocked")
    g = goal(criteria=(), required_commitment_ids=(COMMITMENT_ID,))
    start = changed(owner(), g, commitment(status="completed"), optional, frame(status="completed"))
    completed = g.model_copy(update={"revision": 2, "status": "completed"})
    state = rebuild([start, changed(completed, sequence=2)])
    assert get(state, GOAL_ID).status == "completed"
    assert get(state, optional.id).status == "blocked"
    with pytest.raises(InvalidTransition):
        rebuild([changed(owner(), g, commitment(status="blocked")), changed(completed, sequence=2)])


def test_completion_requires_required_criterion_evidence():
    g = goal()
    completed = g.model_copy(update={"revision": 2, "status": "completed"})
    with pytest.raises(InvalidTransition):
        rebuild([initial(), changed(completed, sequence=2)])
    result = VerificationResult(
        id="f" * 32,
        created_at=NOW,
        criterion_id=criterion().id,
        verifier_id=criterion().verifier_id,
        disposition="accept",
        reason_code="checked",
    )
    state = rebuild(
        [
            changed(owner(), g, commitment(status="completed"), frame(status="completed"), result),
            changed(completed, sequence=2),
        ]
    )
    assert get(state, GOAL_ID).status == "completed"


def test_only_explicit_invalidation_reopens_completion():
    g = goal(criteria=(), status="completed")
    start = changed(owner(), g, commitment(status="completed"), frame(status="completed"))
    reopened = event(
        "completion.invalidated",
        sequence=2,
        payload={"record_ids": [TASK_ID, COMMITMENT_ID, GOAL_ID], "reason": "evidence_retracted"},
    )
    state = rebuild([start, reopened])
    assert get(state, GOAL_ID).status == "active"
    assert get(state, TASK_ID).status == "ready"
    assert get(state, COMMITMENT_ID).status == "ready"
    assert get(rebuild([start]), GOAL_ID).status == "completed"


def test_updates_require_next_revision_and_stable_identity():
    start = initial()
    with pytest.raises(InvalidTransition):
        rebuild([start, changed(goal(revision=3), sequence=2)])
    with pytest.raises(InvalidTransition):
        rebuild([start, changed(goal(revision=2, created_at=NOW.replace(year=2025)), sequence=2)])


@pytest.mark.parametrize(
    "bad",
    [
        event("unknown"),
        event(sequence=2),
        event(payload={"records": []}),
        event(
            payload={
                "records": [
                    {
                        "kind": "goal",
                        "owner_id": OWNER_ID,
                        "outcome": "x",
                        "worker_id": "x",
                        "limits": {},
                    }
                ]
            }
        ),
        event("goal.paused", payload={"surprise": True}),
    ],
)
def test_invalid_event_contracts_fail_closed(bad):
    with pytest.raises(InvalidTransition):
        reduce_event(StateView(), bad)


def test_replay_is_deterministic_and_store_materialization_matches(tmp_path: Path):
    events = [initial(), event("goal.paused", sequence=2), event("goal.resumed", sequence=3)]
    expected = rebuild(events)
    assert rebuild(iter(events)) == expected
    with Store.open(tmp_path / "store") as store:
        for e in events:
            store.commit(
                CommitBatch(
                    events=(e,),
                    expected_revisions=tuple(
                        Ref(id=r.id, revision=r.revision) for r in store.view().records
                    ),
                )
            )
        assert store.view() == expected
        assert rebuild(store.events()) == store.view()


def test_store_rejects_materialization_that_disagrees_with_journal(tmp_path: Path):
    with Store.open(tmp_path / "store") as store:
        with pytest.raises(InvalidTransition):
            store.commit(CommitBatch(events=(initial(),), records=(goal(outcome="different"),)))
        assert store.view() == StateView()
        assert store.events() == ()


@pytest.mark.parametrize("status", ["paused", "cancelled"])
def test_goal_controls_cannot_be_bypassed_with_raw_postimages(status):
    with pytest.raises(InvalidTransition):
        rebuild([initial(), changed(goal(revision=2, status=status), sequence=2)])


def test_paused_goal_cannot_dispatch_authorized_action():
    events = [initial(action(status="authorized")), event("goal.paused", sequence=2)]
    with pytest.raises(InvalidTransition):
        rebuild([*events, changed(action(revision=2, status="dispatching"), sequence=3)])


def test_atomic_reference_validation_is_independent_of_record_order():
    records = initial().model_dump(mode="json")["payload"]["records"]
    forward = rebuild([event(payload={"records": records})])
    reverse = rebuild([event(payload={"records": records[::-1]})])
    assert reverse == forward


def test_store_requires_cas_for_derived_cascades(tmp_path: Path):
    with Store.open(tmp_path / "store") as store:
        store.commit(CommitBatch(events=(initial(),)))
        before = store.view()
        with pytest.raises(ConflictError):
            store.commit(
                CommitBatch(
                    events=(event("goal.paused", sequence=2),),
                    expected_revisions=(Ref(id=GOAL_ID),),
                )
            )
        assert store.view() == before


def test_store_open_rejects_checksummed_materialization_mismatch(tmp_path: Path):
    import hashlib
    import sqlite3

    from signal_gating.executive import StoreError
    from signal_gating.executive.store import canonical_json

    root = tmp_path / "store"
    with Store.open(root) as store:
        store.commit(CommitBatch(events=(initial(),)))
    payload = canonical_json(goal(outcome="unrecorded change"))
    with sqlite3.connect(root / "executive.sqlite3") as db:
        db.execute(
            "UPDATE records SET payload=?, checksum=? WHERE id=?",
            (payload, hashlib.sha256(payload.encode()).hexdigest(), GOAL_ID),
        )
    with pytest.raises(StoreError, match="journal"):
        Store.open(root)


def test_initial_postimages_cannot_generate_nested_identity_or_signal_time():
    from .factories import signal

    raw = goal().model_dump(mode="json")
    del raw["criteria"][0]["created_at"]
    with pytest.raises(InvalidTransition):
        rebuild([event(payload={"records": [owner().model_dump(mode="json"), raw]})])
    raw_signal = signal().model_dump(mode="json")
    del raw_signal["received_at"]
    with pytest.raises(InvalidTransition):
        rebuild([event(payload={"records": [raw_signal]})])


def test_audit_event_preserves_state_and_has_a_closed_bounded_shape():
    audit = event(
        "audit.recorded", payload={"reason_code": "rejected", "details": {"reason": "out of scope"}}
    )
    assert rebuild([audit]) == StateView(sequence=1)
    with pytest.raises(InvalidTransition):
        rebuild(
            [event("audit.recorded", payload={"reason_code": "x", "details": {}, "records": []})]
        )
    with pytest.raises(InvalidTransition):
        rebuild(
            [
                event(
                    "audit.recorded",
                    payload={"reason_code": "x", "details": {"text": "雪" * 22000}},
                )
            ]
        )


def test_current_verification_result_is_unique_per_criterion():
    first = VerificationResult(
        id="e" * 32,
        created_at=NOW,
        criterion_id=criterion().id,
        verifier_id=criterion().verifier_id,
        disposition="accept",
        reason_code="checked",
    )
    second = first.model_copy(update={"id": "f" * 32})
    with pytest.raises(InvalidTransition):
        rebuild([initial(first, second)])


def test_verification_repair_revises_current_result_and_retains_history():
    result = VerificationResult(
        id="e" * 32,
        created_at=NOW,
        criterion_id=criterion().id,
        verifier_id=criterion().verifier_id,
        disposition="insufficient_evidence",
        reason_code="gap",
    )
    events = [changed(owner(), goal(), commitment(status="completed"), result)]
    events.append(
        changed(
            result.model_copy(
                update={"revision": 2, "disposition": "accept", "reason_code": "checked"}
            ),
            sequence=2,
        )
    )
    events.append(changed(goal(revision=2, status="completed"), sequence=3))
    assert get(rebuild(events), GOAL_ID).status == "completed"
    assert get(rebuild(events[:1]), result.id).disposition == "insufficient_evidence"


def test_supported_belief_dependency_can_resume():
    from signal_gating.executive import Belief, Scope

    belief = Belief(
        id="e" * 32,
        created_at=NOW,
        claim="Input verified",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
    )
    task = frame(status="running", dependencies=(Ref(id=belief.id, kind="belief"),))
    state = rebuild(
        [
            initial(belief, task=task),
            event("goal.paused", sequence=2),
            event("goal.resumed", sequence=3),
        ]
    )
    assert get(state, TASK_ID).status == "ready"


def test_invalidation_while_paused_preserves_pause():
    start = initial(task=frame(status="completed"))
    invalidation = event(
        "completion.invalidated",
        sequence=3,
        payload={"record_ids": [TASK_ID], "reason": "evidence_removed"},
    )
    state = rebuild([start, event("goal.paused", sequence=2), invalidation])
    assert get(state, GOAL_ID).status == "paused"
    assert get(state, TASK_ID).status == "suspended"


def test_rebuild_rejects_repeated_journal_identity():
    first = event("audit.recorded", payload={"reason_code": "noop", "details": {}})
    with pytest.raises(InvalidTransition):
        rebuild([first, first.model_copy(update={"sequence": 2})])


@pytest.mark.parametrize(
    "extra",
    [
        action(grant_id="f" * 32),
        action(reservation_id="f" * 32),
        commitment(id="e" * 32, budget_id="f" * 32),
    ],
)
def test_missing_typed_optional_references_are_rejected(extra):
    with pytest.raises(InvalidTransition):
        rebuild([initial(extra)])


def test_criterion_ids_are_globally_unique_across_goals():
    other = goal(id="e" * 32)
    with pytest.raises(InvalidTransition):
        rebuild([initial(other)])


@pytest.mark.parametrize("index", [0, 1])
def test_missing_parent_budget_and_shared_principal_references_are_rejected(index):
    from signal_gating.executive import ArtifactRef, Budget, Limits, Scope

    extras = (
        Budget(
            id="e" * 32,
            created_at=NOW,
            scope=Scope(goal_id=GOAL_ID),
            limits=Limits(),
            parent_id="f" * 32,
        ),
        ArtifactRef(
            id="e" * 32,
            created_at=NOW,
            digest="a" * 64,
            storage_location="artifacts/x",
            media_type="text/plain",
            origin="fixture",
            scope=Scope(goal_id=GOAL_ID),
            size_bytes=0,
            shared_with=("f" * 32,),
        ),
    )
    with pytest.raises(InvalidTransition):
        rebuild([initial(extras[index])])


@pytest.mark.parametrize("reverse", [False, True])
def test_invalidation_checks_dependencies_after_reopening_whole_bundle(tmp_path: Path, reverse):
    dependent_id = "a" * 32
    dependent = frame(
        id=dependent_id,
        status="completed",
        dependencies=(Ref(id=TASK_ID, kind="task"),),
    )
    start = initial(dependent, task=frame(status="completed"))
    identifiers = [TASK_ID, dependent_id]
    invalidation = event(
        "completion.invalidated",
        sequence=2,
        payload={
            "record_ids": identifiers[::-1] if reverse else identifiers,
            "reason": "evidence_retracted",
        },
    )
    state = rebuild([start, invalidation])
    assert (get(state, TASK_ID).status, get(state, TASK_ID).revision) == ("ready", 2)
    assert (get(state, dependent_id).status, get(state, dependent_id).revision) == ("blocked", 2)
    assert get(state, dependent_id).wake_condition == WakeCondition(
        kind="dependency_change",
        refs=(Ref(id=TASK_ID, kind="task"),),
        reason_code="completion_invalidated",
    )
    root = tmp_path / "store"
    with Store.open(root) as store:
        store.commit(CommitBatch(events=(start,)))
        store.commit(
            CommitBatch(
                events=(invalidation,),
                expected_revisions=(Ref(id=TASK_ID), Ref(id=dependent_id)),
            )
        )
        assert store.view() == state
        assert rebuild(store.events()) == state
    with Store.open_readonly(root) as snapshot:
        assert snapshot.view() == state
        assert rebuild(snapshot.events()) == state
