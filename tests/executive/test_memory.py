"""Evidence admission and transitive invalidation through the real journal."""

from signal_gating.executive import (
    ArtifactRef,
    Belief,
    CommitBatch,
    Commitment,
    JournalEvent,
    Ref,
    Scope,
    StateView,
    VerificationResult,
)
from signal_gating.executive.memory import MemoryPolicy, invalidate_dependents
from signal_gating.executive.reducers import rebuild
from signal_gating.executive.store import Store

from .factories import (
    COMMITMENT_ID,
    EPISODE_ID,
    GOAL_ID,
    NOW,
    OWNER_ID,
    TASK_ID,
    criterion,
    frame,
    goal,
    owner,
    packet,
    proposal,
)


def ref(record):
    return Ref(id=record.id, revision=record.revision)


def event(records):
    return JournalEvent(
        sequence=1,
        event_type="state.changed",
        producer_id="test",
        principal_id=OWNER_ID,
        subject=ref(owner()),
        payload={"records": [r.model_dump(mode="json") for r in records]},
    )


def artifact(**changes):
    return ArtifactRef(
        id="d" * 32,
        created_at=NOW,
        digest="0" * 64,
        storage_location="artifacts/" + "0" * 64,
        media_type="text/plain",
        origin="owner",
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
        size_bytes=1,
        **changes,
    )


def test_unknown_or_invalid_evidence_never_establishes_support():
    change = proposal(belief=proposal().belief.model_copy(update={"epistemic_status": "supported"}))
    policy = MemoryPolicy(clock=lambda: NOW)
    assert (
        policy.evaluate(change, StateView(records=(owner(), goal()))).disposition
        == "request_evidence"
    )
    bad = artifact(valid=False)
    change = change.model_copy(
        update={"belief": change.belief.model_copy(update={"evidence": (ref(bad),)})}
    )
    assert (
        MemoryPolicy(evidence_validators={OWNER_ID: lambda c, s: True}, clock=lambda: NOW)
        .evaluate(change, StateView(records=(owner(), goal(), bad)))
        .reason_code
        == "invalid_evidence"
    )


def test_only_owner_registered_validator_can_support_current_accessible_evidence():
    a = artifact()
    change = proposal(belief=proposal().belief.model_copy(update={"evidence": (ref(a),)}))
    state = StateView(records=(owner(), goal(), a))
    assert MemoryPolicy(clock=lambda: NOW).evaluate(change, state).disposition == "request_evidence"
    assert (
        MemoryPolicy(evidence_validators={OWNER_ID: lambda c, s: True}, clock=lambda: NOW)
        .evaluate(change, state)
        .disposition
        == "admit"
    )
    assert (
        MemoryPolicy(evidence_validators={OWNER_ID: lambda c, s: "yes"}, clock=lambda: NOW)
        .evaluate(change, state)
        .disposition
        == "request_evidence"
    )


def test_trusted_conflict_detector_disputes_supported_claims():
    a = artifact()
    old = Belief(
        id="e" * 32,
        claim="The test passed",
        scope=a.scope,
        evidence=(ref(a),),
        epistemic_status="supported",
    )
    change = proposal(
        belief=proposal().belief.model_copy(
            update={"claim": "The test failed", "evidence": (ref(a),), "scope": a.scope}
        )
    )
    state = StateView(records=(owner(), goal(), a, old))
    plain = MemoryPolicy(evidence_validators={OWNER_ID: lambda c, s: True}, clock=lambda: NOW)
    assert plain.evaluate(change, state).disposition == "admit"
    policy = MemoryPolicy(
        evidence_validators={OWNER_ID: lambda c, s: True},
        conflict_detectors={OWNER_ID: lambda a, b, s: True},
        clock=lambda: NOW,
    )
    assert policy.evaluate(change, state).disposition == "dispute"
    assert ref(old) in policy.evaluate(change, state).evidence


def completed_records(paused=False):
    a = artifact()
    b = Belief(
        id="a" * 32,
        claim="Checked",
        scope=a.scope,
        evidence=(ref(a),),
        epistemic_status="supported",
    )
    c = criterion(inputs=(ref(b),), required_evidence=(ref(a),))
    g = goal(criteria=(c,), status="paused" if paused else "completed")
    v = VerificationResult(
        id="e" * 32,
        criterion_id=c.id,
        verifier_id=c.verifier_id,
        disposition="accept",
        reason_code="checked",
        read_set=(ref(b),),
        evidence=(ref(a),),
    )
    commitment = Commitment(
        id=COMMITMENT_ID,
        goal_id=GOAL_ID,
        responsible_principal_id=OWNER_ID,
        dependencies=(ref(b),),
        status="completed",
    )
    task = frame(status="completed", dependencies=(ref(b),))
    return owner(), g, commitment, task, a, b, v


def test_revoked_artifact_retracts_support_reopens_completion_and_preserves_history(tmp_path):
    from signal_gating.executive.artifacts import ArtifactStore

    with Store.open(tmp_path / "store") as store:
        original = store.commit(CommitBatch(events=(event(completed_records()),)))[0]
        a = next(r for r in store.view().records if isinstance(r, ArtifactRef))
        ArtifactStore(store).delete(a, principal=owner())
        records = {r.id: r for r in store.view().records}
        assert records["a" * 32].epistemic_status == "retracted"
        assert records[GOAL_ID].status == "active"
        assert records[TASK_ID].status == "blocked"
        assert records["e" * 32].disposition == "insufficient_evidence"
        assert original in store.events()
        assert rebuild(store.events()) == store.view()
        with Store.open_readonly(store.root) as snapshot:
            assert snapshot.view() == store.view()


def test_invalidation_preserves_pause_and_unrelated_artifacts():
    records = completed_records(paused=True)
    state = StateView(records=records)
    events = invalidate_dependents(state, (ref(records[4]),))
    from signal_gating.executive.reducers import reduce_event

    for offset, item in enumerate(events, 1):
        state = reduce_event(state, item.model_copy(update={"sequence": offset}))
    current = {r.id: r for r in state.records}
    assert current[GOAL_ID].status == "paused"
    assert current[TASK_ID].status == "suspended"
    assert current[records[4].id] == records[4]


def test_owner_correction_reopens_goal_and_invalidates_finding(tmp_path):
    from signal_gating.executive import ReviseGoal
    from signal_gating.executive.intake import Intake

    with Store.open(tmp_path / "store") as store:
        store.commit(CommitBatch(events=(event(completed_records()),)))
        batch = Intake(store, clock=lambda: NOW).accept(
            ReviseGoal(goal_id=GOAL_ID, expected_revision=1, outcome="Corrected outcome"),
            principal=owner(),
            state=store.view(),
        )
        assert batch.response_receipt.disposition == "accept"
        store.commit(batch)
        current = {r.id: r for r in store.view().records}
        assert current[GOAL_ID].status == "active"
        assert current["e" * 32].disposition == "insufficient_evidence"
        assert current[TASK_ID].status in {"suspended", "ready", "blocked"}
        assert rebuild(store.events()) == store.view()


def test_criterion_inputs_invalidate_even_when_finding_omits_dependency():
    from signal_gating.executive.reducers import reduce_event

    values = completed_records()
    values = tuple(
        r.model_copy(update={"read_set": (), "evidence": ()})
        if isinstance(r, VerificationResult)
        else r
        for r in values
    )
    state = StateView(records=values)
    for item in invalidate_dependents(state, (ref(values[4]),)):
        state = reduce_event(state, item.model_copy(update={"sequence": state.sequence + 1}))
    assert (
        next(r for r in state.records if isinstance(r, VerificationResult)).disposition
        == "insufficient_evidence"
    )
    assert next(r for r in state.records if r.id == GOAL_ID).status == "active"


def test_removed_completed_criterion_retains_history_and_reopens(tmp_path):
    from signal_gating.executive import ReviseGoal
    from signal_gating.executive.intake import Intake

    values = completed_records()
    old_criterion = values[1].criteria[0].model_copy(update={"revision": 2})
    values = tuple(
        r.model_copy(update={"criteria": (old_criterion,)}) if r.id == GOAL_ID else r
        for r in values
    )
    with Store.open(tmp_path / "store") as store:
        store.commit(CommitBatch(events=(event(values),)))
        batch = Intake(store, clock=lambda: NOW).accept(
            ReviseGoal(goal_id=GOAL_ID, expected_revision=1, criteria=()),
            principal=owner(),
            state=store.view(),
        )
        assert batch.response_receipt.disposition == "accept"
        store.commit(batch)
        records = {r.id: r for r in store.view().records}
        assert records[old_criterion.id] == old_criterion
        assert records[GOAL_ID].status == "active"
        assert records["e" * 32].disposition == "insufficient_evidence"
        assert rebuild(store.events()) == store.view()


def test_running_episode_is_fenced_by_transitive_evidence_invalidation(tmp_path):
    from datetime import timedelta

    from signal_gating.executive import Episode
    from signal_gating.executive.artifacts import ArtifactStore

    values = list(completed_records())
    values[1] = values[1].model_copy(update={"status": "active"})
    values[2] = values[2].model_copy(update={"status": "pending"})
    values[3] = values[3].model_copy(update={"status": "running", "fencing_token": 1})
    context = packet(goal=values[1], task=values[3], beliefs=(values[5],))
    episode = Episode(
        id=EPISODE_ID,
        task_id=TASK_ID,
        fencing_token=1,
        context_packet_id=context.id,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        allowed_proposal_types=("belief_change",),
        deadline=NOW + timedelta(minutes=5),
    )
    with Store.open(tmp_path / "store") as store:
        store.commit(CommitBatch(events=(event((*values, context, episode)),)))
        ArtifactStore(store).delete(values[4], principal=owner())
        current = {r.id: r for r in store.view().records}
        assert current[episode.id].status == "fenced"
        assert current[TASK_ID].status == "suspended"
        assert current[TASK_ID].fencing_token == 2
        assert rebuild(store.events()) == store.view()


def test_owner_evidence_and_reverify_requests_do_not_grant_verification(tmp_path):
    from signal_gating.executive import AttachEvidence, Reverify
    from signal_gating.executive.intake import Intake

    values = completed_records(paused=True)
    with Store.open(tmp_path / "store") as store:
        store.commit(CommitBatch(events=(event(values),)))
        for command in (
            AttachEvidence(goal_id=GOAL_ID, evidence=(ref(values[4]),)),
            Reverify(goal_id=GOAL_ID),
        ):
            batch = Intake(store, clock=lambda: NOW).accept(
                command, principal=owner(), state=store.view()
            )
            assert batch.response_receipt.disposition == "accept"
            store.commit(batch)
        current = {r.id: r for r in store.view().records}
        assert current[GOAL_ID].status == "paused"
        assert current["e" * 32] == values[6]
        requests = [r for r in current.values() if r.kind == "signal"]
        assert {r.signal_kind for r in requests} == {"attach_evidence", "reverify"}
        assert rebuild(store.events()) == store.view()


def test_replaced_completed_criterion_keeps_old_finding_as_unverified_history(tmp_path):
    from signal_gating.executive import ReviseGoal
    from signal_gating.executive.intake import Intake

    values = completed_records()
    replacement = criterion(id="f" * 32, description="Corrected criterion")
    with Store.open(tmp_path / "store") as store:
        store.commit(CommitBatch(events=(event(values),)))
        batch = Intake(store, clock=lambda: NOW).accept(
            ReviseGoal(goal_id=GOAL_ID, expected_revision=1, criteria=(replacement,)),
            principal=owner(),
            state=store.view(),
        )
        assert batch.response_receipt.disposition == "accept"
        store.commit(batch)
        current = {r.id: r for r in store.view().records}
        assert current[values[1].criteria[0].id] == values[1].criteria[0]
        assert current[GOAL_ID].criteria == (replacement,)
        assert current[GOAL_ID].status == "active"
        assert current["e" * 32].disposition == "insufficient_evidence"
        assert rebuild(store.events()) == store.view()


def test_completed_goal_correction_can_block_on_new_conflict_then_preserve_owner_pause(tmp_path):
    from signal_gating.executive import (
        Constraint,
        PauseGoal,
        PolicyDecision,
        ResolveConflict,
        ReviseGoal,
    )
    from signal_gating.executive.authority import Authority, RuleRegistry
    from signal_gating.executive.intake import Intake

    constraint = Constraint(
        description="Conflicting correction", enforcement="rule", enforcement_id="conflict"
    )
    authority = Authority(
        rules=RuleRegistry(
            {
                "conflict": lambda inputs: PolicyDecision(
                    gate="execution", disposition="deny", reason_code="constraint_conflict"
                )
            }
        )
    )
    with Store.open(tmp_path / "store") as store:
        store.commit(CommitBatch(events=(event(completed_records()),)))
        intake = Intake(store, authority=authority, clock=lambda: NOW)
        batch = intake.accept(
            ReviseGoal(goal_id=GOAL_ID, expected_revision=1, constraints=(constraint,)),
            principal=owner(),
            state=store.view(),
        )
        assert batch.response_receipt.reason_code == "constraint_conflict"
        store.commit(batch)
        current = {r.id: r for r in store.view().records}
        assert current[GOAL_ID].status == "blocked"
        assert current[TASK_ID].wake_condition.reason_code == "constraint_conflict"
        store.commit(
            intake.accept(
                PauseGoal(goal_id=GOAL_ID, expected_revision=current[GOAL_ID].revision),
                principal=owner(),
                state=store.view(),
            )
        )
        current = {r.id: r for r in store.view().records}
        store.commit(
            intake.accept(
                ResolveConflict(
                    goal_id=GOAL_ID,
                    expected_goal_revision=current[GOAL_ID].revision,
                    conflicting_revisions=(ref(constraint),),
                    resolved_constraints=(),
                    explanation="Owner resolved",
                ),
                principal=owner(),
                state=store.view(),
            )
        )
        current = {r.id: r for r in store.view().records}
        assert current[GOAL_ID].status == "paused"
        assert current[TASK_ID].status == "suspended"
        assert current[TASK_ID].wake_condition is None
        assert current["e" * 32].disposition == "insufficient_evidence"
        assert rebuild(store.events()) == store.view()
