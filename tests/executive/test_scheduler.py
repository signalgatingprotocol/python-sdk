"""Deterministic scheduler behavior, including real journal persistence."""

from datetime import timedelta

import pytest

from signal_gating.executive import (
    Belief,
    CommitBatch,
    Commitment,
    DecisionReceipt,
    ExecutiveSignal,
    Intake,
    JournalEvent,
    KernelConfig,
    PauseGoal,
    Ref,
    ResumeGoal,
    Reverify,
    ReviseGoal,
    Scope,
    StateView,
    TaskFrame,
    WakeCondition,
)
from signal_gating.executive.reducers import rebuild, reduce_event
from signal_gating.executive.scheduler import AttentionPolicy, Scheduler, wake
from signal_gating.executive.store import Store

from .factories import COMMITMENT_ID, GOAL_ID, NOW, OWNER_ID, action, frame, goal, owner, signal


class Clock:
    def __init__(self):
        self.now = NOW

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def state(*records):
    return StateView(
        records=(
            owner(),
            goal(),
            Commitment(
                id=COMMITMENT_ID,
                created_at=NOW,
                goal_id=GOAL_ID,
                responsible_principal_id=OWNER_ID,
            ),
            *records,
        )
    )


def changed(view, *records, principal_id=OWNER_ID):
    return JournalEvent(
        sequence=view.sequence + 1,
        created_at=NOW,
        event_type="state.changed",
        producer_id="fixture",
        principal_id=principal_id,
        subject=Ref(id=GOAL_ID),
        payload={"records": [r.model_dump(mode="json") for r in records]},
    )


def apply(view, batch):
    for event in batch.events:
        view = reduce_event(view, event)
    return view


def seed(store, *records):
    view = state(*records)
    store.commit(CommitBatch(events=(changed(StateView(), *view.records),)))
    return store.view()


def test_weighted_cursor_shared_by_actions_and_tasks_survives_restart(tmp_path):
    frames = [frame(id=f"{100 + i:032x}", priority=i) for i in range(15)]
    frames = [t.model_copy(update={"deadline": NOW}) if i < 5 else t for i, t in enumerate(frames)]
    effects = [
        action(id=f"{200 + i:032x}", task_id=frames[i].id, status="outcome_unknown")
        for i in range(8)
    ]
    with Store.open(tmp_path / "state") as store:
        view = seed(store, *frames, *effects)
        scheduler = Scheduler(KernelConfig(max_active_episodes=1))
        classes = []
        for _ in range(7):
            plan = scheduler.plan(view, NOW)
            assert len(plan) == 1
            item = plan[0]
            classes.append(item.work_class)
            assert scheduler.select(view, NOW) == (() if item.kind == "action" else (item.id,))
            assert scheduler.select_actions(view, NOW) == (
                (item.id,) if item.kind == "action" else ()
            )
            batch = scheduler.prepare_selection(view, NOW)
            assert {r.id for r in batch.expected_revisions} == {r.id for r in view.records}
            store.commit(batch)
            view = store.view()
            scheduler = Scheduler(KernelConfig(max_active_episodes=1))
        assert classes == ["reconcile"] * 4 + ["urgent"] * 2 + ["ordinary"]
        assert rebuild(store.events()) == view
    with Store.open(tmp_path / "state") as store:
        assert scheduler.plan(store.view(), NOW) == scheduler.plan(view, NOW)


def test_deadline_aging_and_advisory_task_priority():
    young = frame(id="a" * 32, priority=999)
    old = frame(id="b" * 32, created_at=NOW - timedelta(seconds=120))
    deadline = frame(id="c" * 32, deadline=NOW + timedelta(seconds=30))
    scheduler = Scheduler(KernelConfig(max_active_episodes=3))
    selected = scheduler.select(state(young, old, deadline), NOW)
    assert selected == (deadline.id, old.id, young.id)
    assert scheduler.plan(state(young), NOW)[0].work_class == "ordinary"


def test_filters_ineligible_and_counts_running_episodes():
    ready = frame()
    blocked = frame(id="a" * 32, status="blocked")
    suspended = frame(id="b" * 32, status="suspended")
    running = frame(id="c" * 32, status="running")
    stale = frame(id="d" * 32, dependencies=(Ref(id="e" * 32),))
    scheduler = Scheduler(KernelConfig(max_active_episodes=2))
    assert scheduler.select(state(ready, blocked, suspended, running, stale), NOW) == (ready.id,)
    assert Scheduler(KernelConfig(max_active_episodes=1)).select(state(ready, running), NOW) == ()
    paused = state(ready, action(status="outcome_unknown"))
    paused = paused.model_copy(
        update={
            "records": tuple(
                r.model_copy(update={"status": "paused"}) if r.kind == "goal" else r
                for r in paused.records
            )
        }
    )
    assert scheduler.select(paused, NOW) == ()
    assert scheduler.select_actions(paused, NOW) == (action().id,)


def test_attention_coalescing_defer_archive_receipts_keep_commitments():
    policy = AttentionPolicy(clock=lambda: NOW)
    original = signal()
    view = state(original, frame())
    prepared = policy.prepare(original, view)
    view = apply(view, prepared)
    equivalent = signal(id="9" * 32, producer_id="other-producer")
    assert policy.decide(equivalent, view).disposition == "coalesce"
    assert policy.decide(signal(id="a" * 32, principal_id="b" * 32), view).disposition != "coalesce"
    unscoped = signal(id="d" * 32, scope=Scope())
    view = reduce_event(view, changed(view, unscoped))
    deferred = policy.prepare(unscoped, view)
    receipt = next(
        r
        for r in apply(view, deferred).records
        if isinstance(r, DecisionReceipt) and r.id == deferred.response_receipt.id
    )
    assert receipt.disposition == "defer" and receipt.wake_condition is not None
    archived_signal = signal(id="e" * 32, signal_kind="telemetry")
    view = reduce_event(view, changed(view, archived_signal))
    archived = policy.prepare(archived_signal, view)
    after = apply(view, archived)
    assert archived.response_receipt.disposition == "archive"
    assert next(r for r in after.records if isinstance(r, Commitment)) == next(
        r for r in view.records if isinstance(r, Commitment)
    )
    assert archived.response_receipt.policy_version == "deterministic-v1"
    assert policy.prepare(original, view).events == ()


@pytest.mark.parametrize(
    "kind", ["time", "dependency_change", "new_evidence", "human_decision", "budget"]
)
def test_typed_wakes_only_once_and_no_busy_loop(kind):
    clock = Clock()
    dependency = Belief(
        id="a" * 32,
        created_at=NOW,
        claim="supported",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
    )
    condition = WakeCondition(
        kind=kind,
        at=NOW + timedelta(seconds=60) if kind == "time" else None,
        refs=(Ref(id=dependency.id),) if kind == "dependency_change" else (),
        reason_code="waiting",
    )
    task = frame(status="suspended", wake_condition=condition)
    view = state(task, dependency)
    assert wake(view, None, clock.now) == ()
    clock.advance(60)
    event = None
    if kind == "dependency_change":
        event = changed(view, dependency.model_copy(update={"revision": 2}))
    elif kind in {"new_evidence", "human_decision"}:
        event = changed(
            view, signal(signal_kind="attach_evidence" if kind == "new_evidence" else "reverify")
        )
    elif kind == "budget":
        from signal_gating.executive import Budget, Limits

        budget = Budget(
            id="b" * 32, created_at=NOW, scope=Scope(goal_id=GOAL_ID), limits=Limits(calls=3)
        )
        event = changed(view, budget)
    if event is not None:
        view = reduce_event(view, event)
    events = wake(view, event, clock.now)
    assert events
    for update in events:
        view = reduce_event(view, update)
    assert Scheduler().select(view, clock.now) == (task.id,)
    assert wake(view, event, clock.now) == ()


def test_dependency_wake_refreshes_only_satisfied_current_inputs():
    belief = Belief(
        id="a" * 32,
        created_at=NOW,
        claim="supported",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
        revision=2,
    )
    task = frame(
        status="suspended",
        dependencies=(Ref(id=belief.id),),
        wake_condition=WakeCondition(
            kind="dependency_change",
            refs=(Ref(id=belief.id),),
            reason_code="dependency_invalidated",
        ),
    )
    view = state(task, belief)
    events = wake(view, None, NOW)
    assert events
    for event in events:
        view = reduce_event(view, event)
    resumed = next(r for r in view.records if r.id == task.id)
    assert resumed.dependencies == (Ref(id=belief.id, revision=2),)
    retracted = belief.model_copy(update={"epistemic_status": "retracted"})
    assert wake(state(task, retracted), None, NOW) == ()


def test_repeated_failure_survives_pause_resume_and_material_change(tmp_path):
    clock = Clock()
    scheduler = Scheduler(clock=lambda: clock.now)
    with Store.open(tmp_path / "state") as store:
        view = seed(store, frame())
        store.commit(scheduler.record_failure(view, frame().id, "same_failure", clock.now))
        intake = Intake(store, clock=lambda: clock.now)
        for command in (
            PauseGoal(goal_id=GOAL_ID, expected_revision=1),
            ResumeGoal(goal_id=GOAL_ID, expected_revision=2),
        ):
            store.commit(intake.accept(command, principal=owner(), state=store.view()))
        clock.advance(60)
        view = store.view()
        store.commit(
            CommitBatch(
                events=wake(view, None, clock.now),
                expected_revisions=tuple(Ref(id=r.id, revision=r.revision) for r in view.records),
            )
        )
        store.commit(scheduler.record_failure(store.view(), frame().id, "same_failure", clock.now))
        task = next(r for r in store.view().records if isinstance(r, TaskFrame))
        assert task.status == "blocked" and task.failure_count == 2
        assert scheduler.select(store.view(), clock.now) == ()
        assert wake(store.view(), None, clock.now + timedelta(days=1)) == ()
        store.commit(
            intake.accept(
                ReviseGoal(goal_id=GOAL_ID, expected_revision=3, outcome="Changed outcome"),
                principal=owner(),
                state=store.view(),
            )
        )
        view = store.view()
        awakened = wake(view, None, clock.now)
        assert awakened
        store.commit(
            CommitBatch(
                events=awakened,
                expected_revisions=tuple(Ref(id=r.id, revision=r.revision) for r in view.records),
            )
        )
        assert scheduler.select(store.view(), clock.now) == (task.id,)
        store.commit(scheduler.record_failure(store.view(), task.id, "same_failure", clock.now))
        task = next(r for r in store.view().records if isinstance(r, TaskFrame))
        assert task.failure_count == 1
        assert rebuild(store.events()) == store.view()


def test_control_backlog_overload_is_durable_and_bounded(tmp_path):
    scheduler = Scheduler()
    with Store.open(tmp_path / "state") as store:
        view = seed(store, frame())
        store.commit(scheduler.record_control_backlog(view, 65, NOW))
        view = store.view()
        assert scheduler.select(view, NOW) == ()
        store.commit(scheduler.record_control_backlog(view, 65, NOW + timedelta(seconds=10)))
        view = store.view()
        alerts = [
            r
            for r in view.records
            if isinstance(r, DecisionReceipt) and r.reason_code == "control_overload"
        ]
        assert len(alerts) == 1
        assert scheduler.record_control_backlog(view, 65, NOW + timedelta(seconds=20)).events == ()
        store.commit(scheduler.record_control_backlog(view, 0, NOW + timedelta(seconds=20)))
        assert scheduler.select(store.view(), NOW) == (frame().id,)
        assert scheduler.select_controls(tuple(range(100))) == tuple(range(64))


def test_intake_correction_and_requests_do_not_verify_or_override_pause(tmp_path):
    scheduler = Scheduler()
    with Store.open(tmp_path / "state") as store:
        seed(store, frame())
        intake = Intake(store, clock=lambda: NOW)
        store.commit(
            intake.accept(
                ReviseGoal(goal_id=GOAL_ID, expected_revision=1, outcome="Corrected"),
                principal=owner(),
                state=store.view(),
            )
        )
        view = store.view()
        assert scheduler.select(view, NOW) == ()
        store.commit(
            CommitBatch(
                events=wake(view, None, NOW),
                expected_revisions=tuple(Ref(id=r.id, revision=r.revision) for r in view.records),
            )
        )
        assert scheduler.select(store.view(), NOW) == (frame().id,)
        store.commit(
            intake.accept(
                PauseGoal(goal_id=GOAL_ID, expected_revision=2),
                principal=owner(),
                state=store.view(),
            )
        )
        request = Reverify(goal_id=GOAL_ID)
        store.commit(intake.accept(request, principal=owner(), state=store.view()))
        view = store.view()
        assert wake(view, store.events()[-1], NOW) == ()
        assert scheduler.select(view, NOW) == ()
        typed = next(
            r
            for r in view.records
            if isinstance(r, ExecutiveSignal) and r.signal_kind == "reverify"
        )
        assert AttentionPolicy(clock=lambda: NOW).decide(typed, view).disposition == "process"
        assert not any(r.kind == "verification" for r in view.records)
        assert rebuild(store.events()) == view


def test_real_intake_evidence_wakes_evaluation_after_receipt_event(tmp_path):
    from signal_gating.executive import AttachEvidence

    from .test_memory import artifact

    with Store.open(tmp_path / "state") as store:
        evidence = artifact()
        task = frame(
            status="suspended",
            wake_condition=WakeCondition(kind="new_evidence", reason_code="missing_evidence"),
        )
        seed(store, evidence, task)
        intake = Intake(store, clock=lambda: NOW)
        committed = store.commit(
            intake.accept(
                AttachEvidence(goal_id=GOAL_ID, task_id=task.id, evidence=(Ref(id=evidence.id),)),
                principal=owner(),
                state=store.view(),
            )
        )
        view = store.view()
        assert committed[0].sequence < view.sequence
        events = wake(view, committed[0], NOW)
        assert events
        store.commit(
            CommitBatch(
                events=events,
                expected_revisions=tuple(Ref(id=r.id, revision=r.revision) for r in view.records),
            )
        )
        assert Scheduler().select(store.view(), NOW) == (task.id,)
        assert not any(r.kind == "verification" for r in store.view().records)


def test_deferred_signal_receipt_wakes_after_restart_without_poll_churn(tmp_path):
    with Store.open(tmp_path / "state") as store:
        seed(store, frame(), signal())
        intake = Intake(store, clock=lambda: NOW)
        store.commit(
            intake.accept(
                PauseGoal(goal_id=GOAL_ID, expected_revision=1),
                principal=owner(),
                state=store.view(),
            )
        )
        policy = AttentionPolicy(clock=lambda: NOW)
        batch = policy.prepare(signal(), store.view())
        store.commit(batch)
        receipt_id = batch.response_receipt.id
        assert wake(store.view(), None, NOW) == ()
        resumed = store.commit(
            intake.accept(
                ResumeGoal(goal_id=GOAL_ID, expected_revision=2),
                principal=owner(),
                state=store.view(),
            )
        )
    with Store.open(tmp_path / "state") as store:
        view = store.view()
        events = wake(view, resumed[0], NOW)
        assert events
        store.commit(
            CommitBatch(
                events=events,
                expected_revisions=tuple(Ref(id=r.id, revision=r.revision) for r in view.records),
            )
        )
        receipt = next(r for r in store.view().records if r.id == receipt_id)
        assert receipt.disposition == "process" and receipt.wake_condition is None
        assert wake(store.view(), resumed[0], NOW) == ()
        assert rebuild(store.events()) == store.view()


def test_expired_belief_is_not_runnable_or_awakened():
    belief = Belief(
        id="a" * 32,
        created_at=NOW,
        claim="temporarily supported",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
        valid_until=NOW,
    )
    task = frame(dependencies=(Ref(id=belief.id),))
    assert Scheduler().select(state(task, belief), NOW) == ()
    suspended = task.model_copy(update={"status": "suspended"})
    assert wake(state(suspended, belief), None, NOW) == ()


def test_owner_requests_never_clear_no_progress_blocker(tmp_path):
    with Store.open(tmp_path / "state") as store:
        seed(store, frame())
        scheduler = Scheduler()
        for _ in range(2):
            store.commit(scheduler.record_failure(store.view(), frame().id, "same", NOW))
        intake = Intake(store, clock=lambda: NOW)
        committed = store.commit(
            intake.accept(Reverify(goal_id=GOAL_ID), principal=owner(), state=store.view())
        )
        assert all(wake(store.view(), event, NOW) == () for event in committed)
        assert Scheduler().select(store.view(), NOW) == ()


def test_prepared_selection_and_wake_use_original_cas(tmp_path):
    from signal_gating.executive import ConflictError

    with Store.open(tmp_path / "state") as store:
        view = seed(store, frame())
        batch = Scheduler().prepare_selection(view, NOW)
        intake = Intake(store, clock=lambda: NOW)
        store.commit(
            intake.accept(
                PauseGoal(goal_id=GOAL_ID, expected_revision=1),
                principal=owner(),
                state=store.view(),
            )
        )
        with pytest.raises(ConflictError):
            store.commit(batch)
        assert not any(r.kind == "scheduler" for r in store.view().records)


def test_task7_invalidation_reevaluates_current_dependencies_without_future_event(tmp_path):
    from signal_gating.executive.memory import invalidate_dependents

    belief = Belief(
        id="a" * 32,
        created_at=NOW,
        claim="Checked input",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
    )
    task = frame(dependencies=(Ref(id=belief.id),))
    with Store.open(tmp_path / "state") as store:
        view = seed(store, task, belief)
        revised = belief.model_copy(update={"revision": 2, "claim": "Corrected checked input"})
        event = changed(view, revised)
        after = reduce_event(view, event)
        cascade = invalidate_dependents(after, (Ref(id=belief.id, revision=2),))
        store.commit(
            CommitBatch(
                events=(event, *cascade),
                expected_revisions=tuple(Ref(id=r.id, revision=r.revision) for r in view.records),
            )
        )
        view = store.view()
        assert next(r for r in view.records if r.id == task.id).status == "suspended"
        assert Scheduler().select(view, NOW) == ()
        awakened = wake(view, None, NOW)
        assert awakened
        store.commit(
            CommitBatch(
                events=awakened,
                expected_revisions=tuple(Ref(id=r.id, revision=r.revision) for r in view.records),
            )
        )
        assert Scheduler().select(store.view(), NOW) == (task.id,)
        assert next(r for r in store.view().records if r.id == task.id).dependencies == (
            Ref(id=belief.id, revision=2),
        )
        assert rebuild(store.events()) == store.view()


def test_same_facts_new_revision_does_not_reset_failure_allowance(tmp_path):
    belief = Belief(
        id="a" * 32,
        created_at=NOW,
        claim="Same checked input",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
    )
    task = frame(dependencies=(Ref(id=belief.id),))
    with Store.open(tmp_path / "state") as store:
        seed(store, task, belief)
        scheduler = Scheduler()
        store.commit(scheduler.record_failure(store.view(), task.id, "same", NOW))
        view = store.view()
        store.commit(
            CommitBatch(
                events=(changed(view, belief.model_copy(update={"revision": 2})),),
                expected_revisions=(Ref(id=belief.id),),
            )
        )
        store.commit(scheduler.record_failure(store.view(), task.id, "same", NOW))
        persisted = next(r for r in store.view().records if r.id == task.id)
        assert persisted.status == "blocked" and persisted.failure_count == 2


def test_owner_intake_designates_urgent_work_without_a_deadline(tmp_path):
    from signal_gating.executive import CreateGoal

    with Store.open(tmp_path / "state") as store:
        store.commit(CommitBatch(events=(changed(StateView(), owner()),)))
        commitment = Commitment(
            id=COMMITMENT_ID,
            created_at=NOW,
            goal_id=GOAL_ID,
            responsible_principal_id=OWNER_ID,
        )
        urgent = frame(urgent=True)
        ordinary = frame(id="a" * 32)
        intake = Intake(store, clock=lambda: NOW)
        accepted = intake.accept(
            CreateGoal(goal=goal(), commitments=(commitment,), tasks=(urgent, ordinary)),
            principal=owner(),
            state=store.view(),
        )
        assert accepted.response_receipt.disposition == "accept"
        store.commit(accepted)
        persisted = next(r for r in store.view().records if r.id == urgent.id)
        assert persisted.urgent and persisted.deadline is None
        plan = Scheduler().plan(store.view(), NOW)
        assert [(item.id, item.work_class) for item in plan] == [
            (urgent.id, "urgent"),
            (ordinary.id, "ordinary"),
        ]
        assert rebuild(store.events()) == store.view()


def test_ordinary_enqueue_order_precedes_advisory_task_priority():
    older = frame(id="a" * 32, priority=0, created_at=NOW - timedelta(seconds=20))
    newer = frame(id="b" * 32, priority=999, created_at=NOW - timedelta(seconds=10))
    assert Scheduler().select(state(older, newer), NOW) == (older.id, newer.id)


@pytest.mark.parametrize(
    "support",
    ["criterion_inputs", "criterion_required_evidence", "finding_read_set", "finding_evidence"],
)
@pytest.mark.parametrize("status", ["ready", "suspended"])
def test_criterion_support_expiration_fences_selection_and_wake(tmp_path, support, status):
    from signal_gating.executive import VerificationResult

    belief = Belief(
        id="a" * 32,
        created_at=NOW,
        claim="Temporarily checked support",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
        valid_until=NOW + timedelta(seconds=10),
    )
    support_ref = Ref(id=belief.id, kind="belief")
    criterion = goal().criteria[0]
    if support.startswith("criterion_"):
        field = support.removeprefix("criterion_")
        criterion = criterion.model_copy(update={field: (support_ref,)})
    finding = VerificationResult(
        id="b" * 32,
        created_at=NOW,
        criterion_id=criterion.id,
        verifier_id=criterion.verifier_id,
        disposition="accept",
        reason_code="checked",
        **(
            {support.removeprefix("finding_"): (support_ref,)}
            if support.startswith("finding_")
            else {}
        ),
    )
    task = frame(status=status, dependencies=(Ref(id=criterion.id, kind="criterion"),))
    initial = state(task, belief, finding)
    initial = initial.model_copy(
        update={
            "records": tuple(
                r.model_copy(update={"criteria": (criterion,)}) if r.kind == "goal" else r
                for r in initial.records
            )
        }
    )
    with Store.open(tmp_path / "state") as store:
        store.commit(CommitBatch(events=(changed(StateView(), *initial.records),)))
        view = store.view()
        if status == "ready":
            assert Scheduler().select(view, NOW) == (task.id,)
            assert Scheduler().select(view, NOW + timedelta(seconds=11)) == ()
        else:
            assert wake(view, None, NOW)
            assert wake(view, None, NOW + timedelta(seconds=11)) == ()
        assert rebuild(store.events()) == view
        with Store.open_readonly(store.root) as snapshot:
            assert snapshot.view() == view
