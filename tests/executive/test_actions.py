"""Durable action admission and ambiguous fake effects, without live tools."""

import asyncio
from datetime import timedelta

import pytest

from signal_gating.executive import (
    ActionIntent,
    ActionObservation,
    Budget,
    CapabilitySpec,
    CommitBatch,
    Commitment,
    DecideAction,
    Episode,
    EpisodeResult,
    Limits,
    PauseGoal,
    Scope,
    Usage,
)
from signal_gating.executive.actions import CapabilityRegistry, Dispatcher
from signal_gating.executive.authority import Authority
from signal_gating.executive.budgets import BudgetPolicy
from signal_gating.executive.intake import Intake
from signal_gating.executive.proposals import ProposalValidator
from signal_gating.executive.reducers import rebuild
from signal_gating.executive.store import Store

from .factories import (
    COMMITMENT_ID,
    EPISODE_ID,
    GOAL_ID,
    NOW,
    OWNER_ID,
    TASK_ID,
    frame,
    goal,
    grant,
    owner,
    packet,
)
from .test_memory import event, ref

WORKER = "e" * 32
BUDGET = "f" * 32
ACTION = "b" * 32
QUOTE = Limits(input_tokens=0, output_tokens=0, calls=1, elapsed_seconds=10, cost_microunits=1)


class Fake:
    def __init__(self):
        self.execute_count = 0
        self.lookup_count = 0
        self.effect = False
        self.failure = False
        self.started = asyncio.Event()
        self.finish = None
        self.store = None
        self.observed = []

    async def execute(self, action):
        self.execute_count += 1
        self.observed.append(action)
        assert get(self.store, action.id).status in {"dispatching", "reconciling"}
        assert any(x.action_id == action.id for x in self.store.outbox())
        assert get(self.store, action.reservation_id).subject_id == action.id
        assert not self.store._connection.in_transaction
        self.effect = True
        self.started.set()
        if self.finish is not None:
            try:
                await asyncio.shield(self.finish)
            except asyncio.CancelledError:
                await asyncio.shield(self.finish)
        if self.failure:
            raise ConnectionError("sensitive provider payload")
        return self.observation(action)

    def observation(self, action, **changes):
        return ActionObservation(
            action_id=action.id,
            expected_revision=action.revision,
            outcome="succeeded",
            external_receipt="fake-effect",
            usage=Usage.zero().model_copy(update={"calls": 1, "cost_microunits": 1}),
            **changes,
        )

    async def reconcile(self, action):
        self.lookup_count += 1
        assert not self.store._connection.in_transaction
        return (
            self.observation(action)
            if self.effect
            else ActionObservation(
                action_id=action.id, expected_revision=action.revision, outcome="outcome_unknown"
            )
        )

    async def cancel(self, action):
        return ActionObservation(
            action_id=action.id, expected_revision=action.revision, outcome="outcome_unknown"
        )


@pytest.fixture
def setup(tmp_path, request):
    with Store.open(tmp_path / "store") as store:
        fake = Fake()
        fake.store = store
        registry = CapabilityRegistry()
        spec = CapabilitySpec(
            id="workspace.read",
            binding_version="1",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            effect_class="read_only",
            reconciliation="lookup",
        )
        options = dict(getattr(request, "param", {}))
        support_path = options.pop("_support_path", None)
        constraint = options.pop("_constraint", False)
        spec = spec.model_copy(update=options)

        def register(adapter=fake, capability=spec):
            registry.register(
                capability,
                adapter,
                resource_scope=lambda i: Scope(resource_paths=(i.arguments["path"],)),
                quote=QUOTE,
                quote_source="fixture-prices",
                quote_version="v1",
            )
            return registry.specs["workspace.read"]

        bound = register()
        task = frame(status="running", fencing_token=1)
        target = goal(budget_id=BUDGET)
        support = []
        if support_path:
            from signal_gating.executive import Belief, VerificationResult

            belief = Belief(
                id="d" * 32,
                created_at=NOW,
                claim="current support",
                scope=Scope(goal_id=GOAL_ID),
                epistemic_status="supported",
                valid_until=NOW + timedelta(seconds=5),
            )
            criterion = target.criteria[0]
            if support_path == "criterion_inputs":
                criterion = criterion.model_copy(update={"inputs": (ref(belief),)})
            elif support_path == "criterion_evidence":
                criterion = criterion.model_copy(update={"required_evidence": (ref(belief),)})
            target = target.model_copy(update={"criteria": (criterion,)})
            finding = VerificationResult(
                id="a" * 32,
                created_at=NOW,
                criterion_id=criterion.id,
                verifier_id="fixture",
                disposition="accept",
                reason_code="fixture",
                read_set=(ref(belief),) if support_path == "finding_reads" else (),
                evidence=(ref(belief),) if support_path == "finding_evidence" else (),
            )
            support = [belief, finding]
            if support_path == "finding_artifact":
                from .test_memory import artifact

                missing_blob = artifact().model_copy(
                    update={"id": "8" * 32, "shared_with": (WORKER,)}
                )
                finding = finding.model_copy(update={"evidence": (ref(missing_blob),)})
                support = [belief, finding, missing_blob]
            task = task.model_copy(update={"dependencies": (ref(criterion),)})
        if constraint:
            from signal_gating.executive import Constraint

            target = target.model_copy(
                update={
                    "constraints": (
                        Constraint(
                            id="9" * 32,
                            created_at=NOW,
                            description="hard rule",
                            enforcement="rule",
                            enforcement_id="fixture-rule",
                        ),
                    )
                }
            )
        permission = grant(
            subject_id=WORKER,
            scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID, resource_paths=("data",)),
        )
        context = packet(
            goal=target,
            task=task,
            capabilities=(bound,),
            mandatory_read_set=(
                ref(target),
                ref(task),
                ref(permission),
                *task.dependencies,
                *(ref(c) for c in target.constraints),
            ),
        )
        episode = Episode(
            id=EPISODE_ID,
            task_id=TASK_ID,
            fencing_token=1,
            context_packet_id=context.id,
            worker_id="fixture",
            provider="fake",
            model="fake",
            allowed_proposal_types=("action_intent",),
            deadline=NOW + timedelta(minutes=5),
        )
        budget = Budget(
            id=BUDGET,
            scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
            limits=Limits(
                input_tokens=0, output_tokens=0, calls=10, elapsed_seconds=100, cost_microunits=10
            ),
            actual=Usage.zero(),
            reserved=Usage.zero().model_copy(
                update=dict(
                    input_tokens=0, output_tokens=0, calls=0, elapsed_seconds=0, cost_microunits=0
                )
            ),
        )
        store.commit(
            CommitBatch(
                events=(
                    event(
                        (
                            owner(),
                            owner(id=WORKER, roles=("worker",)),
                            target,
                            Commitment(
                                id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID
                            ),
                            task,
                            permission,
                            context,
                            episode,
                            budget,
                            *support,
                        )
                    ),
                )
            )
        )
        intent = ActionIntent(
            id=ACTION,
            episode_id=EPISODE_ID,
            fencing_token=1,
            read_set=context.mandatory_read_set,
            capability_id="workspace.read",
            arguments={"path": "data/file.txt"},
            scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID, resource_paths=("data/file.txt",)),
        )
        batch = ProposalValidator(
            clock=lambda: NOW, worker_principals={"fixture": WORKER}
        ).validate(
            EpisodeResult(proposals=(intent,), stop_reason="proposed"), episode, store.view()
        )
        assert batch.response_receipt.disposition == "accept"
        store.commit(batch)
        current = [NOW]
        authority = Authority(worker_principals={"fixture": WORKER})
        dispatcher = Dispatcher(
            store,
            authority=authority,
            budgets=BudgetPolicy(clock=lambda: current[0]),
            registry=registry,
            clock=lambda: current[0],
        )
        yield store, fake, registry, dispatcher, register, current


def get(store, identifier):
    return next(r for r in store.view().records if r.id == identifier)


def update(store, record, **changes):
    store.commit(
        CommitBatch(
            events=(
                event((record.model_copy(update={"revision": record.revision + 1, **changes}),)),
            ),
            expected_revisions=(ref(record),),
        )
    )


@pytest.mark.asyncio
async def test_real_proposal_dispatch_reserves_and_settles_atomically(setup):
    store, fake, registry, dispatcher, register, current = setup
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.disposition == "accept"
    assert fake.execute_count == 1
    assert get(store, ACTION).status == "succeeded"
    assert get(store, get(store, ACTION).reservation_id).status == "settled"
    assert store.view().records == rebuild(store.events()).records


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["revoke", "binding", "principal", "digest", "scope", "pause", "expiry"]
)
async def test_current_admission_blocks_before_any_effect(setup, change):
    store, fake, registry, dispatcher, register, current = setup
    action = get(store, ACTION)
    if change == "revoke":
        update(store, get(store, "7" * 32), revoked=True)
    elif change == "binding":
        register()
    elif change == "principal":
        dispatcher.authority = Authority(worker_principals={"fixture": OWNER_ID})
    elif change == "digest":
        update(store, action, arguments={"path": "data/other.txt"})
    elif change == "scope":
        update(store, action, scope=Scope(resource_paths=("elsewhere",)))
    elif change == "pause":
        store.commit(
            Intake(store, clock=lambda: NOW).accept(
                PauseGoal(goal_id=GOAL_ID, expected_revision=1),
                principal=owner(),
                state=store.view(),
            )
        )
    else:
        current[0] = NOW + timedelta(minutes=6)
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.disposition in {"deny", "stale", "require_review"}
    assert fake.execute_count == 0
    assert not store.outbox()


@pytest.mark.asyncio
async def test_disconnect_is_unknown_and_safe_lookup_does_not_repeat(setup):
    store, fake, registry, dispatcher, register, current = setup
    fake.failure = True
    await dispatcher.dispatch_one(ACTION)
    assert get(store, ACTION).status == "outcome_unknown"
    assert get(store, get(store, ACTION).reservation_id).status == "uncertain"
    await dispatcher.reconcile_one(ACTION)
    assert get(store, ACTION).status == "succeeded"
    assert fake.execute_count == 1


@pytest.mark.asyncio
async def test_pause_inflight_preserves_observation_without_resuming_goal(setup):
    store, fake, registry, dispatcher, register, current = setup
    fake.finish = asyncio.get_running_loop().create_future()
    running = asyncio.create_task(dispatcher.dispatch_one(ACTION))
    await fake.started.wait()
    store.commit(
        Intake(store, clock=lambda: NOW).accept(
            PauseGoal(goal_id=GOAL_ID, expected_revision=1), principal=owner(), state=store.view()
        )
    )
    fake.finish.set_result(None)
    await running
    assert get(store, GOAL_ID).status == "paused"
    assert get(store, ACTION).status == "succeeded"
    assert fake.execute_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"effect_class": "consequential"}], indirect=True)
async def test_no_owner_approval_stays_pending_and_digest_mismatch_is_denied(setup):
    store, fake, registry, dispatcher, register, current = setup
    assert (await dispatcher.dispatch_one(ACTION)).disposition == "require_review"
    assert (await dispatcher.dispatch_one(ACTION)).disposition == "require_review"
    action = get(store, ACTION)
    assert action.status == "awaiting_review"
    command = DecideAction(
        action_id=ACTION,
        expected_revision=action.revision,
        decision="approve",
        arguments_digest="0" * 64,
        scope=action.scope,
        constraint_revision=action.constraint_revision,
    )
    batch = Intake(store, clock=lambda: current[0]).accept(
        command, principal=owner(), state=store.view()
    )
    assert batch.response_receipt.disposition == "stale"
    store.commit(batch)
    assert get(store, ACTION).status == "awaiting_review"
    current[0] += timedelta(minutes=6)
    await dispatcher.dispatch_one(ACTION)
    assert fake.execute_count == 0
    assert not store.outbox()


def approve(store):
    action = get(store, ACTION)
    command = DecideAction(
        action_id=ACTION,
        expected_revision=action.revision,
        decision="approve",
        arguments_digest=action.arguments_digest,
        scope=action.scope,
        constraint_revision=action.constraint_revision,
    )
    batch = Intake(store, clock=lambda: NOW).accept(command, principal=owner(), state=store.view())
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"effect_class": "consequential"}], indirect=True)
async def test_owner_approval_allows_only_exact_reviewed_arguments(setup):
    store, fake, registry, dispatcher, register, current = setup
    await dispatcher.dispatch_one(ACTION)
    approve(store)
    import hashlib
    import json

    args = {"path": "data/different.txt"}
    digest = hashlib.sha256(
        json.dumps(args, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    update(
        store,
        get(store, ACTION),
        arguments=args,
        arguments_digest=digest,
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID, resource_paths=("data/different.txt",)),
    )
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.disposition in {"stale", "require_review", "deny"}
    assert fake.execute_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"effect_class": "consequential"}], indirect=True)
async def test_owner_approval_does_not_bypass_current_grant(setup):
    store, fake, registry, dispatcher, register, current = setup
    await dispatcher.dispatch_one(ACTION)
    approve(store)
    update(store, get(store, "7" * 32), revoked=True)
    await dispatcher.dispatch_one(ACTION)
    assert fake.execute_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"effect_class": "consequential"}], indirect=True)
async def test_exact_owner_approval_can_dispatch(setup):
    store, fake, registry, dispatcher, register, current = setup
    await dispatcher.dispatch_one(ACTION)
    approve(store)
    receipt = await dispatcher.dispatch_one(ACTION)
    assert fake.execute_count == 1, receipt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        {
            "reconciliation": "none",
            "idempotency": "durable",
            "idempotency_scope": "fake-tenant-action-key",
            "idempotency_retention_seconds": 20,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("expired", [False, True])
async def test_durable_retry_requires_scope_and_unexpired_window(setup, expired):
    store, fake, registry, dispatcher, register, current = setup
    fake.failure = True
    await dispatcher.dispatch_one(ACTION)
    current[0] += timedelta(seconds=21 if expired else 1)
    fake.failure = False
    await dispatcher.reconcile_one(ACTION)
    assert fake.execute_count == (1 if expired else 2)
    assert get(store, ACTION).status == ("needs_human" if expired else "succeeded")
    if not expired:
        first, second = fake.observed
        assert first.idempotency_key == second.idempotency_key
        assert first.id == second.id
        assert first.reservation_id != second.reservation_id


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"reconciliation": "none"}], indirect=True)
async def test_read_only_with_no_durable_scope_is_never_blindly_retried(setup):
    store, fake, registry, dispatcher, register, current = setup
    fake.failure = True
    await dispatcher.dispatch_one(ACTION)
    await dispatcher.reconcile_one(ACTION)
    assert get(store, ACTION).status == "needs_human"
    assert fake.execute_count == 1


@pytest.mark.asyncio
async def test_caller_cancellation_is_unknown_and_late_completion_is_bounded(setup):
    store, fake, registry, dispatcher, register, current = setup
    fake.finish = asyncio.get_running_loop().create_future()
    run = asyncio.create_task(dispatcher.dispatch_one(ACTION))
    await fake.started.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert get(store, ACTION).status == "outcome_unknown"
    assert get(store, get(store, ACTION).reservation_id).status == "uncertain"
    await dispatcher.dispatch_one(ACTION)
    await dispatcher.reconcile_one(ACTION)
    assert fake.execute_count == 1 and fake.lookup_count == 0
    fake.finish.set_result(None)
    await dispatcher.late_observations_ready.wait()
    late = dispatcher.take_late_observations()
    assert len(late) == 1
    issued, observation = late[0]
    dispatcher.record_observation(issued, observation)
    assert get(store, ACTION).status == "succeeded"
    assert get(store, issued.reservation_id).status == "settled"
    await dispatcher.close()


@pytest.mark.asyncio
async def test_committed_dispatch_crash_reopens_and_looks_up_with_restored_binding(
    setup, monkeypatch
):
    store, fake, registry, dispatcher, register, current = setup
    original = dispatcher.record_observation

    def crash(*args):
        raise RuntimeError("injected crash before receipt")

    monkeypatch.setattr(dispatcher, "record_observation", crash)
    with pytest.raises(RuntimeError, match="injected crash"):
        await dispatcher.dispatch_one(ACTION)
    assert get(store, ACTION).status == "dispatching"
    root = store.root
    trusted_spec = registry.specs["workspace.read"]
    store.close()
    with Store.open(root) as restored:
        fake.store = restored
        rebuilt_registry = CapabilityRegistry()
        rebuilt_registry.register(
            trusted_spec,
            fake,
            resource_scope=lambda i: Scope(resource_paths=(i.arguments["path"],)),
            quote=QUOTE,
            quote_source="fixture-prices",
            quote_version="v1",
        )
        recovered = Dispatcher(
            restored,
            authority=Authority(worker_principals={"fixture": WORKER}),
            budgets=BudgetPolicy(clock=lambda: NOW),
            registry=rebuilt_registry,
            clock=lambda: NOW,
        )
        assert get(restored, ACTION).status == "dispatching"
        await recovered.reconcile_one(ACTION)
        assert get(restored, ACTION).status == "succeeded"
        assert fake.execute_count == 1
        assert fake.lookup_count == 1
        assert restored.view().records == rebuild(restored.events()).records
    monkeypatch.setattr(dispatcher, "record_observation", original)


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["revoke", "pause", "overage", "binding"])
async def test_rechecks_between_authorization_and_durable_dispatch(setup, monkeypatch, race):
    store, fake, registry, dispatcher, register, current = setup
    original = dispatcher._commit

    def commit(action, disposition, reason, **kwargs):
        receipt = original(action, disposition, reason, **kwargs)
        if reason == "action_authorized":
            if race == "revoke":
                update(store, get(store, "7" * 32), revoked=True)
            elif race == "pause":
                store.commit(
                    Intake(store, clock=lambda: NOW).accept(
                        PauseGoal(goal_id=GOAL_ID, expected_revision=1),
                        principal=owner(),
                        state=store.view(),
                    )
                )
            elif race == "binding":
                register()
            else:
                policy = dispatcher.budgets
                store.commit(
                    policy.reserve(
                        store.view(),
                        BUDGET,
                        QUOTE,
                        "9" * 32,
                        subject_id=GOAL_ID,
                        trusted_maximum_quote=True,
                    )
                )
                store.commit(
                    policy.settle(
                        store.view(),
                        "9" * 32,
                        Usage.zero().model_copy(update={"cost_microunits": 2}),
                    )
                )
        return receipt

    monkeypatch.setattr(dispatcher, "_commit", commit)
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.disposition in {"deny", "stale"}
    assert fake.execute_count == 0
    assert get(store, ACTION).status == "cancelled"
    assert get(store, get(store, ACTION).reservation_id).subject_id == ACTION


@pytest.mark.asyncio
async def test_authorization_outbox_failure_rolls_back_reservation(setup):
    store, fake, registry, dispatcher, register, current = setup
    before = store.view()
    store._connection.execute(
        "CREATE TEMP TRIGGER reject_outbox BEFORE INSERT ON outbox "
        "BEGIN SELECT RAISE(ABORT, 'injected outbox failure'); END"
    )
    with pytest.raises(Exception, match="Batch identity or reference conflicts"):
        await dispatcher.dispatch_one(ACTION)
    assert store.view() == before
    assert not store.outbox()
    assert fake.execute_count == 0


@pytest.mark.asyncio
async def test_committed_dispatch_before_call_recovers_unknown_without_repeating(
    setup, monkeypatch
):
    store, fake, registry, dispatcher, register, current = setup
    original = dispatcher._commit

    def commit(action, disposition, reason, **kwargs):
        result = original(action, disposition, reason, **kwargs)
        if reason == "action_dispatching":
            raise RuntimeError("injected crash after dispatch commit")
        return result

    monkeypatch.setattr(dispatcher, "_commit", commit)
    with pytest.raises(RuntimeError, match="injected crash"):
        await dispatcher.dispatch_one(ACTION)
    assert get(store, ACTION).status == "dispatching"
    assert fake.execute_count == 0
    monkeypatch.setattr(dispatcher, "_commit", original)
    await dispatcher.reconcile_one(ACTION)
    assert get(store, ACTION).status == "needs_human"
    assert fake.execute_count == 0 and fake.lookup_count == 1
    assert get(store, get(store, ACTION).prior_reservation_ids[0]).status == "uncertain"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "close"])
async def test_timeout_and_close_do_not_wait_for_cancellation_resistant_effect(setup, mode):
    store, fake, registry, dispatcher, register, current = setup
    timer = asyncio.get_running_loop().create_future()

    async def wait_until(deadline):
        await timer

    dispatcher.wait_until = wait_until
    fake.finish = asyncio.get_running_loop().create_future()
    run = asyncio.create_task(dispatcher.dispatch_one(ACTION))
    await fake.started.wait()
    if mode == "timeout":
        timer.set_result(None)
    else:
        await dispatcher.close()
    await run
    assert get(store, ACTION).status == "outcome_unknown"
    assert not fake.finish.done()
    assert len(dispatcher._pending) == 1
    fake.finish.set_result(None)
    await dispatcher.late_observations_ready.wait()
    issued, observation = dispatcher.take_late_observations()[0]
    dispatcher.record_observation(issued, observation)
    assert get(store, ACTION).status == "succeeded"


@pytest.mark.asyncio
async def test_duplicate_dispatch_is_bounded_while_call_runs(setup):
    store, fake, registry, dispatcher, register, current = setup
    fake.finish = asyncio.get_running_loop().create_future()
    run = asyncio.create_task(dispatcher.dispatch_one(ACTION))
    await fake.started.wait()
    for _ in range(3):
        assert (await dispatcher.dispatch_one(ACTION)).disposition == "defer"
        assert (await dispatcher.reconcile_one(ACTION)).disposition == "defer"
    assert fake.execute_count == 1
    fake.finish.set_result(None)
    await run


@pytest.mark.asyncio
async def test_overage_is_recorded_and_future_admission_fails(setup, monkeypatch):
    store, fake, registry, dispatcher, register, current = setup
    original = fake.observation

    def observation(action):
        return original(action).model_copy(
            update={"usage": Usage.zero().model_copy(update={"calls": 3, "cost_microunits": 3})}
        )

    monkeypatch.setattr(fake, "observation", observation)
    await dispatcher.dispatch_one(ACTION)
    assert get(store, BUDGET).actual.cost_microunits == 3
    assert get(store, BUDGET).accounting_uncertain
    assert any(e.payload.get("reason_code") == "adapter_bound_exceeded" for e in store.events())
    second = get(store, ACTION).model_copy(
        update={
            "id": "a" * 32,
            "revision": 1,
            "status": "proposed",
            "idempotency_key": "second",
            "reservation_id": None,
            "dispatched_at": None,
            "dispatch_attempts": 0,
        }
    )
    store.commit(CommitBatch(events=(event((second,)),)))
    receipt = await dispatcher.dispatch_one(second.id)
    assert receipt.reason_code == "adapter_bound_exceeded"
    assert fake.execute_count == 1


@pytest.mark.asyncio
async def test_unknown_usage_is_never_released_on_success(setup, monkeypatch):
    store, fake, registry, dispatcher, register, current = setup
    original = fake.observation
    monkeypatch.setattr(
        fake,
        "observation",
        lambda action: original(action).model_copy(update={"usage": Usage(calls=1)}),
    )
    await dispatcher.dispatch_one(ACTION)
    assert get(store, ACTION).status == "succeeded"
    reservation = get(store, get(store, ACTION).reservation_id)
    assert reservation.status == "uncertain"
    assert get(store, BUDGET).reserved.cost_microunits == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pause", [False, True])
async def test_safe_lookup_is_separately_reserved_and_preserves_control(setup, pause):
    from signal_gating.executive import CancelGoal

    store, fake, registry, dispatcher, register, current = setup
    fake.failure = True
    await dispatcher.dispatch_one(ACTION)
    prior_reservation = get(store, ACTION).reservation_id
    command = (
        PauseGoal(goal_id=GOAL_ID, expected_revision=1)
        if pause
        else CancelGoal(goal_id=GOAL_ID, expected_revision=1)
    )
    store.commit(
        Intake(store, clock=lambda: NOW).accept(command, principal=owner(), state=store.view())
    )
    await dispatcher.reconcile_one(ACTION)
    assert fake.lookup_count == 1 and fake.execute_count == 1
    assert get(store, GOAL_ID).status == ("paused" if pause else "cancelled")
    assert get(store, ACTION).status == "succeeded"
    assert get(store, ACTION).reservation_id != prior_reservation
    assert get(store, prior_reservation).status == "uncertain"
    assert get(store, get(store, ACTION).reservation_id).status == "settled"


@pytest.mark.asyncio
async def test_same_trusted_restored_version_replacement_rotates_identity(setup):
    store, fake, registry, dispatcher, register, current = setup
    trusted_spec = registry.specs["workspace.read"]
    registry.register(
        trusted_spec,
        fake,
        resource_scope=lambda i: Scope(resource_paths=(i.arguments["path"],)),
        quote=QUOTE,
        quote_source="fixture",
        quote_version="v1",
    )
    assert registry.specs["workspace.read"].binding_id != trusted_spec.binding_id
    await dispatcher.dispatch_one(ACTION)
    assert fake.execute_count == 0


@pytest.mark.asyncio
async def test_invalid_schema_arguments_never_reach_resource_extractor(setup):
    import hashlib
    import json

    store, fake, registry, dispatcher, register, current = setup
    args = {"path": 1}
    digest = hashlib.sha256(
        json.dumps(args, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    update(store, get(store, ACTION), arguments=args, arguments_digest=digest)
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.reason_code == "invalid_action_arguments"
    assert fake.execute_count == 0


@pytest.mark.asyncio
async def test_exception_payload_is_not_in_journal(setup):
    store, fake, registry, dispatcher, register, current = setup
    fake.failure = True
    await dispatcher.dispatch_one(ACTION)
    assert "sensitive provider payload" not in repr(store.events())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        {
            "reconciliation": "none",
            "idempotency": "durable",
            "idempotency_scope": "fake-tenant-action-key",
            "idempotency_retention_seconds": 20,
        }
    ],
    indirect=True,
)
async def test_old_attempt_late_usage_settles_original_reservation_only(setup):
    store, fake, registry, dispatcher, register, current = setup
    fake.failure = True
    await dispatcher.dispatch_one(ACTION)
    first = fake.observed[0]
    await dispatcher.reconcile_one(ACTION)
    second = fake.observed[1]
    assert get(store, ACTION).status == "needs_human"
    late = ActionObservation(
        action_id=first.id,
        expected_revision=first.revision,
        outcome="failed",
        usage=Usage(status="unknown", calls=2, cost_microunits=3),
    )
    dispatcher.record_observation(first, late)
    assert get(store, ACTION).status == "needs_human"
    assert get(store, first.reservation_id).usage.cost_microunits == 3
    assert get(store, second.reservation_id).usage.cost_microunits is None
    assert get(store, second.reservation_id).status == "uncertain"
    assert get(store, BUDGET).actual.cost_microunits == 3
    assert get(store, BUDGET).reserved.cost_microunits == 1
    assert fake.execute_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        {"_support_path": "criterion_inputs"},
        {"_support_path": "criterion_evidence"},
        {"_support_path": "finding_reads"},
        {"_support_path": "finding_evidence"},
    ],
    indirect=True,
)
async def test_transitive_expired_support_blocks_without_revision_change(setup):
    store, fake, registry, dispatcher, register, current = setup
    before = store.view()
    current[0] += timedelta(seconds=6)
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.reason_code == "expired_action_support"
    assert fake.execute_count == 0
    assert get(store, "d" * 32).revision == get_from_view(before, "d" * 32).revision


def get_from_view(state, identifier):
    return next(r for r in state.records if r.id == identifier)


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"_support_path": "criterion_inputs"}], indirect=True)
async def test_current_transitive_support_is_dispatchable(setup):
    store, fake, registry, dispatcher, register, current = setup
    assert (await dispatcher.dispatch_one(ACTION)).disposition == "accept"
    assert fake.execute_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"_constraint": True}], indirect=True)
async def test_current_hard_rule_is_rechecked_and_model_payload_cannot_override(setup):
    from signal_gating.executive import PolicyDecision
    from signal_gating.executive.authority import RuleRegistry

    store, fake, registry, dispatcher, register, current = setup
    verdict = [False]
    rules = RuleRegistry(
        {
            "fixture-rule": lambda inputs: PolicyDecision(
                gate="execution",
                disposition="authorize" if verdict[0] else "require_review",
                reason_code="trusted-rule",
            )
        }
    )
    dispatcher.authority = Authority(rules=rules, worker_principals={"fixture": WORKER})
    assert (await dispatcher.dispatch_one(ACTION)).reason_code == "trusted-rule"
    assert fake.execute_count == 0
    verdict[0] = True
    await dispatcher.dispatch_one(ACTION)
    assert fake.execute_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"_constraint": True}], indirect=True)
async def test_rule_conflict_blocks_goal_and_retains_exact_constraint_evidence(setup):
    from signal_gating.executive import PolicyDecision
    from signal_gating.executive.authority import RuleRegistry
    from signal_gating.executive.intake import Intake

    store, fake, registry, dispatcher, register, current = setup
    constraint = get(store, GOAL_ID).constraints[0]
    dispatcher.authority = Authority(
        worker_principals={"fixture": WORKER},
        rules=RuleRegistry(
            {
                "fixture-rule": lambda inputs: PolicyDecision(
                    gate="execution",
                    disposition="deny",
                    reason_code="constraint_conflict",
                    evidence=(ref(constraint),),
                )
            }
        ),
    )
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.evidence == (ref(constraint),)
    assert receipt.state_revisions == (ref(goal()), ref(constraint))
    assert get(store, GOAL_ID).status == "blocked"
    assert get(store, TASK_ID).status == "blocked"
    assert get(store, TASK_ID).wake_condition.refs == (ref(constraint),)
    assert Intake._conflict(get(store, GOAL_ID), store.view()) == {ref(constraint)}
    assert fake.execute_count == 0


@pytest.mark.asyncio
async def test_missing_quote_or_extractor_cannot_authorize(setup):
    store, fake, registry, dispatcher, register, current = setup
    trusted_spec = registry.specs["workspace.read"]
    for options in (
        {"resource_scope": lambda i: Scope(resource_paths=(i.arguments["path"],))},
        {"quote": QUOTE, "quote_source": "fixture", "quote_version": "v1"},
    ):
        rebuilt = CapabilityRegistry()
        rebuilt.register(trusted_spec, fake, **options)
        dispatcher.registry = rebuilt
        receipt = await dispatcher.dispatch_one(ACTION)
        assert receipt.disposition == "deny" or get(store, ACTION).status == "denied"
        assert fake.execute_count == 0


def test_human_only_and_unsupported_schema_registration_fail_closed():
    registry = CapabilityRegistry()
    base = CapabilitySpec(
        id="fixture", binding_version="v1", input_schema={}, effect_class="human_only"
    )
    with pytest.raises(ValueError, match="Human-only"):
        registry.register(base, Fake())
    with pytest.raises(ValueError, match="Unsupported"):
        registry.register(
            base.model_copy(update={"effect_class": "read_only", "input_schema": {"pattern": "x"}}),
            Fake(),
        )


@pytest.mark.asyncio
async def test_accepted_required_action_remains_task_blocker_until_settled(setup):
    from signal_gating.executive.scheduler import wake

    store, fake, registry, dispatcher, register, current = setup
    task = get(store, TASK_ID)
    assert task.pending_action_ids == (ACTION,)
    assert not wake(store.view(), None, NOW)
    await dispatcher.dispatch_one(ACTION)
    assert wake(store.view(), None, NOW)


@pytest.mark.asyncio
async def test_legacy_proposal_without_trusted_principal_cannot_dispatch(setup):
    store, fake, registry, dispatcher, register, current = setup
    update(store, get(store, ACTION), origin_principal_id=None)
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.reason_code == "action_principal_binding_changed"
    assert fake.execute_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"_support_path": "finding_artifact"}], indirect=True)
async def test_finding_only_artifact_bytes_are_rechecked_before_dispatch(setup):
    store, fake, registry, dispatcher, register, current = setup
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.reason_code == "inaccessible_action_artifact"
    assert fake.execute_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        {
            "input_schema": {
                "type": "object",
                "properties": {"path": {"enum": [1]}},
                "required": ["path"],
            }
        }
    ],
    indirect=True,
)
async def test_schema_enum_does_not_confuse_boolean_with_integer(setup):
    import hashlib

    store, fake, registry, dispatcher, register, current = setup
    update(
        store,
        get(store, ACTION),
        arguments={"path": True},
        arguments_digest=hashlib.sha256(b'{"path":true}').hexdigest(),
    )
    receipt = await dispatcher.dispatch_one(ACTION)
    assert receipt.reason_code == "invalid_action_arguments"
    assert fake.execute_count == 0
