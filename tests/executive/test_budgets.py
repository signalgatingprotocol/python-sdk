"""Hierarchical accounting uses the real journal, reducer, CAS and intake."""

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from signal_gating.executive import (
    Budget,
    BudgetError,
    CommitBatch,
    Commitment,
    ConflictError,
    JournalEvent,
    Limits,
    Ref,
    Reservation,
    ResolveUsage,
    Scope,
    Usage,
)
from signal_gating.executive.budgets import BudgetPolicy
from signal_gating.executive.intake import Intake
from signal_gating.executive.reducers import rebuild
from signal_gating.executive.store import Store

from .factories import COMMITMENT_ID, GOAL_ID, NOW, OWNER_ID, action, frame, goal, owner

ROOT = "a" * 32
LEFT = "b" * 32
RIGHT = "c" * 32


def maxima(**changes):
    values = dict(input_tokens=0, output_tokens=0, calls=0, elapsed_seconds=0, cost_microunits=0)
    values.update(changes)
    return Limits(**values)


def known(**changes):
    values = Usage.zero().model_dump()
    values.update(changes)
    return Usage(**values)


def changed(records):
    return JournalEvent(
        created_at=NOW,
        sequence=1,
        event_type="state.changed",
        producer_id="fixture",
        principal_id=OWNER_ID,
        subject=Ref(id=ROOT),
        payload={"records": [r.model_dump(mode="json") for r in records]},
    )


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "store") as current:
        budgets = tuple(
            Budget(
                id=identifier,
                created_at=NOW,
                scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
                limits=maxima(calls=10, cost_microunits=100),
                parent_id=parent,
            )
            for identifier, parent in ((ROOT, None), (LEFT, ROOT), (RIGHT, ROOT))
        )
        current.commit(CommitBatch(events=(changed((owner(), goal(), *budgets)),)))
        yield current


def get(store, identifier):
    return next(r for r in store.view().records if r.id == identifier)


def reserve(store, budget=LEFT, calls=6, **options):
    batch = BudgetPolicy(clock=lambda: NOW).reserve(
        store.view(),
        budget,
        maxima(calls=calls),
        uuid4().hex,
        trusted_maximum_quote=True,
        **options,
    )
    store.commit(batch)
    return next(
        r
        for r in store.view().records
        if isinstance(r, Reservation) and r.id == batch.events[0].subject.id
    )


def test_concurrent_siblings_cas_prevents_oversubscription(store):
    before = store.view()
    policy = BudgetPolicy(clock=lambda: NOW)
    with ThreadPoolExecutor(max_workers=2) as workers:
        batches = list(
            workers.map(
                lambda budget: policy.reserve(
                    before, budget, maxima(calls=6), uuid4().hex, trusted_maximum_quote=True
                ),
                (LEFT, RIGHT),
            )
        )
    accepted = 0
    for batch in batches:
        try:
            store.commit(batch)
            accepted += 1
        except ConflictError as error:
            assert error.code == "stale_revision"
    assert accepted == 1
    assert get(store, ROOT).reserved.calls == 6
    with pytest.raises(BudgetError, match="calls"):
        policy.reserve(
            store.view(), RIGHT, maxima(calls=6), uuid4().hex, trusted_maximum_quote=True
        )
    assert rebuild(store.events()) == store.view()


def test_settlement_releases_unused_bound_and_is_idempotent(store):
    reservation = reserve(store)
    policy = BudgetPolicy(clock=lambda: NOW)
    store.commit(policy.settle(store.view(), reservation.id, known(calls=3)))
    settled = get(store, reservation.id)
    assert settled.status == "settled"
    for identifier in (ROOT, LEFT):
        assert get(store, identifier).reserved.calls == 0
        assert get(store, identifier).actual.calls == 3
    before = store.view()
    store.commit(policy.settle(before, reservation.id, known(calls=3)))
    assert store.view().records == before.records
    with pytest.raises(BudgetError):
        policy.settle(store.view(), reservation.id, known(calls=4))
    reserve(store, budget=RIGHT, calls=7)


def test_unknown_usage_retains_entire_reservation_across_restart(store):
    reservation = reserve(store)
    policy = BudgetPolicy(clock=lambda: NOW)
    store.commit(policy.mark_unknown(store.view(), reservation.id))
    assert get(store, reservation.id).status == "uncertain"
    assert get(store, ROOT).reserved.calls == 6
    assert get(store, ROOT).accounting_uncertain
    with Store.open_readonly(store.root) as snapshot:
        assert get(snapshot, reservation.id).status != "released"
        with pytest.raises(BudgetError):
            policy.reserve(
                snapshot.view(), RIGHT, maxima(calls=5), uuid4().hex, trusted_maximum_quote=True
            )


def test_partial_usage_reconciles_reported_quantities_and_retains_remainder(store):
    reservation = reserve(store)
    policy = BudgetPolicy(clock=lambda: NOW)
    partial = Usage(status="unknown", calls=2)
    store.commit(policy.settle(store.view(), reservation.id, partial))
    assert get(store, reservation.id).status == "uncertain"
    assert get(store, ROOT).actual.calls == 2
    assert get(store, ROOT).reserved.calls == 4
    assert get(store, ROOT).reserved.cost_microunits == 0
    assert get(store, ROOT).accounting_uncertain
    store.commit(policy.settle(store.view(), reservation.id, known(calls=2)))
    assert get(store, ROOT).actual.calls == 2
    assert not get(store, ROOT).accounting_uncertain


def test_estimated_usage_retains_unspent_maximum(store):
    reservation = reserve(store)
    store.commit(
        BudgetPolicy().settle(store.view(), reservation.id, known(status="estimated", calls=2))
    )
    assert get(store, ROOT).actual.calls == 2
    assert get(store, ROOT).reserved.calls == 4
    assert get(store, reservation.id).status == "uncertain"


@pytest.mark.parametrize("operation", ("reserve", "settle"))
def test_currency_mismatch_fails_without_mutation(store, operation):
    reservation = reserve(store)
    before = store.view()
    with pytest.raises(BudgetError, match="currency"):
        if operation == "reserve":
            BudgetPolicy().reserve(
                before, LEFT, maxima(currency="EUR"), uuid4().hex, trusted_maximum_quote=True
            )
        else:
            BudgetPolicy().settle(before, reservation.id, known(currency="EUR"))
    assert store.view() == before


@pytest.mark.parametrize("limits,trusted", ((Limits(calls=1), True), (maxima(calls=1), False)))
def test_strict_money_requires_trusted_finite_maximum(store, limits, trusted):
    with pytest.raises(BudgetError):
        BudgetPolicy().reserve(
            store.view(), LEFT, limits, uuid4().hex, trusted_maximum_quote=trusted
        )


def test_unbounded_request_cannot_escape_finite_ancestor(store):
    with pytest.raises(BudgetError):
        BudgetPolicy(strict_money=False).reserve(store.view(), LEFT, Limits(calls=1), uuid4().hex)


def test_actual_overage_is_recorded_and_blocks_even_with_remaining_budget(store):
    reservation = reserve(store, calls=2)
    pending = reserve(store, budget=RIGHT, calls=1)
    policy = BudgetPolicy()
    store.commit(policy.settle(store.view(), reservation.id, known(calls=3)))
    assert get(store, ROOT).actual.calls == 3
    assert get(store, ROOT).reserved.calls == 1
    assert get(store, ROOT).accounting_uncertain
    with pytest.raises(BudgetError, match="bound"):
        policy.reserve(store.view(), RIGHT, maxima(calls=1), pending.id, trusted_maximum_quote=True)
    with pytest.raises(BudgetError, match="bound"):
        policy.reserve(
            store.view(), RIGHT, maxima(calls=1), uuid4().hex, trusted_maximum_quote=True
        )


def test_reservation_and_action_authorization_commit_atomically(store):
    permission = action(id="d" * 32)
    commitment = Commitment(
        id=COMMITMENT_ID, created_at=NOW, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID
    )
    store.commit(CommitBatch(events=(changed((commitment, frame(), permission)),)))
    before = store.view()
    reservation_id = uuid4().hex
    batch = BudgetPolicy().reserve(
        before,
        LEFT,
        maxima(calls=6),
        reservation_id,
        subject_id=permission.id,
        trusted_maximum_quote=True,
    )
    authorized = permission.model_copy(
        update={"revision": 2, "status": "authorized", "reservation_id": reservation_id}
    )
    event = batch.events[0]
    combined = event.model_copy(
        update={
            "payload": {
                "records": [
                    *event.model_dump(mode="json")["payload"]["records"],
                    authorized.model_dump(mode="json"),
                ]
            }
        }
    )
    combined_batch = CommitBatch(
        events=(combined,),
        expected_revisions=(*batch.expected_revisions, Ref(id=permission.id, revision=1)),
    )
    store.commit(combined_batch)
    assert get(store, permission.id).status == "authorized"
    assert get(store, reservation_id).subject_id == permission.id
    assert get(store, ROOT).reserved.calls == 6
    assert rebuild(store.events()) == store.view()


def test_real_owner_resolution_callback_settles_ancestors_atomically(store):
    reservation = reserve(store)
    policy = BudgetPolicy(clock=lambda: NOW)
    store.commit(policy.mark_unknown(store.view(), reservation.id))
    command = ResolveUsage(
        reservation_id=reservation.id,
        expected_revision=2,
        accept_reserved_maximum=True,
        explanation="Charge the maximum",
    )
    intake = Intake(store, usage_resolver=policy.resolve_usage)
    batch = intake.accept(command, principal=owner(), state=store.view())
    assert batch.response_receipt.disposition == "accept"
    assert get(store, reservation.id).status == "uncertain"
    store.commit(batch)
    assert get(store, reservation.id).status == "settled"
    assert get(store, ROOT).actual.calls == 6
    assert get(store, ROOT).reserved.calls == 0
    assert not get(store, ROOT).accounting_uncertain
    assert rebuild(store.events()) == store.view()
    with Store.open_readonly(store.root) as snapshot:
        assert snapshot.view() == store.view()


def test_unknown_money_stays_unknown_without_finite_ceilings(store):
    unlimited = Budget(
        id="d" * 32, scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID), limits=Limits()
    )
    store.commit(CommitBatch(events=(changed((unlimited,)),)))
    policy = BudgetPolicy(strict_money=False)
    identifier = uuid4().hex
    store.commit(policy.reserve(store.view(), unlimited.id, Limits(calls=1), identifier))
    assert get(store, unlimited.id).reserved.cost_microunits is None
    assert get(store, unlimited.id).accounting_uncertain
    store.commit(policy.mark_unknown(store.view(), identifier))
    command = ResolveUsage(
        reservation_id=identifier,
        expected_revision=2,
        accept_reserved_maximum=True,
        explanation="Use full maximum",
    )
    with pytest.raises(BudgetError, match="unknown maximum"):
        policy.resolve_usage(command, store.view())
    store.commit(policy.settle(store.view(), identifier, known(calls=1, cost_microunits=9)))
    assert get(store, unlimited.id).actual.cost_microunits == 9
    assert get(store, unlimited.id).reserved.cost_microunits == 0
    assert not get(store, unlimited.id).accounting_uncertain


def test_partial_settlement_keeps_other_sibling_hold(store):
    first = reserve(store, calls=3)
    second = reserve(store, budget=RIGHT, calls=4)
    policy = BudgetPolicy()
    store.commit(policy.mark_unknown(store.view(), first.id))
    store.commit(policy.settle(store.view(), second.id, known(calls=2)))
    assert get(store, ROOT).reserved.calls == 3
    assert get(store, ROOT).actual.calls == 2
    assert get(store, ROOT).accounting_uncertain
    store.commit(policy.settle(store.view(), first.id, known(calls=1)))
    assert get(store, ROOT).reserved.calls == 0
    assert get(store, ROOT).actual.calls == 3
    assert not get(store, ROOT).accounting_uncertain


def test_duplicate_reservation_id_cannot_rebind_subject_or_maximum(store):
    reservation = reserve(store)
    policy = BudgetPolicy()
    before = store.view()
    store.commit(
        policy.reserve(before, LEFT, maxima(calls=6), reservation.id, trusted_maximum_quote=True)
    )
    assert store.view().records == before.records
    with pytest.raises(BudgetError, match="different binding"):
        policy.reserve(
            store.view(),
            LEFT,
            maxima(calls=6),
            reservation.id,
            subject_id=GOAL_ID,
            trusted_maximum_quote=True,
        )


def test_stale_settlement_rolls_back_all_accounting(store):
    reservation = reserve(store)
    policy = BudgetPolicy()
    stale = policy.settle(store.view(), reservation.id, known(calls=2))
    store.commit(policy.mark_unknown(store.view(), reservation.id))
    before = store.view()
    with pytest.raises(ConflictError):
        store.commit(stale)
    assert store.view() == before


def test_monetary_accounting_stays_exact_above_float_integer_precision(store):
    maximum = 2**60 + 7
    large = Budget(
        id="d" * 32,
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
        limits=maxima(calls=2, cost_microunits=maximum),
    )
    store.commit(CommitBatch(events=(changed((large,)),)))
    policy = BudgetPolicy()
    identifier = uuid4().hex
    store.commit(
        policy.reserve(
            store.view(),
            large.id,
            maxima(calls=1, cost_microunits=maximum),
            identifier,
            trusted_maximum_quote=True,
        )
    )
    assert get(store, large.id).reserved.cost_microunits == maximum
    store.commit(
        policy.settle(store.view(), identifier, known(calls=1, cost_microunits=maximum - 1))
    )
    assert get(store, large.id).actual.cost_microunits == maximum - 1
    assert isinstance(get(store, large.id).actual.cost_microunits, int)


@pytest.mark.parametrize("bad", ("missing", "stale", "invalid", "foreign"))
def test_owner_usage_evidence_must_be_current_and_accessible(store, bad):
    from signal_gating.executive import ArtifactRef

    reservation = reserve(store)
    policy = BudgetPolicy()
    store.commit(policy.mark_unknown(store.view(), reservation.id))
    artifact = ArtifactRef(
        id="e" * 32,
        digest="0" * 64,
        storage_location="fixture",
        media_type="text/plain",
        origin="owner",
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
        size_bytes=0,
    )
    if bad == "invalid":
        artifact = artifact.model_copy(update={"valid": False})
    if bad == "foreign":
        artifact = artifact.model_copy(update={"scope": Scope()})
    if bad != "missing":
        store.commit(CommitBatch(events=(changed((artifact,)),)))
    command = ResolveUsage(
        reservation_id=reservation.id,
        expected_revision=2,
        usage=known(calls=2),
        evidence=(Ref(id=artifact.id, revision=2 if bad == "stale" else 1),),
        explanation="Measured",
    )
    batch = Intake(store, usage_resolver=policy.resolve_usage).accept(
        command, principal=owner(), state=store.view()
    )
    assert batch.response_receipt.disposition in ("deny", "stale")
    store.commit(batch)
    assert get(store, reservation.id).status == "uncertain"
    assert get(store, ROOT).reserved.calls == 6


def test_owner_known_usage_retains_evidence_in_real_settlement(store):
    from signal_gating.executive import ArtifactRef

    reservation = reserve(store)
    artifact = ArtifactRef(
        id="e" * 32,
        digest="0" * 64,
        storage_location="fixture",
        media_type="text/plain",
        origin="owner",
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
        size_bytes=0,
    )
    store.commit(CommitBatch(events=(changed((artifact,)),)))
    command = ResolveUsage(
        reservation_id=reservation.id,
        expected_revision=1,
        usage=known(calls=2),
        evidence=(Ref(id=artifact.id),),
        explanation="Metered usage",
    )
    batch = Intake(store, usage_resolver=BudgetPolicy().resolve_usage).accept(
        command, principal=owner(), state=store.view()
    )
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)
    assert get(store, reservation.id).evidence == command.evidence
    assert get(store, ROOT).actual.calls == 2


def test_unattributed_existing_hold_is_never_released_by_new_reservation(store):
    root = get(store, ROOT)
    store.commit(
        CommitBatch(
            events=(
                changed(
                    (
                        root.model_copy(
                            update={"revision": root.revision + 1, "reserved": known(calls=8)}
                        ),
                    )
                ),
            ),
            expected_revisions=(Ref(id=root.id, revision=root.revision),),
        )
    )
    reserve(store, calls=1)
    assert get(store, ROOT).reserved.calls == 9
    with pytest.raises(BudgetError):
        reserve(store, budget=RIGHT, calls=2)


def test_accepting_maximum_cannot_erase_observed_overage(store):
    reservation = reserve(store, calls=2)
    policy = BudgetPolicy()
    store.commit(policy.settle(store.view(), reservation.id, Usage(calls=3)))
    command = ResolveUsage(
        reservation_id=reservation.id,
        expected_revision=2,
        accept_reserved_maximum=True,
        explanation="Charge maximum",
    )
    with pytest.raises(BudgetError):
        policy.resolve_usage(command, store.view())
    assert get(store, ROOT).actual.calls == 3


@pytest.mark.parametrize("problem", ("currency", "cycle"))
def test_ancestor_tree_must_be_consistent(store, problem):
    left = get(store, LEFT)
    if problem == "currency":
        updates = {
            "limits": maxima(currency="EUR"),
            "actual": known(currency="EUR"),
            "reserved": known(currency="EUR"),
        }
        record = left.model_copy(update={"revision": left.revision + 1, **updates})
    else:
        root = get(store, ROOT)
        record = root.model_copy(update={"revision": root.revision + 1, "parent_id": LEFT})
    store.commit(
        CommitBatch(
            events=(changed((record,)),),
            expected_revisions=(Ref(id=record.id, revision=record.revision - 1),),
        )
    )
    with pytest.raises(BudgetError):
        BudgetPolicy().reserve(
            store.view(), LEFT, maxima(), uuid4().hex, trusted_maximum_quote=True
        )


def test_partial_observations_do_not_erase_prior_measured_quantities(store):
    reservation = reserve(store, calls=2)
    policy = BudgetPolicy()
    store.commit(policy.settle(store.view(), reservation.id, Usage(calls=3)))
    store.commit(policy.settle(store.view(), reservation.id, Usage(cost_microunits=1)))
    assert get(store, ROOT).actual.calls == 3
    assert get(store, reservation.id).usage.calls == 3
    with pytest.raises(BudgetError, match="bound"):
        policy.reserve(
            store.view(), RIGHT, maxima(calls=1), uuid4().hex, trusted_maximum_quote=True
        )


def test_unknown_hold_cannot_hide_unattributed_existing_balance(store):
    unlimited = Budget(
        id="d" * 32,
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
        limits=Limits(),
        reserved=known(cost_microunits=7),
    )
    store.commit(CommitBatch(events=(changed((unlimited,)),)))
    before = store.view()
    with pytest.raises(BudgetError, match="untracked"):
        BudgetPolicy(strict_money=False).reserve(before, unlimited.id, Limits(), uuid4().hex)
    assert store.view() == before


def test_nonstrict_untrusted_estimate_cannot_authorize_finite_money_ceiling(store):
    with pytest.raises(BudgetError):
        BudgetPolicy(strict_money=False).reserve(
            store.view(), LEFT, maxima(calls=1, cost_microunits=1), uuid4().hex
        )


def test_nonstrict_untrusted_finite_estimate_remains_unknown_liability(store):
    unlimited = Budget(
        id="d" * 32, scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID), limits=Limits()
    )
    store.commit(CommitBatch(events=(changed((unlimited,)),)))
    policy = BudgetPolicy(strict_money=False)
    identifier = uuid4().hex
    store.commit(
        policy.reserve(store.view(), unlimited.id, maxima(calls=1, cost_microunits=1), identifier)
    )
    assert get(store, unlimited.id).reserved.cost_microunits is None
    assert get(store, unlimited.id).accounting_uncertain
    command = ResolveUsage(
        reservation_id=identifier,
        expected_revision=1,
        accept_reserved_maximum=True,
        explanation="Use maximum",
    )
    with pytest.raises(BudgetError):
        policy.resolve_usage(command, store.view())


def test_reservation_cannot_bind_itself_as_subject(store):
    before = store.view()
    identifier = uuid4().hex
    with pytest.raises(BudgetError) as rejected:
        batch = BudgetPolicy().reserve(
            before,
            LEFT,
            maxima(calls=1),
            identifier,
            subject_id=identifier,
            trusted_maximum_quote=True,
        )
        # The real reducer previously accepted this self-reference, because
        # the reservation was visible in its own resulting post-image bundle.
        store.commit(batch)
    assert rejected.value.code == "invalid_reservation_subject"
    assert store.view() == before
