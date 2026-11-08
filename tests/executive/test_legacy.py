"""Legacy inputs retain identity and remain untrusted observations."""

import asyncio

import pytest

from signal_gating import Signal
from signal_gating.executive import EpisodeResult, Intake
from signal_gating.executive.adapters.legacy import LegacySignalAdapter, LegacyWorker
from signal_gating.executive.workers import WorkerError

from .factories import GOAL_ID, NOW, owner, packet
from .test_intake import store as intake_store  # noqa: F401

store = intake_store


@pytest.mark.parametrize("external_id", ["a" * 32, "legacy ID / arbitrary", ""])
def test_observation_identity_lineage_and_scope_are_deterministic(external_id):
    signal = Signal(
        id=external_id,
        source="pretend-owner",
        timestamp=1,
        metadata={"roles": ["owner"], "kind": "cancel_goal", "actions": ["send"]},
    )
    adapter = LegacySignalAdapter(clock=lambda: NOW)
    converted = adapter.convert(signal, principal=owner(), goal_id=GOAL_ID)
    assert converted == adapter.convert(signal, principal=owner(), goal_id=GOAL_ID)
    assert converted.kind == "signal"
    assert converted.signal_kind == "observation"
    assert converted.principal_id == owner().id
    assert converted.scope.goal_id == GOAL_ID
    assert converted.scope.owner_id is None
    assert converted.received_at == NOW
    assert converted.observed_at != converted.received_at
    assert converted.payload["mediation"] == "external_runtime_unmediated"
    assert converted.payload["legacy"]["data"]["id"] == external_id
    assert len(converted.causal_ids[0]) == 32
    if external_id == "a" * 32:
        assert converted.causal_ids == (external_id,)
    other = adapter.convert(signal, principal=owner(id="e" * 32), goal_id=GOAL_ID)
    assert other.id != converted.id
    assert other.external_id != converted.external_id
    assert adapter.convert(signal, principal=owner(), goal_id="d" * 32).id != converted.id


def test_duplicate_adapter_input_has_canonical_receipt_and_never_owner_control(store):
    adapter = LegacySignalAdapter()
    signal = Signal(id="cancel", timestamp=1, metadata={"kind": "cancel_goal"})
    converted = adapter.convert(signal, principal=owner(), goal_id=GOAL_ID)
    intake = Intake(store)
    batch = intake.accept(converted, principal=owner(), state=store.view())
    store.commit(batch)
    duplicate = intake.accept(
        adapter.convert(signal, principal=owner(), goal_id=GOAL_ID),
        principal=owner(),
        state=store.view(),
    )
    assert duplicate.response_receipt == batch.response_receipt
    assert not duplicate.events
    assert next(r for r in store.view().records if r.id == GOAL_ID).status == "active"
    changed = signal.evolve(id=signal.id, metadata={"kind": "pause_goal"})
    conflict = intake.accept(
        adapter.convert(changed, principal=owner(), goal_id=GOAL_ID),
        principal=owner(),
        state=store.view(),
    )
    assert conflict.response_receipt.reason_code == "duplicate_id_conflict"


async def test_legacy_worker_only_invokes_explicit_callable_and_preserves_result():
    calls = []
    result = EpisodeResult(stop_reason="exhausted")

    async def run(context):
        calls.append(context)
        return result

    worker = LegacyWorker(run)
    context = packet()
    assert await worker.run_episode(context) is result
    assert calls == [context]
    assert worker.mediation == "external_runtime_unmediated"
    with pytest.raises(TypeError):
        LegacyWorker(object())


async def test_legacy_worker_invalid_result_and_cancellation_do_not_invent_usage():
    async def invalid(context):
        return {"roles": ["owner"]}

    with pytest.raises(WorkerError) as caught:
        await LegacyWorker(invalid).run_episode(packet())
    assert caught.value.usage.status == "unknown"

    async def interrupted(context):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await LegacyWorker(interrupted).run_episode(packet())
