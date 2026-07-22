"""Contract tests for the mesh-wide delivery and identity trust boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from signal_gating import (
    Agent,
    AgentContext,
    AgentPool,
    Gate,
    Mesh,
    MeshEvent,
    Receipt,
    Signal,
    TrajectoryRecorder,
    TrajectoryReplayRunner,
)
from signal_gating.errors import ChannelClosed, MeshError


class Message(Signal):
    value: int = 0


def reply_with_value(agent: Agent, value: int, *, delay: float = 0.0) -> None:
    @agent.on(Message)
    async def reply(_signal: Message, ctx: AgentContext) -> None:
        if delay:
            await asyncio.sleep(delay)
        await ctx.reply(Message(value=value))


async def test_blocked_inject_fails_immediately_and_is_observable() -> None:
    worker = Agent("worker")
    received: list[Message] = []
    events: list[MeshEvent] = []

    @worker.on(Message)
    async def handle(signal: Message) -> None:
        received.append(signal)

    mesh = Mesh([worker])
    mesh.intercept(lambda _signal, _source, _target: None)
    mesh.record(events.append)
    signal = Message(value=1)

    async with mesh:
        with pytest.raises(MeshError) as excinfo:
            await asyncio.wait_for(mesh.inject(worker, signal), timeout=0.25)

    message = str(excinfo.value)
    assert "inject" in message
    assert "interceptor" in message
    assert "'mesh' -> 'worker'" in message
    assert "Message" in message
    assert signal.id in message
    assert received == []
    assert [event.action for event in events] == ["intercepted"]
    assert events[0].metadata["delivery_action"] == "inject"
    assert events[0].metadata["signal_id"] == signal.id


async def test_transform_is_delivered_and_recorded_as_the_final_signal() -> None:
    worker = Agent("worker")
    received: list[Message] = []
    recorder = TrajectoryRecorder()

    @worker.on(Message)
    async def handle(signal: Message) -> None:
        received.append(signal)

    mesh = Mesh([worker])
    mesh.intercept(
        lambda signal, _source, _target: signal.evolve(value=signal.value + 10)
        if isinstance(signal, Message)
        else signal
    )
    mesh.record(recorder)

    async with mesh:
        await mesh.inject(worker, Message(value=1))

    assert [signal.value for signal in received] == [11]
    assert [receipt.action for receipt in recorder.receipts] == ["inject"]
    replayed = recorder.receipts[0].to_signal()
    assert isinstance(replayed, Message)
    assert replayed.value == 11
    inject_spans = [span for span in mesh.tracer.get_agent_spans("mesh") if span.action == "inject"]
    assert len(inject_spans) == 1
    assert inject_spans[0].signal_id == received[0].id


async def test_edge_block_is_a_silent_observable_drop() -> None:
    source, target = Agent("source"), Agent("target")
    received: list[Message] = []
    events: list[MeshEvent] = []

    @target.on(Message)
    async def handle(signal: Message) -> None:
        received.append(signal)

    mesh = Mesh([source, target])
    mesh.connect(source, target)
    mesh.intercept(lambda _signal, _source, _target: None)
    mesh.record(events.append)

    async with mesh:
        await source.emit(Message(value=1))

    assert received == []
    assert [event.action for event in events] == ["intercepted"]
    assert events[0].metadata["delivery_action"] == "routed"


async def test_publish_counts_only_allowed_deliveries() -> None:
    allowed, blocked = Agent("allowed"), Agent("blocked")
    allowed_values: list[int] = []
    blocked_values: list[int] = []

    @allowed.on(Message)
    async def handle_allowed(signal: Message) -> None:
        allowed_values.append(signal.value)

    @blocked.on(Message)
    async def handle_blocked(signal: Message) -> None:
        blocked_values.append(signal.value)

    mesh = Mesh([allowed, blocked])
    mesh.create_topic("events")
    mesh.subscribe(allowed, "events")
    mesh.subscribe(blocked, "events")
    mesh.intercept(
        lambda signal, _source, target: None
        if target == "blocked"
        else signal.evolve(value=signal.value + 1)
    )

    async with mesh:
        count = await mesh.publish("events", Message(value=1))

    assert count == 1
    assert allowed_values == [2]
    assert blocked_values == []


async def test_publish_continues_after_destination_closes_during_send() -> None:
    closing, healthy = Agent("closing"), Agent("healthy")
    healthy_values: list[int] = []

    @healthy.on(Message)
    async def handle_healthy(signal: Message) -> None:
        healthy_values.append(signal.value)

    mesh = Mesh([closing, healthy])
    mesh.create_topic("events")
    mesh.subscribe(closing, "events")
    mesh.subscribe(healthy, "events")

    def close_before_send(signal: Signal, _source: str, target: str) -> Signal:
        if target == "closing":
            closing.inbox.close()
        return signal

    mesh.intercept(close_before_send)

    async with mesh:
        count = await mesh.publish("events", Message(value=1))

    assert count == 1
    assert healthy_values == [1]


async def test_publish_propagates_channel_closed_from_interceptor() -> None:
    worker = Agent("worker")
    mesh = Mesh([worker])
    mesh.create_topic("events")
    mesh.subscribe(worker, "events")

    def raise_channel_closed(
        _signal: Signal, _source: str, _target: str
    ) -> Signal:
        raise ChannelClosed()

    mesh.intercept(raise_channel_closed)

    async with mesh:
        with pytest.raises(ChannelClosed):
            await mesh.publish("events", Message(value=1))


async def test_request_send_block_fails_without_leaking_capture() -> None:
    worker = Agent("worker")
    reply_with_value(worker, 2)
    mesh = Mesh([worker])
    mesh.intercept(lambda _signal, _source, _target: None)

    async with mesh:
        with pytest.raises(MeshError, match="request_sent"):
            await asyncio.wait_for(
                mesh.request(worker, Message(value=1), timeout=10), timeout=0.25
            )

    assert worker._outbox == []


async def test_request_response_block_fails_without_timeout_or_capture_leak() -> None:
    worker = Agent("worker")
    reply_with_value(worker, 2)
    mesh = Mesh([worker])
    mesh.intercept(
        lambda signal, source, target: None
        if source == "worker" and target == "mesh"
        else signal
    )

    async with mesh:
        with pytest.raises(MeshError, match="response_received"):
            await asyncio.wait_for(
                mesh.request(worker, Message(value=1), timeout=10), timeout=0.25
            )

    assert worker._outbox == []


async def test_request_returns_and_records_the_transformed_response() -> None:
    worker = Agent("worker")
    reply_with_value(worker, 2)
    recorder = TrajectoryRecorder()
    mesh = Mesh([worker])
    mesh.intercept(
        lambda signal, source, target: signal.evolve(value=signal.value + 10)
        if isinstance(signal, Message) and source == "worker" and target == "mesh"
        else signal
    )
    mesh.record(recorder)

    async with mesh:
        result = await mesh.request(worker, Message(value=1), timeout=1)

    assert isinstance(result, Message)
    assert result.value == 12
    response = next(
        receipt for receipt in recorder.receipts if receipt.action == "response_received"
    )
    assert response.payload == {"value": 12}


async def test_scatter_response_block_fails_immediately_and_cleans_captures() -> None:
    first, second = Agent("first"), Agent("second")
    reply_with_value(first, 1)
    reply_with_value(second, 2)
    mesh = Mesh([first, second])
    mesh.intercept(
        lambda signal, source, target: None
        if source == "second" and target == "mesh"
        else signal
    )

    async with mesh:
        with pytest.raises(MeshError, match="scatter_response"):
            await asyncio.wait_for(
                mesh.scatter(Message(), [first, second], timeout=10), timeout=0.5
            )

    assert first._outbox == []
    assert second._outbox == []


async def test_race_skips_blocked_send_when_an_allowed_target_replies() -> None:
    blocked, allowed = Agent("blocked"), Agent("allowed")
    reply_with_value(blocked, 1)
    reply_with_value(allowed, 2)
    mesh = Mesh([blocked, allowed])
    mesh.intercept(
        lambda signal, source, target: None
        if source == "mesh" and target == "blocked"
        else signal
    )

    async with mesh:
        result = await mesh.race(Message(), [blocked, allowed], timeout=1)

    assert isinstance(result, Message)
    assert result.value == 2
    assert blocked._outbox == []
    assert allowed._outbox == []


async def test_race_skips_blocked_response_when_another_response_is_allowed() -> None:
    blocked, allowed = Agent("blocked"), Agent("allowed")
    reply_with_value(blocked, 1)
    reply_with_value(allowed, 2, delay=0.02)
    mesh = Mesh([blocked, allowed])
    mesh.intercept(
        lambda signal, source, target: None
        if source == "blocked" and target == "mesh"
        else signal
    )

    async with mesh:
        result = await mesh.race(Message(), [blocked, allowed], timeout=1)

    assert isinstance(result, Message)
    assert result.value == 2


async def test_race_final_send_block_after_response_block_fails_immediately() -> None:
    first, second = Agent("first"), Agent("second")
    reply_with_value(first, 1)
    first_response_blocked = asyncio.Event()

    async def block_in_sequence(
        signal: Signal, source: str, target: str
    ) -> Signal | None:
        if source == "first" and target == "mesh":
            first_response_blocked.set()
            return None
        if source == "mesh" and target == "second":
            await first_response_blocked.wait()
            return None
        return signal

    mesh = Mesh([first, second])
    mesh.intercept(block_in_sequence)

    async with mesh:
        with pytest.raises(MeshError, match="race_response"):
            await asyncio.wait_for(
                mesh.race(Message(), [first, second], timeout=10), timeout=0.5
            )

    assert first._outbox == []
    assert second._outbox == []


async def test_race_duplicate_targets_use_independent_candidate_ids() -> None:
    worker = Agent("worker")
    correlation_ids: list[str] = []

    @worker.on(Message)
    async def reply_in_order(signal: Message) -> None:
        correlation_ids.append(signal.correlation_id)
        await worker.reply(signal, Message(value=len(correlation_ids)))

    def block_first_response(
        signal: Signal, source: str, target: str
    ) -> Signal | None:
        if (
            source == "worker"
            and target == "mesh"
            and correlation_ids
            and signal.correlation_id == correlation_ids[0]
        ):
            return None
        return signal

    mesh = Mesh([worker])
    mesh.intercept(block_first_response)

    async with mesh:
        result = await mesh.race(Message(), [worker, worker], timeout=1)

    assert isinstance(result, Message)
    assert result.value == 2
    assert len(correlation_ids) == 2
    assert len(set(correlation_ids)) == 2
    assert worker._outbox == []


async def test_race_send_error_propagates_instead_of_trying_another_target() -> None:
    invalid, allowed = Agent("invalid"), Agent("allowed")
    reply_with_value(invalid, 1)
    reply_with_value(allowed, 2)
    mesh = Mesh([invalid, allowed])
    mesh.intercept(
        lambda signal, source, target: "not a signal"
        if source == "mesh" and target == "invalid"
        else signal
    )  # type: ignore[arg-type]

    async with mesh:
        with pytest.raises(MeshError, match="must return Signal or None"):
            await mesh.race(Message(), [invalid, allowed], timeout=1)

    assert invalid._outbox == []
    assert allowed._outbox == []


async def test_race_response_error_fails_before_slower_allowed_response() -> None:
    invalid, allowed = Agent("invalid"), Agent("allowed")
    reply_with_value(invalid, 1)
    reply_with_value(allowed, 2, delay=0.5)
    mesh = Mesh([invalid, allowed])
    mesh.intercept(
        lambda signal, source, target: "not a signal"
        if source == "invalid" and target == "mesh"
        else signal
    )  # type: ignore[arg-type]

    async with mesh:
        with pytest.raises(MeshError, match="must return Signal or None"):
            await asyncio.wait_for(
                mesh.race(Message(), [invalid, allowed], timeout=10), timeout=0.25
            )

    assert invalid._outbox == []
    assert allowed._outbox == []


async def test_race_fails_immediately_when_every_send_is_blocked() -> None:
    first, second = Agent("first"), Agent("second")
    mesh = Mesh([first, second])
    mesh.intercept(lambda _signal, _source, _target: None)

    async with mesh:
        with pytest.raises(MeshError, match="race_sent"):
            await asyncio.wait_for(
                mesh.race(Message(), [first, second], timeout=10), timeout=0.25
            )

    assert first._outbox == []
    assert second._outbox == []


async def test_race_fails_when_every_response_is_blocked() -> None:
    first, second = Agent("first"), Agent("second")
    reply_with_value(first, 1)
    reply_with_value(second, 2)
    mesh = Mesh([first, second])
    mesh.intercept(
        lambda signal, _source, target: None if target == "mesh" else signal
    )

    async with mesh:
        with pytest.raises(MeshError, match="race_response"):
            await asyncio.wait_for(
                mesh.race(Message(), [first, second], timeout=10), timeout=0.5
            )

    assert first._outbox == []
    assert second._outbox == []


async def test_workflow_variants_and_tools_inherit_the_boundary() -> None:
    worker = Agent("worker")

    @worker.tool()
    async def echo(value: int) -> int:
        return value

    mesh = Mesh([worker])
    mesh.intercept(lambda _signal, _source, _target: None)

    async with mesh:
        operations = [
            mesh.workflow(Message(), [worker], timeout=10),
            mesh.branch_workflow(
                Message(), lambda _signal: "only", {"only": [worker]}, timeout=10
            ),
            mesh.map_reduce(Message(), [worker], worker, timeout=10),
            mesh.call_tool(worker, "echo", value=1, timeout=10),
        ]
        for operation in operations:
            with pytest.raises(MeshError):
                await asyncio.wait_for(operation, timeout=0.25)

    assert worker._outbox == []


async def test_replay_respects_the_current_mesh_boundary() -> None:
    worker = Agent("worker")
    received: list[Message] = []

    @worker.on(Message)
    async def handle(signal: Message) -> None:
        received.append(signal)

    receipt = Receipt.from_signal(
        Message(value=1), source="mesh", target="worker", action="inject"
    )
    mesh = Mesh([worker])
    mesh.intercept(lambda _signal, _source, _target: None)

    async with mesh:
        with pytest.raises(MeshError, match="replay_delivered"):
            await TrajectoryReplayRunner([receipt]).replay_into(mesh)

    assert received == []


async def test_deliver_replay_preserves_boundary_and_receipt_metadata() -> None:
    worker = Agent("worker")
    received: list[Message] = []
    recorder = TrajectoryRecorder()

    @worker.on(Message)
    async def handle(signal: Message) -> None:
        received.append(signal)

    original = Message(value=1)
    receipt = Receipt.from_signal(
        original,
        source="original-source",
        target="worker",
        action="request_sent",
    )
    mesh = Mesh([worker])
    mesh.intercept(
        lambda signal, _source, _target: signal.evolve(value=signal.value + 1)
        if isinstance(signal, Message)
        else signal
    )
    mesh.record(recorder)

    async with mesh:
        delivery_result = await mesh._deliver_replay(
            worker,
            receipt.to_signal(),
            original_action=receipt.action,
            original_source=receipt.source,
            original_signal_id=receipt.signal_id,
        )

    assert delivery_result is None
    assert [signal.value for signal in received] == [2]
    assert len(recorder.receipts) == 1
    replay_receipt = recorder.receipts[0]
    assert replay_receipt.action == "replay_delivered"
    assert (replay_receipt.source, replay_receipt.target) == ("mesh", "worker")
    assert replay_receipt.payload == {"value": 2}
    assert replay_receipt.metadata == {
        "original_action": "request_sent",
        "original_source": "original-source",
        "original_signal_id": original.id,
    }
    replay_spans = [
        span
        for span in mesh.tracer.get_agent_spans("mesh")
        if span.action == "replay_delivered"
    ]
    assert len(replay_spans) == 1
    assert replay_spans[0].gate == "replay"
    assert replay_spans[0].signal_id == received[0].id


async def test_deliver_replay_rejects_same_name_agent_impostor() -> None:
    worker = Agent("worker")
    impostor = Agent("worker")
    signal = Message(value=1)
    mesh = Mesh([worker])

    with pytest.raises(MeshError, match="not the registered instance"):
        await mesh._deliver_replay(
            impostor,
            signal,
            original_action="inject",
            original_source="mesh",
            original_signal_id=signal.id,
        )


async def test_replay_runner_delegates_without_delivery_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = Agent("worker")
    receipt = Receipt.from_signal(
        Message(value=1),
        source="original-source",
        target="worker",
        action="inject",
    )
    mesh = Mesh([worker])
    calls: list[tuple[Agent | str, Signal, str, str, str]] = []

    async def deliver_replay(
        target: Agent | str,
        signal: Signal,
        *,
        original_action: str,
        original_source: str,
        original_signal_id: str,
    ) -> None:
        calls.append(
            (
                target,
                signal,
                original_action,
                original_source,
                original_signal_id,
            )
        )

    async def forbidden_deliver(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trajectory layer called Mesh._deliver directly")

    def forbidden_raise(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trajectory layer handled a delivery outcome")

    monkeypatch.setattr(mesh, "_deliver_replay", deliver_replay)
    monkeypatch.setattr(mesh, "_deliver", forbidden_deliver)
    monkeypatch.setattr(mesh, "_raise_if_blocked", forbidden_raise)

    result = await TrajectoryReplayRunner([receipt]).replay_into(mesh)

    assert result.delivered == 1
    assert len(calls) == 1
    target, signal, original_action, original_source, original_signal_id = calls[0]
    assert target is worker
    assert signal == receipt.to_signal()
    assert (original_action, original_source, original_signal_id) == (
        receipt.action,
        receipt.source,
        receipt.signal_id,
    )


def test_invalid_interceptor_result_is_rejected() -> None:
    worker = Agent("worker")
    mesh = Mesh([worker])
    mesh.intercept(lambda _signal, _source, _target: "not a signal")  # type: ignore[arg-type]

    with pytest.raises(MeshError, match="must return Signal or None"):
        asyncio.run(mesh.inject(worker, Message()))


async def test_interceptor_runs_before_edge_gate_and_rejection_is_recorded() -> None:
    source, target = Agent("source"), Agent("target")
    gate_values: list[int] = []
    events: list[MeshEvent] = []

    async def reject(signal: Signal) -> None:
        assert isinstance(signal, Message)
        gate_values.append(signal.value)
        return None

    mesh = Mesh([source, target])
    mesh.intercept(
        lambda signal, _source, _target: signal.evolve(value=signal.value + 1)
        if isinstance(signal, Message)
        else signal
    )
    mesh.connect(source, target, Gate(reject, name="reject"))
    mesh.record(events.append)

    async with mesh:
        await source.emit(Message(value=1))

    assert gate_values == [2]
    assert [event.action for event in events] == ["edge_rejected"]
    assert events[0].metadata["delivery_action"] == "routed"


async def test_failed_enqueue_does_not_record_a_success_event() -> None:
    worker = Agent("worker")
    events: list[MeshEvent] = []
    mesh = Mesh([worker])
    mesh.record(events.append)
    worker.inbox.close()

    with pytest.raises(ChannelClosed):
        await mesh.inject(worker, Message())

    assert events == []


def test_same_name_agent_impostors_are_rejected_by_sync_apis() -> None:
    canonical, target = Agent("canonical"), Agent("target")
    impostor = Agent("canonical")
    mesh = Mesh([canonical, target])
    mesh.create_topic("events")

    operations: list[Callable[[], object]] = [
        lambda: mesh.connect(impostor, target),
        lambda: mesh.disconnect(impostor, target),
        lambda: mesh.route(impostor, [(lambda _signal: True, target)]),
        lambda: mesh.load_balance(impostor, [target]),
        lambda: mesh.subscribe(impostor, "events"),
        lambda: mesh.unsubscribe(impostor, "events"),
        lambda: mesh.declare_capabilities(impostor, "work"),
        lambda: mesh.agent_capabilities(impostor),
        lambda: mesh.discover_tools(impostor),
    ]

    for operation in operations:
        with pytest.raises(MeshError, match="not the registered instance"):
            operation()

    assert mesh.edges == []
    assert canonical._outbox == []
    assert mesh.list_topics()["events"] == []


async def test_same_name_agent_impostors_are_rejected_by_async_apis() -> None:
    canonical = Agent("canonical")
    impostor = Agent("canonical")
    mesh = Mesh([canonical])

    with pytest.raises(MeshError, match="not the registered instance"):
        await mesh.inject(impostor, Message())
    with pytest.raises(MeshError, match="not the registered instance"):
        await mesh.request(impostor, Message())
    with pytest.raises(MeshError, match="not the registered instance"):
        await mesh.remove(impostor)

    assert mesh.get("canonical") is canonical


def test_pool_identity_and_agent_pool_name_collisions_are_rejected() -> None:
    named_agent = Agent("workers")
    mesh = Mesh([named_agent])
    with pytest.raises(MeshError, match="conflicts with an existing agent"):
        mesh.add_pool(AgentPool("workers", size=1))

    target = Agent("target")
    pool = AgentPool("pool", size=1)
    mesh = Mesh([target])
    mesh.add_pool(pool)

    with pytest.raises(MeshError, match="conflicts with an existing pool"):
        mesh.add(Agent("pool"))
    with pytest.raises(MeshError, match="not the registered instance"):
        mesh.connect(target, AgentPool("pool", size=1))
    with pytest.raises(MeshError, match="not the registered instance"):
        mesh.connect(Agent(pool.workers[0].name), target)
