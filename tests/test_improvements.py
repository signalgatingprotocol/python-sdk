"""Tests for codebase improvements: bug fixes, performance, new primitives."""

import asyncio

import pytest

from signal_gating import Agent, AgentContext, Gate, Mesh, MeshError, Signal
from signal_gating.channel import PriorityChannel

# --- Signal types ---


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


# === Signal.child() lineage tracking ===


class TestSignalChild:
    def test_child_preserves_trace_id(self):
        parent = TaskSignal(task="analyze")
        child = parent.child(task="analyze_part_1")
        assert child.trace_id == parent.trace_id

    def test_child_records_parent_id(self):
        parent = TaskSignal(task="analyze")
        child = parent.child(task="sub_analyze")
        assert child.parent_id == parent.id

    def test_child_gets_new_id(self):
        parent = Signal()
        child = parent.child(priority=10)
        assert child.id != parent.id

    def test_child_chain(self):
        """Multi-level lineage: grandchild tracks parent, not grandparent."""
        root = Signal()
        child = root.child(priority=1)
        grandchild = child.child(priority=2)
        assert grandchild.parent_id == child.id
        assert grandchild.trace_id == root.trace_id

    def test_parent_id_default_empty(self):
        s = Signal()
        assert s.parent_id == ""

    def test_parent_id_hidden_in_repr(self):
        s = Signal()
        assert "parent_id" not in repr(s)

    def test_child_with_subclass(self):
        parent = TaskSignal(task="build")
        child = parent.child(task="build_step_1")
        assert isinstance(child, TaskSignal)
        assert child.task == "build_step_1"


# === Gate.when() conditional branching ===


class TestGateWhen:
    async def test_when_then_branch(self):
        gate = Gate.when(
            lambda s: s.priority >= 5,
            then=Gate.transform(lambda s: s.evolve(priority=99)),
        )
        result = await gate.process(Signal(priority=8))
        assert result is not None
        assert result.priority == 99

    async def test_when_otherwise_branch(self):
        gate = Gate.when(
            lambda s: s.priority >= 5,
            then=Gate.transform(lambda s: s.evolve(priority=99)),
            otherwise=Gate.transform(lambda s: s.evolve(priority=1)),
        )
        result = await gate.process(Signal(priority=2))
        assert result is not None
        assert result.priority == 1

    async def test_when_no_otherwise_passes_through(self):
        gate = Gate.when(
            lambda s: s.priority >= 100,
            then=Gate.block(),
        )
        s = Signal(priority=5)
        result = await gate.process(s)
        assert result is not None
        assert result.priority == 5

    async def test_when_then_can_reject(self):
        gate = Gate.when(
            lambda s: s.priority >= 5,
            then=Gate.block(),
        )
        result = await gate.process(Signal(priority=10))
        assert result is None

    async def test_when_composable(self):
        """Gate.when() composes with other gates via >>."""
        pipeline = Gate.by_priority(3) >> Gate.when(
            lambda s: s.priority >= 8,
            then=Gate.transform(lambda s: s.evolve(priority=100)),
            otherwise=Gate.transform(lambda s: s.evolve(priority=50)),
        )
        high = await pipeline.process(Signal(priority=10))
        assert high is not None and high.priority == 100

        medium = await pipeline.process(Signal(priority=5))
        assert medium is not None and medium.priority == 50

        low = await pipeline.process(Signal(priority=1))
        assert low is None


# === Gate.sample() probabilistic sampling ===


class TestGateSample:
    async def test_sample_rate_zero_blocks_all(self):
        gate = Gate.sample(0.0)
        results = [await gate.process(Signal()) for _ in range(10)]
        assert all(r is None for r in results)

    async def test_sample_rate_one_passes_all(self):
        gate = Gate.sample(1.0)
        results = [await gate.process(Signal()) for _ in range(10)]
        assert all(r is not None for r in results)

    async def test_sample_rate_partial(self):
        gate = Gate.sample(0.5)
        results = [await gate.process(Signal()) for _ in range(1000)]
        passed = sum(1 for r in results if r is not None)
        # Should be roughly 50% — allow wide margin for randomness
        assert 300 < passed < 700


# === Agent.emit_many() batch emission ===


class TestEmitMany:
    async def test_emit_many_delivers_all(self):
        sender = Agent("sender")
        receiver = Agent("receiver")
        received: list[str] = []

        @receiver.on(TaskSignal)
        async def handle(s: TaskSignal):
            received.append(s.task)

        mesh = Mesh([sender, receiver])
        mesh.connect(sender, receiver)

        async with mesh:
            await sender.emit_many([
                TaskSignal(task=f"batch-{i}") for i in range(5)
            ])
            await asyncio.sleep(0.1)

        assert received == [f"batch-{i}" for i in range(5)]


# === Agent.set_tracer() encapsulation ===


class TestSetTracer:
    def test_set_tracer(self):
        from signal_gating import Tracer

        agent = Agent("test")
        tracer = Tracer()
        agent.set_tracer(tracer)
        assert agent._tracer is tracer  # noqa: SLF001


# === DLQ.replay() ===


class TestDLQReplay:
    async def test_replay_sends_to_channel(self):
        agent = Agent("worker", gates=[Gate.by_priority(100)])

        @agent.on(Signal)
        async def handle(s: Signal):
            pass

        await agent.start()
        await agent.inbox.send(Signal(priority=1))
        await agent.inbox.send(Signal(priority=2))
        await asyncio.sleep(0.05)
        await agent.stop()

        assert agent.dead_letters.count == 2

        # Replay into a new channel
        from signal_gating.channel import Channel

        ch: Channel[Signal] = Channel(Signal)
        count = await agent.dead_letters.replay(ch)
        assert count == 2
        assert ch.pending == 2
        assert agent.dead_letters.count == 0  # Cleared after replay

    async def test_replay_into_priority_channel(self):
        agent = Agent("worker", gates=[Gate.by_priority(100)])

        @agent.on(Signal)
        async def handle(s: Signal):
            pass

        await agent.start()
        await agent.inbox.send(Signal(priority=3))
        await asyncio.sleep(0.05)
        await agent.stop()

        ch: PriorityChannel[Signal] = PriorityChannel(Signal, buffer_size=100)
        count = await agent.dead_letters.replay(ch)
        assert count == 1
        received = await ch.receive()
        assert received.priority == 3


# === Dynamic Topology: Mesh.remove() and Mesh.disconnect() ===


class TestDynamicTopology:
    async def test_remove_agent(self):
        a = Agent("a")
        b = Agent("b")
        mesh = Mesh([a, b])
        mesh.connect(a, b)

        async with mesh:
            await mesh.remove(b)
            assert len(mesh.agents) == 1
            assert len(mesh.edges) == 0

    async def test_remove_agent_by_name(self):
        a = Agent("a")
        mesh = Mesh([a])
        await mesh.remove("a")
        assert len(mesh.agents) == 0

    async def test_remove_nonexistent_raises(self):
        mesh = Mesh()
        with pytest.raises(MeshError):
            await mesh.remove("ghost")

    def test_disconnect(self):
        a = Agent("a")
        b = Agent("b")
        c = Agent("c")
        mesh = Mesh([a, b, c])
        mesh.connect(a, b)
        mesh.connect(a, c)

        removed = mesh.disconnect(a, b)
        assert removed == 1
        assert len(mesh.edges) == 1
        assert mesh.edges[0].target.name == "c"

    def test_disconnect_by_name(self):
        a = Agent("a")
        b = Agent("b")
        mesh = Mesh([a, b])
        mesh.connect(a, b)

        removed = mesh.disconnect("a", "b")
        assert removed == 1
        assert len(mesh.edges) == 0

    def test_disconnect_nonexistent_returns_zero(self):
        a = Agent("a")
        b = Agent("b")
        mesh = Mesh([a, b])
        removed = mesh.disconnect(a, b)
        assert removed == 0

    async def test_remove_cleans_capabilities(self):
        a = Agent("a")
        mesh = Mesh([a])
        mesh.declare_capabilities(a, "analysis")
        assert len(mesh.find_capable("analysis")) == 1

        await mesh.remove(a)
        assert len(mesh.find_capable("analysis")) == 0


# === PriorityChannel event-driven backpressure ===


class TestPriorityChannelEventDriven:
    async def test_send_wait_no_polling(self):
        """Verify send_wait is event-driven by checking it resolves quickly."""
        ch: PriorityChannel[Signal] = PriorityChannel(Signal, buffer_size=1)
        await ch.send(Signal(priority=1))

        sent = False

        async def delayed_send():
            nonlocal sent
            await ch.send_wait(Signal(priority=5), timeout=2.0)
            sent = True

        task = asyncio.create_task(delayed_send())
        await asyncio.sleep(0.02)
        assert not sent

        await ch.receive()  # Free space -> triggers _has_space event
        await asyncio.sleep(0.02)
        assert sent  # Should resolve almost immediately, no 10ms polling delay

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# === Scatter fix: no double-send ===


class TestScatterFix:
    async def test_scatter_sends_once(self):
        """Each target should receive exactly ONE signal per scatter."""
        worker = Agent("worker")
        receive_count = 0

        @worker.on(Signal)
        async def handle(s: Signal, ctx: AgentContext):
            nonlocal receive_count
            receive_count += 1
            await ctx.reply(ResultSignal(result="done"))

        mesh = Mesh([worker])

        async with mesh:
            results = await mesh.scatter(Signal(), [worker], timeout=2.0)

        # Worker should have received exactly 1 signal, not 2
        assert receive_count == 1
        assert len(results) == 1


# === Tracer indexed lookups ===


class TestTracerIndexed:
    def test_get_trace_indexed(self):
        from signal_gating import Tracer

        tracer = Tracer()
        for i in range(100):
            tracer.record(f"trace-{i % 10}", f"sig-{i}", "agent", "gate", "passed")

        # Each trace should have 10 spans
        trace = tracer.get_trace("trace-0")
        assert len(trace) == 10

    def test_get_agent_spans_indexed(self):
        from signal_gating import Tracer

        tracer = Tracer()
        for i in range(100):
            tracer.record(f"trace-{i}", f"sig-{i}", f"agent-{i % 5}", "gate", "passed")

        spans = tracer.get_agent_spans("agent-0")
        assert len(spans) == 20

    def test_eviction_rebuilds_indexes(self):
        from signal_gating import Tracer

        tracer = Tracer(max_spans=10)
        for i in range(20):
            tracer.record(f"trace-{i}", f"sig-{i}", "agent", "gate", "passed")

        assert tracer.span_count == 10
        # Only traces 10-19 should remain
        assert len(tracer.get_trace("trace-0")) == 0
        assert len(tracer.get_trace("trace-15")) == 1

    def test_clear_clears_indexes(self):
        from signal_gating import Tracer

        tracer = Tracer()
        tracer.record("t1", "s1", "a", "g", "passed")
        tracer.clear()
        assert tracer.get_trace("t1") == []
        assert tracer.get_agent_spans("a") == []

    def test_summary_uses_indexes(self):
        from signal_gating import Tracer

        tracer = Tracer()
        tracer.record("t1", "s1", "a", "g", "passed")
        tracer.record("t2", "s2", "b", "g", "rejected")
        s = tracer.summary()
        assert s["unique_traces"] == 2
        assert s["unique_agents"] == 2


# === Handler context detection robustness ===


class TestHandlerContextDetection:
    async def test_handler_with_context_annotation(self):
        agent = Agent("test")
        received_ctx = []

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received_ctx.append(ctx)

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert len(received_ctx) == 1
        assert received_ctx[0].agent_name == "test"

    async def test_handler_without_context(self):
        agent = Agent("test")
        received: list[Signal] = []

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal)

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert len(received) == 1
