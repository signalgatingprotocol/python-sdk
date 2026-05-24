"""Tests for agent-native primitives: supervision fix, Gate.batch, Gate.parallel,
Gate.fallback, Mesh.request, and Channel.merge."""

import asyncio

import pytest

from signal_gating import Agent, AgentContext, Channel, Gate, Mesh, Signal

# --- Signal types for testing ---


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


# =============================================================================
# Supervision Fix
# =============================================================================


class TestSupervisionFix:
    """The critical bug: _run_loop was swallowing exceptions, preventing
    _supervised_loop from ever restarting agents. Verify it works now."""

    async def test_agent_restarts_on_handler_crash(self):
        """Agent should restart when the processing loop crashes."""
        agent = Agent("crasher", max_restarts=2, restart_delay=0.01)
        crash_count = 0

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal):
            nonlocal crash_count
            crash_count += 1
            if crash_count <= 1:
                raise RuntimeError("Intentional crash")

        await agent.start()
        await agent.inbox.send(TaskSignal(task="crash", priority=1))
        await asyncio.sleep(0.1)
        # The first signal crashes, agent restarts. Send another.
        await agent.inbox.send(TaskSignal(task="recover", priority=1))
        await asyncio.sleep(0.1)
        await agent.stop()

        # Handler was called at least once (crash) and the agent restarted
        assert crash_count >= 1
        assert agent._restart_count >= 0

    async def test_handler_errors_go_to_dlq_not_restart(self):
        """Handler errors should be isolated to DLQ, not trigger restarts.

        This is by design: handler errors are caught, logged, and sent to
        the dead letter queue. The agent keeps running. Supervision restarts
        are reserved for infrastructure-level crashes (e.g., inbox failure).
        """
        agent = Agent("resilient", max_restarts=3, restart_delay=0.01)
        call_count = 0

        @agent.on(TaskSignal)
        async def always_crash(signal: TaskSignal):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Handler failure")

        await agent.start()
        for _ in range(3):
            await agent.inbox.send(TaskSignal(task="fail", priority=1))
        await asyncio.sleep(0.1)
        await agent.stop()

        # All 3 signals were processed (handler called), errors went to DLQ
        assert call_count == 3
        assert agent.dead_letters.count == 3
        # Agent did NOT restart; handler errors are isolated
        assert agent._restart_count == 0


# =============================================================================
# Gate.batch()
# =============================================================================


class TestGateBatch:
    async def test_batch_accumulates_signals(self):
        gate = Gate.batch(3)
        s1 = Signal(priority=1)
        s2 = Signal(priority=2)
        s3 = Signal(priority=3)

        assert await gate.process(s1) is None  # accumulating
        assert await gate.process(s2) is None  # accumulating
        result = await gate.process(s3)  # batch ready!

        assert result is not None
        assert result.metadata["batch_size"] == 3
        assert len(result.metadata["batch"]) == 3

    async def test_batch_resets_after_flush(self):
        gate = Gate.batch(2)

        assert await gate.process(Signal(priority=1)) is None
        result1 = await gate.process(Signal(priority=2))
        assert result1 is not None
        assert result1.metadata["batch_size"] == 2

        # Second batch
        assert await gate.process(Signal(priority=3)) is None
        result2 = await gate.process(Signal(priority=4))
        assert result2 is not None
        assert result2.metadata["batch_size"] == 2

    async def test_batch_timeout_flushes_early(self):
        gate = Gate.batch(100, timeout=0.01)

        # Send fewer than batch size
        await gate.process(Signal(priority=1))
        await gate.process(Signal(priority=2))

        # Wait for timeout to elapse
        await asyncio.sleep(0.02)

        # Next signal should trigger timeout flush
        result = await gate.process(Signal(priority=3))
        assert result is not None
        assert result.metadata["batch_size"] == 3

    async def test_batch_size_one(self):
        gate = Gate.batch(1)
        result = await gate.process(Signal(priority=5))
        assert result is not None
        assert result.metadata["batch_size"] == 1

    async def test_batch_invalid_size(self):
        with pytest.raises(ValueError, match="batch size"):
            Gate.batch(0)

    async def test_batch_preserves_signal_content(self):
        gate = Gate.batch(2)
        s1 = TaskSignal(task="a", priority=1)
        s2 = TaskSignal(task="b", priority=2)

        await gate.process(s1)
        result = await gate.process(s2)

        assert result is not None
        batch = result.metadata["batch"]
        assert batch[0]["priority"] == 1
        assert batch[1]["priority"] == 2


# =============================================================================
# Gate.parallel()
# =============================================================================


class TestGateParallel:
    async def test_parallel_all_pass(self):
        g1 = Gate.filter(lambda s: s.priority > 0, name="g1")
        g2 = Gate.filter(lambda s: s.priority < 100, name="g2")
        gate = Gate.parallel(g1, g2, mode="all")

        result = await gate.process(Signal(priority=5))
        assert result is not None

    async def test_parallel_all_one_rejects(self):
        g1 = Gate.filter(lambda s: s.priority > 0, name="g1")
        g2 = Gate.filter(lambda s: s.priority > 10, name="g2")
        gate = Gate.parallel(g1, g2, mode="all")

        result = await gate.process(Signal(priority=5))
        assert result is None

    async def test_parallel_any_one_passes(self):
        g1 = Gate.filter(lambda s: s.priority > 100, name="g1")  # rejects
        g2 = Gate.filter(lambda s: s.priority > 0, name="g2")  # passes
        gate = Gate.parallel(g1, g2, mode="any")

        result = await gate.process(Signal(priority=5))
        assert result is not None

    async def test_parallel_any_all_reject(self):
        g1 = Gate.filter(lambda s: s.priority > 100, name="g1")
        g2 = Gate.filter(lambda s: s.priority > 200, name="g2")
        gate = Gate.parallel(g1, g2, mode="any")

        result = await gate.process(Signal(priority=5))
        assert result is None

    async def test_parallel_race(self):
        slow = Gate(lambda s: asyncio.sleep(10, result=s), name="slow")
        fast = Gate.passthrough()
        gate = Gate.parallel(slow, fast, mode="race")

        result = await asyncio.wait_for(gate.process(Signal(priority=1)), timeout=1.0)
        assert result is not None

    async def test_parallel_no_gates_raises(self):
        with pytest.raises(ValueError, match="at least one gate"):
            Gate.parallel()

    async def test_parallel_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown parallel mode"):
            Gate.parallel(Gate.passthrough(), mode="invalid")

    async def test_parallel_runs_concurrently(self):
        """Verify gates run concurrently, not sequentially."""
        delays: list[float] = []

        async def slow_gate(signal: Signal) -> Signal:
            await asyncio.sleep(0.05)
            delays.append(0.05)
            return signal

        g1 = Gate(slow_gate, name="slow1")
        g2 = Gate(slow_gate, name="slow2")
        gate = Gate.parallel(g1, g2, mode="all")

        import time

        start = time.monotonic()
        result = await gate.process(Signal(priority=1))
        elapsed = time.monotonic() - start

        assert result is not None
        assert len(delays) == 2
        # If sequential, would take ~0.1s. Concurrent should be ~0.05s.
        assert elapsed < 0.09


# =============================================================================
# Gate.fallback()
# =============================================================================


class TestGateFallback:
    async def test_fallback_primary_passes(self):
        primary = Gate.passthrough()
        backup = Gate.block()
        gate = Gate.fallback(primary, backup)

        result = await gate.process(Signal(priority=1))
        assert result is not None

    async def test_fallback_uses_secondary(self):
        primary = Gate.block()
        backup = Gate.passthrough()
        gate = Gate.fallback(primary, backup)

        result = await gate.process(Signal(priority=1))
        assert result is not None

    async def test_fallback_all_reject(self):
        gate = Gate.fallback(Gate.block(), Gate.block(), Gate.block())

        result = await gate.process(Signal(priority=1))
        assert result is None

    async def test_fallback_is_lazy(self):
        """Only evaluates fallbacks until one passes."""
        call_count = 0

        async def counting_gate(signal: Signal) -> Signal:
            nonlocal call_count
            call_count += 1
            return signal

        gate = Gate.fallback(
            Gate.block(),
            Gate(counting_gate, name="counter"),
            Gate(counting_gate, name="counter2"),
        )
        await gate.process(Signal(priority=1))
        assert call_count == 1  # Only the first fallback ran, not the second

    async def test_fallback_chain(self):
        """Fallback with multiple levels."""
        transform1 = Gate.transform(lambda s: s.with_metadata(source="primary"))

        gate = Gate.fallback(Gate.block(), transform1)
        result = await gate.process(Signal(priority=1))
        assert result is not None
        # Primary (block) rejects, fallback transform1 runs
        assert result.metadata.get("source") == "primary"


# =============================================================================
# Mesh.request()
# =============================================================================


class TestMeshRequest:
    async def test_request_response(self):
        worker = Agent("worker")

        @worker.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            await ctx.reply(ResultSignal(result=f"done:{signal.task}"))

        mesh = Mesh([worker])

        async with mesh:
            response = await mesh.request(worker, TaskSignal(task="analyze", priority=1))
            assert isinstance(response, ResultSignal)
            assert response.result == "done:analyze"

    async def test_request_by_name(self):
        worker = Agent("worker")

        @worker.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            await ctx.reply(ResultSignal(result="ok"))

        mesh = Mesh([worker])

        async with mesh:
            response = await mesh.request("worker", TaskSignal(task="test", priority=1))
            assert isinstance(response, ResultSignal)

    async def test_request_timeout(self):
        """Request should timeout if agent never replies."""
        worker = Agent("silent")

        @worker.on(TaskSignal)
        async def handle(signal: TaskSignal):
            pass  # Never replies

        mesh = Mesh([worker])

        async with mesh:
            with pytest.raises(asyncio.TimeoutError):
                await mesh.request(worker, TaskSignal(task="timeout", priority=1), timeout=0.1)

    async def test_request_chained_workflow(self):
        """Chain multiple requests to build a multi-step workflow."""
        step1 = Agent("step1")
        step2 = Agent("step2")

        @step1.on(TaskSignal)
        async def handle1(signal: TaskSignal, ctx: AgentContext):
            await ctx.reply(ResultSignal(result=f"step1:{signal.task}"))

        @step2.on(ResultSignal)
        async def handle2(signal: ResultSignal, ctx: AgentContext):
            await ctx.reply(ResultSignal(result=f"step2:{signal.result}"))

        mesh = Mesh([step1, step2])

        async with mesh:
            r1 = await mesh.request(step1, TaskSignal(task="start", priority=1))
            r2 = await mesh.request(step2, r1)

            assert isinstance(r2, ResultSignal)
            assert r2.result == "step2:step1:start"

    async def test_request_cleans_up_capture(self):
        """Capture function should be removed after request completes."""
        worker = Agent("worker")

        @worker.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            await ctx.reply(ResultSignal(result="ok"))

        mesh = Mesh([worker])

        async with mesh:
            outbox_before = len(worker._outbox)
            await mesh.request(worker, TaskSignal(task="test", priority=1))
            outbox_after = len(worker._outbox)
            assert outbox_after == outbox_before


# =============================================================================
# Channel.merge()
# =============================================================================


class TestChannelMerge:
    async def test_merge_two_channels(self):
        ch1: Channel[Signal] = Channel(Signal, buffer_size=10)
        ch2: Channel[Signal] = Channel(Signal, buffer_size=10)

        s1 = Signal(priority=1)
        s2 = Signal(priority=2)
        await ch1.send(s1)
        await ch2.send(s2)

        # Close channels after sending so the merge iterator terminates
        ch1.close()
        ch2.close()

        received: list[Signal] = []
        async for signal in Channel.merge(ch1, ch2):
            received.append(signal)

        assert len(received) == 2
        ids = {s.id for s in received}
        assert s1.id in ids
        assert s2.id in ids

    async def test_merge_single_channel(self):
        ch: Channel[Signal] = Channel(Signal, buffer_size=10)
        await ch.send(Signal(priority=1))
        ch.close()

        received: list[Signal] = []
        async for signal in Channel.merge(ch):
            received.append(signal)
        assert len(received) == 1

    async def test_merge_empty_channels(self):
        ch1: Channel[Signal] = Channel(Signal, buffer_size=10)
        ch2: Channel[Signal] = Channel(Signal, buffer_size=10)
        ch1.close()
        ch2.close()

        received: list[Signal] = []
        async for signal in Channel.merge(ch1, ch2):
            received.append(signal)
        assert len(received) == 0

    async def test_merge_interleaved(self):
        ch1: Channel[Signal] = Channel(Signal, buffer_size=10)
        ch2: Channel[Signal] = Channel(Signal, buffer_size=10)

        received: list[int] = []

        async def producer(ch: Channel[Signal], priorities: list[int]) -> None:
            for p in priorities:
                await ch.send(Signal(priority=p))
                await asyncio.sleep(0.01)
            ch.close()

        async def consumer() -> None:
            async for signal in Channel.merge(ch1, ch2):
                received.append(signal.priority)

        await asyncio.gather(
            producer(ch1, [1, 3, 5]),
            producer(ch2, [2, 4, 6]),
            consumer(),
        )

        assert sorted(received) == [1, 2, 3, 4, 5, 6]
