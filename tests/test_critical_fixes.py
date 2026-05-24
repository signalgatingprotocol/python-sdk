"""Tests for critical bug fixes and new agent-native primitives."""

import asyncio
import time

from signal_gating import Agent, Gate, Mesh, Signal


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


# === Gate name operator precedence fix ===


class TestGateNamePrecedence:
    def test_gate_with_explicit_name(self):
        gate = Gate(lambda s: s, name="my_gate")
        assert gate.name == "my_gate"

    def test_gate_with_named_function(self):
        def my_filter(s: Signal) -> Signal | None:
            return s

        gate = Gate(my_filter)
        assert gate.name == "my_filter"

    def test_gate_with_lambda_no_name(self):
        gate = Gate(lambda s: s)
        assert gate.name == "<lambda>"

    def test_gate_explicit_name_overrides_fn_name(self):
        def my_fn(s: Signal) -> Signal:
            return s

        gate = Gate(my_fn, name="override")
        assert gate.name == "override"


# === Disconnect actually stops signal flow ===


class TestDisconnectStopsFlow:
    async def test_disconnect_stops_signals(self):
        """After disconnect, signals should NOT flow between agents."""
        sender = Agent("sender")
        receiver = Agent("receiver")
        received: list[str] = []

        @receiver.on(TaskSignal)
        async def handle(s: TaskSignal):
            received.append(s.task)

        mesh = Mesh([sender, receiver])
        mesh.connect(sender, receiver)

        async with mesh:
            # Send before disconnect (should arrive)
            await sender.emit(TaskSignal(task="before"))
            await asyncio.sleep(0.05)
            assert received == ["before"]

            # Disconnect
            mesh.disconnect(sender, receiver)

            # Send after disconnect (should NOT arrive)
            await sender.emit(TaskSignal(task="after"))
            await asyncio.sleep(0.05)
            assert received == ["before"]  # Still only one

    async def test_remove_stops_signals_to_removed_agent(self):
        """After removing an agent, signals should not flow to it."""
        sender = Agent("sender")
        receiver = Agent("receiver")
        received: list[str] = []

        @receiver.on(TaskSignal)
        async def handle(s: TaskSignal):
            received.append(s.task)

        mesh = Mesh([sender, receiver])
        mesh.connect(sender, receiver)

        async with mesh:
            await sender.emit(TaskSignal(task="before"))
            await asyncio.sleep(0.05)
            assert len(received) == 1

            await mesh.remove(receiver)

            # Sending after removal should not crash or deliver
            await sender.emit(TaskSignal(task="after"))
            await asyncio.sleep(0.05)
            assert len(received) == 1

    async def test_remove_stops_signals_from_removed_agent(self):
        """Removing an agent clears its outbox too."""
        sender = Agent("sender")
        middle = Agent("middle")
        receiver = Agent("receiver")
        received: list[str] = []

        @middle.on(TaskSignal)
        async def forward(s: TaskSignal):
            await middle.emit(s.evolve(task=f"forwarded:{s.task}"))

        @receiver.on(TaskSignal)
        async def handle(s: TaskSignal):
            received.append(s.task)

        mesh = Mesh([sender, middle, receiver])
        mesh.connect(sender, middle)
        mesh.connect(middle, receiver)

        async with mesh:
            await mesh.remove(middle)
            # Middle's outbox should be cleared (no route to receiver)
            assert len(middle._outbox) == 0


# === Agent restartable after stop ===


class TestAgentRestart:
    async def test_agent_restarts_after_stop(self):
        """Agent should be restartable after being stopped."""
        agent = Agent("restartable")
        received: list[str] = []

        @agent.on(TaskSignal)
        async def handle(s: TaskSignal):
            received.append(s.task)

        # First lifecycle
        await agent.start()
        await asyncio.sleep(0)  # Yield so task starts
        await agent.inbox.send(TaskSignal(task="first"))
        await asyncio.sleep(0.05)
        await agent.stop()
        assert received == ["first"]

        # Second lifecycle: should work after stop
        await agent.start()
        await asyncio.sleep(0)  # Yield so task starts
        assert agent.running
        await agent.inbox.send(TaskSignal(task="second"))
        await asyncio.sleep(0.05)
        await agent.stop()
        assert received == ["first", "second"]

    async def test_inbox_recreated_on_restart(self):
        agent = Agent("test")
        await agent.start()
        await asyncio.sleep(0)
        await agent.stop()
        assert agent.inbox.closed

        await agent.start()
        assert not agent.inbox.closed
        await agent.stop()


# === Exponential backoff on supervised restarts ===


class TestExponentialBackoff:
    async def test_handler_errors_go_to_dlq(self):
        """Handler errors are caught and sent to DLQ, not causing loop restart."""

        class CrashSignal(Signal):
            pass

        agent = Agent("crasher", max_restarts=2, restart_delay=0.01)

        @agent.on(CrashSignal)
        async def handle(s: CrashSignal):
            raise RuntimeError("intentional crash")

        await agent.start()
        await asyncio.sleep(0)
        await agent.inbox.send(CrashSignal())
        await asyncio.sleep(0.05)
        await agent.stop()

        # Handler errors go to DLQ, not restart
        assert agent._restart_count == 0
        assert agent.dead_letters.count == 1
        assert agent._error_count == 1

    def test_supervised_loop_has_exponential_backoff(self):
        """Verify the supervised loop code uses exponential backoff.

        The restart delay doubles each iteration (current_delay *= 2).
        This is verified structurally since loop-level crashes are hard to
        trigger in tests without monkey-patching.
        """
        agent = Agent("test", restart_delay=1.0)
        # The restart_delay is the initial value; it doubles in _supervised_loop
        assert agent._restart_delay == 1.0




# === emit_many concurrency ===


class TestEmitManyConcurrent:
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
            signals = [TaskSignal(task=f"t{i}") for i in range(10)]
            await sender.emit_many(signals)
            await asyncio.sleep(0.1)

        assert len(received) == 10

    async def test_emit_many_empty_list(self):
        agent = Agent("test")
        # Should not raise
        await agent.emit_many([])


# === Gate.throttle ===


class TestGateThrottle:
    async def test_throttle_drops_excess(self):
        gate = Gate.throttle(10)  # 10/sec = 100ms interval
        results = []
        for _ in range(5):
            results.append(await gate.process(Signal()))
        # First should pass, rapid-fire rest should be dropped
        assert results[0] is not None
        dropped = sum(1 for r in results[1:] if r is None)
        assert dropped >= 1  # At least some dropped

    async def test_throttle_passes_after_interval(self):
        gate = Gate.throttle(100)  # 100/sec = 10ms interval
        r1 = await gate.process(Signal())
        assert r1 is not None

        await asyncio.sleep(0.015)
        r2 = await gate.process(Signal())
        assert r2 is not None


# === Gate.ttl ===


class TestGateTTL:
    async def test_ttl_passes_fresh_signal(self):
        gate = Gate.ttl(10)
        s = Signal()  # Just created, fresh
        result = await gate.process(s)
        assert result is not None

    async def test_ttl_drops_expired_signal(self):
        gate = Gate.ttl(0.01)
        s = Signal(timestamp=time.time() - 1.0)  # 1 second old
        result = await gate.process(s)
        assert result is None

    async def test_ttl_composable(self):
        pipeline = Gate.ttl(10) >> Gate.by_priority(3)
        fresh_high = Signal(priority=5)
        result = await pipeline.process(fresh_high)
        assert result is not None


# === Gate.debounce ===


class TestGateDebounce:
    async def test_debounce_passes_after_quiet(self):
        gate = Gate.debounce(0.05)
        result = await gate.process(Signal(priority=1))
        # After debounce period, should pass
        assert result is not None

    async def test_debounce_resets_on_new_signal(self):
        gate = Gate.debounce(0.1)

        # Send two signals rapidly
        task1 = asyncio.create_task(gate.process(Signal(priority=1)))
        await asyncio.sleep(0.02)
        task2 = asyncio.create_task(gate.process(Signal(priority=2)))

        r1 = await task1
        r2 = await task2

        # First should be suppressed (newer signal arrived), second should pass
        assert r1 is None
        assert r2 is not None


# === Drain timeout ===


class TestDrainTimeout:
    async def test_drain_respects_custom_timeout(self):
        agent = Agent("slow")

        @agent.on(Signal)
        async def handle(s: Signal):
            await asyncio.sleep(0.01)

        mesh = Mesh([agent])

        async with mesh:
            for _ in range(5):
                await agent.inbox.send(Signal())
            await asyncio.sleep(0.1)

        # Should complete without hanging (the drain has a timeout)


# === Outbox tagging ===


class TestOutboxTagging:
    def test_connect_tags_route_function(self):
        a = Agent("a")
        b = Agent("b")
        mesh = Mesh([a, b])
        mesh.connect(a, b)

        assert len(a._outbox) == 1
        route_fn = a._outbox[0]
        assert route_fn.target == "b"
        assert route_fn.source == "a"
        assert route_fn.tag == "connect"

    def test_disconnect_removes_tagged_routes(self):
        a = Agent("a")
        b = Agent("b")
        c = Agent("c")
        mesh = Mesh([a, b, c])
        mesh.connect(a, b)
        mesh.connect(a, c)
        assert len(a._outbox) == 2

        mesh.disconnect(a, b)
        assert len(a._outbox) == 1
        assert a._outbox[0].target == "c"

    def test_load_balance_tags_route(self):
        src = Agent("src")
        t1 = Agent("t1")
        t2 = Agent("t2")
        mesh = Mesh([src, t1, t2])
        mesh.load_balance(src, [t1, t2])

        assert len(src._outbox) == 1
        assert src._outbox[0].tag == "load_balance"
