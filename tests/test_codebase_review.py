"""Tests for codebase review improvements: bug fixes and new features."""

import asyncio

import pytest

from signal_gating import (
    Agent,
    AgentContext,
    AgentPool,
    Gate,
    Mesh,
    Pipeline,
    Signal,
)

# --- Signal types for testing ---


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


# =============================================================================
# BUG FIX: Gate.throttle() concurrency safety
# =============================================================================


class TestThrottleConcurrency:
    """Gate.throttle() was using mutable state without a lock.
    Under concurrent access, reads/writes to state dict could interleave.
    """

    async def test_throttle_concurrent_access(self):
        """Multiple concurrent calls should not corrupt throttle state."""
        gate = Gate.throttle(10, name="test_throttle")
        signal = Signal()

        # Fire 50 concurrent signals through the throttle
        results = await asyncio.gather(
            *(gate.process(signal) for _ in range(50))
        )

        # At 10/sec, some should pass and some should be dropped
        passed = [r for r in results if r is not None]
        dropped = [r for r in results if r is None]
        assert len(passed) >= 1  # At least the first one passes
        assert len(dropped) >= 1  # Some should be throttled

    async def test_throttle_sequential_still_works(self):
        """Throttle should still work correctly for sequential calls."""
        gate = Gate.throttle(1000, name="fast_throttle")
        signal = Signal()

        result = await gate.process(signal)
        assert result is not None


# =============================================================================
# BUG FIX: AgentPool counter separation
# =============================================================================


class TestPoolCounterSeparation:
    """AgentPool was sharing itertools.count() between worker naming and
    round-robin selection, causing biased distribution.
    """

    async def test_round_robin_starts_at_first_worker(self):
        """Round-robin should start at worker[0], not worker[N]."""
        pool = AgentPool("test", size=3)
        first = pool.select_worker()
        assert first.name == "test[0]"

    async def test_round_robin_cycles_evenly(self):
        """Round-robin should cycle through all workers evenly."""
        pool = AgentPool("test", size=3)
        workers = [pool.select_worker() for _ in range(6)]
        names = [w.name for w in workers]
        assert names == [
            "test[0]", "test[1]", "test[2]",
            "test[0]", "test[1]", "test[2]",
        ]

    async def test_scaling_doesnt_break_robin(self):
        """Scaling up should not affect round-robin counter."""
        pool = AgentPool("test", size=2)
        # Select first worker
        first = pool.select_worker()
        assert first.name == "test[0]"

        # Scale up
        pool.scale_to(4)

        # Round-robin should continue from where it was
        second = pool.select_worker()
        assert second.name == "test[1]"


# =============================================================================
# NEW: Gate.tap() (side-effect observability)
# =============================================================================


class TestGateTap:
    """Gate.tap() executes a side-effect without affecting signal flow."""

    async def test_tap_passes_signal_unchanged(self):
        signal = TaskSignal(task="test", priority=5)
        seen: list[Signal] = []
        gate = Gate.tap(lambda s: seen.append(s))

        result = await gate.process(signal)
        assert result is signal  # Same object, unchanged
        assert len(seen) == 1
        assert seen[0] is signal

    async def test_tap_with_async_fn(self):
        seen: list[Signal] = []

        async def async_tap(s: Signal) -> None:
            await asyncio.sleep(0)
            seen.append(s)

        gate = Gate.tap(async_tap)
        signal = Signal()
        result = await gate.process(signal)
        assert result is signal
        assert len(seen) == 1

    async def test_tap_composes_with_other_gates(self):
        """Tap should compose cleanly with >> and other operators."""
        log: list[str] = []
        pipeline = (
            Gate.tap(lambda s: log.append("before"))
            >> Gate.by_priority(3)
            >> Gate.tap(lambda s: log.append("after"))
        )

        # High priority: passes through
        result = await pipeline.process(Signal(priority=5))
        assert result is not None
        assert log == ["before", "after"]

        # Low priority: rejected by priority gate, second tap never runs
        log.clear()
        result = await pipeline.process(Signal(priority=1))
        assert result is None
        assert log == ["before"]

    async def test_tap_name(self):
        gate = Gate.tap(lambda s: None)
        assert gate.name == "tap"

        gate = Gate.tap(lambda s: None, name="audit")
        assert gate.name == "audit"


# =============================================================================
# NEW: Mesh.pipe() (fluent chain-connect API)
# =============================================================================


class TestMeshPipe:
    """Mesh.pipe() connects agents in a linear chain."""

    async def test_pipe_connects_linearly(self):
        a = Agent("a")
        b = Agent("b")
        c = Agent("c")
        mesh = Mesh([a, b, c])
        mesh.pipe(a, b, c)

        assert len(mesh.edges) == 2
        assert mesh.edges[0].source.name == "a"
        assert mesh.edges[0].target.name == "b"
        assert mesh.edges[1].source.name == "b"
        assert mesh.edges[1].target.name == "c"

    async def test_pipe_with_gate(self):
        a = Agent("a")
        b = Agent("b")
        c = Agent("c")
        mesh = Mesh([a, b, c])
        gate = Gate.by_priority(3)
        mesh.pipe(a, b, c, gate=gate)

        for edge in mesh.edges:
            assert edge.gate is not None

    async def test_pipe_signals_flow_through(self):
        received: list[str] = []

        a = Agent("a")
        b = Agent("b")
        c = Agent("c")

        @b.on(TaskSignal)
        async def handle_b(signal: TaskSignal, ctx: AgentContext):
            received.append(f"b:{signal.task}")
            await ctx.emit(signal.child(task=f"{signal.task}_processed"))

        @c.on(TaskSignal)
        async def handle_c(signal: TaskSignal):
            received.append(f"c:{signal.task}")

        mesh = Mesh([a, b, c])
        mesh.pipe(a, b, c)

        async with mesh:
            await a.emit(TaskSignal(task="go"))
            await asyncio.sleep(0.1)

        assert "b:go" in received
        assert "c:go_processed" in received

    async def test_pipe_requires_two_agents(self):
        a = Agent("a")
        mesh = Mesh([a])
        with pytest.raises(Exception, match="at least 2"):
            mesh.pipe(a)

    async def test_pipe_by_name(self):
        a = Agent("a")
        b = Agent("b")
        c = Agent("c")
        mesh = Mesh([a, b, c])
        mesh.pipe("a", "b", "c")
        assert len(mesh.edges) == 2


# =============================================================================
# NEW: Mesh.visualize() (topology introspection)
# =============================================================================


class TestMeshVisualize:
    """Mesh.visualize() returns a text representation of the topology."""

    async def test_basic_visualization(self):
        a = Agent("fetcher")
        b = Agent("parser")
        c = Agent("storer")
        mesh = Mesh([a, b, c])
        mesh.connect(a, b)
        mesh.connect(b, c)

        viz = mesh.visualize()
        assert "fetcher" in viz
        assert "parser" in viz
        assert "storer" in viz
        assert "3 agents" in viz
        assert "2 edges" in viz

    async def test_visualization_shows_gates(self):
        a = Agent("a")
        b = Agent("b")
        mesh = Mesh([a, b])
        mesh.connect(a, b, gate=Gate.by_priority(5, name="prio"))

        viz = mesh.visualize()
        assert "prio" in viz

    async def test_visualization_shows_capabilities(self):
        a = Agent("analyst")
        mesh = Mesh([a])
        mesh.declare_capabilities(a, "analysis", "summarization")

        viz = mesh.visualize()
        assert "analysis" in viz
        assert "summarization" in viz

    async def test_visualization_shows_topics(self):
        a = Agent("a")
        b = Agent("b")
        mesh = Mesh([a, b])
        mesh.create_topic("events")
        mesh.subscribe(a, "events")

        viz = mesh.visualize()
        assert "events" in viz


# =============================================================================
# NEW: Agent.on_error hook
# =============================================================================


class TestAgentOnError:
    """Agent.on_error registers hooks called when signal processing fails."""

    async def test_on_error_hook_called_on_handler_failure(self):
        errors: list[tuple[str, str]] = []
        agent = Agent("test")

        @agent.on(TaskSignal)
        async def bad_handler(signal: TaskSignal):
            raise ValueError("boom")

        @agent.on_error
        async def capture_error(signal: Signal, error: Exception):
            errors.append((type(signal).__name__, str(error)))

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="fail"))
            await asyncio.sleep(0.1)

        assert len(errors) == 1
        assert errors[0] == ("TaskSignal", "boom")

    async def test_on_error_sync_hook(self):
        errors: list[str] = []
        agent = Agent("test")

        @agent.on(TaskSignal)
        async def bad_handler(signal: TaskSignal):
            raise RuntimeError("sync_boom")

        @agent.on_error
        def sync_handler(signal: Signal, error: Exception):
            errors.append(str(error))

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="fail"))
            await asyncio.sleep(0.1)

        assert "sync_boom" in errors

    async def test_on_error_multiple_hooks(self):
        count = {"a": 0, "b": 0}
        agent = Agent("test")

        @agent.on(TaskSignal)
        async def bad_handler(signal: TaskSignal):
            raise ValueError("x")

        @agent.on_error
        def hook_a(signal: Signal, error: Exception):
            count["a"] += 1

        @agent.on_error
        def hook_b(signal: Signal, error: Exception):
            count["b"] += 1

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="fail"))
            await asyncio.sleep(0.1)

        assert count["a"] == 1
        assert count["b"] == 1

    async def test_on_error_still_adds_to_dlq(self):
        """Error hooks don't replace DLQ; they supplement it."""
        agent = Agent("test")

        @agent.on(TaskSignal)
        async def bad_handler(signal: TaskSignal):
            raise ValueError("x")

        @agent.on_error
        def hook(signal: Signal, error: Exception):
            pass

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="fail"))
            await asyncio.sleep(0.1)

        assert agent.dead_letters.count == 1


# =============================================================================
# NEW: Pipeline operator overloading
# =============================================================================


class TestPipelineOperators:
    """Pipeline supports >> for composition with Gates and other Pipelines."""

    async def test_pipeline_rshift_gate(self):
        p = Pipeline([Gate.by_priority(3)])
        p2 = p >> Gate.passthrough()
        assert len(p2) == 2
        assert isinstance(p2, Pipeline)

    async def test_pipeline_rshift_pipeline(self):
        p1 = Pipeline([Gate.by_priority(3)])
        p2 = Pipeline([Gate.passthrough()])
        p3 = p1 >> p2
        assert len(p3) == 2

    async def test_gate_rshift_pipeline(self):
        gate = Gate.by_priority(3)
        pipeline = Pipeline([Gate.passthrough()])
        result = gate >> pipeline
        assert isinstance(result, Pipeline)
        assert len(result) == 2

    async def test_composed_pipeline_processes_correctly(self):
        p = (
            Pipeline([Gate.by_priority(3)])
            >> Gate.passthrough()
        )

        # High priority passes
        result = await p.process(Signal(priority=5))
        assert result is not None

        # Low priority rejected
        result = await p.process(Signal(priority=1))
        assert result is None

    async def test_original_pipeline_unchanged(self):
        """>> creates new pipelines, doesn't mutate originals."""
        p1 = Pipeline([Gate.by_priority(3)])
        p2 = p1 >> Gate.passthrough()
        assert len(p1) == 1
        assert len(p2) == 2


# =============================================================================
# NEW: AgentPool.on_error
# =============================================================================


class TestPoolOnError:
    """AgentPool.on_error propagates to all workers."""

    async def test_pool_on_error_fires_for_all_workers(self):
        errors: list[str] = []
        pool = AgentPool("test", size=2)

        @pool.on(TaskSignal)
        async def bad_handler(signal: TaskSignal):
            raise ValueError("pool_boom")

        @pool.on_error
        def capture(signal: Signal, error: Exception):
            errors.append(str(error))

        mesh = Mesh()
        mesh.add_pool(pool)

        async with mesh:
            # Send to first worker
            await pool.workers[0].inbox.send(TaskSignal(task="fail"))
            await asyncio.sleep(0.1)

        assert "pool_boom" in errors

    async def test_pool_on_error_applies_to_scaled_workers(self):
        errors: list[str] = []
        pool = AgentPool("test", size=1)

        @pool.on(TaskSignal)
        async def bad_handler(signal: TaskSignal):
            raise ValueError("scale_boom")

        @pool.on_error
        def capture(signal: Signal, error: Exception):
            errors.append(str(error))

        # Scale up: new worker should also have the error hook
        pool.scale_to(2)
        new_worker = pool.workers[1]
        assert len(new_worker._on_error_hooks) == 1
