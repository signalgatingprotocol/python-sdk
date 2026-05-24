"""Tests for agent-native enhancements: Gate.window, Gate.map, Mesh.workflow, Mesh.race."""

import asyncio

import pytest

from signal_gating import Agent, Gate, Mesh, Signal

# -- Signal types for testing --


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


# =============================================================================
# Gate.window() tests
# =============================================================================


class TestGateWindow:
    async def test_window_rejects_below_min_signals(self):
        """Window gate rejects when fewer than min_signals in the window."""
        gate = Gate.window(seconds=10, min_signals=3)
        sig = Signal()
        result = await gate.process(sig)
        assert result is None  # Only 1 signal, need 3

    async def test_window_passes_at_min_signals(self):
        """Window gate passes once min_signals threshold is reached."""
        gate = Gate.window(seconds=10, min_signals=3)
        for _ in range(2):
            await gate.process(Signal())
        result = await gate.process(Signal(priority=5))
        assert result is not None
        assert result.metadata["window_size"] == 3

    async def test_window_enriches_with_rate_metadata(self):
        """Window gate adds window_size and window_rate metadata."""
        gate = Gate.window(seconds=10, min_signals=1)
        result = await gate.process(Signal())
        assert result is not None
        assert "window_size" in result.metadata
        assert "window_rate" in result.metadata
        assert result.metadata["window_size"] == 1
        assert result.metadata["window_rate"] == round(1 / 10, 4)

    async def test_window_accumulates_signals(self):
        """Window size grows as signals arrive within the window."""
        gate = Gate.window(seconds=60, min_signals=1)
        for i in range(5):
            result = await gate.process(Signal())
        assert result is not None
        assert result.metadata["window_size"] == 5

    async def test_window_evicts_old_signals(self):
        """Signals outside the time window are evicted."""
        gate = Gate.window(seconds=0.05, min_signals=1)
        await gate.process(Signal())
        await asyncio.sleep(0.1)  # Let window expire
        result = await gate.process(Signal())
        assert result is not None
        assert result.metadata["window_size"] == 1  # Old signal evicted

    async def test_window_rejects_invalid_seconds(self):
        """Window raises ValueError for non-positive seconds."""
        with pytest.raises(ValueError, match="window seconds must be > 0"):
            Gate.window(seconds=0)
        with pytest.raises(ValueError, match="window seconds must be > 0"):
            Gate.window(seconds=-1)

    async def test_window_default_min_signals(self):
        """Default min_signals is 1; every signal passes with enrichment."""
        gate = Gate.window(seconds=60)
        result = await gate.process(Signal())
        assert result is not None

    async def test_window_composable_with_operators(self):
        """Window gate composes with other gates via operators."""
        window = Gate.window(seconds=60, min_signals=1)
        priority = Gate.by_priority(3)
        composed = window >> priority

        # Low priority rejected after window
        result = await composed.process(Signal(priority=1))
        assert result is None

        # High priority passes through both
        result = await composed.process(Signal(priority=5))
        assert result is not None

    async def test_window_concurrent_safety(self):
        """Window gate handles concurrent signal processing safely."""
        gate = Gate.window(seconds=60, min_signals=1)
        signals = [Signal() for _ in range(20)]
        results = await asyncio.gather(*(gate.process(s) for s in signals))
        passed = [r for r in results if r is not None]
        assert len(passed) == 20  # All should pass with min_signals=1


# =============================================================================
# Gate.map() tests
# =============================================================================


class TestGateMap:
    async def test_map_sync_transform(self):
        """Map gate works with sync transformation functions."""
        gate = Gate.map(lambda s: s.evolve(priority=99))
        result = await gate.process(Signal(priority=1))
        assert result is not None
        assert result.priority == 99

    async def test_map_async_transform(self):
        """Map gate works with async transformation functions."""

        async def async_enrich(signal: Signal) -> Signal:
            await asyncio.sleep(0)  # Simulate async work
            return signal.with_metadata(enriched=True)

        gate = Gate.map(async_enrich)
        result = await gate.process(Signal())
        assert result is not None
        assert result.metadata["enriched"] is True

    async def test_map_can_reject(self):
        """Map gate can reject signals by returning None."""
        gate = Gate.map(lambda s: s if s.priority > 5 else None)
        assert await gate.process(Signal(priority=3)) is None
        assert await gate.process(Signal(priority=8)) is not None

    async def test_map_async_rejection(self):
        """Map gate supports async rejection."""

        async def conditional(signal: Signal) -> Signal | None:
            return signal if signal.priority > 0 else None

        gate = Gate.map(conditional)
        assert await gate.process(Signal(priority=0)) is None
        assert await gate.process(Signal(priority=1)) is not None

    async def test_map_composable(self):
        """Map gate composes with other gates."""
        add_priority = Gate.map(lambda s: s.evolve(priority=s.priority + 10))
        check = Gate.by_priority(15)
        composed = add_priority >> check

        result = await composed.process(Signal(priority=3))
        assert result is None  # 3 + 10 = 13 < 15

        result = await composed.process(Signal(priority=6))
        assert result is not None  # 6 + 10 = 16 >= 15

    async def test_map_preserves_signal_type(self):
        """Map gate preserves the signal subclass type."""

        gate = Gate.map(lambda s: s.with_metadata(mapped=True))
        result = await gate.process(TaskSignal(task="test"))
        assert result is not None
        assert isinstance(result, TaskSignal)
        assert result.task == "test"  # type: ignore[attr-defined]

    async def test_map_name(self):
        """Map gate has correct default and custom names."""
        gate = Gate.map(lambda s: s)
        assert gate.name == "map"

        gate = Gate.map(lambda s: s, name="my_map")
        assert gate.name == "my_map"


# =============================================================================
# Gate.transform() async support tests
# =============================================================================


class TestGateTransformAsync:
    async def test_transform_accepts_async(self):
        """Transform gate now properly accepts async functions."""

        async def async_transform(signal: Signal) -> Signal:
            await asyncio.sleep(0)
            return signal.evolve(priority=42)

        gate = Gate.transform(async_transform)
        result = await gate.process(Signal())
        assert result is not None
        assert result.priority == 42

    async def test_transform_still_works_sync(self):
        """Transform gate still works with sync functions."""
        gate = Gate.transform(lambda s: s.evolve(priority=7))
        result = await gate.process(Signal())
        assert result is not None
        assert result.priority == 7


# =============================================================================
# Mesh.workflow() tests
# =============================================================================


class TestMeshWorkflow:
    async def test_workflow_single_step(self):
        """Workflow with a single step executes correctly."""
        agent = Agent("worker")

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal):
            await agent.reply(signal, ResultSignal(result=f"done:{signal.task}"))

        mesh = Mesh([agent])
        async with mesh:
            result = await mesh.workflow(
                TaskSignal(task="build"),
                steps=[agent],
                timeout=5.0,
            )
            assert isinstance(result, ResultSignal)
            assert result.result == "done:build"

    async def test_workflow_multi_step_chain(self):
        """Workflow chains results through multiple agents sequentially."""
        step1 = Agent("fetcher")
        step2 = Agent("parser")
        step3 = Agent("storer")

        @step1.on(Signal)
        async def fetch(signal: Signal):
            await step1.reply(signal, signal.with_metadata(fetched=True))

        @step2.on(Signal)
        async def parse(signal: Signal):
            await step2.reply(signal, signal.with_metadata(parsed=True))

        @step3.on(Signal)
        async def store(signal: Signal):
            await step3.reply(signal, signal.with_metadata(stored=True))

        mesh = Mesh([step1, step2, step3])
        async with mesh:
            result = await mesh.workflow(
                Signal(),
                steps=[step1, step2, step3],
                timeout=5.0,
            )
            # Each step enriched the signal
            assert result.metadata.get("fetched") is True
            assert result.metadata.get("parsed") is True
            assert result.metadata.get("stored") is True

    async def test_workflow_preserves_trace_lineage(self):
        """Workflow maintains signal trace_id across all steps."""
        a = Agent("a")
        b = Agent("b")

        @a.on(Signal)
        async def handle_a(signal: Signal):
            await a.reply(signal, signal.with_metadata(step="a"))

        @b.on(Signal)
        async def handle_b(signal: Signal):
            await b.reply(signal, signal.with_metadata(step="b"))

        mesh = Mesh([a, b])
        initial = Signal()
        async with mesh:
            await mesh.workflow(initial, steps=[a, b], timeout=5.0)
            # Trace is recorded
            assert mesh.tracer.span_count > 0

    async def test_workflow_timeout(self):
        """Workflow times out when an agent doesn't respond."""
        slow = Agent("slow")

        @slow.on(Signal)
        async def handle(signal: Signal):
            await asyncio.sleep(10)  # Never responds in time

        mesh = Mesh([slow])
        async with mesh:
            with pytest.raises(asyncio.TimeoutError):
                await mesh.workflow(
                    Signal(), steps=[slow], timeout=0.1, step_timeout=0.1,
                )

    async def test_workflow_empty_steps_raises(self):
        """Workflow raises MeshError for empty steps list."""
        from signal_gating.errors import MeshError

        mesh = Mesh()
        async with mesh:
            with pytest.raises(MeshError, match="at least one step"):
                await mesh.workflow(Signal(), steps=[])

    async def test_workflow_with_string_agent_names(self):
        """Workflow accepts agent names as strings."""
        agent = Agent("worker")

        @agent.on(Signal)
        async def handle(signal: Signal):
            await agent.reply(signal, signal.with_metadata(processed=True))

        mesh = Mesh([agent])
        async with mesh:
            result = await mesh.workflow(
                Signal(), steps=["worker"], timeout=5.0,
            )
            assert result.metadata.get("processed") is True

    async def test_workflow_traces_steps(self):
        """Workflow records trace spans for each step."""
        agent = Agent("worker")

        @agent.on(Signal)
        async def handle(signal: Signal):
            await agent.reply(signal, signal.with_metadata(done=True))

        mesh = Mesh([agent])
        async with mesh:
            await mesh.workflow(Signal(), steps=[agent], timeout=5.0)

        workflow_spans = [
            s for s in mesh.tracer._spans if s.gate == "workflow"
        ]
        assert len(workflow_spans) >= 2  # step_start + step_complete


# =============================================================================
# Mesh.race() tests
# =============================================================================


class TestMeshRace:
    async def test_race_returns_first_response(self):
        """Race returns the response from the fastest agent."""
        fast = Agent("fast")
        slow = Agent("slow")

        @fast.on(Signal)
        async def handle_fast(signal: Signal):
            await fast.reply(signal, ResultSignal(result="fast_wins"))

        @slow.on(Signal)
        async def handle_slow(signal: Signal):
            await asyncio.sleep(5)
            await slow.reply(signal, ResultSignal(result="slow_wins"))

        mesh = Mesh([fast, slow])
        async with mesh:
            result = await mesh.race(
                Signal(), targets=[fast, slow], timeout=3.0,
            )
            assert isinstance(result, ResultSignal)
            assert result.result == "fast_wins"

    async def test_race_single_target(self):
        """Race with single target works like a simple request."""
        agent = Agent("only")

        @agent.on(Signal)
        async def handle(signal: Signal):
            await agent.reply(signal, ResultSignal(result="done"))

        mesh = Mesh([agent])
        async with mesh:
            result = await mesh.race(
                Signal(), targets=[agent], timeout=5.0,
            )
            assert isinstance(result, ResultSignal)
            assert result.result == "done"

    async def test_race_timeout(self):
        """Race times out when no agent responds in time."""
        slow = Agent("slow")

        @slow.on(Signal)
        async def handle(signal: Signal):
            await asyncio.sleep(10)

        mesh = Mesh([slow])
        async with mesh:
            with pytest.raises(asyncio.TimeoutError):
                await mesh.race(Signal(), targets=[slow], timeout=0.1)

    async def test_race_empty_targets_raises(self):
        """Race raises MeshError for empty targets list."""
        from signal_gating.errors import MeshError

        mesh = Mesh()
        async with mesh:
            with pytest.raises(MeshError, match="at least one target"):
                await mesh.race(Signal(), targets=[])

    async def test_race_cleans_up_outbox(self):
        """Race cleans up capture functions from agent outboxes after completion."""
        agent = Agent("worker")

        @agent.on(Signal)
        async def handle(signal: Signal):
            await agent.reply(signal, ResultSignal(result="done"))

        mesh = Mesh([agent])
        async with mesh:
            outbox_before = len(agent._outbox)
            await mesh.race(Signal(), targets=[agent], timeout=5.0)
            # Capture functions should be cleaned up
            assert len(agent._outbox) == outbox_before

    async def test_race_with_string_names(self):
        """Race accepts agent names as strings."""
        agent = Agent("worker")

        @agent.on(Signal)
        async def handle(signal: Signal):
            await agent.reply(signal, ResultSignal(result="done"))

        mesh = Mesh([agent])
        async with mesh:
            result = await mesh.race(
                Signal(), targets=["worker"], timeout=5.0,
            )
            assert isinstance(result, ResultSignal)

    async def test_race_traces_execution(self):
        """Race records trace spans for the race."""
        agent = Agent("worker")

        @agent.on(Signal)
        async def handle(signal: Signal):
            await agent.reply(signal, ResultSignal(result="done"))

        mesh = Mesh([agent])
        async with mesh:
            await mesh.race(Signal(), targets=[agent], timeout=5.0)

        race_spans = [s for s in mesh.tracer._spans if s.gate == "race"]
        assert len(race_spans) >= 1

    async def test_race_multiple_fast_agents(self):
        """Race handles multiple agents responding quickly."""
        agents = []
        for i in range(3):
            a = Agent(f"agent_{i}")

            @a.on(Signal)
            async def handle(signal: Signal, _a=a, _i=i):
                await _a.reply(signal, ResultSignal(result=f"agent_{_i}"))

            agents.append(a)

        mesh = Mesh(agents)
        async with mesh:
            result = await mesh.race(
                Signal(), targets=agents, timeout=5.0,
            )
            assert isinstance(result, ResultSignal)
            assert result.result.startswith("agent_")


# =============================================================================
# Integration tests: combining new features
# =============================================================================


class TestIntegration:
    async def test_window_gate_in_mesh(self):
        """Window gate works correctly within a mesh agent."""
        sender = Agent("sender")
        receiver = Agent("receiver", gates=[Gate.window(seconds=60, min_signals=3)])
        received: list[Signal] = []

        @receiver.on(Signal)
        async def handle(signal: Signal):
            received.append(signal)

        mesh = Mesh([sender, receiver])
        mesh.connect(sender, receiver)

        async with mesh:
            for i in range(5):
                await sender.emit(Signal(priority=i))
            await asyncio.sleep(0.1)

        # First 2 signals rejected (< min_signals), last 3 passed
        assert len(received) == 3

    async def test_map_gate_in_pipeline(self):
        """Map gate works in a pipeline with other gates."""
        from signal_gating import Pipeline

        pipeline = Pipeline([
            Gate.map(lambda s: s.evolve(priority=s.priority * 2)),
            Gate.by_priority(10),
        ])

        assert await pipeline.process(Signal(priority=3)) is None  # 6 < 10
        result = await pipeline.process(Signal(priority=6))  # 12 >= 10
        assert result is not None
        assert result.priority == 12

    async def test_workflow_then_race(self):
        """Workflow and race can be combined for complex orchestration."""
        prep = Agent("prep")
        fast = Agent("fast_analyzer")
        slow = Agent("slow_analyzer")

        @prep.on(Signal)
        async def handle_prep(signal: Signal):
            await prep.reply(signal, signal.with_metadata(prepared=True))

        @fast.on(Signal)
        async def handle_fast(signal: Signal):
            await fast.reply(signal, ResultSignal(result="fast_analysis"))

        @slow.on(Signal)
        async def handle_slow(signal: Signal):
            await asyncio.sleep(5)
            await slow.reply(signal, ResultSignal(result="slow_analysis"))

        mesh = Mesh([prep, fast, slow])
        async with mesh:
            # Step 1: Prepare via workflow
            prepared = await mesh.workflow(
                Signal(), steps=[prep], timeout=5.0,
            )
            assert prepared.metadata.get("prepared") is True

            # Step 2: Race analyzers
            result = await mesh.race(
                prepared, targets=[fast, slow], timeout=3.0,
            )
            assert isinstance(result, ResultSignal)
            assert result.result == "fast_analysis"
