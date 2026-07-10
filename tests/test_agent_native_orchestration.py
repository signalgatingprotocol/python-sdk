"""Tests for agent-native orchestration: tool calling, map_reduce, branch_workflow."""

import asyncio
import threading

import pytest

from signal_gating import (
    Agent,
    AgentContext,
    Mesh,
    Signal,
    ToolCallSignal,
    ToolResultSignal,
    ToolSpec,
)
from signal_gating.errors import ChannelClosed, MeshError

# --- Signal types for tests ---


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


# --- Agent.tool() ---


class TestAgentTool:
    """Tests for the Agent.tool() decorator and tool registry."""

    def test_tool_registration(self):
        agent = Agent("worker")

        @agent.tool(description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        tools = agent.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "add"
        assert tools[0].description == "Add two numbers"
        assert "a" in tools[0].parameters
        assert "b" in tools[0].parameters

    def test_tool_custom_name(self):
        agent = Agent("worker")

        @agent.tool(name="calculator", description="Do math")
        async def add(a: int, b: int) -> int:
            return a + b

        assert agent.get_tool("calculator") is not None
        assert agent.get_tool("add") is None

    def test_tool_schema_export(self):
        agent = Agent("worker")

        @agent.tool(description="Analyze data")
        async def analyze(data: str, depth: int = 1) -> dict:
            return {}

        schema = agent.tools_schema()
        assert len(schema) == 1
        assert schema[0]["name"] == "analyze"
        assert schema[0]["description"] == "Analyze data"
        assert schema[0]["parameters"]["data"]["required"] is True
        assert schema[0]["parameters"]["depth"]["required"] is False
        assert schema[0]["parameters"]["depth"]["default"] == 1

    def test_tool_docstring_as_description(self):
        agent = Agent("worker")

        @agent.tool()
        async def analyze(data: str) -> str:
            """Analyze the given data thoroughly."""
            return data

        tools = agent.list_tools()
        assert tools[0].description == "Analyze the given data thoroughly."

    def test_multiple_tools(self):
        agent = Agent("worker")

        @agent.tool(description="Tool A")
        async def tool_a() -> str:
            return "a"

        @agent.tool(description="Tool B")
        async def tool_b() -> str:
            return "b"

        assert len(agent.list_tools()) == 2
        assert agent.get_tool("tool_a") is not None
        assert agent.get_tool("tool_b") is not None

    async def test_tool_call_signal_handling(self):
        """Test that an agent with tools automatically handles ToolCallSignal."""
        worker = Agent("worker")

        @worker.tool(description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        collector = Agent("collector")
        results: list[ToolResultSignal] = []

        @collector.on(ToolResultSignal)
        async def collect(signal: ToolResultSignal):
            results.append(signal)

        mesh = Mesh([worker, collector])
        mesh.connect(worker, collector)

        async with mesh:
            call = ToolCallSignal(
                tool_name="add",
                arguments={"a": 3, "b": 4},
                correlation_id="test123",
            )
            await worker.inbox.send(call)
            await asyncio.sleep(0.1)

        assert len(results) == 1
        assert results[0].tool_name == "add"
        assert results[0].result == 7
        assert results[0].error == ""

    async def test_tool_call_unknown_tool(self):
        """Test that calling an unknown tool returns an error."""
        worker = Agent("worker")

        @worker.tool(description="Dummy")
        async def dummy() -> str:
            return "ok"

        collector = Agent("collector")
        results: list[ToolResultSignal] = []

        @collector.on(ToolResultSignal)
        async def collect(signal: ToolResultSignal):
            results.append(signal)

        mesh = Mesh([worker, collector])
        mesh.connect(worker, collector)

        async with mesh:
            call = ToolCallSignal(
                tool_name="nonexistent",
                arguments={},
                correlation_id="test456",
            )
            await worker.inbox.send(call)
            await asyncio.sleep(0.1)

        assert len(results) == 1
        assert results[0].error == "Unknown tool: nonexistent"

    async def test_tool_call_with_error(self):
        """Test that tool execution errors are caught and returned."""
        worker = Agent("worker")

        @worker.tool(description="Fails")
        async def broken(x: int) -> int:
            raise ValueError("bad input")

        collector = Agent("collector")
        results: list[ToolResultSignal] = []

        @collector.on(ToolResultSignal)
        async def collect(signal: ToolResultSignal):
            results.append(signal)

        mesh = Mesh([worker, collector])
        mesh.connect(worker, collector)

        async with mesh:
            call = ToolCallSignal(
                tool_name="broken",
                arguments={"x": 1},
                correlation_id="test789",
            )
            await worker.inbox.send(call)
            await asyncio.sleep(0.1)

        assert len(results) == 1
        assert "ValueError: bad input" in results[0].error

    async def test_sync_tool(self):
        """Test that synchronous tool functions work."""
        worker = Agent("worker")

        @worker.tool(description="Sync add")
        def add_sync(a: int, b: int) -> int:
            return a + b

        collector = Agent("collector")
        results: list[ToolResultSignal] = []

        @collector.on(ToolResultSignal)
        async def collect(signal: ToolResultSignal):
            results.append(signal)

        mesh = Mesh([worker, collector])
        mesh.connect(worker, collector)

        async with mesh:
            call = ToolCallSignal(
                tool_name="add_sync",
                arguments={"a": 10, "b": 20},
                correlation_id="sync_test",
            )
            await worker.inbox.send(call)
            await asyncio.sleep(0.1)

        assert len(results) == 1
        assert results[0].result == 30

    async def test_sync_tool_runs_off_event_loop(self):
        """A blocking sync tool must not stall every agent in the mesh."""
        worker = Agent("worker")
        event_loop_thread = threading.get_ident()

        @worker.tool(description="Report the execution thread")
        def execution_thread() -> int:
            return threading.get_ident()

        mesh = Mesh([worker])
        async with mesh:
            tool_thread = await mesh.call_tool(worker, "execution_thread")

        assert tool_thread != event_loop_thread


# --- Mesh.call_tool() ---


class TestMeshCallTool:
    """Tests for Mesh.call_tool(), the agent-native RPC primitive."""

    async def test_call_tool_basic(self):
        worker = Agent("worker")

        @worker.tool(description="Multiply")
        async def multiply(a: int, b: int) -> int:
            return a * b

        mesh = Mesh([worker])

        async with mesh:
            result = await mesh.call_tool(worker, "multiply", a=6, b=7)
            assert result == 42

    async def test_call_tool_error_raises(self):
        worker = Agent("worker")

        @worker.tool(description="Fails")
        async def fail() -> None:
            raise RuntimeError("oops")

        mesh = Mesh([worker])

        async with mesh:
            with pytest.raises(Exception, match="oops"):
                await mesh.call_tool(worker, "fail")

    async def test_discover_tools(self):
        analyst = Agent("analyst")
        coder = Agent("coder")

        @analyst.tool(description="Analyze data")
        async def analyze(data: str) -> str:
            return data

        @coder.tool(description="Write code")
        async def code(spec: str) -> str:
            return spec

        @coder.tool(description="Review code")
        async def review(code: str) -> str:
            return code

        mesh = Mesh([analyst, coder])

        all_tools = mesh.discover_tools()
        assert "analyst" in all_tools
        assert "coder" in all_tools
        assert len(all_tools["analyst"]) == 1
        assert len(all_tools["coder"]) == 2

        analyst_tools = mesh.discover_tools(analyst)
        assert "analyst" in analyst_tools
        assert len(analyst_tools) == 1

    async def test_discover_tools_empty(self):
        agent = Agent("empty")
        mesh = Mesh([agent])
        all_tools = mesh.discover_tools()
        assert len(all_tools) == 0


# --- Mesh.scatter() ---


class TestMeshScatter:
    """Tests for fail-fast, all-or-nothing scatter semantics."""

    async def test_timeout_names_missing_targets_and_cleans_up_captures(self):
        responsive = Agent("responsive")
        silent_second = Agent("silent-second")
        silent_first = Agent("silent-first")

        @responsive.on(TaskSignal)
        async def respond(signal: TaskSignal, ctx: AgentContext):
            await ctx.reply(ResultSignal(result=f"done:{signal.task}"))

        @silent_second.on(TaskSignal)
        @silent_first.on(TaskSignal)
        async def do_not_reply(_signal: TaskSignal):
            return

        mesh = Mesh([responsive, silent_second, silent_first])
        async with mesh:
            with pytest.raises(
                asyncio.TimeoutError,
                match=(
                    r"Scatter timed out after 0\.05s waiting for agents: "
                    r"'silent-second', 'silent-first'"
                ),
            ):
                await mesh.scatter(
                    TaskSignal(task="analyze"),
                    [responsive, silent_second, silent_first],
                    timeout=0.05,
                )

        assert responsive._outbox == []
        assert silent_second._outbox == []
        assert silent_first._outbox == []

    async def test_duplicate_resolved_target_fails_before_dispatch(self):
        worker = Agent("worker")
        handled = 0

        @worker.on(Signal)
        async def count_handler(_signal: Signal):
            nonlocal handled
            handled += 1

        mesh = Mesh([worker])
        async with mesh:
            with pytest.raises(
                MeshError,
                match=r"scatter targets must be unique; duplicate agent: 'worker'",
            ):
                await mesh.scatter(Signal(), [worker, "worker"])

            assert handled == 0
            assert worker._outbox == []

    async def test_dispatch_failure_cleans_up_all_captures(self):
        first = Agent("first")
        stopped = Agent("stopped")
        first_processed = asyncio.Event()

        @first.on(Signal)
        async def observe_partial_dispatch(_signal: Signal):
            first_processed.set()

        mesh = Mesh([first, stopped])
        async with mesh:
            await stopped.stop()

            with pytest.raises(ChannelClosed):
                await mesh.scatter(Signal(), [first, stopped])

            await asyncio.wait_for(first_processed.wait(), timeout=0.5)
            assert first._outbox == []
            assert stopped._outbox == []

    async def test_cancellation_cleans_up_all_captures(self):
        silent = Agent("silent")
        started = asyncio.Event()
        release = asyncio.Event()

        @silent.on(Signal)
        async def wait_without_reply(_signal: Signal):
            started.set()
            await release.wait()

        mesh = Mesh([silent])
        async with mesh:
            scatter = asyncio.create_task(mesh.scatter(Signal(), [silent]))
            await asyncio.wait_for(started.wait(), timeout=0.5)
            scatter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await scatter

            assert silent._outbox == []
            release.set()


# --- Mesh.map_reduce() ---


class TestMeshMapReduce:
    """Tests for Mesh.map_reduce(), the parallel map-reduce primitive."""

    async def test_map_reduce_basic(self):
        mapper1 = Agent("mapper1")
        mapper2 = Agent("mapper2")
        reducer = Agent("reducer")

        @mapper1.on(Signal)
        async def m1(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal.with_metadata(analysis="m1_result"))

        @mapper2.on(Signal)
        async def m2(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal.with_metadata(analysis="m2_result"))

        @reducer.on(Signal)
        async def reduce(signal: Signal, ctx: AgentContext):
            responses = signal.metadata.get("responses", [])
            combined = [r.get("metadata", {}).get("analysis", "") for r in responses]
            await ctx.reply(signal.with_metadata(combined=combined))

        mesh = Mesh([mapper1, mapper2, reducer])

        async with mesh:
            result = await mesh.map_reduce(
                Signal(),
                mappers=[mapper1, mapper2],
                reducer=reducer,
                timeout=5.0,
            )
            assert "combined" in result.metadata
            combined = result.metadata["combined"]
            assert "m1_result" in combined
            assert "m2_result" in combined

    async def test_map_reduce_tracing(self):
        mapper = Agent("mapper")
        reducer = Agent("reducer")

        @mapper.on(Signal)
        async def m(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal.with_metadata(done=True))

        @reducer.on(Signal)
        async def r(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal.with_metadata(reduced=True))

        mesh = Mesh([mapper, reducer])

        async with mesh:
            await mesh.map_reduce(Signal(), [mapper], reducer, timeout=5.0)

        # Verify tracing recorded map_reduce events
        spans = mesh.tracer._spans
        actions = [s.action for s in spans]
        assert "reduce_start" in actions
        assert "reduce_complete" in actions

    async def test_map_reduce_empty_mappers_raises(self):
        reducer = Agent("reducer")
        mesh = Mesh([reducer])

        async with mesh:
            with pytest.raises(Exception, match="at least one mapper"):
                await mesh.map_reduce(Signal(), [], reducer)

    async def test_mapper_timeout_never_invokes_reducer(self):
        responsive = Agent("responsive")
        silent = Agent("silent")
        reducer = Agent("reducer")
        reducer_called = False

        @responsive.on(Signal)
        async def respond(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal.with_metadata(mapper="responsive"))

        @silent.on(Signal)
        async def do_not_reply(_signal: Signal):
            return

        @reducer.on(Signal)
        async def reduce(signal: Signal, ctx: AgentContext):
            nonlocal reducer_called
            reducer_called = True
            await ctx.reply(signal)

        mesh = Mesh([responsive, silent, reducer])
        async with mesh:
            with pytest.raises(
                asyncio.TimeoutError,
                match=r"waiting for agents: 'silent'",
            ):
                await mesh.map_reduce(
                    Signal(),
                    mappers=[responsive, silent],
                    reducer=reducer,
                    timeout=1.0,
                    mapper_timeout=0.05,
                )

        assert reducer_called is False


# --- Mesh.branch_workflow() ---


class TestMeshBranchWorkflow:
    """Tests for Mesh.branch_workflow(), conditional branching workflows."""

    async def test_branch_workflow_routes_correctly(self):
        fast_agent = Agent("fast")
        thorough_agent = Agent("thorough")
        reviewer = Agent("reviewer")

        @fast_agent.on(Signal)
        async def fast(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal.with_metadata(path="fast"))

        @thorough_agent.on(Signal)
        async def thorough(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal.with_metadata(path="thorough_step1"))

        @reviewer.on(Signal)
        async def review(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal.with_metadata(
                path=signal.metadata.get("path", "") + "+reviewed"
            ))

        mesh = Mesh([fast_agent, thorough_agent, reviewer])

        async with mesh:
            # High priority -> thorough path
            result = await mesh.branch_workflow(
                Signal(priority=9),
                router=lambda s: "critical" if s.priority >= 8 else "normal",
                branches={
                    "critical": [thorough_agent, reviewer],
                    "normal": [fast_agent],
                },
                timeout=5.0,
            )
            assert "reviewed" in result.metadata.get("path", "")

            # Low priority -> fast path
            result2 = await mesh.branch_workflow(
                Signal(priority=2),
                router=lambda s: "critical" if s.priority >= 8 else "normal",
                branches={
                    "critical": [thorough_agent, reviewer],
                    "normal": [fast_agent],
                },
                timeout=5.0,
            )
            assert result2.metadata.get("path") == "fast"

    async def test_branch_workflow_unknown_branch_raises(self):
        agent = Agent("a")
        mesh = Mesh([agent])

        async with mesh:
            with pytest.raises(Exception, match="unknown branch"):
                await mesh.branch_workflow(
                    Signal(),
                    router=lambda s: "nonexistent",
                    branches={"other": [agent]},
                    timeout=5.0,
                )

    async def test_branch_workflow_empty_branches_raises(self):
        mesh = Mesh()

        async with mesh:
            with pytest.raises(Exception, match="at least one branch"):
                await mesh.branch_workflow(
                    Signal(),
                    router=lambda s: "x",
                    branches={},
                )

    async def test_branch_workflow_tracing(self):
        agent = Agent("a")

        @agent.on(Signal)
        async def handle(signal: Signal, ctx: AgentContext):
            await ctx.reply(signal)

        mesh = Mesh([agent])

        async with mesh:
            await mesh.branch_workflow(
                Signal(),
                router=lambda s: "main",
                branches={"main": [agent]},
                timeout=5.0,
            )

        actions = [s.action for s in mesh.tracer._spans]
        assert "branch_selected" in actions


# --- ToolCallSignal / ToolResultSignal ---


class TestToolSignals:
    """Tests for the tool protocol signal types."""

    def test_tool_call_signal_fields(self):
        sig = ToolCallSignal(tool_name="analyze", arguments={"data": "test"})
        assert sig.tool_name == "analyze"
        assert sig.arguments == {"data": "test"}

    def test_tool_result_signal_fields(self):
        sig = ToolResultSignal(tool_name="analyze", result={"output": 42})
        assert sig.tool_name == "analyze"
        assert sig.result == {"output": 42}
        assert sig.error == ""

    def test_tool_result_signal_error(self):
        sig = ToolResultSignal(tool_name="broken", error="ValueError: bad")
        assert sig.error == "ValueError: bad"
        assert sig.result is None

    def test_tool_signals_are_immutable(self):
        sig = ToolCallSignal(tool_name="test", arguments={})
        with pytest.raises(Exception):
            sig.tool_name = "other"  # type: ignore[misc]


# --- ToolSpec ---


class TestToolSpec:
    def test_tool_spec_creation(self):
        spec = ToolSpec(name="test", description="A test tool")
        assert spec.name == "test"
        assert spec.description == "A test tool"
        assert spec.parameters == {}
        assert spec.fn is None

    def test_tool_spec_with_params(self):
        spec = ToolSpec(
            name="analyze",
            description="Analyze data",
            parameters={"data": {"type": "str", "required": True}},
        )
        assert spec.parameters["data"]["required"] is True
