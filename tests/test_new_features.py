"""Tests for new features: AgentContext, once(), scatter/gather, interceptors,
graceful shutdown, capability discovery, PriorityChannel.send_wait."""

import asyncio

import pytest

from signal_gating import Agent, AgentContext, Mesh, Signal
from signal_gating.channel import PriorityChannel

# --- Signal types for testing ---


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


# --- AgentContext Tests ---


class TestAgentContext:
    async def test_handler_receives_context(self):
        """Handlers with 2 params get AgentContext injected."""
        agent = Agent("worker")
        received_ctx = []

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received_ctx.append(ctx)

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert len(received_ctx) == 1
        assert received_ctx[0].agent_name == "worker"

    async def test_context_state_access(self):
        """Context provides access to agent state."""
        agent = Agent("worker")

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            count = ctx.state.get("count", 0)
            ctx.state["count"] = count + 1

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="a"))
            await agent.inbox.send(TaskSignal(task="b"))
            await asyncio.sleep(0.05)

        assert agent.state["count"] == 2

    async def test_context_emit(self):
        """Context can emit signals downstream."""
        sender = Agent("sender")
        receiver = Agent("receiver")
        received: list[Signal] = []

        @sender.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            await ctx.emit(ResultSignal(result=f"done:{signal.task}"))

        @receiver.on(ResultSignal)
        async def receive(signal: ResultSignal):
            received.append(signal)

        mesh = Mesh([sender, receiver])
        mesh.connect(sender, receiver)
        async with mesh:
            await sender.inbox.send(TaskSignal(task="work"))
            await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0].result == "done:work"

    async def test_context_reply(self):
        """Context can reply to the originating signal."""
        requester = Agent("requester")
        responder = Agent("responder")

        @responder.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            await ctx.reply(ResultSignal(result=f"completed:{signal.task}"))

        mesh = Mesh([requester, responder])
        mesh.connect(requester, responder)
        mesh.connect(responder, requester)

        async with mesh:
            response = await requester.request(
                TaskSignal(task="analyze"), timeout=2.0
            )

        assert isinstance(response, ResultSignal)
        assert response.result == "completed:analyze"

    async def test_handler_without_context_still_works(self):
        """Existing 1-param handlers continue to work."""
        agent = Agent("worker")
        received: list[Signal] = []

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal)

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert len(received) == 1


# --- once() Tests ---


class TestOnceHandler:
    async def test_once_fires_once(self):
        """Handler registered with once() fires exactly once."""
        agent = Agent("worker")
        calls: list[str] = []

        @agent.once(TaskSignal)
        async def handle(signal: TaskSignal):
            calls.append(signal.task)

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="first"))
            await asyncio.sleep(0.05)
            await agent.inbox.send(TaskSignal(task="second"))
            await asyncio.sleep(0.05)

        assert calls == ["first"]

    async def test_once_alongside_regular_handler(self):
        """once() handler doesn't interfere with regular handlers."""
        agent = Agent("worker")
        once_calls: list[str] = []
        regular_calls: list[str] = []

        @agent.on(TaskSignal)
        async def regular(signal: TaskSignal):
            regular_calls.append(signal.task)

        @agent.once(TaskSignal)
        async def one_time(signal: TaskSignal):
            once_calls.append(signal.task)

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="a"))
            await asyncio.sleep(0.05)
            await agent.inbox.send(TaskSignal(task="b"))
            await asyncio.sleep(0.05)

        assert once_calls == ["a"]
        assert regular_calls == ["a", "b"]


# --- Mesh Interceptor Tests ---


class TestMeshInterceptors:
    async def test_interceptor_sees_all_signals(self):
        """Interceptors see every signal flowing through edges."""
        a = Agent("a")
        b = Agent("b")
        intercepted: list[tuple[str, str]] = []

        def log_interceptor(
            signal: Signal, source: str, target: str
        ) -> Signal | None:
            intercepted.append((source, target))
            return signal

        @b.on(TaskSignal)
        async def handle(signal: TaskSignal):
            pass

        mesh = Mesh([a, b])
        mesh.intercept(log_interceptor)
        mesh.connect(a, b)

        async with mesh:
            await a.emit(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert intercepted == [("a", "b")]

    async def test_interceptor_can_block_signals(self):
        """Interceptors returning None block signal routing."""
        a = Agent("a")
        b = Agent("b")
        received: list[Signal] = []

        def block_all(signal: Signal, source: str, target: str) -> Signal | None:
            return None

        @b.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal)

        mesh = Mesh([a, b])
        mesh.intercept(block_all)
        mesh.connect(a, b)

        async with mesh:
            await a.emit(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert received == []

    async def test_async_interceptor(self):
        """Async interceptors work correctly."""
        a = Agent("a")
        b = Agent("b")
        received: list[Signal] = []

        async def async_interceptor(
            signal: Signal, source: str, target: str
        ) -> Signal | None:
            await asyncio.sleep(0.001)
            return signal

        @b.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal)

        mesh = Mesh([a, b])
        mesh.intercept(async_interceptor)
        mesh.connect(a, b)

        async with mesh:
            await a.emit(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert len(received) == 1

    async def test_interceptor_chain(self):
        """Multiple interceptors run in order; any can block."""
        a = Agent("a")
        b = Agent("b")
        order: list[int] = []

        def first(signal: Signal, source: str, target: str) -> Signal | None:
            order.append(1)
            return signal

        def second(signal: Signal, source: str, target: str) -> Signal | None:
            order.append(2)
            return signal

        @b.on(TaskSignal)
        async def handle(signal: TaskSignal):
            pass

        mesh = Mesh([a, b])
        mesh.intercept(first)
        mesh.intercept(second)
        mesh.connect(a, b)

        async with mesh:
            await a.emit(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert order == [1, 2]


# --- Capability Discovery Tests ---


class TestCapabilityDiscovery:
    def test_declare_and_find(self):
        """Can declare capabilities and find agents by capability."""
        analyst = Agent("analyst")
        coder = Agent("coder")
        mesh = Mesh([analyst, coder])

        mesh.declare_capabilities(analyst, "analysis", "summarization")
        mesh.declare_capabilities(coder, "code_generation", "debugging")

        found = mesh.find_capable("analysis")
        assert len(found) == 1
        assert found[0].name == "analyst"

        found = mesh.find_capable("debugging")
        assert len(found) == 1
        assert found[0].name == "coder"

    def test_multiple_agents_same_capability(self):
        """Multiple agents can share the same capability."""
        a = Agent("a")
        b = Agent("b")
        mesh = Mesh([a, b])

        mesh.declare_capabilities(a, "processing")
        mesh.declare_capabilities(b, "processing")

        found = mesh.find_capable("processing")
        assert len(found) == 2

    def test_find_nonexistent_capability(self):
        """Finding a non-existent capability returns empty list."""
        mesh = Mesh([Agent("a")])
        assert mesh.find_capable("nonexistent") == []

    def test_agent_capabilities(self):
        """Can query all capabilities for a specific agent."""
        agent = Agent("multi")
        mesh = Mesh([agent])
        mesh.declare_capabilities(agent, "a", "b", "c")

        caps = mesh.agent_capabilities(agent)
        assert caps == {"a", "b", "c"}

    def test_declare_capabilities_by_name(self):
        """Can declare capabilities using agent name string."""
        agent = Agent("worker")
        mesh = Mesh([agent])
        mesh.declare_capabilities("worker", "task_processing")

        found = mesh.find_capable("task_processing")
        assert len(found) == 1
        assert found[0].name == "worker"


# --- Graceful Shutdown Tests ---


class TestGracefulShutdown:
    async def test_drain_on_stop(self):
        """drain=True waits for pending signals to be processed."""
        agent = Agent("worker")
        processed: list[str] = []

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal):
            await asyncio.sleep(0.01)
            processed.append(signal.task)

        mesh = Mesh([agent])
        await mesh.start()

        for i in range(5):
            await agent.inbox.send(TaskSignal(task=f"task-{i}"))

        await mesh.stop(drain=True)
        assert len(processed) == 5

    async def test_stop_without_drain(self):
        """Default stop doesn't wait for pending signals."""
        agent = Agent("worker")
        mesh = Mesh([agent])
        await mesh.start()
        await mesh.stop()
        assert not agent.running


# --- PriorityChannel.send_wait Tests ---


class TestPriorityChannelSendWait:
    async def test_send_wait_basic(self):
        """send_wait works for basic sending."""
        ch: PriorityChannel[Signal] = PriorityChannel(Signal, buffer_size=10)
        sig = Signal(priority=5)
        await ch.send_wait(sig)
        received = await ch.receive()
        assert received.priority == 5

    async def test_send_wait_backpressure(self):
        """send_wait blocks when channel is full, resumes when space opens."""
        ch: PriorityChannel[Signal] = PriorityChannel(Signal, buffer_size=1)
        await ch.send(Signal(priority=1))

        sent = False

        async def delayed_send():
            nonlocal sent
            await ch.send_wait(Signal(priority=5), timeout=2.0)
            sent = True

        task = asyncio.create_task(delayed_send())
        await asyncio.sleep(0.05)
        assert not sent  # Blocked because channel is full

        await ch.receive()  # Free up space
        await asyncio.sleep(0.05)
        assert sent  # Now it should have sent

        received = await ch.receive()
        assert received.priority == 5
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_send_wait_on_closed_channel(self):
        """send_wait raises ChannelClosed on closed channel."""
        from signal_gating.errors import ChannelClosed

        ch: PriorityChannel[Signal] = PriorityChannel(Signal)
        ch.close()
        with pytest.raises(ChannelClosed):
            await ch.send_wait(Signal())
