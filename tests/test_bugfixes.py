"""Tests for bug fixes in this release."""

import asyncio

from signal_gating import Agent, AgentContext, Mesh, Signal


class TaskSignal(Signal):
    task: str


class TestOnceWithAgentContext:
    """Fix: once() handlers now correctly detect AgentContext parameters.

    Previously, the once() wrapper function used (*args, **kwargs) which
    broke the handler introspection that detects AgentContext. Now the
    wrapper sets __wrapped__ so introspection finds the real function.
    """

    async def test_once_handler_with_context(self):
        agent = Agent("worker")
        received_ctx: list[str] = []

        @agent.once(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received_ctx.append(ctx.agent_name)

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert received_ctx == ["worker"]

    async def test_once_handler_with_context_fires_once(self):
        agent = Agent("worker")
        calls: list[str] = []

        @agent.once(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            calls.append(ctx.agent_name)

        mesh = Mesh([agent])
        async with mesh:
            await agent.inbox.send(TaskSignal(task="first"))
            await asyncio.sleep(0.05)
            await agent.inbox.send(TaskSignal(task="second"))
            await asyncio.sleep(0.05)

        assert calls == ["worker"]  # Only fires once


class TestMeshGetRunningLoop:
    """Fix: Mesh.stop(drain=True) now uses get_running_loop() instead of
    the deprecated get_event_loop()."""

    async def test_drain_stop_works(self):
        agent = Agent("worker")
        processed: list[str] = []

        @agent.on(TaskSignal)
        async def handle(signal: TaskSignal):
            await asyncio.sleep(0.01)
            processed.append(signal.task)

        mesh = Mesh([agent])
        await mesh.start()

        for i in range(3):
            await agent.inbox.send(TaskSignal(task=f"task-{i}"))

        await mesh.stop(drain=True)
        assert len(processed) == 3
