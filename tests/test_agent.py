"""Tests for Agent signal processing."""

import asyncio

from signal_gating import Agent, Gate, Signal


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


async def test_agent_handler():
    agent = Agent("worker")
    received = []

    @agent.on(TaskSignal)
    async def handle(signal: TaskSignal):
        received.append(signal.task)

    await agent.start()
    await agent.inbox.send(TaskSignal(task="build"))
    await asyncio.sleep(0.05)
    await agent.stop()

    assert received == ["build"]


async def test_agent_with_gate():
    agent = Agent("worker", gates=[Gate.by_priority(5)])
    received = []

    @agent.on(TaskSignal)
    async def handle(signal: TaskSignal):
        received.append(signal.task)

    await agent.start()
    await agent.inbox.send(TaskSignal(task="low", priority=1))
    await agent.inbox.send(TaskSignal(task="high", priority=10))
    await asyncio.sleep(0.05)
    await agent.stop()

    assert received == ["high"]


async def test_agent_emit():
    producer = Agent("producer")
    consumer = Agent("consumer")
    received = []

    @consumer.on(TaskSignal)
    async def handle(signal: TaskSignal):
        received.append(signal.task)

    # Wire producer output to consumer input
    producer._add_output(lambda s: consumer.inbox.send(s))

    await producer.start()
    await consumer.start()

    await producer.emit(TaskSignal(task="hello"))
    await asyncio.sleep(0.05)

    await producer.stop()
    await consumer.stop()

    assert received == ["hello"]


async def test_agent_stats():
    agent = Agent("worker", gates=[Gate.by_priority(5)])

    @agent.on(Signal)
    async def handle(signal: Signal):
        pass

    await agent.start()
    await agent.inbox.send(Signal(priority=1))  # rejected
    await agent.inbox.send(Signal(priority=10))  # processed
    await asyncio.sleep(0.05)
    await agent.stop()

    stats = agent.stats
    assert stats["processed"] == 1
    assert stats["rejected"] == 1


async def test_agent_on_any():
    agent = Agent("catch_all")
    received = []

    @agent.on_any
    async def handle(signal: Signal):
        received.append(type(signal).__name__)

    await agent.start()
    await agent.inbox.send(TaskSignal(task="t"))
    await agent.inbox.send(ResultSignal(result="r"))
    await asyncio.sleep(0.05)
    await agent.stop()

    assert "TaskSignal" in received
    assert "ResultSignal" in received


async def test_agent_source_tagging():
    agent = Agent("tagger")
    emitted = []

    agent._add_output(lambda s: emitted.append(s) or asyncio.sleep(0))  # type: ignore

    await agent.emit(TaskSignal(task="tagged"))
    assert emitted[0].source == "tagger"
