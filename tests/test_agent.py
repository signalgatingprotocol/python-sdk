"""Tests for Agent signal processing."""

import asyncio

import pytest

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


# --- New: Middleware ---


async def test_agent_middleware():
    agent = Agent("mw_agent")
    log = []

    async def logging_mw(signal, next_fn):
        log.append(f"before:{signal.priority}")
        result = await next_fn(signal)
        log.append(f"after:{signal.priority}")
        return result

    agent.use(logging_mw)

    @agent.on(Signal)
    async def handle(signal: Signal):
        log.append(f"handler:{signal.priority}")

    await agent.start()
    await agent.inbox.send(Signal(priority=42))
    await asyncio.sleep(0.05)
    await agent.stop()

    assert "before:42" in log
    assert "handler:42" in log
    assert "after:42" in log


async def test_agent_middleware_chain():
    agent = Agent("chain_agent")
    order = []

    async def mw_a(signal, next_fn):
        order.append("a_before")
        result = await next_fn(signal)
        order.append("a_after")
        return result

    async def mw_b(signal, next_fn):
        order.append("b_before")
        result = await next_fn(signal)
        order.append("b_after")
        return result

    agent.use(mw_a)
    agent.use(mw_b)

    @agent.on(Signal)
    async def handle(signal: Signal):
        order.append("handler")

    await agent.start()
    await agent.inbox.send(Signal())
    await asyncio.sleep(0.05)
    await agent.stop()

    # Middleware should wrap in order: a wraps b wraps handler
    assert order == ["a_before", "b_before", "handler", "b_after", "a_after"]


# --- New: Agent State ---


async def test_agent_state():
    agent = Agent("stateful")
    agent.state["counter"] = 0

    @agent.on(Signal)
    async def handle(signal: Signal):
        agent.state["counter"] += 1

    await agent.start()
    await agent.inbox.send(Signal())
    await agent.inbox.send(Signal())
    await agent.inbox.send(Signal())
    await asyncio.sleep(0.05)
    await agent.stop()

    assert agent.state["counter"] == 3


# --- New: Dead Letter Queue ---


async def test_dead_letter_queue_on_gate_rejection():
    agent = Agent("dlq_agent", gates=[Gate.by_priority(100)])

    @agent.on(Signal)
    async def handle(signal: Signal):
        pass

    await agent.start()
    await agent.inbox.send(Signal(priority=1))
    await asyncio.sleep(0.05)
    await agent.stop()

    assert agent.dead_letters.count == 1
    entry = agent.dead_letters.entries[0]
    assert entry["reason"] == "gate_rejected"
    assert entry["agent"] == "dlq_agent"


async def test_dead_letter_queue_on_handler_error():
    agent = Agent("error_agent")

    @agent.on(Signal)
    async def handle(signal: Signal):
        raise ValueError("boom")

    await agent.start()
    await agent.inbox.send(Signal())
    await asyncio.sleep(0.1)
    await agent.stop()

    assert agent.dead_letters.count == 1
    entry = agent.dead_letters.entries[0]
    assert entry["reason"] == "handler_error"
    assert "ValueError: boom" in entry["error"]


# --- New: Stats include errors and dead letters ---


async def test_agent_stats_extended():
    agent = Agent("stats_agent", gates=[Gate.by_priority(5)])

    @agent.on(Signal)
    async def handle(signal: Signal):
        pass

    await agent.start()
    await agent.inbox.send(Signal(priority=1))
    await agent.inbox.send(Signal(priority=10))
    await asyncio.sleep(0.05)
    await agent.stop()

    stats = agent.stats
    assert "errors" in stats
    assert "dead_letters" in stats
    assert stats["dead_letters"] == 1  # one gate rejection
    assert stats["restarts"] == 0


# --- Lifecycle Hooks ---


async def test_agent_on_start_hook():
    agent = Agent("lifecycle")
    events: list[str] = []

    @agent.on_start
    async def setup():
        events.append("started")

    @agent.on(Signal)
    async def handle(s: Signal):
        pass

    await agent.start()
    await asyncio.sleep(0.01)
    await agent.stop()

    assert "started" in events


async def test_agent_on_stop_hook():
    agent = Agent("lifecycle")
    events: list[str] = []

    @agent.on_stop
    async def cleanup():
        events.append("stopped")

    @agent.on(Signal)
    async def handle(s: Signal):
        pass

    await agent.start()
    await asyncio.sleep(0.01)
    await agent.stop()

    assert "stopped" in events


async def test_agent_lifecycle_order():
    agent = Agent("lifecycle")
    events: list[str] = []

    @agent.on_start
    async def setup():
        events.append("start")

    @agent.on_stop
    async def cleanup():
        events.append("stop")

    @agent.on(Signal)
    async def handle(s: Signal):
        events.append("process")

    await agent.start()
    await agent.inbox.send(Signal())
    await asyncio.sleep(0.05)
    await agent.stop()

    assert events == ["start", "process", "stop"]


async def test_agent_sync_lifecycle_hooks():
    agent = Agent("sync_hooks")
    events: list[str] = []

    @agent.on_start
    def setup():
        events.append("sync_start")

    @agent.on_stop
    def cleanup():
        events.append("sync_stop")

    await agent.start()
    await asyncio.sleep(0.01)
    await agent.stop()

    assert events == ["sync_start", "sync_stop"]


# --- Request/Response ---


async def test_agent_request_response():
    from signal_gating import Mesh

    requester = Agent("requester")
    responder = Agent("responder")

    @responder.on(TaskSignal)
    async def handle(signal: TaskSignal):
        await responder.reply(signal, ResultSignal(result=f"done:{signal.task}"))

    mesh = Mesh([requester, responder])
    mesh.connect(requester, responder)
    mesh.connect(responder, requester)  # Return path

    async with mesh:
        response = await requester.request(TaskSignal(task="analyze"), timeout=2.0)
        assert isinstance(response, ResultSignal)
        assert response.result == "done:analyze"


async def test_agent_request_timeout():
    agent = Agent("lonely")
    # No connections, so request will timeout
    await agent.start()
    with pytest.raises(asyncio.TimeoutError):
        await agent.request(Signal(), timeout=0.05)
    await agent.stop()


async def test_agent_reply_without_correlation():
    agent = Agent("replier")
    emitted: list[Signal] = []
    agent._add_output(lambda s: emitted.append(s) or asyncio.sleep(0))  # type: ignore

    # Reply to a signal without correlation_id just emits normally
    original = Signal()
    await agent.reply(original, ResultSignal(result="ok"))
    assert len(emitted) == 1


# --- Priority Inbox ---


async def test_agent_priority_inbox():
    agent = Agent("priority_worker", priority_inbox=True)
    received_order: list[int] = []

    @agent.on(Signal)
    async def handle(signal: Signal):
        received_order.append(signal.priority)

    # Queue up signals before starting so they're all available
    await agent.inbox.send(Signal(priority=1))
    await agent.inbox.send(Signal(priority=10))
    await agent.inbox.send(Signal(priority=5))

    await agent.start()
    await asyncio.sleep(0.05)
    await agent.stop()

    assert received_order == [10, 5, 1]
