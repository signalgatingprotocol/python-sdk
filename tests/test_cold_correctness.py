"""Cold correctness: adversarial tests for silent foundational bugs.

Each test here targets a real, subtle bug that the broader suite missed:
they exercise lifecycle edge cases, concurrent topology mutation, and
close semantics under pressure. If any of these regresses, signals are
silently lost, agents brick, or stop() leaks tasks.
"""

from __future__ import annotations

import asyncio

import pytest

from signal_gating import Agent, AgentContext, Channel, Gate, Mesh, Signal
from signal_gating.errors import ChannelClosed


class Ping(Signal):
    n: int = 0


# --- Channel: close must wake up blocked receivers, even when the buffer is full.


async def test_channel_close_wakes_blocked_receiver():
    ch: Channel[Ping] = Channel(Ping, buffer_size=1)

    async def consumer() -> str:
        try:
            await ch.receive()
            return "got_signal"
        except ChannelClosed:
            return "closed"

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let consumer block on receive
    ch.close()
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result == "closed"


async def test_channel_close_with_full_buffer_does_not_strand_receiver():
    """Pre-fill to capacity so the old sentinel approach would have failed."""
    ch: Channel[Ping] = Channel(Ping, buffer_size=2)
    await ch.send(Ping(n=1))
    await ch.send(Ping(n=2))

    received: list[int] = []

    async def consumer() -> None:
        try:
            while True:
                sig = await ch.receive()
                received.append(sig.n)
        except ChannelClosed:
            return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    ch.close()
    await asyncio.wait_for(task, timeout=1.0)
    assert received == [1, 2]


async def test_channel_drain_after_close_returns_remaining():
    ch: Channel[Ping] = Channel(Ping, buffer_size=4)
    for i in range(3):
        await ch.send(Ping(n=i))
    ch.close()
    drained = await ch.drain()
    assert [s.n for s in drained] == [0, 1, 2]
    with pytest.raises(ChannelClosed):
        await ch.receive()


async def test_channel_close_is_idempotent():
    ch: Channel[Ping] = Channel(Ping)
    ch.close()
    ch.close()  # must not raise
    assert ch.closed


# --- Agent: restart contract.


async def test_agent_restart_after_clean_stop():
    agent = Agent("worker")
    received: list[int] = []

    @agent.on(Ping)
    async def handle(sig: Ping) -> None:
        received.append(sig.n)

    await agent.start()
    await agent.inbox.send(Ping(n=1))
    await asyncio.sleep(0.05)
    await agent.stop()

    # Restart should work: inbox is recreated, handlers preserved.
    await agent.start()
    await asyncio.sleep(0)  # let the supervised task start
    assert agent.running
    await agent.inbox.send(Ping(n=2))
    await asyncio.sleep(0.05)
    await agent.stop()

    assert received == [1, 2]


async def test_agent_restart_after_max_restarts_exceeded():
    """The README promises restartable agents. Even after the supervisor
    gives up, the operator must be able to call start() again and recover."""

    def explode(_sig: Signal) -> Signal | None:
        raise RuntimeError("gate boom")

    agent = Agent(
        "flaky",
        max_restarts=1,
        restart_delay=0.0,
        gates=[Gate(explode, name="explode")],
    )

    await agent.start()
    # Drive crashes until the supervisor gives up.
    for _ in range(5):
        await agent.inbox.send(Ping())
    for _ in range(100):
        if agent._task is not None and agent._task.done():
            break
        await asyncio.sleep(0.01)
    assert agent._task is not None and agent._task.done()
    assert agent._restart_count > agent._max_restarts

    # Operator fixes the bug (swaps to a passing gate) and restarts.
    agent.gates = []
    successes: list[int] = []

    @agent.on(Ping)
    async def fixed(sig: Ping) -> None:
        successes.append(sig.n)

    await agent.start()
    await asyncio.sleep(0)
    assert agent.running
    assert agent._restart_count == 0  # counter was reset
    # Drain anything left in the inbox from before the restart, then send fresh.
    await asyncio.sleep(0.02)
    successes.clear()
    await agent.inbox.send(Ping(n=99))
    await asyncio.sleep(0.05)
    await agent.stop()
    assert successes == [99]


async def test_agent_stop_awaits_cancellation():
    """stop() must not return while the run task is still executing."""
    agent = Agent("slow")
    inside = asyncio.Event()
    release = asyncio.Event()

    @agent.on(Ping)
    async def handle(sig: Ping) -> None:
        inside.set()
        await release.wait()

    await agent.start()
    await agent.inbox.send(Ping())
    await inside.wait()
    # Handler is stuck. stop() with a tight timeout must cancel and *wait*
    # for the cancellation to fully propagate before returning.
    stop_task = asyncio.create_task(agent.stop(timeout=0.05))
    release.set()  # let handler resume so cancel can land
    await asyncio.wait_for(stop_task, timeout=2.0)
    assert agent._task is None


# --- Agent: outbox snapshot under concurrent mutation.


async def test_emit_does_not_corrupt_under_concurrent_disconnect():
    """If we disconnect while emit() is iterating, snapshotting must save us."""
    a = Agent("source")
    b = Agent("b")
    c = Agent("c")
    received_b: list[int] = []
    received_c: list[int] = []

    @b.on(Ping)
    async def hb(sig: Ping) -> None:
        received_b.append(sig.n)

    @c.on(Ping)
    async def hc(sig: Ping) -> None:
        received_c.append(sig.n)

    mesh = Mesh([a, b, c])
    mesh.connect(a, b)
    mesh.connect(a, c)

    async with mesh:
        # Emit a burst while concurrently mutating topology.
        async def emit_burst() -> None:
            for i in range(20):
                await a.emit(Ping(n=i))
                await asyncio.sleep(0)

        async def churn() -> None:
            for _ in range(10):
                await asyncio.sleep(0.001)
                mesh.disconnect(a, c)
                mesh.connect(a, c)

        await asyncio.gather(emit_burst(), churn())
        await asyncio.sleep(0.1)

    # No crash, no IndexError. b receives every signal because it was never
    # disconnected; c receives a subset (depending on churn timing). The
    # invariant under test is "no exception, no skipped iteration".
    assert sorted(received_b) == list(range(20))
    assert all(0 <= n < 20 for n in received_c)


# --- Handler context detection: cached, robust to once() wrappers.


async def test_once_handler_with_context_still_receives_context():
    agent = Agent("ctx")
    captured: list[str] = []

    @agent.once(Ping)
    async def handle(sig: Ping, ctx: AgentContext) -> None:
        captured.append(ctx.agent_name)

    await agent.start()
    await agent.inbox.send(Ping())
    await asyncio.sleep(0.05)
    await agent.stop()
    assert captured == ["ctx"]


async def test_handler_without_context_unaffected():
    agent = Agent("noctx")
    seen: list[int] = []

    @agent.on(Ping)
    async def handle(sig: Ping) -> None:
        seen.append(sig.n)

    await agent.start()
    await agent.inbox.send(Ping(n=7))
    await asyncio.sleep(0.05)
    await agent.stop()
    assert seen == [7]


async def test_double_start_is_idempotent():
    agent = Agent("idem")
    await agent.start()
    first_task = agent._task
    await agent.start()  # must be a no-op
    assert agent._task is first_task
    await agent.stop()
