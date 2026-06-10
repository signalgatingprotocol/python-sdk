"""Regression tests for correctness refinements.

Each test pins a fixed defect:
- Channel.receive() losing a dequeued signal when cancelled mid-wait
- Mesh.remove() leaving the agent subscribed to topics
- Mesh.publish() aborting the broadcast on a stopped subscriber
- Mesh.stop(drain=True) ignoring handlers that are mid-flight
- Agent.start() spawning duplicate loops under concurrent calls
- Agent.once() retrying on later signals after the handler raised
- Gate.circuit_breaker() serializing all traffic through its state lock
- LLMAgent hanging forever on an unresponsive model
"""

import asyncio

import pytest

from signal_gating import Agent, Channel, Gate, Mesh, Signal
from signal_gating.errors import AgentError
from signal_gating.llm import LLMAgent


class Ping(Signal):
    n: int = 0


class Pong(Signal):
    n: int = 0


async def test_channel_receive_cancellation_does_not_lose_signal():
    ch = Channel(Signal)
    receiver = asyncio.create_task(ch.receive())
    await asyncio.sleep(0)  # receiver enters the slow path

    sig = Signal()
    await ch.send(sig)
    await asyncio.sleep(0)  # internal getter dequeues the item
    receiver.cancel()

    try:
        result = await receiver
    except asyncio.CancelledError:
        result = None

    if result is not None:
        # receive() won the race against cancellation; nothing was lost.
        assert result.id == sig.id
    else:
        # Cancelled after dequeue: the signal must be back in the channel.
        assert ch.pending == 1
        assert (await ch.receive()).id == sig.id


async def test_mesh_remove_cleans_topic_subscriptions():
    keeper, goner = Agent("keeper"), Agent("goner")
    mesh = Mesh([keeper, goner])
    mesh.create_topic("events")
    mesh.subscribe(keeper, "events")
    mesh.subscribe(goner, "events")

    seen: list[int] = []

    @keeper.on(Ping)
    async def handle(sig: Ping) -> None:
        seen.append(sig.n)

    async with mesh:
        await mesh.remove(goner)
        assert mesh.list_topics() == {"events": ["keeper"]}
        count = await mesh.publish("events", Ping(n=1))
        assert count == 1
        await asyncio.sleep(0.05)

    assert seen == [1]


async def test_publish_skips_stopped_subscriber():
    healthy, stopped = Agent("healthy"), Agent("stopped")
    mesh = Mesh([healthy, stopped])
    mesh.create_topic("events")
    mesh.subscribe(healthy, "events")
    mesh.subscribe(stopped, "events")

    seen: list[int] = []

    @healthy.on(Ping)
    async def handle(sig: Ping) -> None:
        seen.append(sig.n)

    async with mesh:
        await stopped.stop()  # inbox closes; agent stays subscribed
        count = await mesh.publish("events", Ping(n=7))
        assert count == 1  # only the reachable subscriber
        await asyncio.sleep(0.05)

    assert seen == [7]


async def test_drain_waits_for_in_flight_handler_emission():
    a, b = Agent("a"), Agent("b")
    seen: list[int] = []

    @a.on(Ping)
    async def forward(sig: Ping) -> None:
        await asyncio.sleep(0.05)  # mid-flight when stop(drain=True) begins
        await a.emit(Pong(n=sig.n))

    @b.on(Pong)
    async def collect(sig: Pong) -> None:
        seen.append(sig.n)

    mesh = Mesh([a, b])
    mesh.connect(a, b)
    await mesh.start()
    await a.inbox.send(Ping(n=3))
    await asyncio.sleep(0.01)  # a pops the signal; its inbox is now empty
    await mesh.stop(drain=True)

    assert seen == [3]


async def test_concurrent_start_spawns_single_loop():
    agent = Agent("solo")

    @agent.on_start
    async def slow_setup() -> None:
        await asyncio.sleep(0.02)  # yields between the check and create_task

    await asyncio.gather(agent.start(), agent.start())
    loops = [t for t in asyncio.all_tasks() if t.get_name() == "agent:solo"]
    assert len(loops) == 1
    await agent.stop()


async def test_once_handler_never_retried_after_exception():
    agent = Agent("flaky")
    calls: list[int] = []

    @agent.once(Ping)
    async def explode(sig: Ping) -> None:
        calls.append(sig.n)
        raise RuntimeError("boom")

    await agent.start()
    await agent.inbox.send(Ping(n=1))
    await agent.inbox.send(Ping(n=2))
    await asyncio.sleep(0.05)
    await agent.stop()

    assert calls == [1]  # at most once, even though the first call raised


async def test_circuit_breaker_does_not_serialize_closed_state():
    entered = 0
    release = asyncio.Event()

    async def slow(sig: Signal) -> Signal:
        nonlocal entered
        entered += 1
        await release.wait()
        return sig

    breaker = Gate.circuit_breaker(Gate(slow))
    t1 = asyncio.create_task(breaker.process(Signal()))
    t2 = asyncio.create_task(breaker.process(Signal()))
    await asyncio.sleep(0.01)
    assert entered == 2  # both in flight; the state lock must not serialize
    release.set()
    r1, r2 = await asyncio.gather(t1, t2)
    assert r1 is not None and r2 is not None


async def test_circuit_breaker_half_open_admits_single_probe():
    entered = 0
    release = asyncio.Event()
    mode = {"reject": True}

    async def inner(sig: Signal) -> Signal | None:
        nonlocal entered
        entered += 1
        if mode["reject"]:
            return None
        await release.wait()
        return sig

    breaker = Gate.circuit_breaker(
        Gate(inner), failure_threshold=1, recovery_timeout=0.05
    )
    assert await breaker.process(Signal()) is None  # trips the breaker
    assert entered == 1
    await asyncio.sleep(0.06)  # recovery window elapses -> half-open

    mode["reject"] = False
    probe = asyncio.create_task(breaker.process(Signal()))
    await asyncio.sleep(0.01)
    # While the probe is in flight, other signals are rejected without
    # reaching the inner gate.
    assert await breaker.process(Signal()) is None
    assert entered == 2
    release.set()
    assert await probe is not None
    # Probe success closed the circuit again.
    assert await breaker.process(Signal()) is not None


class _HangingCompletions:
    async def create(self, *, model, messages, **kw):
        await asyncio.sleep(3600)


class _HangingChat:
    completions = _HangingCompletions()


class _HangingClient:
    chat = _HangingChat()


async def test_llm_agent_timeout_raises_instead_of_hanging():
    class Topic(Signal):
        text: str = ""

    agent = LLMAgent(
        "a", client=_HangingClient(), model="m", on=Topic, emit=Topic, timeout=0.05
    )
    with pytest.raises(AgentError, match="timed out"):
        await agent._dispatch(Topic(text="q"))


def test_llm_agent_rejects_non_positive_timeout():
    class Topic(Signal):
        text: str = ""

    with pytest.raises(ValueError, match="timeout"):
        LLMAgent("a", client=_HangingClient(), model="m", on=Topic, emit=Topic, timeout=0)
