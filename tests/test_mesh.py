"""Tests for Mesh agent network topology."""

import asyncio

import pytest

from signal_gating import Agent, Gate, Mesh, MeshError, Signal


class TaskSignal(Signal):
    task: str


async def test_mesh_lifecycle():
    agent = Agent("worker")

    @agent.on(Signal)
    async def handle(s: Signal):
        pass

    mesh = Mesh([agent])
    async with mesh:
        assert agent.running
    assert not agent.running


async def test_mesh_connect():
    producer = Agent("producer")
    consumer = Agent("consumer")
    received = []

    @consumer.on(TaskSignal)
    async def handle(signal: TaskSignal):
        received.append(signal.task)

    mesh = Mesh([producer, consumer])
    mesh.connect(producer, consumer)

    async with mesh:
        await producer.emit(TaskSignal(task="hello"))
        await asyncio.sleep(0.05)

    assert received == ["hello"]


async def test_mesh_gated_connect():
    producer = Agent("producer")
    consumer = Agent("consumer")
    received = []

    @consumer.on(TaskSignal)
    async def handle(signal: TaskSignal):
        received.append(signal.task)

    mesh = Mesh([producer, consumer])
    mesh.connect(producer, consumer, gate=Gate.by_priority(5))

    async with mesh:
        await producer.emit(TaskSignal(task="low", priority=1))
        await producer.emit(TaskSignal(task="high", priority=10))
        await asyncio.sleep(0.05)

    assert received == ["high"]


async def test_mesh_fan_out():
    source = Agent("source")
    a = Agent("a")
    b = Agent("b")
    received_a: list[str] = []
    received_b: list[str] = []

    @a.on(TaskSignal)
    async def handle_a(s: TaskSignal):
        received_a.append(s.task)

    @b.on(TaskSignal)
    async def handle_b(s: TaskSignal):
        received_b.append(s.task)

    mesh = Mesh([source, a, b])
    mesh.broadcast_connect(source, [a, b])

    async with mesh:
        await source.emit(TaskSignal(task="broadcast"))
        await asyncio.sleep(0.05)

    assert received_a == ["broadcast"]
    assert received_b == ["broadcast"]


async def test_mesh_fan_in():
    a = Agent("a")
    b = Agent("b")
    target = Agent("target")
    received: list[str] = []

    @target.on(TaskSignal)
    async def handle(s: TaskSignal):
        received.append(s.source)

    mesh = Mesh([a, b, target])
    mesh.converge_connect([a, b], target)

    async with mesh:
        await a.emit(TaskSignal(task="from_a"))
        await b.emit(TaskSignal(task="from_b"))
        await asyncio.sleep(0.05)

    assert "a" in received
    assert "b" in received


async def test_mesh_inject():
    agent = Agent("worker")
    received = []

    @agent.on(Signal)
    async def handle(s: Signal):
        received.append(s.priority)

    mesh = Mesh([agent])
    async with mesh:
        await mesh.inject(agent, Signal(priority=42))
        await asyncio.sleep(0.05)

    assert received == [42]


async def test_mesh_topology():
    a = Agent("a")
    b = Agent("b")
    mesh = Mesh([a, b])
    mesh.connect(a, b, gate=Gate.passthrough())

    topo = mesh.topology()
    assert len(topo["agents"]) == 2
    assert len(topo["edges"]) == 1
    assert topo["edges"][0]["source"] == "a"
    assert topo["edges"][0]["target"] == "b"


async def test_mesh_duplicate_agent():
    agent = Agent("worker")
    mesh = Mesh([agent])
    with pytest.raises(MeshError):
        mesh.add(agent)


async def test_mesh_unknown_agent():
    mesh = Mesh()
    with pytest.raises(MeshError):
        mesh.get("nonexistent")


async def test_mesh_connect_by_name():
    a = Agent("a")
    b = Agent("b")
    mesh = Mesh([a, b])
    mesh.connect("a", "b")
    assert len(mesh.edges) == 1
