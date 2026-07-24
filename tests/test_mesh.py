"""Tests for Mesh agent network topology."""

import asyncio

import pytest

from signal_gating import Agent, AgentPool, Gate, Mesh, MeshError, Signal, Tracer


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


async def test_mesh_context_exit_drains_multi_hop_work():
    source = Agent("source")
    middle = Agent("middle")
    sink = Agent("sink")
    received: list[str] = []

    @middle.on(TaskSignal)
    async def forward(signal: TaskSignal):
        # Keep the handler active long enough for a concurrent shutdown to
        # close downstream inboxes before this emission.
        await asyncio.sleep(0.01)
        await middle.emit(TaskSignal(task=f"{signal.task}:forwarded"))

    @sink.on(TaskSignal)
    async def collect(signal: TaskSignal):
        received.append(signal.task)

    mesh = Mesh([source, middle, sink])
    mesh.connect(source, middle)
    mesh.connect(middle, sink)

    async with mesh:
        await source.emit(TaskSignal(task="hello"))

    assert received == ["hello:forwarded"]


async def test_wait_idle_timeout_identifies_active_agents():
    worker = Agent("blocked-worker")
    entered = asyncio.Event()
    release = asyncio.Event()

    @worker.on(TaskSignal)
    async def block(_signal: TaskSignal):
        entered.set()
        await release.wait()

    mesh = Mesh([worker])
    await mesh.start()
    try:
        await worker.inbox.send(TaskSignal(task="blocked"))
        await entered.wait()

        with pytest.raises(
            asyncio.TimeoutError,
            match=r"active agents: blocked-worker\(pending=0, busy=True\)",
        ):
            await mesh.wait_idle(timeout=0.01)
    finally:
        release.set()
        await mesh.stop(drain=True)


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


# --- New: Integrated Tracing ---


async def test_mesh_auto_traces_signal_flow():
    producer = Agent("producer")
    consumer = Agent("consumer")
    received = []

    @consumer.on(TaskSignal)
    async def handle(signal: TaskSignal):
        received.append(signal.task)

    mesh = Mesh([producer, consumer])
    mesh.connect(producer, consumer)

    async with mesh:
        await producer.emit(TaskSignal(task="traced"))
        await asyncio.sleep(0.05)

    # Mesh should have auto-traced the signal flow
    assert mesh.tracer.span_count > 0
    summary = mesh.tracer.summary()
    assert summary["total_spans"] > 0
    assert "routed" in summary["actions"]


async def test_mesh_traces_edge_rejection():
    producer = Agent("producer")
    consumer = Agent("consumer")

    @consumer.on(Signal)
    async def handle(s: Signal):
        pass

    mesh = Mesh([producer, consumer])
    mesh.connect(producer, consumer, gate=Gate.by_priority(100))

    async with mesh:
        await producer.emit(Signal(priority=1))
        await asyncio.sleep(0.05)

    actions = mesh.tracer.summary().get("actions", {})
    assert "edge_rejected" in actions


async def test_mesh_custom_tracer():
    tracer = Tracer(max_spans=50)
    a = Agent("a")
    b = Agent("b")

    @b.on(Signal)
    async def handle(s: Signal):
        pass

    mesh = Mesh([a, b], tracer=tracer)
    mesh.connect(a, b)

    async with mesh:
        await a.emit(Signal())
        await asyncio.sleep(0.05)

    # The custom tracer should have been used
    assert tracer.span_count > 0
    assert mesh.tracer is tracer


# --- Content-based Routing ---


async def test_mesh_content_routing():
    router = Agent("router")
    fast = Agent("fast")
    slow = Agent("slow")
    fast_received: list[str] = []
    slow_received: list[str] = []

    @fast.on(TaskSignal)
    async def handle_fast(s: TaskSignal):
        fast_received.append(s.task)

    @slow.on(TaskSignal)
    async def handle_slow(s: TaskSignal):
        slow_received.append(s.task)

    mesh = Mesh([router, fast, slow])
    mesh.route(router, [
        (lambda s: s.priority >= 5, fast),
        (lambda s: s.priority < 5, slow),
    ])

    async with mesh:
        await router.emit(TaskSignal(task="urgent", priority=8))
        await router.emit(TaskSignal(task="lazy", priority=2))
        await asyncio.sleep(0.05)

    assert fast_received == ["urgent"]
    assert slow_received == ["lazy"]


async def test_mesh_content_routing_default():
    router = Agent("router")
    special = Agent("special")
    fallback = Agent("fallback")
    special_received: list[int] = []
    fallback_received: list[int] = []

    @special.on(Signal)
    async def handle_special(s: Signal):
        special_received.append(s.priority)

    @fallback.on(Signal)
    async def handle_fallback(s: Signal):
        fallback_received.append(s.priority)

    mesh = Mesh([router, special, fallback])
    mesh.route(
        router,
        [(lambda s: s.priority >= 10, special)],
        default=fallback,
    )

    async with mesh:
        await router.emit(Signal(priority=10))
        await router.emit(Signal(priority=3))
        await asyncio.sleep(0.05)

    assert special_received == [10]
    assert fallback_received == [3]


async def test_mesh_content_routing_no_match_no_default():
    router = Agent("router")
    target = Agent("target")
    received: list[int] = []

    @target.on(Signal)
    async def handle(s: Signal):
        received.append(s.priority)

    mesh = Mesh([router, target])
    mesh.route(router, [
        (lambda s: s.priority >= 100, target),
    ])

    async with mesh:
        await router.emit(Signal(priority=1))
        await asyncio.sleep(0.05)

    assert received == []  # No match, no default, signal dropped


async def test_mesh_content_routing_by_name():
    router = Agent("router")
    target = Agent("target")
    received: list[int] = []

    @target.on(Signal)
    async def handle(s: Signal):
        received.append(s.priority)

    mesh = Mesh([router, target])
    mesh.route("router", [
        (lambda s: True, "target"),
    ])

    async with mesh:
        await router.emit(Signal(priority=42))
        await asyncio.sleep(0.05)

    assert received == [42]


async def test_mesh_content_routing_traces():
    router = Agent("router")
    target = Agent("target")

    @target.on(Signal)
    async def handle(s: Signal):
        pass

    mesh = Mesh([router, target])
    mesh.route(router, [(lambda s: True, target)])

    async with mesh:
        await router.emit(Signal())
        await asyncio.sleep(0.05)

    actions = mesh.tracer.summary().get("actions", {})
    assert "routed" in actions


# --- Load Balancing ---


async def test_mesh_load_balance():
    dispatcher = Agent("dispatcher")
    w1 = Agent("w1")
    w2 = Agent("w2")
    w3 = Agent("w3")
    w1_received: list[str] = []
    w2_received: list[str] = []
    w3_received: list[str] = []

    @w1.on(TaskSignal)
    async def h1(s: TaskSignal):
        w1_received.append(s.task)

    @w2.on(TaskSignal)
    async def h2(s: TaskSignal):
        w2_received.append(s.task)

    @w3.on(TaskSignal)
    async def h3(s: TaskSignal):
        w3_received.append(s.task)

    mesh = Mesh([dispatcher, w1, w2, w3])
    mesh.load_balance(dispatcher, [w1, w2, w3])

    async with mesh:
        for i in range(6):
            await dispatcher.emit(TaskSignal(task=f"job-{i}"))
        await asyncio.sleep(0.1)

    # Round-robin: each worker gets exactly 2 jobs
    assert len(w1_received) == 2
    assert len(w2_received) == 2
    assert len(w3_received) == 2
    assert w1_received == ["job-0", "job-3"]
    assert w2_received == ["job-1", "job-4"]
    assert w3_received == ["job-2", "job-5"]


async def test_mesh_load_balance_with_gate():
    dispatcher = Agent("dispatcher")
    w1 = Agent("w1")
    w2 = Agent("w2")
    received: list[str] = []

    @w1.on(TaskSignal)
    async def h1(s: TaskSignal):
        received.append(f"w1:{s.task}")

    @w2.on(TaskSignal)
    async def h2(s: TaskSignal):
        received.append(f"w2:{s.task}")

    mesh = Mesh([dispatcher, w1, w2])
    mesh.load_balance(dispatcher, [w1, w2], gate=Gate.by_priority(5))

    async with mesh:
        await dispatcher.emit(TaskSignal(task="low", priority=1))
        await dispatcher.emit(TaskSignal(task="high", priority=10))
        await asyncio.sleep(0.05)

    # Low priority is filtered out, only high gets through
    assert len(received) == 1
    assert "high" in received[0]


async def test_mesh_load_balance_by_name():
    dispatcher = Agent("dispatcher")
    w1 = Agent("w1")
    w2 = Agent("w2")

    @w1.on(Signal)
    async def h1(s: Signal):
        pass

    @w2.on(Signal)
    async def h2(s: Signal):
        pass

    mesh = Mesh([dispatcher, w1, w2])
    mesh.load_balance("dispatcher", ["w1", "w2"])
    # Should not raise (just verifying the API accepts strings)


# --- Mesh Health ---


async def test_mesh_health():
    a = Agent("a")
    b = Agent("b")

    @a.on(Signal)
    async def ha(s: Signal):
        pass

    @b.on(Signal)
    async def hb(s: Signal):
        pass

    mesh = Mesh([a, b])
    mesh.connect(a, b)

    async with mesh:
        health = mesh.health()
        assert health["healthy"] is True
        assert health["running"] is True
        assert health["total_agents"] == 2
        assert health["total_edges"] == 1
        assert "a" in health["agents"]
        assert "b" in health["agents"]
        assert health["agents"]["a"]["healthy"] is True


class TestDrainOnStop:
    """Mesh.stop(drain=True) waits for pending signals to complete."""

    async def test_drain_stop_works(self):
        agent = Agent("worker")
        processed: list[str] = []

        class DrainTask(Signal):
            task: str

        @agent.on(DrainTask)
        async def handle(signal: DrainTask):
            await asyncio.sleep(0.01)
            processed.append(signal.task)

        mesh = Mesh([agent])
        await mesh.start()

        for i in range(3):
            await agent.inbox.send(DrainTask(task=f"task-{i}"))

        await mesh.stop(drain=True)
        assert len(processed) == 3


class TestRemoveHardening:
    async def test_scale_pool_purges_pool_membership(self):
        pool = AgentPool("workers", size=3)
        mesh = Mesh()
        mesh.add_pool(pool)
        victim = pool.workers[-1]
        removed = await mesh.scale_pool(pool, 2)
        assert removed == [victim]
        assert victim.name not in pool.worker_names
        assert pool.size == 2

    async def test_remove_prunes_load_balance_target(self):
        src, a, b = Agent("src"), Agent("a"), Agent("b")
        got: list[str] = []
        for agent in (a, b):
            @agent.on(Signal)
            async def handle(signal: Signal, _name=agent.name):
                got.append(_name)
        mesh = Mesh([src, a, b])
        mesh.load_balance(src, [a, b])
        async with mesh:
            await mesh.remove(b)
            for _ in range(4):
                await src.emit(Signal())
            await asyncio.sleep(0.05)
        assert got == ["a", "a", "a", "a"]

    async def test_remove_prunes_route_branch_falls_to_default(self):
        src, hot, cold = Agent("src"), Agent("hot"), Agent("cold")
        got: list[str] = []
        for agent in (hot, cold):
            @agent.on(Signal)
            async def handle(signal: Signal, _name=agent.name):
                got.append(_name)
        mesh = Mesh([src, hot, cold])
        mesh.route(src, [(lambda s: s.priority >= 5, hot)], default=cold)
        async with mesh:
            await mesh.remove(hot)
            await src.emit(Signal(priority=9))
            await asyncio.sleep(0.05)
        assert got == ["cold"]
