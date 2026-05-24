"""Tests for AgentPool: horizontal scaling primitive."""

import asyncio

import pytest

from signal_gating import Agent, AgentContext, AgentPool, Gate, Mesh, Signal


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


# === Pool Creation ===


class TestPoolCreation:
    def test_pool_creates_workers(self):
        pool = AgentPool("workers", size=3)
        assert pool.size == 3
        assert len(pool.workers) == 3

    def test_pool_worker_naming(self):
        pool = AgentPool("workers", size=3)
        names = pool.worker_names
        assert names == ["workers[0]", "workers[1]", "workers[2]"]

    def test_pool_default_size(self):
        pool = AgentPool("workers")
        assert pool.size == 3

    def test_pool_invalid_size(self):
        with pytest.raises(ValueError, match="at least 1"):
            AgentPool("workers", size=0)

    def test_pool_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            AgentPool("workers", strategy="random")

    def test_pool_repr(self):
        pool = AgentPool("workers", size=2)
        r = repr(pool)
        assert "workers" in r
        assert "size=2" in r
        assert "round_robin" in r

    def test_pool_with_gates(self):
        pool = AgentPool("workers", size=2, gates=[Gate.by_priority(5)])
        for worker in pool.workers:
            assert len(worker.gates) == 1

    def test_pool_with_priority_inbox(self):
        pool = AgentPool("workers", size=2, priority_inbox=True)
        for worker in pool.workers:
            assert worker._priority_inbox is True


# === Handler Registration ===


class TestPoolHandlers:
    def test_on_handler_registered_on_all_workers(self):
        pool = AgentPool("workers", size=3)

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal):
            pass

        for worker in pool.workers:
            assert TaskSignal in worker._handlers
            assert len(worker._handlers[TaskSignal]) == 1

    def test_on_any_registered_on_all_workers(self):
        pool = AgentPool("workers", size=2)

        @pool.on_any
        async def handle(signal: Signal):
            pass

        for worker in pool.workers:
            assert Signal in worker._handlers

    async def test_handlers_process_signals(self):
        pool = AgentPool("workers", size=2)
        received: list[str] = []

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal.task)

        mesh = Mesh()
        mesh.add_pool(pool)

        async with mesh:
            # Send directly to first worker
            await pool.workers[0].inbox.send(TaskSignal(task="hello"))
            await asyncio.sleep(0.05)

        assert received == ["hello"]

    async def test_handler_with_context(self):
        pool = AgentPool("workers", size=2)
        received_names: list[str] = []

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received_names.append(ctx.agent_name)

        mesh = Mesh()
        mesh.add_pool(pool)

        async with mesh:
            await pool.workers[0].inbox.send(TaskSignal(task="a"))
            await pool.workers[1].inbox.send(TaskSignal(task="b"))
            await asyncio.sleep(0.05)

        assert "workers[0]" in received_names
        assert "workers[1]" in received_names

    def test_middleware_applied_to_all_workers(self):
        pool = AgentPool("workers", size=2)

        async def mw(signal: Signal, next_fn):  # type: ignore
            return await next_fn(signal)

        pool.use(mw)
        for worker in pool.workers:
            assert len(worker._middleware) == 1


# === Mesh Integration ===


class TestPoolMeshIntegration:
    async def test_connect_to_pool_load_balances(self):
        coordinator = Agent("coordinator")
        pool = AgentPool("workers", size=3)
        received: dict[str, list[str]] = {w.name: [] for w in pool.workers}

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received[ctx.agent_name].append(signal.task)

        mesh = Mesh([coordinator])
        mesh.add_pool(pool)
        mesh.connect(coordinator, pool)

        async with mesh:
            for i in range(6):
                await coordinator.emit(TaskSignal(task=f"job-{i}"))
            await asyncio.sleep(0.1)

        # Round-robin: each worker should get 2 jobs
        for worker_name, tasks in received.items():
            assert len(tasks) == 2, f"{worker_name} got {len(tasks)} tasks"

    async def test_connect_from_pool(self):
        pool = AgentPool("workers", size=2)
        collector = Agent("collector")
        collected: list[str] = []

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            await ctx.emit(ResultSignal(result=f"done:{signal.task}"))

        @collector.on(ResultSignal)
        async def collect(signal: ResultSignal):
            collected.append(signal.result)

        mesh = Mesh([collector])
        mesh.add_pool(pool)
        mesh.connect(pool, collector)

        async with mesh:
            await pool.workers[0].inbox.send(TaskSignal(task="a"))
            await pool.workers[1].inbox.send(TaskSignal(task="b"))
            await asyncio.sleep(0.1)

        assert len(collected) == 2
        assert "done:a" in collected
        assert "done:b" in collected

    async def test_add_pool_duplicate_raises(self):
        pool = AgentPool("workers", size=1)
        mesh = Mesh()
        mesh.add_pool(pool)
        with pytest.raises(Exception):
            mesh.add_pool(pool)

    async def test_get_pool(self):
        pool = AgentPool("workers", size=1)
        mesh = Mesh()
        mesh.add_pool(pool)
        assert mesh.get_pool("workers") is pool

    async def test_get_pool_not_found(self):
        mesh = Mesh()
        with pytest.raises(Exception):
            mesh.get_pool("nonexistent")

    async def test_connect_pool_by_name(self):
        coordinator = Agent("coordinator")
        pool = AgentPool("workers", size=2)
        received: list[str] = []

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal.task)

        mesh = Mesh([coordinator])
        mesh.add_pool(pool)
        mesh.connect("coordinator", "workers")  # pool name resolves to pool

        async with mesh:
            await coordinator.emit(TaskSignal(task="test"))
            await asyncio.sleep(0.05)

        assert len(received) == 1


# === Scaling ===


class TestPoolScaling:
    def test_scale_up(self):
        pool = AgentPool("workers", size=2)

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal):
            pass

        new_workers = pool.scale_to(5)
        assert pool.size == 5
        assert len(new_workers) == 3
        # New workers should have handlers
        for w in new_workers:
            assert TaskSignal in w._handlers

    def test_scale_down(self):
        pool = AgentPool("workers", size=5)
        removed = pool.scale_to(2)
        assert pool.size == 2
        assert len(removed) == 3

    def test_scale_to_same_size(self):
        pool = AgentPool("workers", size=3)
        result = pool.scale_to(3)
        assert result == []
        assert pool.size == 3

    def test_scale_to_zero_raises(self):
        pool = AgentPool("workers", size=3)
        with pytest.raises(ValueError, match="at least 1"):
            pool.scale_to(0)

    async def test_scale_up_async(self):
        pool = AgentPool("workers", size=1)

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal):
            pass

        new = await pool.scale_up(2)
        assert pool.size == 3
        assert len(new) == 2

    async def test_scale_down_async_stops_workers(self):
        pool = AgentPool("workers", size=3)

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal):
            pass

        await pool.start()
        removed = await pool.scale_down(2)
        assert pool.size == 1
        assert len(removed) == 2
        for w in removed:
            assert not w.running
        await pool.stop()


# === Distribution Strategies ===


class TestPoolStrategies:
    async def test_least_loaded_strategy(self):
        pool = AgentPool("workers", size=3, strategy="least_loaded")
        received: dict[str, int] = {}

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received[ctx.agent_name] = received.get(ctx.agent_name, 0) + 1
            await asyncio.sleep(0.05)  # Simulate work

        mesh = Mesh()
        mesh.add_pool(pool)

        async with mesh:
            # Send directly using select_worker, which picks least loaded
            for i in range(6):
                worker = pool.select_worker()
                await worker.inbox.send(TaskSignal(task=f"job-{i}"))
            await asyncio.sleep(0.5)

        # All 6 should be processed
        total = sum(received.values())
        assert total == 6


# === Observability ===


class TestPoolObservability:
    async def test_pool_health(self):
        pool = AgentPool("workers", size=2)

        @pool.on(Signal)
        async def handle(s: Signal):
            pass

        await pool.start()
        health = pool.health()
        assert health["pool"] == "workers"
        assert health["healthy"] is True
        assert health["size"] == 2
        assert len(health["workers"]) == 2
        await pool.stop()

    async def test_pool_stats(self):
        pool = AgentPool("workers", size=2)

        @pool.on(Signal)
        async def handle(s: Signal):
            pass

        await pool.start()
        await pool.workers[0].inbox.send(Signal())
        await asyncio.sleep(0.05)
        await pool.stop()

        stats = pool.stats
        assert stats["pool"] == "workers"
        assert stats["total_processed"] >= 1
        assert stats["size"] == 2


# === Lifecycle ===


class TestPoolLifecycle:
    async def test_start_stop(self):
        pool = AgentPool("workers", size=2)

        @pool.on(Signal)
        async def handle(s: Signal):
            pass

        await pool.start()
        for w in pool.workers:
            assert w.running

        await pool.stop()
        for w in pool.workers:
            assert not w.running

    async def test_lifecycle_hooks(self):
        pool = AgentPool("workers", size=2)
        started: list[str] = []
        stopped: list[str] = []

        @pool.on_start
        async def on_start():
            started.append("start")

        @pool.on_stop
        async def on_stop():
            stopped.append("stop")

        @pool.on(Signal)
        async def handle(s: Signal):
            pass

        await pool.start()
        await pool.stop()

        # Each worker fires its own hook
        assert len(started) == 2
        assert len(stopped) == 2
