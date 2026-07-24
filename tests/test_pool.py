"""Tests for AgentPool: horizontal scaling primitive."""

import asyncio
import re

import pytest

from signal_gating import Agent, AgentContext, AgentPool, Gate, Mesh, MeshError, Signal


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

    def test_add_pool_preflights_worker_collisions_with_pool_namespace(self):
        existing_pool = AgentPool("batch[1]", size=1)
        mesh = Mesh()
        mesh.add_pool(existing_pool)
        agents_before = mesh.agents
        pools_before = dict(mesh._pools)

        with pytest.raises(MeshError, match=r"pool names: 'batch\[1\]'"):
            mesh.add_pool(AgentPool("batch", size=2))

        assert mesh.agents == agents_before
        assert mesh._pools == pools_before

    def test_add_pool_preflights_duplicate_worker_names(self):
        malformed_pool = AgentPool("batch", size=2)
        malformed_pool._workers[1] = Agent("batch[0]")
        mesh = Mesh()

        with pytest.raises(MeshError, match=r"duplicate worker names: 'batch\[0\]'"):
            mesh.add_pool(malformed_pool)

        assert mesh.agents == []
        assert mesh._pools == {}

    def test_pool_cannot_attach_to_two_meshes(self):
        pool = AgentPool("workers", size=1)
        first_mesh = Mesh()
        second_mesh = Mesh()
        first_mesh.add_pool(pool)

        with pytest.raises(MeshError, match="already attached"):
            second_mesh.add_pool(pool)

        assert second_mesh.agents == []
        assert second_mesh._pools == {}

    async def test_remove_rejects_pool_final_worker(self):
        pool = AgentPool("workers", size=1)
        mesh = Mesh()
        mesh.add_pool(pool)

        with pytest.raises(MeshError, match="final worker"):
            await mesh.remove(pool.workers[0])

        assert mesh.agents == pool.workers
        assert pool.size == 1

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
    def test_unattached_pool_discard_removes_worker(self):
        pool = AgentPool("workers", size=2)

        assert pool.discard("workers[0]") is True
        assert pool.worker_names == ["workers[1]"]

    async def test_attached_pool_rejects_direct_scaling_and_discard(self):
        pool = AgentPool("workers", size=2)
        mesh = Mesh()
        mesh.add_pool(pool)
        error = (
            "Pool 'workers' is attached to a mesh; "
            "use await mesh.scale_pool('workers', size)"
        )

        with pytest.raises(MeshError, match=re.escape(error)):
            pool.scale_to(3)
        with pytest.raises(MeshError, match=re.escape(error)):
            await pool.scale_up()
        with pytest.raises(MeshError, match=re.escape(error)):
            await pool.scale_down()
        with pytest.raises(MeshError, match=re.escape(error)):
            pool.discard("workers[1]")

        assert pool.size == 2

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

    async def test_mesh_live_scaling_preserves_all_pool_connection_shapes(self):
        dispatcher = Agent("dispatcher")
        collector = Agent("collector")
        ingress = AgentPool("ingress", size=1)
        producers = AgentPool("producers", size=1)
        consumers = AgentPool("consumers", size=1)
        ingress_received: dict[str, list[str]] = {}
        collected: list[str] = []
        consumed: dict[str, list[str]] = {}
        gate_calls = {"to_ingress": 0, "from_ingress": 0, "pool_to_pool": 0}

        def counting_gate(name: str) -> Gate:
            def count(signal: Signal) -> Signal:
                gate_calls[name] += 1
                return signal

            return Gate(count, name=name)

        @ingress.on(TaskSignal)
        async def handle_ingress(signal: TaskSignal, ctx: AgentContext):
            ingress_received.setdefault(ctx.agent_name, []).append(signal.task)
            await ctx.emit(ResultSignal(result=f"ingress:{signal.task}"))

        @collector.on(ResultSignal)
        async def collect(signal: ResultSignal):
            collected.append(signal.result)

        @producers.on(TaskSignal)
        async def produce(signal: TaskSignal, ctx: AgentContext):
            await ctx.emit(ResultSignal(result=f"{ctx.agent_name}:{signal.task}"))

        @consumers.on(ResultSignal)
        async def consume(signal: ResultSignal, ctx: AgentContext):
            consumed.setdefault(ctx.agent_name, []).append(signal.result)

        mesh = Mesh([dispatcher, collector])
        mesh.add_pool(ingress)
        mesh.add_pool(producers)
        mesh.add_pool(consumers)
        mesh.connect(dispatcher, ingress, counting_gate("to_ingress"))
        mesh.connect(ingress, collector, counting_gate("from_ingress"))
        mesh.connect(producers, consumers, counting_gate("pool_to_pool"))

        async def exercise_batch(label: str) -> None:
            await dispatcher.emit(TaskSignal(task=label))
            for worker in producers.workers:
                await worker.inbox.send(TaskSignal(task=label))
            await mesh.wait_idle()

        async with mesh:
            await exercise_batch("one")

            new_ingress = await mesh.scale_pool(ingress, 3)
            new_producers = await mesh.scale_pool("producers", 3)
            new_consumers = await mesh.scale_pool(consumers, 3)

            for worker in new_ingress + new_producers + new_consumers:
                assert mesh.get(worker.name) is worker
                assert worker._tracer is mesh.tracer
                assert worker.running

            for index in range(3):
                await dispatcher.emit(TaskSignal(task=f"three-{index}"))
            for worker in producers.workers:
                await worker.inbox.send(TaskSignal(task="three"))
            await mesh.wait_idle()

            removed_ingress = await mesh.scale_pool("ingress", 1)
            removed_producers = await mesh.scale_pool(producers, 1)
            removed_consumers = await mesh.scale_pool("consumers", 1)
            assert [worker.name for worker in removed_ingress] == [
                "ingress[2]",
                "ingress[1]",
            ]
            assert [worker.name for worker in removed_producers] == [
                "producers[2]",
                "producers[1]",
            ]
            assert [worker.name for worker in removed_consumers] == [
                "consumers[2]",
                "consumers[1]",
            ]

            await exercise_batch("one-again")

        assert set(ingress_received) == {
            "ingress[0]",
            "ingress[1]",
            "ingress[2]",
        }
        assert len(collected) == 5
        assert set(consumed) == {
            "consumers[0]",
            "consumers[1]",
            "consumers[2]",
        }
        assert sum(map(len, consumed.values())) == 5
        assert gate_calls == {
            "to_ingress": 5,
            "from_ingress": 5,
            "pool_to_pool": 5,
        }

    async def test_mesh_scale_pool_while_stopped_registers_without_starting(self):
        pool = AgentPool("workers", size=1)
        collector = Agent("collector")
        mesh = Mesh([collector])
        mesh.add_pool(pool)
        mesh.connect(pool, collector)

        created = await mesh.scale_pool(pool, 3)

        assert [worker.name for worker in created] == ["workers[1]", "workers[2]"]
        assert pool.worker_names == ["workers[0]", "workers[1]", "workers[2]"]
        for worker in created:
            assert mesh.get(worker.name) is worker
            assert worker._tracer is mesh.tracer
            assert not worker.running
            assert any(
                getattr(route, "target", None) == collector.name
                for route in worker._outbox
            )

        removed = await mesh.scale_pool("workers", 1)
        assert removed == list(reversed(created))
        assert pool.worker_names == ["workers[0]"]

    async def test_mesh_scale_down_severs_removed_workers_and_prunes_policies(self):
        source = Agent("source")
        target = Agent("target")
        pool = AgentPool("workers", size=3)
        downstream = AgentPool("downstream", size=1)
        source_pool = AgentPool("source-pool", size=1)
        received: list[str] = []
        survivor = pool.workers[0]
        removed_targets = pool.workers[1:]

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal.task)

        mesh = Mesh([source, target])
        mesh.add_pool(pool)
        mesh.add_pool(downstream)
        mesh.add_pool(source_pool)
        mesh.connect(source, pool)
        mesh.connect(pool, target)
        mesh.connect(removed_targets[-1], downstream)
        mesh.connect(source_pool, removed_targets[-1])
        mesh.connect(source, removed_targets[-1])
        mesh.connect(removed_targets[-1], target)
        mesh.load_balance(source, [survivor, *removed_targets])
        mesh.create_topic("work")
        for worker in removed_targets:
            mesh.subscribe(worker, "work")
            mesh.declare_capabilities(worker, "ephemeral")

        async with mesh:
            removed = await mesh.scale_pool(pool, 1)
            assert removed == list(reversed(removed_targets))
            assert all(not worker.running for worker in removed)

            for index in range(6):
                await source.emit(TaskSignal(task=f"job-{index}"))
            await mesh.wait_idle()

        assert pool.workers == [survivor]
        assert received
        assert all(worker not in mesh.agents for worker in removed_targets)
        for worker in removed_targets:
            with pytest.raises(MeshError, match="not found"):
                mesh.get(worker.name)
            assert worker.name not in mesh._capabilities
            assert worker not in mesh._topics["work"]
            assert worker._outbox == []
            assert all(
                edge.source is not worker and edge.target is not worker
                for edge in mesh.edges
            )
            assert all(
                getattr(route, "target", None) != worker.name
                for agent in mesh.agents
                for route in agent._outbox
            )

        assert any(
            connection.source is pool and connection.target is target
            for connection in mesh._pool_connections
        )
        assert all(
            connection.source not in removed_targets
            and connection.target not in removed_targets
            for connection in mesh._pool_connections
        )

    async def test_mesh_scale_pool_serializes_concurrent_resizes(self):
        pool = AgentPool("workers", size=1)
        mesh = Mesh()
        mesh.add_pool(pool)
        entered_start = asyncio.Event()
        release_start = asyncio.Event()

        async with mesh:

            @pool.on_start
            async def pause_new_worker_start():
                entered_start.set()
                await release_start.wait()

            first = asyncio.create_task(mesh.scale_pool(pool, 2))
            await entered_start.wait()
            second = asyncio.create_task(mesh.scale_pool("workers", 3))
            await asyncio.sleep(0)

            assert not second.done()
            assert pool.size == 1

            release_start.set()
            first_created, second_created = await asyncio.gather(first, second)

            assert [worker.name for worker in first_created] == ["workers[1]"]
            assert [worker.name for worker in second_created] == ["workers[2]"]
            assert pool.worker_names == ["workers[0]", "workers[1]", "workers[2]"]

    async def test_mesh_scale_pool_validates_size(self):
        pool = AgentPool("workers", size=1)
        mesh = Mesh()
        mesh.add_pool(pool)

        with pytest.raises(ValueError, match="Pool size must be at least 1"):
            await mesh.scale_pool(pool, 0)


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
            # Send directly using select_worker (which picks least loaded)
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
