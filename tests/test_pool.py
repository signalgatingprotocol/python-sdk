"""Tests for AgentPool: horizontal scaling primitive."""

import asyncio
import re

import pytest

from signal_gating import (
    Agent,
    AgentContext,
    AgentError,
    AgentPool,
    Gate,
    Mesh,
    MeshError,
    Signal,
)


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


class TestPoolDisconnection:
    async def test_disconnect_agent_to_pool_removes_route_and_policy(self):
        source = Agent("source")
        pool = AgentPool("workers", size=2)
        received: list[str] = []

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal.task)

        mesh = Mesh([source])
        mesh.add_pool(pool)
        mesh.connect(source, pool)

        async with mesh:
            await source.emit(TaskSignal(task="before"))
            await mesh.wait_idle()
            assert mesh.disconnect(source, pool) == 1
            await source.emit(TaskSignal(task="after"))
            await mesh.wait_idle()

        assert received == ["before"]
        assert mesh.disconnect(source, pool) == 0

    async def test_disconnect_pool_to_agent_survives_source_scaling(self):
        pool = AgentPool("workers", size=1)
        target = Agent("target")
        received: list[str] = []

        @target.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal.task)

        mesh = Mesh([target])
        mesh.add_pool(pool)
        mesh.connect(pool, target)
        assert mesh.disconnect("workers", "target") == 1
        await mesh.scale_pool(pool, 3)

        async with mesh:
            for worker in pool.workers:
                await worker.emit(TaskSignal(task=worker.name))
            await mesh.wait_idle()

        assert received == []

    async def test_disconnect_pool_to_pool_isolated_and_reconnects_once(self):
        sources = AgentPool("sources", size=2)
        targets = AgentPool("targets", size=2)
        audit = Agent("audit")
        received: list[str] = []
        audited: list[str] = []

        @targets.on(TaskSignal)
        async def handle(signal: TaskSignal):
            received.append(signal.task)

        @audit.on(TaskSignal)
        async def handle_audit(signal: TaskSignal):
            audited.append(signal.task)

        mesh = Mesh([audit])
        mesh.add_pool(sources)
        mesh.add_pool(targets)
        mesh.connect(sources, targets)
        mesh.connect(sources, audit)
        assert mesh.disconnect("sources", targets) == 1
        mesh.connect(sources, targets)

        async with mesh:
            await sources.workers[0].emit(TaskSignal(task="one"))
            await mesh.wait_idle()

        assert received == ["one"]
        assert audited == ["one"]


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

    async def test_pool_workers_have_independent_builtin_gate_state(self):
        pool = AgentPool("workers", size=2, gates=[Gate.throttle(1)])
        received: list[str] = []

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext) -> None:
            received.append(ctx.agent_name)

        mesh = Mesh()
        mesh.add_pool(pool)
        async with mesh:
            await pool.workers[0].inbox.send(TaskSignal(task="a"))
            await pool.workers[1].inbox.send(TaskSignal(task="b"))
            await mesh.wait_idle()

        assert sorted(received) == ["workers[0]", "workers[1]"]
        assert pool.workers[0].gates[0] is not pool.workers[1].gates[0]

    async def test_pool_workers_have_independent_deduplicate_state(self):
        pool = AgentPool("workers", size=2, gates=[Gate.deduplicate()])
        left, right = (worker.gates[0] for worker in pool.workers)

        assert await left.process(TaskSignal(task="same")) is not None
        assert await right.process(TaskSignal(task="same")) is not None

    async def test_pool_workers_have_independent_rate_limit_state(self, monkeypatch):
        sleep_delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            sleep_delays.append(delay)

        monkeypatch.setattr("signal_gating.gate.time.monotonic", lambda: 100.0)
        monkeypatch.setattr("signal_gating.gate.asyncio.sleep", record_sleep)
        pool = AgentPool("workers", size=2, gates=[Gate.rate_limit(1)])
        left, right = (worker.gates[0] for worker in pool.workers)

        assert await left.process(TaskSignal(task="left")) is not None
        assert await right.process(TaskSignal(task="right")) is not None
        assert sleep_delays == []

    async def test_pool_workers_have_independent_batch_state(self):
        pool = AgentPool("workers", size=2, gates=[Gate.batch(2)])
        left, right = (worker.gates[0] for worker in pool.workers)

        assert await left.process(TaskSignal(task="left-1")) is None
        assert await right.process(TaskSignal(task="right-1")) is None
        left_batch = await left.process(TaskSignal(task="left-2"))
        right_batch = await right.process(TaskSignal(task="right-2"))

        assert left_batch is not None
        assert left_batch.metadata["batch_size"] == 2
        assert right_batch is not None
        assert right_batch.metadata["batch_size"] == 2

    async def test_pool_workers_have_independent_debounce_state(self):
        pool = AgentPool("workers", size=2, gates=[Gate.debounce(0.01)])
        left, right = (worker.gates[0] for worker in pool.workers)

        results = await asyncio.gather(
            left.process(TaskSignal(task="left")),
            right.process(TaskSignal(task="right")),
        )

        assert all(result is not None for result in results)

    async def test_pool_workers_have_independent_window_state(self):
        pool = AgentPool(
            "workers", size=2, gates=[Gate.window(seconds=60, min_signals=2)]
        )
        left, right = (worker.gates[0] for worker in pool.workers)

        assert await left.process(TaskSignal(task="left-1")) is None
        assert await right.process(TaskSignal(task="right-1")) is None
        left_window = await left.process(TaskSignal(task="left-2"))
        right_window = await right.process(TaskSignal(task="right-2"))

        assert left_window is not None
        assert left_window.metadata["window_size"] == 2
        assert right_window is not None
        assert right_window.metadata["window_size"] == 2

    async def test_pool_workers_have_independent_circuit_breaker_state(self):
        calls = 0

        async def reject(signal: Signal) -> None:
            nonlocal calls
            calls += 1
            return None

        pool = AgentPool(
            "workers",
            size=2,
            gates=[
                Gate.circuit_breaker(
                    Gate(reject), failure_threshold=1, recovery_timeout=60
                )
            ],
        )
        left, right = (worker.gates[0] for worker in pool.workers)

        assert await left.process(TaskSignal(task="left")) is None
        assert await right.process(TaskSignal(task="right")) is None
        assert calls == 2

    @pytest.mark.parametrize(
        "build_gate",
        [
            pytest.param(
                lambda: Gate.throttle(1) >> Gate.passthrough(), id="chain"
            ),
            pytest.param(lambda: Gate.throttle(1) | Gate.block(), id="either"),
            pytest.param(
                lambda: Gate.throttle(1) & Gate.passthrough(), id="both"
            ),
            pytest.param(
                lambda: Gate.retry(Gate.throttle(1), max_attempts=1), id="retry"
            ),
            pytest.param(
                lambda: Gate.timeout(Gate.throttle(1), seconds=1), id="timeout"
            ),
            pytest.param(
                lambda: Gate.when(lambda signal: True, Gate.throttle(1)),
                id="when",
            ),
            pytest.param(
                lambda: Gate.parallel(
                    Gate.throttle(1), Gate.passthrough(), mode="all"
                ),
                id="parallel",
            ),
            pytest.param(
                lambda: Gate.fallback(Gate.throttle(1), Gate.block()),
                id="fallback",
            ),
        ],
    )
    async def test_pool_workers_fork_nested_builtin_gate_state(self, build_gate):
        pool = AgentPool("workers", size=2, gates=[build_gate()])
        left, right = (worker.gates[0] for worker in pool.workers)

        assert await left.process(TaskSignal(task="left")) is not None
        assert await right.process(TaskSignal(task="right")) is not None

    async def test_pool_workers_fork_inverted_builtin_gate_state(self):
        pool = AgentPool("workers", size=2, gates=[~Gate.deduplicate()])
        left, right = (worker.gates[0] for worker in pool.workers)

        assert await left.process(TaskSignal(task="same")) is None
        assert await right.process(TaskSignal(task="same")) is None

    async def test_pool_workers_share_caller_owned_callable_state(self):
        calls = 0

        def first_only(signal: Signal) -> Signal | None:
            nonlocal calls
            calls += 1
            return signal if calls == 1 else None

        configured_gate = Gate(first_only, name="caller-owned")
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        left, right = (worker.gates[0] for worker in pool.workers)

        assert left is not configured_gate
        assert right is not configured_gate
        assert left is not right
        assert left.name == "caller-owned"
        assert await left.process(TaskSignal(task="left")) is not None
        assert await right.process(TaskSignal(task="right")) is None
        assert calls == 2

    async def test_pool_fork_preserves_custom_gate_subclass_behavior(self):
        class RejectingGate(Gate):
            async def process(self, signal: Signal) -> None:
                return None

        configured_gate = RejectingGate(lambda signal: signal, name="rejecting")
        pool = AgentPool("workers", size=2, gates=[configured_gate])

        for worker in pool.workers:
            forked = worker.gates[0]
            assert forked is not configured_gate
            assert isinstance(forked, RejectingGate)
            assert await forked.process(TaskSignal(task="blocked")) is None

    def test_gate_fork_ignores_untrustworthy_subclass_copy_hook(self):
        class SelfCopyGate(Gate):
            def __copy__(self):
                return self

        configured_gate = SelfCopyGate(lambda signal: signal, name="self-copy")
        first_fork = configured_gate.fork()
        second_fork = configured_gate.fork()
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        wrappers = [
            configured_gate,
            first_fork,
            second_fork,
            *(worker.gates[0] for worker in pool.workers),
        ]

        assert len({id(gate) for gate in wrappers}) == len(wrappers)
        assert all(isinstance(gate, SelfCopyGate) for gate in wrappers)

    async def test_stateful_subclass_forks_preserve_slotted_runtime_policy(self):
        class PolicyThrottle(Gate):
            __slots__ = ("policy",)

            def __init__(self, fn, name=""):
                super().__init__(fn, name)
                self.policy = "deny"

            async def process(self, signal: Signal) -> Signal | None:
                if self.policy != "allow":
                    return None
                return await super().process(signal)

        configured_gate = PolicyThrottle.throttle(1)
        configured_gate.name = "policy-throttle"
        configured_gate.policy = "allow"
        first_fork = configured_gate.fork()
        second_fork = configured_gate.fork()
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        forks = [
            first_fork,
            second_fork,
            *(worker.gates[0] for worker in pool.workers),
        ]

        assert len({id(gate) for gate in forks}) == len(forks)
        assert all(isinstance(gate, PolicyThrottle) for gate in forks)
        assert [gate.policy for gate in forks] == ["allow"] * len(forks)
        assert [gate.name for gate in forks] == ["policy-throttle"] * len(forks)

        assert await configured_gate.process(TaskSignal(task="configured")) is not None
        for index, gate in enumerate(forks):
            assert await gate.process(TaskSignal(task=f"first-{index}")) is not None
            assert await gate.process(TaskSignal(task=f"second-{index}")) is None

    async def test_stateful_forks_preserve_legacy_constructor_fn_wrapper(self):
        audited_tasks: list[str] = []

        class AuditedThrottle(Gate):
            def __init__(self, fn, name=""):
                async def audited(signal: TaskSignal) -> Signal | None:
                    audited_tasks.append(signal.task)
                    return await fn(signal)

                super().__init__(audited, name)

        configured_gate = AuditedThrottle.throttle(1)
        first_fork = configured_gate.fork()
        second_fork = configured_gate.fork()
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        forks = [
            first_fork,
            second_fork,
            *(worker.gates[0] for worker in pool.workers),
        ]

        for index, gate in enumerate(forks):
            assert isinstance(gate, AuditedThrottle)
            assert await gate.process(TaskSignal(task=f"first-{index}")) is not None
            assert await gate.process(TaskSignal(task=f"second-{index}")) is None

        assert audited_tasks == [
            task
            for index in range(len(forks))
            for task in (f"first-{index}", f"second-{index}")
        ]

    async def test_stateful_forks_preserve_overridden_factory_decoration(self):
        decorated_tasks: list[str] = []

        class DecoratedThrottle(Gate):
            @classmethod
            def throttle(cls, max_per_second, name="throttle"):
                gate = super().throttle(max_per_second, name=name)
                builtin_fn = gate._fn

                async def decorated(signal: TaskSignal) -> Signal | None:
                    decorated_tasks.append(signal.task)
                    return await builtin_fn(signal)

                gate._fn = decorated
                return gate

        configured_gate = DecoratedThrottle.throttle(1)
        first_fork = configured_gate.fork()
        second_fork = configured_gate.fork()
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        forks = [
            first_fork,
            second_fork,
            *(worker.gates[0] for worker in pool.workers),
        ]

        for index, gate in enumerate(forks):
            assert isinstance(gate, DecoratedThrottle)
            assert await gate.process(TaskSignal(task=f"first-{index}")) is not None
            assert await gate.process(TaskSignal(task=f"second-{index}")) is None

        assert decorated_tasks == [
            task
            for index in range(len(forks))
            for task in (f"first-{index}", f"second-{index}")
        ]

    async def test_stateful_forks_preserve_fresh_dict_executor(self, monkeypatch):
        class ExecutorThrottle(Gate):
            def __init__(self, fn, name=""):
                super().__init__(fn, name)
                self.executor = fn

            async def process(self, signal: Signal) -> Signal | None:
                return await self.executor(signal)

        monkeypatch.setattr("signal_gating.gate.time.monotonic", lambda: 100.0)
        configured_gate = ExecutorThrottle.throttle(1)
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        forks = [
            configured_gate.fork(),
            configured_gate.fork(),
            *(worker.gates[0] for worker in pool.workers),
        ]

        assert len({id(gate) for gate in forks}) == len(forks)
        for index, gate in enumerate(forks):
            assert await gate.process(TaskSignal(task=f"first-{index}")) is not None
            assert await gate.process(TaskSignal(task=f"second-{index}")) is None

    async def test_stateful_forks_preserve_fresh_slotted_executor(self, monkeypatch):
        class SlottedExecutorThrottle(Gate):
            __slots__ = ("executor",)

            def __init__(self, fn, name=""):
                super().__init__(fn, name)
                self.executor = fn

            async def process(self, signal: Signal) -> Signal | None:
                return await self.executor(signal)

        monkeypatch.setattr("signal_gating.gate.time.monotonic", lambda: 100.0)
        configured_gate = SlottedExecutorThrottle.throttle(1)
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        forks = [
            configured_gate.fork(),
            configured_gate.fork(),
            *(worker.gates[0] for worker in pool.workers),
        ]

        assert len({id(gate) for gate in forks}) == len(forks)
        for index, gate in enumerate(forks):
            assert await gate.process(TaskSignal(task=f"first-{index}")) is not None
            assert await gate.process(TaskSignal(task=f"second-{index}")) is None

    async def test_stateful_forks_do_not_restore_absent_dict_executor(
        self, monkeypatch
    ):
        builds = 0

        class OptionalExecutorThrottle(Gate):
            def __init__(self, fn, name=""):
                nonlocal builds
                super().__init__(fn, name)
                builds += 1
                if builds == 1:
                    self.executor = fn

            async def process(self, signal: Signal) -> Signal | None:
                executor = getattr(self, "executor", self._fn)
                return await executor(signal)

        monkeypatch.setattr("signal_gating.gate.time.monotonic", lambda: 100.0)
        configured_gate = OptionalExecutorThrottle.throttle(1)
        forks = [configured_gate.fork(), configured_gate.fork()]
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        forks.extend(worker.gates[0] for worker in pool.workers)

        assert all(not hasattr(gate, "executor") for gate in forks)
        for index, gate in enumerate(forks):
            assert await gate.process(TaskSignal(task=f"first-{index}")) is not None
            assert await gate.process(TaskSignal(task=f"second-{index}")) is None

    async def test_stateful_forks_do_not_restore_absent_slotted_executor(
        self, monkeypatch
    ):
        builds = 0

        class OptionalSlottedExecutorThrottle(Gate):
            __slots__ = ("executor",)

            def __init__(self, fn, name=""):
                nonlocal builds
                super().__init__(fn, name)
                builds += 1
                if builds == 1:
                    self.executor = fn

            async def process(self, signal: Signal) -> Signal | None:
                executor = getattr(self, "executor", self._fn)
                return await executor(signal)

        monkeypatch.setattr("signal_gating.gate.time.monotonic", lambda: 100.0)
        configured_gate = OptionalSlottedExecutorThrottle.throttle(1)
        forks = [configured_gate.fork(), configured_gate.fork()]
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        forks.extend(worker.gates[0] for worker in pool.workers)

        assert all(not hasattr(gate, "executor") for gate in forks)
        for index, gate in enumerate(forks):
            assert await gate.process(TaskSignal(task=f"first-{index}")) is not None
            assert await gate.process(TaskSignal(task=f"second-{index}")) is None

    async def test_stateful_forks_preserve_rebuilt_none_executor(self, monkeypatch):
        builds = 0

        class OptionalExecutorThrottle(Gate):
            def __init__(self, fn, name=""):
                nonlocal builds
                super().__init__(fn, name)
                builds += 1
                self.executor = fn if builds == 1 else None

            async def process(self, signal: Signal) -> Signal | None:
                executor = self.executor if callable(self.executor) else self._fn
                return await executor(signal)

        monkeypatch.setattr("signal_gating.gate.time.monotonic", lambda: 100.0)
        configured_gate = OptionalExecutorThrottle.throttle(1)
        forks = [configured_gate.fork(), configured_gate.fork()]
        pool = AgentPool("workers", size=2, gates=[configured_gate])
        forks.extend(worker.gates[0] for worker in pool.workers)

        assert all(gate.executor is None for gate in forks)
        for index, gate in enumerate(forks):
            assert await gate.process(TaskSignal(task=f"first-{index}")) is not None
            assert await gate.process(TaskSignal(task=f"second-{index}")) is None

    @pytest.mark.parametrize(
        "build_gate",
        [
            pytest.param(lambda cls: cls.rate_limit(1000), id="rate-limit"),
            pytest.param(lambda cls: cls.deduplicate(), id="deduplicate"),
            pytest.param(
                lambda cls: cls.retry(Gate.passthrough(), max_attempts=1),
                id="retry",
            ),
            pytest.param(
                lambda cls: cls.circuit_breaker(Gate.passthrough()),
                id="circuit-breaker",
            ),
            pytest.param(
                lambda cls: cls.timeout(Gate.passthrough(), seconds=1),
                id="timeout",
            ),
            pytest.param(
                lambda cls: cls.when(lambda signal: True, Gate.passthrough()),
                id="when",
            ),
            pytest.param(lambda cls: cls.throttle(1000), id="throttle"),
            pytest.param(lambda cls: cls.batch(1), id="batch"),
            pytest.param(
                lambda cls: cls.parallel(Gate.passthrough()), id="parallel"
            ),
            pytest.param(
                lambda cls: cls.fallback(Gate.passthrough()), id="fallback"
            ),
            pytest.param(lambda cls: cls.debounce(0), id="debounce"),
            pytest.param(lambda cls: cls.window(1), id="window"),
        ],
    )
    def test_builtin_gate_factories_support_legacy_subclass_constructor(
        self, build_gate
    ):
        class LegacyConstructorGate(Gate):
            def __init__(self, fn, name=""):
                super().__init__(fn, name)

        configured_gate = build_gate(LegacyConstructorGate)
        forked = configured_gate.fork()

        assert isinstance(configured_gate, LegacyConstructorGate)
        assert isinstance(forked, LegacyConstructorGate)
        assert forked is not configured_gate

    @pytest.mark.parametrize(
        "build_gate",
        [
            pytest.param(lambda: Gate.throttle(1), id="built-in"),
            pytest.param(
                lambda: Gate.throttle(1) >> Gate.passthrough(), id="composite"
            ),
        ],
    )
    def test_pool_forks_preserve_reassigned_public_gate_name(self, build_gate):
        configured_gate = build_gate()
        configured_gate.name = "renamed-by-caller"

        pool = AgentPool("workers", size=2, gates=[configured_gate])

        assert [worker.gates[0].name for worker in pool.workers] == [
            "renamed-by-caller",
            "renamed-by-caller",
        ]

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

    async def test_remove_rejects_any_attached_pool_worker(self):
        pool = AgentPool("workers", size=2)
        mesh = Mesh()
        mesh.add_pool(pool)
        workers_before = pool.workers

        with pytest.raises(MeshError, match="use await mesh.scale_pool"):
            await mesh.remove(pool.workers[-1])

        assert mesh.agents == workers_before
        assert pool.workers == workers_before

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
    async def test_scaled_worker_gets_fresh_builtin_gate_state(self):
        pool = AgentPool("workers", size=1, gates=[Gate.throttle(1)])
        received: list[str] = []

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext) -> None:
            received.append(ctx.agent_name)

        mesh = Mesh()
        mesh.add_pool(pool)
        original = pool.workers[0]

        async with mesh:
            await original.inbox.send(TaskSignal(task="original"))
            await mesh.wait_idle()
            added = await mesh.scale_pool(pool, 2)
            assert len(added) == 1
            await added[0].inbox.send(TaskSignal(task="scaled"))
            await mesh.wait_idle()

        assert received == ["workers[0]", "workers[1]"]
        assert added[0].gates[0] is not original.gates[0]

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

    async def test_pool_connection_added_during_start_wires_staged_worker(self):
        pool = AgentPool("workers", size=1)
        early = Agent("early")
        late = Agent("late")
        early_results: list[str] = []
        late_results: list[str] = []
        entered_start = asyncio.Event()
        release_start = asyncio.Event()

        @pool.on(TaskSignal)
        async def produce(signal: TaskSignal, ctx: AgentContext):
            await ctx.emit(ResultSignal(result=signal.task))

        @early.on(ResultSignal)
        async def collect_early(signal: ResultSignal):
            early_results.append(signal.result)

        @late.on(ResultSignal)
        async def collect_late(signal: ResultSignal):
            late_results.append(signal.result)

        mesh = Mesh([early, late])
        mesh.add_pool(pool)
        mesh.connect(pool, early)

        async with mesh:

            @pool.on_start
            async def pause_new_worker_start():
                entered_start.set()
                await release_start.wait()

            scaling = asyncio.create_task(mesh.scale_pool(pool, 2))
            await entered_start.wait()
            mesh.connect(pool, late)
            release_start.set()
            new_workers = await scaling

            await new_workers[0].inbox.send(TaskSignal(task="staged"))
            await mesh.wait_idle()

        assert early_results == ["staged"]
        assert late_results == ["staged"]

    async def test_mesh_scale_up_start_failure_rolls_back_staged_workers(self):
        pool = AgentPool("workers", size=1)
        collector = Agent("collector")
        mesh = Mesh([collector])
        mesh.add_pool(pool)
        mesh.connect(pool, collector)
        original_agents = mesh.agents
        original_edges = mesh.edges
        original_outputs = list(pool.workers[0]._outbox)
        entered_start = asyncio.Event()
        release_start = asyncio.Event()

        async with mesh:

            @pool.on_start
            async def fail_new_worker_start():
                entered_start.set()
                await release_start.wait()
                raise RuntimeError("staged start failed")

            scaling = asyncio.create_task(mesh.scale_pool(pool, 3))
            await entered_start.wait()
            staged = [agent for agent in mesh.agents if agent not in original_agents]
            assert [worker.name for worker in staged] == ["workers[1]", "workers[2]"]
            assert pool.worker_names == ["workers[0]"]

            release_start.set()
            with pytest.raises(AgentError, match="staged start failed"):
                await scaling

            assert mesh.agents == original_agents
            assert mesh.edges == original_edges
            assert pool.worker_names == ["workers[0]"]
            assert pool.workers[0].running
            assert pool.workers[0]._outbox == original_outputs
            for worker in staged:
                assert not worker.running
                assert worker._outbox == []

    async def test_mesh_scale_up_cancellation_rolls_back_staged_workers(self):
        pool = AgentPool("workers", size=1)
        mesh = Mesh()
        mesh.add_pool(pool)
        original_agents = mesh.agents
        entered_start = asyncio.Event()
        release_start = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()

        async with mesh:

            @pool.on_start
            async def pause_new_worker_start():
                entered_start.set()
                await release_start.wait()

            @pool.on_stop
            async def pause_staged_worker_cleanup():
                cleanup_started.set()
                await release_cleanup.wait()

            scaling = asyncio.create_task(mesh.scale_pool(pool, 2))
            await entered_start.wait()
            staged = [agent for agent in mesh.agents if agent not in original_agents]
            scaling.cancel()
            await cleanup_started.wait()
            scaling.cancel()
            await asyncio.sleep(0)
            scaling.cancel()
            await asyncio.sleep(0)
            completed_before_release = scaling.done()
            release_cleanup.set()

            with pytest.raises(asyncio.CancelledError):
                await scaling

            assert not completed_before_release
            assert mesh.agents == original_agents
            assert pool.worker_names == ["workers[0]"]
            assert all(not worker.running for worker in staged)
            assert all(worker._outbox == [] for worker in staged)

            release_start.set()

    async def test_mesh_scale_down_stops_new_routing_and_drains_current_work(self):
        source = Agent("source")
        collector = Agent("collector")
        pool = AgentPool("workers", size=2)
        handled: dict[str, list[str]] = {}
        collected: list[str] = []
        retiring_started = asyncio.Event()
        survivor_processed = asyncio.Event()
        release_retiring = asyncio.Event()

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            handled.setdefault(ctx.agent_name, []).append(signal.task)
            if signal.task == "retiring":
                retiring_started.set()
                await release_retiring.wait()
            if signal.task == "after-scale-started":
                survivor_processed.set()
            await ctx.emit(ResultSignal(result=f"{ctx.agent_name}:{signal.task}"))

        @collector.on(ResultSignal)
        async def collect(signal: ResultSignal):
            collected.append(signal.result)

        mesh = Mesh([source, collector])
        mesh.add_pool(pool)
        mesh.connect(source, pool)
        mesh.connect(pool, collector)
        retiring = pool.workers[-1]

        async with mesh:
            await source.emit(TaskSignal(task="warmup"))
            await mesh.wait_idle()
            await source.emit(TaskSignal(task="retiring"))
            await retiring_started.wait()
            await retiring.inbox.send(TaskSignal(task="queued"))

            scaling = asyncio.create_task(mesh.scale_pool(pool, 1))
            while pool.size != 1:
                await asyncio.sleep(0)
            assert not scaling.done()

            await source.emit(TaskSignal(task="after-scale-started"))
            await survivor_processed.wait()
            assert handled[retiring.name] == ["retiring"]

            release_retiring.set()
            removed = await scaling
            assert removed == [retiring]

        assert not retiring.running
        assert retiring not in mesh.agents
        assert handled[retiring.name] == ["retiring", "queued"]
        assert f"{retiring.name}:retiring" in collected
        assert f"{retiring.name}:queued" in collected

    async def test_mesh_scale_down_cancellation_finishes_retirement_cleanup(self):
        pool = AgentPool("workers", size=2)
        mesh = Mesh()
        mesh.add_pool(pool)
        retiring = pool.workers[-1]
        retiring_started = asyncio.Event()
        release_retiring = asyncio.Event()

        @pool.on(TaskSignal)
        async def block_retiring(signal: TaskSignal, ctx: AgentContext):
            if ctx.agent_name == retiring.name:
                retiring_started.set()
                await release_retiring.wait()

        async with mesh:
            await retiring.inbox.send(TaskSignal(task="block"))
            await retiring_started.wait()

            scaling = asyncio.create_task(mesh.scale_pool(pool, 1))
            while pool.size != 1:
                await asyncio.sleep(0)
            scaling.cancel()
            await asyncio.sleep(0)
            scaling.cancel()
            await asyncio.sleep(0)
            completed_before_release = scaling.done()

            release_retiring.set()
            with pytest.raises(asyncio.CancelledError):
                await scaling

            assert not completed_before_release
            assert not retiring.running
            assert retiring not in mesh.agents
            assert retiring._outbox == []

    async def test_mesh_scale_down_stops_worker_during_restart_backoff(self):
        crashed = asyncio.Event()

        def crash_gate(signal: Signal) -> Signal:
            crashed.set()
            raise RuntimeError("infrastructure failure")

        pool = AgentPool(
            "workers",
            size=2,
            gates=[Gate(crash_gate)],
            max_restarts=1,
            restart_delay=0.2,
        )
        mesh = Mesh()
        mesh.add_pool(pool)
        retiring = pool.workers[-1]
        stop_calls = 0

        @pool.on_stop
        async def record_stop():
            nonlocal stop_calls
            stop_calls += 1

        await mesh.start()
        try:
            await retiring.inbox.send(TaskSignal(task="crash"))
            await crashed.wait()
            while retiring._restart_count == 0 or retiring.running:
                await asyncio.sleep(0)

            supervisor = retiring._task
            assert supervisor is not None and not supervisor.done()

            removed = await mesh.scale_pool(pool, 1)

            assert removed == [retiring]
            assert supervisor.done()
            assert retiring._task is None
            assert retiring.inbox.closed
            assert stop_calls == 1
        finally:
            if retiring._task is not None or not retiring.inbox.closed:
                await retiring.stop()
            await mesh.stop()

    async def test_mesh_scale_down_stopped_mesh_still_stops_removed_worker(self):
        pool = AgentPool("workers", size=2)
        mesh = Mesh()
        mesh.add_pool(pool)
        retiring = pool.workers[-1]
        stop_calls = 0

        @pool.on_stop
        async def record_stop():
            nonlocal stop_calls
            stop_calls += 1

        try:
            removed = await mesh.scale_pool(pool, 1)

            assert removed == [retiring]
            assert retiring.inbox.closed
            assert stop_calls == 1
        finally:
            if not retiring.inbox.closed:
                await retiring.stop()

    async def test_mesh_scale_down_drains_target_selected_before_async_gate(self):
        source = Agent("source")
        pool = AgentPool("workers", size=2)
        gate_entered = asyncio.Event()
        release_gate = asyncio.Event()
        gate_calls: list[str] = []
        received: dict[str, list[str]] = {}

        async def pause_race(signal: Signal) -> Signal:
            assert isinstance(signal, TaskSignal)
            gate_calls.append(signal.task)
            if signal.task == "race":
                gate_entered.set()
                await release_gate.wait()
            return signal

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received.setdefault(ctx.agent_name, []).append(signal.task)

        mesh = Mesh([source])
        mesh.add_pool(pool)
        mesh.connect(source, pool, Gate(pause_race, name="pause-race"))
        survivor, retiring = pool.workers

        async with mesh:
            await source.emit(TaskSignal(task="warmup"))
            await mesh.wait_idle()

            delivery = asyncio.create_task(source.emit(TaskSignal(task="race")))
            await gate_entered.wait()
            scaling = asyncio.create_task(mesh.scale_pool(pool, 1))
            while pool.size != 1:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            completed_before_release = scaling.done()

            release_gate.set()
            await delivery
            removed = await scaling
            await mesh.wait_idle()

            assert removed == [retiring]
            assert not completed_before_release

        assert received == {
            survivor.name: ["warmup"],
            retiring.name: ["race"],
        }
        assert gate_calls == ["warmup", "race"]

    async def test_mesh_pool_interceptor_target_stays_stable_during_retirement(self):
        source = Agent("source")
        pool = AgentPool("workers", size=2)
        interceptor_entered = asyncio.Event()
        release_interceptor = asyncio.Event()
        intercepted_targets: list[str] = []
        received: dict[str, list[str]] = {}

        async def pause_interceptor(
            signal: Signal,
            source_name: str,
            target_name: str,
        ) -> Signal:
            if isinstance(signal, TaskSignal) and signal.task == "race":
                intercepted_targets.append(target_name)
                interceptor_entered.set()
                await release_interceptor.wait()
            return signal

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received.setdefault(ctx.agent_name, []).append(signal.task)

        mesh = Mesh([source])
        mesh.add_pool(pool)
        mesh.connect(source, pool)
        mesh.intercept(pause_interceptor)
        survivor, retiring = pool.workers

        async with mesh:
            await source.emit(TaskSignal(task="warmup"))
            await mesh.wait_idle()

            delivery = asyncio.create_task(source.emit(TaskSignal(task="race")))
            await interceptor_entered.wait()
            scaling = asyncio.create_task(mesh.scale_pool(pool, 1))
            while pool.size != 1:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            completed_before_release = scaling.done()

            release_interceptor.set()
            await delivery
            removed = await scaling
            await mesh.wait_idle()

            assert removed == [retiring]
            assert not completed_before_release

        assert intercepted_targets == [retiring.name]
        assert received == {
            survivor.name: ["warmup"],
            retiring.name: ["race"],
        }

    async def test_mesh_scale_down_waits_for_selected_pool_send(self, monkeypatch):
        source = Agent("source")
        pool = AgentPool("workers", size=2)
        received: dict[str, list[str]] = {}
        send_entered = asyncio.Event()
        release_send = asyncio.Event()

        @pool.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            received.setdefault(ctx.agent_name, []).append(signal.task)

        mesh = Mesh([source])
        mesh.add_pool(pool)
        mesh.connect(source, pool)
        survivor, retiring = pool.workers

        async with mesh:
            await source.emit(TaskSignal(task="warmup"))
            await mesh.wait_idle()
            original_send = retiring.inbox.send

            async def hold_selected_send(signal: Signal):
                send_entered.set()
                await release_send.wait()
                await original_send(signal)

            monkeypatch.setattr(retiring.inbox, "send", hold_selected_send)
            delivery = asyncio.create_task(source.emit(TaskSignal(task="selected")))
            await send_entered.wait()
            scaling = asyncio.create_task(mesh.scale_pool(pool, 1))
            while pool.size != 1:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            completed_before_release = scaling.done()

            release_send.set()
            await delivery
            removed = await scaling
            await mesh.wait_idle()

            assert removed == [retiring]
            assert not completed_before_release

        assert received == {
            survivor.name: ["warmup"],
            retiring.name: ["selected"],
        }

    async def test_mesh_rejects_direct_removal_while_pool_worker_retires(self):
        pool = AgentPool("workers", size=2)
        mesh = Mesh()
        mesh.add_pool(pool)
        retiring = pool.workers[-1]
        stop_entered = asyncio.Event()
        release_stop = asyncio.Event()

        @pool.on_stop
        async def pause_retirement():
            stop_entered.set()
            await release_stop.wait()

        async with mesh:
            scaling = asyncio.create_task(mesh.scale_pool(pool, 1))
            await stop_entered.wait()
            scaling_result: list[Agent] | BaseException | None = None
            try:
                with pytest.raises(
                    MeshError,
                    match="use await mesh.scale_pool\\('workers', size\\)",
                ):
                    await mesh.remove(retiring)
            finally:
                release_stop.set()
                [scaling_result] = await asyncio.gather(
                    scaling,
                    return_exceptions=True,
                )

            assert scaling_result == [retiring]
            assert retiring not in mesh.agents
            assert not retiring.running

    async def test_mesh_rejects_direct_removal_of_staged_pool_worker(self):
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

            scaling = asyncio.create_task(mesh.scale_pool(pool, 2))
            await entered_start.wait()
            staged = mesh.get("workers[1]")
            added: list[Agent] = []
            try:
                with pytest.raises(
                    MeshError,
                    match="use await mesh.scale_pool\\('workers', size\\)",
                ):
                    await mesh.remove(staged)
            finally:
                release_start.set()
                added = await scaling
                for worker in added:
                    if worker not in mesh.agents:
                        await worker.stop()

            assert added == [staged]
            assert pool.workers[-1] is staged
            assert mesh.get(staged.name) is staged
            assert staged.running

    async def test_mesh_scale_up_inherits_configuration_added_during_start(self):
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

            scaling = asyncio.create_task(mesh.scale_pool(pool, 2))
            await entered_start.wait()

            @pool.on(ResultSignal)
            async def late_handler(signal: ResultSignal):
                pass

            @pool.on_any
            async def late_any_handler(signal: Signal):
                pass

            @pool.once(TaskSignal)
            async def late_once_handler(signal: TaskSignal):
                pass

            async def late_middleware(signal: Signal, call_next):
                return await call_next(signal)

            pool.use(late_middleware)

            @pool.on_start
            async def late_start_hook():
                pass

            @pool.on_stop
            async def late_stop_hook():
                pass

            @pool.on_error
            async def late_error_hook(signal: Signal, error: Exception):
                pass

            release_start.set()
            added = await scaling
            incumbent, staged = pool.workers

            assert added == [staged]
            assert late_handler in staged._handlers[ResultSignal]
            assert late_any_handler in staged._handlers[Signal]
            assert any(
                getattr(handler, "__wrapped__", None) is late_once_handler
                for handler in staged._handlers[TaskSignal]
            )
            assert late_middleware in staged._middleware
            assert late_start_hook in staged._on_start_hooks
            assert late_stop_hook in staged._on_stop_hooks
            assert late_error_hook in staged._on_error_hooks
            assert staged._handlers.keys() == incumbent._handlers.keys()
            assert len(staged._middleware) == len(incumbent._middleware)
            assert len(staged._on_start_hooks) == len(incumbent._on_start_hooks)
            assert len(staged._on_stop_hooks) == len(incumbent._on_stop_hooks)
            assert len(staged._on_error_hooks) == len(incumbent._on_error_hooks)

    async def test_mesh_scale_pool_validates_size(self):
        pool = AgentPool("workers", size=1)
        mesh = Mesh()
        mesh.add_pool(pool)

        assert await mesh.scale_pool(pool, pool.size) == []

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
