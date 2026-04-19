"""AgentPool — elastic horizontal scaling for agent workloads.

A pool manages N workers that share the same handler configuration, distributes
signals across them, and can scale up or down at runtime.

    pool = AgentPool("workers", size=3, gates=[Gate.by_priority(3)])

    @pool.on(TaskSignal)
    async def handle(signal: TaskSignal, ctx: AgentContext):
        await ctx.emit(ResultSignal(result="done"))

    mesh = Mesh()
    mesh.add_pool(pool)
    mesh.connect(coordinator, pool)  # Load-balanced across workers
    mesh.connect(pool, collector)    # All workers emit to collector
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from signal_gating.agent import Agent, ErrorHook, Handler, LifecycleHook, Middleware
from signal_gating.gate import Gate
from signal_gating.signal import Signal

T = TypeVar("T", bound=Signal)

logger = logging.getLogger("signal_gating.pool")


class AgentPool:
    """An elastic pool of identical agents for horizontal scaling.

    Each worker in the pool shares the same handler and gate configuration
    but maintains independent state. Signals are distributed across workers
    using configurable strategies (round-robin or least-loaded).

    The pool integrates with Mesh as a first-class primitive:
        - ``mesh.add_pool(pool)`` adds all workers
        - ``mesh.connect(source, pool)`` load-balances to workers
        - ``mesh.connect(pool, target)`` all workers emit to target

    Workers are named ``{pool_name}[0]``, ``{pool_name}[1]``, etc.
    """

    def __init__(
        self,
        name: str,
        size: int = 3,
        gates: list[Gate] | None = None,
        buffer_size: int = 1000,
        max_restarts: int = 3,
        restart_delay: float = 1.0,
        priority_inbox: bool = False,
        strategy: str = "round_robin",
    ):
        if size < 1:
            raise ValueError("Pool size must be at least 1")
        if strategy not in ("round_robin", "least_loaded"):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        self.name = name
        self._size = size
        self._gates = gates or []
        self._buffer_size = buffer_size
        self._max_restarts = max_restarts
        self._restart_delay = restart_delay
        self._priority_inbox = priority_inbox
        self._strategy = strategy

        # Handler registry — applied to all workers on creation
        self._handler_registry: list[tuple[type[Signal], Handler]] = []
        self._any_handlers: list[Handler] = []
        self._once_handlers: list[tuple[type[Signal], Handler]] = []
        self._middleware: list[Middleware] = []
        self._on_start_hooks: list[LifecycleHook] = []
        self._on_stop_hooks: list[LifecycleHook] = []
        self._on_error_hooks: list[ErrorHook] = []

        # Create workers
        self._workers: list[Agent] = []
        self._name_counter = itertools.count()
        self._robin_counter = itertools.count()
        for _ in range(size):
            self._workers.append(self._create_worker())

    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def workers(self) -> list[Agent]:
        return list(self._workers)

    @property
    def worker_names(self) -> list[str]:
        return [w.name for w in self._workers]

    def _create_worker(self) -> Agent:
        """Create a new worker with the pool's configuration."""
        index = next(self._name_counter)
        worker = Agent(
            name=f"{self.name}[{index}]",
            gates=list(self._gates),
            buffer_size=self._buffer_size,
            max_restarts=self._max_restarts,
            restart_delay=self._restart_delay,
            priority_inbox=self._priority_inbox,
        )
        # Apply all registered handlers
        for signal_type, handler in self._handler_registry:
            worker._handlers.setdefault(signal_type, []).append(handler)
        for handler in self._any_handlers:
            worker._handlers.setdefault(Signal, []).append(handler)
        for signal_type, handler in self._once_handlers:
            worker.once(signal_type)(handler)
        for mw in self._middleware:
            worker.use(mw)
        for start_hook in self._on_start_hooks:
            worker._on_start_hooks.append(start_hook)
        for stop_hook in self._on_stop_hooks:
            worker._on_stop_hooks.append(stop_hook)
        for err_hook in self._on_error_hooks:
            worker._on_error_hooks.append(err_hook)
        return worker

    # --- Handler Registration (mirrors Agent API) ---

    def on(self, signal_type: type[T]) -> Callable[[Handler], Handler]:
        """Register a handler for a signal type across all pool workers."""

        def decorator(fn: Handler) -> Handler:
            self._handler_registry.append((signal_type, fn))
            for worker in self._workers:
                worker._handlers.setdefault(signal_type, []).append(fn)
            return fn

        return decorator

    def on_any(self, fn: Handler) -> Handler:
        """Register a handler for all signal types across all pool workers."""
        self._any_handlers.append(fn)
        for worker in self._workers:
            worker._handlers.setdefault(Signal, []).append(fn)
        return fn

    def once(self, signal_type: type[T]) -> Callable[[Handler], Handler]:
        """Register a once-handler across all pool workers."""

        def decorator(fn: Handler) -> Handler:
            self._once_handlers.append((signal_type, fn))
            for worker in self._workers:
                worker.once(signal_type)(fn)
            return fn

        return decorator

    def on_start(self, fn: LifecycleHook) -> LifecycleHook:
        """Register a start hook for all pool workers."""
        self._on_start_hooks.append(fn)
        for worker in self._workers:
            worker._on_start_hooks.append(fn)
        return fn

    def on_stop(self, fn: LifecycleHook) -> LifecycleHook:
        """Register a stop hook for all pool workers."""
        self._on_stop_hooks.append(fn)
        for worker in self._workers:
            worker._on_stop_hooks.append(fn)
        return fn

    def on_error(self, fn: ErrorHook) -> ErrorHook:
        """Register an error hook for all pool workers."""
        self._on_error_hooks.append(fn)
        for worker in self._workers:
            worker._on_error_hooks.append(fn)
        return fn

    def use(self, middleware: Middleware) -> None:
        """Add middleware to all pool workers."""
        self._middleware.append(middleware)
        for worker in self._workers:
            worker.use(middleware)

    # --- Scaling ---

    def scale_to(self, size: int) -> list[Agent]:
        """Scale the pool to the specified number of workers.

        Returns newly created workers (if scaling up) or removed workers
        (if scaling down). Scaling down returns stopped workers — the caller
        must await their stop() if they are running.

            pool.scale_to(10)   # Handle traffic spike
            pool.scale_to(2)    # Scale back down
        """
        if size < 1:
            raise ValueError("Pool size must be at least 1")

        if size == len(self._workers):
            return []

        if size > len(self._workers):
            # Scale up
            new_workers: list[Agent] = []
            for _ in range(size - len(self._workers)):
                worker = self._create_worker()
                self._workers.append(worker)
                new_workers.append(worker)
            logger.info(
                "Pool '%s' scaled up to %d workers (+%d)",
                self.name, len(self._workers), len(new_workers),
            )
            return new_workers

        # Scale down — remove from the end
        removed: list[Agent] = []
        while len(self._workers) > size:
            worker = self._workers.pop()
            removed.append(worker)
        logger.info(
            "Pool '%s' scaled down to %d workers (-%d)",
            self.name, len(self._workers), len(removed),
        )
        return removed

    async def scale_up(self, count: int = 1) -> list[Agent]:
        """Add workers to the pool. Returns the new workers."""
        return self.scale_to(len(self._workers) + count)

    async def scale_down(self, count: int = 1) -> list[Agent]:
        """Remove workers from the pool. Stops them gracefully. Returns removed workers."""
        removed = self.scale_to(max(1, len(self._workers) - count))
        await asyncio.gather(*(w.stop() for w in removed if w.running))
        return removed

    # --- Routing ---

    def select_worker(self, signal: Signal | None = None) -> Agent:
        """Select a worker based on the pool's distribution strategy.

        round_robin: Distributes evenly regardless of load.
        least_loaded: Routes to the worker with the fewest pending signals.
        """
        if self._strategy == "least_loaded":
            return min(self._workers, key=lambda w: w.inbox.pending)
        # round_robin (default)
        return self._workers[next(self._robin_counter) % len(self._workers)]

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start all workers in the pool."""
        await asyncio.gather(*(w.start() for w in self._workers))
        logger.info("Pool '%s' started with %d workers", self.name, len(self._workers))

    async def stop(self) -> None:
        """Stop all workers gracefully."""
        await asyncio.gather(*(w.stop() for w in self._workers))
        logger.info("Pool '%s' stopped", self.name)

    # --- Observability ---

    def health(self) -> dict[str, Any]:
        """Aggregate health across all pool workers."""
        worker_health = {w.name: w.health() for w in self._workers}
        all_healthy = all(h["healthy"] for h in worker_health.values())
        total_pending = sum(w.inbox.pending for w in self._workers)
        return {
            "pool": self.name,
            "healthy": all_healthy,
            "size": len(self._workers),
            "strategy": self._strategy,
            "total_pending": total_pending,
            "workers": worker_health,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Aggregate stats across all pool workers."""
        worker_stats = [w.stats for w in self._workers]
        return {
            "pool": self.name,
            "size": len(self._workers),
            "strategy": self._strategy,
            "total_processed": sum(s["processed"] for s in worker_stats),
            "total_rejected": sum(s["rejected"] for s in worker_stats),
            "total_errors": sum(s["errors"] for s in worker_stats),
            "total_pending": sum(s["pending"] for s in worker_stats),
            "total_dead_letters": sum(s["dead_letters"] for s in worker_stats),
            "workers": worker_stats,
        }

    def __repr__(self) -> str:
        return (
            f"AgentPool({self.name!r}, size={len(self._workers)}, "
            f"strategy={self._strategy!r})"
        )
