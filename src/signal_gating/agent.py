"""Agents — autonomous signal processors that form the backbone of the protocol."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Coroutine
from inspect import isawaitable
from typing import Any, TypeVar, get_type_hints
from uuid import uuid4

from signal_gating.channel import Channel, PriorityChannel
from signal_gating.errors import AgentError
from signal_gating.gate import Gate
from signal_gating.signal import Signal

T = TypeVar("T", bound=Signal)
Handler = Callable[..., Coroutine[Any, Any, Any]]
NextFn = Callable[..., Coroutine[Any, Any, Signal | None]]
Middleware = Callable[[Signal, NextFn], Coroutine[Any, Any, Signal | None]]
LifecycleHook = Callable[[], Any]

logger = logging.getLogger("signal_gating.agent")


class AgentContext:
    """Context passed to signal handlers, providing agent capabilities without closures.

    Instead of closing over the agent variable:

        @worker.on(TaskSignal)
        async def handle(signal: TaskSignal):
            await worker.emit(ResultSignal(...))  # requires closure

    Use the context pattern:

        @worker.on(TaskSignal)
        async def handle(signal: TaskSignal, ctx: AgentContext):
            await ctx.emit(ResultSignal(...))     # no closure needed
            ctx.state["count"] = ctx.state.get("count", 0) + 1
    """

    __slots__ = ("_agent", "signal")

    def __init__(self, agent: Agent, signal: Signal) -> None:
        self._agent = agent
        self.signal = signal

    @property
    def agent_name(self) -> str:
        return self._agent.name

    @property
    def state(self) -> dict[str, Any]:
        return self._agent.state

    async def emit(self, signal: Signal) -> None:
        """Emit a signal to all connected downstream agents."""
        await self._agent.emit(signal)

    async def reply(self, response: Signal) -> None:
        """Reply to the current signal with a correlated response."""
        await self._agent.reply(self.signal, response)

    async def request(self, signal: Signal, timeout: float = 30.0) -> Signal:
        """Emit a signal and wait for a correlated response."""
        return await self._agent.request(signal, timeout=timeout)


class DeadLetterQueue:
    """Collects signals that failed processing or were rejected by gates.

    Every production agent system needs to know what went wrong and where.
    The DLQ captures failed signals with context for debugging, replay, or alerting.
    """

    def __init__(self, max_size: int = 10000):
        self._entries: list[dict[str, Any]] = []
        self._signals: list[Signal] = []
        self._max_size = max_size

    def add(
        self,
        signal: Signal,
        reason: str,
        agent: str = "",
        error: Exception | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "signal_id": signal.id,
            "signal_type": type(signal).__name__,
            "trace_id": signal.trace_id,
            "agent": agent,
            "reason": reason,
            "timestamp": time.time(),
        }
        if error is not None:
            entry["error"] = f"{type(error).__name__}: {error}"
        self._entries.append(entry)
        self._signals.append(signal)
        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size :]
            self._signals = self._signals[-self._max_size :]

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    @property
    def signals(self) -> list[Signal]:
        """Access the original signal objects for replay or inspection."""
        return list(self._signals)

    @property
    def count(self) -> int:
        return len(self._entries)

    def drain(self) -> list[Signal]:
        """Remove and return all signals for replay. Clears the DLQ."""
        signals = list(self._signals)
        self._entries.clear()
        self._signals.clear()
        return signals

    async def replay(self, channel: Channel[Signal] | PriorityChannel[Signal]) -> int:
        """Replay all dead-lettered signals back into a channel.

        This is the agent-native recovery pattern: when signals fail,
        fix the issue, then replay them without losing any work.

        Returns the number of signals replayed.
        """
        signals = self.drain()
        for signal in signals:
            await channel.send(signal)
        return len(signals)

    def clear(self) -> None:
        self._entries.clear()
        self._signals.clear()


class Agent:
    """An autonomous entity that processes signals through gates.

    Agents are the primary actors in the Signal Gating Protocol. They:
    - Receive signals through an inbox channel
    - Apply gates to filter/transform incoming signals
    - Run middleware pipeline on each signal
    - Dispatch signals to registered handlers
    - Emit new signals to connected agents
    - Track state across processing cycles
    - Auto-restart on failure (supervision)
    - Route failed signals to dead letter queue

    Usage:
        agent = Agent("worker")

        @agent.on(TaskSignal)
        async def handle_task(signal: TaskSignal) -> None:
            result = await do_work(signal.task)
            await agent.emit(ResultSignal(result=result))

        async with mesh:
            await agent.emit(TaskSignal(task="build"))
    """

    def __init__(
        self,
        name: str,
        gates: list[Gate] | None = None,
        buffer_size: int = 1000,
        max_restarts: int = 3,
        restart_delay: float = 1.0,
        priority_inbox: bool = False,
    ):
        self.name = name
        self.gates = gates or []
        self._buffer_size = buffer_size
        self._priority_inbox = priority_inbox
        self.inbox: Channel[Signal] | PriorityChannel[Signal] = self._make_inbox()
        self._handlers: dict[type[Signal], list[Handler]] = {}
        self._handler_context_cache: dict[int, bool] = {}
        self._middleware: list[Middleware] = []
        self._outbox: list[Callable[[Signal], Coroutine[Any, Any, None]]] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._processed_count = 0
        self._rejected_count = 0
        self._error_count = 0
        self._restart_count = 0

        # Agent state — persistent memory across signal processing cycles
        self.state: dict[str, Any] = {}

        # Supervision
        self._max_restarts = max_restarts
        self._restart_delay = restart_delay

        # Dead letter queue
        self.dead_letters = DeadLetterQueue()

        # Tracer (set by mesh or manually)
        self._tracer: Any = None

        # Lifecycle hooks
        self._on_start_hooks: list[LifecycleHook] = []
        self._on_stop_hooks: list[LifecycleHook] = []

        # Request/response pending futures
        self._pending_requests: dict[str, asyncio.Future[Signal]] = {}

    def _make_inbox(self) -> Channel[Signal] | PriorityChannel[Signal]:
        """Create a fresh inbox channel."""
        if self._priority_inbox:
            return PriorityChannel(Signal, buffer_size=self._buffer_size)
        return Channel(Signal, buffer_size=self._buffer_size)

    def set_tracer(self, tracer: Any) -> None:
        """Attach a tracer for observability. Called by Mesh or manually."""
        self._tracer = tracer

    @property
    def running(self) -> bool:
        return self._running

    @property
    def healthy(self) -> bool:
        """Quick health check: is the agent running and not over-erroring?"""
        if not self._running:
            return False
        if self._restart_count > self._max_restarts:
            return False
        return True

    def health(self) -> dict[str, Any]:
        """Detailed health status for monitoring and readiness probes."""
        return {
            "name": self.name,
            "healthy": self.healthy,
            "running": self._running,
            "restarts": self._restart_count,
            "max_restarts": self._max_restarts,
            "error_count": self._error_count,
            "dead_letters": self.dead_letters.count,
            "inbox_depth": self.inbox.pending,
            "error_rate": (
                self._error_count / self._processed_count
                if self._processed_count > 0
                else 0.0
            ),
        }

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "running": self._running,
            "processed": self._processed_count,
            "rejected": self._rejected_count,
            "errors": self._error_count,
            "restarts": self._restart_count,
            "pending": self.inbox.pending,
            "dead_letters": self.dead_letters.count,
            "handlers": {t.__name__: len(h) for t, h in self._handlers.items()},
        }

    def on(self, signal_type: type[T]) -> Callable[[Handler], Handler]:
        """Register a handler for a signal type.

            @agent.on(TaskSignal)
            async def handle(signal: TaskSignal) -> None:
                ...
        """

        def decorator(fn: Handler) -> Handler:
            self._handlers.setdefault(signal_type, []).append(fn)
            return fn

        return decorator

    def on_any(self, fn: Handler) -> Handler:
        """Register a handler for all signal types."""
        self._handlers.setdefault(Signal, []).append(fn)
        return fn

    def once(self, signal_type: type[T]) -> Callable[[Handler], Handler]:
        """Register a handler that fires only once, then auto-removes itself.

            @agent.once(StartupSignal)
            async def handle(signal: StartupSignal):
                print("First signal received — won't fire again")
        """

        def decorator(fn: Handler) -> Handler:
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                handlers = self._handlers.get(signal_type, [])
                if wrapper in handlers:
                    handlers.remove(wrapper)
                return result

            self._handlers.setdefault(signal_type, []).append(wrapper)
            return fn

        return decorator

    def on_start(self, fn: LifecycleHook) -> LifecycleHook:
        """Register a hook called when the agent starts.

            @agent.on_start
            async def setup():
                agent.state["db"] = await connect_db()
        """
        self._on_start_hooks.append(fn)
        return fn

    def on_stop(self, fn: LifecycleHook) -> LifecycleHook:
        """Register a hook called when the agent stops.

            @agent.on_stop
            async def cleanup():
                await agent.state["db"].close()
        """
        self._on_stop_hooks.append(fn)
        return fn

    async def request(self, signal: Signal, timeout: float = 30.0) -> Signal:
        """Emit a signal and wait for a correlated response.

        This is the agent request/response pattern. The signal is tagged with a
        correlation ID. When a response with a matching correlation ID arrives
        in this agent's inbox, the future resolves.

        Requires a return path in the mesh (bidirectional connection).

            response = await planner.request(
                TaskSignal(task="analyze data"), timeout=5.0
            )
        """
        cid = uuid4().hex
        request_signal = signal.evolve(correlation_id=cid)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Signal] = loop.create_future()
        self._pending_requests[cid] = future
        try:
            await self.emit(request_signal)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_requests.pop(cid, None)

    async def reply(self, original: Signal, response: Signal) -> None:
        """Reply to a request signal with a correlated response.

        The response signal inherits the correlation ID from the original,
        enabling the requesting agent to match it.

            @worker.on(TaskSignal)
            async def handle(signal: TaskSignal):
                result = await process(signal.task)
                await worker.reply(signal, ResultSignal(result=result))
        """
        if original.correlation_id:
            reply_signal = response.evolve(correlation_id=original.correlation_id)
            await self.emit(reply_signal)
        else:
            await self.emit(response)

    def use(self, middleware: Middleware) -> None:
        """Add middleware to the processing pipeline.

        Middleware wraps signal dispatch, enabling cross-cutting concerns:

            async def logging_middleware(signal, next_fn):
                print(f"Processing: {signal}")
                result = await next_fn(signal)
                print(f"Done: {signal}")
                return result

            agent.use(logging_middleware)
        """
        self._middleware.append(middleware)

    async def emit(self, signal: Signal) -> None:
        """Emit a signal to all connected downstream agents."""
        tagged = signal.with_source(self.name) if not signal.source else signal
        for send_fn in self._outbox:
            await send_fn(tagged)

    async def emit_many(self, signals: list[Signal]) -> None:
        """Emit multiple signals concurrently.

        Uses asyncio.gather for true parallel emission — all signals are sent
        to all downstream agents simultaneously instead of sequentially.
        Essential for high-throughput agent patterns like fan-out and map operations.
        """
        if signals:
            await asyncio.gather(*(self.emit(signal) for signal in signals))

    async def _apply_gates(self, signal: Signal) -> Signal | None:
        """Apply all gates sequentially. Returns None if any gate rejects."""
        current: Signal | None = signal
        for gate in self.gates:
            if current is None:
                return None
            start = time.monotonic()
            current = await gate.process(current)
            elapsed_ms = (time.monotonic() - start) * 1000

            if self._tracer is not None:
                action = "passed" if current is not None else "rejected"
                self._tracer.record(
                    trace_id=signal.trace_id,
                    signal_id=signal.id,
                    agent=self.name,
                    gate=gate.name,
                    action=action,
                    duration_ms=elapsed_ms,
                )
        return current

    async def _dispatch(self, signal: Signal) -> None:
        """Dispatch a signal to matching handlers, with middleware."""
        dispatched = False
        for signal_type, handlers in self._handlers.items():
            if isinstance(signal, signal_type):
                for handler in handlers:
                    await self._run_handler_with_middleware(handler, signal)
                dispatched = True

        if not dispatched:
            logger.debug(
                "Agent '%s': no handler for %s", self.name, type(signal).__name__
            )

    def _handler_wants_context(self, handler: Handler) -> bool:
        """Check if a handler's signature includes an AgentContext parameter.

        Results are cached per handler identity for performance — this avoids
        expensive introspection on every signal dispatch.
        """
        handler_id = id(handler)
        cached = self._handler_context_cache.get(handler_id)
        if cached is not None:
            return cached

        result = self._inspect_handler_context(handler)
        self._handler_context_cache[handler_id] = result
        return result

    @staticmethod
    def _inspect_handler_context(handler: Handler) -> bool:
        """Introspect a handler's signature for AgentContext parameter."""
        try:
            fn = getattr(handler, "__wrapped__", handler)
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            if len(params) < 2:
                return False
            try:
                hints = get_type_hints(fn)
            except Exception:
                hints = getattr(fn, "__annotations__", {})
            for param in params[1:]:
                ann = hints.get(param.name)
                if ann is AgentContext:
                    return True
                if isinstance(ann, str) and ann == "AgentContext":
                    return True
            return False
        except (ValueError, TypeError):
            return False

    async def _run_handler_with_middleware(
        self, handler: Handler, signal: Signal
    ) -> None:
        """Execute a single handler wrapped in the middleware chain."""

        async def call_handler(sig: Signal) -> Signal | None:
            if self._handler_wants_context(handler):
                ctx = AgentContext(self, sig)
                await handler(sig, ctx)
            else:
                await handler(sig)
            return sig

        chain: NextFn = call_handler
        for mw in reversed(self._middleware):
            outer_chain = chain

            async def make_chain(
                s: Signal, _mw: Middleware = mw, _next: NextFn = outer_chain
            ) -> Signal | None:
                return await _mw(s, _next)

            chain = make_chain

        await chain(signal)

    async def _run_loop(self) -> None:
        """Main processing loop."""
        self._running = True
        try:
            async for signal in self.inbox:
                start = time.monotonic()
                gated = await self._apply_gates(signal)
                if gated is None:
                    self._rejected_count += 1
                    self.dead_letters.add(signal, "gate_rejected", self.name)
                    continue

                # Request/response: resolve pending futures for correlated responses
                if gated.correlation_id:
                    future = self._pending_requests.pop(gated.correlation_id, None)
                    if future is not None and not future.done():
                        future.set_result(gated)
                        self._processed_count += 1
                        continue

                self._processed_count += 1
                try:
                    await self._dispatch(gated)
                except Exception as e:
                    self._error_count += 1
                    self.dead_letters.add(gated, "handler_error", self.name, e)
                    logger.error(
                        "Handler error in agent '%s': %s", self.name, e, exc_info=True
                    )
                    continue

                if self._tracer is not None:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    self._tracer.record(
                        trace_id=signal.trace_id,
                        signal_id=signal.id,
                        agent=self.name,
                        gate="dispatch",
                        action="processed",
                        duration_ms=elapsed_ms,
                    )
        except Exception as e:
            if self._running:
                logger.error("Agent '%s' loop error: %s", self.name, e, exc_info=True)
        finally:
            self._running = False

    async def start(self) -> None:
        """Start the agent's processing loop with supervision.

        Can be called after stop() — a fresh inbox is created automatically
        if the previous one was closed, enabling agent restart patterns.
        """
        if self._task is not None:
            return
        # Recreate inbox if it was closed (enables restart after stop)
        if self.inbox.closed:
            self.inbox = self._make_inbox()
        for hook in self._on_start_hooks:
            try:
                result = hook()
                if isawaitable(result):
                    await result
            except Exception as e:
                logger.error(
                    "Agent '%s' on_start hook failed: %s", self.name, e, exc_info=True
                )
                raise AgentError(self.name, f"on_start hook failed: {e}") from e
        self._task = asyncio.create_task(
            self._supervised_loop(), name=f"agent:{self.name}"
        )

    async def _supervised_loop(self) -> None:
        """Run the processing loop with automatic restart on failure.

        Uses exponential backoff: each restart waits longer than the last,
        preventing thundering-herd restarts in production systems.
        """
        current_delay = self._restart_delay
        while self._restart_count <= self._max_restarts:
            try:
                await self._run_loop()
                return  # Clean exit (channel closed)
            except Exception as e:
                self._restart_count += 1
                if self._restart_count > self._max_restarts:
                    logger.error(
                        "Agent '%s' exceeded max restarts (%d). Shutting down.",
                        self.name,
                        self._max_restarts,
                    )
                    return
                logger.warning(
                    "Agent '%s' crashed (%s), restarting (%d/%d) in %.1fs...",
                    self.name,
                    e,
                    self._restart_count,
                    self._max_restarts,
                    current_delay,
                )
                await asyncio.sleep(current_delay)
                current_delay *= 2  # Exponential backoff

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
        self.inbox.close()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        for hook in self._on_stop_hooks:
            try:
                result = hook()
                if isawaitable(result):
                    await result
            except Exception as e:
                logger.error(
                    "Agent '%s' on_stop hook failed: %s", self.name, e, exc_info=True
                )

    def _add_output(self, send_fn: Callable[[Signal], Coroutine[Any, Any, None]]) -> None:
        """Internal: register an output destination."""
        self._outbox.append(send_fn)

    def _remove_outputs(
        self, *, target: str | None = None, tag: str | None = None
    ) -> int:
        """Internal: remove output destinations by target name or tag.

        Returns the number of removed entries.
        """
        before = len(self._outbox)
        self._outbox = [
            fn for fn in self._outbox
            if not (
                (target is not None and getattr(fn, "_mesh_target", None) == target)
                or (tag is not None and getattr(fn, "_mesh_tag", None) == tag)
            )
        ]
        return before - len(self._outbox)

    def __repr__(self) -> str:
        n_handlers = sum(len(h) for h in self._handlers.values())
        return (
            f"Agent({self.name!r}, gates={len(self.gates)}, handlers={n_handlers})"
        )
