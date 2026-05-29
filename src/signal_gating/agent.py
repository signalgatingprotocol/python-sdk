"""Agents: autonomous signal processors that form the backbone of the protocol."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from inspect import isawaitable
from pathlib import Path
from typing import Any, TypeVar, get_type_hints
from uuid import uuid4

from signal_gating.channel import Channel, PriorityChannel
from signal_gating.errors import AgentError
from signal_gating.gate import Gate
from signal_gating.signal import Signal
from signal_gating.tracing import Tracer

T = TypeVar("T", bound=Signal)
Handler = Callable[..., Coroutine[Any, Any, Any]]
NextFn = Callable[..., Coroutine[Any, Any, Signal | None]]
Middleware = Callable[[Signal, NextFn], Coroutine[Any, Any, Signal | None]]
LifecycleHook = Callable[[], Any]
ErrorHook = Callable[[Signal, Exception], Any]
ToolFn = Callable[..., Any]


# --- Tool Protocol Signals ---


class ToolCallSignal(Signal):
    """Signal requesting invocation of a named tool on a target agent.

    This is the wire format for agent-to-agent tool calling. An agent emits
    this signal to invoke a tool on another agent. The target agent processes
    it, executes the tool, and replies with a ToolResultSignal.

    Used automatically by ``Agent.call_tool()`` and ``Mesh.call_tool()``.
    """

    tool_name: str
    arguments: dict[str, Any] = {}


class ToolResultSignal(Signal):
    """Signal carrying the result of a tool invocation.

    Emitted by agents in response to a ToolCallSignal. Carries the tool's
    return value (or error) back to the caller.
    """

    tool_name: str
    result: Any = None
    error: str = ""


@dataclass(slots=True)
class ToolSpec:
    """Specification for a tool exposed by an agent.

    When an agent registers a tool, it becomes discoverable and callable by
    other agents through the mesh. Bridges signal-based communication and
    structured function calling for LLM-based agents.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    fn: ToolFn | None = field(default=None, repr=False)

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

    def to_jsonl(self, path: str | Path) -> int:
        """Persist the dead-lettered signals, with failure context, as JSON Lines.

        Each line is ``{"entry": <context>, "signal": <wire envelope>}``. Because
        signals are written in their wire form, ``load_jsonl`` reconstructs them
        as their original types. This is the durability half of recovery: persist
        on shutdown (or on a schedule), survive a crash or redeploy, then reload
        and ``replay``. Returns the number of records written.
        """
        out = Path(path)
        with out.open("w", encoding="utf-8") as f:
            for entry, signal in zip(self._entries, self._signals):
                record = {"entry": entry, "signal": signal.to_wire()}
                f.write(json.dumps(record, default=str) + "\n")
        return len(self._signals)

    def load_jsonl(self, path: str | Path, *, strict: bool = True) -> int:
        """Load dead-lettered signals from a JSONL file, appending to this queue.

        Reconstructs each signal as its original type via the registry, so the
        reloaded signals dispatch to the same handlers on ``replay``. Import the
        modules that define your signal types before loading so they are
        registered; with ``strict=True`` (default) an unknown type raises
        ``UnknownSignalType`` rather than silently degrading to a base ``Signal``.
        Honors ``max_size``, keeping the most recent entries. Returns the count
        loaded.
        """
        src = Path(path)
        loaded = 0
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                signal = Signal.from_wire(record["signal"], strict=strict)
                self._entries.append(record.get("entry", {}))
                self._signals.append(signal)
                loaded += 1
        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size :]
            self._signals = self._signals[-self._max_size :]
        return loaded

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
        self._middleware: list[Middleware] = []
        self._outbox: list[Callable[[Signal], Coroutine[Any, Any, None]]] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._processed_count = 0
        self._rejected_count = 0
        self._error_count = 0
        self._restart_count = 0

        # Agent state: persistent memory across signal processing cycles
        self.state: dict[str, Any] = {}

        # Supervision
        self._max_restarts = max_restarts
        self._restart_delay = restart_delay

        # Dead letter queue
        self.dead_letters = DeadLetterQueue()

        # Tracer (set by mesh or manually)
        self._tracer: Tracer | None = None

        # Lifecycle hooks
        self._on_start_hooks: list[LifecycleHook] = []
        self._on_stop_hooks: list[LifecycleHook] = []
        self._on_error_hooks: list[ErrorHook] = []

        # Request/response pending futures
        self._pending_requests: dict[str, asyncio.Future[Signal]] = {}

        # Tool registry: agent-native function calling
        self._tools: dict[str, ToolSpec] = {}

        # Per-handler cache of "does this handler want an AgentContext?".
        # Computed once at registration; avoids per-signal inspect calls.
        self._handler_wants_ctx: dict[Handler, bool] = {}

    def _make_inbox(self) -> Channel[Signal] | PriorityChannel[Signal]:
        """Create a fresh inbox channel."""
        if self._priority_inbox:
            return PriorityChannel(Signal, buffer_size=self._buffer_size)
        return Channel(Signal, buffer_size=self._buffer_size)

    def set_tracer(self, tracer: Tracer) -> None:
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
            self._register_handler(signal_type, fn)
            return fn

        return decorator

    def on_any(self, fn: Handler) -> Handler:
        """Register a handler for all signal types."""
        self._register_handler(Signal, fn)
        return fn

    def once(self, signal_type: type[T]) -> Callable[[Handler], Handler]:
        """Register a handler that fires only once, then auto-removes itself.

            @agent.once(StartupSignal)
            async def handle(signal: StartupSignal):
                print("First signal received. Won't fire again.")
        """

        def decorator(fn: Handler) -> Handler:
            wants_ctx = self._detect_wants_context(fn)

            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                handlers = self._handlers.get(signal_type, [])
                if wrapper in handlers:
                    handlers.remove(wrapper)
                self._handler_wants_ctx.pop(wrapper, None)
                return result

            wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
            self._handlers.setdefault(signal_type, []).append(wrapper)
            # Wrapper inherits the wrapped handler's context preference.
            self._handler_wants_ctx[wrapper] = wants_ctx
            return fn

        return decorator

    def _register_handler(self, signal_type: type[Signal], fn: Handler) -> None:
        """Internal: register a handler and cache its context preference."""
        self._handlers.setdefault(signal_type, []).append(fn)
        self._handler_wants_ctx[fn] = self._detect_wants_context(fn)

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

    def on_error(self, fn: ErrorHook) -> ErrorHook:
        """Register a hook called when signal processing fails.

        Error hooks receive the signal and exception, enabling custom error
        handling like alerting, retry logic, or signal rerouting. Hooks run
        AFTER the signal is added to the dead letter queue.

        Supports both sync and async hooks.

            @agent.on_error
            async def handle_error(signal: Signal, error: Exception):
                await alert_service.notify(f"Failed: {error}")
        """
        self._on_error_hooks.append(fn)
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
        # Snapshot the outbox: connect/disconnect can mutate it concurrently
        # (e.g. while we're awaiting a slow downstream send). Iterating the
        # live list would skip or double-deliver entries.
        for send_fn in tuple(self._outbox):
            await send_fn(tagged)

    async def emit_many(self, signals: list[Signal]) -> None:
        """Emit multiple signals concurrently.

        Uses asyncio.gather for true parallel emission. All signals are sent
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

    @staticmethod
    def _detect_wants_context(handler: Handler) -> bool:
        """Inspect a handler's signature once to see if it expects AgentContext."""
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

        wants_ctx = self._handler_wants_ctx.get(handler)
        if wants_ctx is None:
            # Defensive: handler registered through an internal path. Compute once.
            wants_ctx = self._detect_wants_context(handler)
            self._handler_wants_ctx[handler] = wants_ctx

        async def call_handler(sig: Signal) -> Signal | None:
            if wants_ctx:
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
                    for hook in self._on_error_hooks:
                        try:
                            result = hook(gated, e)
                            if isawaitable(result):
                                await result
                        except Exception as hook_err:
                            logger.error(
                                "Agent '%s' on_error hook failed: %s",
                                self.name, hook_err, exc_info=True,
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
                raise  # Propagate to _supervised_loop for restart
        finally:
            self._running = False

    async def start(self) -> None:
        """Start the agent's processing loop with supervision.

        Idempotent: returns immediately if the agent is already running.
        Restartable: if a previous run exited (cleanly or via max restarts),
        the task slot is cleared, the restart counter is reset, and a fresh
        inbox is created if the prior one was closed.
        """
        # Already running.
        if self._task is not None and not self._task.done():
            return
        # Previous run finished. Collect it before starting fresh so that
        # exceptions don't get silently buried in the task object.
        if self._task is not None and self._task.done():
            self._task = None
        self._restart_count = 0
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

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the agent gracefully.

        Closes the inbox so the run loop drains and exits. If the loop does
        not exit within ``timeout`` seconds, it is cancelled and we wait for
        cancellation to fully propagate before returning. The agent is left
        in a clean restartable state.
        """
        self._running = False
        self.inbox.close()
        task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            except asyncio.CancelledError:
                # Outer caller cancelled us. Propagate after best-effort cleanup.
                if not task.done():
                    task.cancel()
                raise
            except Exception:
                # Loop raised; we still consider the agent stopped.
                pass
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

    # --- Tool Registry ---

    def tool(
        self,
        name: str | None = None,
        description: str = "",
    ) -> Callable[[ToolFn], ToolFn]:
        """Register a function as a tool that other agents can discover and call.

        Tools let agents expose structured capabilities that other agents
        (or LLMs) can discover and invoke. When a tool is registered, the agent
        automatically handles ToolCallSignal for that tool name, executes the
        function, and replies with a ToolResultSignal.

        Supports both sync and async tool functions.

            @worker.tool(description="Analyze data and return insights")
            async def analyze(data: str, depth: int = 1) -> dict:
                return {"insights": await run_analysis(data, depth)}

            # Other agents can discover and call this tool:
            tools = worker.list_tools()
            result = await mesh.call_tool(worker, "analyze", data="revenue Q4")

        Args:
            name: Tool name (defaults to function name).
            description: Human-readable description of what the tool does.

        Returns:
            Decorator that registers the function as a tool.
        """
        agent = self

        def decorator(fn: ToolFn) -> ToolFn:
            tool_name = name or fn.__name__
            # Extract parameter schema from function signature
            sig = inspect.signature(fn)
            params: dict[str, Any] = {}
            try:
                hints = get_type_hints(fn)
            except Exception:
                hints = getattr(fn, "__annotations__", {})
            for pname, param in sig.parameters.items():
                param_info: dict[str, Any] = {}
                if pname in hints:
                    ann = hints[pname]
                    param_info["type"] = getattr(ann, "__name__", str(ann))
                if param.default is not inspect.Parameter.empty:
                    param_info["default"] = param.default
                param_info["required"] = param.default is inspect.Parameter.empty
                params[pname] = param_info

            spec = ToolSpec(
                name=tool_name,
                description=description or (fn.__doc__ or "").strip(),
                parameters=params,
                fn=fn,
            )
            agent._tools[tool_name] = spec

            # Auto-register handler for ToolCallSignal if not already present
            if ToolCallSignal not in agent._handlers:
                @agent.on(ToolCallSignal)
                async def _handle_tool_call(
                    signal: ToolCallSignal, ctx: AgentContext,
                ) -> None:
                    tool = agent._tools.get(signal.tool_name)
                    if tool is None or tool.fn is None:
                        await ctx.reply(ToolResultSignal(
                            tool_name=signal.tool_name,
                            error=f"Unknown tool: {signal.tool_name}",
                        ))
                        return
                    try:
                        result = tool.fn(**signal.arguments)
                        if isawaitable(result):
                            result = await result
                        await ctx.reply(ToolResultSignal(
                            tool_name=signal.tool_name,
                            result=result,
                        ))
                    except Exception as e:
                        await ctx.reply(ToolResultSignal(
                            tool_name=signal.tool_name,
                            error=f"{type(e).__name__}: {e}",
                        ))

            return fn

        return decorator

    def list_tools(self) -> list[ToolSpec]:
        """List all tools exposed by this agent.

        Returns tool specifications that can be used by LLMs for tool selection,
        by other agents for capability discovery, or by orchestrators for
        building dynamic workflows.

            tools = agent.list_tools()
            for tool in tools:
                print(f"{tool.name}: {tool.description}")
                print(f"  params: {tool.parameters}")
        """
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolSpec | None:
        """Get a specific tool spec by name."""
        return self._tools.get(name)

    def tools_schema(self) -> list[dict[str, Any]]:
        """Export all tools as a JSON-serializable schema.

        This is designed to be directly usable as the ``tools`` parameter
        in LLM API calls. Each tool's parameter types and descriptions
        are extracted from the function signature and docstring.

            # Feed directly to an LLM
            schema = agent.tools_schema()
            response = await llm.complete(messages, tools=schema)
        """
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in self._tools.values()
        ]

    def _add_output(self, send_fn: Callable[[Signal], Coroutine[Any, Any, None]]) -> None:
        """Internal: register an output destination."""
        self._outbox.append(send_fn)

    def _remove_outputs(
        self, *, target: str | None = None, tag: str | None = None
    ) -> int:
        """Internal: remove output destinations by target name or tag.

        Only RouteFn-wrapped outputs carry metadata; plain callables are kept.
        Returns the number of removed entries.
        """
        from signal_gating.mesh import RouteFn

        def should_remove(fn: Any) -> bool:
            if not isinstance(fn, RouteFn):
                return False
            if target is not None and fn.target == target:
                return True
            if tag is not None and fn.tag == tag:
                return True
            return False

        before = len(self._outbox)
        self._outbox = [fn for fn in self._outbox if not should_remove(fn)]
        return before - len(self._outbox)

    def __repr__(self) -> str:
        n_handlers = sum(len(h) for h in self._handlers.values())
        return (
            f"Agent({self.name!r}, gates={len(self.gates)}, handlers={n_handlers})"
        )
