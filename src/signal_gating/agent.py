"""Agents — autonomous signal processors that form the backbone of the protocol."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from signal_gating.channel import Channel
from signal_gating.errors import AgentError
from signal_gating.gate import Gate
from signal_gating.signal import Signal

T = TypeVar("T", bound=Signal)
Handler = Callable[..., Coroutine[Any, Any, Any]]

logger = logging.getLogger("signal_gating.agent")


class Agent:
    """An autonomous entity that processes signals through gates.

    Agents are the primary actors in the Signal Gating Protocol. They:
    - Receive signals through an inbox channel
    - Apply gates to filter/transform incoming signals
    - Dispatch signals to registered handlers
    - Emit new signals to connected agents

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
    ):
        self.name = name
        self.gates = gates or []
        self.inbox: Channel[Signal] = Channel(Signal, buffer_size=buffer_size)
        self._handlers: dict[type[Signal], list[Handler]] = {}
        self._outbox: list[Callable[[Signal], Coroutine[Any, Any, None]]] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._processed_count = 0
        self._rejected_count = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "running": self._running,
            "processed": self._processed_count,
            "rejected": self._rejected_count,
            "pending": self.inbox.pending,
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

    async def emit(self, signal: Signal) -> None:
        """Emit a signal to all connected downstream agents."""
        tagged = signal.with_source(self.name) if not signal.source else signal
        for send_fn in self._outbox:
            await send_fn(tagged)

    async def _apply_gates(self, signal: Signal) -> Signal | None:
        """Apply all gates sequentially. Returns None if any gate rejects."""
        current: Signal | None = signal
        for gate in self.gates:
            if current is None:
                return None
            current = await gate.process(current)
        return current

    async def _dispatch(self, signal: Signal) -> None:
        """Dispatch a signal to matching handlers."""
        dispatched = False
        for signal_type, handlers in self._handlers.items():
            if isinstance(signal, signal_type):
                for handler in handlers:
                    try:
                        await handler(signal)
                    except Exception as e:
                        logger.error(f"Handler error in agent '{self.name}': {e}", exc_info=True)
                        raise AgentError(self.name, str(e)) from e
                dispatched = True

        if not dispatched:
            logger.debug(f"Agent '{self.name}': no handler for {type(signal).__name__}")

    async def _run_loop(self) -> None:
        """Main processing loop."""
        self._running = True
        try:
            async for signal in self.inbox:
                gated = await self._apply_gates(signal)
                if gated is None:
                    self._rejected_count += 1
                    continue
                self._processed_count += 1
                await self._dispatch(gated)
        except Exception as e:
            if self._running:
                logger.error(f"Agent '{self.name}' loop error: {e}", exc_info=True)
        finally:
            self._running = False

    async def start(self) -> None:
        """Start the agent's processing loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop(), name=f"agent:{self.name}")

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

    def _add_output(self, send_fn: Callable[[Signal], Coroutine[Any, Any, None]]) -> None:
        """Internal: register an output destination."""
        self._outbox.append(send_fn)

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, gates={len(self.gates)}, handlers={sum(len(h) for h in self._handlers.values())})"
