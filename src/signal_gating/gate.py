"""Gates — composable predicates that control signal flow.

Gates are the core innovation of the Signal Gating Protocol. They decide whether
a signal passes through, gets transformed, or gets rejected.

Compose gates with operators:
    gate1 >> gate2    # Chain: signal passes through gate1, then gate2
    gate1 | gate2     # Either: signal passes if either gate accepts
    gate1 & gate2     # Both: signal must pass both gates
    ~gate1            # Invert: passes only if gate1 rejects
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from inspect import isawaitable

from signal_gating.signal import Signal

GateFn = Callable[[Signal], Signal | None | Awaitable[Signal | None]]


class Gate:
    """A composable unit that controls signal flow.

    A gate receives a signal and either:
    - Returns the signal (possibly transformed) to pass it through
    - Returns None to reject it
    """

    def __init__(self, fn: GateFn, name: str = ""):
        self._fn = fn
        self.name = name or fn.__name__ if hasattr(fn, "__name__") else "gate"

    async def process(self, signal: Signal) -> Signal | None:
        """Process a signal through this gate. Returns None if rejected."""
        result = self._fn(signal)
        if isawaitable(result):
            result = await result
        return result

    def __rshift__(self, other: Gate) -> Gate:
        """Chain gates: signal flows through self, then other."""
        left, right = self, other

        async def chained(signal: Signal) -> Signal | None:
            result = await left.process(signal)
            if result is None:
                return None
            return await right.process(result)

        return Gate(chained, name=f"{self.name}>>{other.name}")

    def __or__(self, other: Gate) -> Gate:
        """Either gate: signal passes if either gate accepts."""
        left, right = self, other

        async def either(signal: Signal) -> Signal | None:
            result = await left.process(signal)
            if result is not None:
                return result
            return await right.process(signal)

        return Gate(either, name=f"({self.name}|{other.name})")

    def __and__(self, other: Gate) -> Gate:
        """Both gates: signal must pass both (uses result from second)."""
        left, right = self, other

        async def both(signal: Signal) -> Signal | None:
            r1 = await left.process(signal)
            if r1 is None:
                return None
            r2 = await right.process(signal)
            if r2 is None:
                return None
            return r2

        return Gate(both, name=f"({self.name}&{other.name})")

    def __invert__(self) -> Gate:
        """Invert gate: passes only if this gate rejects."""
        inner = self

        async def inverted(signal: Signal) -> Signal | None:
            result = await inner.process(signal)
            return signal if result is None else None

        return Gate(inverted, name=f"~{self.name}")

    def __repr__(self) -> str:
        return f"Gate({self.name!r})"

    # --- Factory Methods ---

    @classmethod
    def filter(cls, predicate: Callable[[Signal], bool], name: str = "filter") -> Gate:
        """Create a gate that passes signals matching a predicate."""

        def fn(signal: Signal) -> Signal | None:
            return signal if predicate(signal) else None

        return cls(fn, name=name)

    @classmethod
    def transform(cls, fn: Callable[[Signal], Signal], name: str = "transform") -> Gate:
        """Create a gate that transforms passing signals."""
        return cls(fn, name=name)

    @classmethod
    def rate_limit(cls, max_per_second: float, name: str = "rate_limit") -> Gate:
        """Create a gate that enforces a rate limit."""
        min_interval = 1.0 / max_per_second
        state: dict[str, float] = {"last": 0.0}
        lock = asyncio.Lock()

        async def fn(signal: Signal) -> Signal | None:
            async with lock:
                now = time.monotonic()
                elapsed = now - state["last"]
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
                state["last"] = time.monotonic()
                return signal

        return cls(fn, name=name)

    @classmethod
    def deduplicate(cls, window: float = 60.0, name: str = "dedup") -> Gate:
        """Create a gate that drops duplicate signals within a time window."""
        seen: dict[str, float] = {}
        lock = asyncio.Lock()

        async def fn(signal: Signal) -> Signal | None:
            async with lock:
                now = time.monotonic()
                # Evict expired entries
                expired = [k for k, t in seen.items() if now - t > window]
                for k in expired:
                    del seen[k]
                # Check for duplicate
                content = signal.model_dump_json(exclude={"id", "timestamp", "trace_id"})
                key = f"{type(signal).__name__}:{content}"
                if key in seen:
                    return None
                seen[key] = now
                return signal

        return cls(fn, name=name)

    @classmethod
    def by_type(cls, *signal_types: type[Signal], name: str = "type_filter") -> Gate:
        """Create a gate that only passes specific signal types."""

        def fn(signal: Signal) -> Signal | None:
            return signal if isinstance(signal, signal_types) else None

        return cls(fn, name=name)

    @classmethod
    def by_priority(cls, min_priority: int = 0, name: str = "priority_filter") -> Gate:
        """Create a gate that only passes signals above a priority threshold."""
        return cls.filter(lambda s: s.priority >= min_priority, name=name)

    @classmethod
    def retry(
        cls,
        gate: Gate,
        max_attempts: int = 3,
        delay: float = 0.1,
        backoff: float = 2.0,
        name: str = "retry",
    ) -> Gate:
        """Wrap a gate with retry logic and exponential backoff."""

        async def fn(signal: Signal) -> Signal | None:
            last_result: Signal | None = None
            current_delay = delay
            for attempt in range(max_attempts):
                last_result = await gate.process(signal)
                if last_result is not None:
                    return last_result
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            return last_result

        return cls(fn, name=name)

    @classmethod
    def circuit_breaker(
        cls,
        gate: Gate,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "circuit_breaker",
    ) -> Gate:
        """Wrap a gate with circuit breaker pattern.

        After `failure_threshold` consecutive rejections, the circuit opens
        and rejects all signals for `recovery_timeout` seconds. Then it
        enters half-open state and lets one signal through to test recovery.
        """
        state: dict[str, float | int | str] = {
            "failures": 0,
            "opened_at": 0.0,
            "status": "closed",  # closed | open | half_open
        }
        lock = asyncio.Lock()

        async def fn(signal: Signal) -> Signal | None:
            async with lock:
                now = time.monotonic()
                status = state["status"]

                if status == "open":
                    elapsed = now - float(state["opened_at"])
                    if elapsed < recovery_timeout:
                        return None
                    state["status"] = "half_open"

                result = await gate.process(signal)

                if result is not None:
                    state["failures"] = 0
                    state["status"] = "closed"
                    return result

                state["failures"] = int(state["failures"]) + 1
                if int(state["failures"]) >= failure_threshold:
                    state["status"] = "open"
                    state["opened_at"] = now
                return None

        return cls(fn, name=name)

    @classmethod
    def timeout(cls, gate: Gate, seconds: float, name: str = "timeout") -> Gate:
        """Wrap a gate with a timeout — rejects if processing takes too long."""

        async def fn(signal: Signal) -> Signal | None:
            try:
                return await asyncio.wait_for(gate.process(signal), timeout=seconds)
            except asyncio.TimeoutError:
                return None

        return cls(fn, name=name)

    @classmethod
    def passthrough(cls) -> Gate:
        """A gate that passes everything unchanged."""
        return cls(lambda s: s, name="passthrough")

    @classmethod
    def block(cls) -> Gate:
        """A gate that blocks everything."""
        return cls(lambda s: None, name="block")
