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
from typing import Any

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
        self.name = name or (fn.__name__ if hasattr(fn, "__name__") else "gate")

    async def process(self, signal: Signal) -> Signal | None:
        """Process a signal through this gate. Returns None if rejected."""
        result = self._fn(signal)
        if isawaitable(result):
            result = await result
        return result

    def __rshift__(self, other: Gate) -> Gate:
        """Chain gates: signal flows through self, then other."""
        if not isinstance(other, Gate):
            return NotImplemented
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
    def transform(cls, fn: GateFn, name: str = "transform") -> Gate:
        """Create a gate that transforms passing signals.

        Accepts both sync and async transform functions:

            gate = Gate.transform(lambda s: s.evolve(priority=10))
            gate = Gate.transform(async_enrich)  # async works too
        """
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

    @classmethod
    def when(
        cls,
        condition: Callable[[Signal], bool],
        then: Gate,
        otherwise: Gate | None = None,
        name: str = "when",
    ) -> Gate:
        """Conditional branching gate: route signals through different paths based on a predicate.

        If no `otherwise` gate is provided, non-matching signals pass through unchanged.

            gate = Gate.when(
                lambda s: s.priority >= 8,
                then=Gate.transform(enrich_urgent),
                otherwise=Gate.transform(enrich_normal),
            )
        """

        async def fn(signal: Signal) -> Signal | None:
            if condition(signal):
                return await then.process(signal)
            if otherwise is not None:
                return await otherwise.process(signal)
            return signal

        return cls(fn, name=name)

    @classmethod
    def sample(cls, rate: float, name: str = "sample") -> Gate:
        """Probabilistic sampling gate — passes signals at the given rate (0.0-1.0).

        Essential for high-throughput systems where you want to observe
        or process only a fraction of signals:

            gate = Gate.sample(0.1)  # Process ~10% of signals
        """
        import random

        def fn(signal: Signal) -> Signal | None:
            return signal if random.random() < rate else None  # noqa: S311

        return cls(fn, name=name)

    @classmethod
    def throttle(cls, max_per_second: float, name: str = "throttle") -> Gate:
        """Throttle gate — drops signals that exceed the rate instead of sleeping.

        Unlike `rate_limit` which applies backpressure (sleeps), throttle
        silently drops excess signals. Use when dropping is preferable to
        queuing.

            gate = Gate.throttle(100)  # Allow max 100 signals/sec, drop rest
        """
        min_interval = 1.0 / max_per_second
        state: dict[str, float] = {"last": 0.0}
        lock = asyncio.Lock()

        async def fn(signal: Signal) -> Signal | None:
            async with lock:
                now = time.monotonic()
                if now - state["last"] < min_interval:
                    return None
                state["last"] = now
                return signal

        return cls(fn, name=name)

    @classmethod
    def ttl(cls, seconds: float, name: str = "ttl") -> Gate:
        """Time-to-live gate — drops signals older than the specified age.

        Prevents stale signals from consuming agent resources. Critical for
        systems where signal freshness determines value:

            gate = Gate.ttl(30)  # Drop signals older than 30 seconds
        """

        def fn(signal: Signal) -> Signal | None:
            age = time.time() - signal.timestamp
            return signal if age <= seconds else None

        return cls(fn, name=name)

    @classmethod
    def tap(
        cls,
        fn: Callable[[Signal], Any],
        name: str = "tap",
    ) -> Gate:
        """Tap gate — execute a side-effect without affecting signal flow.

        The signal always passes through unchanged. The tap function is called
        for observation only. Essential for logging, metrics, debugging, and
        auditing signal flow without modifying the processing pipeline.

        Supports both sync and async tap functions.

            gate = Gate.tap(lambda s: print(f"Saw: {s}"))
            gate = Gate.tap(lambda s: metrics.increment("signals", tags={"type": type(s).__name__}))

            # Async tap
            gate = Gate.tap(lambda s: audit_log.write(s))
        """

        async def tap_fn(signal: Signal) -> Signal:
            result = fn(signal)
            if isawaitable(result):
                await result
            return signal

        return cls(tap_fn, name=name)

    @classmethod
    def batch(cls, size: int, timeout: float = 0.0, name: str = "batch") -> Gate:
        """Accumulate signals into batches, releasing when size or timeout is reached.

        Signals accumulate silently (returning None) until the batch threshold
        is met. Then the last signal is returned with metadata containing the
        full batch.

        The returned signal carries:
            - metadata["batch"]: list of serialized signals in the batch
            - metadata["batch_size"]: number of signals in the batch

        With timeout > 0, the batch also flushes if ``timeout`` seconds have
        elapsed since the first signal in the current batch was received.
        The timeout is checked on each incoming signal (not via background task),
        so it prevents signals from sitting in the buffer indefinitely in
        low-throughput scenarios.

            gate = Gate.batch(10)             # flush every 10 signals
            gate = Gate.batch(10, timeout=5)  # flush every 10 signals or 5s
        """
        if size < 1:
            raise ValueError("batch size must be >= 1")
        state: dict[str, Any] = {"buffer": [], "first_at": 0.0}
        lock = asyncio.Lock()

        async def fn(signal: Signal) -> Signal | None:
            async with lock:
                now = time.monotonic()
                if not state["buffer"]:
                    state["first_at"] = now
                state["buffer"].append(signal)

                timed_out = timeout > 0 and (now - state["first_at"]) >= timeout
                if len(state["buffer"]) >= size or timed_out:
                    batch = list(state["buffer"])
                    state["buffer"] = []
                    state["first_at"] = 0.0
                    return signal.with_metadata(
                        batch=[s.model_dump() for s in batch],
                        batch_size=len(batch),
                    )
                return None  # Accumulating

        return cls(fn, name=name)

    @classmethod
    def parallel(
        cls,
        *gates: Gate,
        mode: str = "all",
        name: str = "parallel",
    ) -> Gate:
        """Run multiple gates concurrently on the same signal.

        Use when you need to check multiple conditions simultaneously instead
        of sequentially, or race gates against each other.

        Modes:
            ``all``: Signal must pass ALL gates. Returns the last result.
            ``any``: Signal passes if ANY gate accepts. Returns first non-None.
            ``race``: Whichever gate completes first wins. Others are cancelled.

            gate = Gate.parallel(auth_gate, rate_gate, schema_gate, mode="all")
            gate = Gate.parallel(cache_gate, compute_gate, mode="any")
            gate = Gate.parallel(local_gate, remote_gate, mode="race")
        """
        if not gates:
            raise ValueError("parallel requires at least one gate")
        if mode not in ("all", "any", "race"):
            raise ValueError(f"Unknown parallel mode: {mode!r}")
        gate_list = list(gates)

        async def fn(signal: Signal) -> Signal | None:
            if mode == "race":
                tasks = [asyncio.create_task(g.process(signal)) for g in gate_list]
                try:
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in pending:
                        t.cancel()
                    for t in done:
                        return t.result()
                    return None
                except Exception:
                    for t in tasks:
                        t.cancel()
                    raise

            results = await asyncio.gather(*(g.process(signal) for g in gate_list))

            if mode == "all":
                for r in results:
                    if r is None:
                        return None
                return results[-1]
            else:  # any
                for r in results:
                    if r is not None:
                        return r
                return None

        return cls(fn, name=name)

    @classmethod
    def fallback(
        cls,
        primary: Gate,
        *fallbacks: Gate,
        name: str = "fallback",
    ) -> Gate:
        """Try primary gate first, then fallbacks in order until one passes.

        Unlike ``|`` (OR) which evaluates both sides, fallback is lazy — it
        only tries the next gate if the previous one rejected. This matters
        when gates have side effects or are expensive.

            gate = Gate.fallback(
                cache_lookup,      # Try cache first
                database_lookup,   # Fall back to database
                default_value,     # Last resort
            )
        """
        all_gates = [primary, *fallbacks]

        async def fn(signal: Signal) -> Signal | None:
            for gate in all_gates:
                result = await gate.process(signal)
                if result is not None:
                    return result
            return None

        return cls(fn, name=name)

    @classmethod
    def debounce(cls, seconds: float, name: str = "debounce") -> Gate:
        """Debounce gate — only passes a signal after a quiet period.

        Waits `seconds` after receiving a signal. If another signal arrives
        during the wait, the timer resets. Only the last signal in a burst
        passes through. Essential for noisy signal sources.

            gate = Gate.debounce(0.5)  # Wait 500ms of silence before passing
        """
        state: dict[str, Any] = {"last_signal": None, "last_time": 0.0}
        lock = asyncio.Lock()

        async def fn(signal: Signal) -> Signal | None:
            async with lock:
                now = time.monotonic()
                state["last_signal"] = signal
                state["last_time"] = now

            # Wait for quiet period
            await asyncio.sleep(seconds)

            async with lock:
                # Only pass if no newer signal arrived
                if state["last_time"] <= now:
                    result: Signal | None = state["last_signal"]
                    return result
                return None

        return cls(fn, name=name)

    @classmethod
    def window(
        cls,
        seconds: float,
        min_signals: int = 1,
        name: str = "window",
    ) -> Gate:
        """Sliding time window gate — passes signals enriched with window context.

        Maintains a rolling window of the last ``seconds`` seconds. Each
        incoming signal is checked against the window: if at least
        ``min_signals`` exist within the window, the signal passes through
        enriched with rate and count metadata. Otherwise it is rejected.

        This is THE observability primitive for agent systems — detect
        bursts, anomalies, and patterns in real-time signal flow.

            # Detect bursts: only pass when 5+ signals arrive in 10 seconds
            gate = Gate.window(seconds=10, min_signals=5)

            # Enrich every signal with rate context
            gate = Gate.window(seconds=60, min_signals=1)
            # signal.metadata["window_size"]  → count of signals in last 60s
            # signal.metadata["window_rate"]  → signals per second
        """
        if seconds <= 0:
            raise ValueError("window seconds must be > 0")
        state: dict[str, list[tuple[float, str]]] = {"buffer": []}
        lock = asyncio.Lock()

        async def fn(signal: Signal) -> Signal | None:
            async with lock:
                now = time.monotonic()
                state["buffer"].append((now, signal.id))
                # Evict signals outside the window
                state["buffer"] = [
                    (t, sid) for t, sid in state["buffer"] if now - t <= seconds
                ]
                window_size = len(state["buffer"])
                if window_size >= min_signals:
                    return signal.with_metadata(
                        window_size=window_size,
                        window_rate=round(window_size / seconds, 4),
                    )
                return None

        return cls(fn, name=name)

    @classmethod
    def map(
        cls,
        fn: Callable[[Signal], Any],
        name: str = "map",
    ) -> Gate:
        """Async-first transformation gate that can also reject signals.

        Like ``transform()`` but semantically designed for async operations
        and transformations that may fail (return None). This is the primary
        gate for LLM-powered agent transformations where the transform
        involves I/O, API calls, or other async work.

            # Async LLM enrichment
            async def enrich(signal):
                result = await llm.analyze(signal.metadata["text"])
                return signal.with_metadata(analysis=result)

            gate = Gate.map(enrich)

            # Sync works too
            gate = Gate.map(lambda s: s.evolve(priority=s.priority + 1))

            # Return None to reject
            gate = Gate.map(lambda s: s if s.priority > 0 else None)
        """

        async def map_fn(signal: Signal) -> Signal | None:
            raw = fn(signal)
            if isawaitable(raw):
                raw = await raw
            result: Signal | None = raw
            return result

        return cls(map_fn, name=name)
