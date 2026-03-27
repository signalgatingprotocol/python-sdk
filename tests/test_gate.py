"""Tests for Gate composability and factory methods."""

import asyncio

import pytest

from signal_gating import Gate, Signal


class PrioritySignal(Signal):
    label: str = ""


@pytest.fixture
def signal():
    return PrioritySignal(label="test", priority=5)


@pytest.fixture
def low_signal():
    return PrioritySignal(label="low", priority=1)


async def test_filter_pass(signal):
    gate = Gate.filter(lambda s: s.priority > 3)
    result = await gate.process(signal)
    assert result is not None
    assert result.priority == 5


async def test_filter_reject(low_signal):
    gate = Gate.filter(lambda s: s.priority > 3)
    result = await gate.process(low_signal)
    assert result is None


async def test_transform(signal):
    gate = Gate.transform(lambda s: s.evolve(priority=s.priority * 2))
    result = await gate.process(signal)
    assert result is not None
    assert result.priority == 10


async def test_chain_operator(signal):
    double = Gate.transform(lambda s: s.evolve(priority=s.priority * 2))
    add_one = Gate.transform(lambda s: s.evolve(priority=s.priority + 1))
    chained = double >> add_one
    result = await chained.process(signal)
    assert result is not None
    assert result.priority == 11  # (5 * 2) + 1


async def test_chain_short_circuits(low_signal):
    reject = Gate.filter(lambda s: s.priority > 3)
    transform = Gate.transform(lambda s: s.evolve(priority=99))
    chained = reject >> transform
    result = await chained.process(low_signal)
    assert result is None


async def test_or_operator(signal, low_signal):
    high = Gate.filter(lambda s: s.priority > 10)
    medium = Gate.filter(lambda s: s.priority > 3)
    either = high | medium
    assert await either.process(signal) is not None
    assert await either.process(low_signal) is None


async def test_and_operator(signal):
    check_priority = Gate.filter(lambda s: s.priority > 3)
    check_label = Gate.filter(lambda s: s.label == "test")
    both = check_priority & check_label
    result = await both.process(signal)
    assert result is not None


async def test_and_operator_rejects(signal):
    check_priority = Gate.filter(lambda s: s.priority > 3)
    check_label = Gate.filter(lambda s: s.label == "wrong")
    both = check_priority & check_label
    result = await both.process(signal)
    assert result is None


async def test_invert_operator(signal, low_signal):
    high = Gate.filter(lambda s: s.priority > 3)
    not_high = ~high
    assert await not_high.process(signal) is None
    assert await not_high.process(low_signal) is not None


async def test_passthrough(signal):
    gate = Gate.passthrough()
    result = await gate.process(signal)
    assert result is signal


async def test_block(signal):
    gate = Gate.block()
    result = await gate.process(signal)
    assert result is None


async def test_by_type():
    class AlphaSignal(Signal):
        pass

    class BetaSignal(Signal):
        pass

    gate = Gate.by_type(AlphaSignal)
    assert await gate.process(AlphaSignal()) is not None
    assert await gate.process(BetaSignal()) is None


async def test_by_priority(signal, low_signal):
    gate = Gate.by_priority(min_priority=3)
    assert await gate.process(signal) is not None
    assert await gate.process(low_signal) is None


async def test_deduplicate():
    gate = Gate.deduplicate(window=60)
    s1 = Signal(priority=1)
    s2 = Signal(priority=1)  # Different id but same content hash
    # First should pass
    assert await gate.process(s1) is not None
    # The dedup uses model_dump excluding id/timestamp, so same priority = duplicate
    assert await gate.process(s2) is None


async def test_async_gate(signal):
    async def async_check(s: Signal) -> Signal | None:
        return s if s.priority > 0 else None

    gate = Gate(async_check, name="async_check")
    result = await gate.process(signal)
    assert result is not None


async def test_complex_composition(signal):
    pipeline = (
        Gate.by_priority(3)
        >> Gate.transform(lambda s: s.evolve(priority=s.priority + 10))
        >> Gate.filter(lambda s: s.priority > 10)
    )
    result = await pipeline.process(signal)
    assert result is not None
    assert result.priority == 15


# --- New: Retry Gate ---


async def test_retry_gate_passes_on_success():
    inner = Gate.passthrough()
    gate = Gate.retry(inner, max_attempts=3)
    s = Signal(priority=5)
    result = await gate.process(s)
    assert result is not None
    assert result.priority == 5


async def test_retry_gate_retries_on_rejection():
    call_count = 0

    async def flaky(signal: Signal) -> Signal | None:
        nonlocal call_count
        call_count += 1
        return signal if call_count >= 3 else None

    inner = Gate(flaky, name="flaky")
    gate = Gate.retry(inner, max_attempts=3, delay=0.01, backoff=1.0)
    result = await gate.process(Signal())
    assert result is not None
    assert call_count == 3


async def test_retry_gate_exhausts_attempts():
    inner = Gate.block()
    gate = Gate.retry(inner, max_attempts=2, delay=0.01)
    result = await gate.process(Signal())
    assert result is None


# --- New: Circuit Breaker Gate ---


async def test_circuit_breaker_closes_after_threshold():
    inner = Gate.block()  # always rejects
    gate = Gate.circuit_breaker(
        inner, failure_threshold=3, recovery_timeout=0.1
    )

    # First 3 attempts go through the inner gate (all rejected)
    for _ in range(3):
        result = await gate.process(Signal())
        assert result is None

    # Now circuit is open — rejects without calling inner
    result = await gate.process(Signal())
    assert result is None


async def test_circuit_breaker_recovers():
    call_count = 0

    async def recovering(signal: Signal) -> Signal | None:
        nonlocal call_count
        call_count += 1
        # Start failing, then succeed
        return signal if call_count > 3 else None

    inner = Gate(recovering, name="recovering")
    gate = Gate.circuit_breaker(
        inner, failure_threshold=3, recovery_timeout=0.05
    )

    # Trip the breaker
    for _ in range(3):
        await gate.process(Signal())

    # Wait for recovery timeout
    await asyncio.sleep(0.06)

    # Should enter half-open and succeed (call_count will be 4)
    result = await gate.process(Signal())
    assert result is not None


# --- New: Timeout Gate ---


async def test_timeout_gate_passes_fast_gate():
    inner = Gate.passthrough()
    gate = Gate.timeout(inner, seconds=1.0)
    result = await gate.process(Signal())
    assert result is not None


async def test_timeout_gate_rejects_slow_gate():
    async def slow(signal: Signal) -> Signal | None:
        await asyncio.sleep(10)
        return signal

    inner = Gate(slow, name="slow")
    gate = Gate.timeout(inner, seconds=0.01)
    result = await gate.process(Signal())
    assert result is None
