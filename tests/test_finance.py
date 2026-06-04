"""Tests for finance-oriented signals and gates."""

import pytest

from signal_gating import Gate, MarketDecision, MarketGate, MarketTick, Signal


def test_market_tick_normalizes_symbol_and_wire_type() -> None:
    tick = MarketTick(symbol=" aapl ", venue=" xnas ", bid=100.0, ask=100.02)

    assert tick.symbol == "AAPL"
    assert tick.venue == "XNAS"
    assert tick.wire_type() == "sgp.finance.market_tick.v1"

    restored = Signal.from_wire(tick.to_wire())
    assert isinstance(restored, MarketTick)
    assert restored == tick


def test_market_decision_validates_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        MarketDecision(symbol="AAPL", action="buy", confidence=1.2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_market_tick_rejects_non_finite_market_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        MarketTick(symbol="AAPL", bid=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_market_decision_rejects_non_finite_edge_and_notional(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        MarketDecision(symbol="AAPL", action="buy", expected_edge_bps=value)
    with pytest.raises(ValueError, match="finite"):
        MarketDecision(symbol="AAPL", action="buy", notional=value)


async def test_market_freshness_uses_event_time_not_signal_creation_time() -> None:
    now = 1_000.0
    fresh = MarketTick(symbol="AAPL", bid=100.0, ask=100.01, event_ts=now - 0.100)
    stale = MarketTick(symbol="AAPL", bid=100.0, ask=100.01, event_ts=now - 2.0)

    gate = MarketGate.freshness(max_age_ms=250, clock=lambda: now)

    fresh_result = await gate.process(fresh)
    assert fresh_result is not None
    assert fresh_result.metadata["market_age_ms"] == 100.0
    assert await gate.process(stale) is None


async def test_market_freshness_rejects_far_future_ticks() -> None:
    now = 1_000.0
    tick = MarketTick(symbol="AAPL", bid=100.0, ask=100.01, event_ts=now + 2.0)
    gate = MarketGate.freshness(max_age_ms=250, max_future_ms=500, clock=lambda: now)

    assert await gate.process(tick) is None


async def test_market_monotonic_sequence_is_per_symbol_and_venue() -> None:
    gate = MarketGate.monotonic_sequence()

    first = MarketTick(symbol="AAPL", venue="XNAS", bid=100.0, ask=100.01, sequence=10)
    next_tick = first.evolve(sequence=11)
    duplicate = first.evolve(sequence=10)
    other_symbol = MarketTick(
        symbol="MSFT",
        venue="XNAS",
        bid=200.0,
        ask=200.02,
        sequence=10,
    )

    assert await gate.process(first) is not None
    assert await gate.process(next_tick) is not None
    assert await gate.process(duplicate) is None
    assert await gate.process(other_symbol) is not None


async def test_market_quote_sanity_rejects_crossed_and_wide_quotes() -> None:
    gate = MarketGate.quote_sanity(max_spread_bps=10)
    tight = MarketTick(symbol="AAPL", bid=100.0, ask=100.05)
    crossed = MarketTick(symbol="AAPL", bid=100.05, ask=100.0)
    wide = MarketTick(symbol="AAPL", bid=100.0, ask=101.0)

    tight_result = await gate.process(tight)
    assert tight_result is not None
    assert tight_result.metadata["market_mid"] == 100.025
    assert tight_result.metadata["market_spread_bps"] < 10
    assert await gate.process(crossed) is None
    assert await gate.process(wide) is None


async def test_market_quote_sanity_can_reject_locked_quotes() -> None:
    locked = MarketTick(symbol="AAPL", bid=100.0, ask=100.0)

    assert await MarketGate.quote_sanity().process(locked) is not None
    assert await MarketGate.quote_sanity(allow_locked=False).process(locked) is None


async def test_market_gates_reject_non_finite_generic_signal_values() -> None:
    class RawQuote(Signal):
        bid: float
        ask: float

    class RawDecision(Signal):
        action: str
        notional: float
        expected_edge_bps: float = 1.0
        max_slippage_bps: float = 0.0
        confidence: float = 1.0

    quote = RawQuote(bid=float("nan"), ask=100.0)
    decision = RawDecision(action="buy", notional=float("inf"))

    with pytest.raises(ValueError, match="bid must be finite"):
        await MarketGate.quote_sanity().process(quote)
    with pytest.raises(ValueError, match="notional must be finite"):
        await MarketGate.notional_limit(1_000.0).process(decision)


async def test_market_decision_edge_enriches_net_edge_and_rejects_weak_alpha() -> None:
    gate = MarketGate.decision_edge(min_net_edge_bps=3.0, min_confidence=0.7)
    strong = MarketDecision(
        symbol="AAPL",
        action="buy",
        expected_edge_bps=8.0,
        max_slippage_bps=2.0,
        confidence=0.8,
        notional=25_000.0,
    )
    weak = strong.evolve(expected_edge_bps=4.0, max_slippage_bps=2.0)
    low_confidence = strong.evolve(confidence=0.5)

    strong_result = await gate.process(strong)
    assert strong_result is not None
    assert strong_result.metadata["market_net_edge_bps"] == 6.0
    assert await gate.process(weak) is None
    assert await gate.process(low_confidence) is None


async def test_market_decision_edge_passes_control_actions() -> None:
    gate = MarketGate.decision_edge(min_net_edge_bps=10.0, min_confidence=1.0)

    hold = MarketDecision(symbol="AAPL", action="hold", confidence=0.0)
    cancel = MarketDecision(symbol="AAPL", action="cancel", confidence=0.0)

    assert await gate.process(hold) is hold
    assert await gate.process(cancel) is cancel


async def test_market_notional_limit_rejects_budget_breaches() -> None:
    gate = MarketGate.notional_limit(50_000.0)

    small = MarketDecision(symbol="AAPL", action="buy", notional=49_999.0)
    large = MarketDecision(symbol="AAPL", action="buy", notional=50_001.0)

    small_result = await gate.process(small)
    assert small_result is not None
    assert small_result.metadata["market_notional_limit"] == 50_000.0
    assert await gate.process(large) is None


async def test_market_exposure_limit_tracks_cumulative_gross_exposure_by_key() -> None:
    gate = MarketGate.exposure_limit(50_000.0)

    first = MarketDecision(symbol="AAPL", action="buy", notional=25_000.0)
    second = MarketDecision(symbol="AAPL", action="buy", notional=25_000.0)
    breach = MarketDecision(symbol="AAPL", action="buy", notional=1.0)
    other_symbol = MarketDecision(symbol="MSFT", action="buy", notional=50_000.0)

    first_result = await gate.process(first)
    second_result = await gate.process(second)

    assert first_result is not None
    assert first_result.metadata["market_exposure_used"] == 25_000.0
    assert first_result.metadata["market_exposure_remaining"] == 25_000.0
    assert second_result is not None
    assert second_result.metadata["market_exposure_used"] == 50_000.0
    assert second_result.metadata["market_exposure_remaining"] == 0.0
    assert await gate.process(breach) is None
    assert await gate.process(other_symbol) is not None


async def test_market_exposure_limit_can_roll_off_a_window() -> None:
    now = 1_000.0
    gate = MarketGate.exposure_limit(
        50_000.0,
        window_seconds=1.0,
        clock=lambda: now,
    )

    first = MarketDecision(symbol="AAPL", action="buy", notional=40_000.0)
    second = MarketDecision(symbol="AAPL", action="buy", notional=20_000.0)

    assert await gate.process(first) is not None
    assert await gate.process(second) is None

    now = 1_001.1
    second_result = await gate.process(second)

    assert second_result is not None
    assert second_result.metadata["market_exposure_used"] == 20_000.0
    assert second_result.metadata["market_exposure_window_seconds"] == 1.0


async def test_market_exposure_limit_net_mode_offsets_sells() -> None:
    gate = MarketGate.exposure_limit(50_000.0, mode="net")

    buy = MarketDecision(symbol="AAPL", action="buy", notional=40_000.0)
    rejected_buy = MarketDecision(symbol="AAPL", action="buy", notional=20_000.0)
    sell = MarketDecision(symbol="AAPL", action="sell", notional=30_000.0)
    accepted_buy = MarketDecision(symbol="AAPL", action="buy", notional=20_000.0)

    assert await gate.process(buy) is not None
    assert await gate.process(rejected_buy) is None

    sell_result = await gate.process(sell)
    accepted_result = await gate.process(accepted_buy)

    assert sell_result is not None
    assert sell_result.metadata["market_exposure_delta"] == -30_000.0
    assert sell_result.metadata["market_exposure_used"] == 10_000.0
    assert accepted_result is not None
    assert accepted_result.metadata["market_exposure_used"] == 30_000.0


async def test_market_gates_compose_with_generic_gate_algebra() -> None:
    gate = (
        MarketGate.freshness(max_age_ms=500, clock=lambda: 100.0)
        >> MarketGate.monotonic_sequence()
        >> MarketGate.quote_sanity(max_spread_bps=15)
        >> Gate.by_priority(5)
    )
    tick = MarketTick(
        symbol="AAPL",
        bid=100.0,
        ask=100.05,
        event_ts=99.9,
        sequence=1,
        priority=7,
    )

    result = await gate.process(tick)

    assert result is not None
    assert result.metadata["market_age_ms"] == 100.0
    assert result.metadata["market_sequence"] == 1
    assert result.metadata["market_spread_bps"] < 15
