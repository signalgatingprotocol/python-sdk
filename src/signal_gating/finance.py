"""Finance-oriented signal types and gates.

These helpers keep market-signal control out of the generic gate algebra while
making the common production invariants easy to compose: event-time freshness,
monotonic market-data sequence, quote sanity, edge-after-costs, and notional
limits.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Literal

from pydantic import field_validator

from signal_gating.gate import Gate
from signal_gating.signal import Signal

MarketAction = Literal["buy", "sell", "hold", "cancel"]
MarketKeyFn = Callable[[Signal], object]
ClockFn = Callable[[], float]


class MarketTick(Signal):
    """A normalized quote/trade update from a market-data feed."""

    __signal_type__ = "sgp.finance.market_tick.v1"

    symbol: str
    venue: str = ""
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last: float | None = None
    last_size: float | None = None
    event_ts: float | None = None
    sequence: int | None = None

    @field_validator("symbol")
    @classmethod
    def _symbol_required(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("symbol must not be empty")
        return cleaned

    @field_validator("venue")
    @classmethod
    def _venue_clean(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("bid", "ask", "bid_size", "ask_size", "last", "last_size")
    @classmethod
    def _non_negative_float(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("market values must be non-negative")
        return value

    @field_validator("event_ts")
    @classmethod
    def _event_ts_positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("event_ts must be positive")
        return value

    @field_validator("sequence")
    @classmethod
    def _sequence_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("sequence must be non-negative")
        return value


class MarketDecision(Signal):
    """A strategy or risk decision intended for an execution-facing agent."""

    __signal_type__ = "sgp.finance.market_decision.v1"

    symbol: str
    action: MarketAction
    confidence: float = 0.0
    expected_edge_bps: float = 0.0
    max_slippage_bps: float = 0.0
    notional: float = 0.0
    reason: str = ""

    @field_validator("symbol")
    @classmethod
    def _symbol_required(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("symbol must not be empty")
        return cleaned

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @field_validator("max_slippage_bps", "notional")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value


class MarketGate:
    """Factory methods for finance-specific gates."""

    @classmethod
    def freshness(
        cls,
        max_age_ms: float,
        *,
        clock: ClockFn = time.time,
        timestamp_attr: str = "event_ts",
        max_future_ms: float = 1000.0,
        name: str = "market_freshness",
    ) -> Gate:
        """Drop market signals whose event timestamp is stale or far in the future.

        ``Gate.ttl`` measures from ``Signal.timestamp``. Market feeds need
        event-time freshness because ingestion, transport, and agent scheduling
        delay can otherwise make old prices look fresh.
        """
        if max_age_ms < 0:
            raise ValueError("max_age_ms must be >= 0")
        if max_future_ms < 0:
            raise ValueError("max_future_ms must be >= 0")

        def fn(signal: Signal) -> Signal | None:
            raw_ts = getattr(signal, timestamp_attr, None)
            event_ts = raw_ts if isinstance(raw_ts, int | float) else signal.timestamp
            age_ms = (clock() - float(event_ts)) * 1000.0
            if age_ms < -max_future_ms or age_ms > max_age_ms:
                return None
            return signal.with_metadata(market_age_ms=round(age_ms, 3))

        return Gate(fn, name=name)

    @classmethod
    def monotonic_sequence(
        cls,
        *,
        key: MarketKeyFn | None = None,
        sequence_attr: str = "sequence",
        name: str = "market_monotonic_sequence",
    ) -> Gate:
        """Drop duplicate or out-of-order feed updates per symbol/venue key."""
        last_seen: dict[object, int] = {}
        lock = asyncio.Lock()
        key_fn = key or _market_key

        async def fn(signal: Signal) -> Signal | None:
            raw_sequence = getattr(signal, sequence_attr, None)
            if raw_sequence is None:
                return signal
            sequence = int(raw_sequence)
            signal_key = key_fn(signal)
            async with lock:
                previous = last_seen.get(signal_key)
                if previous is not None and sequence <= previous:
                    return None
                last_seen[signal_key] = sequence
            return signal.with_metadata(
                market_sequence=sequence,
                market_sequence_key=str(signal_key),
            )

        return Gate(fn, name=name)

    @classmethod
    def quote_sanity(
        cls,
        *,
        max_spread_bps: float | None = None,
        allow_locked: bool = True,
        name: str = "market_quote_sanity",
    ) -> Gate:
        """Drop crossed, negative, or excessively wide quote updates."""
        if max_spread_bps is not None and max_spread_bps < 0:
            raise ValueError("max_spread_bps must be >= 0")

        def fn(signal: Signal) -> Signal | None:
            bid = _optional_float_attr(signal, "bid")
            ask = _optional_float_attr(signal, "ask")
            last = _optional_float_attr(signal, "last")
            sizes = (
                _optional_float_attr(signal, "bid_size"),
                _optional_float_attr(signal, "ask_size"),
                _optional_float_attr(signal, "last_size"),
            )
            values = tuple(v for v in (bid, ask, last, *sizes) if v is not None)
            if any(v < 0 for v in values):
                return None
            if bid is None or ask is None:
                return signal
            if bid > ask or (bid == ask and not allow_locked):
                return None
            mid = (bid + ask) / 2.0
            if mid <= 0:
                return None
            spread_bps = ((ask - bid) / mid) * 10_000.0
            if max_spread_bps is not None and spread_bps > max_spread_bps:
                return None
            return signal.with_metadata(
                market_mid=round(mid, 10),
                market_spread_bps=round(spread_bps, 6),
            )

        return Gate(fn, name=name)

    @classmethod
    def decision_edge(
        cls,
        min_net_edge_bps: float,
        *,
        min_confidence: float = 0.0,
        name: str = "market_decision_edge",
    ) -> Gate:
        """Require actionable decisions to clear edge, slippage, and confidence.

        ``hold`` and ``cancel`` are control actions and pass through; ``buy`` and
        ``sell`` must have positive edge after estimated slippage.
        """
        if min_confidence < 0 or min_confidence > 1:
            raise ValueError("min_confidence must be between 0 and 1")

        def fn(signal: Signal) -> Signal | None:
            action = getattr(signal, "action", None)
            if action in {"hold", "cancel"}:
                return signal
            expected_edge_bps = _required_float_attr(signal, "expected_edge_bps")
            max_slippage_bps = _required_float_attr(signal, "max_slippage_bps")
            confidence = _required_float_attr(signal, "confidence")
            net_edge_bps = expected_edge_bps - max_slippage_bps
            if confidence < min_confidence or net_edge_bps < min_net_edge_bps:
                return None
            return signal.with_metadata(market_net_edge_bps=round(net_edge_bps, 6))

        return Gate(fn, name=name)

    @classmethod
    def notional_limit(
        cls,
        max_notional: float,
        *,
        notional_attr: str = "notional",
        name: str = "market_notional_limit",
    ) -> Gate:
        """Drop decisions whose notional exceeds a configured budget."""
        if max_notional < 0:
            raise ValueError("max_notional must be >= 0")

        def fn(signal: Signal) -> Signal | None:
            notional = _required_float_attr(signal, notional_attr)
            if notional > max_notional:
                return None
            return signal.with_metadata(market_notional_limit=max_notional)

        return Gate(fn, name=name)


def _market_key(signal: Signal) -> tuple[str, str, str]:
    symbol = str(getattr(signal, "symbol", type(signal).__name__))
    venue = str(getattr(signal, "venue", ""))
    return (type(signal).wire_type(), symbol, venue)


def _optional_float_attr(signal: Signal, attr: str) -> float | None:
    value = getattr(signal, attr, None)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise TypeError(f"{attr} must be numeric")
    return float(value)


def _required_float_attr(signal: Signal, attr: str) -> float:
    value = getattr(signal, attr, None)
    if not isinstance(value, int | float):
        raise TypeError(f"{attr} must be numeric")
    return float(value)


__all__ = [
    "MarketAction",
    "MarketDecision",
    "MarketGate",
    "MarketKeyFn",
    "MarketTick",
]
