"""Financial-physics mesh: freshness, microstructure gates, risk, receipts.

This example is intentionally deterministic and offline. It models the minimum
shape of a market-signal control loop:

1. A feed agent emits normalized market ticks.
2. Freshness, sequence, quote-sanity, and priority gates decide what an analyst
   is allowed to see.
3. The analyst converts order-book imbalance into a decision with estimated
   edge after slippage.
4. Risk gates admit only decisions with enough net edge, bounded notional, and
   bounded cumulative exposure.
5. A TrajectoryRecorder produces tamper-evident receipts for audit/replay.

Run:

    python examples/financial_physics_mesh.py
"""

from __future__ import annotations

import asyncio
import time

from signal_gating import Agent, MarketDecision, MarketGate, MarketTick, Mesh, TrajectoryRecorder


def priority_from_spread(tick: MarketTick) -> MarketTick:
    """Escalate tight, liquid quotes; keep wide/noisy updates low priority."""
    if tick.bid is None or tick.ask is None:
        return tick.evolve(priority=0)
    bid_size = tick.bid_size or 0.0
    ask_size = tick.ask_size or 0.0
    mid = (tick.bid + tick.ask) / 2.0
    if mid <= 0:
        return tick.evolve(priority=0)
    spread_bps = ((tick.ask - tick.bid) / mid) * 10_000.0
    priority = 8 if spread_bps < 8 and bid_size + ask_size >= 1_000 else 3
    return tick.evolve(priority=priority)


async def main() -> None:
    now = time.time()
    ticks = [
        MarketTick(
            symbol="AAPL",
            venue="XNAS",
            bid=100.00,
            ask=100.04,
            bid_size=2_400,
            ask_size=800,
            event_ts=now - 0.050,
            sequence=101,
        ),
        MarketTick(
            symbol="AAPL",
            venue="XNAS",
            bid=100.02,
            ask=100.05,
            bid_size=2_800,
            ask_size=700,
            event_ts=now - 0.040,
            sequence=102,
        ),
        MarketTick(
            symbol="AAPL",
            venue="XNAS",
            bid=99.10,
            ask=101.20,
            bid_size=1_000,
            ask_size=1_000,
            event_ts=now - 0.030,
            sequence=103,
        ),  # rejected: too wide
        MarketTick(
            symbol="AAPL",
            venue="XNAS",
            bid=100.03,
            ask=100.06,
            bid_size=2_500,
            ask_size=700,
            event_ts=now - 1.000,
            sequence=104,
        ),  # rejected: stale
        MarketTick(
            symbol="AAPL",
            venue="XNAS",
            bid=100.04,
            ask=100.07,
            bid_size=2_900,
            ask_size=600,
            event_ts=now - 0.020,
            sequence=102,
        ),  # rejected: duplicate/out-of-order sequence
    ]

    feed = Agent("feed")
    analyst = Agent("microstructure_analyst")
    risk = Agent("risk_gate")
    executor = Agent("executor")
    accepted: list[MarketDecision] = []
    done = asyncio.Event()

    @analyst.on(MarketTick)
    async def analyze_tick(tick: MarketTick) -> None:
        bid_size = tick.bid_size or 0.0
        ask_size = tick.ask_size or 0.0
        imbalance = (bid_size - ask_size) / max(bid_size + ask_size, 1.0)
        expected_edge_bps = imbalance * 10.0
        action = "buy" if expected_edge_bps > 0 else "hold"
        decision = MarketDecision(
            symbol=tick.symbol,
            action=action,
            confidence=min(0.99, 0.55 + abs(imbalance)),
            expected_edge_bps=expected_edge_bps,
            max_slippage_bps=2.0,
            notional=25_000.0,
            reason=f"book_imbalance={imbalance:.3f}",
            priority=tick.priority,
            parent_id=tick.id,
            trace_id=tick.trace_id,
        )
        await analyst.emit(decision)

    @risk.on(MarketDecision)
    async def approve(decision: MarketDecision) -> None:
        await risk.emit(decision.with_metadata(risk_approved=True))

    @executor.on(MarketDecision)
    async def execute(decision: MarketDecision) -> None:
        accepted.append(decision)
        print(
            "ACCEPTED "
            f"{decision.action.upper()} {decision.symbol} "
            f"net_edge={decision.metadata['market_net_edge_bps']:.2f}bps "
            f"notional=${decision.notional:,.0f}"
        )
        if len(accepted) >= 2:
            done.set()

    recorder = TrajectoryRecorder()
    mesh = Mesh([feed, analyst, risk, executor])
    mesh.record(recorder)
    mesh.connect(
        feed,
        analyst,
        gate=(
            MarketGate.freshness(max_age_ms=250)
            >> MarketGate.monotonic_sequence()
            >> MarketGate.quote_sanity(max_spread_bps=12)
        ),
    )
    mesh.connect(
        analyst,
        risk,
        gate=MarketGate.decision_edge(min_net_edge_bps=3.0, min_confidence=0.75),
    )
    mesh.connect(
        risk,
        executor,
        gate=MarketGate.notional_limit(50_000.0) >> MarketGate.exposure_limit(50_000.0),
    )

    async with mesh:
        for tick in ticks:
            await feed.emit(priority_from_spread(tick))
        await asyncio.wait_for(done.wait(), timeout=3.0)

    print(f"accepted_decisions={len(accepted)}")
    print(f"receipts={len(recorder.receipts)}")
    print(f"all_receipts_verify={all(receipt.verify() for receipt in recorder.receipts)}")
    for trace_id, receipts in recorder.trajectories().items():
        actions = " -> ".join(f"{r.action}:{r.source}->{r.target}" for r in receipts)
        print(f"trace={trace_id[:8]} {actions}")


if __name__ == "__main__":
    asyncio.run(main())
