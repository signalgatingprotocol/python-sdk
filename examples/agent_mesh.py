"""Agent mesh example: fan-out, fan-in, and topology inspection."""

import asyncio

from signal_gating import Agent, Mesh, Signal


class DataSignal(Signal):
    payload: str
    stage: str = "raw"


async def main():
    # Create a processing mesh:
    #   ingester -> [validator, enricher] -> aggregator
    ingester = Agent("ingester")
    validator = Agent("validator")
    enricher = Agent("enricher")
    aggregator = Agent("aggregator")

    results: list[str] = []

    @validator.on(DataSignal)
    async def validate(s: DataSignal):
        print(f"  [validator] Validating: {s.payload}")
        await validator.emit(s.evolve(stage="validated"))

    @enricher.on(DataSignal)
    async def enrich(s: DataSignal):
        print(f"  [enricher] Enriching: {s.payload}")
        await enricher.emit(s.evolve(stage="enriched", payload=f"{s.payload}+metadata"))

    @aggregator.on(DataSignal)
    async def aggregate(s: DataSignal):
        results.append(f"{s.source}:{s.stage}:{s.payload}")
        print(f"  [aggregator] Received from {s.source}: {s.payload} ({s.stage})")

    # Build mesh topology
    mesh = Mesh([ingester, validator, enricher, aggregator])

    # Fan-out: ingester sends to both validator and enricher
    mesh.broadcast_connect(ingester, [validator, enricher])

    # Fan-in: both processors feed into aggregator
    mesh.converge_connect([validator, enricher], aggregator)

    # Inspect topology
    print("Mesh topology:")
    topo = mesh.topology()
    for edge in topo["edges"]:
        print(f"  {edge['source']} -> {edge['target']}")

    print("\nProcessing signals...")
    async with mesh:
        await ingester.emit(DataSignal(payload="sensor-reading-42", priority=5))
        await asyncio.sleep(0.1)

    print(f"\nAggregator received {len(results)} signals:")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    asyncio.run(main())
