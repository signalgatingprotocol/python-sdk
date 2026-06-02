"""Basic pipeline example: composing gates to control signal flow."""

import asyncio

from signal_gating import Gate, Pipeline, Signal


class AlertSignal(Signal):
    message: str


async def main():
    # Build a pipeline that filters low-priority alerts and enriches the rest
    pipeline = Pipeline([
        Gate.by_priority(min_priority=3),
        Gate.deduplicate(window=10),
        Gate.transform(lambda s: s.with_metadata(reviewed=True)),
    ])

    alerts = [
        AlertSignal(message="disk space low", priority=2),
        AlertSignal(message="CPU spike", priority=7),
        AlertSignal(message="memory warning", priority=5),
        AlertSignal(message="CPU spike", priority=7),  # duplicate
    ]

    for alert in alerts:
        result = await pipeline.process(alert)
        if result:
            print(f"PASSED: {alert.message} (priority={alert.priority})")
        else:
            print(f"REJECTED: {alert.message} (priority={alert.priority})")


if __name__ == "__main__":
    asyncio.run(main())
