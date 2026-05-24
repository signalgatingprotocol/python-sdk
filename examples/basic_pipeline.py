"""Basic pipeline example: composing gates to control signal flow."""

import asyncio

from signal_gating import Gate, Pipeline, Signal


class AlertSignal(Signal):
    message: str
    severity: int = 0


async def main():
    # Build a pipeline that filters low-severity alerts and enriches the rest
    pipeline = Pipeline([
        Gate.by_priority(min_priority=3),
        Gate.deduplicate(window=10),
        Gate.transform(lambda s: s.with_metadata(reviewed=True)),
    ])

    alerts = [
        AlertSignal(message="disk space low", severity=2, priority=2),
        AlertSignal(message="CPU spike", severity=7, priority=7),
        AlertSignal(message="memory warning", severity=5, priority=5),
        AlertSignal(message="CPU spike", severity=7, priority=7),  # duplicate
    ]

    for alert in alerts:
        result = await pipeline.process(alert)
        if result:
            print(f"PASSED: {alert.message} (priority={alert.priority})")
        else:
            print(f"REJECTED: {alert.message} (priority={alert.priority})")


if __name__ == "__main__":
    asyncio.run(main())
