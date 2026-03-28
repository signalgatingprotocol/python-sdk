"""Pipelines — composable chains of gates for building signal processing flows."""

from __future__ import annotations

from signal_gating.gate import Gate
from signal_gating.signal import Signal


class Pipeline:
    """An ordered sequence of gates that signals flow through.

    Pipelines provide a structured way to compose gate logic:

        pipeline = Pipeline([
            Gate.by_priority(min_priority=5),
            Gate.deduplicate(window=30),
            Gate.transform(enrich_signal),
        ])

        result = await pipeline.process(signal)
    """

    def __init__(self, gates: list[Gate] | None = None):
        self._gates: list[Gate] = gates or []

    def add(self, gate: Gate) -> Pipeline:
        """Add a gate to the end of the pipeline. Returns self for chaining."""
        self._gates.append(gate)
        return self

    async def process(self, signal: Signal) -> Signal | None:
        """Run a signal through all gates in sequence.

        Returns the (possibly transformed) signal, or None if any gate rejects.
        """
        current: Signal | None = signal
        for gate in self._gates:
            if current is None:
                return None
            current = await gate.process(current)
        return current

    def to_gate(self) -> Gate:
        """Convert this pipeline into a single composable Gate."""
        if not self._gates:
            return Gate.passthrough()
        result = self._gates[0]
        for gate in self._gates[1:]:
            result = result >> gate
        return result

    def __rshift__(self, other: Gate | Pipeline) -> Pipeline:
        """Compose pipeline with a gate or another pipeline using >>."""
        if isinstance(other, Pipeline):
            return Pipeline(self._gates + other._gates)
        return Pipeline(self._gates + [other])

    def __rrshift__(self, other: Gate) -> Pipeline:
        """Allow Gate >> Pipeline composition."""
        return Pipeline([other] + self._gates)

    def __len__(self) -> int:
        return len(self._gates)

    def __repr__(self) -> str:
        names = [g.name for g in self._gates]
        return f"Pipeline({' >> '.join(names)})"
