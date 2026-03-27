"""Tracing — observability for signal flow through the protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """A single span in a signal's trace through the system."""

    trace_id: str
    signal_id: str
    agent: str
    gate: str
    action: str  # "passed", "rejected", "transformed", "error"
    timestamp: float = field(default_factory=time.monotonic)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects trace spans for signal flow observability.

    Usage:
        tracer = Tracer()

        # Attach to a mesh
        mesh = Mesh()
        # ... record spans as signals flow ...

        # Inspect traces
        for span in tracer.get_trace(trace_id):
            print(f"{span.agent} -> {span.gate}: {span.action}")
    """

    def __init__(self, max_spans: int = 10000):
        self._spans: list[Span] = []
        self._max_spans = max_spans

    def record(
        self,
        trace_id: str,
        signal_id: str,
        agent: str,
        gate: str,
        action: str,
        duration_ms: float = 0.0,
        **metadata: Any,
    ) -> Span:
        """Record a trace span."""
        span = Span(
            trace_id=trace_id,
            signal_id=signal_id,
            agent=agent,
            gate=gate,
            action=action,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        self._spans.append(span)
        if len(self._spans) > self._max_spans:
            self._spans = self._spans[-self._max_spans:]
        return span

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a given trace ID."""
        return [s for s in self._spans if s.trace_id == trace_id]

    def get_agent_spans(self, agent: str) -> list[Span]:
        """Get all spans for a given agent."""
        return [s for s in self._spans if s.agent == agent]

    def clear(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()

    @property
    def span_count(self) -> int:
        return len(self._spans)

    def summary(self) -> dict[str, Any]:
        """Summary statistics across all recorded spans."""
        if not self._spans:
            return {"total_spans": 0}
        actions: dict[str, int] = {}
        agents: set[str] = set()
        traces: set[str] = set()
        for s in self._spans:
            actions[s.action] = actions.get(s.action, 0) + 1
            agents.add(s.agent)
            traces.add(s.trace_id)
        return {
            "total_spans": len(self._spans),
            "unique_traces": len(traces),
            "unique_agents": len(agents),
            "actions": actions,
        }
