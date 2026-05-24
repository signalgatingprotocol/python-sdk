"""Tracing: observability for signal flow through the protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
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

    Uses indexed lookups for O(1) trace and agent queries instead of O(n) scans.

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
        # Indexed lookups for O(1) access
        self._by_trace: dict[str, list[Span]] = {}
        self._by_agent: dict[str, list[Span]] = {}

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
        self._by_trace.setdefault(trace_id, []).append(span)
        self._by_agent.setdefault(agent, []).append(span)
        if len(self._spans) > self._max_spans:
            self._evict()
        return span

    def _evict(self) -> None:
        """Evict oldest spans to stay within max_spans, rebuilding indexes."""
        self._spans = self._spans[-self._max_spans :]
        self._by_trace.clear()
        self._by_agent.clear()
        for span in self._spans:
            self._by_trace.setdefault(span.trace_id, []).append(span)
            self._by_agent.setdefault(span.agent, []).append(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a given trace ID. O(1) lookup."""
        return list(self._by_trace.get(trace_id, []))

    def get_agent_spans(self, agent: str) -> list[Span]:
        """Get all spans for a given agent. O(1) lookup."""
        return list(self._by_agent.get(agent, []))

    def clear(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()
        self._by_trace.clear()
        self._by_agent.clear()

    @property
    def span_count(self) -> int:
        return len(self._spans)

    def summary(self) -> dict[str, Any]:
        """Summary statistics across all recorded spans, including latency percentiles."""
        if not self._spans:
            return {"total_spans": 0}
        actions: dict[str, int] = {}
        durations: list[float] = []
        for s in self._spans:
            actions[s.action] = actions.get(s.action, 0) + 1
            if s.duration_ms > 0:
                durations.append(s.duration_ms)
        result: dict[str, Any] = {
            "total_spans": len(self._spans),
            "unique_traces": len(self._by_trace),
            "unique_agents": len(self._by_agent),
            "actions": actions,
        }
        if durations:
            durations.sort()
            n = len(durations)
            result["latency_ms"] = {
                "min": durations[0],
                "max": durations[-1],
                "mean": sum(durations) / n,
                "p50": durations[n // 2],
                "p95": durations[int(n * 0.95)],
                "p99": durations[min(int(n * 0.99), n - 1)],
            }
        return result
