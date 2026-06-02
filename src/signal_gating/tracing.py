"""Tracing: observability for signal flow through the protocol."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

logger = logging.getLogger("signal_gating.tracing")

SpanSink = Callable[["Span"], None]


def _otel_attribute_value(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, tuple | list) and all(
        item is None or isinstance(item, str | bool | int | float)
        for item in value
    ):
        return list(value)
    return json.dumps(value, default=str, sort_keys=True)


@dataclass(slots=True)
class Span:
    """A single span in a signal's trace through the system."""

    trace_id: str
    signal_id: str
    agent: str
    gate: str
    action: str  # "passed", "rejected", "transformed", "error"
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this span."""
        return {
            "trace_id": self.trace_id,
            "signal_id": self.signal_id,
            "agent": self.agent,
            "gate": self.gate,
            "action": self.action,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }

    def to_otel_attributes(self, *, namespace: str = "sgp") -> dict[str, Any]:
        """Convert this span into OpenTelemetry-safe attributes.

        SGP trace IDs are protocol-level lineage identifiers, not OpenTelemetry
        trace IDs. They are exported as attributes so backends can correlate
        agent runs without forcing a specific OTel context model.
        """
        prefix = namespace.rstrip(".")
        attrs: dict[str, Any] = {
            f"{prefix}.trace_id": self.trace_id,
            f"{prefix}.signal_id": self.signal_id,
            f"{prefix}.agent": self.agent,
            f"{prefix}.gate": self.gate,
            f"{prefix}.action": self.action,
            f"{prefix}.duration_ms": self.duration_ms,
        }
        for key, value in self.metadata.items():
            attrs[f"{prefix}.{key}"] = _otel_attribute_value(value)
        return attrs


class OpenTelemetrySpanExporter:
    """Callable sink that exports SGP spans as OpenTelemetry spans.

    The OpenTelemetry import is lazy. Core users do not pay the dependency cost;
    production users can install ``signal-gating[otel]`` or provide their own
    compatible tracer object.
    """

    def __init__(
        self,
        *,
        tracer: Any | None = None,
        tracer_name: str = "signal_gating",
        attributes_namespace: str = "sgp",
        span_name: Callable[[Span], str] | None = None,
    ) -> None:
        if tracer is None:
            try:
                otel_trace = import_module("opentelemetry.trace")
            except ImportError as e:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "OpenTelemetrySpanExporter requires opentelemetry-api. "
                    "Install it with: pip install 'signal-gating[otel]'"
                ) from e
            tracer = otel_trace.get_tracer(tracer_name)
        self._tracer = tracer
        self._attributes_namespace = attributes_namespace
        self._span_name = span_name or self._default_span_name

    def __call__(self, span: Span) -> None:
        """Export one SGP span into the configured OpenTelemetry tracer."""
        name = self._span_name(span)
        start_ns = int(span.timestamp * 1_000_000_000)
        end_ns = int(
            (span.timestamp + max(span.duration_ms, 0.0) / 1000.0) * 1_000_000_000
        )
        otel_span = self._tracer.start_span(
            name,
            start_time=start_ns,
            attributes=span.to_otel_attributes(namespace=self._attributes_namespace),
        )
        try:
            if span.action == "error":
                try:
                    otel_trace = import_module("opentelemetry.trace")
                    status_type = getattr(otel_trace, "Status")
                    status_code_type = getattr(otel_trace, "StatusCode")

                    otel_span.set_status(
                        status_type(status_code_type.ERROR, span.action)
                    )
                except Exception:  # pragma: no cover - exporter compatibility
                    logger.debug("Could not set OpenTelemetry status", exc_info=True)
        finally:
            otel_span.end(end_time=end_ns)

    @staticmethod
    def _default_span_name(span: Span) -> str:
        return f"signal_gating.{span.gate}.{span.action}"


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

    def __init__(
        self,
        max_spans: int = 10000,
        sinks: list[SpanSink] | None = None,
    ):
        self._spans: list[Span] = []
        self._max_spans = max_spans
        self._sinks: list[SpanSink] = list(sinks or [])
        self._sink_errors = 0
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
        self._publish(span)
        return span

    def add_sink(self, sink: SpanSink) -> None:
        """Stream subsequently recorded spans into ``sink``.

        Sinks are best-effort observers: if a sink raises, tracing records the
        failure and keeps the agent system moving.
        """
        self._sinks.append(sink)

    def remove_sink(self, sink: SpanSink) -> bool:
        """Remove a sink. Returns ``True`` if it was registered."""
        try:
            self._sinks.remove(sink)
            return True
        except ValueError:
            return False

    def export(self, sink: SpanSink) -> int:
        """Replay all retained spans into a sink. Returns the number exported."""
        exported = 0
        for span in list(self._spans):
            try:
                sink(span)
                exported += 1
            except Exception:
                self._sink_errors += 1
                logger.warning("Tracer sink failed during export", exc_info=True)
        return exported

    @property
    def sink_errors(self) -> int:
        """Number of sink failures suppressed by this tracer."""
        return self._sink_errors

    def _publish(self, span: Span) -> None:
        for sink in tuple(self._sinks):
            try:
                sink(span)
            except Exception:
                self._sink_errors += 1
                logger.warning("Tracer sink failed", exc_info=True)

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
