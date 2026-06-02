"""Tests for signal tracing and observability."""

import time

from signal_gating import MeshEvent, OpenTelemetrySpanExporter, Signal, Span, SpanSink, Tracer


def test_record_span():
    tracer = Tracer()
    span = tracer.record("trace-1", "sig-1", "agent-a", "priority_filter", "passed")
    assert span.trace_id == "trace-1"
    assert span.action == "passed"
    assert tracer.span_count == 1
    assert span.timestamp <= time.time()


def test_span_to_dict():
    span = Span(
        trace_id="t",
        signal_id="s",
        agent="a",
        gate="g",
        action="passed",
        timestamp=123.0,
        duration_ms=4.5,
        metadata={"target": "b"},
    )

    assert span.to_dict() == {
        "trace_id": "t",
        "signal_id": "s",
        "agent": "a",
        "gate": "g",
        "action": "passed",
        "timestamp": 123.0,
        "duration_ms": 4.5,
        "metadata": {"target": "b"},
    }


def test_get_trace():
    tracer = Tracer()
    tracer.record("trace-1", "sig-1", "agent-a", "gate-1", "passed")
    tracer.record("trace-1", "sig-1", "agent-b", "gate-2", "transformed")
    tracer.record("trace-2", "sig-2", "agent-a", "gate-1", "rejected")

    trace = tracer.get_trace("trace-1")
    assert len(trace) == 2


def test_get_agent_spans():
    tracer = Tracer()
    tracer.record("t1", "s1", "agent-a", "g1", "passed")
    tracer.record("t2", "s2", "agent-b", "g1", "passed")
    tracer.record("t3", "s3", "agent-a", "g2", "rejected")

    spans = tracer.get_agent_spans("agent-a")
    assert len(spans) == 2


def test_max_spans():
    tracer = Tracer(max_spans=5)
    for i in range(10):
        tracer.record(f"t{i}", f"s{i}", "a", "g", "passed")
    assert tracer.span_count == 5


def test_summary():
    tracer = Tracer()
    tracer.record("t1", "s1", "a", "g", "passed")
    tracer.record("t1", "s2", "b", "g", "rejected")
    tracer.record("t2", "s3", "a", "g", "passed")

    s = tracer.summary()
    assert s["total_spans"] == 3
    assert s["unique_traces"] == 2
    assert s["unique_agents"] == 2
    assert s["actions"]["passed"] == 2
    assert s["actions"]["rejected"] == 1


def test_clear():
    tracer = Tracer()
    tracer.record("t1", "s1", "a", "g", "passed")
    tracer.clear()
    assert tracer.span_count == 0


def test_summary_latency_percentiles():
    tracer = Tracer()
    for i in range(100):
        tracer.record("t1", f"s{i}", "a", "g", "passed", duration_ms=float(i + 1))

    s = tracer.summary()
    assert "latency_ms" in s
    lat = s["latency_ms"]
    assert lat["min"] == 1.0
    assert lat["max"] == 100.0
    assert lat["mean"] == 50.5
    assert lat["p50"] == 51.0
    assert lat["p95"] == 96.0
    assert lat["p99"] == 100.0


def test_summary_no_latency_when_zero_durations():
    tracer = Tracer()
    tracer.record("t1", "s1", "a", "g", "passed")  # duration_ms=0
    s = tracer.summary()
    assert "latency_ms" not in s


def test_record_streams_to_sinks():
    received: list[Span] = []
    tracer = Tracer(sinks=[received.append])

    span = tracer.record("t1", "s1", "a", "g", "passed")

    assert received == [span]


def test_record_event_maps_mesh_event_to_safe_span_metadata():
    tracer = Tracer()
    signal = Signal(priority=7, correlation_id="cid", parent_id="pid")
    event = MeshEvent(
        action="request_sent",
        signal=signal,
        source="mesh",
        target="worker",
        event_kind="mesh",
        metadata={
            "gate": "risk_check",
            "tool_name": "rank",
            "argument_names": ["symbol"],
            "responses": [{"raw": "not exported"}],
            "result": {"raw": "not exported"},
        },
    )

    span = tracer.record_event(event)

    assert span.trace_id == signal.trace_id
    assert span.signal_id == signal.id
    assert span.agent == "mesh->worker"
    assert span.gate == "mesh_event"
    assert span.action == "request_sent"
    assert span.metadata["event_kind"] == "mesh"
    assert span.metadata["source"] == "mesh"
    assert span.metadata["target"] == "worker"
    assert span.metadata["signal_type"] == "Signal"
    assert span.metadata["priority"] == 7
    assert span.metadata["correlation_id"] == "cid"
    assert span.metadata["parent_id"] == "pid"
    assert span.metadata["event_gate"] == "risk_check"
    assert span.metadata["tool_name"] == "rank"
    assert span.metadata["argument_names"] == ["symbol"]
    assert "responses" not in span.metadata
    assert "result" not in span.metadata


def test_add_remove_sink():
    received: list[Span] = []
    tracer = Tracer()

    sink = received.append
    assert tracer.remove_sink(sink) is False
    tracer.add_sink(sink)
    assert tracer.remove_sink(sink) is True
    tracer.record("t1", "s1", "a", "g", "passed")
    assert received == []


def test_sink_failure_does_not_block_recording():
    def broken_sink(_span: Span) -> None:
        raise RuntimeError("collector offline")

    tracer = Tracer(sinks=[broken_sink])
    span = tracer.record("t1", "s1", "a", "g", "passed")

    assert span.trace_id == "t1"
    assert tracer.span_count == 1
    assert tracer.sink_errors == 1


def test_export_replays_retained_spans():
    tracer = Tracer()
    first = tracer.record("t1", "s1", "a", "g", "passed")
    second = tracer.record("t2", "s2", "b", "g", "rejected")
    received: list[Span] = []

    assert tracer.export(received.append) == 2
    assert received == [first, second]


def test_span_to_otel_attributes_encodes_complex_metadata():
    span = Span(
        trace_id="t",
        signal_id="s",
        agent="agent",
        gate="route",
        action="routed",
        duration_ms=1.25,
        metadata={"target": "worker", "payload": {"symbol": "SPY"}},
    )

    attrs = span.to_otel_attributes()

    assert attrs["sgp.trace_id"] == "t"
    assert attrs["sgp.signal_id"] == "s"
    assert attrs["sgp.agent"] == "agent"
    assert attrs["sgp.gate"] == "route"
    assert attrs["sgp.action"] == "routed"
    assert attrs["sgp.duration_ms"] == 1.25
    assert attrs["sgp.target"] == "worker"
    assert attrs["sgp.payload"] == '{"symbol": "SPY"}'


def test_opentelemetry_exporter_uses_span_times_and_attributes():
    class FakeOtelSpan:
        def __init__(self) -> None:
            self.end_time: int | None = None

        def end(self, *, end_time: int | None = None) -> None:
            self.end_time = end_time

    class FakeTracer:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.last_span = FakeOtelSpan()

        def start_span(
            self,
            name: str,
            *,
            start_time: int,
            attributes: dict[str, object],
        ) -> FakeOtelSpan:
            self.calls.append(
                {"name": name, "start_time": start_time, "attributes": attributes}
            )
            return self.last_span

    fake = FakeTracer()
    exporter = OpenTelemetrySpanExporter(
        tracer=fake,
        attributes_namespace="sgp",
        span_name=lambda span: f"custom.{span.action}",
    )
    span = Span(
        trace_id="t",
        signal_id="s",
        agent="agent",
        gate="route",
        action="routed",
        timestamp=10.0,
        duration_ms=25.5,
        metadata={"target": "worker"},
    )

    exporter(span)

    assert fake.calls == [
        {
            "name": "custom.routed",
            "start_time": 10_000_000_000,
            "attributes": {
                "sgp.trace_id": "t",
                "sgp.signal_id": "s",
                "sgp.agent": "agent",
                "sgp.gate": "route",
                "sgp.action": "routed",
                "sgp.duration_ms": 25.5,
                "sgp.target": "worker",
            },
        }
    ]
    assert fake.last_span.end_time == 10_025_500_000


def test_exports():
    assert OpenTelemetrySpanExporter is not None
    assert SpanSink is not None
