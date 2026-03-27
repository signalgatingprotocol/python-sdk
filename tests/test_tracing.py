"""Tests for signal tracing and observability."""

from signal_gating import Tracer


def test_record_span():
    tracer = Tracer()
    span = tracer.record("trace-1", "sig-1", "agent-a", "priority_filter", "passed")
    assert span.trace_id == "trace-1"
    assert span.action == "passed"
    assert tracer.span_count == 1


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
