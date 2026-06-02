"""Tests for Signal core type."""

import pytest

from signal_gating import Signal


class TaskSignal(Signal):
    task: str
    urgency: int = 0


def test_signal_creation():
    s = Signal()
    assert s.id
    assert s.timestamp > 0
    assert s.priority == 0
    assert s.source == ""
    assert s.trace_id


def test_signal_subclass():
    s = TaskSignal(task="build", urgency=5)
    assert s.task == "build"
    assert s.urgency == 5
    assert isinstance(s, Signal)


def test_signal_evolve():
    s = TaskSignal(task="build", priority=3)
    s2 = s.evolve(priority=10)
    assert s2.priority == 10
    assert s2.task == "build"
    assert s2.id != s.id  # New ID
    assert s2.trace_id == s.trace_id  # Same trace


def test_signal_with_source():
    s = Signal()
    s2 = s.with_source("planner")
    assert s2.source == "planner"
    assert s.source == ""  # Original unchanged (immutable)


def test_signal_with_metadata():
    s = Signal()
    s2 = s.with_metadata(region="us-east", tier="premium")
    assert s2.metadata["region"] == "us-east"
    assert s2.metadata["tier"] == "premium"


def test_signal_metadata_immutable():
    s = Signal().with_metadata(region="us-east")
    with pytest.raises(TypeError):
        s.metadata["region"] = "eu-west"  # type: ignore[index]


def test_signal_immutable():
    s = Signal()
    try:
        s.priority = 10  # type: ignore
        assert False, "Should have raised"
    except Exception:
        pass


def test_signal_repr():
    s = TaskSignal(task="test", priority=5)
    r = repr(s)
    assert "TaskSignal" in r
    assert "test" in r


def test_signal_repr_hides_defaults():
    s = Signal()
    r = repr(s)
    # Should hide id, timestamp, trace_id, and default-valued fields
    assert "id=" not in r
    assert "timestamp=" not in r
    assert "trace_id=" not in r
    assert "source=" not in r
    assert "priority=" not in r
    assert "metadata=" not in r
    assert "correlation_id=" not in r
    assert r == "Signal()"


def test_signal_repr_shows_non_defaults():
    s = Signal(priority=5, source="agent-a")
    r = repr(s)
    assert "priority=5" in r
    assert "source='agent-a'" in r


def test_signal_correlation_id():
    s = Signal()
    assert s.correlation_id == ""
    s2 = s.evolve(correlation_id="req-123")
    assert s2.correlation_id == "req-123"
    assert s.correlation_id == ""  # Original unchanged


def test_signal_correlation_id_preserved_in_evolve():
    s = Signal(correlation_id="req-abc")
    s2 = s.evolve(priority=10)
    assert s2.correlation_id == "req-abc"
    assert s2.priority == 10
