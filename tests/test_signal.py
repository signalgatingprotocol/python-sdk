"""Tests for Signal core type."""

import warnings

import pytest

from signal_gating import Signal


class TaskSignal(Signal):
    task: str
    urgency: int = 0


class NestedSignal(Signal):
    payload: dict[str, object]
    items: list[dict[str, int]]
    labels: set[str]


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


def test_signal_recursively_freezes_mutable_fields_and_input_aliases() -> None:
    payload = {"auth": {"amount": 1}}
    items = [{"value": 1}]
    labels = {"approved"}
    signal = NestedSignal(payload=payload, items=items, labels=labels)

    payload["auth"]["amount"] = 1000  # type: ignore[index]
    items[0]["value"] = 1000
    labels.add("tampered")

    assert signal.payload["auth"] == {"amount": 1}
    assert signal.items == [{"value": 1}]
    assert signal.labels == {"approved"}

    with pytest.raises(TypeError):
        signal.payload["auth"]["amount"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        signal.items[0]["value"] = 2
    with pytest.raises(TypeError):
        signal.items.append({"value": 2})
    with pytest.raises(TypeError):
        signal.labels.add("blocked")


def test_nested_signal_wire_roundtrip_without_serializer_warnings() -> None:
    signal = NestedSignal(
        payload={"auth": {"amount": 1}},
        items=[{"value": 1}],
        labels={"approved"},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = Signal.from_wire(signal.to_wire())

    assert type(restored) is NestedSignal
    assert restored.payload == {"auth": {"amount": 1}}
    assert restored.items == [{"value": 1}]
    assert restored.labels == {"approved"}


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
