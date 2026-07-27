"""Tests for Signal core type."""

import copy
import warnings
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import Field, field_serializer

from signal_gating import Signal


class TaskSignal(Signal):
    task: str
    urgency: int = 0


class NestedSignal(Signal):
    payload: dict[str, object]
    items: list[dict[str, int]]
    labels: set[str]


class CopySemanticsSignal(Signal):
    payload: dict[str, int]
    excluded_required: str = Field(exclude=True)
    defaulted: int = 7


class CustomSerializedCopySignal(Signal):
    value: str

    @field_serializer("value")
    def _serialize_value(self, value: str) -> dict[str, str]:
        return {"wire_value": value}


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
    assert signal.items[:] == [{"value": 1}]
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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda signal: dict.__setitem__(signal.payload, "tampered", True),
        lambda signal: list.append(signal.items, {"value": 2}),
        lambda signal: set.add(signal.labels, "tampered"),
    ],
)
def test_signal_rejects_builtin_base_class_mutation(
    mutate: Callable[[NestedSignal], Any],
) -> None:
    signal = NestedSignal(
        payload={"auth": {"amount": 1}},
        items=[{"value": 1}],
        labels={"approved"},
    )

    with pytest.raises(TypeError):
        mutate(signal)

    assert signal.payload == {"auth": {"amount": 1}}
    assert signal.items == [{"value": 1}]
    assert signal.labels == {"approved"}


@pytest.mark.parametrize(
    "reinitialize",
    [
        lambda signal: dict.__init__(signal.payload, {"tampered": True}),
        lambda signal: list.__init__(signal.items, [{"value": 2}]),
        lambda signal: set.__init__(signal.labels, {"tampered"}),
    ],
)
def test_signal_rejects_builtin_base_class_reinitialization(
    reinitialize: Callable[[NestedSignal], Any],
) -> None:
    signal = NestedSignal(
        payload={"auth": {"amount": 1}},
        items=[{"value": 1}],
        labels={"approved"},
    )

    with pytest.raises(TypeError):
        reinitialize(signal)

    assert signal.payload == {"auth": {"amount": 1}}
    assert signal.items == [{"value": 1}]
    assert signal.labels == {"approved"}


def test_signal_model_copy_update_freezes_aliases_without_validation() -> None:
    signal = NestedSignal(payload={}, items=[], labels=set())
    payload = {"auth": {"amount": 1}}
    items = [{"value": "not-validated"}]
    labels = {"approved"}

    copied = signal.model_copy(
        update={"payload": payload, "items": items, "labels": labels}  # type: ignore[dict-item]
    )

    payload["auth"]["amount"] = 1000  # type: ignore[index]
    items[0]["value"] = "tampered"
    labels.add("tampered")

    assert copied.payload == {"auth": {"amount": 1}}
    assert copied.items == [{"value": "not-validated"}]
    assert copied.labels == {"approved"}
    with pytest.raises(TypeError):
        copied.items.append({"value": 2})


def test_signal_model_copy_update_preserves_excluded_and_unset_fields() -> None:
    signal = CopySemanticsSignal(payload={"value": 1}, excluded_required="secret")

    copied = signal.model_copy(update={"priority": 2})

    assert copied.excluded_required == "secret"
    assert copied.defaulted == 7
    assert copied.model_fields_set == signal.model_fields_set | {"priority"}
    assert "defaulted" not in copied.model_dump(exclude_unset=True)


def test_signal_model_copy_update_does_not_revalidate_custom_serializer_output() -> None:
    signal = CustomSerializedCopySignal(value="native")

    copied = signal.model_copy(update={"priority": 2})

    assert copied.value == "native"
    assert copied.model_dump()["value"] == {"wire_value": "native"}


def test_signal_model_copy_preserves_native_shallow_and_deep_identity() -> None:
    signal = NestedSignal(
        payload={"auth": {"amount": 1}},
        items=[{"value": 1}],
        labels={"approved"},
    )

    shallow = signal.model_copy(update={"priority": 2})
    deep = signal.model_copy(update={"priority": 2}, deep=True)

    assert shallow.payload is signal.payload
    assert shallow.items is signal.items
    assert shallow.labels is signal.labels
    assert deep.payload is not signal.payload
    assert deep.items is not signal.items
    assert deep.labels is not signal.labels


def test_signal_supports_deepcopy_without_losing_immutability() -> None:
    signal = NestedSignal(
        payload={"auth": {"amount": 1}},
        items=[{"value": 1}],
        labels={"approved"},
    )

    copied = copy.deepcopy(signal)

    assert copied == signal
    assert copied.payload is not signal.payload
    assert copied.payload["auth"] is not signal.payload["auth"]
    with pytest.raises(TypeError):
        copied.payload["auth"]["amount"] = 2  # type: ignore[index]


def test_signal_model_copy_deep_succeeds_without_losing_immutability() -> None:
    signal = NestedSignal(
        payload={"auth": {"amount": 1}},
        items=[{"value": 1}],
        labels={"approved"},
    )

    copied = signal.model_copy(deep=True)

    assert copied == signal
    assert copied.items is not signal.items
    assert copied.items[0] is not signal.items[0]
    with pytest.raises(TypeError):
        copied.items.append({"value": 2})


def test_signal_dump_preserves_python_container_shapes_without_warnings() -> None:
    class ContainerShapeSignal(Signal):
        mutable_list: list[int]
        fixed_tuple: tuple[int, ...]
        mutable_set: set[str]
        fixed_frozenset: frozenset[str]

    signal = ContainerShapeSignal(
        mutable_list=[1, 2],
        fixed_tuple=(1, 2),
        mutable_set={"a", "b"},
        fixed_frozenset=frozenset({"a", "b"}),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        python_dump = signal.model_dump()
        json_dump = signal.model_dump(mode="json")
        restored = Signal.from_wire(signal.to_wire())

    assert type(python_dump["mutable_list"]) is list
    assert type(python_dump["fixed_tuple"]) is tuple
    assert type(python_dump["mutable_set"]) is set
    assert type(python_dump["fixed_frozenset"]) is frozenset
    assert type(json_dump["mutable_list"]) is list
    assert type(json_dump["fixed_tuple"]) is list
    assert type(json_dump["mutable_set"]) is list
    assert type(json_dump["fixed_frozenset"]) is list
    assert restored == signal


def test_frozen_list_preserves_safe_builtin_read_operations() -> None:
    signal = NestedSignal(payload={}, items=[{"value": 1}], labels=set())

    sliced = signal.items[:]
    added = signal.items + [{"value": 2}]
    reverse_added = [{"value": 0}] + signal.items
    multiplied = signal.items * 2
    reverse_multiplied = 2 * signal.items

    assert type(sliced) is list
    assert sliced == [{"value": 1}]
    assert type(added) is list
    assert added == [{"value": 1}, {"value": 2}]
    assert reverse_added == [{"value": 0}, {"value": 1}]
    assert multiplied == [{"value": 1}, {"value": 1}]
    assert reverse_multiplied == multiplied


def test_frozen_mapping_union_returns_plain_dict() -> None:
    signal = NestedSignal(payload={"left": 1}, items=[], labels=set())

    merged = signal.payload | {"right": 2}
    reverse_merged = {"left": 0, "first": True} | signal.payload

    assert type(merged) is dict
    assert merged == {"left": 1, "right": 2}
    assert type(reverse_merged) is dict
    assert reverse_merged == {"left": 1, "first": True}


def test_frozen_set_preserves_standard_nonmutating_operations() -> None:
    signal = NestedSignal(payload={}, items=[], labels={"a", "b"})

    assert type(signal.labels.union({"c"})) is set
    assert signal.labels.union({"c"}) == {"a", "b", "c"}
    assert type(signal.labels.intersection({"b", "c"})) is set
    assert signal.labels.intersection({"b", "c"}) == {"b"}
    assert type(signal.labels.difference({"b"})) is set
    assert signal.labels.difference({"b"}) == {"a"}
    assert type(signal.labels.symmetric_difference({"b", "c"})) is set
    assert signal.labels.symmetric_difference({"b", "c"}) == {"a", "c"}
    assert signal.labels.issubset({"a", "b", "c"})
    assert signal.labels.issuperset({"a"})
    assert signal.labels.isdisjoint({"c"})
    assert type(signal.labels | {"c"}) is set
    assert type(signal.labels & {"b", "c"}) is set
    assert type(signal.labels - {"b"}) is set
    assert type(signal.labels ^ {"b", "c"}) is set


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
