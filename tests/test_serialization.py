"""Tests for the signal wire format, type registry, and durable DLQ recovery."""

from __future__ import annotations

import asyncio
import json

import pytest

from signal_gating import (
    WIRE_VERSION,
    Agent,
    Channel,
    Mesh,
    Signal,
    SignalSerializationError,
    UnknownSignalType,
    from_wire,
    lookup_signal,
    register_signal,
    registered_signals,
    to_wire,
)


class WireTask(Signal):
    task: str
    urgency: int = 0


class WireResult(Signal):
    output: str
    score: float = 0.0


# --- Wire envelope shape ---


def test_envelope_shape():
    sig = WireTask(task="build")
    wire = sig.to_wire()
    assert wire["sgp"] == WIRE_VERSION
    assert wire["type"] == "WireTask"
    assert wire["data"]["task"] == "build"
    # The envelope carries every field, not just domain fields.
    assert "id" in wire["data"] and "trace_id" in wire["data"]


def test_module_level_and_method_to_wire_agree():
    sig = WireTask(task="x", urgency=2)
    assert to_wire(sig) == sig.to_wire()


# --- Faithful round-trips ---


def test_roundtrip_preserves_subclass_and_all_fields():
    sig = WireTask(task="build", urgency=7, priority=4, source="planner")
    back = Signal.from_wire(sig.to_wire())
    assert type(back) is WireTask
    assert back == sig  # full fidelity: id, trace_id, timestamp, domain fields


def test_json_roundtrip():
    sig = WireResult(output="done", score=0.9, correlation_id="abc")
    restored = Signal.from_json(sig.to_json())
    assert type(restored) is WireResult
    assert restored == sig


def test_from_json_accepts_bytes():
    sig = WireTask(task="bytes")
    raw = sig.to_json().encode("utf-8")
    assert Signal.from_json(raw) == sig


def test_metadata_and_lineage_survive():
    parent = WireTask(task="root")
    child = parent.child(task="leaf").with_metadata(region="us-east")
    back = Signal.from_wire(child.to_wire())
    assert back.parent_id == parent.id
    assert back.trace_id == parent.trace_id
    assert back.metadata["region"] == "us-east"


def test_from_wire_is_classmethod_independent_of_caller():
    """from_wire returns the registered type regardless of which class calls it."""
    wire = WireResult(output="z").to_wire()
    assert type(WireTask.from_wire(wire)) is WireResult
    assert type(Signal.from_wire(wire)) is WireResult


# --- Registry ---


def test_subclasses_auto_register():
    assert lookup_signal("WireTask") is WireTask
    assert lookup_signal("WireResult") is WireResult


def test_base_and_builtins_registered():
    reg = registered_signals()
    assert reg["Signal"] is Signal
    # Tool protocol signals self-register on import of the package.
    assert "ToolCallSignal" in reg
    assert "ToolResultSignal" in reg


def test_registered_signals_returns_copy():
    reg = registered_signals()
    reg["Bogus"] = WireTask
    assert "Bogus" not in registered_signals()


def test_explicit_wire_name_via_class_attr():
    class Versioned(Signal):
        __signal_type__ = "task.v2"
        payload: str

    assert Versioned.wire_type() == "task.v2"
    assert lookup_signal("task.v2") is Versioned
    wire = Versioned(payload="p").to_wire()
    assert wire["type"] == "task.v2"
    assert type(Signal.from_wire(wire)) is Versioned


def test_signal_type_not_inherited():
    """A subclass must not silently inherit a parent's pinned wire name."""

    class Parent(Signal):
        __signal_type__ = "pinned.parent"

    class Child(Parent):
        pass

    assert Parent.wire_type() == "pinned.parent"
    assert Child.wire_type() == "Child"


def test_register_decorator_with_custom_name():
    @register_signal(name="custom.alias")
    class Aliased(Signal):
        value: int = 0

    assert lookup_signal("custom.alias") is Aliased


def test_collision_without_override_raises():
    register_signal(WireTask, name="collide.me")
    with pytest.raises(SignalSerializationError):
        register_signal(WireResult, name="collide.me")


def test_collision_with_override_succeeds():
    register_signal(WireTask, name="override.me")
    register_signal(WireResult, name="override.me", override=True)
    assert lookup_signal("override.me") is WireResult


def test_reregistering_same_class_is_idempotent():
    register_signal(WireTask, name="idem.me")
    register_signal(WireTask, name="idem.me")  # no raise
    assert lookup_signal("idem.me") is WireTask


# --- Unknown types ---


def test_unknown_type_strict_raises():
    wire = {"sgp": WIRE_VERSION, "type": "GhostSignal", "data": {"x": 1}}
    with pytest.raises(UnknownSignalType) as exc:
        Signal.from_wire(wire)
    assert exc.value.type_name == "GhostSignal"


def test_unknown_type_nonstrict_preserves_data():
    wire = {
        "sgp": WIRE_VERSION,
        "type": "GhostSignal",
        "data": {"source": "ghost", "priority": 3, "spooky": True, "n": [1, 2]},
    }
    sig = Signal.from_wire(wire, strict=False)
    assert type(sig) is Signal
    assert sig.source == "ghost" and sig.priority == 3
    assert sig.metadata["_sgp_type"] == "GhostSignal"
    assert sig.metadata["_sgp_unmapped"] == {"spooky": True, "n": [1, 2]}


# --- Malformed envelopes ---


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        {"type": "WireTask", "data": {}},  # missing version
        {"sgp": 999, "type": "WireTask", "data": {}},  # wrong version
        {"sgp": WIRE_VERSION, "data": {}},  # missing type
        {"sgp": WIRE_VERSION, "type": "WireTask"},  # missing data
        {"sgp": WIRE_VERSION, "type": "WireTask", "data": "nope"},  # data not dict
    ],
)
def test_malformed_envelope_raises(bad):
    with pytest.raises(SignalSerializationError):
        from_wire(bad)


def test_invalid_payload_for_known_type_raises_serialization_error():
    # WireTask.task is required; omit it.
    wire = {"sgp": WIRE_VERSION, "type": "WireTask", "data": {"urgency": 1}}
    with pytest.raises(SignalSerializationError):
        Signal.from_wire(wire)


# --- Durable dead-letter recovery ---


def test_dlq_persist_and_reload_roundtrip(tmp_path):
    from signal_gating import DeadLetterQueue

    dlq = DeadLetterQueue()
    dlq.add(WireTask(task="a", urgency=1), "handler_error", "worker")
    dlq.add(WireResult(output="b"), "gate_rejected", "worker")

    path = tmp_path / "dlq.jsonl"
    assert dlq.to_jsonl(path) == 2

    fresh = DeadLetterQueue()
    assert fresh.load_jsonl(path) == 2
    sigs = fresh.signals
    assert [type(s).__name__ for s in sigs] == ["WireTask", "WireResult"]
    assert sigs[0].task == "a" and sigs[0].urgency == 1
    # Failure context survives alongside the signal.
    assert fresh.entries[0]["reason"] == "handler_error"
    assert fresh.entries[1]["reason"] == "gate_rejected"


def test_dlq_persisted_file_is_valid_jsonl(tmp_path):
    from signal_gating import DeadLetterQueue

    dlq = DeadLetterQueue()
    dlq.add(WireTask(task="x"), "handler_error", "w")
    path = tmp_path / "dlq.jsonl"
    dlq.to_jsonl(path)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["entry"]["reason"] == "handler_error"
    assert rec["signal"]["type"] == "WireTask"


def test_dlq_load_strict_unknown_raises(tmp_path):
    from signal_gating import DeadLetterQueue

    path = tmp_path / "dlq.jsonl"
    rec = {"entry": {"reason": "x"}, "signal": {"sgp": WIRE_VERSION, "type": "Gone", "data": {}}}
    path.write_text(json.dumps(rec) + "\n")
    dlq = DeadLetterQueue()
    with pytest.raises(UnknownSignalType):
        dlq.load_jsonl(path)


def test_dlq_load_nonstrict_tolerates_unknown(tmp_path):
    from signal_gating import DeadLetterQueue

    path = tmp_path / "dlq.jsonl"
    rec = {
        "entry": {"reason": "x"},
        "signal": {"sgp": WIRE_VERSION, "type": "Gone", "data": {"foo": 1}},
    }
    path.write_text(json.dumps(rec) + "\n")
    dlq = DeadLetterQueue()
    assert dlq.load_jsonl(path, strict=False) == 1
    assert dlq.signals[0].metadata["_sgp_type"] == "Gone"


def test_dlq_load_respects_max_size(tmp_path):
    from signal_gating import DeadLetterQueue

    path = tmp_path / "dlq.jsonl"
    with path.open("w") as f:
        for i in range(5):
            rec = {"entry": {"reason": "r"}, "signal": WireTask(task=str(i)).to_wire()}
            f.write(json.dumps(rec) + "\n")
    dlq = DeadLetterQueue(max_size=3)
    dlq.load_jsonl(path)
    assert dlq.count == 3
    # Most recent kept.
    assert [s.task for s in dlq.signals] == ["2", "3", "4"]


def test_dlq_load_skips_blank_lines(tmp_path):
    from signal_gating import DeadLetterQueue

    path = tmp_path / "dlq.jsonl"
    rec = {"entry": {}, "signal": WireTask(task="solo").to_wire()}
    path.write_text(json.dumps(rec) + "\n\n  \n")
    dlq = DeadLetterQueue()
    assert dlq.load_jsonl(path) == 1


async def test_end_to_end_persist_reload_replay(tmp_path):
    """Failures persist across a simulated restart, then replay as real types."""

    class FlakySignal(Signal):
        payload: str

    failing = Agent("worker")

    @failing.on(FlakySignal)
    async def boom(signal: FlakySignal) -> None:
        raise RuntimeError("kaboom")

    mesh = Mesh([failing])
    async with mesh:
        await failing.inbox.send(FlakySignal(payload="one"))
        await failing.inbox.send(FlakySignal(payload="two"))
        await asyncio.sleep(0.05)

    assert failing.dead_letters.count == 2

    # Persist (shutdown), then reload into a brand-new queue (restart).
    path = tmp_path / "dlq.jsonl"
    failing.dead_letters.to_jsonl(path)

    from signal_gating import DeadLetterQueue

    recovered = DeadLetterQueue()
    recovered.load_jsonl(path)
    assert [type(s).__name__ for s in recovered.signals] == ["FlakySignal", "FlakySignal"]

    # Replay into a channel — signals come back as their original type.
    channel: Channel[Signal] = Channel(Signal)
    n = await recovered.replay(channel)
    assert n == 2
    first = await channel.receive()
    assert isinstance(first, FlakySignal)
    assert first.payload == "one"
