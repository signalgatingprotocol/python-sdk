import asyncio
import json

import pytest

import signal_gating
from signal_gating import Agent, Mesh, MeshError, Signal, SignalSerializationError
from signal_gating.trajectory import (
    Receipt,
    ReplayDelivery,
    TrajectoryRecorder,
    TrajectoryReplayRunner,
)


class Ping(Signal):
    n: int = 0


class StablePing(Signal):
    __signal_type__ = "tests.stable_ping"
    n: int = 0


# ---------------------------------------------------------------------------
# Task 1: Receipt unit tests
# ---------------------------------------------------------------------------


def test_from_signal_extracts_envelope_and_domain_payload():
    sig = Ping(n=7, priority=3)
    r = Receipt.from_signal(sig, source="a", target="b")
    assert r.trace_id == sig.trace_id
    assert r.signal_id == sig.id
    assert r.parent_id == sig.parent_id
    assert r.signal_type == "Ping"
    assert r.source == "a" and r.target == "b"
    assert r.priority == 3
    assert r.payload == {"n": 7}            # domain fields only; no envelope keys
    assert "trace_id" not in r.payload and "source" not in r.payload


def test_from_signal_uses_stable_wire_type_for_audit_projection():
    sig = StablePing(n=9)
    r = Receipt.from_signal(sig, source="a", target="b")
    assert r.signal_type == "tests.stable_ping"
    assert r.wire["type"] == "tests.stable_ping"


def test_digest_verifies_and_detects_tampering():
    r = Receipt.from_signal(Ping(n=1), source="a", target="b")
    assert r.verify() is True
    tampered = Receipt(**{**r.to_dict(), "payload": {"n": 999}})
    assert tampered.verify() is False


def test_to_dict_is_json_serializable():
    r = Receipt.from_signal(Ping(n=1), source="a", target="b")
    d = r.to_dict()
    assert json.loads(json.dumps(d, default=str))["signal_type"] == "Ping"
    assert d["digest"] == r.digest


# ---------------------------------------------------------------------------
# Task 2: TrajectoryRecorder integration tests
# ---------------------------------------------------------------------------


async def _run_relay_mesh(recorder: TrajectoryRecorder) -> tuple[Ping, list[Ping]]:
    """seed -> a -> b -> c; relays thread lineage via child()."""
    a, b, c = Agent("a"), Agent("b"), Agent("c")
    seen: list[Ping] = []
    done = asyncio.Event()

    @a.on(Ping)
    async def a_relay(sig: Ping) -> None:
        await a.emit(sig.child(n=sig.n + 1))

    @b.on(Ping)
    async def b_relay(sig: Ping) -> None:
        await b.emit(sig.child(n=sig.n + 1))

    @c.on(Ping)
    async def c_sink(sig: Ping) -> None:
        seen.append(sig)
        done.set()

    mesh = Mesh([a, b, c])
    mesh.intercept(recorder)
    mesh.connect(a, b)
    mesh.connect(b, c)

    seed = Ping(n=0)
    async with mesh:
        await mesh.inject(a, seed)
        await asyncio.wait_for(done.wait(), timeout=3.0)
    return seed, seen


async def test_recorder_captures_each_hop():
    recorder = TrajectoryRecorder()
    seed, seen = await _run_relay_mesh(recorder)
    assert len(seen) == 1
    rs = recorder.receipts
    assert len(rs) == 2                       # a->b and b->c (seed inject is not a hop)
    assert (rs[0].source, rs[0].target) == ("a", "b")
    assert (rs[1].source, rs[1].target) == ("b", "c")
    assert rs[0].payload == {"n": 1}
    assert rs[1].payload == {"n": 2}
    assert all(r.signal_type == "Ping" for r in rs)


async def test_trajectories_group_by_trace_and_chain_lineage():
    recorder = TrajectoryRecorder()
    seed, _ = await _run_relay_mesh(recorder)
    traj = recorder.trajectories()
    assert list(traj.keys()) == [seed.trace_id]      # one run
    run = traj[seed.trace_id]
    assert len(run) == 2
    assert run[0].parent_id == seed.id               # first hop descends from the seed
    assert run[1].parent_id == run[0].signal_id      # lineage chains hop-to-hop


async def test_all_receipts_verify():
    recorder = TrajectoryRecorder()
    await _run_relay_mesh(recorder)
    assert all(r.verify() for r in recorder.receipts)


async def test_export_jsonl_round_trips(tmp_path) -> None:
    recorder = TrajectoryRecorder()
    await _run_relay_mesh(recorder)
    out = tmp_path / "runs.jsonl"
    n = recorder.export_jsonl(out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert n == len(lines) == 2
    first = json.loads(lines[0])
    assert first["source"] == "a" and first["payload"] == {"n": 1}


async def test_recorder_is_pure_observer():
    """Attaching a recorder must not drop signals."""
    recorder = TrajectoryRecorder()
    _, seen = await _run_relay_mesh(recorder)
    assert len(seen) == 1                            # delivery unaffected


async def test_mesh_record_captures_connected_route_without_interceptor():
    recorder = TrajectoryRecorder()
    a, b = Agent("a"), Agent("b")
    done = asyncio.Event()

    @b.on(Ping)
    async def b_sink(_sig: Ping) -> None:
        done.set()

    mesh = Mesh([a, b])
    mesh.record(recorder)
    mesh.connect(a, b)

    async with mesh:
        await a.emit(Ping(n=3))
        await asyncio.wait_for(done.wait(), timeout=3.0)

    assert len(recorder.receipts) == 1
    receipt = recorder.receipts[0]
    assert (receipt.source, receipt.target) == ("a", "b")
    assert receipt.action == "routed"
    assert receipt.event_kind == "signal"
    assert receipt.payload == {"n": 3}


async def test_workflow_without_connect_records_mesh_requests_and_replies():
    recorder = TrajectoryRecorder()
    first, second = Agent("first"), Agent("second")

    @first.on(Ping)
    async def handle_first(sig: Ping) -> None:
        await first.reply(sig, sig.child(n=sig.n + 1))

    @second.on(Ping)
    async def handle_second(sig: Ping) -> None:
        await second.reply(sig, sig.child(n=sig.n + 1))

    mesh = Mesh([first, second])
    mesh.record(recorder)
    seed = Ping(n=0)

    async with mesh:
        result = await mesh.workflow(seed, [first, second])

    assert result.n == 2
    assert mesh.edges == []
    actions = [r.action for r in recorder.receipts]
    assert actions == [
        "workflow_step_start",
        "request_sent",
        "response_received",
        "workflow_step_complete",
        "workflow_step_start",
        "request_sent",
        "response_received",
        "workflow_step_complete",
    ]
    assert all(r.trace_id == seed.trace_id for r in recorder.receipts)
    assert recorder.receipts[1].source == "mesh"
    assert recorder.receipts[1].target == "first"
    assert recorder.receipts[2].source == "first"
    assert recorder.receipts[2].target == "mesh"
    assert recorder.receipts[2].parent_id == recorder.receipts[1].signal_id


async def test_call_tool_records_tool_call_and_result_without_connect():
    recorder = TrajectoryRecorder()
    worker = Agent("worker")

    @worker.tool(description="Increment a value")
    async def inc(n: int) -> int:
        return n + 1

    mesh = Mesh([worker])
    mesh.record(recorder)

    async with mesh:
        result = await mesh.call_tool(worker, "inc", n=4)

    assert result == 5
    actions = [r.action for r in recorder.receipts]
    assert actions == [
        "tool_call_start",
        "request_sent",
        "response_received",
        "tool_call_complete",
    ]
    assert recorder.receipts[0].signal_type == "ToolCallSignal"
    assert recorder.receipts[0].metadata["tool_name"] == "inc"
    assert recorder.receipts[-1].signal_type == "ToolResultSignal"
    assert recorder.receipts[-1].metadata["error"] is False


async def test_scatter_records_fanout_and_all_replies_once():
    recorder = TrajectoryRecorder()
    agents = [Agent("a"), Agent("b"), Agent("c")]

    for agent in agents:
        @agent.on(Ping)
        async def handle(sig: Ping, current: Agent = agent) -> None:
            await current.reply(sig, sig.child(n=sig.n + 1))

    mesh = Mesh(agents)
    mesh.record(recorder)

    async with mesh:
        responses = await mesh.scatter(Ping(n=0), agents)

    assert [r.n for r in responses] == [1, 1, 1]  # type: ignore[attr-defined]
    sends = [r for r in recorder.receipts if r.action == "scatter_sent"]
    replies = [r for r in recorder.receipts if r.action == "scatter_response"]
    assert [r.target for r in sends] == ["a", "b", "c"]
    assert sorted(r.source for r in replies) == ["a", "b", "c"]
    assert len(sends) == len(replies) == 3
    assert len({r.metadata["correlation_id"] for r in sends}) == 3


async def test_race_records_sends_and_single_winner():
    recorder = TrajectoryRecorder()
    fast, slow = Agent("fast"), Agent("slow")

    @fast.on(Ping)
    async def fast_handle(sig: Ping) -> None:
        await fast.reply(sig, sig.child(n=1))

    @slow.on(Ping)
    async def slow_handle(sig: Ping) -> None:
        await asyncio.sleep(0.1)
        await slow.reply(sig, sig.child(n=2))

    mesh = Mesh([fast, slow])
    mesh.record(recorder)

    async with mesh:
        result = await mesh.race(Ping(n=0), [fast, slow])

    assert result.n == 1  # type: ignore[attr-defined]
    sends = [r for r in recorder.receipts if r.action == "race_sent"]
    replies = [r for r in recorder.receipts if r.action == "race_response"]
    winners = [r for r in recorder.receipts if r.action == "race_winner"]
    assert [r.target for r in sends] == ["fast", "slow"]
    assert [r.source for r in replies] == ["fast"]
    assert len(winners) == 1


async def test_publish_and_inject_record_direct_deliveries():
    recorder = TrajectoryRecorder()
    sub_a, sub_b = Agent("sub_a"), Agent("sub_b")
    worker = Agent("worker")

    mesh = Mesh([sub_a, sub_b, worker])
    mesh.record(recorder)
    mesh.create_topic("events")
    mesh.subscribe(sub_a, "events")
    mesh.subscribe(sub_b, "events")

    async with mesh:
        count = await mesh.publish("events", Ping(n=1))
        await mesh.inject(worker, Ping(n=2))

    assert count == 2
    actions = [r.action for r in recorder.receipts]
    assert actions == ["published", "published", "inject"]
    assert [r.target for r in recorder.receipts] == ["sub_a", "sub_b", "worker"]
    assert recorder.receipts[0].metadata["topic"] == "events"


async def test_mesh_record_sink_failure_does_not_block_delivery():
    def broken_sink(_event: object) -> None:
        raise RuntimeError("offline")

    worker = Agent("worker")
    seen: list[int] = []

    @worker.on(Ping)
    async def handle(sig: Ping) -> None:
        seen.append(sig.n)

    mesh = Mesh([worker])
    mesh.record(broken_sink)

    async with mesh:
        await mesh.inject(worker, Ping(n=5))
        await asyncio.sleep(0.05)

    assert seen == [5]
    assert mesh.event_sink_errors == 1


def test_clear_empties_receipts():
    recorder = TrajectoryRecorder()
    recorder._receipts.append(Receipt.from_signal(Ping(n=1), "a", "b"))  # type: ignore[attr-defined]
    recorder.clear()
    assert recorder.receipts == []


# ---------------------------------------------------------------------------
# Task 3: Export tests
# ---------------------------------------------------------------------------


def test_exports():
    assert hasattr(signal_gating, "Receipt")
    assert hasattr(signal_gating, "ReplayDelivery")
    assert hasattr(signal_gating, "TrajectoryRecorder")
    assert hasattr(signal_gating, "TrajectoryReplayRunner")
    assert "Receipt" in signal_gating.__all__
    assert "ReplayDelivery" in signal_gating.__all__
    assert "TrajectoryRecorder" in signal_gating.__all__
    assert "TrajectoryReplayRunner" in signal_gating.__all__


# ---------------------------------------------------------------------------
# Task 4: Faithful replay via the wire format
# ---------------------------------------------------------------------------


def test_receipt_carries_wire_and_reconstructs_typed_signal():
    sig = Ping(n=7, priority=3)
    r = Receipt.from_signal(sig, source="a", target="b")
    # The audit projection is unchanged...
    assert r.payload == {"n": 7}
    # ...and the wire envelope is the faithful, self-describing form.
    assert r.wire == {"sgp": 1, "type": "Ping", "data": sig.model_dump(mode="json")}

    back = r.to_signal()
    assert isinstance(back, Ping)
    assert back.n == 7
    assert back.id == sig.id                  # identity preserved, not just shape
    assert back.trace_id == sig.trace_id
    assert back.priority == 3
    assert back == sig


def test_to_signal_preserves_full_lineage():
    parent = Ping(n=0)
    child = parent.child(n=1, priority=5)
    r = Receipt.from_signal(child, source="a", target="b")
    back = r.to_signal()
    assert back.parent_id == parent.id
    assert back.trace_id == parent.trace_id
    assert back.id == child.id


def test_digest_covers_wire_envelope():
    r = Receipt.from_signal(Ping(n=1), source="a", target="b")
    assert r.verify() is True
    tampered_wire = {**r.wire, "data": {**r.wire["data"], "n": 999}}
    tampered = Receipt(**{**r.to_dict(), "wire": tampered_wire})
    assert tampered.verify() is False         # tampering with replay data is caught


def test_from_dict_round_trips_and_stays_verifiable():
    r = Receipt.from_signal(Ping(n=4, priority=2), source="x", target="y")
    rebuilt = Receipt.from_dict(r.to_dict())
    assert rebuilt == r
    assert rebuilt.verify() is True
    assert rebuilt.to_signal() == Ping.from_wire(r.wire)


async def test_export_then_load_replays_typed_signals(tmp_path) -> None:
    recorder = TrajectoryRecorder()
    await _run_relay_mesh(recorder)
    out = tmp_path / "runs.jsonl"
    recorder.export_jsonl(out)

    # A fresh recorder, as if after a process restart, reloads and replays.
    reloaded = TrajectoryRecorder()
    n = reloaded.load_jsonl(out)
    assert n == 2
    assert all(r.verify() for r in reloaded.receipts)   # digests survive the round-trip

    signals = reloaded.replay()
    assert [type(s).__name__ for s in signals] == ["Ping", "Ping"]
    assert [s.n for s in signals] == [1, 2]             # type: ignore[attr-defined]
    # Reconstructed signals match the originally captured ones, identity and all.
    assert signals == [r.to_signal() for r in recorder.receipts]


async def test_replay_runner_replays_inject_into_fresh_mesh() -> None:
    recorder = TrajectoryRecorder()
    original = Agent("worker")

    mesh = Mesh([original])
    mesh.record(recorder)
    async with mesh:
        await mesh.inject(original, Ping(n=7))

    seen: list[Ping] = []
    replayed = Agent("worker")

    @replayed.on(Ping)
    async def handle(sig: Ping) -> None:
        seen.append(sig)

    replay_mesh = Mesh([replayed])
    replay_recorder = TrajectoryRecorder()
    replay_mesh.record(replay_recorder)
    runner = TrajectoryReplayRunner.from_recorder(recorder)

    async with replay_mesh:
        result = await runner.replay_into(replay_mesh)
        await asyncio.sleep(0.05)

    assert result.attempted == 1
    assert result.delivered == 1
    assert result.skipped == 0
    assert result.failed == 0
    assert result.receipts == [recorder.receipts[0]]
    assert result.deliveries == [
        ReplayDelivery(
            receipt_index=0,
            action="inject",
            trace_id=recorder.receipts[0].trace_id,
            signal_id=recorder.receipts[0].signal_id,
            signal_type="Ping",
            target="worker",
            status="delivered",
        )
    ]
    assert result.actions == {"inject": 1}
    assert result.trace_ids == (recorder.receipts[0].trace_id,)
    assert len(seen) == 1
    assert seen[0].n == 7
    assert seen[0].id == recorder.receipts[0].signal_id
    assert seen[0].trace_id == recorder.receipts[0].trace_id
    assert [r.action for r in replay_recorder.receipts] == ["replay_delivered"]
    assert replay_recorder.receipts[0].metadata["original_action"] == "inject"


async def test_replay_runner_delivers_request_entries_without_session_resume() -> None:
    recorder = TrajectoryRecorder()
    original = Agent("worker")

    @original.on(Ping)
    async def original_handle(sig: Ping) -> None:
        await original.reply(sig, sig.child(n=sig.n + 1))

    mesh = Mesh([original])
    mesh.record(recorder)
    async with mesh:
        response = await mesh.request(original, Ping(n=2))

    assert response.n == 3  # type: ignore[attr-defined]

    seen: list[Ping] = []
    replies: list[Ping] = []
    replayed = Agent("worker")

    @replayed.on(Ping)
    async def replay_handle(sig: Ping) -> None:
        seen.append(sig)
        response = sig.child(n=sig.n + 1).evolve(correlation_id=sig.correlation_id)
        replies.append(response)

    replay_mesh = Mesh([replayed])
    runner = TrajectoryReplayRunner.from_recorder(recorder)

    async with replay_mesh:
        result = await runner.replay_into(replay_mesh)
        await asyncio.sleep(0.05)

    assert result.attempted == 1
    assert result.delivered == 1
    assert result.skipped == 1  # response_received is audit data, not an entry
    assert len(seen) == 1
    assert seen[0].n == 2
    assert seen[0].correlation_id == recorder.receipts[0].metadata["correlation_id"]
    assert replies[0].correlation_id == seen[0].correlation_id
    assert replies[0].trace_id == seen[0].trace_id
    assert replies[0].parent_id == seen[0].id


async def test_replay_runner_skips_non_entry_events() -> None:
    recorder = TrajectoryRecorder()
    worker = Agent("worker")
    mesh = Mesh([worker])
    mesh.record(recorder)
    async with mesh:
        await mesh.inject(worker, Ping(n=1))

    runner = TrajectoryReplayRunner(
        [
            *recorder.receipts,
            Receipt.from_signal(
                Ping(n=2),
                source="worker",
                target="mesh",
                action="response_received",
            ),
        ]
    )

    assert [r.action for r in runner.replayable_receipts()] == ["inject"]


async def test_replay_runner_verifies_before_delivery() -> None:
    valid = Receipt.from_signal(Ping(n=0), source="mesh", target="worker", action="inject")
    receipt = Receipt.from_signal(Ping(n=1), source="mesh", target="worker", action="inject")
    tampered = Receipt(**{**receipt.to_dict(), "payload": {"n": 999}})
    worker = Agent("worker")
    seen: list[int] = []

    @worker.on(Ping)
    async def handle(sig: Ping) -> None:
        seen.append(sig.n)

    mesh = Mesh([worker])
    runner = TrajectoryReplayRunner([valid, tampered])

    async with mesh:
        with pytest.raises(SignalSerializationError, match="receipt digest mismatch"):
            await runner.replay_into(mesh)
        await asyncio.sleep(0.05)

    assert seen == []


async def test_replay_runner_missing_target_policy() -> None:
    receipt = Receipt.from_signal(Ping(n=1), source="mesh", target="missing", action="inject")
    present = Receipt.from_signal(Ping(n=2), source="mesh", target="present", action="inject")
    runner = TrajectoryReplayRunner([receipt, present])
    present_agent = Agent("present")
    seen: list[int] = []

    @present_agent.on(Ping)
    async def handle(sig: Ping) -> None:
        seen.append(sig.n)

    mesh = Mesh([present_agent])

    async with mesh:
        with pytest.raises(MeshError):
            await runner.replay_into(mesh)
        result = await runner.replay_into(mesh, strict_targets=False)
        await asyncio.sleep(0.05)

    assert result.attempted == 2
    assert result.delivered == 1
    assert result.skipped == 1
    assert result.failed == 1
    assert result.missing_targets == ["missing"]
    assert seen == [2]


async def test_replay_runner_loads_from_jsonl(tmp_path) -> None:
    recorder = TrajectoryRecorder()
    worker = Agent("worker")
    mesh = Mesh([worker])
    mesh.record(recorder)
    async with mesh:
        await mesh.inject(worker, Ping(n=8))

    out = tmp_path / "runs.jsonl"
    recorder.export_jsonl(out)

    seen: list[int] = []
    replayed = Agent("worker")

    @replayed.on(Ping)
    async def handle(sig: Ping) -> None:
        seen.append(sig.n)

    replay_mesh = Mesh([replayed])
    runner = TrajectoryReplayRunner.from_jsonl(out)

    async with replay_mesh:
        result = await runner.replay_into(replay_mesh)
        await asyncio.sleep(0.05)

    assert result.delivered == 1
    assert seen == [8]


async def test_load_jsonl_rejects_tampered_receipt_by_default(tmp_path) -> None:
    recorder = TrajectoryRecorder()
    recorder._receipts.append(Receipt.from_signal(Ping(n=1), "a", "b"))  # type: ignore[attr-defined]
    out = tmp_path / "runs.jsonl"
    recorder.export_jsonl(out)

    line = json.loads(out.read_text(encoding="utf-8"))
    line["wire"]["data"]["n"] = 999
    out.write_text(json.dumps(line) + "\n", encoding="utf-8")

    reloaded = TrajectoryRecorder()
    with pytest.raises(SignalSerializationError, match="receipt digest mismatch"):
        reloaded.load_jsonl(out)
    assert reloaded.receipts == []


async def test_replay_rejects_tampered_receipts_by_default(tmp_path) -> None:
    recorder = TrajectoryRecorder()
    recorder._receipts.append(Receipt.from_signal(Ping(n=1), "a", "b"))  # type: ignore[attr-defined]
    out = tmp_path / "runs.jsonl"
    recorder.export_jsonl(out)

    line = json.loads(out.read_text(encoding="utf-8"))
    line["payload"]["n"] = 999
    out.write_text(json.dumps(line) + "\n", encoding="utf-8")

    reloaded = TrajectoryRecorder()
    assert reloaded.load_jsonl(out, verify=False) == 1
    with pytest.raises(SignalSerializationError, match="receipt digest mismatch"):
        reloaded.replay()
    signals = reloaded.replay(verify=False)
    assert len(signals) == 1
    assert isinstance(signals[0], Ping)
    assert signals[0].n == 1


def test_replay_unknown_type_strict_then_lenient():
    from signal_gating import UnknownSignalType

    # A receipt referencing a type that is not registered in this process.
    bogus = {
        "trace_id": "t",
        "signal_id": "s",
        "parent_id": "",
        "signal_type": "GhostSignal",
        "source": "a",
        "target": "b",
        "priority": 0,
        "timestamp": 0.0,
        "payload": {"ghost": True},
        "wire": {"sgp": 1, "type": "GhostSignal", "data": {"ghost": True}},
        "digest": "",
    }
    r = Receipt.from_dict(bogus)
    try:
        r.to_signal()
    except UnknownSignalType:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("strict replay should reject an unknown type")

    degraded = r.to_signal(strict=False)
    assert isinstance(degraded, Signal)
    assert degraded.metadata["_sgp_type"] == "GhostSignal"
    assert degraded.metadata["_sgp_unmapped"] == {"ghost": True}
