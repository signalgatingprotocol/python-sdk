import asyncio
import json

import signal_gating
from signal_gating import Agent, Mesh, Signal
from signal_gating.trajectory import Receipt, TrajectoryRecorder


class Ping(Signal):
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
    assert hasattr(signal_gating, "TrajectoryRecorder")
    assert "Receipt" in signal_gating.__all__
    assert "TrajectoryRecorder" in signal_gating.__all__
