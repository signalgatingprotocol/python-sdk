import asyncio
import json

import pytest

from signal_gating import Gate, Signal
from signal_gating.errors import (
    BudgetExceeded,
    SignalSerializationError,
    TaskRejected,
    TeamError,
)
from signal_gating.taskboard import TaskBoard, TaskOpened
from signal_gating.trajectory import domain_payload


def test_error_hierarchy():
    err = TaskRejected("t1", "no_empty_results")
    assert err.task_id == "t1" and err.gate_name == "no_empty_results"
    assert BudgetExceeded(1000, "k").budget == 1000
    assert issubclass(TeamError, Exception)


def test_domain_payload_excludes_envelope():
    class Probe(Signal):
        text: str = ""

    assert domain_payload(Probe(text="x", priority=9)) == {"text": "x"}


async def test_lifecycle_and_dependencies():
    board = TaskBoard("t")
    first = await board.open("first")
    second = await board.open("second", depends_on=(first,))
    assert [t.id for t in board.claimable()] == [first]

    task = await board.claim("alice")
    assert task is not None and task.id == first
    assert board.task(first).status == "in_progress"
    assert await board.claim("bob") is None          # second is dep-blocked

    await board.complete(first, "alice", result={"ok": True})
    assert board.task(first).status == "completed"
    assert [t.id for t in board.claimable()] == [second]   # unblocked, nothing to do

    await board.claim("bob", task_id=second)
    await board.release(second, "bob", reason="meeting")
    assert board.task(second).status == "pending"


async def test_priority_and_targeted_claim():
    board = TaskBoard("t")
    low = await board.open("low", priority=1)
    high = await board.open("high", priority=9)
    assert (await board.claim("a")).id == high
    assert await board.claim("b", task_id=high) is None    # already claimed
    assert (await board.claim("b", task_id=low)).id == low


async def test_concurrent_claims_never_double_assign():
    board = TaskBoard("t")
    for i in range(5):
        await board.open(f"task-{i}")
    winners = await asyncio.gather(*(board.claim(f"m{i}") for i in range(10)))
    claimed = [t.id for t in winners if t is not None]
    assert len(claimed) == 5 and len(set(claimed)) == 5


async def test_gates_reject_with_task_rejected():
    board = TaskBoard(
        "t",
        complete_gate=Gate(lambda s: s if s.result else None, name="no_empty_results"),
    )
    tid = await board.open("x")
    await board.claim("a", task_id=tid)
    with pytest.raises(TaskRejected) as exc:
        await board.complete(tid, "a", result={})
    assert exc.value.gate_name == "no_empty_results"
    assert board.task(tid).status == "in_progress"


def test_pinned_wire_names():
    assert TaskOpened.wire_type() == "sgp.task.opened"


async def _populated(tmp_path):
    board = TaskBoard("t")
    a = await board.open("a")
    b = await board.open("b", depends_on=(a,))
    await board.claim("alice", task_id=a)
    await board.complete(a, "alice", result={"n": 1})
    await board.claim("bob", task_id=b)
    path = tmp_path / "ledger.jsonl"
    board.export_jsonl(path)
    return board, path, a, b


async def test_jsonl_round_trip_reconstructs_state(tmp_path):
    board, path, a, b = await _populated(tmp_path)
    loaded = TaskBoard.load_jsonl(path, release_in_progress=False)
    assert {t.id: t.status for t in loaded.tasks()} == {a: "completed", b: "in_progress"}
    assert loaded.head_digest == board.head_digest


async def test_release_in_progress_recovery(tmp_path):
    _, path, a, b = await _populated(tmp_path)
    loaded = TaskBoard.load_jsonl(path)        # default True
    assert loaded.task(b).status == "pending"
    last = loaded.events[-1]
    assert last.wire_type() == "sgp.task.released" and last.reason == "recovered"


async def test_chain_breaks_on_edit_reorder_delete(tmp_path):
    _, path, *_ = await _populated(tmp_path)
    lines = path.read_text().splitlines()

    edited = lines.copy()
    record = json.loads(edited[2])
    record["event"]["data"]["member"] = "mallory"
    edited[2] = json.dumps(record, sort_keys=True)

    for mutation in (
        edited,                                        # in-place edit
        [lines[0], lines[2], lines[1], *lines[3:]],    # reorder
        [lines[0], *lines[2:]],                        # interior delete
    ):
        path.write_text("\n".join(mutation) + "\n")
        with pytest.raises(SignalSerializationError):
            TaskBoard.load_jsonl(path)


async def test_observers_are_advisory():
    board = TaskBoard("t")
    seen: list[str] = []

    def observer(event):
        seen.append(event.wire_type())
        raise RuntimeError("boom")

    unsubscribe = board.on_event(observer)
    await board.open("x")
    assert seen == ["sgp.task.opened"] and board.observer_errors == 1
    unsubscribe()
    await board.open("y")
    assert len(seen) == 1


def test_public_exports():
    import signal_gating as sg

    for name in (
        "TaskBoard",
        "Task",
        "Team",
        "Script",
        "ScriptContext",
        "CheckpointStore",
        "TaskRejected",
        "TeamError",
        "BudgetExceeded",
        "domain_payload",
    ):
        assert hasattr(sg, name), name
