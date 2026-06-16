"""Regression tests for correctness defects found in the concurrency/contract audit.

Each test pins a specific bug fixed in the audit pass so it cannot silently
regress. Grouped by module.
"""

import asyncio

import pytest

from signal_gating import (
    Agent,
    AgentContext,
    Channel,
    Gate,
    Mesh,
    PriorityChannel,
    Signal,
    TaskAssigned,
    TaskResult,
    Team,
    TrajectoryRecorder,
)
from signal_gating.errors import ChannelClosed, SignalSerializationError
from signal_gating.llm import _param_schema
from signal_gating.script import Script
from signal_gating.taskboard import TaskBoard


class _Msg(Signal):
    n: int = 0


# --- signal.py: frozen Signal must be hashable -----------------------------------


def test_signal_is_hashable():
    s = Signal()
    assert isinstance(hash(s), int)
    assert s in {s}


def test_distinct_signals_coexist_in_a_set():
    a, b = Signal(), Signal()
    assert len({a, b}) == 2


def test_signal_with_metadata_is_hashable():
    s = _Msg(n=1).with_metadata(region="us-east")
    # Metadata is the unhashable MappingProxyType; hashing must still work.
    assert s in {s}


# --- gate.py: operator type guards -----------------------------------------------


def test_gate_or_returns_notimplemented_for_non_gate():
    g = Gate.passthrough()
    assert g.__or__(5) is NotImplemented
    with pytest.raises(TypeError):
        _ = g | 5


def test_gate_and_returns_notimplemented_for_non_gate():
    g = Gate.passthrough()
    assert g.__and__("x") is NotImplemented
    with pytest.raises(TypeError):
        _ = g & "x"


# --- gate.py: parallel race cancels and awaits losers cleanly --------------------


async def test_parallel_race_completes_without_pending_warning():
    async def slow_fn(s):
        await asyncio.sleep(0.2)
        return s

    fast = Gate.transform(lambda s: s)
    slow_gate = Gate(slow_fn, name="slow")
    race = Gate.parallel(fast, slow_gate, mode="race")
    result = await race.process(_Msg(n=1))
    assert result is not None
    # Give any improperly-abandoned task a chance to surface a warning.
    await asyncio.sleep(0.25)


# --- channel.py: send_wait must wake a blocked producer on close -----------------


async def test_channel_send_wait_wakes_on_close():
    ch: Channel = Channel(buffer_size=1)
    await ch.send(_Msg(n=1))  # fill the buffer

    async def producer():
        with pytest.raises(ChannelClosed):
            await ch.send_wait(_Msg(n=2))

    task = asyncio.create_task(producer())
    await asyncio.sleep(0.05)  # let the producer block on a full buffer
    ch.close()
    await asyncio.wait_for(task, timeout=1.0)


async def test_priority_channel_send_wait_wakes_on_close():
    ch: PriorityChannel = PriorityChannel(buffer_size=1)
    await ch.send(_Msg(n=1))  # fill the buffer

    async def producer():
        with pytest.raises(ChannelClosed):
            await ch.send_wait(_Msg(n=2))

    task = asyncio.create_task(producer())
    await asyncio.sleep(0.05)
    ch.close()
    await asyncio.wait_for(task, timeout=1.0)


# --- agent.py: request/response resolves before inbox gates ----------------------


async def test_gated_requester_still_receives_reply():
    requester = Agent("req", gates=[Gate.by_priority(5)])
    responder = Agent("resp")

    @responder.on(_Msg)
    async def handle(signal: _Msg, ctx: AgentContext):
        # Reply with a LOW priority that the requester's by_priority(5) gate
        # would otherwise reject, dead-lettering the reply.
        await ctx.reply(_Msg(n=signal.n + 1, priority=0))

    mesh = Mesh([requester, responder])
    mesh.connect(requester, responder)
    mesh.connect(responder, requester)  # return path

    async with mesh:
        reply = await requester.request(_Msg(n=1, priority=10), timeout=2.0)
    assert reply.n == 2


# --- mesh.py: interceptors run on content-routed and load-balanced edges ---------


async def test_content_route_applies_interceptors():
    a, b = Agent("a"), Agent("b")
    received: list[int] = []

    @b.on(_Msg)
    async def handle(signal: _Msg):
        received.append(signal.n)

    seen: list[tuple[str, str]] = []
    mesh = Mesh([a, b])
    mesh.intercept(lambda sig, src, tgt: (seen.append((src, tgt)), sig)[1])
    mesh.route(a, [(lambda s: True, b)])

    async with mesh:
        await a.emit(_Msg(n=1))
        await asyncio.sleep(0.05)

    assert ("a", "b") in seen
    assert received == [1]


async def test_content_route_interceptor_can_block():
    a, b = Agent("a"), Agent("b")
    received: list[int] = []

    @b.on(_Msg)
    async def handle(signal: _Msg):
        received.append(signal.n)

    mesh = Mesh([a, b])
    mesh.intercept(lambda sig, src, tgt: None)  # block everything
    mesh.route(a, [(lambda s: True, b)])

    async with mesh:
        await a.emit(_Msg(n=1))
        await asyncio.sleep(0.05)

    assert received == []


async def test_load_balance_applies_interceptors():
    a, b = Agent("a"), Agent("b")
    received: list[int] = []

    @b.on(_Msg)
    async def handle(signal: _Msg):
        received.append(signal.n)

    seen: list[tuple[str, str]] = []
    mesh = Mesh([a, b])
    mesh.intercept(lambda sig, src, tgt: (seen.append((src, tgt)), sig)[1])
    mesh.load_balance(a, [b])

    async with mesh:
        await a.emit(_Msg(n=1))
        await asyncio.sleep(0.05)

    assert ("a", "b") in seen
    assert received == [1]


# --- mesh.py: scatter with no targets is a clean empty result --------------------


async def test_scatter_empty_targets_returns_empty():
    mesh = Mesh([Agent("a")])
    async with mesh:
        assert await mesh.scatter(_Msg(n=1), []) == []


# --- mesh.py: start() is exception-safe ------------------------------------------


async def test_mesh_start_failure_stops_started_agents():
    good = Agent("good")
    bad = Agent("bad")

    @bad.on_start
    async def boom():
        raise RuntimeError("startup failed")

    mesh = Mesh([good, bad])
    with pytest.raises(Exception):
        await mesh.start()

    assert not good.running
    assert not mesh._running


# --- trajectory.py: load_jsonl reports the offending line ------------------------


def test_load_jsonl_malformed_line_is_diagnosable(tmp_path):
    bad = tmp_path / "trajectory.jsonl"
    bad.write_text("not valid json\n")
    rec = TrajectoryRecorder()
    with pytest.raises(SignalSerializationError) as excinfo:
        rec.load_jsonl(bad)
    assert "trajectory.jsonl:1" in str(excinfo.value)


# --- taskboard.py: unknown task ids raise ValueError, not KeyError ---------------


async def test_complete_unknown_task_raises_valueerror():
    board = TaskBoard("b")
    with pytest.raises(ValueError):
        await board.complete("nonexistent", "member")


async def test_release_unknown_task_raises_valueerror():
    board = TaskBoard("b")
    with pytest.raises(ValueError):
        await board.release("nonexistent", "member")


# --- script.py: fan_out with no targets raises a clear error ---------------------


async def test_fan_out_empty_targets_raises():
    mesh = Mesh([Agent("w")])

    async def flow(ctx):
        return await ctx.fan_out([], [_Msg(n=1)])

    async with mesh:
        with pytest.raises(ValueError):
            await Script("s", mesh, flow).run()


# --- team.py: a cancelling shutdown releases the in-flight task ------------------


async def test_shutdown_releases_in_flight_task():
    worker = Agent("w")

    @worker.on(TaskAssigned)
    async def work(signal: TaskAssigned, ctx: AgentContext):
        await asyncio.sleep(5.0)  # still running when shutdown's timeout fires
        await ctx.reply(TaskResult(task_id=signal.task_id, result={}))

    mesh = Mesh([worker])
    team = Team("t", mesh, task_timeout=10.0)
    team.enroll(worker)

    async with mesh:
        async with team:
            tid = await team.open("slow")
            for _ in range(100):
                await asyncio.sleep(0.02)
                if team.board.task(tid).status == "in_progress":
                    break
            assert team.board.task(tid).status == "in_progress"

            await team.shutdown("w", timeout=0.2)
            # Released, not stranded in_progress forever.
            assert team.board.task(tid).status == "pending"


# --- llm.py: container tool params get correct JSON-Schema types -----------------


def test_param_schema_maps_containers():
    assert _param_schema("list") == {"type": "array", "items": {}}
    assert _param_schema("dict") == {"type": "object"}
    assert _param_schema("str") == {"type": "string"}
    assert _param_schema("int") == {"type": "integer"}
    # Unknown/custom types fall back to string.
    assert _param_schema("MyModel") == {"type": "string"}
