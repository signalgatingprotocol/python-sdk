"""Tests for the Script workflow runtime: checkpoint store, step keys, runtime."""

import asyncio

import pytest

from signal_gating import Agent, AgentContext, Mesh, Signal
from signal_gating.errors import BudgetExceeded, MeshError, SignalSerializationError
from signal_gating.script import CheckpointStore, Script, step_key


class Ping(Signal):
    # Pinned wire name: other test modules define their own `Ping`, and the
    # registry's last-definition-wins would break from_wire on full-suite runs.
    __signal_type__ = "tests.script_ping"
    text: str = ""


def test_step_key_determinism_and_occurrence():
    a = step_key("s", "scan", Ping(text="x"), occurrence=0, target="t")
    same = step_key("s", "scan", Ping(text="x", priority=9), occurrence=0, target="t")
    assert a == same                                  # envelope fields excluded
    assert a != step_key("s", "scan", Ping(text="x"), occurrence=1, target="t")
    assert a != step_key("s", "scan", Ping(text="y"), occurrence=0, target="t")
    assert a != step_key("s", "scan", Ping(text="x"), occurrence=0)  # run vs fan_out


def test_store_round_trip_and_last_wins(tmp_path):
    path = tmp_path / "cp.jsonl"
    store = CheckpointStore(path)
    store.put("k1", "s", "scan", Ping(text="one"))
    store.put("k1", "s", "scan", Ping(text="two"))
    reloaded = CheckpointStore(path)
    assert reloaded.get("k1").text == "two"           # duplicate keys: last wins
    assert reloaded.get("nope") is None


def test_store_detects_tampering(tmp_path):
    path = tmp_path / "cp.jsonl"
    CheckpointStore(path).put("k", "s", "p", Ping(text="x"))
    path.write_text(path.read_text().replace('"x"', '"evil"'))
    with pytest.raises(SignalSerializationError):
        CheckpointStore(path)


class Pong(Signal):
    __signal_type__ = "tests.script_pong"
    text: str = ""


def make_echo(name: str = "echo") -> Agent:
    agent = Agent(name)

    @agent.on(Ping)
    async def handle(signal: Ping, ctx: AgentContext):
        await ctx.reply(Pong(text=signal.text.upper()))

    return agent


async def test_resume_skips_completed_steps(tmp_path):
    async def flow(ctx):
        async with ctx.phase("scan"):
            return [
                (await ctx.run("echo", Ping(text=t))).text for t in ("a", "b", "a")
            ]

    requests: list[object] = []

    def fresh_mesh() -> Mesh:
        mesh = Mesh([make_echo()])
        mesh.record(
            lambda e: requests.append(e) if e.action == "request_sent" else None
        )
        return mesh

    mesh = fresh_mesh()
    async with mesh:
        store = CheckpointStore(tmp_path / "cp.jsonl")
        assert await Script("s", mesh, flow, store=store).run() == ["A", "B", "A"]
    assert len(requests) == 3        # occurrence keys: the duplicate "a" re-executes

    requests.clear()
    mesh = fresh_mesh()
    async with mesh:
        store = CheckpointStore(tmp_path / "cp.jsonl")
        assert await Script("s", mesh, flow, store=store).run() == ["A", "B", "A"]
    assert requests == []            # full cache hit: zero mesh requests


async def test_fan_out_respects_concurrency_and_order():
    gate = asyncio.Event()
    in_flight = 0
    peak = 0
    workers = []
    for i in range(4):
        agent = Agent(f"w{i}")

        @agent.on(Ping)
        async def handle(signal: Ping, ctx: AgentContext):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await gate.wait()
            in_flight -= 1
            await ctx.reply(Pong(text=signal.text))

        workers.append(agent)

    async def flow(ctx):
        async with ctx.phase("p"):
            pending = asyncio.ensure_future(
                ctx.fan_out(
                    [f"w{i}" for i in range(4)],
                    [Ping(text=str(n)) for n in range(8)],
                )
            )
            await asyncio.sleep(0.1)
            gate.set()
            return [r.text for r in await pending]

    mesh = Mesh(workers)
    async with mesh:
        out = await Script("s", mesh, flow, max_concurrency=2).run()
    assert out == [str(n) for n in range(8)]     # input order preserved
    assert peak <= 2


async def test_failed_fan_out_does_not_checkpoint_unfinished_siblings(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    waiting = Agent("waiting")
    blocked = Agent("blocked")

    @waiting.on(Ping)
    async def handle(signal: Ping, ctx: AgentContext):
        started.set()
        await release.wait()
        await ctx.reply(Pong(text=signal.text))

    async def intercept(signal: Signal, source: str, target: str):
        if isinstance(signal, Ping) and signal.text == "fail":
            await started.wait()
            return None
        return signal

    async def flow(ctx):
        async with ctx.phase("scan"):
            return await ctx.fan_out(
                ["waiting", "blocked"],
                [Ping(text="slow"), Ping(text="fail")],
            )

    path = tmp_path / "cp.jsonl"
    mesh = Mesh([waiting, blocked])
    mesh.intercept(intercept)
    async with mesh:
        try:
            with pytest.raises(MeshError, match="request_sent"):
                await Script("s", mesh, flow, store=CheckpointStore(path)).run()
        finally:
            release.set()
        await mesh.wait_idle(timeout=2)
        await asyncio.sleep(0)

    assert len(CheckpointStore(path)) == 0


@pytest.mark.parametrize("failure", [True, False], ids=["step-failure", "caller-cancellation"])
async def test_fan_out_finishes_cleanup_despite_repeated_cancellation(tmp_path, failure):
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def intercept(signal: Signal, source: str, target: str):
        if isinstance(signal, Ping) and signal.text == "slow":
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_finished.set()
        if isinstance(signal, Ping) and signal.text == "fail":
            await started.wait()
            return None
        return signal

    async def flow(ctx):
        async with ctx.phase("scan"):
            signals = [Ping(text="slow")]
            if failure:
                signals.append(Ping(text="fail"))
            return await ctx.fan_out(["waiting", "blocked"], signals)

    path = tmp_path / "cp.jsonl"
    mesh = Mesh([Agent("waiting"), Agent("blocked")])
    mesh.intercept(intercept)
    async with mesh:
        pending = asyncio.create_task(Script("s", mesh, flow, store=CheckpointStore(path)).run())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            if not failure:
                pending.cancel("initial cancellation")
            await asyncio.wait_for(cleanup_started.wait(), timeout=2)
            pending.cancel("repeated cancellation")
            await asyncio.sleep(0)
            pending.cancel("repeated cancellation")
            await asyncio.sleep(0)
            assert not pending.done()
        finally:
            release_cleanup.set()
        error = MeshError if failure else asyncio.CancelledError
        message = "request_sent" if failure else "initial cancellation"
        with pytest.raises(error, match=message):
            await asyncio.wait_for(pending, timeout=2)
        assert cleanup_finished.is_set()
        assert len(mesh.get("waiting")._outbox) == 0

    assert len(CheckpointStore(path)) == 0


async def test_budget_exceeded_keeps_checkpoints(tmp_path):
    async def flow(ctx):
        async with ctx.phase("p"):
            for n in range(5):
                await ctx.run("echo", Ping(text=str(n)))

    mesh = Mesh([make_echo()])
    async with mesh:
        store = CheckpointStore(tmp_path / "cp.jsonl")
        with pytest.raises(BudgetExceeded):
            await Script("s", mesh, flow, budget=3, store=store).run()
    assert len(CheckpointStore(tmp_path / "cp.jsonl")) == 3


async def test_spawn_restores_topology():
    def factory() -> Agent:
        return make_echo("ephemeral")

    async def flow(ctx):
        async with ctx.phase("p"):
            replies = await asyncio.gather(
                *(ctx.spawn(factory, Ping(text=f"s{i}")) for i in range(3))
            )
            return sorted(r.text for r in replies)

    mesh = Mesh()
    before = mesh.topology()
    async with mesh:
        out = await Script("s", mesh, flow).run()
    assert out == ["S0", "S1", "S2"]             # concurrent spawns, unique names
    assert mesh.topology() == before             # no residue


async def test_failed_step_surfaces_as_timeout_without_checkpoint(tmp_path):
    flaky = Agent("flaky")

    @flaky.on(Ping)
    async def handle(signal: Ping):
        raise RuntimeError("boom")               # dead-letters; never replies

    async def flow(ctx):
        async with ctx.phase("p"):
            await ctx.run("flaky", Ping(text="x"), timeout=0.2)

    mesh = Mesh([flaky])
    async with mesh:
        store = CheckpointStore(tmp_path / "cp.jsonl")
        with pytest.raises(asyncio.TimeoutError):
            await Script("s", mesh, flow, store=store).run()
    assert len(CheckpointStore(tmp_path / "cp.jsonl")) == 0
    assert flaky.dead_letters.count == 1
