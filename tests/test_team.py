"""Tests for Team: steward-driven peer coordination over a TaskBoard."""

import asyncio

import pytest

from signal_gating import Agent, AgentContext, Mesh
from signal_gating.errors import TeamError
from signal_gating.taskboard import TaskBoard
from signal_gating.team import MemberIdle, TaskAssigned, TaskResult, Team


def make_worker(name: str, fail: set[str] | None = None) -> Agent:
    agent = Agent(name)

    @agent.on(TaskAssigned)
    async def work(signal: TaskAssigned, ctx: AgentContext):
        if fail and signal.brief in fail:
            raise RuntimeError(f"cannot do {signal.brief}")
        await ctx.reply(TaskResult(task_id=signal.task_id, result={"by": name}))

    return agent


async def drained(board: TaskBoard) -> None:
    done = asyncio.Event()

    def check(_event):
        if board.tasks() and all(t.status == "completed" for t in board.tasks()):
            done.set()

    board.on_event(check)
    check(None)
    await asyncio.wait_for(done.wait(), timeout=5.0)


async def test_assign_executes_and_completes_on_board():
    mesh = Mesh([make_worker("w")])
    team = Team("t", mesh)
    team.enroll(mesh.get("w"))
    async with mesh:
        async with team:
            tid = await team.open("review channel.py", payload={"path": "channel.py"})
            await team.assign(tid, "w")
            await drained(team.board)
    assert dict(team.board.task(tid).result) == {"by": "w"}


async def test_two_members_drain_dependent_backlog_without_double_work():
    a, b = make_worker("a"), make_worker("b")
    mesh = Mesh([a, b])
    team = Team("t", mesh)
    team.enroll(a)
    team.enroll(b)
    async with mesh:
        async with team:
            first = await team.open("t0")
            for i in range(1, 5):
                await team.open(f"t{i}", depends_on=(first,))
            await drained(team.board)
    assert all(t.member in ("a", "b") for t in team.board.tasks())
    claims = [e for e in team.board.events if e.wire_type() == "sgp.task.claimed"]
    assert len(claims) == 5                      # zero double-claims


async def test_crash_releases_task_and_peer_recovers():
    crasher = make_worker("crasher", fail={"hard"})
    helper = make_worker("helper")
    mesh = Mesh([crasher, helper])
    team = Team("t", mesh, task_timeout=0.3)
    team.enroll(crasher)
    async with mesh:
        async with team:
            released = asyncio.Event()
            team.board.on_event(
                lambda e: released.set()
                if e.wire_type() == "sgp.task.released"
                else None
            )
            tid = await team.open("hard")
            await team.assign(tid, "crasher")
            await asyncio.wait_for(released.wait(), timeout=5.0)
            team.enroll(helper)                  # the release re-pends; helper wakes
            await team.assign(tid, "helper")
            await drained(team.board)
    assert team.board.task(tid).member == "helper"
    assert crasher.dead_letters.count >= 1


async def test_member_idle_reaches_lead_edge_triggered():
    lead, worker = Agent("lead"), make_worker("w")
    idles: list[str] = []

    @lead.on(MemberIdle)
    async def on_idle(signal: MemberIdle):
        idles.append(signal.member)

    mesh = Mesh([lead, worker])
    team = Team("t", mesh)
    team.lead(lead)
    team.enroll(worker)
    async with mesh:
        async with team:
            tid = await team.open("x")
            await team.assign(tid, "w")
            await drained(team.board)
            await asyncio.sleep(0.05)            # deliver the idle notification
    assert idles == ["w"]                        # edge-triggered: exactly one


async def test_shutdown_and_dissolve():
    worker = make_worker("w")
    mesh = Mesh([worker])
    team = Team("t", mesh)
    team.enroll(worker)
    async with mesh:
        await team.start()
        await team.shutdown("w")
        assert not worker.running
        await team.dissolve()
        with pytest.raises(TeamError):
            await team.dissolve()
