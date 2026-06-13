"""Agent team example: a 3-member review team draining a shared task board.

Three reviewers (security, performance, tests) each carry one obligation:
handle TaskAssigned and reply a TaskResult. The Team's stewards claim tasks
from the board as members free up; a summary task depends on all three
reviews and unblocks automatically when the last one completes. The lead
holds no machinery — it just hears MemberIdle when the backlog drains.
"""

import asyncio

from signal_gating import (
    Agent,
    AgentContext,
    MemberIdle,
    Mesh,
    TaskAssigned,
    TaskBoard,
    TaskResult,
    Team,
)

FILES = ["channel.py", "gate.py", "mesh.py"]


def make_reviewer(name: str, angle: str) -> Agent:
    agent = Agent(name)

    @agent.on(TaskAssigned)
    async def review(signal: TaskAssigned, ctx: AgentContext):
        path = signal.payload.get("path", "<summary>")
        print(f"  [{name}] {signal.brief} ({path})")
        await ctx.reply(
            TaskResult(
                task_id=signal.task_id,
                result={"angle": angle, "path": path, "findings": f"{angle} ok"},
            )
        )

    return agent


async def drained(board: TaskBoard) -> None:
    done = asyncio.Event()

    def check(_event) -> None:
        if board.tasks() and all(t.status == "completed" for t in board.tasks()):
            done.set()

    board.on_event(check)
    check(None)
    await done.wait()


async def main():
    security = make_reviewer("security", "security")
    performance = make_reviewer("performance", "performance")
    tests = make_reviewer("tests", "coverage")
    lead = Agent("lead")

    @lead.on(MemberIdle)
    async def on_idle(signal: MemberIdle):
        print(f"  [lead] {signal.member} is idle")

    mesh = Mesh([security, performance, tests, lead])
    team = Team("review", mesh, task_timeout=5.0)
    team.lead(lead)
    for member in (security, performance, tests):
        team.enroll(member, role="reviewer")

    async with mesh:
        async with team:
            review_ids = [
                await team.open(f"review {path}", payload={"path": path})
                for path in FILES
            ]
            await team.open("summarize findings", depends_on=tuple(review_ids))
            await drained(team.board)
            await asyncio.sleep(0.05)  # let idle notifications land

    print("\nBoard results:")
    for task in team.board.tasks():
        print(f"  {task.brief}: {dict(task.result)} (by {task.member})")
    print(f"\nLedger head digest: {team.board.head_digest}")


asyncio.run(main())
