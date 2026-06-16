"""Team: steward-driven coordination of peer agents over a TaskBoard.

The protocol lives in team-owned steward coroutines, one per member — never
inside member agents. Members carry exactly one obligation: handle
TaskAssigned and reply a TaskResult.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import TYPE_CHECKING, Any, ClassVar

from signal_gating.errors import TaskRejected, TeamError
from signal_gating.signal import Signal
from signal_gating.taskboard import Task, TaskBoard, _PayloadSignal, _ResultSignal

if TYPE_CHECKING:
    from signal_gating.agent import Agent
    from signal_gating.mesh import Mesh


class Mail(Signal):
    __signal_type__: ClassVar[str] = "sgp.team.mail"
    to: str = ""
    sender: str = ""
    body: str = ""


class TaskAssigned(_PayloadSignal):
    __signal_type__: ClassVar[str] = "sgp.team.task_assigned"
    task_id: str = ""
    brief: str = ""


class TaskResult(_ResultSignal):
    __signal_type__: ClassVar[str] = "sgp.team.task_result"
    task_id: str = ""


class MemberIdle(Signal):
    __signal_type__: ClassVar[str] = "sgp.team.member_idle"
    member: str = ""


class _Steward:
    def __init__(self) -> None:
        self.wake = asyncio.Event()
        self.assigned: deque[Task] = deque()
        self.stopping = False
        self.worked = False          # edge trigger for MemberIdle
        # Tasks this member failed (timeout or gate rejection). Excluded from
        # untargeted claims so a failing member does not instantly re-claim
        # the task it just released — the release re-pends work for peers,
        # and an explicit assign() can still retry it on this member.
        self.failed: set[str] = set()
        self.runner: asyncio.Task[None] | None = None
        # The task currently being executed (claimed/assigned, awaiting its
        # handler's reply). Tracked so a cancelling shutdown can release it
        # instead of stranding it in_progress forever.
        self.current: Task | None = None


class Team:
    """A named set of peer agents draining a shared TaskBoard.

    Each enrolled member gets one team-owned steward coroutine that claims
    tasks from the board, executes them via ``mesh.request`` against the
    member's ``TaskAssigned`` handler, and records the outcome (completion,
    or release on timeout / gate rejection) back on the board. A handler
    that raises is dead-lettered by the member's own supervision and never
    replies, which surfaces as the same timeout — a crashed teammate never
    strands a claimed task.
    """

    def __init__(
        self,
        name: str,
        mesh: Mesh,
        board: TaskBoard | None = None,
        *,
        task_timeout: float = 60.0,
    ) -> None:
        self.name = name
        self.board = board if board is not None else TaskBoard(name)
        self._mesh = mesh
        self._task_timeout = task_timeout
        self._lead: str | None = None
        self._members: dict[str, str] = {}
        self._stewards: dict[str, _Steward] = {}
        self._started = False
        self._dissolved = False
        self._unsubscribe = self.board.on_event(self._on_board_event)
        self.idle_errors = 0

    # -- membership ------------------------------------------------------------

    @property
    def members(self) -> dict[str, str]:
        return dict(self._members)

    def lead(self, agent: Agent | str) -> None:
        """Name the conventional MemberIdle recipient. Optional; a lead holds
        no machinery and a stopped lead degrades nothing but idle delivery."""
        self._lead = agent if isinstance(agent, str) else agent.name

    def enroll(self, agent: Agent, role: str = "member") -> None:
        if self._dissolved:
            raise TeamError(f"team {self.name!r} is dissolved")
        if agent.name in self._members:
            raise TeamError(f"member {agent.name!r} already enrolled")
        self._members[agent.name] = role
        steward = _Steward()
        self._stewards[agent.name] = steward
        if self._started:
            steward.runner = asyncio.ensure_future(self._run_steward(agent.name))

    # -- coordination ------------------------------------------------------------

    async def open(self, brief: str, **kwargs: Any) -> str:
        return await self.board.open(brief, **kwargs)

    async def assign(self, task_id: str, member: str) -> None:
        if member not in self._members:
            raise TeamError(f"unknown member {member!r}")
        task = await self.board.claim(member, task_id=task_id)
        if task is None:
            raise TeamError(f"task {task_id!r} is not claimable")
        steward = self._stewards[member]
        steward.assigned.append(task)
        steward.wake.set()

    async def send(self, to: str, body: str, *, sender: str = "") -> None:
        await self._mesh.inject(to, Mail(to=to, sender=sender, body=body))

    # -- lifecycle -----------------------------------------------------------------

    async def start(self) -> None:
        if self._dissolved:
            raise TeamError(f"team {self.name!r} is dissolved")
        self._started = True
        for name, steward in self._stewards.items():
            if steward.runner is None:
                steward.runner = asyncio.ensure_future(self._run_steward(name))

    async def shutdown(self, member: str, timeout: float | None = None) -> None:
        """Team-side shutdown: stop claiming, drain the in-flight task, release
        anything claimed-but-unstarted, then stop the agent from outside."""
        steward = self._stewards[member]
        steward.stopping = True
        steward.wake.set()
        runner = steward.runner
        if runner is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(runner),
                    timeout if timeout is not None else self._task_timeout,
                )
            except asyncio.TimeoutError:
                runner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await runner
            except asyncio.CancelledError:
                # shield() raises this both when the runner itself was
                # cancelled and when *we* are cancelled; only re-raise the
                # latter so a cancelled steward never poisons shutdown.
                if not runner.cancelled():
                    raise
        # If the runner was cancelled mid-execution, its in-flight task is
        # still in_progress; release it so a peer can reclaim it rather than
        # leaving it stranded forever.
        if steward.current is not None:
            in_flight = steward.current
            steward.current = None
            with contextlib.suppress(ValueError):
                await self.board.release(in_flight.id, member, reason="shutdown")
        while steward.assigned:
            task = steward.assigned.popleft()
            await self.board.release(task.id, member, reason="shutdown")
        agent = self._mesh.get(member)
        if agent.running:
            await agent.stop()

    async def dissolve(self) -> None:
        if self._dissolved:
            raise TeamError(f"team {self.name!r} is already dissolved")
        for member in list(self._members):
            await self.shutdown(member)
        self._unsubscribe()
        self._dissolved = True

    async def __aenter__(self) -> Team:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if not self._dissolved:
            await self.dissolve()

    # -- the steward loop -----------------------------------------------------------

    def _on_board_event(self, _event: Signal) -> None:
        # Event.set() cannot raise, so this advisory observer is loss-free.
        # Every event wakes stewards — including TaskReleased, which re-pends
        # work, and TaskCompleted, which may unblock dependents.
        for steward in self._stewards.values():
            steward.wake.set()

    async def _claim_next(self, member: str, steward: _Steward) -> Task | None:
        # Untargeted claim, skipping tasks this member already failed; the
        # snapshot-then-targeted-claim loop tolerates peers winning a race
        # for any candidate (the board hands out each task exactly once).
        for candidate in self.board.claimable():
            if candidate.id in steward.failed:
                continue
            task = await self.board.claim(member, task_id=candidate.id)
            if task is not None:
                return task
        return None

    async def _run_steward(self, member: str) -> None:
        steward = self._stewards[member]
        while not steward.stopping:
            task = (
                steward.assigned.popleft()
                if steward.assigned
                else await self._claim_next(member, steward)
            )
            if task is None:
                if steward.worked:
                    steward.worked = False
                    await self._notify_idle(member)
                steward.wake.clear()
                await steward.wake.wait()
                continue
            opened = self.board._opened[task.id]
            assigned = TaskAssigned(
                task_id=task.id,
                brief=task.brief,
                payload=dict(task.payload),
                trace_id=opened.trace_id,        # caller threads lineage
                parent_id=opened.id,
            )
            steward.current = task
            try:
                reply = await self._mesh.request(
                    member, assigned, timeout=self._task_timeout
                )
                result = dict(reply.result) if isinstance(reply, TaskResult) else {}
                await self.board.complete(task.id, member, result=result)
            except asyncio.TimeoutError:
                # Crashed handlers dead-letter and never reply; same surface.
                steward.failed.add(task.id)
                await self.board.release(task.id, member, reason="timeout")
            except TaskRejected as err:
                steward.failed.add(task.id)
                await self.board.release(
                    task.id, member, reason=f"complete_gate:{err.gate_name}"
                )
            finally:
                steward.worked = True
            # Cleared only after a normal/handled exit. On cancellation (a
            # timed-out shutdown) this line is skipped, leaving `current` set
            # so shutdown can release the still-in_progress task.
            steward.current = None

    async def _notify_idle(self, member: str) -> None:
        if self._lead is None:
            return
        try:
            await self._mesh.inject(self._lead, MemberIdle(member=member))
        except Exception:
            self.idle_errors += 1        # a dead lead must not poison stewards
