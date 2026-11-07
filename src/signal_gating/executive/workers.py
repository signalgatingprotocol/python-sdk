"""Bounded proposal computation, without state writes or capability execution.

The serialized coordinator must commit Episode, context and BudgetPolicy
reservation atomically BEFORE run; recheck admission/current authority at dispatch;
then commit acceptance plus policy.settle, or policy.mark_unknown on cancellation,
process loss or absent usage. WorkerError.usage retains observations even when a
result is rejected. CancelledError also requires mark_unknown, never a zero charge.

is_current(episode, packet) reads authoritative current state synchronously. It
must check episode/task status, revision, IDs, current fence, worker binding,
reservation/admission, freshness and authority. The runner checks immediately
before dispatch and before returning; final acceptance still needs coordinator
CAS. No callback is a substitute for the sole writer's atomic commit.

Cancellation is a request, not proof that a provider stopped. Detached invocations
continue to count against max_active until they actually finish. close cancels
once and returns without joining them; the host owns eventual client/transport
shutdown. A malicious/uncooperative same-process coroutine cannot be killed by
this library. Use process isolation where hard host shutdown is required.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Protocol

from .errors import ExecutiveError
from .messages import EpisodeResult
from .records import ContextPacket, Episode, Usage, utc_now


class Worker(Protocol):
    async def run_episode(self, packet: ContextPacket) -> EpisodeResult: ...


class WorkerError(ExecutiveError):
    """Stable safe diagnostic carrying accounting observations, never raw output."""

    def __init__(self, code: str, message: str, *, usage: Usage | None = None) -> None:
        self.usage = usage if usage is not None else Usage()
        super().__init__(code, message)


async def _wait_until(deadline: datetime) -> None:
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[None] = loop.create_future()

    def wake() -> None:
        if not ready.done():
            ready.set_result(None)

    timer = loop.call_later(max(0, (deadline - utc_now()).total_seconds()), wake)
    try:
        await ready
    finally:
        timer.cancel()


def _drain(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


class EpisodeRunner:
    def __init__(
        self,
        *,
        is_current: Callable[[Episode, ContextPacket], bool],
        clock: Callable[[], datetime] = utc_now,
        wait_until: Callable[[datetime], Awaitable[None]] = _wait_until,
        max_active: int = 2,
    ) -> None:
        if type(max_active) is not int or max_active <= 0:
            raise ValueError("max_active must be a positive integer")
        self.is_current = is_current
        self.clock = clock
        self.wait_until = wait_until
        self.max_active = max_active
        self._pending: set[asyncio.Task[EpisodeResult]] = set()
        self._cancel_requested: set[asyncio.Task[EpisodeResult]] = set()
        self._closed = False
        self._shutdown = asyncio.Event()
        self._detached: dict[asyncio.Task[EpisodeResult], tuple[Episode, Usage]] = {}
        self._late_usage: list[tuple[Episode, Usage]] = []
        self.late_usage_ready = asyncio.Event()

    def _cancel(self, task: asyncio.Task[EpisodeResult]) -> None:
        if not task.done() and task not in self._cancel_requested:
            self._cancel_requested.add(task)
            task.cancel()

    def _finished(self, task: asyncio.Task[EpisodeResult]) -> None:
        self._pending.discard(task)
        self._cancel_requested.discard(task)
        detached = self._detached.pop(task, None)
        if detached is not None:
            episode, unknown = detached
            self._late_usage.append((episode, self._observed(task, unknown)))
            self.late_usage_ready.set()
        _drain(task)

    @staticmethod
    def _observed(task: asyncio.Task[EpisodeResult], fallback: Usage) -> Usage:
        if task.done() and not task.cancelled():
            try:
                result = task.result()
                if isinstance(result, EpisodeResult):
                    return result.usage
            except WorkerError as exc:
                return exc.usage
            except Exception:
                pass
        return fallback

    def take_late_usage(self) -> tuple[tuple[Episode, Usage], ...]:
        """Drain accounting only, never stale proposals. No state writes occur.

        The coordinator must serialize settlement/mark_unknown from these
        observations; draining is not proof of reconciliation. Until drained,
        observations count against max_active to bound retained memory.
        """
        result = tuple(self._late_usage)
        self._late_usage.clear()
        self.late_usage_ready.clear()
        return result

    def _check(self, episode: Episode, packet: ContextPacket, usage: Usage) -> None:
        if self._closed:
            raise WorkerError("runner_closed", "Episode runner is closed", usage=usage)
        if self.clock() >= episode.deadline:
            raise WorkerError("episode_timeout", "Episode deadline expired", usage=usage)
        if (
            episode.status != "running"
            or episode.id != packet.episode_id
            or episode.context_packet_id != packet.id
            or episode.task_id != packet.task.id
            or episode.fencing_token != packet.fencing_token
            or packet.task.fencing_token != packet.fencing_token
            or packet.task.status != "running"
            or not self.is_current(episode, packet)
        ):
            raise WorkerError("episode_fenced", "Episode is no longer current", usage=usage)

    async def _invoke(
        self, episode: Episode, packet: ContextPacket, worker: Worker
    ) -> EpisodeResult:
        self._check(episode, packet, Usage.zero(currency=packet.remaining_limits.currency))
        return await worker.run_episode(packet)

    async def _deadline(self, at: datetime) -> None:
        await self.wait_until(at)

    async def run(self, episode: Episode, packet: ContextPacket, worker: Worker) -> EpisodeResult:
        if self._closed:
            raise WorkerError("runner_closed", "Episode runner is closed")
        self._check(episode, packet, Usage.zero(currency=packet.remaining_limits.currency))
        if (
            sum(not task.done() for task in self._pending) + len(self._late_usage)
            >= self.max_active
        ):
            raise WorkerError("worker_capacity", "Worker concurrency capacity exhausted")
        task = asyncio.create_task(self._invoke(episode, packet, worker))
        self._pending.add(task)
        task.add_done_callback(self._finished)
        deadline = asyncio.create_task(self._deadline(episode.deadline))
        deadline.add_done_callback(_drain)
        shutdown = asyncio.create_task(self._shutdown.wait())
        shutdown.add_done_callback(_drain)
        unknown = Usage(calls=1, currency=packet.remaining_limits.currency, source="episode-runner")
        try:
            completed, _ = await asyncio.wait(
                (task, deadline, shutdown), return_when=asyncio.FIRST_COMPLETED
            )
            observed = self._observed(task, unknown)
            if self._closed:
                raise WorkerError("runner_closed", "Episode runner is closed", usage=observed)
            if deadline in completed:
                # A failed timer also fails closed; never depend on child cancellation.
                raise WorkerError("episode_timeout", "Episode deadline expired", usage=observed)
            try:
                result = task.result()
            except WorkerError:
                raise
            except Exception:
                raise WorkerError(
                    "worker_failure", "Worker invocation failed", usage=unknown
                ) from None
            if not isinstance(result, EpisodeResult):
                raise WorkerError("invalid_worker_result", "Invalid worker result", usage=unknown)
            self._check(episode, packet, result.usage)
            if any(
                proposal.kind not in episode.allowed_proposal_types
                or proposal.episode_id != episode.id
                or proposal.fencing_token != episode.fencing_token
                for proposal in result.proposals
            ):
                raise WorkerError(
                    "invalid_worker_result", "Proposal binding or type rejected", usage=result.usage
                )
            return result
        except asyncio.CancelledError:
            if task.done():
                self._late_usage.append((episode, self._observed(task, unknown)))
                self.late_usage_ready.set()
            raise
        finally:
            if not task.done():
                self._detached[task] = (episode, unknown)
            self._cancel(task)
            deadline.cancel()
            shutdown.cancel()

    async def close(self) -> None:
        """Stop admission and request cancellation, with no unbounded join."""
        self._closed = True
        self._shutdown.set()
        for task in tuple(self._pending):
            self._cancel(task)
