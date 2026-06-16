"""TaskBoard: a durable, gate-checked task ledger folded from signal events."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import (
    Field,
    field_serializer,
    field_validator,
)

from signal_gating.errors import SignalSerializationError, TaskRejected
from signal_gating.gate import Gate
from signal_gating.registry import from_wire
from signal_gating.signal import Signal
from signal_gating.trajectory import _digest

_GENESIS = "sgp-taskboard-genesis"


def _frozen(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


class _PayloadSignal(Signal):
    """Mixin: a frozen, JSON-safe ``payload`` mapping (the Signal.metadata pattern)."""

    payload: Mapping[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("payload", mode="after")
    @classmethod
    def _freeze_payload(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _frozen(value)

    @field_serializer("payload")
    def _serialize_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(value)


class _ResultSignal(Signal):
    """Mixin: a frozen, JSON-safe ``result`` mapping."""

    result: Mapping[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("result", mode="after")
    @classmethod
    def _freeze_result(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _frozen(value)

    @field_serializer("result")
    def _serialize_result(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(value)


class TaskOpened(_PayloadSignal):
    __signal_type__: ClassVar[str] = "sgp.task.opened"
    task_id: str = ""
    brief: str = ""
    depends_on: tuple[str, ...] = ()


class TaskClaimed(Signal):
    __signal_type__: ClassVar[str] = "sgp.task.claimed"
    task_id: str = ""
    member: str = ""


class TaskReleased(Signal):
    __signal_type__: ClassVar[str] = "sgp.task.released"
    task_id: str = ""
    member: str = ""
    reason: str = ""


class TaskCompleted(_ResultSignal):
    __signal_type__: ClassVar[str] = "sgp.task.completed"
    task_id: str = ""
    member: str = ""


@dataclass(frozen=True)
class Task:
    """Frozen view of one task, derived from the event fold."""

    id: str
    brief: str
    status: str                  # "pending" | "in_progress" | "completed"
    member: str
    priority: int
    depends_on: tuple[str, ...]
    payload: Mapping[str, Any]
    result: Mapping[str, Any]


class TaskBoard:
    def __init__(
        self,
        name: str,
        *,
        open_gate: Gate | None = None,
        complete_gate: Gate | None = None,
    ) -> None:
        self.name = name
        self._open_gate = open_gate
        self._complete_gate = complete_gate
        self._lock = asyncio.Lock()
        self._events: list[Signal] = []
        self._tasks: dict[str, Task] = {}      # cached fold, updated per append
        self._opened: dict[str, TaskOpened] = {}
        self._records: list[dict[str, Any]] = []
        self._observers: list[Callable[[Signal], None]] = []
        self.observer_errors = 0

    # -- queries (sync, over the cached fold) ---------------------------------

    @property
    def events(self) -> list[Signal]:
        return list(self._events)

    @property
    def head_digest(self) -> str:
        """The chain head. Store it externally to make tail truncation detectable;
        the chain itself detects edits, reordering, and interior deletion."""
        if not self._records:
            return _GENESIS
        digest: str = self._records[-1]["digest"]
        return digest

    # -- ledger ----------------------------------------------------------------

    def on_event(self, fn: Callable[[Signal], None]) -> Callable[[], None]:
        """Register an advisory observer; returns an unsubscribe callable.

        Observers cannot block a transition; their exceptions are counted.
        """
        self._observers.append(fn)
        return lambda: self._observers.remove(fn)

    def export_jsonl(self, path: str | Path) -> int:
        with open(path, "w", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return len(self._records)

    @classmethod
    def load_jsonl(
        cls, path: str | Path, *, release_in_progress: bool = True
    ) -> TaskBoard:
        board = cls(Path(path).stem)
        prev = _GENESIS
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                record: dict[str, Any] = json.loads(line)
                core = {
                    "seq": record["seq"], "prev": record["prev"], "event": record["event"]
                }
                if (
                    record["seq"] != n
                    or record["prev"] != prev
                    or record["digest"] != _digest(core)
                ):
                    raise SignalSerializationError(
                        f"taskboard ledger chain broken at seq {n}"
                    )
                prev = record["digest"]
                board._append(from_wire(record["event"]))
        if release_in_progress:
            for task in [t for t in board.tasks() if t.status == "in_progress"]:
                opened = board._opened[task.id]
                board._append(
                    TaskReleased(
                        task_id=task.id, member=task.member, reason="recovered",
                        trace_id=opened.trace_id, parent_id=opened.id,
                    )
                )
        return board

    def task(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def claimable(self) -> list[Task]:
        return sorted(
            (
                t
                for t in self._tasks.values()
                if t.status == "pending"
                and all(self._tasks[d].status == "completed" for d in t.depends_on)
            ),
            key=lambda t: -t.priority,
        )

    # -- transitions ----------------------------------------------------------

    async def open(
        self,
        brief: str,
        *,
        depends_on: tuple[str, ...] = (),
        priority: int = 0,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        event = TaskOpened(
            task_id=uuid4().hex,
            brief=brief,
            depends_on=tuple(depends_on),
            priority=priority,
            payload=dict(payload or {}),
        )
        event.to_wire()                              # JSON-safety at transition time
        await self._check(self._open_gate, event)    # gates run OUTSIDE the lock
        async with self._lock:
            for dep in event.depends_on:
                if dep not in self._tasks:
                    raise ValueError(f"unknown dependency {dep!r}")
            self._append(event)
        return event.task_id

    async def claim(self, member: str, task_id: str | None = None) -> Task | None:
        async with self._lock:
            pool = self.claimable()
            if task_id is not None:
                pool = [t for t in pool if t.id == task_id]
            if not pool:
                return None
            chosen = pool[0]
            opened = self._opened[chosen.id]
            self._append(
                TaskClaimed(
                    task_id=chosen.id,
                    member=member,
                    trace_id=opened.trace_id,
                    parent_id=opened.id,
                )
            )
            return self._tasks[chosen.id]

    async def complete(
        self, task_id: str, member: str, *, result: Mapping[str, Any] | None = None
    ) -> None:
        opened = self._opened.get(task_id)
        if opened is None:
            raise ValueError(f"unknown task {task_id!r}")
        event = TaskCompleted(
            task_id=task_id,
            member=member,
            result=dict(result or {}),
            trace_id=opened.trace_id,
            parent_id=opened.id,
        )
        event.to_wire()
        await self._check(self._complete_gate, event)
        async with self._lock:
            self._require(task_id, member, "in_progress")
            self._append(event)

    async def release(self, task_id: str, member: str, *, reason: str = "") -> None:
        opened = self._opened.get(task_id)
        if opened is None:
            raise ValueError(f"unknown task {task_id!r}")
        async with self._lock:
            self._require(task_id, member, "in_progress")
            self._append(
                TaskReleased(
                    task_id=task_id,
                    member=member,
                    reason=reason,
                    trace_id=opened.trace_id,
                    parent_id=opened.id,
                )
            )

    # -- internals --------------------------------------------------------------

    async def _check(self, gate: Gate | None, event: Signal) -> None:
        if gate is not None and await gate.process(event) is None:
            raise TaskRejected(getattr(event, "task_id", ""), gate.name)

    def _require(self, task_id: str, member: str, status: str) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        if task.status != status or task.member != member:
            raise ValueError(
                f"task {task_id!r} is {task.status!r}/{task.member!r}, "
                f"expected {status!r}/{member!r}"
            )

    def _append(self, event: Signal) -> None:
        # Live transitions already hold the lock; load_jsonl is single-threaded.
        core = {
            "seq": len(self._records),
            "prev": self.head_digest,
            "event": event.to_wire(),
        }
        self._records.append({**core, "digest": _digest(core)})
        self._events.append(event)
        self._fold(event)
        for observer in list(self._observers):
            try:
                observer(event)
            except Exception:
                self.observer_errors += 1

    def _fold(self, event: Signal) -> None:
        if isinstance(event, TaskOpened):
            self._opened[event.task_id] = event
            self._tasks[event.task_id] = Task(
                id=event.task_id, brief=event.brief, status="pending", member="",
                priority=event.priority, depends_on=event.depends_on,
                payload=event.payload, result=_frozen({}),
            )
        elif isinstance(event, TaskClaimed):
            task = self._tasks[event.task_id]
            self._tasks[task.id] = replace(task, status="in_progress", member=event.member)
        elif isinstance(event, TaskReleased):
            task = self._tasks[event.task_id]
            self._tasks[task.id] = replace(task, status="pending", member="")
        elif isinstance(event, TaskCompleted):
            task = self._tasks[event.task_id]
            self._tasks[task.id] = replace(
                task, status="completed", member=event.member, result=event.result
            )
