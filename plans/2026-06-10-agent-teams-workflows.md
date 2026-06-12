# Agent Teams & Scripted Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship the three orchestration-state subsystems from the spec: `TaskBoard` (durable, gate-checked, hash-chained task ledger), `Team` (steward-driven peer coordination over a board), and `Script` (checkpointed, budget-bounded workflow runtime).

**Architecture:** Coordination state is a fold over an append-only log of signals. `taskboard.py` holds the event signals (pinned `sgp.task.*` wire names) and the chained ledger; `team.py` holds the protocol signals and team-owned steward coroutines that execute tasks via `mesh.request` against a member's `TaskAssigned` handler; `script.py` holds `Script`/`ScriptContext`/`CheckpointStore` with content-addressed, occurrence-indexed step keys built on the (promoted) `domain_payload` projection from `trajectory.py`. `Mesh.remove()` is hardened for spawn churn. No base `Signal`/`Gate`/`Channel`/`Agent` changes.

**Tech Stack:** Python 3.10+, pydantic v2, pytest + pytest-asyncio (auto mode), ruff, mypy strict.

**Spec:** `specs/2026-06-10-agent-teams-workflows-design.md`

---

## Conventions

- **Commits:** per CLAUDE.md, commit only when the user authorizes. Commit steps are ready-to-run; treat as "stage + commit once authorized." Branch is `claude/agent-teams-workflows-d7kbfn` (off `main`, not default).
- **Working dir:** the repo root. Tests: `pytest` (asyncio auto mode — `async def test_*` runs without decorators).
- **mypy:** runs on `src/` only; test files need not be strict-typed.

## File structure

| File | Change |
| --- | --- |
| `src/signal_gating/mesh.py` | Harden `remove()`: pool purge + prunable `route()`/`load_balance()` routes. |
| `src/signal_gating/pool.py` | Add `AgentPool.discard(name)`. |
| `src/signal_gating/errors.py` | Add `TaskRejected`, `TeamError`, `BudgetExceeded`. |
| `src/signal_gating/trajectory.py` | Promote `_domain_payload` to public `domain_payload()`. |
| `src/signal_gating/taskboard.py` | New: event signals, `Task` view, `TaskBoard`, chained ledger. |
| `src/signal_gating/team.py` | New: protocol signals, `Team`, stewards. |
| `src/signal_gating/script.py` | New: `CheckpointStore`, `Script`, `ScriptContext`. |
| `src/signal_gating/__init__.py` | Export the above. |
| `tests/test_taskboard.py`, `tests/test_team.py`, `tests/test_script.py` | New (+ `remove()` cases appended to `tests/test_mesh.py`). |
| `examples/agent_team.py`, `examples/scripted_workflow.py` | New. |
| `README.md` | New "Teams" and "Scripted workflows" sections. |

---

## Task 1: Harden `Mesh.remove()` for spawn churn

`remove()` exists (mesh.py:128) and handles `connect`-tagged routes, topics, capabilities. It misses: pool worker lists, and the `target=None` routing closures created by `load_balance()`/`route()` that hold direct agent references and keep sending to a removed agent's closed inbox.

**Files:**
- Modify: `src/signal_gating/mesh.py`, `src/signal_gating/pool.py`
- Test: `tests/test_mesh.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mesh.py`:

```python
class TestRemoveHardening:
    async def test_remove_purges_pool_membership(self):
        pool = AgentPool("workers", size=3)
        mesh = Mesh()
        mesh.add_pool(pool)
        victim = pool.workers[1]
        await mesh.remove(victim)
        assert victim.name not in pool.worker_names
        assert pool.size == 2

    async def test_remove_prunes_load_balance_target(self):
        src, a, b = Agent("src"), Agent("a"), Agent("b")
        got: list[str] = []
        for agent in (a, b):
            @agent.on(Signal)
            async def handle(signal: Signal, _name=agent.name):
                got.append(_name)
        mesh = Mesh([src, a, b])
        mesh.load_balance(src, [a, b])
        async with mesh:
            await mesh.remove(b)
            for _ in range(4):
                await src.emit(Signal())
            await asyncio.sleep(0.05)
        assert got == ["a", "a", "a", "a"]

    async def test_remove_prunes_route_branch_falls_to_default(self):
        src, hot, cold = Agent("src"), Agent("hot"), Agent("cold")
        got: list[str] = []
        for agent in (hot, cold):
            @agent.on(Signal)
            async def handle(signal: Signal, _name=agent.name):
                got.append(_name)
        mesh = Mesh([src, hot, cold])
        mesh.route(src, [(lambda s: s.priority >= 5, hot)], default=cold)
        async with mesh:
            await mesh.remove(hot)
            await src.emit(Signal(priority=9))
            await asyncio.sleep(0.05)
        assert got == ["cold"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mesh.py::TestRemoveHardening -q`
Expected: FAIL — the pool still lists the removed worker; load-balanced/routed signals target the closed inbox or vanish.

- [ ] **Step 3: Implement pruning**

In `src/signal_gating/pool.py`, add to `AgentPool`:

```python
    def discard(self, name: str) -> bool:
        """Remove a worker by name without stopping it (mesh.remove() stops it).

        Returns True if a worker was removed. The pool does not backfill;
        call scale_to() to restore capacity.
        """
        before = len(self._workers)
        self._workers = [w for w in self._workers if w.name != name]
        return len(self._workers) != before
```

In `src/signal_gating/mesh.py`, give `RouteFn` an optional prune hook:

```python
@dataclass
class RouteFn:
    fn: Callable[[Signal], Coroutine[Any, Any, None]]
    source: str
    target: str | None = None
    tag: str = ""
    # Called with a removed agent's name; mutates captured route state and
    # returns True when the route has no remaining targets and must be dropped.
    prune: Callable[[str], bool] | None = None
```

In `route()`, capture the resolved structures mutably and register the hook (the closure reads `resolved` and `default_box[0]` instead of a local `resolved_default`):

```python
        resolved: list[tuple[Callable[[Signal], bool], Agent]] = [
            (pred, self._resolve(tgt)) for pred, tgt in routes
        ]
        default_box: list[Agent | None] = [self._resolve(default) if default else None]

        def prune(name: str) -> bool:
            resolved[:] = [(p, t) for p, t in resolved if t.name != name]
            if default_box[0] is not None and default_box[0].name == name:
                default_box[0] = None
            return not resolved and default_box[0] is None

        src._add_output(
            RouteFn(fn=content_route, source=src.name, tag="content_route", prune=prune)
        )
```

In `load_balance()` likewise (`resolved[:] = [t for t in resolved if t.name != name]`; empty list → return True). In `remove()`, after the existing `_remove_outputs` pass over other agents:

```python
        for other in self._agents.values():
            other._outbox = [
                fn
                for fn in other._outbox
                if not (
                    isinstance(fn, RouteFn)
                    and fn.prune is not None
                    and fn.prune(resolved.name)
                )
            ]
        for pool in self._pools.values():
            pool.discard(resolved.name)
```

Extend the `remove()` docstring with the in-flight hazard: clearing the removed agent's `_outbox` drops any live `mesh.request` capture functions, so a requester awaiting that agent times out rather than erroring.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mesh.py -q` — all pass, including the pre-existing remove tests.

- [ ] **Step 5: Commit** (when authorized)

`git add src/signal_gating/mesh.py src/signal_gating/pool.py tests/test_mesh.py && git commit -m "feat(mesh): harden remove() for pool membership and closure routes"`

---

## Task 2: Errors and the public payload projection

**Files:**
- Modify: `src/signal_gating/errors.py`, `src/signal_gating/trajectory.py`
- Test: `tests/test_taskboard.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_taskboard.py` with just:

```python
from signal_gating import Signal
from signal_gating.errors import BudgetExceeded, TaskRejected, TeamError
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_taskboard.py -q`
Expected: FAIL — `ImportError: cannot import name 'TaskRejected'`.

- [ ] **Step 3: Implement**

In `errors.py` (follows the `GateRejected`/`CircuitOpenError` precedent; `N818` is already ignored):

```python
class TaskRejected(SignalGatingError):
    """A TaskBoard gate refused a task transition.

    Gate combinators collapse names, so gate_name is the outermost gate's
    name — the algebra cannot know which leaf rejected.
    """

    def __init__(self, task_id: str, gate_name: str = "") -> None:
        self.task_id = task_id
        self.gate_name = gate_name
        super().__init__(f"task {task_id!r} rejected by gate {gate_name!r}")


class TeamError(SignalGatingError):
    """Team protocol misuse (duplicate enrollment, bad assign, reuse after dissolve)."""


class BudgetExceeded(SignalGatingError):
    """A Script run exceeded its agent budget."""

    def __init__(self, budget: int, key: str) -> None:
        self.budget = budget
        self.key = key
        super().__init__(f"script budget of {budget} steps exceeded at step {key!r}")
```

In `trajectory.py`, rename `_domain_payload` to `domain_payload` with a public docstring (the volatile-envelope-excluding projection — drops `id`, `source`, `timestamp`, `priority`, `trace_id`, `correlation_id`, `parent_id`, `metadata`; a priority or metadata tweak deliberately does not change the projection) and keep `_domain_payload = domain_payload` as an alias for existing internal callers.

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_taskboard.py tests/test_trajectory.py -q`

- [ ] **Step 5: Commit** (when authorized)

`git add src/signal_gating/errors.py src/signal_gating/trajectory.py tests/test_taskboard.py && git commit -m "feat(errors,trajectory): orchestration errors and public domain_payload"`

---

## Task 3: `taskboard.py` — event signals and `TaskBoard` core

**Files:**
- Create: `src/signal_gating/taskboard.py`
- Test: `tests/test_taskboard.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_taskboard.py`:

```python
import asyncio

import pytest

from signal_gating import Gate
from signal_gating.taskboard import TaskBoard, TaskOpened


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_taskboard.py -q` — `ModuleNotFoundError: No module named 'signal_gating.taskboard'`.

- [ ] **Step 3: Implement the module**

Create `src/signal_gating/taskboard.py`:

```python
"""TaskBoard: a durable, gate-checked task ledger folded from signal events."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import Field, field_serializer, field_validator

from signal_gating.errors import TaskRejected
from signal_gating.gate import Gate
from signal_gating.signal import Signal


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

    # -- queries (sync, over the cached fold) ---------------------------------

    @property
    def events(self) -> list[Signal]:
        return list(self._events)

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
        opened = self._opened[task_id]
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
        opened = self._opened[task_id]
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
        task = self._tasks[task_id]
        if task.status != status or task.member != member:
            raise ValueError(
                f"task {task_id!r} is {task.status!r}/{task.member!r}, "
                f"expected {status!r}/{member!r}"
            )

    def _append(self, event: Signal) -> None:
        self._events.append(event)
        self._fold(event)

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
```

(`Gate(fn, name)` is the raw constructor — gate.py:34. If `Gate.filter` already threads a `name` kwarg, the test may use it instead; check gate.py:101 and keep whichever reads better.)

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_taskboard.py -q`

- [ ] **Step 5: Commit** (when authorized)

`git add src/signal_gating/taskboard.py tests/test_taskboard.py && git commit -m "feat(taskboard): signal-sourced task ledger with gates and targeted claims"`

---

## Task 4: Chained JSONL ledger, observers, crash recovery

**Files:**
- Modify: `src/signal_gating/taskboard.py`
- Test: `tests/test_taskboard.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

from signal_gating.errors import SignalSerializationError


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
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_taskboard.py -q` (`AttributeError: ... no attribute 'export_jsonl'`).

- [ ] **Step 3: Implement**

Add imports and constants to `taskboard.py`:

```python
import json
from collections.abc import Callable
from pathlib import Path

from signal_gating.errors import SignalSerializationError
from signal_gating.registry import from_wire
from signal_gating.trajectory import _digest

_GENESIS = "sgp-taskboard-genesis"
```

Add to `TaskBoard.__init__`:

```python
        self._records: list[dict[str, Any]] = []
        self._observers: list[Callable[[Signal], None]] = []
        self.observer_errors = 0
```

Add the ledger surface:

```python
    @property
    def head_digest(self) -> str:
        """The chain head. Store it externally to make tail truncation detectable;
        the chain itself detects edits, reordering, and interior deletion."""
        return self._records[-1]["digest"] if self._records else _GENESIS

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
                record = json.loads(line)
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
```

Extend `_append` to chain and notify (live transitions already hold the lock; `load_jsonl` is single-threaded):

```python
    def _append(self, event: Signal) -> None:
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
```

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_taskboard.py -q`

- [ ] **Step 5: Commit** (when authorized)

`git add src/signal_gating/taskboard.py tests/test_taskboard.py && git commit -m "feat(taskboard): hash-chained JSONL ledger, observers, crash recovery"`

---

## Task 5: `team.py` — protocol signals, `Team`, stewards

**Files:**
- Create: `src/signal_gating/team.py`
- Test: `tests/test_team.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_team.py`:

```python
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
    assert len(crasher.dead_letters) >= 1


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
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_team.py -q` (`ModuleNotFoundError: No module named 'signal_gating.team'`).

- [ ] **Step 3: Implement the module**

Create `src/signal_gating/team.py`:

```python
"""Team: steward-driven coordination of peer agents over a TaskBoard.

The protocol lives in team-owned steward coroutines, one per member — never
inside member agents. Members carry exactly one obligation: handle
TaskAssigned and reply a TaskResult.
"""

from __future__ import annotations

import asyncio
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
        self.runner: asyncio.Task[None] | None = None


class Team:
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
        if steward.runner is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(steward.runner), timeout or self._task_timeout
                )
            except asyncio.TimeoutError:
                steward.runner.cancel()
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

    async def _run_steward(self, member: str) -> None:
        steward = self._stewards[member]
        while not steward.stopping:
            task = (
                steward.assigned.popleft()
                if steward.assigned
                else await self.board.claim(member)
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
            try:
                reply = await self._mesh.request(
                    member, assigned, timeout=self._task_timeout
                )
                result = dict(reply.result) if isinstance(reply, TaskResult) else {}
                await self.board.complete(task.id, member, result=result)
            except asyncio.TimeoutError:
                # Crashed handlers dead-letter and never reply; same surface.
                await self.board.release(task.id, member, reason="timeout")
            except TaskRejected as err:
                await self.board.release(
                    task.id, member, reason=f"complete_gate:{err.gate_name}"
                )
            finally:
                steward.worked = True

    async def _notify_idle(self, member: str) -> None:
        if self._lead is None:
            return
        try:
            await self._mesh.inject(self._lead, MemberIdle(member=member))
        except Exception:
            self.idle_errors += 1        # a dead lead must not poison stewards
```

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_team.py -q`

- [ ] **Step 5: Commit** (when authorized)

`git add src/signal_gating/team.py tests/test_team.py && git commit -m "feat(team): steward-driven teams with request/reply task execution"`

---

## Task 6: `script.py` — `CheckpointStore` and step keys

**Files:**
- Create: `src/signal_gating/script.py`
- Test: `tests/test_script.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_script.py`:

```python
import pytest

from signal_gating import Signal
from signal_gating.errors import SignalSerializationError
from signal_gating.script import CheckpointStore, step_key


class Ping(Signal):
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
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_script.py -q` (`ModuleNotFoundError`).

- [ ] **Step 3: Implement store and keys**

Create `src/signal_gating/script.py` (first half):

```python
"""Script: a script-held workflow runtime with checkpointed, resumable steps.

Unrelated to mesh.workflow(), which is a one-shot step chain; a Script is a
user-authored coroutine that owns the loop and the branching, with the runtime
contributing bounded fan-out, an agent budget, and resume-as-cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from signal_gating.errors import SignalSerializationError
from signal_gating.registry import from_wire
from signal_gating.signal import Signal
from signal_gating.trajectory import _digest, domain_payload


def step_key(
    script: str,
    phase: str,
    signal: Signal,
    *,
    occurrence: int,
    target: str | None = None,
) -> str:
    """Content-addressed step identity: stable across runs, volatile-field-free.

    ``target`` is set only by ctx.run (a caller-named agent); fan_out/spawn
    omit it because worker assignment is incidental to a step's identity.
    """
    return _digest(
        {
            "script": script,
            "phase": phase,
            "target": target,
            "type": signal.wire_type(),
            "payload": domain_payload(signal),
            "occurrence": occurrence,
        }
    )


class CheckpointStore:
    """Append-only JSONL of completed step results; per-record sha256 digests.

    ``path=None`` is a per-run in-memory store: no persistence, no resume.
    Duplicate keys on load: last record wins.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._results: dict[str, Signal] = {}
        if self._path is not None and self._path.exists():
            with open(self._path, encoding="utf-8") as fh:
                for n, line in enumerate(fh):
                    record = json.loads(line)
                    core = {k: record[k] for k in ("key", "script", "phase", "result")}
                    if record["digest"] != _digest(core):
                        raise SignalSerializationError(
                            f"checkpoint record {n} failed digest verification"
                        )
                    self._results[record["key"]] = from_wire(record["result"])

    def __len__(self) -> int:
        return len(self._results)

    def get(self, key: str) -> Signal | None:
        return self._results.get(key)

    def put(self, key: str, script: str, phase: str, result: Signal) -> None:
        self._results[key] = result
        if self._path is None:
            return
        core = {"key": key, "script": script, "phase": phase, "result": result.to_wire()}
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**core, "digest": _digest(core)}, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_script.py -q`

- [ ] **Step 5: Commit** (when authorized)

`git add src/signal_gating/script.py tests/test_script.py && git commit -m "feat(script): checkpoint store with content-addressed step keys"`

---

## Task 7: `Script` + `ScriptContext` — phases, run, fan_out, spawn, limits

**Files:**
- Modify: `src/signal_gating/script.py`
- Test: `tests/test_script.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_script.py`:

```python
import asyncio

from signal_gating import Agent, AgentContext, Mesh
from signal_gating.errors import BudgetExceeded
from signal_gating.script import Script


class Pong(Signal):
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
    assert len(flaky.dead_letters) == 1
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_script.py -q` (`ImportError: cannot import name 'Script'`).

- [ ] **Step 3: Implement the runtime**

Append to `src/signal_gating/script.py`:

```python
import asyncio
import itertools
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from signal_gating.agent import Agent
from signal_gating.errors import BudgetExceeded
from signal_gating.mesh import Mesh


class ScriptContext:
    def __init__(self, script: Script, args: Any) -> None:
        self.args = args
        self._script = script
        self._phase = ""
        self._occurrences: dict[str, int] = {}
        self._spawn_counter = itertools.count()

    @asynccontextmanager
    async def phase(self, name: str) -> AsyncIterator[ScriptContext]:
        previous, self._phase = self._phase, name
        try:
            yield self
        finally:
            self._phase = previous

    async def run(
        self,
        target: Agent | str,
        signal: Signal,
        *,
        timeout: float = 30.0,
        key: str | None = None,
    ) -> Signal:
        """One checkpointed mesh.request. Failure — including a dead-lettered
        target handler that never replies — surfaces as asyncio.TimeoutError;
        the script decides what that means. The runtime never retries."""
        name = target if isinstance(target, str) else target.name
        resolved = key if key is not None else self._key(signal, target=name)
        return await self._execute(resolved, name, signal, timeout)

    async def fan_out(
        self,
        targets: Sequence[Agent | str],
        signals: Sequence[Signal],
        *,
        timeout: float = 30.0,
    ) -> list[Signal]:
        """Round-robin signals across targets under the concurrency semaphore;
        results in input order. One target is serial by construction."""
        names = [t if isinstance(t, str) else t.name for t in targets]
        steps = [
            (self._key(signal), names[i % len(names)], signal)
            for i, signal in enumerate(signals)
        ]
        return list(
            await asyncio.gather(
                *(self._execute(k, name, s, timeout) for k, name, s in steps)
            )
        )

    async def spawn(
        self,
        factory: Callable[[], Agent],
        signal: Signal,
        *,
        timeout: float = 30.0,
    ) -> Signal:
        """Ephemeral agent: build, uniquely rename, add, start, one checkpointed
        request, then stop and mesh.remove() — topology as found."""
        key = self._key(signal)
        cached = self._script._store.get(key)
        if cached is not None:
            return cached
        agent = factory()
        agent.name = f"{agent.name}#{next(self._spawn_counter)}"
        mesh = self._script._mesh
        mesh.add(agent)
        try:
            await agent.start()
            return await self._execute(key, agent.name, signal, timeout)
        finally:
            await mesh.remove(agent)

    def _key(self, signal: Signal, *, target: str | None = None) -> str:
        content = step_key(
            self._script.name, self._phase, signal, occurrence=0, target=target
        )
        occurrence = self._occurrences.get(content, 0)
        self._occurrences[content] = occurrence + 1
        return step_key(
            self._script.name, self._phase, signal,
            occurrence=occurrence, target=target,
        )

    async def _execute(
        self, key: str, target: str, signal: Signal, timeout: float
    ) -> Signal:
        cached = self._script._store.get(key)
        if cached is not None:
            return cached
        self._script._spent += 1
        if self._script._spent > self._script._budget:
            raise BudgetExceeded(self._script._budget, key)
        async with self._script._semaphore:
            result = await self._script._mesh.request(target, signal, timeout=timeout)
        self._script._store.put(key, self._script.name, self._phase, result)
        return result


class Script:
    def __init__(
        self,
        name: str,
        mesh: Mesh,
        fn: Callable[[ScriptContext], Awaitable[Any]],
        *,
        max_concurrency: int = 16,
        budget: int = 1000,
        store: CheckpointStore | None = None,
    ) -> None:
        self.name = name
        self._mesh = mesh
        self._fn = fn
        self._budget = budget
        self._spent = 0
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._store = store if store is not None else CheckpointStore()

    async def run(self, args: Any = None) -> Any:
        self._spent = 0
        return await self._fn(ScriptContext(self, args))
```

(`Agent.name` is a plain attribute — agent.py:308 — so the unique-rename in `spawn` is a straight assignment. Note `_key` consumes one occurrence slot per issued step in program order, which keeps numbering deterministic across runs.)

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_script.py -q`

- [ ] **Step 5: Commit** (when authorized)

`git add src/signal_gating/script.py tests/test_script.py && git commit -m "feat(script): checkpointed workflow runtime with fan-out, spawn, budget"`

---

## Task 8: Exports, examples, README, full verification gate

**Files:**
- Modify: `src/signal_gating/__init__.py`, `README.md`
- Create: `examples/agent_team.py`, `examples/scripted_workflow.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_taskboard.py`:

```python
def test_public_exports():
    import signal_gating as sg

    for name in (
        "TaskBoard", "Task", "Team", "Script", "ScriptContext", "CheckpointStore",
        "TaskRejected", "TeamError", "BudgetExceeded", "domain_payload",
    ):
        assert hasattr(sg, name), name
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_taskboard.py::test_public_exports -q`.

- [ ] **Step 3: Implement**

Add imports + `__all__` entries in `__init__.py` for: `TaskBoard`, `Task`, `TaskOpened`, `TaskClaimed`, `TaskReleased`, `TaskCompleted` (from `taskboard`), `Team`, `Mail`, `TaskAssigned`, `TaskResult`, `MemberIdle` (from `team`), `Script`, `ScriptContext`, `CheckpointStore`, `step_key` (from `script`), `TaskRejected`, `TeamError`, `BudgetExceeded` (from `errors`), `domain_payload` (from `trajectory`).

`examples/agent_team.py` — a 3-member review team: `security`, `performance`, `tests` agents, each with a `TaskAssigned` handler replying a `TaskResult` of stub findings for `payload["path"]`; a `lead` agent printing `MemberIdle`; main opens one task per file plus a summary task `depends_on` all three, runs `async with mesh:` then `async with team:`, waits for the board to drain (the `drained` idiom from `tests/test_team.py`), prints each `Task.result` and `board.head_digest`.

`examples/scripted_workflow.py` — a checkpointed sweep: two stub `scanner` agents and one `verifier`; the flow fans `ScanReq(path=...)` over 10 fake paths in a `scan` phase, dedupes findings, `ctx.run`s each through the verifier in a `verify` phase; `CheckpointStore("sweep-checkpoints.jsonl")`; the module docstring tells the reader to Ctrl-C mid-run and rerun to watch resume skip completed steps.

README: add "Teams" and "Scripted workflows" sections after the existing orchestration content — one paragraph plus one trimmed code block each (lift from the spec's Usage sketches), linking the two examples.

- [ ] **Step 4: Run the full gate**

```
pytest -q
ruff check .
mypy src/
python examples/agent_team.py
python examples/scripted_workflow.py
```

All green; both examples run with deterministic stub agents, no LLM server.

- [ ] **Step 5: Commit** (when authorized)

`git add src/signal_gating/__init__.py examples/ README.md tests/ && git commit -m "feat: export teams/script API, add examples and README sections"`

---

## Done criteria (maps to spec success criteria)

1. `from signal_gating import TaskBoard, Team, Script, CheckpointStore` works; core imports remain stdlib + pydantic only (Task 8).
2. A 2-member team drains a dependent backlog via steward claims with zero double-assignments and survives a mid-run handler crash — release + DLQ, no stall — with tests synchronized on board events, not sleeps (Task 5).
3. An interrupted script rerun completes from checkpoints, re-executing only unfinished steps, asserted by counting `request_sent` mesh events on the second run (Task 7).
4. An edited, reordered, or interiorly-deleted board ledger entry fails chain verification; a tampered checkpoint record fails its digest (Tasks 4 and 6).
5. `pytest`, `ruff check .`, `mypy src/` (strict) all pass (Task 8 gate).
6. Both examples run end-to-end with deterministic stub agents (Task 8).
