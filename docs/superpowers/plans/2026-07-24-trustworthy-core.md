# Trustworthy Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SGP's existing core safe to trust under deadlines, mutable caller inputs, steward failures, and live pool scaling, then give adopters one explicit stable import surface.

**Architecture:** Preserve the existing `Signal` → `Gate` → `Agent` → `Mesh` model and its public method signatures. Fix reliability at the narrow ownership boundaries: request methods own their complete deadline, `Signal` owns recursively frozen container values, `Team` owns claimed-task recovery, and `AgentPool` owns per-worker built-in gate state. Add a small `signal_gating.core` facade without removing any compatibility imports from the package root.

**Tech Stack:** Python 3.10+, asyncio, Pydantic 2, pytest/pytest-asyncio, Ruff, strict mypy.

## Global Constraints

- Python support remains `>=3.10`; do not use `asyncio.timeout()`.
- Do not add a mandatory runtime dependency beyond `pydantic>=2.0`.
- Keep every existing public method signature and every existing package-root import working.
- A public `timeout` covers the whole operation, including interceptors, gates, outbox delivery, inbox backpressure, and response waiting.
- Timeout and cancellation cleanup must remove every temporary capture and pending request entry.
- Signal container immutability must be recursive for mappings, lists, and sets while preserving Pydantic serialization without warnings.
- Built-in stateful gates used by `AgentPool` must have independent state per worker, including workers added later by live scaling.
- A team steward must not erase a wakeup or strand a claimed task when mesh delivery raises.
- `signal_gating.core.__all__` is exactly `Agent`, `Gate`, `Mesh`, `MeshEvent`, `Receipt`, `Signal`, and `TrajectoryRecorder`, in that order.
- Follow strict TDD for every behavior change: record the expected RED failure, implement the minimum fix, then record GREEN output.
- Run `.venv/bin/ruff check .`, `.venv/bin/mypy src/`, and `.venv/bin/pytest -q` before the final handoff.

---

## File Structure

- `src/signal_gating/agent.py`: make `Agent.request()` own one end-to-end deadline.
- `src/signal_gating/mesh.py`: make request/scatter/race deadlines cover dispatch as well as response waiting.
- `src/signal_gating/_immutable.py`: recursively frozen JSON-container implementations used by signals.
- `src/signal_gating/signal.py`: freeze every signal field after Pydantic validation.
- `src/signal_gating/team.py`: preserve steward wakeups and release work after delivery failures.
- `src/signal_gating/gate.py`: expose `Gate.fork()` and fresh-state factories for built-in stateful/composite gates.
- `src/signal_gating/pool.py`: fork configured gates for each worker.
- `src/signal_gating/core.py`: explicit stable core facade.
- `tests/test_mesh_delivery_contract.py`: end-to-end mesh deadline regressions.
- `tests/test_agent.py`: agent outbox deadline regression.
- `tests/test_signal.py`: recursive immutability and serialization regressions.
- `tests/test_team.py`: steward failure and wakeup regressions.
- `tests/test_pool.py`: independent gate-state regressions, including scale-up.
- `tests/test_public_api.py`: stable core import contract.
- `README.md`: distinguish the stable core from experimental orchestration modules.

---

### Task 1: Make request deadlines cover delivery and response waiting

**Files:**
- Modify: `src/signal_gating/agent.py:512-524`
- Modify: `src/signal_gating/mesh.py:1194-1439`
- Modify: `src/signal_gating/mesh.py:1531-1693`
- Test: `tests/test_agent.py`
- Test: `tests/test_mesh_delivery_contract.py`

**Interfaces:**
- Consumes: existing `Agent.request(signal, timeout)`, `Mesh.request(target, signal, timeout)`, `Mesh.scatter(signal, targets, timeout)`, and `Mesh.race(signal, targets, timeout)` signatures.
- Produces: the same return values and exceptions, with each timeout measured from method entry and all temporary correlation captures removed on every exit.

- [ ] **Step 1: Write failing deadline tests**

Add a reusable permanently blocked interceptor and tests with an outer safety timeout. The inner SGP timeout must fire first:

```python
async def never_returns(
    signal: Signal, _source: str, _target: str
) -> Signal:
    await asyncio.Event().wait()
    return signal


async def test_request_timeout_bounds_interceptor_and_removes_capture() -> None:
    worker = Agent("worker")
    mesh = Mesh([worker])
    mesh.intercept(never_returns)

    async with mesh:
        started = asyncio.get_running_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mesh.request(worker, Message(value=1), timeout=0.02),
                timeout=0.20,
            )
        elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.10
    assert worker._outbox == []
```

Add equivalent mesh tests for `scatter()` and `race()` with two targets. Assert all installed capture functions are removed from both target outboxes. Add this agent test:

```python
async def test_scatter_timeout_bounds_dispatch_and_removes_captures() -> None:
    workers = [Agent("worker-a"), Agent("worker-b")]
    mesh = Mesh(workers)
    mesh.intercept(never_returns)

    async with mesh:
        started = asyncio.get_running_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mesh.scatter(Message(value=1), workers, timeout=0.02),
                timeout=0.20,
            )
        elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.10
    assert all(worker._outbox == [] for worker in workers)


async def test_race_timeout_bounds_dispatch_and_removes_captures() -> None:
    workers = [Agent("worker-a"), Agent("worker-b")]
    mesh = Mesh(workers)
    mesh.intercept(never_returns)

    async with mesh:
        started = asyncio.get_running_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mesh.race(Message(value=1), workers, timeout=0.02),
                timeout=0.20,
            )
        elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.10
    assert all(worker._outbox == [] for worker in workers)
```

The request test must also measure elapsed time and assert `elapsed < 0.10`; otherwise both the inner and outer timeout raise the same exception and the regression would not be detected. Add this agent test:

```python
async def test_agent_request_timeout_bounds_blocked_outbox_and_cleans_pending() -> None:
    agent = Agent("requester")

    async def blocked(_signal: Signal) -> None:
        await asyncio.Event().wait()

    agent._add_output(blocked)

    started = asyncio.get_running_loop().time()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            agent.request(Signal(), timeout=0.02),
            timeout=0.20,
        )

    assert asyncio.get_running_loop().time() - started < 0.10
    assert agent._pending_requests == {}
```

- [ ] **Step 2: Run the focused tests and capture RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_mesh_delivery_contract.py::test_request_timeout_bounds_interceptor_and_removes_capture \
  tests/test_mesh_delivery_contract.py::test_scatter_timeout_bounds_dispatch_and_removes_captures \
  tests/test_mesh_delivery_contract.py::test_race_timeout_bounds_dispatch_and_removes_captures \
  tests/test_agent.py::test_agent_request_timeout_bounds_blocked_outbox_and_cleans_pending
```

Expected: elapsed-time assertions fail because the outer `0.20` second guard fires; after the fix, the inner `0.02` second deadline fires first.

- [ ] **Step 3: Put delivery and response wait under one timeout**

Restructure each method so `asyncio.wait_for()` wraps an inner coroutine beginning before the first delivery await. Keep cleanup in the method's existing outer `finally` block:

```python
async def operation() -> Signal:
    outcome = await self._deliver(...)
    self._raise_if_blocked(...)
    return await future

try:
    return await asyncio.wait_for(operation(), timeout=timeout)
finally:
    try:
        resolved._outbox.remove(capture)
    except ValueError:
        pass
```

For `Agent.request()`, the inner operation is `await self.emit(request_signal)` followed by `await future`. The existing `_pending_requests.pop(cid, None)` stays in the outer `finally`.

For `scatter()` and `race()`, wrap the complete dispatch-and-wait algorithm in one inner coroutine. On timeout, preserve the existing actionable missing-target message by deriving names from unresolved futures before raising `asyncio.TimeoutError`. Their outer `finally` blocks must remove every capture and consume completed-future exceptions as they do today.

- [ ] **Step 4: Run focused GREEN and related orchestration tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_mesh_delivery_contract.py \
  tests/test_agent.py \
  tests/test_agent_native_primitives.py \
  tests/test_agent_native_orchestration.py \
  tests/test_llm_tools.py
```

Expected: all selected tests pass with no warning output.

- [ ] **Step 5: Commit the bounded-deadline fix**

```bash
git add src/signal_gating/agent.py src/signal_gating/mesh.py \
  tests/test_agent.py tests/test_mesh_delivery_contract.py
git commit -m "fix: enforce end-to-end orchestration deadlines"
```

---

### Task 2: Make Signal container values recursively immutable

**Files:**
- Create: `src/signal_gating/_immutable.py`
- Modify: `src/signal_gating/signal.py:1-60`
- Test: `tests/test_signal.py`

**Interfaces:**
- Consumes: every Pydantic-validated field on `Signal` and its subclasses.
- Produces: `deep_freeze(value: Any) -> Any`; private `_FrozenDict`, `_FrozenList`, and `_FrozenSet` container subclasses; recursively immutable signal values that retain `dict`, `list`, and `set` runtime compatibility and serialize normally.

- [ ] **Step 1: Write failing recursive immutability tests**

Add a representative nested signal and test both mutation and input aliasing:

```python
class NestedSignal(Signal):
    payload: dict[str, object]
    items: list[dict[str, int]]
    labels: set[str]


def test_signal_recursively_freezes_mutable_fields_and_input_aliases() -> None:
    payload = {"auth": {"amount": 1}}
    items = [{"value": 1}]
    labels = {"approved"}
    signal = NestedSignal(payload=payload, items=items, labels=labels)

    payload["auth"]["amount"] = 1000  # type: ignore[index]
    items[0]["value"] = 1000
    labels.add("tampered")

    assert signal.payload["auth"] == {"amount": 1}
    assert signal.items == [{"value": 1}]
    assert signal.labels == {"approved"}

    with pytest.raises(TypeError):
        signal.payload["auth"]["amount"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        signal.items[0]["value"] = 2
    with pytest.raises(TypeError):
        signal.items.append({"value": 2})
    with pytest.raises(TypeError):
        signal.labels.add("blocked")
```

Add a wire round-trip test inside `warnings.catch_warnings()` with `warnings.simplefilter("error")`. It must call `signal.to_wire()`, reconstruct with `Signal.from_wire()`, and compare every nested value without emitting a Pydantic serializer warning.

- [ ] **Step 2: Run the focused tests and capture RED**

Run:

```bash
.venv/bin/pytest -q tests/test_signal.py
```

Expected: nested input aliases and nested container mutations change the signal today, so the new test fails.

- [ ] **Step 3: Implement JSON-container-compatible frozen subclasses**

Create `src/signal_gating/_immutable.py` with these interfaces:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn


def _immutable(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise TypeError("signal containers are immutable")


class _FrozenDict(dict[Any, Any]):
    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class _FrozenList(list[Any]):
    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable


class _FrozenSet(set[Any]):
    add = _immutable
    clear = _immutable
    difference_update = _immutable
    discard = _immutable
    intersection_update = _immutable
    pop = _immutable
    remove = _immutable
    symmetric_difference_update = _immutable
    update = _immutable
    __ior__ = _immutable
    __iand__ = _immutable
    __ixor__ = _immutable
    __isub__ = _immutable


def deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(deep_freeze(item) for item in value)
    if isinstance(value, set):
        return _FrozenSet(deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(deep_freeze(item) for item in value)
    return value
```

Use normal base-container constructors so construction succeeds before mutation methods are disabled. Keep these classes private.

- [ ] **Step 4: Freeze every validated Signal field**

In `Signal`, add an inherited Pydantic `model_validator(mode="after")`:

```python
@model_validator(mode="after")
def _freeze_containers(self) -> Signal:
    for field_name in type(self).model_fields:
        object.__setattr__(self, field_name, deep_freeze(getattr(self, field_name)))
    return self
```

Keep the existing metadata serializer so metadata remains a plain JSON object on the wire. Update the class docstring to state that nested mappings, lists, and sets are frozen too.

- [ ] **Step 5: Run focused GREEN and serialization coverage**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_signal.py \
  tests/test_serialization.py \
  tests/test_llm_tools.py \
  tests/test_taskboard.py \
  tests/test_trajectory.py
```

Expected: all selected tests pass with no serializer warnings.

- [ ] **Step 6: Commit recursive signal immutability**

```bash
git add src/signal_gating/_immutable.py src/signal_gating/signal.py tests/test_signal.py
git commit -m "fix: make signal payloads recursively immutable"
```

---

### Task 3: Keep team stewards alive and wakeup-safe

**Files:**
- Modify: `src/signal_gating/team.py:220-274`
- Test: `tests/test_team.py`

**Interfaces:**
- Consumes: `_Steward.wake`, `_Steward.current`, `Team._run_steward()`, and existing `TaskBoard.release()` transitions.
- Produces: no public API change; mesh delivery failures release the task with `request_error:<ExceptionType>`, mark it failed for that steward, and allow peer recovery.

- [ ] **Step 1: Write failing delivery-failure and lost-wakeup tests**

For delivery failure, stop an enrolled member before work is assigned, wait for `TaskReleased`, and assert the steward task is still running:

```python
async def test_delivery_failure_releases_task_without_killing_steward() -> None:
    worker = make_worker("worker")
    mesh = Mesh([worker])
    team = Team("t", mesh, task_timeout=0.2)
    team.enroll(worker)

    async with mesh:
        await team.start()
        await worker.stop()
        released = asyncio.Event()
        team.board.on_event(
            lambda event: released.set()
            if event.wire_type() == "sgp.task.released"
            else None
        )
        task_id = await team.open("recover me")
        await asyncio.wait_for(released.wait(), timeout=1.0)
        assert team.board.task(task_id).status == "pending"
        steward = team._stewards["worker"]
        assert steward.runner is not None
        assert not steward.runner.done()
        assert steward.current is None
        await team.dissolve()
```

For the wakeup race, block `MemberIdle` delivery with an interceptor, open a second task while the steward is awaiting that notification, then release the interceptor. Assert the second task completes without another external wakeup.

```python
async def test_board_event_during_idle_notification_is_not_lost() -> None:
    lead, worker = Agent("lead"), make_worker("worker")
    idle_entered = asyncio.Event()
    release_idle = asyncio.Event()

    async def block_idle(signal, _source, _target):
        if isinstance(signal, MemberIdle):
            idle_entered.set()
            await release_idle.wait()
        return signal

    mesh = Mesh([lead, worker])
    mesh.intercept(block_idle)
    team = Team("t", mesh)
    team.lead(lead)
    team.enroll(worker)

    async with mesh:
        async with team:
            first = await team.open("first")
            await drained(team.board)
            assert team.board.task(first).status == "completed"
            await asyncio.wait_for(idle_entered.wait(), timeout=1.0)
            second = await team.open("second")
            release_idle.set()
            await drained(team.board)
            assert team.board.task(second).status == "completed"
```

- [ ] **Step 2: Run the focused tests and capture RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_team.py::test_delivery_failure_releases_task_without_killing_steward \
  tests/test_team.py::test_board_event_during_idle_notification_is_not_lost
```

Expected: the first steward exits with `ChannelClosed`; the second task remains pending because the event is cleared after notification.

- [ ] **Step 3: Clear wakeups before checking work and catch delivery failures**

Move `steward.wake.clear()` to the top of each loop iteration, before checking `assigned` or calling `_claim_next()`. When no task exists, notify idle and then await the event without clearing it again:

```python
while not steward.stopping:
    steward.wake.clear()
    task = (
        steward.assigned.popleft()
        if steward.assigned
        else await self._claim_next(member, steward)
    )
    if task is None:
        if steward.worked:
            steward.worked = False
            await self._notify_idle(member)
        await steward.wake.wait()
        continue
```

After the existing `TaskRejected` branch, catch other `Exception` values, mark the task failed for this steward, and release it:

```python
except Exception as error:
    steward.failed.add(task.id)
    await self.board.release(
        task.id,
        member,
        reason=f"request_error:{type(error).__name__}",
    )
```

Do not catch `asyncio.CancelledError`; shutdown relies on cancellation leaving `steward.current` set until `shutdown()` releases it.

- [ ] **Step 4: Run focused GREEN and the complete team/taskboard suites**

Run:

```bash
.venv/bin/pytest -q tests/test_team.py tests/test_taskboard.py
```

Expected: all tests pass and steward task exceptions are fully observed.

- [ ] **Step 5: Commit steward resilience**

```bash
git add src/signal_gating/team.py tests/test_team.py
git commit -m "fix: preserve team work across steward failures"
```

---

### Task 4: Give every pool worker independent built-in gate state

**Files:**
- Modify: `src/signal_gating/gate.py:20-687`
- Modify: `src/signal_gating/pool.py:109-124`
- Test: `tests/test_pool.py`

**Interfaces:**
- Consumes: existing `Gate` factories and composition operators.
- Produces: `Gate.fork() -> Gate`; fresh wrapper instances for all gates and fresh closure state for built-in stateful gates and compositions; `AgentPool` calls `fork()` once per configured gate per worker.

- [ ] **Step 1: Write failing pool isolation tests**

Add a deterministic throttle test that sends the first signal directly to both workers:

```python
async def test_pool_workers_have_independent_builtin_gate_state() -> None:
    pool = AgentPool("workers", size=2, gates=[Gate.throttle(1)])
    received: list[str] = []

    @pool.on(TaskSignal)
    async def handle(signal: TaskSignal, ctx: AgentContext) -> None:
        received.append(ctx.agent_name)

    mesh = Mesh()
    mesh.add_pool(pool)
    async with mesh:
        await pool.workers[0].inbox.send(TaskSignal(task="a"))
        await pool.workers[1].inbox.send(TaskSignal(task="b"))
        await mesh.wait_idle()

    assert sorted(received) == ["workers[0]", "workers[1]"]
    assert pool.workers[0].gates[0] is not pool.workers[1].gates[0]
```

Add a live-scale test: consume one throttled signal on the original worker, scale up, then confirm the new worker's first direct signal passes immediately. Assert its gate is distinct from every existing worker gate.

```python
async def test_scaled_worker_gets_fresh_builtin_gate_state() -> None:
    pool = AgentPool("workers", size=1, gates=[Gate.throttle(1)])
    received: list[str] = []

    @pool.on(TaskSignal)
    async def handle(signal: TaskSignal, ctx: AgentContext) -> None:
        received.append(ctx.agent_name)

    mesh = Mesh()
    mesh.add_pool(pool)
    original = pool.workers[0]

    async with mesh:
        await original.inbox.send(TaskSignal(task="original"))
        await mesh.wait_idle()
        added = await mesh.scale_pool(pool, 2)
        assert len(added) == 1
        await added[0].inbox.send(TaskSignal(task="scaled"))
        await mesh.wait_idle()

    assert received == ["workers[0]", "workers[1]"]
    assert added[0].gates[0] is not original.gates[0]
```

- [ ] **Step 2: Run the focused tests and capture RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_pool.py::TestPoolCreation::test_pool_workers_have_independent_builtin_gate_state \
  tests/test_pool.py::TestPoolScaling::test_scaled_worker_gets_fresh_builtin_gate_state
```

Expected: only one first signal passes because the current workers reference the same throttle gate and closure state.

- [ ] **Step 3: Add Gate.fork() and factory-backed built-in state**

Extend `Gate.__init__` with a private keyword-only factory and add `fork()`:

```python
GateFactory = Callable[[], "Gate"]


def __init__(
    self,
    fn: GateFn,
    name: str = "",
    *,
    _factory: GateFactory | None = None,
) -> None:
    self._fn = fn
    self.name = name or (fn.__name__ if hasattr(fn, "__name__") else "gate")
    self._factory = _factory


def fork(self) -> Gate:
    """Return a fresh gate wrapper, including fresh built-in gate state."""
    if self._factory is not None:
        return self._factory()
    return Gate(self._fn, name=self.name)
```

Every built-in stateful factory must provide a factory that calls the same classmethod with the same arguments: `rate_limit`, `deduplicate`, `circuit_breaker`, `throttle`, `batch`, `debounce`, and `window`. Wrappers/composites that contain gates must fork their children in their factory: `>>`, `|`, `&`, `~`, `retry`, `timeout`, `when`, `parallel`, and `fallback`. Stateless factories still return a new wrapper from `fork()`; caller-owned function closure state remains caller-owned and is documented as such.

Example for throttle:

```python
return cls(
    fn,
    name=name,
    _factory=lambda: cls.throttle(max_per_second, name=name),
)
```

Example for chaining:

```python
return Gate(
    chained,
    name=f"{self.name}>>{other.name}",
    _factory=lambda: self.fork() >> other.fork(),
)
```

- [ ] **Step 4: Fork configured gates when each worker is created**

Change `AgentPool._create_worker()` from `gates=list(self._gates)` to:

```python
gates=[gate.fork() for gate in self._gates]
```

Update the pool docstring to promise independent state for built-in gates, while noting that mutable closure state inside a caller-provided raw `Gate(fn)` remains owned by that callable.

- [ ] **Step 5: Run focused GREEN and all gate/pool tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_gate.py \
  tests/test_agent_native_primitives.py \
  tests/test_agent_native_enhancements.py \
  tests/test_pool.py
```

Expected: all selected tests pass with no timing warnings or leaked tasks.

- [ ] **Step 6: Commit independent pool gate state**

```bash
git add src/signal_gating/gate.py src/signal_gating/pool.py tests/test_pool.py
git commit -m "fix: isolate built-in gate state per pool worker"
```

---

### Task 5: Publish one explicit stable core surface

**Files:**
- Create: `src/signal_gating/core.py`
- Create: `tests/test_public_api.py`
- Modify: `README.md:3-73`
- Modify: `README.md:788-807`

**Interfaces:**
- Consumes: existing public classes from their owning modules.
- Produces: `from signal_gating.core import Agent, Gate, Mesh, MeshEvent, Receipt, Signal, TrajectoryRecorder`; no removals or changes to `signal_gating.__init__` compatibility imports.

- [ ] **Step 1: Write the failing public API contract test**

Create `tests/test_public_api.py`:

```python
from signal_gating import Agent, Gate, Mesh, MeshEvent, Receipt, Signal, TrajectoryRecorder
from signal_gating import core


def test_stable_core_exports_exact_contract() -> None:
    assert core.__all__ == [
        "Agent",
        "Gate",
        "Mesh",
        "MeshEvent",
        "Receipt",
        "Signal",
        "TrajectoryRecorder",
    ]
    assert core.Agent is Agent
    assert core.Gate is Gate
    assert core.Mesh is Mesh
    assert core.MeshEvent is MeshEvent
    assert core.Receipt is Receipt
    assert core.Signal is Signal
    assert core.TrajectoryRecorder is TrajectoryRecorder
```

- [ ] **Step 2: Run the test and capture RED**

Run:

```bash
.venv/bin/pytest -q tests/test_public_api.py
```

Expected: import fails because `signal_gating.core` does not exist.

- [ ] **Step 3: Add the stable facade**

Create `src/signal_gating/core.py`:

```python
"""The stable Signal Gating Protocol core.

This module is the compatibility-focused surface for production integrations.
Broader orchestration helpers remain available from their owning modules and
the package root while the project is alpha.
"""

from signal_gating.agent import Agent
from signal_gating.gate import Gate
from signal_gating.mesh import Mesh, MeshEvent
from signal_gating.signal import Signal
from signal_gating.trajectory import Receipt, TrajectoryRecorder

__all__ = [
    "Agent",
    "Gate",
    "Mesh",
    "MeshEvent",
    "Receipt",
    "Signal",
    "TrajectoryRecorder",
]
```

- [ ] **Step 4: Rewrite the README's adoption path around the stable core**

Immediately after Quick Start, add a `## Stable core` section that tells production adopters to import the seven names from `signal_gating.core`. State that package-root imports remain compatible in `0.1.x`, while pools, teams, scripts, LLM helpers, task boards, and improvement loops are alpha modules whose APIs may change before `1.0`.

Update the architecture section so the diagram names the stable core and shows experimental orchestration as consumers of it. Keep the existing detailed examples; do not delete working documentation.

- [ ] **Step 5: Run public API and documentation-adjacent checks**

Run:

```bash
.venv/bin/pytest -q tests/test_public_api.py tests/test_codebase_review.py
.venv/bin/ruff check src/signal_gating/core.py tests/test_public_api.py
.venv/bin/mypy src/
```

Expected: all commands exit 0 with no warnings.

- [ ] **Step 6: Commit the stable core facade**

```bash
git add src/signal_gating/core.py tests/test_public_api.py README.md
git commit -m "feat: define the stable SGP core surface"
```

---

## Final Verification

- [ ] **Step 1: Run the full suite**

```bash
.venv/bin/pytest -q
```

Expected: all tests pass and output contains no warnings or leaked-task messages.

- [ ] **Step 2: Run static verification**

```bash
.venv/bin/ruff check .
.venv/bin/mypy src/
```

Expected: both commands exit 0.

- [ ] **Step 3: Re-run the two original failure probes as regression evidence**

Run the request interceptor probe with an inner timeout of `0.01` seconds and an outer guard of `0.08` seconds. It must return at the inner deadline. Construct a nested signal from caller-owned dictionaries, mutate the original dictionaries, and confirm the signal retains the admitted values and raises `TypeError` on nested mutation.

- [ ] **Step 4: Review the branch against all Global Constraints**

Confirm every existing public signature and root import remains intact, all temporary correlation state is cleaned on timeout, built-in pool gates are distinct across initial and scaled workers, and `core.__all__` contains exactly seven names.
