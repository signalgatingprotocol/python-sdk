# Live AgentPool Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Close issue #34 by making live pool scaling preserve mesh topology, routing, lifecycle, and discovery invariants.

**Architecture:** `AgentPool` continues to own worker construction and configuration. Once attached, `Mesh` becomes the exclusive owner of membership, routing, and lifecycle changes. Pool-target routes select from current workers at delivery time, while logical pool connection policies configure newly added source workers.

**Tech Stack:** Python 3.10+, asyncio, pytest, Ruff, mypy.

## Global Constraints

- Start from the latest `origin/main` in an isolated `fix/live-pool-scaling` worktree.
- Preserve unattached `AgentPool.scale_to()`, `scale_up()`, and `scale_down()` behavior.
- Attached pools may only be resized through `Mesh.scale_pool()`.
- Pool size must remain at least one.
- No new dependencies, wire-format changes, or unrelated refactors.
- Each behavior follows RED, GREEN, REFACTOR and receives an independent review.
- `Mesh.scale_pool(pool: AgentPool | str, size: int) -> list[Agent]` is the only new public interface.
- Scaling up returns workers in ascending creation order; scaling down returns workers in descending removal order; a no-op returns `[]`.
- Invalid sizes raise `ValueError("Pool size must be at least 1")`.
- Direct attached-pool mutation raises `MeshError("Pool 'workers' is attached to a mesh; use await mesh.scale_pool('workers', size)")`.

---

### Task 1: Establish exclusive mesh ownership

**Files:**
- Modify: `src/signal_gating/pool.py`
- Modify: `src/signal_gating/mesh.py`
- Test: `tests/test_pool.py`

**Interfaces:**
- Produces private `AgentPool` ownership and staged-worker helpers used by Task 2.
- Preserves all unattached pool scaling behavior.

- [ ] Add failing tests proving `scale_to`, `scale_up`, `scale_down`, and `discard` reject direct mutation after `add_pool()`, while unattached behavior remains unchanged.
- [ ] Add a failing test proving one pool instance cannot attach to two meshes and that the second mesh remains unchanged.
- [ ] Add a failing test proving `Mesh.remove()` rejects removal of a pool's final worker.
- [ ] Run `.venv/bin/pytest tests/test_pool.py -q` and confirm failures come from missing ownership enforcement.
- [ ] Add `_mesh_owner`, `_claim()`, `_assert_detached()`, `_prepare_workers()`, `_commit_workers()`, `_take_workers()`, and `_discard_worker()` internals to `AgentPool`.
- [ ] Make `Mesh.add_pool()` claim ownership only after all collision checks pass. Make mesh cleanup use the private discard path.
- [ ] Re-run `.venv/bin/pytest tests/test_pool.py -q`, then the full suite once.
- [ ] Commit as `fix(pool): make attached mesh the scaling owner`.

### Task 2: Preserve live routing and topology

**Files:**
- Modify: `src/signal_gating/mesh.py`
- Test: `tests/test_pool.py`

**Interfaces:**
- Consumes Task 1's ownership and staged-worker helpers.
- Produces `async Mesh.scale_pool(pool: AgentPool | str, size: int) -> list[Agent]`.
- Produces immutable `_PoolConnection(source, target, gate)` policy records.

- [ ] Add failing integration tests for agent-to-pool, pool-to-agent, and pool-to-pool connections while scaling `1 -> 3 -> 1` inside `async with mesh`.
- [ ] Prove new workers are registered, traced, wired, and running when the mesh is running. Prove scaling a stopped mesh registers workers without starting them.
- [ ] Prove removed workers are stopped and absent from membership, `Mesh.get()`, capabilities, edges, topics, and routing closures.
- [ ] Run `.venv/bin/pytest tests/test_pool.py -q` and confirm the failures demonstrate captured target lists and missing mesh-owned scaling.
- [ ] Add `_PoolConnection` records and retain every connection involving a pool.
- [ ] Replace captured target-worker lists with `_connect_to_pool()`, which calls `pool.select_worker(signal)` at delivery time and applies the original gate exactly once.
- [ ] Add `_wire_pool_source_worker()` so new source workers inherit every outbound and pool-to-pool policy.
- [ ] Implement `Mesh.scale_pool()` behind one asyncio lock. Scaling up stages workers, registers and wires them, starts them when required, then commits membership. Scaling down removes workers from selection before stopping and cleaning them up.
- [ ] Prune logical policies when an individual endpoint is removed, while retaining policies owned by a still-attached pool.
- [ ] Run `.venv/bin/pytest tests/test_pool.py tests/test_mesh.py -q`, then the full suite once.
- [ ] Commit as `fix(mesh): preserve invariants during live pool scaling`.

### Task 3: Make failure semantics atomic and document them

**Files:**
- Modify: `src/signal_gating/mesh.py`
- Test: `tests/test_pool.py`
- Modify: `README.md`

**Interfaces:**
- Keeps the Task 2 public signature unchanged.
- Defines rollback and graceful-retirement behavior for callers.

- [ ] Add a failing test where a new worker's start hook fails. Assert original capacity, membership, routes, and running workers are restored and all staged workers are stopped.
- [ ] Add a lifecycle test proving retiring workers stop accepting pool-routed work immediately but drain queued and in-flight work before `scale_pool()` returns.
- [ ] Make scale-up rollback unregister, unwire, and stop every staged worker on any `BaseException`. Make cancellation finish cleanup before propagating.
- [ ] Make scale-down complete retirement cleanup before propagating cancellation, preventing detached running workers.
- [ ] Add a concise README `Live pool scaling` section covering the mesh-owned API, minimum size, running versus stopped behavior, inherited connection policies, graceful retirement, rollback, and direct-mutation errors.
- [ ] Run `.venv/bin/pytest tests/test_pool.py -q`, then `.venv/bin/pytest -q -p no:cacheprovider`, `.venv/bin/ruff check .`, `.venv/bin/mypy --no-incremental src/`, and `git diff --check`.
- [ ] Commit as `docs(pool): define live scaling semantics`.

## Final Verification

- `.venv/bin/pytest -q -p no:cacheprovider`
- `.venv/bin/ruff check .`
- `.venv/bin/mypy --no-incremental src/`
- `git diff --check`
- The full `1 -> 3 -> 1` scenario covers pool source, target, and both sides of pool-to-pool routing.
- A final whole-branch review must approve the merge-base diff.

