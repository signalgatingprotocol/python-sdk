# Agent Teams & Scripted Workflows: Coordinated Multi-Agent Work at Scale

- **Date:** 2026-06-10
- **Status:** Design — approved, proceeding to plan
- **Repo:** `signalgatingprotocol/python-sdk` (branch `claude/agent-teams-workflows-d7kbfn`)
- **Scope:** v1 of the "orchestration at scale" direction. Adds two subsystems on top of the existing mesh: **teams** (long-lived peer agents coordinating through a shared, durable task ledger with direct messaging) and **scripts** (a script-held workflow runtime that fans out bounded swarms of agents with checkpointed, resumable results).

## Problem

The mesh has orchestration *verbs* — `request`, `scatter`, `race`, `workflow`, `map_reduce`, `call_tool` — but no orchestration *state*. Every multi-agent example in the repo coordinates by hand: closures collecting results, `asyncio.sleep` for convergence, attempt counters threaded through `evolve()`. Two distinct shapes of work have no home:

1. **Team-shaped work.** Several long-lived agents share a backlog, claim tasks as they free up, message each other directly, and wind down cleanly. Today there is no task list, no claim semantics, no idle/shutdown protocol — each user reinvents them, badly, with ad-hoc signals.
2. **Script-shaped work.** A sweep over 500 files, a fan-out review, a research run that touches dozens of agents. Today the "plan" lives in the calling coroutine and dies with it: interrupt the run and every completed result is lost, because nothing maps a step to its result durably.

Both reduce to the same missing primitive: **durable, append-only coordination state derived from signals**, which the SDK is already uniquely positioned to provide — signals are immutable, self-describing on the wire, and digest-verified as Receipts.

## Goal

A `TaskBoard` (shared, durable, gate-checked task ledger), a `Team` (coordination protocol over a board: enroll, assign, self-claim, message, idle, shutdown, dissolve — driven by team-owned steward coroutines, not by member internals), and a `Script` (a user-authored async workflow executed against a mesh with bounded concurrency, an agent budget, and content-addressed checkpoints so an interrupted run resumes without redoing finished work). All three compose from existing primitives — `Signal`, `Gate`, `Agent`, `Mesh`, Receipts — through public seams only.

## Non-goals (v1)

- **Cross-process or distributed teams.** The board and mesh are single-process, like everything else in the SDK. The JSONL ledgers are the seam a future distributed layer attaches to.
- **A plan-approval protocol** (teammate drafts in read-only mode, lead approves). Clean v2 extension of the request/reply task contract below.
- **Nested teams or scripts spawning scripts.** One level. A script's ephemeral agents are plain `Agent`s.
- **`AgentPool` targets in scripts.** `mesh.request` resolves single agents only; pool-aware checkpoint keys are a v2 question. Scripts target agents (preregistered or spawned).
- **LLM-aware anything.** An `LLMAgent` enrolls in a team or serves a script exactly like any agent. No token accounting, no model routing.
- **Progress UI.** Observability is the existing surface: tracer spans, mesh events, board events. Rendering is someone else's job.
- **Mid-run input to scripts.** A script runs to completion or stops; for sign-off between stages, run each stage as its own script.
- Any change to `Signal`, `Gate`, `Channel`, or the base `Agent`. `Mesh` changes are limited to hardening the existing `remove()` (below). The existing `mesh.workflow()` step-chain helper is untouched; the `Script` runtime is a different, unrelated thing, and the name avoids the collision.

## The primitive: signal-sourced coordination state

One idea underlies both subsystems: **coordination state is a fold over an append-only log of signals.** A task board is not a mutable table; it is the projection of `TaskOpened`/`TaskClaimed`/`TaskReleased`/`TaskCompleted` events. A script's progress is not runtime bookkeeping; it is the set of checkpoint records its steps have committed. Because every entry is a `Signal`, the wire format, lineage threading, and JSONL persistence come for free — the same machinery Receipts use today.

Consequences:

- **Durability**: `export_jsonl`/`load_jsonl` round-trips the ledger; a restarted process reconstructs the board (with explicit crash-recovery semantics, below).
- **Tamper-evidence**: ledger entries are **hash-chained** — each record's sha256 covers its sequence number, its event envelope, and the previous record's digest. Editing, reordering, or deleting an interior entry breaks the chain on load. (Truncating the tail is detectable only against an externally stored head digest; `board.head_digest` exposes it, storing it elsewhere is the caller's job. Stated honestly in docs.)
- **Auditability**: board events are ordinary signals; a team's history reads as a trajectory.
- **Policy as gates**: transition rules ("no task closes without a result", "max 50 open tasks") are ordinary `Gate`s over the event signal, with the full combinator algebra (`>>`, `|`, `&`, `~`).

## `TaskBoard` (`src/signal_gating/taskboard.py`)

### Event signals

All protocol signals pin `__signal_type__` (`"sgp.task.opened"`, `"sgp.task.claimed"`, …) so a user defining their own `TaskCompleted` class can never collide with the board's wire names in the global registry. All `Mapping` fields use a shared `FrozenPayload` mixin providing the same `field_validator`/`field_serializer` pair `Signal.metadata` uses (per-field freezing is not inherited; the mixin is where the boilerplate lives once).

```python
class TaskOpened(Signal):     # __signal_type__ = "sgp.task.opened"
    task_id: str; brief: str = ""; depends_on: tuple[str, ...] = ()
    payload: Mapping[str, Any] = {}     # arbitrary JSON-safe data for the worker
class TaskClaimed(Signal):    task_id: str; member: str = ""
class TaskReleased(Signal):   task_id: str; member: str = ""; reason: str = ""
class TaskCompleted(Signal):  task_id: str; member: str = ""; result: Mapping[str, Any] = {}
```

Every event after `TaskOpened` is a `child()` of the task's opening event, so a task's full life is one trace. `payload` and `result` are validated JSON-safe **at transition time** (a `to_wire` round-trip check in `open()`/`complete()`), so `export_jsonl` can never fail later on data the board already accepted.

### API

```python
board = TaskBoard(
    "docs-sweep",
    open_gate=None,        # Gate | None — evaluated on TaskOpened; rejection raises TaskRejected
    complete_gate=None,    # Gate | None — evaluated on TaskCompleted; rejection raises TaskRejected
)

tid  = await board.open("fix channel docs", depends_on=(other_tid,), priority=5, payload={...})
task = await board.claim("alice")                # highest-priority claimable task, or None
task = await board.claim("alice", task_id=tid)   # targeted claim; None if not claimable
await board.complete(tid, "alice", result={...})
await board.release(tid, "alice", reason="blocked on review")

board.task(tid) -> Task                          # frozen view: id, brief, status, member, depends_on, result
board.tasks() -> list[Task]
board.claimable() -> list[Task]                  # pending, all deps completed
board.events -> list[Signal]                     # the ledger, in order
board.head_digest -> str                         # chain head, for external anchoring
board.on_event(fn)                               # sync observer per event; advisory, errors swallowed & counted
board.export_jsonl(path)
TaskBoard.load_jsonl(path, release_in_progress=True)
```

- **Statuses are derived**, never stored: `pending` (opened, unclaimed), `in_progress` (claimed), `completed`. A pending task with unresolved `depends_on` is simply not claimable; completing the last dependency unblocks dependents with nothing to do.
- **Claiming is race-free** under one `asyncio.Lock`; concurrent `claim()` calls never double-assign, and the targeted form is what makes lead-directed assignment sound (no claim-whatever-is-on-top race).
- **Gates run outside the lock** (they are pure over the event signal; some built-ins like `Gate.rate_limit` sleep). Only the append and the claim decision hold the lock. Rejection raises `TaskRejected(task_id, gate_name)` — note gate combinators collapse names, so `gate_name` is the outermost gate's name, which is all the gate algebra can know.
- **Observers are pure observers** (the interceptor rule): they cannot block a transition; their exceptions are counted, not raised. Nothing load-bearing may sit behind one — see the team wake-up design below for why this matters.
- **Crash recovery is explicit.** `load_jsonl(release_in_progress=True)` (the default) appends a synthetic `TaskReleased(reason="recovered")` for every task claimed by a process that no longer exists — the fold stays an honest log, and a restart never wedges the board with phantom `in_progress` tasks. Pass `False` to reconstruct verbatim.

### Ledger record shape

A ledger line is not a bare wire envelope (no digest) and not a `Receipt` (no meaningful source/target here). It is a third, minimal record:

```python
{"seq": 14, "event": {<wire envelope>}, "prev": "<digest 13>", "digest": "sha256(seq|prev|event)"}
```

`load_jsonl` verifies the chain; any break raises `SignalSerializationError` with the failing sequence number.

## `Team` (`src/signal_gating/team.py`)

A `Team` is not a new kind of agent and installs **nothing inside member agents** beyond what members already own. The first design draft put the protocol in member-side middleware and self-stopping handlers; review against `agent.py` killed it — middleware wraps *per-handler* (misfires N times, never fires on unhandled signals), an agent cannot await its own `busy` flag or `stop()` itself from inside a handler, and a load-bearing wake-up built on swallowed observer errors stalls silently. The protocol therefore lives in **team-owned steward coroutines**, one per member, running in the Team — the same side of the boundary as `mesh.request`.

### Protocol signals

```python
class Mail(Signal):          to: str = ""; sender: str = ""; body: str = ""   # "sgp.team.mail"
class TaskAssigned(Signal):  task_id: str; brief: str = ""; payload: Mapping[str, Any] = {}
class TaskResult(Signal):    task_id: str; result: Mapping[str, Any] = {}
class MemberIdle(Signal):    member: str = ""
```

No shutdown signals: shutdown is an API call, not a conversation (below).

### API

```python
team = Team("docs-sweep", mesh, board=None,      # builds a TaskBoard when not given one
            task_timeout=60.0)

team.lead(planner)                               # optional; the conventional MemberIdle recipient
team.enroll(writer, role="implementer")          # any Agent, including LLMAgent
team.members -> dict[str, str]                   # name -> role

tid = await team.open("rewrite channel docs", payload={...})   # delegates to the board
await team.assign(tid, "writer")                 # targeted board.claim + steward pickup
await team.send("writer", "skip pool.py", sender="lead")       # Mail via mesh.inject
await team.shutdown("writer", timeout=None)      # team-side: stop claiming, drain, agent.stop()
await team.dissolve()                            # shutdown of all members + steward teardown
async with team: ...                             # start/dissolve
```

### The work contract: request/reply

A member's only obligation is the SDK's most idiomatic one: handle `TaskAssigned` and reply.

```python
@writer.on(TaskAssigned)
async def work(signal: TaskAssigned, ctx: AgentContext):
    ...do the work using signal.payload...
    await ctx.reply(TaskResult(task_id=signal.task_id, result={"summary": ...}))
```

The steward executes a task as `mesh.request(member, assigned, timeout=task_timeout)` where `assigned` is a `child()` of the claim event (one trace from open to completion). On reply: `board.complete(task_id, member, result)`. On timeout: `board.release(task_id, member, reason="timeout")`. A handler that raises is dead-lettered by the member's existing supervision and never replies — which surfaces as the same timeout, so a crashed teammate never strands a claimed task, and the truth of the crash is in the member's DLQ where it always lives. `payload` is plain JSON-safe data; members that want typed domain signals put a wire envelope in it and `Signal.from_wire` it themselves — the board does not reconstruct domain types on anyone's behalf.

### The steward loop

Each enrolled member gets one steward coroutine owned by the Team:

1. **Wake.** The steward waits on an `asyncio.Event`. The board's `on_event` observer for the Team does exactly one thing — `event.set()` — which cannot raise, keeping the observer honestly advisory while making wake-up loss-free. *Every* board event wakes stewards (including `TaskReleased`: a release re-pends work, and a completion may unblock dependents).
2. **Claim.** If the member is not shutting down: take an assigned-but-unstarted task if `assign()` queued one, else `board.claim(member)`. The steward holds **at most one task at a time** — no hoarding; unclaimed work stays visible to peers.
3. **Execute** via the request/reply contract above; record completion or release on the board.
4. **Idle, edge-triggered.** When a claim attempt finds nothing and the previous iteration did work, the steward sends one `MemberIdle` to the lead via `mesh.inject` — best-effort (caught and counted; a stopped or absent lead must not poison stewards). No lead, no send.

`shutdown(member)` is entirely team-side: flag the steward to stop claiming, await its in-flight task (bounded by `timeout`, default `task_timeout`), release anything claimed-but-unstarted, cancel the steward, then `await agent.stop()` from outside — the only place `stop()` can be awaited safely. `dissolve()` is shutdown for everyone plus board observer removal; it raises `TeamError` only on protocol misuse (duplicate enrollment, assigning an unclaimable task, dissolving a dissolved team). Members are peers throughout: any member can `Mail` any other by name, the lead holds zero machinery, and killing the lead degrades exactly one thing — nobody listens to `MemberIdle`.

## `Script` (`src/signal_gating/script.py`)

A script moves the plan into code: a user-authored async function owns the loop and the branching, and intermediate results live in script variables — not in any agent's context. The runtime contributes the three things a bare coroutine lacks: **bounded fan-out, an agent budget, and resumability**. (The name is deliberate: `mesh.workflow()` already means "step chain"; this is a different animal.)

### API

```python
async def audit(ctx: ScriptContext) -> dict:
    files = ctx.args["files"]
    async with ctx.phase("scan"):
        findings = await ctx.fan_out([scan_a, scan_b], [ScanReq(path=f) for f in files])
    async with ctx.phase("verify"):
        verified = [await ctx.run(verifier, Claim(text=c)) for c in dedupe(findings)]
    return summarize(verified)

script = Script("endpoint-audit", mesh, audit,
                max_concurrency=16, budget=1000,
                store=CheckpointStore("audit.jsonl"))   # store=None -> in-memory, no resume
out = await script.run(args={"files": files})           # rerun after interruption -> resumes
```

### `ScriptContext`

- `ctx.args` — invocation input, as passed to `run()`.
- `ctx.phase(name)` — async context manager that namespaces step keys and scopes tracer spans/mesh events for progress observation.
- `await ctx.run(target, signal, *, timeout=30.0, key=None)` — a checkpointed `mesh.request`. **Failure surfaces as `asyncio.TimeoutError`** — including the case where the target handler raised and was dead-lettered (no reply is ever sent on that path; the truth is in the target's DLQ). The script decides what a timeout means; the runtime never retries (`Gate.retry` composes on the target if wanted).
- `await ctx.fan_out(targets, signals, *, timeout=30.0)` — distributes signals round-robin across `targets` under the concurrency semaphore; results in input order. With one target this is serial by construction (an agent's run loop is serial) — parallelism comes from multiple targets or `spawn`.
- `await ctx.spawn(factory, signal, *, timeout=30.0)` — ephemeral agent: the factory builds an `Agent`, the runtime renames it `{name}#{n}` (unique per run, so concurrent spawns from one factory never collide in `mesh.add`), adds it, starts it, executes one checkpointed request, then stops and `mesh.remove()`s it. This is how a script uses dozens-to-hundreds of agents without preregistering them.

### Checkpoint keys

A step key is `sha256(script, phase, [target], wire_type, payload, occurrence)`:

- `payload` is the **domain-fields-only projection** — the existing `_domain_payload` in `trajectory.py`, promoted to a public `domain_payload(signal)` helper. It excludes `id`, `timestamp`, `trace_id`, `source`, `priority`, `correlation_id`, `parent_id`, **and `metadata`** — so volatile identity never invalidates a checkpoint, and neither does a priority or metadata tweak (deliberate; documented).
- `occurrence` is a per-(phase, content) counter assigned in issue order, so two steps with identical payloads in one phase get distinct keys — a retry loop or duplicate input is never silently deduped, no two in-flight steps can share a key, and the numbering is deterministic across runs (issue order is program order).
- `target` is included **only for `ctx.run`**, where the caller named a stable agent; `fan_out` and `spawn` exclude it because worker assignment is incidental (round-robin and generated names would otherwise break resume).
- `key=` overrides everything when the caller knows better.

### Checkpoints and resume

Every completed step appends `{script, phase, key, result: <wire envelope>, digest}` to the store (JSONL; per-record digest, verified on load; duplicate keys: last record wins, stated). On `run()`, a step whose key exists returns `Signal.from_wire(result)` immediately, touching neither the mesh nor the budget. Interruption is therefore free: stop the process, rerun, and only unfinished steps execute. There is no pause state — *resume is cache, not snapshot*. A changed input changes the key, so stale results can never replay against new inputs.

### Limits

- `max_concurrency` (default 16): a semaphore over all in-flight steps.
- `budget` (default 1000): total mesh-touching steps per run; exceeding it raises `BudgetExceeded` (carrying the budget and offending key) with completed checkpoints intact — a runaway loop costs at most one budget.

## Mesh prerequisite: harden the existing `remove()`

`Mesh.remove()` already exists (mesh.py:128) and handles edges, `connect`-tagged routes, topics, and capabilities. It is not yet safe under `ctx.spawn` churn; this work closes the gaps rather than adding anything:

- Purge the removed agent from `Mesh._pools` worker lists.
- `load_balance()` and `route()` register routing closures with `target=None` holding direct agent references; removal must rebuild or filter those target lists so a removed agent's closed inbox is never sent to.
- Document the in-flight hazard: clearing the removed agent's `_outbox` drops any live `mesh.request` capture functions; the requester then times out rather than erroring — acceptable, but it must be written down.

## Error handling

- `TaskRejected(SignalGatingError)` — an open/complete gate refused a transition; carries task id and (outermost) gate name. `TeamError(SignalGatingError)` — team protocol misuse. `BudgetExceeded(SignalGatingError)` — budget and offending step key. Sited in `errors.py` beside `GateRejected`/`CircuitOpenError`, the existing precedent for feature-scoped errors.
- Everything else rides existing paths: failing member handlers dead-letter and surface as task timeouts; failing script steps surface as `TimeoutError` into the script; ledger corruption raises `SignalSerializationError`.

## Testing

`tests/test_taskboard.py`, `tests/test_team.py`, `tests/test_script.py` — deterministic, in-memory, CI-safe, no sleep-based synchronization (await board state / request completion instead):

- Board: open/claim/complete/release lifecycle; targeted claim; dependency blocking and unblock-on-complete; N concurrent `claim()`s never double-assign; gates reject with `TaskRejected` and never run under the lock (instrumented); JSONL round-trip reconstructs identical derived state; an edited, reordered, or interiorly-deleted ledger line breaks the chain on load; `release_in_progress` recovery appends synthetic releases; observer exceptions are swallowed and counted.
- Team: assign executes on the named member and completion lands on the board with the reply's result; a raising handler yields release-on-timeout and a DLQ entry; two members drain a five-task dependent backlog with no double-work and stewards holding ≤1 task throughout; a release wakes an idle peer (the lost-wake-up regression test); final state after drain is one edge-triggered `MemberIdle` per member at the lead; shutdown under a loaded board stops claiming immediately and stops the agent; `dissolve()` is idempotent-hostile (`TeamError` on reuse) and removes the board observer; a stopped lead does not poison stewards.
- Script: identical rerun with a populated store performs zero mesh requests (asserted via mesh event recording); changed input re-executes; duplicate payloads in one phase execute separately (occurrence keys); semaphore holds in-flight ≤ `max_concurrency` (instrumented counter); step N+1 past budget raises `BudgetExceeded`; `spawn` leaves mesh topology exactly as found (including under concurrent spawns from one factory); dead-lettered target surfaces as `TimeoutError` with the checkpoint absent.
- Mesh: `remove()` purges pool membership and load-balance/content-route closures; removal mid-`request` times out the requester cleanly.

## Files

| File | Change |
| --- | --- |
| `src/signal_gating/taskboard.py` | New: `FrozenPayload` mixin, task event signals (pinned wire names), `Task` view, `TaskBoard`, chained ledger. |
| `src/signal_gating/team.py` | New: protocol signals, `Team`, stewards. |
| `src/signal_gating/script.py` | New: `Script`, `ScriptContext`, `CheckpointStore`. |
| `src/signal_gating/trajectory.py` | Promote `_domain_payload` to public `domain_payload()`. No other change. |
| `src/signal_gating/mesh.py` | Harden `remove()` (pools, LB/route closures). No other change. |
| `src/signal_gating/errors.py` | Add `TaskRejected`, `TeamError`, `BudgetExceeded`. |
| `src/signal_gating/__init__.py` | Export the above. |
| `tests/test_taskboard.py`, `tests/test_team.py`, `tests/test_script.py` | New (+ `remove()` hardening cases in `tests/test_mesh.py`). |
| `examples/agent_team.py`, `examples/scripted_workflow.py` | New: a 3-member review team; a checkpointed fan-out sweep. |
| `README.md` | New "Teams" and "Scripted workflows" sections. |

## Success criteria

1. `from signal_gating import TaskBoard, Team, Script, CheckpointStore` works; core still imports only stdlib + pydantic.
2. A 2-member team drains a dependent backlog via steward claims with zero double-assignments, survives a mid-run handler crash (release + DLQ, no stall), then shuts down and dissolves cleanly — asserted without sleeps-as-synchronization.
3. A script interrupted halfway and rerun completes from checkpoints, re-executing only unfinished steps (asserted by counting mesh requests on the second run).
4. An edited, reordered, or interiorly-deleted board ledger entry fails chain verification on load; a tampered checkpoint record fails its digest.
5. `pytest`, `ruff check .`, and `mypy src/` (strict) pass.
6. Both examples run end-to-end with deterministic stub agents (no LLM server).
