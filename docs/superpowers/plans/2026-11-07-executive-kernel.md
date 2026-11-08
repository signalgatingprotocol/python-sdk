# Executive Kernel Implementation Plan

> For agentic workers: implement this plan task by task, with a review gate per task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a usable local executive kernel that preserves intent, selectively updates state, mediates real work, handles corrections and restarts, and verifies completion, with foundations for adaptive control and federation.

**Architecture:** One authoritative SQLite writer commits journal events, materialized state, receipts, reservations, and outbox entries atomically. Bounded workers propose changes against versioned context; kernel transitions enforce authority, freshness, and budgets. The first release includes a real model adapter, local tools, foreground operator controls, and recovery, not just interfaces or a scripted demo.

**Tech Stack:** Python 3.10-3.14, asyncio, stdlib SQLite/argparse/JSON/filesystem/subprocess, existing Pydantic v2 and optional `openai` dependency, pytest/pytest-asyncio, Ruff, mypy, build, twine. No new mandatory runtime dependency.

**Spec:** [Approved executive-kernel design](../specs/2026-11-07-executive-kernel-design.md). Read both documents before execution. Design approved 2026-11-07; this plan awaits execution-method selection.

## Global Constraints

- "Preserve `signal_gating.core` and existing `Signal`, `Gate`, `Agent`, `Mesh`, and receipt behavior."
- "The kernel is the sole writer of authoritative state."
- "External calls never occur inside this transaction."
- "No model-generated floating-point score grants authority."
- "Workers may add dependencies but cannot remove mandatory ones."
- "Never silently truncate a hard constraint."
- "No configured approver means review remains pending."
- "Do not claim exactly-once execution or blindly repeat a non-idempotent effect."
- "In Preston's environment, outbound email is human-only and must never be queued or transmitted by this kernel."
- "Live self-modification of grants, hard constraints, verification criteria, or budget ceilings is prohibited."
- Python minimum remains `>=3.10`; existing CI covers 3.10-3.14. Keep exports in `signal_gating.executive`, not `core` or package-root exports.
- Preserve unrelated changes, including untracked decision-routing files. No attribution in commits or PRs. No credential values in logs, fixtures, or git.

## Review Focus

1. Duplicate IDs with different payloads must report conflict, not acknowledge different commands as identical. Test in Task 5.
2. Pausing a goal with an action in flight must preserve reconciliation without silently resuming work. Test in Tasks 11 and 14.
3. Oversized required context, including Unicode, must fail before inference rather than drop a constraint. Test in Tasks 9 and 10.
4. Symlinks or concurrently changed files must not permit root escape or overwrite of unexamined content. Test in Task 12; explicitly bound protection against hostile host writers.
5. Invalidated completion evidence must reopen dependent findings after the original worker disappears. Test in Tasks 7 and 13.

---

## Release scope and execution baseline

This is one integrated product plan. Storage, scheduling, tools, and verification cannot independently fulfill the intended experience, so they are reviewable tasks within one plan. Tasks 1–14 establish the kernel; Tasks 15–19 make it operable, tested, evaluable, and installable. Do not label it usable after the storage or deterministic-demo milestones alone.

A developer must be able to open a store, supply a goal and completion criteria, run a configured model against a bounded local workspace, revise or pause the goal during work, inspect decisions, resume after process loss, and obtain independently checked artifacts. Python and a foreground CLI are sufficient; no web service or MCP server is required.

Adaptive policies, network transports, and federation are later horizons. Implement their foundations now: versioned schemas, producer identity, causal IDs, scoped authority, commitments distinct from goals, receipts, policy versions, deterministic reducers, and exportable conformance cases. Those fields do not confer remote trust.

At execution start, recheck applicable instructions, status, and open PRs. The latest PR query during planning could not connect; the prior successful listing showed policy fixes, CodeQL, and dependency work. The checkout changed during planning from the design branch to `docs/refresh-install-pin`; the approved design was recovered from commit `ab63a3c`. Never switch branches underneath other work or discard unrelated files. Use a suitable checkout, or an isolated worktree at execution time if needed.

Create a venv; run `python -m pip install -e ".[dev]" build twine`, `make check`, and `make package`. Record pre-existing failures. An earlier local environment lacked `yaml`, and an untracked test imported an unexported `Decision`. These are not kernel regressions. Each task ends in an explicit-path commit. Run full checks at integration gates. Paid model calls remain opt-in; plan approval does not authorize spend.

## File ownership

New runtime files live under `src/signal_gating/executive/`:

| Files | Responsibility |
| --- | --- |
| `__init__.py`, `__main__.py`, `errors.py` | Experimental API, entrypoint, errors |
| `records.py`, `messages.py`, `config.py` | Persistent records, commands/events/proposals, configuration |
| `store.py`, `migrations.py`, `locking.py`, `reducers.py` | Transactions, schema evolution, ownership, transitions |
| `artifacts.py` | Scoped evidence, integrity, retention |
| `intake.py`, `authority.py`, `budgets.py` | Authenticated admission, constraints/grants, reservations |
| `memory.py`, `proposals.py` | Evidence state, invalidation, freshness |
| `scheduler.py`, `context.py` | Attention, wake conditions, context assembly |
| `workers.py`, `adapters/openai_worker.py` | Worker protocol and model adapter |
| `actions.py`, `adapters/local_tools.py` | Dispatch, reconciliation, local capabilities |
| `verification.py` | Criterion checks and completion |
| `kernel.py`, `recovery.py`, `inspection.py` | Coordinator, recovery, operator views |
| `cli.py`, `adapters/legacy.py` | Foreground interface and legacy bridge |
| `demo.py`, `evaluation.py` | Acceptance scenario and comparison harness |

Tests mirror components in `tests/executive/`, with `factories.py` and `conftest.py`. Add `tests/test_installed_executive.py`, `docs/executive-kernel.md`, and `examples/executive/`. Existing-file changes are limited to `pyproject.toml` entrypoints, `Makefile` smoke coverage, and surgical README/security documentation at Task 19.

## Shared contracts and decisions

### Public API

Implement this API in Task 14 using earlier components:

```python
class Kernel:
    @classmethod
    def open(cls, root: Path, *, config: KernelConfig,
             workers: Mapping[str, Worker], capabilities: CapabilityRegistry,
             verifiers: Mapping[str, Verifier], clock: Clock) -> Kernel: ...
    async def __aenter__(self) -> Kernel: ...
    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None: ...
    async def submit(self, command: Command, *, principal: Principal) -> DecisionReceipt: ...
    async def ingest(self, signal: ExecutiveSignal, *, principal: Principal) -> DecisionReceipt: ...
    async def run_until_idle(self) -> RunSummary: ...
    async def serve(self, commands: AsyncIterator[AuthenticatedCommand]) -> RunSummary: ...
    async def stop(self) -> None: ...
    def inspect(self, goal_id: str | None = None) -> KernelView: ...
```

`open` creates/opens storage; new-store ownership requires explicit local bootstrap in configuration. `run_until_idle` returns when nothing is runnable now and no owned episode/dispatch is active, including when tasks await review. `serve` waits for commands and timer wakes until stop or EOF. Neither equates idle with completion. Shutdown fences episodes and persists uncertain effects/usage before releasing ownership.

### Defaults and boundaries

- Use immutable Pydantic records with `extra="forbid"`, validated nested containers, strict integers, finite numbers, UTC timestamps, explicit enums. Internal IDs are UUID hex; external IDs are namespaced by principal and producer. Record revisions and schema version start at 1.
- Keep episode fencing counters separate from descriptive task revisions. Accept an episode's proposal bundle atomically against its pre-bundle read set; an early member must not invalidate later members. External results are separate observations.
- Owner controls are authenticated at the adapter. Workers cannot submit owner commands. Artifact access defaults to owner and goal; sharing is explicit.
- Defaults: 2 active episodes; 120-second episode timeout; 2048 maximum output tokens; 8192 context-token limit; 65,536-byte signal limit; 8,388,608-byte artifact limit; 2 identical failures before blocking; 2 verification repair rounds; 10,000 queued ordinary items. Validate and record overrides.
- Scheduler `deterministic-v1`: process batches of up to 64 controls before ordinary dispatch. When clear, repeat 4 reconciliation / 2 urgent / 1 ordinary selections, skipping empty classes. Ordinary sort: owner priority plus one aging point per 60 seconds capped at 100, deadline, enqueue time, ID. Urgent means owner-designated or deadline within 60 seconds. Report control overload after 10 seconds of continuous blocking; never silently discard a stop command.
- Goal budgets are explicit. Currency uses integer micro-units and a currency code. Strict money mode requires a trusted maximum charge quote. Prices/capacity come from versioned operator configuration, not hardcoded current service assumptions.
- Inject `TokenCounter`. Include a conservative UTF-8 byte counter plus configurable framing reserve, labeled as estimated. A provider-specific counter/capacity configuration is required where that bound is not defensible. Reserve output/framing before packing inputs; estimated billing is separate from context capacity.
- Test writer support on Linux and macOS initially. Reject unsupported writer platforms before mutation; keep ordinary package imports and legacy use portable. Do not use SQLite stores on network filesystems.
- The operator runs one foreground process receiving JSONL controls on stdin and emitting JSON responses/events on stdout. Separate `inspect` uses read-only snapshots. No socket, daemon, or command-spool authentication scheme is required.

### Shared types and test fixtures

Task 1 defines the spec's persistent records as `Record`, plus `ExecutiveSignal` distinct from legacy `Signal`. `Command` is a tagged union: `CreateGoal`, `ReviseGoal`, `PauseGoal`, `ResumeGoal`, `CancelGoal`, `ResolveConflict`, `DecideAction`, `ResolveAction`, `ResolveUsage`, `AttachEvidence`, `Reverify`, `ShareArtifact`, `SetGrant`. `Proposal` is a tagged union: `BeliefChange`, `TaskDecomposition`, `ActionIntent`, `EvidenceRequest`, `CompletionClaim`.

Other DTOs used below are `JournalEvent`, `CommitBatch`, `StateView`, `Ref`, `WakeCondition`, `PolicyDecision`, `ContextPacket`, `ContextManifest`, `ArtifactDraft`, `EpisodeResult`, `Usage`, `Limits`, `Reservation`, `CapabilitySpec`, `ActionObservation`, `VerificationResult`, `Criterion`, `CriterionGap`, `AuthenticatedCommand`, `RunSummary`, `KernelView`, `EvaluationCase`, `EvaluationResult`, `EvaluationReport`. Their required fields follow the design and the interfaces/assertions below. Arbitrary JSON is allowed only at declared payload/capability boundaries, not as untyped authoritative state.

An `ArtifactDraft` carries an episode-local key, media type, and bounded UTF-8 text. `EpisodeResult.artifact_drafts` lets a model propose real file contents without filesystem access. Proposal references may name `draft:<key>`; the kernel stages these bytes outside the state transaction, validates size/scope, and resolves them to new goal-scoped artifact references inside the accepted bundle. They remain model-produced hypotheses/drafts until independently checked. Rejected bundles leave only collectible orphan bytes, never executable file actions.

`ResolveAction` requires action ID, expected action revision, observed outcome (`succeeded` or `failed`), evidence references, and an owner explanation. It records a human attestation, not adapter proof, and never means retry. `ResolveUsage` requires reservation ID, expected revision, known usage and evidence, or explicit acceptance of charging the full reserved maximum; it cannot silently erase uncertainty for free. Both are unavailable to workers.

Configuration declares owner identity, worker IDs/kinds (`fixture` or `openai`), model/environment-variable references, context/output/price limits, workspace root/path rules/named argv checks, criterion verifier IDs, and scheduler overrides. Goal input declares outcome, constraints, criteria, required commitments, selected worker, and explicit budgets. Bootstrap grants are operator-configured and default to none. Goal creation supplies one initial ready task if no decomposition is specified. JSON config never imports arbitrary Python code; custom registries enter through the Python API.

### Episode-to-work progression

Accepting a bundle containing action intents makes the frame wait on those actions. Their observations update progress/evidence references and wake the frame once all required effects are settled. A pure accepted belief/task update schedules the next episode when unfinished runnable work remains. `needs_evidence` requires an evidence-request proposal and waits for its resolution; `blocked` requires a typed blocker/wake condition; `exhausted` suspends on budget/limit state. A completion claim enters verification. An empty `proposed` result counts as a no-progress failure. Conflicting lifecycle requests in one bundle are rejected atomically. These transitions live in reducers and the coordinator, not worker-specific loops.

`Clock` supplies `now() -> datetime` and `monotonic() -> float`; `FakeClock.advance(seconds: float)` and controlled futures replace timing sleeps. `tests/executive/factories.py` supplies `owner()`, `goal()`, `frame()`, `grant()`, `signal()`, `proposal()`, `action()`, `packet()`, `criterion()`, each taking keyword overrides and returning valid records with fixed IDs/times. Tests below use component fixtures created in their owning task, not alternate implementations of the component.

## Task 1: Versioned contracts

**Files:** Create `src/signal_gating/executive/{records,messages,config,errors,__init__}.py`; create `tests/executive/{__init__,factories,conftest,test_contracts}.py`.

**Interfaces:** Produces shared DTOs and `KernelConfig`. Goal statuses: `active`, `paused`, `blocked`, `completed`, `cancelled`, `failed`. Commitments mirror task status and carry responsible principal and required/optional flag. Receipts expose `disposition`, `reason_code`, `event_ids`; run summaries expose completed-goal IDs, blocked-task IDs, pending-review IDs, unknown-action IDs, and usage.

- [ ] Write `test_contract_roundtrip_and_invalid_values` for every tagged type, unsupported versions, bool-as-int, NaN, naive timestamps, negative limits, and mutation.

  ```python
  assert Goal.model_validate_json(goal().model_dump_json()) == goal()
  with pytest.raises(ValidationError):
      KernelConfig(max_active_episodes=True)
  assert proposal().schema_version == 1
  ```

- [ ] Run `python -m pytest tests/executive/test_contracts.py -q`; expect missing imports/contracts.
- [ ] Implement contracts, structured errors, defaults, and fixtures. Keep optional model dependencies out of imports.
- [ ] Run that test and `python -m mypy src/signal_gating/executive`; require success.
- [ ] Commit explicit task files: `feat: define executive kernel contracts`.

## Task 2: Atomic persistence and migrations

**Files:** Create `src/signal_gating/executive/{store,migrations,locking}.py`; `tests/executive/test_store.py`.

**Interfaces:** `Store.open(root: Path) -> Store`; `commit(batch: CommitBatch) -> tuple[JournalEvent, ...]`; `view() -> StateView`; `events(after: int = 0) -> tuple[JournalEvent, ...]`; `close() -> None`. Batches contain expected revisions, events, view writes, inbox acknowledgments, reservations, receipts, and outbox entries.

- [ ] Write rollback, stale-CAS, monotonic sequence, future-schema, migration, and subprocess ownership-lock tests. Inject failure between event and view writes.

  ```python
  before = store.view()
  with pytest.raises(InjectedFailure):
      store.commit(failing_batch)
  assert store.view() == before
  assert store.events() == ()
  ```

- [ ] Run `python -m pytest tests/executive/test_store.py -q`; expect missing store behavior.
- [ ] Implement WAL, foreign keys, `synchronous=FULL`, `BEGIN IMMEDIATE`, canonical JSON, uniqueness constraints, bounded busy timeout, OS-released advisory ownership lock, and read-only snapshots. No model/tool calls inside transactions. Migration preserves event interpretability; unsupported schema versions fail before writes.
- [ ] Run store tests with process kill/reopen; require no partial records or orphaned ownership locks.
- [ ] Commit: `feat: persist executive state atomically`.

## Task 3: Reducers and lifecycle

**Files:** Create `src/signal_gating/executive/reducers.py`; `tests/executive/test_reducers.py`.

**Interfaces:** `reduce_event(state: StateView, event: JournalEvent) -> StateView`; `rebuild(events: Iterable[JournalEvent]) -> StateView`; `validate_transition(record: Record, target: str) -> None`. Time and IDs come from events, not reducer side effects.

- [ ] Write transition, pause/resume/cancel, dependency-cycle, required/optional commitment, and replay tests.

  ```python
  assert rebuild(events) == expected_state
  assert rebuild(events).goals[goal_id].status == "paused"
  with pytest.raises(InvalidTransition):
      validate_transition(completed_frame, "running")
  ```

- [ ] Run `python -m pytest tests/executive/test_reducers.py -q`; expect missing reducer behavior.
- [ ] Implement the design's state machines. Resume only eligible work; preserve blockers. Cancel fences tasks and undispatched intents while retaining effects in flight. Reopening requires an explicit invalidation event. Reject missing references and cycles atomically.
- [ ] Run reducer/store tests; require reconstructed and materialized views to match.
- [ ] Commit: `feat: implement executive lifecycle reducers`.

## Task 4: Scoped artifacts and retention

**Files:** Create `src/signal_gating/executive/artifacts.py`; `tests/executive/test_artifacts.py`.

**Interfaces:** `ArtifactStore.put(data: bytes, *, principal: Principal, goal_id: str, media_type: str) -> ArtifactRef`; `read(ref: ArtifactRef, *, principal: Principal, goal_id: str) -> bytes`; `delete(ref: ArtifactRef, *, principal: Principal) -> JournalEvent`; `collect_orphans() -> int`.

- [ ] Write atomic-write, size-limit, corruption, cross-goal access, shared-byte/separate-reference, and deletion-event tests.

  ```python
  assert artifacts.read(ref, principal=owner(), goal_id=ref.goal_id) == b"evidence"
  with pytest.raises(AccessDenied):
      artifacts.read(ref, principal=other_owner, goal_id=other_goal)
  assert deletion.kind == "artifact_invalidated"
  ```

- [ ] Run `python -m pytest tests/executive/test_artifacts.py -q`; expect missing artifact behavior.
- [ ] Implement content-addressed bytes with separate scoped reference records, restrictive permissions, write/fsync/rename, integrity checks, and explicit cleanup. Revoke/invalidate a reference transactionally before physical deletion; retain bytes with live references. Reads use reference validity, not possession of a hash. Missing/corrupt artifacts cannot satisfy criteria. Required active evidence is not age-deleted automatically.
- [ ] Run artifact/store tests; require orphan cleanup to preserve referenced bytes and denied reads to expose no contents.
- [ ] Commit: `feat: add scoped executive evidence storage`.

## Task 5: Authenticated intake and authority

**Files:** Create `src/signal_gating/executive/{intake,authority}.py`; `tests/executive/{test_intake,test_authority}.py`.

**Interfaces:** `Intake.accept(item: Command | ExecutiveSignal, *, principal: Principal, state: StateView) -> CommitBatch`; `Authority.check_command(principal: Principal, command: Command, state: StateView) -> PolicyDecision`; `check_action(intent: ActionIntent, state: StateView, now: datetime) -> PolicyDecision`; `RuleRegistry.evaluate(rule_id: str, inputs: Mapping[str, Record]) -> PolicyDecision`.

- [ ] Write owner/worker role, evidence-as-instruction, late observation, correction acknowledgment, queue bound, duplicate-ID conflict, and owner-only action/usage resolution tests.

  ```python
  assert duplicate_receipt == original_receipt
  assert conflicting_payload_receipt.reason_code == "duplicate_id_conflict"
  assert worker_control_receipt.disposition == "deny"
  assert unknown_rule_decision.disposition == "require_review"
  ```

- [ ] Run `python -m pytest tests/executive/test_intake.py tests/executive/test_authority.py -q`; expect missing authority/intake behavior.
- [ ] Bind principals from trusted adapters; canonicalize payload digests and deduplicate by principal/producer/ID. Reserve bounded control capacity separately; apply backpressure before acknowledgment when full, never drop an acknowledged stop request. Commit corrections before returning acceptance. Hard rules are registered trusted functions, not imported model text. Conflicting authorized constraints block affected work until resolution names the conflicting revisions.
- [ ] Run tests; require unknown/expired/revoked permissions never authorize, and oversized payloads yield bounded rejection metadata without logging their contents.
- [ ] Commit: `feat: enforce executive intake authority`.

## Task 6: Hierarchical budgets

**Files:** Create `src/signal_gating/executive/budgets.py`; `tests/executive/test_budgets.py`.

**Interfaces:** `BudgetPolicy.reserve(state: StateView, budget_id: str, limits: Limits, reservation_id: str) -> CommitBatch`; `settle(state: StateView, reservation_id: str, usage: Usage) -> CommitBatch`; `mark_unknown(state: StateView, reservation_id: str) -> CommitBatch`.

- [ ] Write concurrent sibling reservation, duplicate settlement, exhaustion, uncertain usage, and currency mismatch tests.

  ```python
  assert accepted_reservations == 1  # two requests for 6 from a limit of 10
  assert state.budgets[root_id].reserved.calls == 6
  assert unknown_reservation.released is False
  ```

- [ ] Run `python -m pytest tests/executive/test_budgets.py -q`; expect missing accounting behavior.
- [ ] Implement integer accounting and ancestor CAS writes in the same transaction as episode/action authorization. Reconcile known usage and retain uncertain remainder. Strict-money mode rejects unknown maximum quotes. An adapter exceeding a promised bound records actual overage and blocks more spending; it must not silently claim enforcement succeeded.
- [ ] Run budget/store tests concurrently; require no oversubscription of declared reliable bounds.
- [ ] Commit: `feat: bound executive resource spending`.

## Task 7: Memory, freshness, and invalidation

**Files:** Create `src/signal_gating/executive/{memory,proposals}.py`; `tests/executive/{test_memory,test_proposals}.py`.

**Interfaces:** `MemoryPolicy.evaluate(change: BeliefChange, state: StateView) -> PolicyDecision`; `ProposalValidator.validate(result: EpisodeResult, episode: Episode, state: StateView) -> CommitBatch`; `invalidate_dependents(state: StateView, changed: tuple[Ref, ...]) -> tuple[JournalEvent, ...]`.

- [ ] Write disputed-belief, missing-evidence, stale/fenced proposal, missing mandatory-read-set, atomic bundle, artifact-draft reference/size, decomposition-cycle, and completed-finding invalidation tests.

  ```python
  assert stale_receipt.reason_code == "stale_dependency"
  assert state.beliefs[claim_id].status == "disputed"
  assert state.goals[completed_goal_id].status == "active"
  assert original_completion_event in history
  ```

- [ ] Run `python -m pytest tests/executive/test_memory.py tests/executive/test_proposals.py -q`; expect missing validation behavior.
- [ ] Implement evidence policies: unknown evidence remains hypothesis; conflicting support becomes disputed; invalidated evidence retracts dependent support. Owner-registered evidence validators establish support. Validate entire bundles and mandatory dependencies before any state write; resolve staged artifact drafts without granting them supported status. Decomposition inherits constraints and parent budgets; grants may narrow only. Invalidation fences dependent episodes and reopens affected criteria while preserving unrelated artifacts.
- [ ] Run memory/proposal/reducer tests; require one invalid bundle member to prevent all its state/action writes.
- [ ] Commit: `feat: validate executive state proposals`.

## Task 8: Attention and scheduling

**Files:** Create `src/signal_gating/executive/scheduler.py`; `tests/executive/test_scheduler.py`.

**Interfaces:** `AttentionPolicy.decide(signal: ExecutiveSignal, state: StateView) -> PolicyDecision`; `Scheduler.select(state: StateView, now: datetime) -> tuple[str, ...]`; `wake(state: StateView, event: JournalEvent | None, now: datetime) -> tuple[JournalEvent, ...]`. Selection returns task IDs up to available episode slots.

- [ ] Write fake-clock tests for 4/2/1 order, deadline/aging ranking, coalescing, all wake variants, overload, and repeated failures.

  ```python
  assert classes[:7] == ["reconcile"] * 4 + ["urgent"] * 2 + ["ordinary"]
  clock.advance(60)
  assert awakened_task_id in selected_after_wake
  assert repeated_failure_task.status == "blocked"
  ```

- [ ] Run `python -m pytest tests/executive/test_scheduler.py -q`; expect missing scheduler behavior.
- [ ] Implement pinned defaults. Every defer needs a typed wake condition; archival does not erase commitments. Relevance judgments become bounded assessment tasks, never recursive assess-the-assessor calls. Record policy version and disposition. Pause/resume alone does not reset no-progress counters. Persist scheduling cursor/aging timestamps for stable restart behavior.
- [ ] Run scheduler/reducer tests; require no selection of paused, blocked, stale, or already-running tasks and no deferred-work busy loop.
- [ ] Commit: `feat: schedule persistent executive attention`.

## Task 9: Context assembly

**Files:** Create `src/signal_gating/executive/context.py`; `tests/executive/test_context.py`.

**Interfaces:** `TokenCounter.count(text: str) -> int`; `ContextAssembler.build(frame: TaskFrame, state: StateView, *, principal: Principal) -> ContextPacket`; `render_packet(packet: ContextPacket) -> tuple[dict[str, str], ...]`. Packets include mandatory reads, manifest, rendered-input artifact reference, bindings, and limits.

- [ ] Write tests for constraint preservation, disputed beliefs, scoped evidence, deterministic ranking, evidence requests, and oversized Unicode required context.

  ```python
  assert constraint_id in packet.manifest.required_ids
  assert secret_id not in packet.manifest.visible_ids
  with pytest.raises(ContextBudgetInsufficient):
      assembler.build(oversized_frame, state, principal=owner())
  ```

- [ ] Run `python -m pytest tests/executive/test_context.py -q`; expect missing context behavior.
- [ ] Pack required authority/task/criteria first; rank optional evidence by task link, support, freshness, then ID. Reserve output/framing before packing. All supplied beliefs are mandatory revision dependencies. Evidence requests retrieve authorized references into a new bounded episode; they cannot request arbitrary paths or turn content into instructions. Context artifacts are kernel-private unless explicitly shared; excluded-artifact manifests expose no inaccessible content.
- [ ] Run context/artifact/proposal tests; require overflow before worker invocation and durable actual-input manifests within retention policy.
- [ ] Commit: `feat: assemble executive working context`.

## Task 10: Bounded workers and real model adapter

**Files:** Create `src/signal_gating/executive/workers.py`, `adapters/__init__.py`, `adapters/openai_worker.py`; `tests/executive/{test_workers,test_openai_worker}.py`.

**Interfaces:** `Worker.run_episode(packet: ContextPacket) -> Awaitable[EpisodeResult]`; `EpisodeRunner.run(episode: Episode, packet: ContextPacket, worker: Worker) -> Awaitable[EpisodeResult]`; `OpenAIWorker(client: LLMClient, *, model: str, counter: TokenCounter, capacity: int, quote: Limits | None)`. Consume the existing `signal_gating.llm.LLMClient` contract; do not change `LLMAgent` behavior.

- [ ] Write fake-client tests for proposal decoding, malformed JSON, unknown tags, context/output limits, timeout, returned tool calls, cancellation, and missing usage.

  ```python
  assert result.stop_reason == "proposed"
  assert fake_client.last_kwargs["max_tokens"] == 2048
  assert capabilities_called == []
  assert missing_usage_result.usage.uncertain is True
  ```

- [ ] Run `python -m pytest tests/executive/test_workers.py tests/executive/test_openai_worker.py -q`; expect missing adapter behavior.
- [ ] Request JSON matching the episode-result schema through the injected chat-completions client and validate it locally. The adapter returns action proposals and never executes provider-selected tools. Do not rely on vendor-only structured-output support. Make the output-limit parameter configurable for endpoints differing from `max_tokens`. Enforce deadlines with Python-3.10-compatible APIs. Fence late results and retain uncertain reservations when usage is missing. Default retries to zero; scheduling owns subsequent attempts.
- [ ] Run fake-client tests without credentials/network. Document a real provider constructor using explicit model, capacity, counter, quote, and environment credentials. Do not execute paid requests during default tests.
- [ ] Commit: `feat: add bounded model executive worker`.

## Task 11: Authorized dispatch and reconciliation

**Files:** Create `src/signal_gating/executive/actions.py`; extend `authority.py`; `tests/executive/test_actions.py`.

**Interfaces:** `CapabilityRegistry.register(spec: CapabilitySpec, adapter: ExecutionAdapter) -> None`; `ExecutionAdapter.execute(action: ActionRecord) -> Awaitable[ActionObservation]`; `reconcile(action: ActionRecord) -> Awaitable[ActionObservation]`; `cancel(action: ActionRecord) -> Awaitable[ActionObservation]`. `Dispatcher.dispatch_one(action_id: str) -> Awaitable[DecisionReceipt]`; `reconcile_one(action_id: str) -> Awaitable[DecisionReceipt]` consume store, authority, budgets, and registry.

- [ ] Write fake-effect tests for revocation, replaced bindings, approval digest mismatch, pause in flight, crash after effect/before receipt, and expired idempotency windows.

  ```python
  assert revoked_adapter.execute_count == 0
  assert recovered_action.status == "outcome_unknown"
  assert unsafe_adapter.execute_count == 1
  assert paused_goal.status == "paused"
  ```

- [ ] Run `python -m pytest tests/executive/test_actions.py -q`; expect missing dispatcher behavior.
- [ ] Authorize/reserve atomically; recheck authority, pause state, dependencies, binding, expiry, and budget immediately before committing `dispatching`. Approvals bind arguments, resources, and relevant revisions. Commit observations separately. Timeout/disconnection after dispatch means unknown unless no effect is proved. Reconcile with stable action ID; retry only inside declared durable idempotency scope/window. Preserve observations for paused/cancelled goals without enabling more actions. No approver leaves review pending; expiry never grants approval.
- [ ] Run action/authority/budget tests; require every external call to have a durable dispatch record and occur outside the transaction.
- [ ] Commit: `feat: mediate executive action lifecycles`.

## Task 12: Useful local capabilities

**Files:** Create `src/signal_gating/executive/adapters/local_tools.py`; `tests/executive/test_local_tools.py`.

**Interfaces:** `LocalWorkspace(root: Path, *, readable: tuple[str, ...], writable: tuple[str, ...], commands: Mapping[str, tuple[str, ...]])`; `.capabilities() -> tuple[tuple[CapabilitySpec, ExecutionAdapter], ...]`. Bind versioned `list_files`, `read_text`, `write_text`, and `run_check`. Writes require a relative path, expected previous SHA-256 or explicit missing marker, and replacement-content artifact. Checks accept configured command IDs plus expected digests of the criterion's declared input files. Their evidence records argv/configuration and before/after input digests; changed inputs invalidate the result.

- [ ] Write tests for useful operations, traversal, symlink components, Unicode filenames, stale hashes, detectable file replacement races, bounded check output, timeout, and write reconciliation.

  ```python
  assert workspace_file.read_text() == "corrected\n"
  assert stale_write.reason_code == "resource_changed"
  assert outside_file.read_bytes() == original_bytes
  assert reconciled_write.status == "succeeded"
  ```

- [ ] Run `python -m pytest tests/executive/test_local_tools.py -q`; expect missing local tools.
- [ ] Use root-relative descriptor operations with symlink rejection on tested platforms. Serialize adapter writes, recheck expected content, and replace temporary same-directory files atomically. Protection from unrelated hostile host writers requires external isolation; do not promise filesystem-wide CAS. Refuse detectable ambiguity. Reconcile using content plus durable action evidence; identical bytes alone do not prove this action executed. Execute configured argv without a shell, with fixed cwd, allowlisted environment, bounded output artifacts, deadline, and process-group cleanup. Configured commands are trusted programs, not a sandbox. Unknown command outcomes remain visible and require resolution; no email capability exists.
- [ ] Run local-tool/action tests in temporary workspaces; require no path escape, no unregistered command, and no falsely confirmed uncertain write.
- [ ] Commit: `feat: add local executive workspace tools`.

## Task 13: Verification and completion

**Files:** Create `src/signal_gating/executive/verification.py`; `tests/executive/test_verification.py`.

**Interfaces:** `Verifier.verify(criterion: Criterion, state: StateView) -> Awaitable[VerificationResult]`; `VerificationService.evaluate(claim: CompletionClaim, state: StateView) -> Awaitable[CommitBatch]`. Built-ins check artifact integrity/existence, registered local-check evidence, and explicit human approvals. Optional model judgments use the bounded worker contract and are labeled as such.

- [ ] Write missing-criterion, unrelated-pass, stale-verifier, human-criterion, optional-commitment, finite-repair, and invalidated-evidence tests.

  ```python
  assert unrelated_pass.disposition == "insufficient_evidence"
  assert complete_state.goals[goal_id].status == "completed"
  assert invalidated_state.goals[goal_id].status == "active"
  assert third_repair_attempt.status == "blocked"
  ```

- [ ] Run `python -m pytest tests/executive/test_verification.py -q`; expect missing verification behavior.
- [ ] Bind criteria to artifact versions, owner-selected verifier/configuration, and constraint revisions. External verifier calls use dispatcher/reservations; results undergo freshness checks. Failed verification creates named gaps and at most two automatic repair rounds without material evidence changes, then blocks. Human criterion approvals name criterion, artifact, and revision; no generic approval flag. Required unsettled effects prevent dependent completion.
- [ ] Run verification/proposal/action tests; require complete goals only when all required criteria and commitments are satisfied.
- [ ] Commit: `feat: verify executive task outcomes`.

## Task 14: Integrated kernel and restart recovery

**Files:** Create `src/signal_gating/executive/{kernel,recovery,inspection}.py`; extend `__init__.py`; `tests/executive/{test_kernel,test_recovery}.py`.

**Interfaces:** Implements the public `Kernel` API. `recover(store: Store, *, registry: CapabilityRegistry, now: datetime) -> CommitBatch`; `inspect_state(state: StateView, goal_id: str | None) -> KernelView`. Consume Tasks 1–13 without duplicating their policies.

- [ ] Write integration tests for live correction during a blocked worker call, suspend/resume, concurrency, exhaustion, timer wakes, stop/EOF, lost leases, uncertain effects, and paused-goal restart.

  ```python
  receipt = await kernel.submit(PauseGoal(goal_id=gid), principal=owner())
  assert receipt.disposition == "accepted"
  assert kernel.inspect(gid).goal.status == "paused"
  assert late_receipt.reason_code == "episode_fenced"
  assert len((await kernel.run_until_idle()).pending_reviews) == 1
  ```

- [ ] Run `python -m pytest tests/executive/test_kernel.py tests/executive/test_recovery.py -q`; expect missing integration behavior.
- [ ] Implement short serialized commits plus bounded async workers outside the loop, condition/event wakeups, durable fencing, and the progression rules above. Stage artifact drafts before bundle validation; never let workers write files directly. Acknowledge human control only after state commit, before ordinary dispatch. Record external observations from cancelled work but reject fenced proposals. Startup validates schema and reconciles unfinished records before scheduling affected tasks. Unknown effects block dependent work; independent observations may proceed. Shutdown fences late workers and preserves uncertain usage/effects without waiting forever. No automatic resumption of paused/cancelled goals.
- [ ] Run all `tests/executive` and `tests/test_public_api.py`; require no leaked asyncio work, idle polling spin, or replay/live-state differences.
- [ ] Commit: `feat: integrate persistent executive kernel`.

## Task 15: Foreground operator interface

**Files:** Create `src/signal_gating/executive/{cli,__main__}.py`; extend `inspection.py`; `tests/executive/test_cli.py`; `examples/executive/{local-config,goal}.json`.

**Interfaces:** `main(argv: Sequence[str] | None = None) -> int`; `load_config(path: Path) -> KernelConfig`. CLI commands: `init --store PATH`, `run --store PATH --config FILE (--goal FILE | --resume GOAL_ID) [--live] [--batch]`, `inspect --store PATH [--goal ID] [--json]`, `export --store PATH --goal ID --output PATH`, `demo --store PATH`. `run` serves tagged JSONL controls; every response includes command ID, receipt, and status. `--goal` creates and reports a stable goal ID; `--resume` loads that goal without creating another commitment, reconciles outstanding work, and resumes only through the explicit owner-authorized resume transition. Cancelled/failed goals require a new goal rather than silent reactivation.

- [ ] Write subprocess tests for another cwd, live correction responsiveness, malformed command recovery, secret redaction, EOF persistence, resume without duplicate commitments, concurrent inspect, and second-writer rejection.

  ```python
  assert paused_response["goal_status"] == "paused"
  assert len(inspected["pending_reviews"]) == 1
  assert "API_KEY_VALUE" not in output
  assert second_writer.returncode == 2
  ```

- [ ] Run `python -m pytest tests/executive/test_cli.py -q`; expect missing CLI behavior.
- [ ] Bind controls to the configured local owner, never a claimed JSON principal. Implement inspection of status/reasons/receipts and all `Command` operations, including evidence-backed uncertain-action and usage resolution, through shared kernel methods. A dedicated stdin thread/queue may keep asyncio responsive. Reject real model execution without `--live` and explicit budgets; load credentials only through environment references. Exit 0 on clean shutdown/completion, 2 on startup/config error, 3 on batch idle requiring intervention, 4 on runtime failure/corruption. Interactive review remains live until EOF/stop. Add `--batch` to run without interactive stdin and return the idle code.
- [ ] Run CLI/kernel tests and exercise a fixture worker with a streamed correction. Require useful operator output and a restart path without relying on checkout-relative files.
- [ ] Commit: `feat: expose executive operator controls`.

## Task 16: Full acceptance scenario and crash testing

**Files:** Create `src/signal_gating/executive/demo.py`; `tests/executive/{test_acceptance,test_faults}.py`; `examples/executive/release-scenario.json` and `release-workspace/` fixture files.

**Interfaces:** `run_demo(root: Path) -> Awaitable[EvaluationResult]`. Use the real kernel/local tools with deterministic workers. Result metrics come from actual receipts, including stale rejection, resumed commitment, compatibility evidence, reconciliation, and forbidden dispatches.

- [ ] Write the design's nine-step release scenario and subprocess crashes before authorization, after authorization, after dispatch, after effect, and before observation commit.

  ```python
  assert result.metrics["stale_proposals_rejected"] == 1
  assert result.metrics["forbidden_dispatches"] == 0
  assert result.metrics["resumed_commitments"] == 1
  assert result.completed is True
  assert rebuilt_state == live_state
  ```

- [ ] Run `python -m pytest tests/executive/test_acceptance.py tests/executive/test_faults.py -q`; expect missing scenario orchestration.
- [ ] Build a fixture project with a checkable Python-3.10 requirement and an incompatible proposal. Label static fixture checks accurately; actual interpreter compatibility is verified by CI's matrix. Kill subprocesses rather than only raising exceptions. Include expired idempotency windows and missing artifacts after restart. Keep the installed demo deterministic and independent of source fixtures by packaging its minimal data within the runtime module or package assets.
- [ ] Run the scenario/fault tests, `python -m signal_gating.executive demo --store <temporary-path>`, and `make check`; require measured fixture output with no credentials.
- [ ] Commit: `test: prove executive continuity and recovery`.

## Task 17: Comparison harness and usefulness criteria

**Files:** Create `src/signal_gating/executive/evaluation.py`; `tests/executive/test_evaluation.py`; `examples/executive/evaluation/{cases.jsonl,protocol.json,README.md}`.

**Interfaces:** `evaluate_cases(cases: Sequence[EvaluationCase], *, arm: str, trials: int, worker_factory: Callable[[], Worker]) -> Awaitable[EvaluationReport]`; `compare_reports(reports: Sequence[EvaluationReport]) -> dict[str, object]`. Arms: `notes`, `rules`, `executive`, `executive_no_memory_gating`, `executive_no_priority_scheduling`. Reuse existing evaluation utilities where compatible; do not add a general eval framework.

- [ ] Write equal-configuration, isolated-store, cost-accounting, held-out split, missing-usage, failed-attempt retention, and empty-suite tests.

  ```python
  assert report.attempts == len(cases) * 3
  assert report.total_calls == report.worker_calls + report.control_calls + report.verifier_calls
  assert missing_usage_report.cost_status == "unknown"
  assert empty_report.accepted is False
  ```

- [ ] Run `python -m pytest tests/executive/test_evaluation.py -q`; expect missing comparison behavior.
- [ ] Implement a competent notes/compaction baseline and a rules baseline adding explicit task state, priority, and deduplication. All arms use identical model/tool/verification settings, caps, stimuli, and trial ordering; document behavioral differences. Replay of recorded decisions is separate from fresh model evaluation. No control arm receives weaker authorization or artificially unrestricted context by default.
- [ ] Create 6 tuning and 24 held-out cases across correction, interruption/resumption, contradictory evidence, repeated failure, restart/unknown effect, and low-overhead tasks. Four held-out cases per category, 3 trials each: 72 attempts per main arm. Freeze the case manifest before live evaluation. Initial usefulness criteria, selected here for planning: zero observed unauthorized effects or hard-constraint violations; at least 90% correct eligible resumption; completion no worse than the stronger baseline; plus either 10 percentage points better verified completion or 20% lower total measured cost at equal-or-better completion. Report paired differences and uncertainty; point thresholds do not establish general superiority. Unknown cost prevents cost claims. Preserve every failure.
- [ ] Run comparison tests and all arms with deterministic workers, clearly labeled `synthetic_fixture`. Provide `python -m signal_gating.executive.evaluation --cases FILE --protocol FILE --output PATH [--live]`; print planned request ceilings before opt-in live work. No paid run is part of default tests. Failed usefulness criteria block benefit claims/expansion, not honest availability of working experimental code.
- [ ] Commit: `feat: evaluate executive control outcomes`.

## Task 18: Legacy adapter and portable contract exports

**Files:** Create `src/signal_gating/executive/adapters/legacy.py`; `tests/executive/{test_legacy,test_schema_export}.py`; schema vectors in `tests/executive/fixtures/contracts/`.

**Interfaces:** `LegacySignalAdapter.convert(signal: signal_gating.Signal, *, principal: Principal, goal_id: str) -> ExecutiveSignal`; `export_contract_schemas() -> dict[str, object]`; `export_goal(store: Store, goal_id: str, *, principal: Principal) -> dict[str, object]`. Put export functions in `inspection.py`. `LegacyWorker` wraps an explicitly supplied callable with the `Worker.run_episode` signature and grants no implicit privileges.

- [ ] Write lineage, authority non-promotion, schema roundtrip, future-version rejection, access-scope, and redaction tests.

  ```python
  assert converted.causal_ids == (legacy.id,)
  assert converted.kind == "observation"
  assert converted.principal_id == authenticated_principal.id
  assert exported["schema_version"] == 1
  assert "credentials" not in json.dumps(exported)
  ```

- [ ] Run `python -m pytest tests/executive/test_legacy.py tests/executive/test_schema_export.py tests/test_public_api.py -q`; expect only new adapter/schema failures.
- [ ] Preserve explicit scope/provenance. An existing runtime acting outside the dispatcher is observation-only, visibly labeled. Export logical records, policy versions, receipts, and authorized artifact manifests without credentials, machine paths, or transferable grants. Ship conformance vectors for duplicates, stale proposals, interruption, and unknown effects. Describe schemas as experimental; do not claim federation or remote enforcement.
- [ ] Run adapter/schema/public API tests; require unchanged legacy exports and provider-independent schema inspection.
- [ ] Commit: `feat: expose portable executive contracts`.

## Task 19: Installed usability and final checks

**Files:** Modify `pyproject.toml` (`[project.scripts]`), `Makefile` (`smoke`), `README.md` (executive section/link), `SECURITY.md` (executive boundary); create `docs/executive-kernel.md`, `tests/test_installed_executive.py`. Reconcile concurrent README work before editing.

**Interfaces:** Register `signal-gating-executive = "signal_gating.executive.cli:main"`. Existing entrypoints/imports remain unchanged.

- [ ] Write installed-wheel tests from another cwd for dependency-light import/demo, restart inspection, config/goal loading, and live-mode diagnostics.

  ```python
  assert completed.returncode == 0
  assert report["forbidden_dispatches"] == 0
  assert report["completed"] is True
  assert legacy_core_exports == expected_legacy_core_exports
  ```

- [ ] Run `python -m pytest tests/test_installed_executive.py -q`; expect missing entrypoint/package behavior.
- [ ] Add entrypoint and credential-free wheel smoke using a temporary directory. Document install, owner bootstrap, a real goal, scoped local commands, explicit live model configuration/budgets, correction, exact approval, pause/restart/resume, inspection, uncertain-effect resolution, and export. Explain host-process mediation and trusted configured commands. Keep adaptive control and federation as subsequent horizons supported by present contracts. Do not publish or automatically bump version.
- [ ] In a clean venv run `make check`, `make package`, `make smoke`. Require Python 3.10–3.14 CI evidence, full scenario/fault tests, and wheel use independent of checkout. Report any unrelated baseline failure explicitly rather than deleting tests or broadening scope.
- [ ] Commit: `feat: ship usable experimental executive kernel`.

## Completion gate and design coverage

The first kernel is complete only when the real-worker workflow (Tasks 10/12/15), recovery (14), actual artifact verification (13), and installable distribution (19) all work. A live acceptance run needs separately authorized credentials/budget and is unverified until performed. Credential-free CI establishes infrastructure behavior, not model quality.

| Design sections | Tasks |
| --- | --- |
| Intent, boundaries, approach (1–4) | Release scope; 14–15, 19 |
| Records, persistence, retention (5–6) | 1–4 |
| Intake, authority, gates (7–8) | 5–7, 11, 13 |
| Attention and continuity (9) | 3, 8, 14 |
| Context and workers (10) | 9–10 |
| Freshness and execution (11) | 7, 11–12 |
| Budgets (12) | 6, 10–11, 14 |
| Verification and feedback (13) | 7, 13, 17 |
| Recovery and interaction (14) | 14–16, 19 |
| Acceptance scenario (15) | 16 |
| Compatibility and horizons (16–17) | 18–19; subsequent learning/federation retained explicitly |
| Evaluation (18) | 16–17 |
| Builder decisions and research (19–20) | Pinned choices and interfaces here; source grounding in the design |

Later plans should use these events, receipts, and evaluations for offline policy promotion, then establish a second implementation and transport conformance before federated commitment negotiation. They must retain scoped authority, explicit acceptance, and uncertain-outcome semantics across partitions. This sequence preserves the long-term vision while making the first kernel useful on its own.

## Self-review and execution handoff

Self-review completed against the approved design: all local-kernel sections map to tasks; all five Review Focus cases have owning tests; interfaces and list-valued summary fields were checked across tasks. The review added artifact-draft ingestion, explicit owner resolution of unknown actions/usage, and episode-to-work progression so the real-worker path is complete. Later adaptive/federation behavior remains explicitly deferred, with present contracts supporting it. This was a self-review, not an implementation or external code review.

Recommended execution: subagent-driven implementation with a fresh review gate per task and a whole-branch review at the end. Transaction, authority, and recovery mistakes cross component boundaries; explicit task interfaces enable independent review. Native execution with final independent review remains available. Select the execution method after reviewing the plan. Writing this document does not start implementation.
