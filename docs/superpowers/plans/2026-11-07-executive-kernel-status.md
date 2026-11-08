# Executive kernel delivery status

This continuation implements Tasks 1–19 of the [implementation plan](2026-11-07-executive-kernel.md): the first usable experimental local executive kernel. The [approved design](../specs/2026-11-07-executive-kernel-design.md) remains the long-term target. The [operator guide](../../executive-kernel.md) covers installed use. No version bump, tag, publication or paid inference is part of this delivery.

## Implemented

- Immutable typed contracts, versioned records, commands, configuration and receipts.
- Single-writer SQLite journal, atomic materialized state, inbox/outbox records, migrations and replay validation.
- Deterministic lifecycle reducers, pause/resume/cancel and completion invalidation events.
- Scoped artifact storage, integrity checks, staging, retention and durable revocation.
- Authenticated intake, deduplication, bounded backpressure, trusted constraints and scoped action-admission policy.
- Hierarchical reservations, ancestor CAS, conservative unknown usage, overage accounting and owner usage resolution.
- Evidence memory, typed proposal validation, contradiction handling and freshness checks.
- Deterministic scheduling, retained commitments, bounded context assembly and advisory observations separated from material dependencies.
- Proposal-only OpenAI worker, explicit model/credential references, trusted per-call bounds, atomic admission and usage settlement.
- Scoped local read/write/check adapters, exact action approval, authenticated check receipts and current artifact verification.
- Sole-writer coordinator, foreground JSONL CLI, in-flight correction, pause/restart/resume and conservative uncertain-effect recovery.
- Integrated credential-free release scenario, crash/fault injection and evaluation harness with integrity-checked evidence and explicit synthetic/observed labels.
- Legacy observation/callable bridges, 69 full versioned DTO/union schemas and owner-only redacted observation export.
- Installed executive console entrypoint, wheel tests from another cwd, temporary credential-free smoke and complete local operator documentation.

Tasks 1–19 received independent scoped review. The subsequent whole-branch review found five integration defects: commitment episode budgets, historical grant fencing, false draft-rejection accounting conflicts, historical evidence requests and child-goal budget ancestry. The coordinated final fix addresses all five with cross-component regressions. Its independent scoped re-review and final-head CI remain controller gates. Self-review and local checks do not substitute for those gates. The retained continuation records preserve all earlier reports and evidence.

## Validation and remaining limits

Credential-free tests exercise real journal/coordinator/adapter/verification/recovery infrastructure with deterministic workers and fake provider responses. Installed-wheel checks remove checkout path injection, verify legacy exports, run the release scenario, load config/goals, restart/inspect/export and reject incomplete live setup before calls. The installed optional real-SDK check constructs/closes a client using a dummy named credential with requests/network blocked. This is constructor compatibility evidence only.

The Task 19 report records clean-environment `make check`, `make package`, `make smoke`, acceptance/crash tests, explicit version validation and outside-checkout installation provenance. Python 3.10–3.14 final-head CI evidence belongs to the delivery gate; do not infer it from one local interpreter or the fixture's static grammar check. Existing SDK compatibility remains unchanged.

Before the final integration fixes, local Python 3.14.3 validation passed all 1,597 tests with lint and strict type checks, and distribution/installed console checks passed. The pre-fix f4313e7 hosted Python 3.10–3.14, workflow lint and wheel smoke jobs also passed. These historical results do not validate the amended runtime. Final-fix source/package and fresh installation evidence is recorded separately below.

The final coordinated fix passed one complete local Python 3.14.3 `make check`: 1,620 tests, Ruff and strict mypy across 61 source files. Sequential `make package smoke` passed both distribution checks, version validation and installed legacy/executive demos. The executive fixture completed with no failures and zero forbidden dispatches. The 23 new regression cases cover all five review findings, current commitment/ancestor ceilings, foreign scope rejection, in-flight revocation/expiry, durable request supersession and goal-ancestry CAS. Fresh corrected-snapshot Git installation and its README pin update are the next delivery step.

Source distributions explicitly exclude the local `.superpowers` continuation workspace. A temporary-project regression seeds a harmless private record, builds the actual source archive and verifies exclusion while retaining executive code. On-disk continuation records remain preserved.

Fresh outside-checkout base and `llm` Git installs verified version 0.1.0 and exact `direct_url.json` commit [030eecc136f8386ba605dfb72851a243f66d641c](https://github.com/signalgatingprotocol/python-sdk/commit/030eecc136f8386ba605dfb72851a243f66d641c). The README Quick Start, legacy demo, completed executive scenario, inspection/export and real-SDK constructor/close check passed from those installs. The `otel` command resolved successfully in a separate fresh environment with `--dry-run`; no telemetry runtime behavior was tested. These are historical Task 19 installation results. The README still selects this earlier snapshot until a corrected runtime commit is pushed and verified through fresh literal Git installs.

Paid/live inference, provider billing correctness, model task quality, production savings and security benefit claims remain unverified. The kernel mediates registered calls within a trusted host process; configured programs and external legacy runtimes are not sandboxed. Unknown effects require reconciliation, not an exactly-once claim or blind retry. Portable exports are observation-only, not backups/imports or executable authority.

Adaptive policy promotion remains offline future work grounded in events, receipts and explicit evaluations. Federation requires a second implementation and transport conformance before commitment negotiation, retaining scoped authority, explicit acceptance and uncertain-outcome semantics across partitions.

## Enforced continuation contracts

- Intake prepares batches. Commit before returning a durable receipt; exact retries retrieve the canonical persisted receipt. Queue admission must be serialized by the coordinator.
- Bind worker configuration IDs to registered principals and trusted argument-to-resource scope extractors. Repeat authority/freshness checks immediately before dispatch.
- BudgetPolicy returns event-only batches. Combine new subjects and reservations in one canonical state.changed bundle with ancestor CAS. Strict money defaults closed. Record versioned trusted quote provenance; unknown usage remains held. Inject BudgetPolicy.resolve_usage into Intake.
- Stage artifacts outside transactions and release staged pins in finally. Verifiers must read through ArtifactStore, not trust artifact reference metadata alone.
- Consume AttachEvidence/Reverify signals. Corrections and invalidated evidence must reopen dependent completion and reevaluate suspended work while preserving owner pause.
- Keep one current VerificationResult per globally unique criterion ID; update its stable ID/revision and retain historical checks in the journal.

## Recorded implementation decisions

These historical rulings preserve implementation context, including the earlier Tasks 1–6 delivery cutoff. The completed continuation scope and current review state above supersede that former cutoff; no historical ruling is deleted.

Ruling: Use the managed isolated worktree and explicit elevated writes there — source checkout changed during planning — costs an isolated branch that must be intentionally integrated later.
Ruling: Worktree is clean of unrelated untracked decision tests; leave them in the source checkout — preserves user work and gives a reliable baseline — integration must not silently adopt them.
Ruling: Stage CLI demo/export integration tests when Tasks 16/18 deliver those functions; Task 15 tests its actual implemented operations — plan references later functions — final Task 19 must verify every command.
Ruling: Use existing local client protocol with structural optional usage extraction, recording unknown usage when absent — protocol does not declare usage — provider billing cannot be guessed.
Ruling: Independent reviews cover each task; no parallel implementation writers — follows requested workflow — slower but avoids shared-interface conflicts.
Ruling: Live paid calls and CI-host evidence are distinct from local validation; do not fabricate either — no credentials/spending authorization supplied — first usable code can be fully tested with fake clients and documented live setup.
Ruling: Proceed with isolated kernel work; baseline timing test is unrelated and will be reported separately if it persists — avoids changing unrelated behavior — final full checks may still expose host-load flakiness.
Ruling: ArtifactDraft validates UTF-8 but staging enforces configured byte limits, including collections — fixed schema ceilings would defeat approved overrides and worker-supplied limits would be untrusted — staging tests must prove pre-persistence rejection.
Ruling: Task 2 may extend CommitBatch with typed InboxAcknowledgment and OutboxEntry and expose store queries — Task1 omitted required atomic inbox/outbox slots — minimal extensions must be independently reviewed. Acknowledgments bind principal/producer/external ID, payload digest and receipt; outbox points at authoritative ActionRecord without duplicate lifecycle state.
Ruling: Store assigns authoritative monotonically increasing journal sequence and returns assigned events — caller placeholder cannot determine commit ordering — future reducers consume persisted sequence.
Ruling: Task 3 integrates deterministic reduction into Store.commit and updates Task2 fixtures to canonical state.changed post-image events — arbitrary unrelated event/record writes violate journal source-of-truth — costs stricter internal API before downstream consumers exist. Store derives changed views and validates explicit supplied records match; preserves sequence/CAS/inbox/outbox guarantees. Validate whole-bundle references/cycles, not input-order partial states.
Ruling: One current VerificationResult per globally unique criterion ID; rechecks revise its stable record ID and journal retains history — avoids ambiguity between past failed and current accepted checks — Task3 enforces uniqueness, Task13 must update rather than append duplicate current records.
Ruling: Task4 adds ArtifactRef.valid=True and artifact_invalidated event_type (empty payload targeting current ref), reducer revokes validity and advances revision — deletion otherwise cannot be journaled or invalidate access — Task7 owns dependent discovery. ArtifactStore reloads authoritative ACL/ref by ID; stage_drafts may create uncommitted refs with configured individual/aggregate limits and GC protection.
Ruling: Task4 may expose Store.check_writable as reused closed/fork/readonly guard before blob mutation. Staged refs are pinned until release_staged; Task14 must release in finally. Readonly artifact byte reads use a fresh existing Store.open_readonly snapshot for ACL/ref validity, not the caller pinned historical view — honors revocations committed before read while retaining operator export — costs fresh snapshot validation on readonly reads.
Ruling: Task5 adds nonpersisted CommitBatch.response_receipt; Kernel returns canonical persisted receipt only after commit, duplicates return original without new writes. Coordinator supplies validated queue depths; capacity before ACK, dedup before backpressure.
Ruling: Authority uses trusted worker-config-ID to registered Principal mapping plus capability specs/resource extractors — worker names and payload roles cannot establish identity or resource authority — Task11/14 must supply operator bindings.
Ruling: ResolveConflict gains explicit mutually exclusive constraint/belief resolution modes with current revision references; constraint mode names goal/revision, exact conflicting constraints, replacements and explanation while preserving unrelated constraints — generic revisions must not silently clear conflicts — modest DTO extension before public release. Trusted rules report conflicts; no free-text contradiction oracle.
Ruling: ResolveUsage uses injected usage_resolver(ResolveUsage, StateView)->CommitBatch after owner authorization. Task6 provides real budget settlement and integration test; Task14 wires it. Missing resolver returns deny/unsupported_operation, never a settlement acknowledgment. Task5 tests authorization and callback plumbing.
Ruling: Intake hashes producer/external identity with uniform namespace+SHA256 for inbox keys, retaining accepted provenance in events/signals — bounds hostile rejection metadata while retaining exact deterministic deduplication — Store.acknowledgment callers must use Intake mapping. Durable denials bind receipts; unknown identities, duplicate conflicts, transient backpressure, and absent usage resolver do not bind settlement acceptance.
Ruling: Permit state.changed promotion of an exact existing pre-event Goal-embedded Constraint to standalone at unchanged revision when no standalone exists — preserves authoritative identity/history for removed rev2+ constraints without inventing revision1 history or duplicating all active constraints — surgical reducer exception requires owner-goal CAS, exact content checks, negative tests and replay equivalence. All genuinely new records still start at revision1.
Ruling: Task5 fix may add surgical Store.commit owner-goal CAS enforcement for revision>1 embedded-constraint promotion plus test_store regression — direct Store callers must not bypass the same freshness requirement — no general transaction redesign; ordinary new revision1 insert behavior unchanged.
Ruling: Task6 reserve adds optional subject_id, default budget_id for an unbound budget-scope reservation; production episodes/actions/verifiers pass actual subject_id and atomically commit reservation with authorization. No self-referential reservation subject. BudgetPolicy strict_money=True by default; reserve trusted_maximum_quote=False requires operator-trusted bound provenance, finite cost for strict mode. Explicit nonstrict mode never interprets unknown liability as zero and must not oversubscribe bounded ancestors. Tokens/calls/cost use integer arithmetic; elapsed_seconds retains existing finite numeric schema.
Ruling: Reservation adds trusted_maximum_quote: bool=False, backward-compatible optional field recording the trusted adapter assertion. Later adapters/coordinator retain versioned configured quote provenance. Nonstrict unknown maximum cannot be admitted against finite ancestor monetary ceilings merely by marking uncertainty afterward.
Ruling: User requested session wrap, PR, independent review, merge and local cleanup. Close this delivery at completed Tasks1–6, explicitly label foundations and retain Tasks7–19 for continuation; do not claim a usable kernel.
