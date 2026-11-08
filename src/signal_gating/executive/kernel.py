"""One local writer coordinating bounded workers and durable control.

Python workers/verifiers/adapters are trusted host code. Fixture workers and
injected deterministic verifiers must be free local computations. Metered model
work requires configured reservations; provider clients must disable retries.
No worker callback receives Store or writes authoritative state.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import TypeAdapter

from .actions import CapabilityRegistry, Dispatcher
from .adapters.local_tools import LocalWorkspace, StoreArtifacts
from .adapters.openai_worker import OpenAIWorker
from .artifacts import ArtifactStore
from .authority import Authority, reference
from .budgets import BudgetPolicy
from .config import KernelConfig
from .context import ContextAssembler, TokenCounter, UTF8ByteCounter
from .errors import ExecutiveError
from .inspection import inspect_state
from .intake import Intake
from .memory import records_by_id
from .messages import (
    AuthenticatedCommand,
    Command,
    CommitBatch,
    CompletionClaim,
    CreateGoal,
    EvidenceRequest,
    ReviseGoal,
)
from .proposals import ProposalValidator
from .records import (
    ActionRecord,
    ArtifactRef,
    Belief,
    Budget,
    CapabilitySpec,
    ContextPacket,
    DecisionReceipt,
    Episode,
    ExecutiveSignal,
    Goal,
    Grant,
    JournalEvent,
    KernelView,
    Limits,
    PersistentRecord,
    Principal,
    Ref,
    Reservation,
    RunSummary,
    Scope,
    StateView,
    TaskFrame,
    Usage,
    WakeCondition,
    utc_now,
)
from .recovery import recover
from .reducers import reduce_event
from .scheduler import AttentionPolicy, Scheduler, _fresh, wake
from .store import Store
from .verification import VerificationService, Verifier
from .workers import EpisodeRunner, Worker, WorkerError, _wait_until

_RECORD = TypeAdapter(PersistentRecord)
_RETRYABLE_VERIFICATION = {
    "late_verification_result",
    "verification_snapshot_changed",
    "verification_configuration_changed",
}


@dataclass
class _VerificationJob:
    sequence: int
    deadline: datetime
    configuration: str
    validity_boundary: datetime | None
    task: asyncio.Task[CommitBatch | str | None]


class Clock(Protocol):
    def now(self) -> datetime: ...
    def monotonic(self) -> float: ...


class SystemClock:
    def now(self) -> datetime:
        return utc_now()

    def monotonic(self) -> float:
        return time.monotonic()


def _reads(state: StateView) -> tuple[Ref, ...]:
    return tuple(reference(r) for r in state.records)


def _project(state: StateView, events: tuple[JournalEvent, ...]) -> StateView:
    for event in events:
        state = reduce_event(state, event.model_copy(update={"sequence": state.sequence + 1}))
    return state


class Kernel:
    """Foreground coordinator. Construct with open, then use as an async context.

    ``store``, ``artifacts``, ``workspace``, ``worker_principals`` and
    ``verifier_principal`` are trusted host integration surfaces. Adapters must
    use this sole Store. Share selected artifacts explicitly with these
    principals using ShareArtifact; publication never broadens access.
    """

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        config: KernelConfig,
        workers: Mapping[str, Worker],
        capabilities: CapabilityRegistry,
        verifiers: Mapping[str, Verifier],
        clock: Clock,
    ) -> Kernel:
        return cls(
            root,
            config=config,
            workers=workers,
            capabilities=capabilities,
            verifiers=verifiers,
            clock=clock,
        )

    def __init__(
        self,
        root: Path,
        *,
        config: KernelConfig,
        workers: Mapping[str, Worker],
        capabilities: CapabilityRegistry,
        verifiers: Mapping[str, Verifier],
        clock: Clock,
    ) -> None:
        self.config, self.clock, self.registry = config, clock, capabilities
        self.workers = dict(workers)
        self.worker_configs = {w.id: w for w in config.workers}
        if self.workers.keys() != self.worker_configs.keys():
            raise ValueError("Every worker requires exactly one trusted WorkerConfig")
        self.store = Store.open(root)
        try:
            owners = [
                r
                for r in self.store.view().records
                if isinstance(r, Principal) and "owner" in r.roles
            ]
            if not owners:
                if (
                    not config.bootstrap_owner
                    or config.owner is None
                    or "owner" not in config.owner.roles
                ):
                    raise ValueError("New kernel requires explicit bootstrap_owner and owner role")
                self.owner = config.owner
            else:
                self.owner = config.owner or owners[0]
                if self.owner not in owners:
                    raise ValueError("Configured owner must match the persisted identity")
            self._closed = False
            self._stopped = False
            self._changed = asyncio.Event()
            self._shutdown = asyncio.Event()
            self._verification_tasks: set[asyncio.Task[CommitBatch]] = set()
            self._verification_jobs: dict[str, _VerificationJob] = {}
            self._verification_retries: set[str] = set()
            self._verification_attempts: dict[str, tuple[int, str, datetime | None]] = {}
            self._verification_progress = self.store.view().sequence
            self._active: dict[str, asyncio.Task[None]] = {}
            self._driving = asyncio.Lock()
            self._seen_sequence = self.store.view().sequence
            self._verification_due: set[str] = set()
            self._pending_signals: list[str] = []
            self.counters: dict[str, TokenCounter] = {"utf8-bytes-v1": UTF8ByteCounter()}
            self.worker_principals = {
                w.id: w.principal
                or Principal(
                    id=uuid5(NAMESPACE_URL, self.owner.id + ":worker:" + w.id).hex,
                    created_at=self.owner.created_at,
                    name=w.id,
                    trust_domain="kernel-local",
                    authenticated_origin="operator-worker-config",
                    roles=("worker",),
                )
                for w in config.workers
            }
            self.verifier_principal = config.verifier_principal or Principal(
                id=uuid5(NAMESPACE_URL, self.owner.id + ":verifier").hex,
                created_at=self.owner.created_at,
                name="kernel-verifier",
                trust_domain="kernel-local",
                authenticated_origin="operator-verifier-config",
                roles=("verifier",),
            )
            if (
                any("worker" not in p.roles for p in self.worker_principals.values())
                or "verifier" not in self.verifier_principal.roles
            ):
                raise ValueError("Configured worker/verifier principals require their exact roles")
            identities = (self.owner, *self.worker_principals.values(), self.verifier_principal)
            if len({p.id for p in identities}) != len(identities):
                raise ValueError("Owner, workers and verifier require distinct identities")
            state = self.store.view()
            indexed = records_by_id(state)
            additions: list[PersistentRecord] = []
            for principal in identities:
                if principal.id in indexed and indexed[principal.id] != principal:
                    raise ValueError("Configured principal differs from persisted identity")
                if principal.id not in indexed:
                    additions.append(principal)
            if not owners:
                additions.extend(config.bootstrap_grants)
            if additions:
                self._commit(self._batch(tuple(additions), state))
            self.artifacts = ArtifactStore(self.store, max_artifact_bytes=config.max_artifact_bytes)
            self.workspace: LocalWorkspace | None = None
            if config.workspace:
                workspace = config.workspace
                if workspace.allow_symlinks:
                    raise ValueError("LocalWorkspace does not support symlinks")
                self.workspace = LocalWorkspace(
                    workspace.root,
                    readable=workspace.readable_paths,
                    writable=workspace.writable_paths,
                    commands={c.id: c.argv for c in workspace.named_checks},
                    check_options={c.id: c for c in workspace.named_checks},
                    artifacts=StoreArtifacts(self.store),
                    clock=clock.monotonic,
                )
                local_ids = {s.id for s, _ in self.workspace.capabilities()}
                if local_ids & capabilities.specs.keys():
                    raise ValueError("Workspace capabilities must be registered by this kernel")
                restored = self._restored_local_specs(self.workspace)
                self.workspace.register(capabilities, restored_specs=restored)
            self.budgets = BudgetPolicy(strict_money=config.strict_money, clock=clock.now)
            self.authority = Authority(
                worker_principals={k: p.id for k, p in self.worker_principals.items()},
                capabilities=capabilities.specs,
                resource_scopes={
                    k: b.resource_scope
                    for k, b in capabilities._bindings.items()
                    if b.resource_scope is not None
                },
            )
            self.intake = Intake(
                self.store,
                authority=self.authority,
                max_signal_bytes=config.max_signal_bytes,
                max_queued_ordinary_items=config.max_queued_ordinary_items,
                max_queued_control_items=config.scheduler.control_batch_size,
                clock=clock.now,
                usage_resolver=self.budgets.resolve_usage,
            )
            self.scheduler = Scheduler(config, clock=clock.now)
            self.runner = EpisodeRunner(
                is_current=self._is_current,
                clock=clock.now,
                wait_until=self._wait_until,
                max_active=config.max_active_episodes,
            )
            self.dispatcher = Dispatcher(
                self.store,
                authority=self.authority,
                budgets=self.budgets,
                registry=capabilities,
                clock=clock.now,
                wait_until=self._wait_until,
                max_active=config.max_active_episodes,
                max_retries=0,
            )
            self.verification = VerificationService(
                self.artifacts,
                principal=self.verifier_principal,
                authority=self.authority,
                local_checks={
                    key: (self.workspace, capabilities)
                    for key in config.verifier_ids
                    if self.workspace is not None
                },
                verifiers=verifiers,
                repair_rounds=config.verification_repair_rounds,
                clock=clock.now,
            )
            self._commit(recover(self.store, registry=capabilities, now=clock.now()))
            self._commit(
                CommitBatch(
                    events=(
                        self._audit(
                            "kernel_configuration",
                            {
                                "config": config.model_dump(
                                    mode="json",
                                    exclude={
                                        "workspace": {"named_checks": {"__all__": {"environment"}}}
                                    },
                                ),
                                "capabilities": [
                                    s.model_dump(mode="json") for s in capabilities.specs.values()
                                ],
                                "workspace_digest": self.workspace.configuration_digest
                                if self.workspace
                                else None,
                            },
                        ),
                    )
                )
            )
            # Requests and claims without a completion receipt survive process loss.
            state = self.store.view()
            completed_subjects = {
                r.subject_id
                for r in state.records
                if isinstance(r, DecisionReceipt)
                and r.policy_version.startswith("verification")
                and r.reason_code not in _RETRYABLE_VERIFICATION
            }
            self._pending_signals = [
                r.id
                for r in state.records
                if isinstance(r, ExecutiveSignal)
                and r.signal_kind in {"completion_claim", "attach_evidence", "reverify"}
                and r.id not in completed_subjects
            ]
            self._verification_due.update(
                r.id for r in state.records if isinstance(r, Goal) and r.status == "completed"
            )
        except BaseException:
            self.store.close()
            raise

    def _restored_local_specs(self, workspace: LocalWorkspace) -> dict[str, CapabilitySpec] | None:
        for event in reversed(self.store.events()):
            if (
                event.producer_id != "kernel"
                or event.event_type != "audit.recorded"
                or event.payload.get("reason_code") != "kernel_configuration"
            ):
                continue
            details = event.model_dump(mode="json")["payload"]["details"]
            if details.get("workspace_digest") != workspace.configuration_digest:
                return None
            ids = {s.id for s, _ in workspace.capabilities()}
            return {
                s.id: s
                for raw in details["capabilities"]
                if (s := CapabilitySpec.model_validate(raw)).id in ids
            }
        return None

    def set_token_counter(self, identifier: str, counter: TokenCounter) -> None:
        """Trusted startup injection for WorkerConfig.token_counter_id."""
        if self._active:
            raise ValueError("Configure token counters before starting work")
        self.counters[identifier] = counter

    def _event(self, records: tuple[PersistentRecord, ...]) -> JournalEvent:
        return JournalEvent(
            created_at=self.clock.now(),
            sequence=1,
            event_type="state.changed",
            producer_id="kernel",
            principal_id=self.owner.id,
            subject=reference(self.owner),
            payload={"records": [r.model_dump(mode="json") for r in records]},
        )

    def _audit(self, reason: str, details: dict[str, Any]) -> JournalEvent:
        return JournalEvent(
            created_at=self.clock.now(),
            sequence=1,
            event_type="audit.recorded",
            producer_id="kernel",
            principal_id=self.owner.id,
            subject=reference(self.owner),
            payload={"reason_code": reason, "details": details},
        )

    def _batch(self, records: tuple[PersistentRecord, ...], state: StateView) -> CommitBatch:
        return CommitBatch(events=(self._event(records),), expected_revisions=_reads(state))

    def _commit(self, batch: CommitBatch) -> DecisionReceipt | None:
        if batch.events or batch.inbox_acknowledgments or batch.outbox:
            self.store.commit(batch)
            self._changed.set()
        if batch.response_receipt:
            receipt = records_by_id(self.store.view()).get(batch.response_receipt.id)
            if isinstance(receipt, DecisionReceipt):
                return receipt
            # Backpressure is deliberately not a durable acknowledgement.
            return batch.response_receipt
        return None

    async def __aenter__(self) -> Kernel:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            await self.stop()
        finally:
            self.store.close()
            self._closed = True

    async def submit(self, command: Command, *, principal: Principal) -> DecisionReceipt:
        if self._closed or self._stopped:
            raise RuntimeError("Kernel is stopped")
        state = self.store.view()
        self.intake.set_queue_depths(ordinary=self._ordinary_depth(), control=0)
        batch = self.intake.accept(command, principal=principal, state=state)
        if (
            isinstance(command, CreateGoal)
            and batch.response_receipt
            and batch.response_receipt.disposition == "accept"
            and batch.events
        ):
            # Normalize postimages after Intake binds the ORIGINAL command digest.
            # Never rewrite the submitted command or its deduplication identity.
            events = []
            for event in batch.events:
                if event.event_type == "state.changed":
                    writes = [
                        _RECORD.validate_python(r)
                        for r in event.model_dump(mode="json")["payload"]["records"]
                    ]
                    for index, record in enumerate(writes):
                        if (
                            isinstance(record, Goal)
                            and record.id == command.goal.id
                            and record.budget_id is None
                        ):
                            budget = Budget(
                                created_at=self.clock.now(),
                                scope=Scope(owner_id=record.owner_id, goal_id=record.id),
                                limits=record.limits,
                                parent_id=next(
                                    (
                                        r.budget_id
                                        for r in state.records
                                        if isinstance(r, Goal) and r.id == record.parent_id
                                    ),
                                    None,
                                ),
                            )
                            writes[index] = record.model_copy(update={"budget_id": budget.id})
                            writes.append(budget)
                    event = event.model_copy(
                        update={"payload": {"records": [r.model_dump(mode="json") for r in writes]}}
                    )
                events.append(event)
            batch = batch.model_copy(update={"events": tuple(events)})
        if (
            isinstance(command, ReviseGoal)
            and command.limits is not None
            and batch.events
            and batch.response_receipt is not None
            and batch.response_receipt.disposition == "accept"
        ):
            target = records_by_id(state).get(command.goal_id)
            ceiling_budget = (
                records_by_id(state).get(target.budget_id or "")
                if isinstance(target, Goal)
                else None
            )
            if isinstance(ceiling_budget, Budget):
                # Owner limit changes revise the enforcing ceiling, never balances.
                ceiling = ceiling_budget.model_copy(
                    update={"revision": ceiling_budget.revision + 1, "limits": command.limits}
                )
                batch = batch.model_copy(
                    update={
                        "events": (*batch.events, self._event((ceiling,))),
                        "expected_revisions": _reads(state),
                    }
                )
        receipt = self._commit(batch)
        assert receipt is not None
        if receipt.disposition == "accept" and command.kind in {
            "approve_criterion",
            "revise_goal",
            "resolve_action",
            "resolve_usage",
            "share_artifact",
        }:
            goal_id = getattr(command, "goal_id", None)
            self._verification_due.update(
                r.id
                for r in self.store.view().records
                if isinstance(r, Goal) and (r.id == goal_id or r.status == "completed")
            )
        self._cancel_fenced()
        return receipt

    def _ordinary_depth(self) -> int:
        state = self.store.view()
        processed = {
            r.subject_id
            for r in state.records
            if isinstance(r, DecisionReceipt)
            and (r.reason_code.startswith("attention_") or r.policy_version == "verification-v1")
        }
        return sum(
            isinstance(r, ExecutiveSignal)
            and r.producer_id != "worker-proposal"
            and r.id not in processed
            for r in state.records
        )

    async def ingest(self, signal: ExecutiveSignal, *, principal: Principal) -> DecisionReceipt:
        if self._closed or self._stopped:
            raise RuntimeError("Kernel is stopped")
        self.intake.set_queue_depths(ordinary=self._ordinary_depth(), control=0)
        receipt = self._commit(
            self.intake.accept(signal, principal=principal, state=self.store.view())
        )
        assert receipt is not None
        return receipt

    def inspect(self, goal_id: str | None = None) -> KernelView:
        return inspect_state(self.store.view(), goal_id)

    async def _wait_until(self, deadline: datetime) -> None:
        timer = getattr(self.clock, "wait_until", None)
        if timer is not None:
            await timer(deadline)
        else:
            # Convert the injected UTC delta to a real timer without assuming
            # its wall-clock epoch equals the host's UTC clock.
            await _wait_until(utc_now() + max(deadline - self.clock.now(), timedelta()))

    def _is_current(self, episode: Episode, packet: ContextPacket) -> bool:
        state = self.store.view()
        records = records_by_id(state)
        task, goal = records.get(episode.task_id), records.get(packet.goal.id)
        principal = self.worker_principals.get(episode.worker_id)
        if (
            self._stopped
            or records.get(episode.id) != episode
            or records.get(packet.id) != packet
            or not isinstance(task, TaskFrame)
            or not isinstance(goal, Goal)
            or task != packet.task
            or goal != packet.goal
            or task.status != "running"
            or goal.status != "active"
            or principal is None
            or records.get(principal.id) != principal
            or not _fresh(
                task.model_copy(
                    update={
                        "dependencies": tuple(
                            dict.fromkeys((*task.dependencies, *packet.mandatory_read_set))
                        )
                    }
                ),
                records,
                self.clock.now(),
            )
        ):
            return False
        if any(
            records.get(ref.id) is None or records[ref.id].revision != ref.revision
            for ref in packet.mandatory_read_set
        ):
            return False
        for ref in packet.mandatory_read_set:
            authority = records.get(ref.id)
            if isinstance(authority, Grant) and (
                authority.revoked
                or (authority.expires_at is not None and self.clock.now() >= authority.expires_at)
            ):
                return False
        reservation = records.get(episode.reservation_id or "")
        if not isinstance(reservation, Reservation) or reservation.subject_id != episode.id:
            return False
        try:
            self.budgets.reserve(
                state,
                reservation.budget_id,
                reservation.maximum,
                reservation.id,
                subject_id=episode.id,
                trusted_maximum_quote=reservation.trusted_maximum_quote,
            )
            for ref in packet.manifest.included:
                artifact = records.get(ref.id)
                if not isinstance(artifact, ArtifactRef):
                    continue
                self.artifacts.read(artifact, principal=principal, goal_id=goal.id)
        except ExecutiveError:
            return False
        return True

    def _cancel_fenced(self) -> None:
        records = records_by_id(self.store.view())
        for identifier, task in self._active.items():
            record = records.get(identifier)
            if isinstance(record, Episode) and record.status != "running" and not task.done():
                task.cancel()

    def _observe(self, episode: Episode, usage: Usage) -> None:
        if episode.reservation_id:
            self._commit(
                self.budgets.observe_usage(self.store.view(), episode.reservation_id, usage)
            )

    def _drain(self) -> None:
        for episode, usage in self.runner.take_late_usage():
            self._observe(episode, usage)
        for issued, observation in self.dispatcher.take_late_observations():
            self.dispatcher.record_observation(issued, observation)

    def _observe_events(self) -> None:
        events = self.store.events(after=self._seen_sequence)
        for event in events:
            images = event.model_dump(mode="json")["payload"].get("records", ())
            # A stale-attempt receipt is not new evidence for another retry.
            if not images or not all(
                raw.get("kind") == "receipt"
                and raw.get("policy_version") == "verification-v1"
                and raw.get("reason_code") in _RETRYABLE_VERIFICATION
                for raw in images
            ):
                self._verification_progress = event.sequence
            state = self.store.view()
            self._commit(
                CommitBatch(
                    events=wake(state, event, self.clock.now()), expected_revisions=_reads(state)
                )
            )
            if event.producer_id not in {
                "verification",
            } and event.event_type in {"state.changed", "artifact_invalidated"}:
                images = event.model_dump(mode="json")["payload"].get("records", ())
                if event.event_type == "artifact_invalidated" or any(
                    raw.get("kind") in {"budget", "reservation", "artifact", "belief"}
                    for raw in images
                ):
                    self._verification_due.update(
                        r.id
                        for r in state.records
                        if isinstance(r, Goal) and r.status == "completed"
                    )
            if event.event_type == "state.changed":
                for raw in event.model_dump(mode="json")["payload"]["records"]:
                    record = _RECORD.validate_python(raw)
                    if (
                        isinstance(record, ExecutiveSignal)
                        and record.signal_kind
                        in {"completion_claim", "attach_evidence", "reverify"}
                        and record.id not in self._pending_signals
                        and record.id not in self._verification_jobs
                    ):
                        self._pending_signals.append(record.id)
                    elif (
                        isinstance(record, ExecutiveSignal)
                        and record.producer_id != "worker-proposal"
                    ):
                        self._commit(
                            AttentionPolicy(clock=self.clock.now).prepare(record, self.store.view())
                        )
        self._seen_sequence = self.store.view().sequence

    async def _verification_call(
        self, child: asyncio.Task[CommitBatch], deadline: datetime
    ) -> CommitBatch | str | None:
        # Only prepare here. The coordinator consumes delivery on its writer turn.
        # Detached, cancellation-resistant children retain their capacity slot.
        timer = asyncio.create_task(self._wait_until(deadline))
        shutdown = asyncio.create_task(self._shutdown.wait())
        try:
            done, _ = await asyncio.wait(
                (child, timer, shutdown), return_when=asyncio.FIRST_COMPLETED
            )
            if self._stopped:
                return None
            if timer in done or self.clock.now() >= deadline:
                return "verification_timeout"
            try:
                return child.result()
            except asyncio.CancelledError:
                return "verification_failed"
            except ExecutiveError as exc:
                if exc.code == "stale_verification_state":
                    return "verification_snapshot_changed"
                return "verification_failed"
            except Exception:
                return "verification_failed"
        finally:
            if not child.done():
                child.cancel()
            timer.cancel()
            shutdown.cancel()
            await asyncio.gather(timer, shutdown, return_exceptions=True)

    def _finish_verifications(self) -> None:
        for identifier, job in tuple(self._verification_jobs.items()):
            if not job.task.done():
                continue
            del self._verification_jobs[identifier]
            self._verification_retries.discard(identifier)
            result = job.task.result()
            if self._stopped or result is None:
                continue
            # Deadline wins even if both child and timer were ready, or delivery
            # waited for this writer turn. No await separates these guards/commit.
            if self.clock.now() >= job.deadline:
                result = "verification_timeout"
            state = self.store.view()
            if isinstance(result, CommitBatch):
                if set(result.expected_revisions) != set(_reads(state)):
                    result = "verification_snapshot_changed"
                elif self.verification._configuration() != job.configuration:
                    result = "verification_configuration_changed"
                elif self._verification_boundary() != job.validity_boundary:
                    result = "verification_snapshot_changed"
            if isinstance(result, str):
                if result not in _RETRYABLE_VERIFICATION:
                    self._verification_failure(identifier, result)
                    continue
                receipt = DecisionReceipt(
                    created_at=self.clock.now(),
                    subject_id=identifier,
                    disposition="stale",
                    reason_code=result,
                    policy_version="verification-v1",
                )
                result = self._batch((receipt,), state).model_copy(
                    update={"response_receipt": receipt}
                )
            self._commit(result)
            response = result.response_receipt
            if response is not None and response.reason_code in _RETRYABLE_VERIFICATION:
                # Other stale-attempt receipts change full snapshots without
                # advancing ordinary progress. Retry that real interference once
                # in isolation; our own newly committed receipt is not a trigger.
                if state.sequence != job.sequence:
                    self._verification_retries.add(identifier)
                subject = records_by_id(state).get(identifier)
                if isinstance(subject, ExecutiveSignal):
                    if identifier not in self._pending_signals:
                        self._pending_signals.append(identifier)
                elif isinstance(subject, Goal):
                    self._verification_due.add(identifier)

    def _verification_failure(self, subject_id: str, reason: str) -> None:
        state = self.store.view()
        subject = records_by_id(state).get(subject_id)
        goal_id = (
            subject.id
            if isinstance(subject, Goal)
            else subject.scope.goal_id
            if isinstance(subject, ExecutiveSignal)
            else None
        )
        writes: list[PersistentRecord] = [
            DecisionReceipt(
                created_at=self.clock.now(),
                subject_id=subject_id,
                disposition="needs_human",
                reason_code=reason,
                policy_version="verification-v1",
            )
        ]
        writes.extend(
            task.model_copy(
                update={
                    "revision": task.revision + 1,
                    "status": "blocked",
                    "wake_condition": WakeCondition(kind="human_decision", reason_code=reason),
                }
            )
            for task in state.records
            if isinstance(task, TaskFrame)
            and task.goal_id == goal_id
            and task.status == "verifying"
        )
        self._commit(self._batch(tuple(writes), state))

    def _verification_boundary(self) -> datetime | None:
        # Time may invalidate a prepared proof without changing any record revision.
        now = self.clock.now()
        return max(
            (
                boundary
                for record in self.store.view().records
                if isinstance(record, Belief)
                for boundary in (record.valid_from, record.valid_until)
                if boundary is not None and boundary <= now
            ),
            default=None,
        )

    def _verify(self) -> None:
        # Ordinary changed-state/configuration/time retries remain concurrent.
        # If only audit interference exhausted a key, retry in isolation once
        # other jobs settle. Our own stale receipt never grants another try.
        key = (
            self._verification_progress,
            self.verification._configuration(),
            self._verification_boundary(),
        )
        self._verification_retries.intersection_update(
            identifier
            for identifier, attempted in self._verification_attempts.items()
            if attempted == key
        )
        if self._verification_retries.intersection(self._verification_jobs):
            return
        for identifier in (*self._pending_signals, *sorted(self._verification_due)):
            if (
                sum(not task.done() for task in self._verification_tasks)
                >= self.config.max_active_episodes
            ):
                break
            retry = identifier in self._verification_retries
            if (
                identifier in self._verification_jobs
                or (self._verification_attempts.get(identifier) == key and not retry)
                or (retry and self._verification_jobs)
            ):
                continue
            state = self.store.view()
            subject = records_by_id(state).get(identifier)
            if isinstance(subject, ExecutiveSignal):
                self._pending_signals.remove(identifier)
                call = (
                    self.verification.evaluate(
                        CompletionClaim.model_validate(subject.payload["proposal"]), state
                    )
                    if subject.signal_kind == "completion_claim"
                    else self.verification.consume_request(subject, state)
                )
            else:
                self._verification_due.discard(identifier)
                call = self.verification.reverify(identifier, state)
            self._verification_attempts[identifier] = key
            child = asyncio.create_task(call)
            self._verification_tasks.add(child)

            def finished(task: asyncio.Task[CommitBatch]) -> None:
                self._verification_tasks.discard(task)
                if not task.cancelled():
                    task.exception()
                self._changed.set()

            child.add_done_callback(finished)
            deadline = self.clock.now() + timedelta(seconds=self.config.episode_timeout_seconds)
            job = asyncio.create_task(self._verification_call(child, deadline))
            job.add_done_callback(lambda _: self._changed.set())
            self._verification_jobs[identifier] = _VerificationJob(
                state.sequence, deadline, key[1], key[2], job
            )
            if retry:
                break

    def _action_slots(self) -> int:
        occupied = self.dispatcher._active_ids | {
            action.id for action, _ in self.dispatcher._detached.values()
        }
        indexed = records_by_id(self.store.view())
        scheduled = sum(
            isinstance(indexed.get(identifier), ActionRecord) and identifier not in occupied
            for identifier in self._active
        )
        return (
            self.config.max_active_episodes
            - len(self.dispatcher._pending)
            - len(self.dispatcher._late)
            - scheduled
        )

    def _remaining(self, budget: Budget, state: StateView) -> Limits:
        indexed = records_by_id(state)
        values: dict[str, Any] = {"currency": budget.limits.currency}
        for key in ("input_tokens", "output_tokens", "calls", "elapsed_seconds", "cost_microunits"):
            available = []
            node = budget
            visited: set[str] = set()
            while node.id not in visited:
                visited.add(node.id)
                maximum = getattr(node.limits, key)
                if maximum is not None:
                    actual, reserved = getattr(node.actual, key), getattr(node.reserved, key)
                    available.append(
                        0
                        if actual is None or reserved is None
                        else max(0, maximum - actual - reserved)
                    )
                parent = indexed.get(node.parent_id or "")
                if not isinstance(parent, Budget):
                    break
                node = parent
            values[key] = min(available) if available else None
        return Limits(**values)

    def _admit(self) -> tuple[tuple[Episode, ContextPacket], ...]:
        original = self.store.view()
        plan = self.scheduler.plan(original, self.clock.now())
        if not plan:
            return ()
        # Detached cancellation-resistant calls and undrained observations still
        # occupy capacity. Do not consume fairness quota or reserve another call.
        worker_slots = (
            self.config.max_active_episodes
            - len(self.runner._pending)
            - len(self.runner._late_usage)
        )
        action_slots = self._action_slots()
        unavailable_actions = self.dispatcher._active_ids | {
            a.id for a, _ in self.dispatcher._detached.values()
        }
        if (
            sum(item.kind == "task" for item in plan) > worker_slots
            or sum(item.kind == "action" for item in plan) > action_slots
            or any(item.id in unavailable_actions for item in plan)
        ):
            return ()
        selection = self.scheduler.prepare_selection(original, self.clock.now())
        state = _project(original, selection.events)
        events = list(selection.events)
        admitted: list[tuple[Episode, ContextPacket]] = []
        pins: list[ArtifactRef] = []
        failing_task: TaskFrame | None = None
        try:
            for selected in plan:
                indexed = records_by_id(state)
                item = indexed[selected.id]
                if isinstance(item, ActionRecord):
                    event = self._event(
                        (
                            item.model_copy(
                                update={"revision": item.revision + 1, "status": "reconciling"}
                            ),
                        )
                    )
                    events.append(event)
                    state = _project(state, (event,))
                    continue
                assert isinstance(item, TaskFrame)
                failing_task = next(
                    r for r in original.records if isinstance(r, TaskFrame) and r.id == item.id
                )
                goal = indexed[item.goal_id]
                assert isinstance(goal, Goal)
                wc = self.worker_configs.get(goal.worker_id)
                if wc is None:
                    raise ExecutiveError("unknown_worker", "Worker is not configured")
                budget = indexed.get(goal.budget_id or "")
                if not isinstance(budget, Budget):
                    raise ExecutiveError("missing_episode_budget", "Goal requires a budget")
                remaining = self._remaining(budget, state)
                duration = min(
                    self.config.episode_timeout_seconds,
                    remaining.elapsed_seconds
                    if remaining.elapsed_seconds is not None
                    else self.config.episode_timeout_seconds,
                )
                worker = self.workers[wc.id]
                if (
                    isinstance(worker, OpenAIWorker)
                    and worker.quote is not None
                    and worker.quote.elapsed_seconds is not None
                ):
                    duration = min(duration, worker.quote.elapsed_seconds)
                if duration <= 0:
                    raise ExecutiveError("budget_exhausted", "No episode time remains")
                running = item.model_copy(
                    update={
                        "revision": item.revision + 1,
                        "status": "running",
                        "fencing_token": item.fencing_token + 1,
                    }
                )
                planned = state.model_copy(
                    update={
                        "records": tuple(running if r.id == item.id else r for r in state.records)
                    }
                )
                counter = self.counters.get(wc.token_counter_id)
                if counter is None:
                    raise ExecutiveError(
                        "unknown_token_counter", "Configure the required token counter"
                    )
                requests = tuple(
                    EvidenceRequest.model_validate(r.payload["proposal"])
                    for r in state.records
                    if isinstance(r, ExecutiveSignal)
                    and r.signal_kind == "evidence_request"
                    and r.scope.goal_id == goal.id
                )
                packet = ContextAssembler(
                    self.artifacts,
                    worker=wc,
                    counter=counter,
                    counter_id=wc.token_counter_id,
                    capabilities=tuple(self.registry.specs.values()),
                    clock=self.clock.now,
                ).build(
                    running,
                    planned,
                    principal=self.worker_principals[wc.id],
                    remaining_limits=remaining,
                    evidence_requests=requests,
                )
                assert packet.rendered_artifact is not None
                pins.append(packet.rendered_artifact)
                episode = Episode(
                    created_at=self.clock.now(),
                    task_id=running.id,
                    id=packet.episode_id,
                    fencing_token=running.fencing_token,
                    context_packet_id=packet.id,
                    worker_id=wc.id,
                    provider=wc.kind,
                    model=wc.model or "fixture",
                    allowed_proposal_types=(
                        "belief_change",
                        "task_decomposition",
                        "action_intent",
                        "evidence_request",
                        "completion_claim",
                    ),
                    deadline=self.clock.now() + timedelta(seconds=duration),
                    reservation_id=uuid4().hex,
                )
                worker = self.workers[wc.id]
                if wc.kind == "openai":
                    if not isinstance(worker, OpenAIWorker):
                        raise ExecutiveError(
                            "invalid_worker_binding", "OpenAI kind requires bounded OpenAIWorker"
                        )
                    reservation = worker.reservation_batch(
                        self.budgets, state, budget.id, episode=episode, packet=packet
                    )
                else:
                    quote = Limits(
                        input_tokens=packet.manifest.input_tokens + wc.framing_tokens_reserved,
                        output_tokens=wc.max_output_tokens,
                        calls=1,
                        elapsed_seconds=duration,
                        cost_microunits=0,
                        currency=budget.limits.currency,
                    )
                    reservation = self.budgets.reserve(
                        state,
                        budget.id,
                        quote,
                        episode.reservation_id or "",
                        subject_id=episode.id,
                        trusted_maximum_quote=True,
                    )
                    events.append(
                        self._audit(
                            "worker_quote",
                            {
                                "source": "trusted-fixture-config",
                                "version": "fixture-v1",
                                "maximum": quote.model_dump(mode="json"),
                                "episode_id": episode.id,
                            },
                        )
                    )
                writes: list[PersistentRecord] = [
                    running,
                    packet.rendered_artifact,
                    packet,
                    episode,
                ]
                for event in reservation.events:
                    if event.event_type == "state.changed":
                        writes.extend(
                            _RECORD.validate_python(r)
                            for r in event.model_dump(mode="json")["payload"]["records"]
                        )
                    else:
                        events.append(event)
                event = self._event(tuple(writes))
                events.append(event)
                state = _project(state, (event,))
                admitted.append((episode, packet))
            self._commit(CommitBatch(events=tuple(events), expected_revisions=_reads(original)))
            for selected in plan:
                if selected.kind == "action":
                    self._launch(selected.id, self._dispatch(selected.id, reconcile=True))
            return tuple(admitted)
        except ExecutiveError as exc:
            if failing_task is None:
                raise
            current = records_by_id(self.store.view())[failing_task.id]
            assert isinstance(current, TaskFrame)
            condition = WakeCondition(
                kind="budget" if "budget" in exc.code or "quote" in exc.code else "human_decision",
                reason_code=exc.code,
            )
            self._commit(
                self._batch(
                    (
                        current.model_copy(
                            update={
                                "revision": current.revision + 1,
                                "status": "blocked",
                                "wake_condition": condition,
                            }
                        ),
                    ),
                    self.store.view(),
                )
            )
            return ()
        finally:
            self.artifacts.release_staged(pins)

    def _launch(self, identifier: str, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._active[identifier] = task
        task.add_done_callback(lambda _: self._changed.set())

    async def _episode(self, episode: Episode, packet: ContextPacket) -> None:
        try:
            result = await self.runner.run(episode, packet, self.workers[episode.worker_id])
            self._observe(episode, result.usage)
            if result.stop_reason == "proposed" and not result.proposals:
                self._fail_episode(episode, "no_progress")
                return
            drafts = self.artifacts.stage_drafts(
                result.artifact_drafts, principal=self.owner, goal_id=packet.goal.id
            )
            try:
                batch = ProposalValidator(
                    staged_drafts=drafts,
                    worker_principals={k: p.id for k, p in self.worker_principals.items()},
                    clock=self.clock.now,
                ).validate(result, episode, self.store.view())
                receipt = self._commit(batch)
            finally:
                self.artifacts.release_staged(drafts.values())
            if receipt is not None and receipt.disposition != "accept":
                self._fail_episode(episode, receipt.reason_code)
        except (WorkerError, ExecutiveError) as exc:
            self._observe(
                episode,
                exc.usage if isinstance(exc, WorkerError) else Usage(source="kernel-worker-error"),
            )
            self._fail_episode(episode, exc.code)
        except asyncio.CancelledError:
            self._observe(episode, Usage(calls=1, source="kernel-cancelled"))
            self._fail_episode(episode, "episode_fenced")

    def _fail_episode(self, episode: Episode, reason: str) -> None:
        state = self.store.view()
        indexed = records_by_id(state)
        current, task = indexed[episode.id], indexed[episode.task_id]
        writes: list[PersistentRecord] = [
            DecisionReceipt(
                created_at=self.clock.now(),
                subject_id=episode.id,
                disposition="stale",
                reason_code=reason,
            )
        ]
        if isinstance(current, Episode) and current.status == "running":
            writes.append(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "status": "expired" if reason == "episode_timeout" else "rejected",
                    }
                )
            )
        if isinstance(task, TaskFrame) and task.status == "running":
            writes.append(
                task.model_copy(
                    update={
                        "revision": task.revision + 1,
                        "status": "suspended",
                        "fencing_token": task.fencing_token + 1,
                    }
                )
            )
        self._commit(self._batch(tuple(writes), state))
        current_task = records_by_id(self.store.view())[episode.task_id]
        if (
            isinstance(current_task, TaskFrame)
            and current_task.status == "suspended"
            and not self._stopped
            and isinstance(current, Episode)
            and current.status == "running"
        ):
            self._commit(
                self.scheduler.record_failure(
                    self.store.view(), current_task.id, reason, self.clock.now()
                )
            )

    async def _dispatch(self, identifier: str, *, reconcile: bool = False) -> None:
        if reconcile:
            await self.dispatcher.reconcile_one(identifier)
        else:
            await self.dispatcher.dispatch_one(identifier)

    async def run_until_idle(self) -> RunSummary:
        async with self._driving:
            self._verification_attempts.clear()
            # A public tick also checks external bytes and temporal support.
            self._verification_due.update(
                r.id
                for r in self.store.view().records
                if isinstance(r, Goal) and r.status == "completed"
            )
            while not self._stopped:
                self._changed.clear()
                self._drain()
                for identifier, task in tuple(self._active.items()):
                    if task.done():
                        del self._active[identifier]
                        task.result()
                self._finish_verifications()
                self._observe_events()
                if self._stopped:
                    break
                state = self.store.view()
                self._commit(
                    CommitBatch(
                        events=wake(state, None, self.clock.now()), expected_revisions=_reads(state)
                    )
                )
                self._cancel_fenced()
                for record in self.store.view().records:
                    if (
                        isinstance(record, ActionRecord)
                        and record.status in {"proposed", "authorized"}
                        and record.id not in self._active
                        and self._action_slots() > 0
                        and len(self._active) < self.config.max_active_episodes
                    ):
                        self._launch(record.id, self._dispatch(record.id))
                if len(self._active) < self.config.max_active_episodes:
                    for episode, packet in self._admit():
                        self._launch(episode.id, self._episode(episode, packet))
                self._verify()
                if self._active or self._verification_jobs:
                    await self._wait_change()
                elif self._changed.is_set():
                    continue
                else:
                    return self.inspect().summary
            return self.inspect().summary.model_copy(update={"stop_reason": "stopped"})

    async def _wait_change(self) -> None:
        waiting: list[asyncio.Task[Any]] = [
            asyncio.create_task(event.wait())
            for event in (
                self._changed,
                self.runner.late_usage_ready,
                self.dispatcher.late_observations_ready,
            )
        ]
        deadline = self._next_wake()
        timer = asyncio.create_task(self._wait_until(deadline)) if deadline else None
        if timer is not None:
            waiting.append(timer)
        try:
            completed, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            if timer is not None and timer in completed:
                timer.result()
            if deadline is not None and self.clock.now() >= deadline:
                # Clock boundaries can invalidate transitive completion proof
                # even while an unrelated owned call remains active. Reuse the
                # service's full current-proof walk rather than another index.
                self._verification_due.update(
                    r.id
                    for r in self.store.view().records
                    if isinstance(r, Goal) and r.status == "completed"
                )
        finally:
            for task in waiting:
                task.cancel()
            await asyncio.gather(*waiting, return_exceptions=True)

    def _next_wake(self) -> datetime | None:
        now = self.clock.now()
        times = []
        for record in self.store.view().records:
            if (
                isinstance(record, TaskFrame)
                and record.wake_condition
                and record.wake_condition.kind == "time"
                and record.wake_condition.at
                and record.wake_condition.at > now
            ):
                times.append(record.wake_condition.at)
            if isinstance(record, Belief):
                times.extend(
                    boundary
                    for boundary in (record.valid_from, record.valid_until)
                    if boundary is not None and boundary > now
                )
        return min(times) if times else None

    async def serve(
        self,
        commands: AsyncIterator[AuthenticatedCommand],
        *,
        on_response: Callable[[AuthenticatedCommand, DecisionReceipt], None] | None = None,
        on_idle: Callable[[KernelView], None] | None = None,
    ) -> RunSummary:
        """Serve until EOF/stop; optional synchronous observers run outside commits.

        Responses are the exact submit return, including non-durable backpressure.
        Idle observers receive the settled current view after run_until_idle.
        Observer exceptions propagate after normal service shutdown.
        """
        eof = False

        async def receive() -> None:
            nonlocal eof
            count = 0
            try:
                async for item in commands:
                    receipt = await self.submit(item.command, principal=item.principal)
                    if on_response is not None:
                        on_response(item, receipt)
                    count += 1
                    if count % self.config.scheduler.control_batch_size == 0:
                        # Bound a synchronous producer's control turn. This is
                        # an event-loop yield, never an idle polling timer.
                        await asyncio.sleep(0)
            finally:
                eof = True
                self._changed.set()
                await self.stop()

        reader = asyncio.create_task(receive())
        try:
            while not self._stopped and not eof:
                await self.run_until_idle()
                if on_idle is not None and not self._stopped and not eof:
                    on_idle(self.inspect())
                if eof or self._stopped:
                    break
                self._changed.clear()
                await self._wait_change()
            if reader.done():
                reader.result()
            await self.stop()
            return self.inspect().summary.model_copy(
                update={"stop_reason": "eof" if eof else "stopped"}
            )
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await self.stop()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._shutdown.set()
        self._changed.set()
        await self.runner.close()
        await self.dispatcher.close()
        if self._active:
            await asyncio.gather(*self._active.values())
            self._active.clear()
        if self._verification_jobs:
            await asyncio.gather(*(job.task for job in self._verification_jobs.values()))
            self._verification_jobs.clear()
        self._drain()
        self._commit(recover(self.store, registry=self.registry, now=self.clock.now()))
