"""Deterministic attention and journal-backed scheduling, without dispatch.

plan/select/select_actions are projections of the SAME ordered selection. Consume
plan once, then atomically commit prepare_selection with any episode/reservation
writes. Never call each projection as a separate scheduling turn. Preparing does
not reserve slots, grant authority, enqueue an effect, or invoke a worker.

All preparation uses complete pre-batch CAS reads. The serialized coordinator
must commit before consuming a selection or acknowledging a decision. wake has
the required event-only interface; callers union its pre-batch reads themselves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Literal, TypeVar

from .authority import reference
from .config import KernelConfig
from .memory import records_by_id
from .messages import CommitBatch
from .records import (
    ActionRecord,
    Belief,
    Budget,
    DecisionReceipt,
    Episode,
    ExecutiveSignal,
    Goal,
    Identifier,
    JournalEvent,
    Model,
    PersistentRecord,
    PolicyDecision,
    Principal,
    Record,
    Ref,
    SchedulerState,
    StateView,
    TaskFrame,
    WakeCondition,
    utc_now,
)
from .reducers import _eligible, reduce_event

T = TypeVar("T")


class ScheduledItem(Model):
    id: Identifier
    kind: Literal["task", "action"]
    work_class: Literal["reconcile", "urgent", "ordinary"]


def _now(value: datetime) -> datetime:
    # Validate timestamps even on empty/idle views.
    validated = WakeCondition(kind="time", at=value, reason_code="clock").at
    assert validated is not None
    return validated


def _reads(state: StateView) -> tuple[Ref, ...]:
    return tuple(reference(record) for record in state.records)


def _changed(
    state: StateView, writes: tuple[PersistentRecord, ...], now: datetime, principal_id: str
) -> JournalEvent:
    event = JournalEvent(
        sequence=state.sequence + 1,
        created_at=now,
        event_type="state.changed",
        producer_id="scheduler",
        principal_id=principal_id,
        subject=reference(writes[0]),
        payload={"records": [record.model_dump(mode="json") for record in writes]},
    )
    reduce_event(state, event)
    return event


def _batch(
    state: StateView,
    writes: tuple[PersistentRecord, ...],
    now: datetime,
    principal_id: str,
    receipt: DecisionReceipt | None = None,
) -> CommitBatch:
    return CommitBatch(
        events=(_changed(state, writes, now, principal_id),) if writes else (),
        expected_revisions=_reads(state),
        response_receipt=receipt,
    )


def _scheduler_state(state: StateView, now: datetime) -> SchedulerState:
    return next(
        (r for r in state.records if isinstance(r, SchedulerState)),
        SchedulerState(created_at=now),
    )


def _revise_scheduler(
    state: StateView, current: SchedulerState, **updates: object
) -> SchedulerState:
    return current.model_copy(
        update={
            "revision": current.revision + int(any(r.id == current.id for r in state.records)),
            **updates,
        }
    )


def _owner(state: StateView) -> str:
    principal = next(
        (r for r in state.records if isinstance(r, Principal) and "owner" in r.roles), None
    )
    if principal is None:
        raise ValueError("Scheduling preparation requires a registered owner")
    return principal.id


class AttentionPolicy:
    """Only deterministic routing. No model relevance judgment or assessment API.

    Unscoped relevance remains a human-decision deferral. A future optional model
    assessment must supply its own bounded task, budget and deadline, never a
    recursive callback through this policy. This class never calls a model.
    """

    def __init__(
        self, config: KernelConfig | None = None, *, clock: Callable[[], datetime] = utc_now
    ) -> None:
        self.config = config or KernelConfig()
        self.clock = clock

    @staticmethod
    def _receipts(signal: ExecutiveSignal, state: StateView) -> tuple[DecisionReceipt, ...]:
        return tuple(
            r
            for r in state.records
            if isinstance(r, DecisionReceipt)
            and r.subject_id == signal.id
            and r.reason_code.startswith("attention_")
        )

    def decide(self, signal: ExecutiveSignal, state: StateView) -> PolicyDecision:
        _now(self.clock())
        records = records_by_id(state)
        goal = records.get(signal.scope.goal_id or "")
        evidence = (reference(goal),) if isinstance(goal, Goal) else ()
        args = {
            "gate": "attention",
            "policy_version": self.config.scheduler.policy_version,
            "evidence": evidence,
        }

        # Equivalent content from different producers is coalesced only within
        # the same authenticated principal/scope and still-current goal revision.
        def equivalent(other: ExecutiveSignal) -> bool:
            return (
                other.signal_kind,
                other.principal_id,
                other.scope,
                other.payload,
                other.artifacts,
            ) == (
                signal.signal_kind,
                signal.principal_id,
                signal.scope,
                signal.payload,
                signal.artifacts,
            )

        for other in state.records:
            if (
                not isinstance(other, ExecutiveSignal)
                or other.id == signal.id
                or not equivalent(other)
            ):
                continue
            if any(
                receipt.disposition in {"process", "coalesce"}
                and receipt.state_revisions == evidence
                for receipt in self._receipts(other, state)
            ):
                return PolicyDecision.model_validate(
                    {
                        **args,
                        "disposition": "coalesce",
                        "reason_code": "attention_coalesced",
                        "evidence": (*evidence, reference(other)),
                    }
                )
        if signal.signal_kind == "telemetry":
            return PolicyDecision.model_validate(
                {**args, "disposition": "archive", "reason_code": "attention_archive_telemetry"}
            )
        principal = records.get(signal.principal_id)
        if (
            isinstance(goal, Goal)
            and isinstance(principal, Principal)
            and signal.scope.owner_id in (None, goal.owner_id)
        ):
            # These are owner REQUESTS. Processing may invoke downstream evidence
            # validation/verification, never acceptance or owner-goal resumption.
            if (
                signal.signal_kind in {"attach_evidence", "reverify"}
                and principal.id == goal.owner_id
                and "owner" in principal.roles
            ):
                return PolicyDecision.model_validate(
                    {**args, "disposition": "process", "reason_code": "attention_owner_request"}
                )
            if goal.status == "active":
                return PolicyDecision.model_validate(
                    {**args, "disposition": "process", "reason_code": "attention_relevant"}
                )
            if goal.status in {"completed", "cancelled", "failed"}:
                return PolicyDecision.model_validate(
                    {**args, "disposition": "archive", "reason_code": "attention_terminal_goal"}
                )
            return PolicyDecision.model_validate(
                {
                    **args,
                    "disposition": "defer",
                    "reason_code": "attention_owner_control",
                    "wake_condition": WakeCondition(
                        kind="human_decision", refs=evidence, reason_code="owner_control"
                    ),
                }
            )
        return PolicyDecision.model_validate(
            {
                **args,
                "disposition": "defer",
                "reason_code": "attention_unscoped",
                "wake_condition": WakeCondition(
                    kind="human_decision", reason_code="scope_required"
                ),
            }
        )

    def prepare(self, signal: ExecutiveSignal, state: StateView) -> CommitBatch:
        """Record one decision for an already authenticated, committed signal."""
        if signal not in state.records:
            raise ValueError("Attention requires a current admitted signal")
        existing = self._receipts(signal, state)
        if existing:
            return CommitBatch(response_receipt=existing[-1])
        now = _now(self.clock())
        decision = self.decide(signal, state)
        receipt = DecisionReceipt(
            created_at=now,
            subject_id=signal.id,
            state_revisions=tuple(
                ref for ref in decision.evidence if ref.id == signal.scope.goal_id
            ),
            policy_version=decision.policy_version,
            disposition=decision.disposition,
            reason_code=decision.reason_code,
            evidence=decision.evidence,
            wake_condition=decision.wake_condition,
        )
        return _batch(state, (receipt,), now, signal.principal_id, receipt)


class Scheduler:
    def __init__(
        self, config: KernelConfig | None = None, *, clock: Callable[[], datetime] = utc_now
    ) -> None:
        self.config = config or KernelConfig()
        self.clock = clock

    @property
    def _cycle(self) -> tuple[str, ...]:
        config = self.config.scheduler
        return (
            ("reconcile",) * config.reconciliation_quota
            + ("urgent",) * config.urgent_quota
            + ("ordinary",) * config.ordinary_quota
        )

    def _plan(self, state: StateView, now: datetime) -> tuple[tuple[ScheduledItem, ...], int]:
        now = _now(now)
        current = _scheduler_state(state, now)
        if current.control_pending:
            return (), current.cursor
        records = {r.id: r for r in state.records}
        indexed = records_by_id(state)
        running = {
            r.task_id for r in state.records if isinstance(r, Episode) and r.status == "running"
        }
        running.update(
            r.id for r in state.records if isinstance(r, TaskFrame) and r.status == "running"
        )
        capacity = max(0, self.config.max_active_episodes - len(running))
        groups: dict[str, list[ScheduledItem]] = {key: [] for key in set(self._cycle)}
        tasks = []
        actions = []
        for item in state.records:
            if isinstance(item, ActionRecord) and item.status == "outcome_unknown":
                # Reconciliation of issued effects is distinct from new work and
                # remains available under owner pause/cancellation. Already
                # reconciling/needs-human effects cannot spin or retry here.
                actions.append(item)
            if not isinstance(item, TaskFrame) or item.status != "ready" or item.id in running:
                continue
            goal = records.get(item.goal_id)
            if (
                not isinstance(goal, Goal)
                or goal.status != "active"
                or not _eligible(item, records)
                or not _fresh(item, indexed, now)
            ):
                continue
            tasks.append((item, goal))
        for item in sorted(actions, key=lambda a: (a.created_at, a.id)):
            groups["reconcile"].append(
                ScheduledItem(id=item.id, kind="action", work_class="reconcile")
            )

        def rank(pair: tuple[TaskFrame, Goal]) -> tuple[int, datetime, int, datetime, str]:
            task, goal = pair
            since = task.eligible_since or task.created_at
            age = min(
                self.config.scheduler.max_aging_points,
                max(
                    0,
                    int(
                        (now - since).total_seconds()
                        // self.config.scheduler.aging_interval_seconds
                    ),
                ),
            )
            deadlines = [d for d in (task.deadline, goal.deadline) if d is not None]
            deadline = min(deadlines) if deadlines else datetime.max.replace(tzinfo=now.tzinfo)
            return (-(goal.priority + age), deadline, -task.priority, since, task.id)

        for task, goal in sorted(tasks, key=rank):
            deadlines = [d for d in (task.deadline, goal.deadline) if d is not None]
            urgent = bool(
                deadlines
                and min(deadlines)
                <= now + timedelta(seconds=self.config.scheduler.urgent_deadline_seconds)
            )
            key: Literal["urgent", "ordinary"] = "urgent" if urgent else "ordinary"
            groups[key].append(ScheduledItem(id=task.id, kind="task", work_class=key))
        selected: list[ScheduledItem] = []
        cursor = current.cursor % len(self._cycle)
        empty = 0
        while len(selected) < capacity and empty < len(self._cycle):
            selected_class = self._cycle[cursor]
            cursor = (cursor + 1) % len(self._cycle)
            if groups[selected_class]:
                selected.append(groups[selected_class].pop(0))
                empty = 0
            else:
                empty += 1
        return tuple(selected), cursor

    def plan(self, state: StateView, now: datetime) -> tuple[ScheduledItem, ...]:
        return self._plan(state, now)[0]

    def select(self, state: StateView, now: datetime) -> tuple[str, ...]:
        return tuple(item.id for item in self.plan(state, now) if item.kind == "task")

    def select_actions(self, state: StateView, now: datetime) -> tuple[str, ...]:
        return tuple(item.id for item in self.plan(state, now) if item.kind == "action")

    def prepare_selection(self, state: StateView, now: datetime) -> CommitBatch:
        now = _now(now)
        selected, cursor = self._plan(state, now)
        if not selected:
            return CommitBatch()
        current = _scheduler_state(state, now)
        writes: list[PersistentRecord] = [_revise_scheduler(state, current, cursor=cursor)]
        records = records_by_id(state)
        for item in selected:
            record = records[item.id]
            if isinstance(record, TaskFrame):
                writes.append(
                    record.model_copy(
                        update={
                            "revision": record.revision + 1,
                            "eligible_since": now,
                            "last_scheduled_at": now,
                        }
                    )
                )
            writes.append(
                DecisionReceipt(
                    created_at=now,
                    subject_id=item.id,
                    policy_version=self.config.scheduler.policy_version,
                    disposition="process",
                    reason_code="scheduler_" + item.work_class,
                    state_revisions=(reference(record),),
                )
            )
        return _batch(state, tuple(writes), now, _owner(state))

    def select_controls(self, controls: tuple[T, ...]) -> tuple[T, ...]:
        """Bound a coordinator-owned queue turn; retain its unconsumed suffix."""
        return controls[: self.config.scheduler.control_batch_size]

    def record_control_backlog(self, state: StateView, pending: int, now: datetime) -> CommitBatch:
        now = _now(now)
        if type(pending) is not int or pending < 0:
            raise ValueError("Control backlog must be a nonnegative integer")
        current = _scheduler_state(state, now)
        since = (current.control_since or now) if pending else None
        overload = bool(
            since is not None
            and (now - since).total_seconds() >= self.config.scheduler.control_overload_seconds
        )
        reported = (current.overload_reported or overload) if pending else False
        if (pending, since, reported) == (
            current.control_pending,
            current.control_since,
            current.overload_reported,
        ):
            return CommitBatch()
        updated = _revise_scheduler(
            state, current, control_pending=pending, control_since=since, overload_reported=reported
        )
        writes: list[PersistentRecord] = [updated]
        if overload and not current.overload_reported:
            writes.append(
                DecisionReceipt(
                    created_at=now,
                    subject_id=current.id,
                    policy_version=self.config.scheduler.policy_version,
                    disposition="needs_human",
                    reason_code="control_overload",
                )
            )
        return _batch(state, tuple(writes), now, _owner(state))

    def record_failure(
        self, state: StateView, task_id: str, reason_code: str, now: datetime
    ) -> CommitBatch:
        """Bound equivalent worker failures. Never retries an external action.

        Material definition/dependency changes reset the count upon a NEW failure;
        descriptive owner status/revision, fences and scheduler bookkeeping do not.
        The coordinator calls this once per observed failure, not once per poll.
        """
        now = _now(now)
        records = records_by_id(state)
        task = records.get(task_id)
        if not isinstance(task, TaskFrame) or task.status not in {
            "ready",
            "running",
            "suspended",
            "blocked",
            "verifying",
        }:
            raise ValueError("Failures require a nonterminal task")
        goal = records.get(task.goal_id)
        if not isinstance(goal, Goal):
            raise ValueError("Failures require the current owning goal")
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("Failures require a nonempty reason code")
        signature = _failure_signature(task, goal, records, reason_code)
        count = task.failure_count + 1 if task.failure_signature == signature else 1
        blocked = count >= self.config.identical_failure_limit
        condition = WakeCondition(
            kind="human_decision" if blocked else "time",
            at=None
            if blocked
            else now + timedelta(seconds=self.config.scheduler.aging_interval_seconds),
            reason_code="repeated_failure" if blocked else "retry_backoff",
        )
        updated = task.model_copy(
            update={
                "revision": task.revision + 1,
                "status": "blocked" if blocked else "suspended",
                "wake_condition": condition,
                "failure_count": count,
                "failure_signature": signature,
                "failure_reason": reason_code,
                "fencing_token": task.fencing_token + int(task.status == "running"),
            }
        )
        writes: list[PersistentRecord] = [updated]
        writes.extend(
            e.model_copy(update={"revision": e.revision + 1, "status": "fenced"})
            for e in state.records
            if isinstance(e, Episode) and e.task_id == task.id and e.status == "running"
        )
        receipt = DecisionReceipt(
            created_at=now,
            subject_id=task.id,
            policy_version=self.config.scheduler.policy_version,
            disposition="defer",
            reason_code="scheduler_repeated_failure" if blocked else "scheduler_retry_backoff",
            wake_condition=condition,
            state_revisions=(reference(task), reference(goal)),
        )
        writes.append(receipt)
        return _batch(state, tuple(writes), now, goal.owner_id, receipt)


def _failure_signature(
    task: TaskFrame,
    goal: Goal,
    records: Mapping[str, Record],
    reason_code: str,
) -> str:
    def material(record: Record) -> object:
        exclude = {"revision", "created_at"}
        if isinstance(record, Goal):
            exclude.update(("status", "priority", "deadline"))
        return record.model_dump(mode="json", exclude=exclude)

    relevant = {ref.id for ref in (*task.dependencies, *task.progress_artifacts)} | set(
        task.pending_action_ids
    )
    content = {
        "reason_code": reason_code,
        "goal": material(goal),
        "subgoal": task.subgoal,
        "dependencies": [(ref.id, ref.kind) for ref in task.dependencies],
        "inputs": [material(records[i]) if i in records else {"id": i} for i in sorted(relevant)],
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _event_records(state: StateView, event: JournalEvent | None) -> tuple[PersistentRecord, ...]:
    """Use only matching authoritative post-images from a committed batch event."""
    if event is None or event.sequence > state.sequence or event.event_type != "state.changed":
        return ()
    raw = event.model_dump(mode="json")["payload"].get("records")
    if not isinstance(raw, (tuple, list)):
        return ()
    records = {r.id: r for r in state.records}
    return tuple(
        records[item["id"]]
        for item in raw
        if isinstance(item, Mapping)
        and item.get("id") in records
        and records[item["id"]].model_dump(mode="json") == item
    )


def _fresh(task: TaskFrame, indexed: Mapping[str, Record], now: datetime) -> bool:
    """Temporal checks supplement the reducer's clock-free lifecycle checks."""
    pending = list(task.dependencies)
    seen: set[str] = set()
    while pending:
        ref = pending.pop()
        if ref.id in seen:
            continue
        seen.add(ref.id)
        item = indexed.get(ref.id)
        if isinstance(item, Belief):
            if (item.valid_from is not None and now < item.valid_from) or (
                item.valid_until is not None and now >= item.valid_until
            ):
                return False
            pending.extend(item.evidence)
    return True


def _woke(
    condition: WakeCondition,
    goal: Goal,
    state: StateView,
    event: JournalEvent | None,
    changed: tuple[PersistentRecord, ...],
    indexed: Mapping[str, Record],
    now: datetime,
) -> bool:
    scoped_signals = [
        r for r in changed if isinstance(r, ExecutiveSignal) and r.scope.goal_id == goal.id
    ]
    if condition.kind == "time":
        return condition.at is not None and now >= condition.at
    if condition.kind == "dependency_change":
        return condition.reason_code == "dependency_invalidated" or any(
            ref.id in indexed and indexed[ref.id].revision != ref.revision for ref in condition.refs
        )
    if condition.kind == "new_evidence":
        return any(
            r.signal_kind in {"attach_evidence", "evidence"}
            and (
                not condition.refs
                or any(ref.id in {a.id for a in r.artifacts} for ref in condition.refs)
            )
            for r in scoped_signals
        )
    if condition.kind == "human_decision":
        # Reverification asks for verification. It cannot override an explicit
        # conflict or grant another identical-failure retry. Those require their
        # dedicated owner resolution, which must remove the named blocker.
        if condition.reason_code in {"repeated_failure", "constraint_conflict"}:
            return False
        if event is None or event.sequence > state.sequence or event.principal_id != goal.owner_id:
            return False
        return any(
            r.principal_id == goal.owner_id and r.signal_kind == "reverify" for r in scoped_signals
        ) or (
            event.event_type in {"goal.resumed", "goal.cancelled"}
            and event.subject.id == goal.id
            and event.subject.revision + 1 == goal.revision
        )
    if condition.kind == "budget":
        # An accounting change wakes reevaluation, not budget admission. Actual
        # maxima/holds/ancestor accounting remain BudgetPolicy's responsibility.
        return any(
            isinstance(r, Budget)
            and r.scope.goal_id == goal.id
            and not r.accounting_uncertain
            and (not condition.refs or r.id in {ref.id for ref in condition.refs})
            for r in changed
        )
    return False


def wake(state: StateView, event: JournalEvent | None, now: datetime) -> tuple[JournalEvent, ...]:
    """Reevaluate frames and deferred attention without changing owner controls.

    Pass each relevant committed event, including events before a batch's final
    receipt. Only matching current post-images are considered. Dependency/time
    wakes also work after restart without an event. Requests wake evaluation,
    never verification success. Callers union complete pre-batch CAS reads.
    """
    now = _now(now)
    records = {r.id: r for r in state.records}
    indexed = records_by_id(state)
    changed = _event_records(state, event)
    writes: list[PersistentRecord] = []
    running = {r.task_id for r in state.records if isinstance(r, Episode) and r.status == "running"}
    for task in state.records:
        if (
            not isinstance(task, TaskFrame)
            or task.status not in {"pending", "suspended", "blocked"}
            or task.id in running
        ):
            continue
        goal = records.get(task.goal_id)
        if not isinstance(goal, Goal) or goal.status != "active":
            continue
        condition = task.wake_condition
        satisfied = condition is None and task.status in {"pending", "suspended"}
        if condition is not None:
            satisfied = _woke(condition, goal, state, event, changed, indexed, now)
            if condition.reason_code == "repeated_failure" and task.failure_reason is not None:
                satisfied = task.failure_signature != _failure_signature(
                    task, goal, indexed, task.failure_reason
                )
        if not satisfied:
            continue
        # Refresh revisions only after the wake. This never establishes support:
        # all current targets must still satisfy the reducer's eligibility check.
        dependencies = tuple(
            ref.model_copy(update={"revision": indexed[ref.id].revision})
            if ref.id in indexed
            else ref
            for ref in task.dependencies
        )
        candidate = task.model_copy(
            update={
                "revision": task.revision + 1,
                "status": "ready",
                "wake_condition": None,
                "dependencies": dependencies,
                "eligible_since": now,
            }
        )
        if _eligible(candidate, records) and _fresh(candidate, indexed, now):
            writes.append(candidate)
    policy = AttentionPolicy(clock=lambda: now)
    for receipt in state.records:
        if (
            not isinstance(receipt, DecisionReceipt)
            or receipt.disposition != "defer"
            or not receipt.reason_code.startswith("attention_")
            or receipt.wake_condition is None
        ):
            continue
        signal = records.get(receipt.subject_id)
        if not isinstance(signal, ExecutiveSignal):
            continue
        goal = records.get(signal.scope.goal_id or "")
        if not isinstance(goal, Goal) or not _woke(
            receipt.wake_condition, goal, state, event, changed, indexed, now
        ):
            continue
        decision = policy.decide(signal, state)
        if decision.disposition == "defer":
            continue
        writes.append(
            receipt.model_copy(
                update={
                    "revision": receipt.revision + 1,
                    "disposition": decision.disposition,
                    "reason_code": decision.reason_code,
                    "policy_version": decision.policy_version,
                    "evidence": decision.evidence,
                    "state_revisions": (reference(goal),),
                    "wake_condition": decision.wake_condition,
                }
            )
        )
    return (_changed(state, tuple(writes), now, _owner(state)),) if writes else ()
