"""Prepare atomic authenticated admission; the coordinator commits before ACK.

accept() does not commit. Its response_receipt is a candidate response, not an
acknowledgment: commit the batch, then load the stored receipt by ID. Exact
retries return the original durable receipt without writes. Queue-full responses
are transient backpressure and have neither events nor acknowledgment bindings.
Use on the Store's owning thread, serialized with the coordinator's queue updates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from .authority import Authority, decision, reference, registered
from .errors import ConflictError, ExecutiveError
from .messages import (
    AttachEvidence,
    CancelGoal,
    Command,
    CommitBatch,
    CreateGoal,
    DecideAction,
    InboxAcknowledgment,
    PauseGoal,
    ResolveAction,
    ResolveConflict,
    ResolveUsage,
    ResumeGoal,
    Reverify,
    ReviseGoal,
    SetGrant,
    ShareArtifact,
)
from .records import (
    ActionRecord,
    ArtifactRef,
    Belief,
    Commitment,
    DecisionReceipt,
    Episode,
    ExecutiveSignal,
    Goal,
    JournalEvent,
    PersistentRecord,
    PolicyDecision,
    Principal,
    Ref,
    Scope,
    StateView,
    TaskFrame,
    WakeCondition,
    utc_now,
)
from .reducers import reduce_event
from .store import Store, canonical_json

UsageResolver = Callable[[ResolveUsage, StateView], CommitBatch]


class Intake:
    def __init__(
        self,
        store: Store,
        *,
        authority: Authority | None = None,
        max_signal_bytes: int = 65_536,
        max_queued_ordinary_items: int = 10_000,
        max_queued_control_items: int = 64,
        clock: Callable[[], datetime] = utc_now,
        usage_resolver: UsageResolver | None = None,
    ) -> None:
        for limit in (max_signal_bytes, max_queued_ordinary_items, max_queued_control_items):
            if type(limit) is not int or limit < 1:
                raise ValueError("Admission limits must be positive integers")
        self.store = store
        self.authority = authority or Authority()
        self.max_signal_bytes = max_signal_bytes
        self.max_queued_ordinary_items = max_queued_ordinary_items
        self.max_queued_control_items = max_queued_control_items
        self.clock = clock
        self.usage_resolver = usage_resolver
        self.set_queue_depths(ordinary=0, control=0)

    def set_queue_depths(self, *, ordinary: int, control: int) -> None:
        """Coordinator-owned pending counts, excluding the candidate input.

        Call immediately before accept in the same serialized coordinator turn.
        Capacity is reserved by the coordinator until processing completes; it
        must never discard a previously acknowledged control to admit another.
        """
        if any(type(value) is not int or value < 0 for value in (ordinary, control)):
            raise ValueError("Queue depths must be nonnegative integers")
        self.ordinary_depth = ordinary
        self.control_depth = control

    @staticmethod
    def _identity(item: Command | ExecutiveSignal) -> str:
        # Tags prevent an external string from colliding with a sequence/UUID.
        if item.external_id is not None:
            key: tuple[str, str | int] = ("external", item.external_id)
        elif isinstance(item, ExecutiveSignal) and item.producer_sequence is not None:
            key = ("sequence", item.producer_sequence)
        else:
            key = ("id", item.id)
        encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return "input-v1:" + hashlib.sha256(encoded).hexdigest()

    @classmethod
    def identity_keys(cls, item: Command | ExecutiveSignal) -> tuple[str, str]:
        """Bounded inbox producer/input keys for Store.acknowledgment lookups.

        These are namespaced SHA-256 fingerprints, not raw wire IDs. Original
        producer provenance is retained by admitted events and signal records.
        The principal ID remains the authenticated adapter's registered ID.
        """
        producer = "producer-v1:" + hashlib.sha256(item.producer_id.encode("utf-8")).hexdigest()
        return producer, cls._identity(item)

    @staticmethod
    def _digest(item: Command | ExecutiveSignal, principal: Principal) -> tuple[str, int]:
        value = item.model_dump(mode="json")
        # Generated transport IDs/times are not part of a stable external input.
        for field in ("id", "created_at", "revision", "received_at"):
            value.pop(field, None)
        if isinstance(item, ExecutiveSignal):
            value["principal_id"] = principal.id
        payload = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), len(payload)

    def accept(
        self, item: Command | ExecutiveSignal, *, principal: Principal, state: StateView
    ) -> CommitBatch:
        if state != self.store.view():
            raise ConflictError("stale_intake_state", "Intake requires the current Store snapshot")
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Intake clock must return an aware timestamp")
        current = registered(principal, state)
        if current is None:
            return self._finish(item, principal, state, now, decision("deny", "unknown_principal"))
        try:
            digest, size = self._digest(item, principal)
            producer_key, identity = self.identity_keys(item)
        except UnicodeError:
            return self._finish(
                item, principal, state, now, decision("deny", "invalid_payload_encoding")
            )
        old = self.store.acknowledgment(principal.id, producer_key, identity)
        if old is not None:
            if old.payload_digest == digest:
                receipt = next((r for r in state.records if r.id == old.receipt_id), None)
                if not isinstance(receipt, DecisionReceipt):
                    raise ConflictError("missing_intake_receipt", "Acknowledgment has no receipt")
                return CommitBatch(response_receipt=receipt)
            return self._finish(
                item, principal, state, now, decision("deny", "duplicate_id_conflict")
            )
        if isinstance(item, ExecutiveSignal):
            # Both identity axes are unique in Store even when external ID is primary.
            for previous in state.records:
                if (
                    isinstance(previous, ExecutiveSignal)
                    and previous.principal_id == principal.id
                    and previous.producer_id == item.producer_id
                    and item.producer_sequence is not None
                    and previous.producer_sequence == item.producer_sequence
                ):
                    return self._finish(
                        item, principal, state, now, decision("deny", "duplicate_sequence_conflict")
                    )
        if (
            size > self.max_signal_bytes
            or len(canonical_json(item).encode("utf-8")) > self.max_signal_bytes
        ):
            return self._finish(item, principal, state, now, decision("deny", "payload_too_large"))
        policy = (
            decision("authorize", "authenticated_evidence")
            if isinstance(item, ExecutiveSignal)
            else self.authority.check_command(principal, item, state)
        )
        if policy.disposition != "authorize":
            return self._finish(item, principal, state, now, policy)
        is_control = not isinstance(item, ExecutiveSignal)
        if (is_control and self.control_depth >= self.max_queued_control_items) or (
            not is_control and self.ordinary_depth >= self.max_queued_ordinary_items
        ):
            return CommitBatch(
                response_receipt=DecisionReceipt(
                    created_at=now,
                    subject_id=item.id,
                    disposition="reject",
                    reason_code="control_queue_full" if is_control else "ordinary_queue_full",
                )
            )
        events: tuple[JournalEvent, ...]
        try:
            if isinstance(item, ExecutiveSignal):
                admitted = item.model_copy(
                    update={
                        "id": uuid4().hex,
                        "revision": 1,
                        "principal_id": principal.id,
                        "received_at": now,
                    }
                )
                self._check_evidence(admitted.artifacts, principal, state, admitted.scope.goal_id)
                records = {r.id: r for r in state.records}
                target = records.get(admitted.scope.goal_id or "")
                if admitted.scope.owner_id not in (None, principal.id) and (
                    not isinstance(target, Goal) or admitted.scope.owner_id != target.owner_id
                ):
                    raise ConflictError("signal_scope_mismatch", "Signal scope is invalid")
                events = (self._changed(item, principal, now, (admitted,)),)
            else:
                events, command_policy = self._command(item, principal, state, now)
                if command_policy is not None:
                    policy = command_policy
            # Validate the complete candidate with exactly Store's reducer before
            # assigning acceptance. CAS still checks all pre-batch reads at commit.
            after = state
            for offset, event in enumerate(events, 1):
                after = reduce_event(
                    after, event.model_copy(update={"sequence": state.sequence + offset})
                )
            if isinstance(item, (ReviseGoal, ResolveConflict)):
                from .memory import invalidate_dependents

                before_records = {r.id: r for r in state.records}
                changed_refs = tuple(
                    reference(r)
                    for r in after.records
                    if isinstance(r, (Goal, Belief))
                    and r.id in before_records
                    and r != before_records[r.id]
                )
                retired_criteria = tuple(
                    reference(c)
                    for old in state.records
                    if isinstance(old, Goal) and any(ref.id == old.id for ref in changed_refs)
                    for c in old.criteria
                )
                events = (
                    *events,
                    *invalidate_dependents(after, (*changed_refs, *retired_criteria)),
                )
            return self._finish(
                item, principal, state, now, policy, events=events, digest=digest, identity=identity
            )
        except ExecutiveError as exc:
            disposition = (
                "stale"
                if exc.code.startswith("stale") or exc.code == "revision_conflict"
                else "deny"
            )
            return self._finish(item, principal, state, now, decision(disposition, exc.code))

    def _changed(
        self,
        item: Command | ExecutiveSignal,
        principal: Principal,
        now: datetime,
        records: tuple[PersistentRecord, ...],
    ) -> JournalEvent:
        return JournalEvent(
            created_at=now,
            sequence=1,
            event_type="state.changed",
            producer_id=item.producer_id,
            principal_id=principal.id,
            causal_ids=item.causal_ids,
            subject=Ref(id=item.id),
            payload={"records": [r.model_dump(mode="json") for r in records]},
        )

    def _finish(
        self,
        item: Command | ExecutiveSignal,
        principal: Principal,
        state: StateView,
        now: datetime,
        policy: PolicyDecision,
        *,
        events: tuple[JournalEvent, ...] = (),
        digest: str | None = None,
        identity: str | None = None,
    ) -> CommitBatch:
        # Durable negative decisions deduplicate too. Transient backpressure
        # bypasses _finish; an absent resolver must remain retryable once wired.
        if digest is None and policy.reason_code not in {
            "unknown_principal",
            "duplicate_id_conflict",
            "duplicate_sequence_conflict",
            "unsupported_operation",
            "invalid_payload_encoding",
        }:
            digest, _ = self._digest(item, principal)
            _, identity = self.identity_keys(item)
        event_id = uuid4().hex
        receipt = DecisionReceipt(
            created_at=now,
            subject_id=item.id,
            disposition=("admit" if isinstance(item, ExecutiveSignal) else "accept")
            if policy.disposition == "authorize"
            else ("dispute" if policy.reason_code == "constraint_conflict" else policy.disposition),
            reason_code="input_committed"
            if policy.disposition == "authorize"
            else policy.reason_code,
            state_revisions=policy.evidence,
            event_ids=tuple(e.id for e in events) + (event_id,),
        )
        event = self._changed(item, principal, now, (receipt,)).model_copy(
            update={"id": event_id, "producer_id": "intake", "causal_ids": ()}
        )
        return CommitBatch(
            events=(*events, event),
            expected_revisions=tuple(reference(r) for r in state.records)
            if events
            else tuple(reference(r) for r in state.records if r.id == principal.id),
            inbox_acknowledgments=(
                InboxAcknowledgment(
                    principal_id=principal.id,
                    producer_id=self.identity_keys(item)[0],
                    external_id=identity,
                    payload_digest=digest,
                    receipt_id=receipt.id,
                ),
            )
            if digest is not None and identity is not None
            else (),
            response_receipt=receipt,
        )

    @staticmethod
    def _check_evidence(
        refs: tuple[Ref, ...], principal: Principal, state: StateView, goal_id: str | None
    ) -> None:
        records = {r.id: r for r in state.records}
        for ref in refs:
            current = records.get(ref.id)
            if (
                current is None
                or current.revision != ref.revision
                or (ref.kind != "record" and current.kind != ref.kind)
            ):
                raise ConflictError("stale_evidence", "Evidence must name a current record")
            if isinstance(current, ArtifactRef) and (
                not current.valid
                or current.scope.goal_id != goal_id
                or (
                    principal.id != current.scope.owner_id
                    and principal.id not in current.shared_with
                )
            ):
                raise ConflictError("evidence_access_denied", "Evidence access denied")

    def _command(
        self, command: Command, principal: Principal, state: StateView, now: datetime
    ) -> tuple[tuple[JournalEvent, ...], PolicyDecision | None]:
        records = {r.id: r for r in state.records}
        target_id = getattr(
            command, "goal_id", getattr(command, "action_id", getattr(command, "artifact_id", ""))
        )
        if isinstance(command, SetGrant):
            target_id = command.grant.id
        elif isinstance(command, ResolveUsage):
            target_id = command.reservation_id
        target = records.get(target_id)
        expected = getattr(command, "expected_revision", None)
        if expected is not None and (target is None or target.revision != expected):
            raise ConflictError("stale_command", "Command target revision changed")
        if (
            isinstance(command, ReviseGoal)
            and isinstance(target, Goal)
            and target.status == "completed"
        ):
            # Reopening must precede ordinary correction/conflict transitions.
            # All events still commit atomically against the original CAS reads.
            from .memory import invalidate_dependents

            reopening = invalidate_dependents(state, (reference(target),))
            reopened = state
            for event in reopening:
                reopened = reduce_event(
                    reopened, event.model_copy(update={"sequence": reopened.sequence + 1})
                )
            current_goal = next(r for r in reopened.records if r.id == target.id)
            corrected, policy = self._command(
                command.model_copy(update={"expected_revision": current_goal.revision}),
                principal,
                reopened,
                now,
            )
            return (*reopening, *corrected), policy
        writes: list[PersistentRecord] = []
        if isinstance(command, CreateGoal):
            target = command.goal
            if target.id in records or target.status != "active" or target.revision != 1:
                raise ConflictError(
                    "invalid_goal_create", "Goal creation requires a new active goal"
                )
            commitments = command.commitments or (
                Commitment(
                    created_at=now, goal_id=target.id, responsible_principal_id=principal.id
                ),
            )
            tasks = command.tasks or (
                TaskFrame(
                    created_at=now,
                    goal_id=target.id,
                    commitment_id=commitments[0].id,
                    subgoal=target.outcome,
                ),
            )
            initial_work = (*commitments, *tasks)
            initial_ids = (target.id, *(work.id for work in initial_work))
            if len(set(initial_ids)) != len(initial_ids) or any(
                work.id in records or work.revision != 1 for work in initial_work
            ):
                raise ConflictError(
                    "invalid_initial_work",
                    "Initial work requires fresh distinct IDs and revision 1",
                )
            if any(c.goal_id != target.id for c in commitments) or any(
                t.goal_id != target.id or t.status != "ready" for t in tasks
            ):
                raise ConflictError(
                    "invalid_initial_work", "Initial work must belong to the new goal"
                )
            writes.extend((target, *commitments, *tasks))
        elif isinstance(command, (PauseGoal, ResumeGoal, CancelGoal)):
            if (
                isinstance(command, ResumeGoal)
                and isinstance(target, Goal)
                and self._conflict(target, state)
            ):
                raise ConflictError(
                    "constraint_conflict_unresolved", "Resolve named constraint revisions first"
                )
            return (
                (
                    JournalEvent(
                        created_at=now,
                        sequence=1,
                        event_type={
                            "pause_goal": "goal.paused",
                            "resume_goal": "goal.resumed",
                            "cancel_goal": "goal.cancelled",
                        }[command.kind],
                        principal_id=principal.id,
                        producer_id=command.producer_id,
                        causal_ids=command.causal_ids,
                        subject=Ref(
                            id=command.goal_id, revision=command.expected_revision, kind="goal"
                        ),
                    ),
                ),
                None,
            )
        elif isinstance(command, ReviseGoal):
            assert isinstance(target, Goal)
            if self._conflict(target, state):
                raise ConflictError(
                    "constraint_conflict_unresolved", "Resolve named constraint revisions first"
                )
            changes = {
                field: getattr(command, field)
                for field in ("outcome", "constraints", "criteria", "limits", "worker_id")
                if getattr(command, field) is not None
            }
            if command.criteria is not None:
                # Retain exact removed definitions for historical finding refs.
                remaining = {c.id for c in command.criteria}
                writes.extend(
                    c for c in target.criteria if c.id not in remaining and c.id not in records
                )
            target = target.model_copy(update={"revision": target.revision + 1, **changes})
            writes.append(target)
            self._fence(target, state, writes)
        elif isinstance(command, ResolveConflict):
            for ref in command.conflicting_revisions:
                current = records.get(ref.id) or next(
                    (
                        c
                        for r in state.records
                        if isinstance(r, Goal)
                        for c in r.constraints
                        if c.id == ref.id
                    ),
                    None,
                )
                if (
                    current is None
                    or current.revision != ref.revision
                    or (ref.kind != "record" and current.kind != ref.kind)
                ):
                    raise ConflictError("stale_conflict", "Conflict revisions changed")
            if command.goal_id is not None:
                assert isinstance(target, Goal)
                conflict = self._conflict(target, state)
                named = {r.id for r in command.conflicting_revisions}
                if command.expected_goal_revision != target.revision or conflict != set(
                    command.conflicting_revisions
                ):
                    raise ConflictError(
                        "stale_conflict", "Resolution must name the complete current conflict"
                    )
                replacements = command.resolved_constraints or ()
                if any(c.id not in named for c in replacements) or len(
                    {c.id for c in replacements}
                ) != len(replacements):
                    raise ConflictError(
                        "invalid_conflict_resolution", "Only named constraints may be replaced"
                    )
                for constraint in replacements:
                    old_constraint = next(c for c in target.constraints if c.id == constraint.id)
                    if (
                        constraint.revision != old_constraint.revision + 1
                        or constraint.created_at != old_constraint.created_at
                    ):
                        raise ConflictError(
                            "stale_conflict", "Replacement constraints must advance one revision"
                        )
                # Historical receipt refs survive removing a named constraint.
                writes.extend(
                    c
                    for c in target.constraints
                    if c.id in named
                    and c.id not in {v.id for v in replacements}
                    and c.id not in records
                )
                target = target.model_copy(
                    update={
                        "revision": target.revision + 1,
                        "constraints": tuple(c for c in target.constraints if c.id not in named)
                        + replacements,
                        "status": "active" if target.status == "blocked" else target.status,
                    }
                )
                writes.append(target)
                for task in state.records:
                    if (
                        isinstance(task, TaskFrame)
                        and task.goal_id == target.id
                        and task.status in ("blocked", "suspended")
                        and task.wake_condition is not None
                        and task.wake_condition.reason_code == "constraint_conflict"
                        and set(task.wake_condition.refs) <= conflict
                    ):
                        writes.append(
                            task.model_copy(
                                update={
                                    "revision": task.revision + 1,
                                    "status": "suspended",
                                    "wake_condition": None,
                                }
                            )
                        )
                # Scheduler reevaluates cleared/suspended frames before new work.
            else:
                for identifier in command.belief_ids:
                    belief = records.get(identifier)
                    if not isinstance(belief, Belief) or belief.epistemic_status != "disputed":
                        raise ConflictError(
                            "invalid_conflict", "Only current disputed beliefs can be resolved"
                        )
                    self._check_evidence(command.evidence, principal, state, belief.scope.goal_id)
                    writes.append(
                        belief.model_copy(
                            update={
                                "revision": belief.revision + 1,
                                "epistemic_status": "supported"
                                if belief.id == command.accepted_belief_id
                                else "retracted",
                                "evidence": command.evidence,
                            }
                        )
                    )
        elif isinstance(command, SetGrant):
            old = records.get(command.grant.id)
            if old is None:
                if command.expected_revision is not None or command.grant.revision != 1:
                    raise ConflictError("stale_grant", "New grants must start at revision one")
            elif (
                command.expected_revision != old.revision
                or command.grant.revision != old.revision + 1
                or command.grant.created_at != old.created_at
            ):
                raise ConflictError("stale_grant", "Grant revision changed")
            writes.append(command.grant)
        elif isinstance(command, ShareArtifact):
            assert isinstance(target, ArtifactRef)
            if not target.valid or any(
                not isinstance(records.get(identifier), Principal)
                for identifier in command.principal_ids
            ):
                raise ConflictError(
                    "invalid_artifact_share",
                    "Sharing requires valid artifact and registered principals",
                )
            writes.append(
                target.model_copy(
                    update={
                        "revision": target.revision + 1,
                        "shared_with": tuple(
                            sorted(set(target.shared_with) | set(command.principal_ids))
                        ),
                    }
                )
            )
        elif isinstance(command, DecideAction):
            assert isinstance(target, ActionRecord)
            target_goal = records.get(target.goal_id)
            if (
                command.arguments_digest != target.arguments_digest
                or command.scope != target.scope
                or command.constraint_revision != target.constraint_revision
                or not isinstance(target_goal, Goal)
                or command.constraint_revision != target_goal.revision
            ):
                raise ConflictError(
                    "stale_action_approval", "Approval binding differs from current action"
                )
            if (
                target.status != "awaiting_review"
                or target_goal.status != "active"
                or (target.expires_at is not None and now >= target.expires_at)
            ):
                raise ConflictError("invalid_action_approval", "Action is not currently reviewable")
            writes.append(
                target.model_copy(
                    update={
                        "revision": target.revision + 1,
                        "status": "authorized" if command.decision == "approve" else "denied",
                        "authorization_revision": target.authorization_revision + 1,
                    }
                )
            )
        elif isinstance(command, ResolveAction):
            assert isinstance(target, ActionRecord)
            if target.status not in (
                "dispatching",
                "outcome_unknown",
                "reconciling",
                "needs_human",
            ):
                raise ConflictError(
                    "invalid_action_resolution", "Only unresolved issued actions can be attested"
                )
            self._check_evidence(command.evidence, principal, state, target.goal_id)
            writes.append(
                target.model_copy(
                    update={
                        "revision": target.revision + 1,
                        "status": command.outcome,
                        "evidence": command.evidence,
                        "external_receipt": f"human-attestation:{principal.id}:{command.id}",
                    }
                )
            )
        elif isinstance(command, ResolveUsage):
            if self.usage_resolver is None:
                raise ConflictError(
                    "unsupported_operation", "Usage settlement resolver is not configured"
                )
            result = self.usage_resolver(command, state)
            if (
                result.response_receipt is not None
                or result.inbox_acknowledgments
                or result.outbox
                or result.records
                or result.signals
                or result.receipts
                or not result.events
            ):
                raise ConflictError(
                    "invalid_usage_resolution",
                    "Resolver must return only event-backed settlement with optional CAS",
                )
            for ref in result.expected_revisions:
                current = records.get(ref.id)
                if current is None or current.revision != ref.revision:
                    raise ConflictError("stale_usage_resolution", "Usage resolver read set changed")
            return result.events, None
        elif isinstance(command, (AttachEvidence, Reverify)):
            self._check_evidence(
                command.evidence if isinstance(command, AttachEvidence) else (),
                principal,
                state,
                command.goal_id,
            )
            # Typed owner requests enter the ordinary evidence/verification cycle;
            # acceptance records the request, not a successful verification.
            writes.append(
                ExecutiveSignal(
                    created_at=now,
                    received_at=now,
                    signal_kind=command.kind,
                    principal_id=principal.id,
                    producer_id=command.producer_id,
                    scope=Scope(owner_id=principal.id, goal_id=command.goal_id),
                    artifacts=command.evidence if isinstance(command, AttachEvidence) else (),
                    payload={"command": command.model_dump(mode="json")},
                )
            )
        conflict_policy: PolicyDecision | None = None
        if isinstance(target, Goal) and isinstance(
            command, (CreateGoal, ReviseGoal, ResolveConflict)
        ):
            check = self.authority.check_constraints(target.constraints, {"goal": target})
            if check.reason_code == "constraint_conflict":
                writes = [r for r in writes if r.id != target.id]
                target = target.model_copy(
                    update={"status": "paused" if target.status == "paused" else "blocked"}
                )
                writes.append(target)
                self._fence(target, state, writes, conflict_refs=check.evidence)
                conflict_policy = decision(
                    "deny", "constraint_conflict", evidence=(reference(target), *check.evidence)
                )
        return (self._changed(command, principal, now, tuple(writes)),), conflict_policy

    @staticmethod
    def _conflict(target: Goal, state: StateView) -> set[Ref]:
        current = {reference(c) for c in target.constraints}
        conflicts: set[Ref] = set()
        for record in state.records:
            if (
                isinstance(record, DecisionReceipt)
                and record.reason_code == "constraint_conflict"
                and any(ref.id == target.id for ref in record.state_revisions)
            ):
                named = {ref for ref in record.state_revisions if ref.id != target.id}
                if named and named <= current:
                    conflicts.update(named)
        return conflicts

    @staticmethod
    def _fence(
        target: Goal,
        state: StateView,
        writes: list[PersistentRecord],
        *,
        conflict_refs: tuple[Ref, ...] = (),
    ) -> None:
        before = {r.id: r for r in state.records}
        pending = {r.id: r for r in writes}
        candidates = {**before, **pending}
        for record in candidates.values():
            if (
                isinstance(record, TaskFrame)
                and record.goal_id == target.id
                and record.status not in ("completed", "cancelled", "failed")
            ):
                original = before.get(record.id)
                revised = record.model_copy(
                    update={
                        "revision": original.revision + 1 if original else 1,
                        "fencing_token": original.fencing_token + 1
                        if isinstance(original, TaskFrame)
                        else record.fencing_token,
                        "status": "pending"
                        if record.status == "pending"
                        else ("blocked" if conflict_refs else "suspended"),
                        "wake_condition": WakeCondition(
                            kind="human_decision",
                            refs=conflict_refs,
                            reason_code="constraint_conflict",
                        )
                        if conflict_refs
                        else record.wake_condition,
                    }
                )
                pending[record.id] = revised
            if (
                isinstance(record, Episode)
                and record.status == "running"
                and any(
                    isinstance(t, TaskFrame) and t.id == record.task_id and t.goal_id == target.id
                    for t in candidates.values()
                )
            ):
                pending[record.id] = record.model_copy(
                    update={"revision": record.revision + 1, "status": "fenced"}
                )
        writes[:] = pending.values()
