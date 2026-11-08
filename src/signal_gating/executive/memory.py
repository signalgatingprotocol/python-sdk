"""Owner-configured evidence admission and journalled dependency invalidation.

Validators and conflict detectors are trusted configuration. Text, confidence,
recency and worker-declared status never establish truth or contradiction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from types import MappingProxyType

from .messages import BeliefChange
from .records import (
    ArtifactRef,
    Belief,
    Commitment,
    ContextPacket,
    Episode,
    Goal,
    JournalEvent,
    PersistentRecord,
    PolicyDecision,
    Principal,
    Record,
    Ref,
    StateView,
    TaskFrame,
    VerificationResult,
    WakeCondition,
    utc_now,
)
from .reducers import reduce_event

EvidenceValidator = Callable[[BeliefChange, StateView], bool]
ConflictDetector = Callable[[Belief, Belief, StateView], bool]


def reference(record: Record) -> Ref:
    return Ref(id=record.id, revision=record.revision)


def records_by_id(state: StateView) -> dict[str, Record]:
    records: dict[str, Record] = {r.id: r for r in state.records}
    for record in state.records:
        if isinstance(record, Goal):
            for embedded in (*record.criteria, *record.constraints):
                records.setdefault(embedded.id, embedded)
    return records


class MemoryPolicy:
    def __init__(
        self,
        *,
        evidence_validators: Mapping[str, EvidenceValidator] | None = None,
        conflict_detectors: Mapping[str, ConflictDetector] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        for registry in (evidence_validators, conflict_detectors):
            if any(not callable(value) for value in (registry or {}).values()):
                raise TypeError("Memory policies must be trusted callables")
        self.evidence_validators = MappingProxyType(dict(evidence_validators or {}))
        self.conflict_detectors = MappingProxyType(dict(conflict_detectors or {}))
        self.clock = clock

    def inputs(self, change: BeliefChange, state: StateView) -> StateView:
        """Project callback inputs so every observable revision can be fenced.

        Trusted callbacks receive the owning goal/principal, declared evidence,
        current scoped beliefs and their evidence, never unrelated global state.
        """
        records = records_by_id(state)
        goal = records.get(change.belief.scope.goal_id or "")
        identifiers = {
            change.belief.id,
            *(ref.id for ref in (*change.belief.evidence, *change.evidence)),
        }
        if isinstance(goal, Goal):
            identifiers.update((goal.id, goal.owner_id))
            for item in state.records:
                if isinstance(item, Belief) and item.scope.goal_id == goal.id:
                    identifiers.add(item.id)
                    identifiers.update(ref.id for ref in item.evidence)
        return StateView(records=tuple(r for r in state.records if r.id in identifiers))

    def evaluate(self, change: BeliefChange, state: StateView) -> PolicyDecision:
        belief = change.belief
        records = records_by_id(state)
        goal = records.get(belief.scope.goal_id or "")
        owner_id = goal.owner_id if isinstance(goal, Goal) else belief.scope.owner_id
        evidence = tuple(dict.fromkeys((*belief.evidence, *change.evidence)))

        def decision(
            disposition: str, reason: str, refs: tuple[Ref, ...] = evidence
        ) -> PolicyDecision:
            return PolicyDecision.model_validate(
                dict(gate="memory", disposition=disposition, reason_code=reason, evidence=refs)
            )

        if not isinstance(goal, Goal) or belief.scope.owner_id not in (None, owner_id):
            return decision("reject", "belief_scope_mismatch", ())
        if not evidence:
            return decision("request_evidence", "missing_evidence", ())
        now = self.clock()
        if (belief.valid_from and now < belief.valid_from) or (
            belief.valid_until and now >= belief.valid_until
        ):
            return decision("request_evidence", "invalid_evidence")
        for ref in evidence:
            item = records.get(ref.id)
            if (
                item is None
                or item.revision != ref.revision
                or (ref.kind != "record" and ref.kind != getattr(item, "kind", ""))
            ):
                return decision("request_evidence", "unknown_evidence", ())
            if isinstance(item, ArtifactRef) and (
                not item.valid
                or item.scope.goal_id != belief.scope.goal_id
                or (item.scope.owner_id != owner_id and owner_id not in item.shared_with)
            ):
                return decision("request_evidence", "invalid_evidence")
            if isinstance(item, Belief) and (
                item.epistemic_status != "supported"
                or (item.valid_from and now < item.valid_from)
                or (item.valid_until and now >= item.valid_until)
            ):
                return decision("request_evidence", "invalid_evidence")
        policy_state = self.inputs(change, state)
        validator = self.evidence_validators.get(owner_id or "")
        try:
            supported = validator is not None and validator(change, policy_state) is True
        except Exception:
            supported = False
        if not supported:
            return decision("request_evidence", "unvalidated_evidence")
        try:
            conflicts = self.conflicts(belief, policy_state)
        except Exception:
            return decision("request_evidence", "conflict_detector_failed")
        if conflicts:
            return decision("dispute", "conflicting_support", conflicts)
        return decision("revise" if belief.id in records else "admit", "evidence_validated")

    def conflicts(self, belief: Belief, state: StateView) -> tuple[Ref, ...]:
        """Return only trusted conflict findings, not supporting evidence refs."""
        goal = records_by_id(state).get(belief.scope.goal_id or "")
        detector = self.conflict_detectors.get(goal.owner_id) if isinstance(goal, Goal) else None
        if detector is None:
            return ()
        return tuple(
            reference(old)
            for old in state.records
            if isinstance(old, Belief)
            and old.id != belief.id
            and old.scope.goal_id == belief.scope.goal_id
            and old.epistemic_status in {"supported", "disputed"}
            and detector(belief, old, state) is True
        )


def invalidate_dependents(state: StateView, changed: tuple[Ref, ...]) -> tuple[JournalEvent, ...]:
    """Prepare a transitive cascade; callers union all pre-batch CAS reads.

    changed identifies materially revised/revoked inputs, not every journal write.
    This function never revokes the inputs themselves or runs external validators.
    Repeated invalidation without a new change has no further state effect.
    """
    if not changed:
        return ()
    records = records_by_id(state)
    affected = {ref.id for ref in changed}
    changed_goals = {
        identifier for identifier in affected if isinstance(records.get(identifier), Goal)
    }
    # A goal correction changes the meaning of its criteria, even if IDs survive.
    for identifier in changed_goals:
        goal = records[identifier]
        assert isinstance(goal, Goal)
        affected.update(c.id for c in goal.criteria)
    while True:
        previous = set(affected)
        for item in state.records:
            if isinstance(item, Goal):
                for criterion in item.criteria:
                    if any(
                        ref.id in affected
                        for ref in (*criterion.inputs, *criterion.required_evidence)
                    ):
                        affected.add(criterion.id)
            dependencies: tuple[Ref, ...] = ()
            if isinstance(item, Belief):
                dependencies = item.evidence
            elif isinstance(item, VerificationResult):
                dependencies = (*item.read_set, *item.evidence, Ref(id=item.criterion_id))
            elif isinstance(item, (TaskFrame, Commitment)):
                dependencies = item.dependencies
                if item.goal_id in changed_goals:
                    affected.add(item.id)
                if isinstance(item, TaskFrame):
                    dependencies += item.progress_artifacts
            elif isinstance(item, ContextPacket):
                # Historical packets remain untouched, but their live episodes fence.
                dependencies = (
                    *item.mandatory_read_set,
                    reference(item.goal),
                    reference(item.task),
                    *(reference(b) for b in item.beliefs),
                )
            elif isinstance(item, Episode):
                dependencies = (Ref(id=item.task_id), Ref(id=item.context_packet_id))
            if any(ref.id in affected for ref in dependencies):
                affected.add(item.id)
        for item in state.records:
            if isinstance(item, Goal) and (
                any(c.id in affected for c in item.criteria)
                or any(
                    isinstance(r, VerificationResult)
                    and r.id in affected
                    and r.criterion_id in {c.id for c in item.criteria}
                    for r in state.records
                )
                or any(
                    isinstance(r, Commitment)
                    and r.goal_id == item.id
                    and r.id in affected
                    and (r.required or r.id in item.required_commitment_ids)
                    for r in state.records
                )
            ):
                affected.add(item.id)
        if affected == previous:
            break
    writes: list[PersistentRecord] = []
    completed: list[str] = []
    for item in state.records:
        if item.id not in affected:
            continue
        updates: dict[str, object] = {}
        if (
            isinstance(item, Belief)
            and item.id not in {r.id for r in changed}
            and item.epistemic_status == "supported"
        ):
            updates = {"epistemic_status": "retracted"}
        elif isinstance(item, VerificationResult) and item.disposition == "accept":
            updates = {
                "disposition": "insufficient_evidence",
                "reason_code": "dependency_invalidated",
                "gaps": (),
            }
        elif isinstance(item, Episode) and item.status == "running":
            updates = {"status": "fenced"}
        elif isinstance(item, (Goal, TaskFrame, Commitment)) and item.status == "completed":
            completed.append(item.id)
        elif isinstance(item, TaskFrame) and item.status in {"ready", "running", "verifying"}:
            updates = {
                "status": "suspended",
                "fencing_token": item.fencing_token + 1,
                "wake_condition": item.wake_condition
                or WakeCondition(
                    kind="dependency_change", refs=changed, reason_code="dependency_invalidated"
                ),
            }
        if updates:
            writes.append(item.model_copy(update={"revision": item.revision + 1, **updates}))
    if not writes and not completed:
        return ()
    principal = next(
        (r for r in state.records if isinstance(r, Principal) and "owner" in r.roles), None
    )
    if principal is None:
        raise ValueError("Invalidation requires a registered owner")
    events = []
    if writes:
        event = JournalEvent(
            sequence=state.sequence + 1,
            event_type="state.changed",
            producer_id="memory",
            principal_id=principal.id,
            subject=changed[0],
            payload={"records": [r.model_dump(mode="json") for r in writes]},
        )
        events.append(event)
        state = reduce_event(state, event)
    if completed:
        events.append(
            JournalEvent(
                sequence=state.sequence + 1,
                event_type="completion.invalidated",
                producer_id="memory",
                principal_id=principal.id,
                subject=changed[0],
                payload={
                    "record_ids": [identifier for identifier in sorted(completed)],
                    "reason": "dependency_invalidated",
                },
            )
        )
    return tuple(events)
