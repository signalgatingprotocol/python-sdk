"""Pure executive replay. The journal is the authoritative state transition log.

Wire v1: state.changed carries {records: [complete persistent post-images]};
controls goal.paused/resumed/cancelled carry {}; completion.invalidated carries
{record_ids: [IDs], reason: str}; audit.recorded carries {reason_code: str,
details: object}; artifact_invalidated carries {} and targets a current valid
artifact. Unknown formats fail closed. IDs/times are supplied, never made
here. Atomic creates are validated against the whole resulting event bundle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn, TypeVar

from pydantic import TypeAdapter, ValidationError

from .errors import InvalidTransition
from .records import (
    ActionRecord,
    ArtifactRef,
    Belief,
    Budget,
    Commitment,
    Constraint,
    ContextPacket,
    Criterion,
    Episode,
    Goal,
    Grant,
    JournalEvent,
    Model,
    PersistentRecord,
    Record,
    Ref,
    Reservation,
    Scope,
    StateView,
    TaskFrame,
    VerificationResult,
    WakeCondition,
)

R = TypeVar("R", bound=Record)

_ADAPTER: TypeAdapter[PersistentRecord] = TypeAdapter(PersistentRecord)
_TASK_EDGES = {
    "pending": {"ready", "cancelled"},
    "ready": {"running", "suspended", "blocked", "cancelled"},
    "running": {"verifying", "suspended", "blocked", "failed", "cancelled"},
    "verifying": {"completed", "ready", "blocked", "suspended", "failed", "cancelled"},
    "suspended": {"ready", "blocked", "cancelled"},
    "blocked": {"ready", "suspended", "cancelled"},
}
_GOAL_EDGES = {
    "active": {"paused", "blocked", "completed", "cancelled", "failed"},
    "paused": {"active", "cancelled"},
    "blocked": {"active", "paused", "completed", "cancelled", "failed"},
}
_ACTION_EDGES = {
    "proposed": {"denied", "awaiting_review", "authorized", "cancelled"},
    "awaiting_review": {"denied", "authorized", "cancelled"},
    "authorized": {"dispatching", "cancelled"},
    "dispatching": {"succeeded", "failed", "outcome_unknown"},
    "outcome_unknown": {"reconciling", "succeeded", "failed"},
    "reconciling": {"succeeded", "failed", "needs_human"},
    "needs_human": {"succeeded", "failed"},
}
_OTHER_EDGES = {
    "episode": {"running": {"accepted", "rejected", "expired", "fenced"}},
    "reservation": {"reserved": {"settled", "uncertain", "released"}, "uncertain": {"settled"}},
}
_TASK_TERMINAL = {"completed", "failed", "cancelled"}
_UNSETTLED_ACTIONS = {"dispatching", "outcome_unknown", "reconciling", "needs_human"}


def _fail(code: str, message: str, record_id: str | None = None) -> NoReturn:
    raise InvalidTransition(code, message, record_id=record_id)


def validate_transition(record: Record, target: str) -> None:
    """Check a lifecycle edge. Terminal completion reopens only via invalidation.

    Contextual preconditions (dependencies, goal controls, evidence) are checked
    by reduce_event against the whole resulting state, not by this edge check.
    """
    if isinstance(record, Goal):
        edges = _GOAL_EDGES
    elif isinstance(record, (TaskFrame, Commitment)):
        edges = _TASK_EDGES
    elif isinstance(record, ActionRecord):
        edges = _ACTION_EDGES
    else:
        edges = _OTHER_EDGES.get(getattr(record, "kind", ""), {})
    source = getattr(record, "status", None)
    if source is None or (source != target and target not in edges.get(source, set())):
        _fail(
            "invalid_transition",
            f"Cannot transition {getattr(record, 'kind', 'record')} from {source} to {target}",
            record.id,
        )


def _replace(record: R, **updates: Any) -> R:
    return record.model_copy(update={"revision": record.revision + 1, **updates})


def _complete(record: Record) -> bool:
    return getattr(record, "status", None) == "completed"


def _dependency_satisfied(record: Record, records: Mapping[str, PersistentRecord]) -> bool:
    if isinstance(record, Belief):
        return record.epistemic_status == "supported"
    if isinstance(record, ArtifactRef):
        return record.valid
    if isinstance(record, Criterion):
        return any(
            isinstance(r, VerificationResult)
            and r.criterion_id == record.id
            and r.disposition == "accept"
            for r in records.values()
        )
    if isinstance(record, ActionRecord):
        return record.status == "succeeded"
    return _complete(record)


def _eligible(task: TaskFrame, records: Mapping[str, PersistentRecord]) -> bool:
    dependencies: dict[str, Record] = dict(records)
    for item in records.values():
        if isinstance(item, Goal):
            dependencies.update((c.id, c) for c in item.criteria)
    return (
        task.wake_condition is None
        and all(
            ref.id in dependencies
            and dependencies[ref.id].revision == ref.revision
            and _dependency_satisfied(dependencies[ref.id], records)
            for ref in task.dependencies
        )
        and all(
            getattr(records.get(identifier), "status", None) == "succeeded"
            for identifier in task.pending_action_ids
        )
    )


def _control(records: dict[str, PersistentRecord], event: JournalEvent) -> None:
    g = records.get(event.subject.id)
    if not isinstance(g, Goal) or event.subject.revision != g.revision:
        _fail(
            "invalid_subject", "Goal control requires the current goal revision", event.subject.id
        )
    target = {"goal.paused": "paused", "goal.resumed": "active", "goal.cancelled": "cancelled"}[
        event.event_type
    ]
    if event.event_type == "goal.resumed" and g.status != "paused":
        _fail("invalid_transition", "Resume requires a paused goal", g.id)
    validate_transition(g, target)
    if g.status == target:
        _fail("invalid_transition", "Goal control must change status", g.id)
    records[g.id] = _replace(g, status=target)
    tasks = [r for r in records.values() if isinstance(r, TaskFrame) and r.goal_id == g.id]
    for task in tasks:
        if task.status in _TASK_TERMINAL:
            continue
        if target == "cancelled":
            records[task.id] = _replace(
                task, status="cancelled", fencing_token=task.fencing_token + 1
            )
        elif target == "paused" and task.status in {"ready", "running", "verifying"}:
            records[task.id] = _replace(
                task, status="suspended", fencing_token=task.fencing_token + 1
            )
        elif target == "active" and task.status == "suspended" and task.wake_condition is None:
            if _eligible(task, records):
                records[task.id] = _replace(task, status="ready")
            else:
                pending = tuple(
                    Ref(id=i, revision=records[i].revision, kind="action")
                    for i in task.pending_action_ids
                    if getattr(records[i], "status", None) != "succeeded"
                )
                records[task.id] = _replace(
                    task,
                    status="blocked",
                    wake_condition=WakeCondition(
                        kind="dependency_change",
                        refs=pending or task.dependencies,
                        reason_code="pending_actions" if pending else "dependencies_unsettled",
                    ),
                )
    if target in {"paused", "cancelled"}:
        task_ids = {task.id for task in tasks}
        for item in tuple(records.values()):
            if isinstance(item, Episode) and item.task_id in task_ids and item.status == "running":
                records[item.id] = _replace(item, status="fenced")
            elif (
                target == "cancelled"
                and isinstance(item, Commitment)
                and item.goal_id == g.id
                and item.status not in _TASK_TERMINAL
            ):
                records[item.id] = _replace(item, status="cancelled")
            elif (
                target == "cancelled"
                and isinstance(item, ActionRecord)
                and item.goal_id == g.id
                and item.status in {"proposed", "awaiting_review", "authorized"}
            ):
                records[item.id] = _replace(item, status="cancelled")


def _explicit_records(value: object, *, record: bool = False) -> None:
    """Prevent Pydantic defaults from introducing time or identity during replay."""
    if isinstance(value, Mapping):
        if record and not {"id", "revision", "created_at", "schema_version", "kind"}.issubset(
            value
        ):
            _fail(
                "invalid_event", "Post-images require explicit identity, version, revision and time"
            )
        if value.get("kind") == "signal" and "received_at" not in value:
            _fail("invalid_event", "Signals require an explicit received_at timestamp")
        for name in (
            "criteria",
            "constraints",
            "restrictions",
            "commitments",
            "beliefs",
            "pending_effects",
        ):
            if name in value:
                _explicit_records(value[name], record=True)
        if value.get("kind") == "context_packet":
            for name in ("goal", "task"):
                _explicit_records(value.get(name), record=True)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _explicit_records(child, record=record)


def _refs(value: object) -> Iterable[Ref]:
    if isinstance(value, Ref):
        yield value
    elif isinstance(value, ContextPacket):
        # Its embedded DTOs describe historical context, not current graph edges.
        yield from value.mandatory_read_set
    elif isinstance(value, Model):
        for name in type(value).model_fields:
            yield from _refs(getattr(value, name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _refs(item)


def _validate_state(records: Mapping[str, PersistentRecord]) -> None:
    embedded: dict[str, Record] = {}
    criterion_ids: set[str] = set()
    for item in records.values():
        if isinstance(item, Goal):
            for criterion in item.criteria:
                if criterion.id in criterion_ids:
                    _fail(
                        "duplicate_criterion", "Criterion IDs must be globally unique", criterion.id
                    )
                criterion_ids.add(criterion.id)
            embedded.update((r.id, r) for r in (*item.criteria, *item.constraints))
    all_records: dict[str, Record] = {**embedded, **records}

    def require(identifier: str, kinds: set[str] | None = None) -> Record:
        found = all_records.get(identifier)
        if found is None or (kinds and getattr(found, "kind", None) not in kinds):
            _fail("missing_reference", "Missing or wrong-kind record reference", identifier)
        return found

    graph: dict[str, tuple[str, ...]] = {}
    verification_criteria: set[str] = set()
    for item in records.values():
        scope = getattr(item, "scope", None)
        if isinstance(scope, Scope):
            if scope.owner_id:
                require(scope.owner_id, {"principal"})
            if scope.goal_id:
                require(scope.goal_id, {"goal"})
            for principal_id in scope.principal_ids:
                require(principal_id, {"principal"})
        for field, kinds in (
            ("budget_id", {"budget"}),
            ("reservation_id", {"reservation"}),
            ("grant_id", {"grant"}),
        ):
            identifier = getattr(item, field, None)
            if identifier:
                require(identifier, kinds)
        if isinstance(item, Budget) and item.parent_id:
            require(item.parent_id, {"budget"})
        if isinstance(item, ArtifactRef):
            for principal_id in item.shared_with:
                require(principal_id, {"principal"})
        if isinstance(item, Grant):
            require(item.issuer_id, {"principal"})
            require(item.subject_id, {"principal"})
        elif isinstance(item, Reservation):
            require(item.subject_id)
        elif item.kind == "signal":
            require(item.principal_id, {"principal"})
        for ref in _refs(item):
            found = require(ref.id)
            if ref.kind != "record" and getattr(found, "kind", None) != ref.kind:
                _fail("missing_reference", "Reference kind disagrees with its target", ref.id)
            if found.revision < ref.revision:
                _fail("missing_reference", "Reference points to a future revision", ref.id)
        if isinstance(item, Goal):
            require(item.owner_id, {"principal"})
            for identifier in item.required_commitment_ids:
                c = require(identifier, {"commitment"})
                if isinstance(c, Commitment) and c.goal_id != item.id:
                    _fail(
                        "invalid_reference",
                        "Required commitment belongs to another goal",
                        identifier,
                    )
            if item.parent_id:
                require(item.parent_id, {"goal"})
            if item.budget_id:
                require(item.budget_id, {"budget"})
        elif isinstance(item, (TaskFrame, Commitment, ActionRecord)):
            require(item.goal_id, {"goal"})
            if isinstance(item, Commitment):
                require(item.responsible_principal_id, {"principal"})
            elif isinstance(item, TaskFrame):
                c = require(item.commitment_id, {"commitment"})
                if isinstance(c, Commitment) and c.goal_id != item.goal_id:
                    _fail("invalid_reference", "Task and commitment goals disagree", item.id)
                for identifier in item.pending_action_ids:
                    a = require(identifier, {"action"})
                    if isinstance(a, ActionRecord) and a.task_id != item.id:
                        _fail(
                            "invalid_reference",
                            "Pending action belongs to another task",
                            identifier,
                        )
            else:
                task = require(item.task_id, {"task"})
                if isinstance(task, TaskFrame) and task.goal_id != item.goal_id:
                    _fail("invalid_reference", "Action and task goals disagree", item.id)
            if isinstance(item, (TaskFrame, Commitment)):
                graph[item.id] = tuple(ref.id for ref in item.dependencies)
        elif isinstance(item, Episode):
            require(item.task_id, {"task"})
            require(item.context_packet_id, {"context_packet"})
        elif isinstance(item, VerificationResult):
            require(item.criterion_id, {"criterion"})
            if item.criterion_id in verification_criteria:
                _fail(
                    "ambiguous_verification",
                    "A criterion has one current verification result",
                    item.criterion_id,
                )
            verification_criteria.add(item.criterion_id)

    # Iterative DFS avoids recursion limits on a valid, long task chain.
    done: set[str] = set()
    for root in graph:
        active: set[str] = set()
        stack = [(root, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                active.remove(node)
                done.add(node)
            elif node in active:
                _fail("dependency_cycle", "Task dependencies must be acyclic", node)
            elif node not in done:
                active.add(node)
                stack.append((node, True))
                stack.extend((dep, False) for dep in graph.get(node, ()))


def _completion(goal: Goal, records: Mapping[str, PersistentRecord]) -> None:
    required = set(goal.required_commitment_ids)
    required.update(
        r.id
        for r in records.values()
        if isinstance(r, Commitment) and r.goal_id == goal.id and r.required
    )
    if any(not _complete(records[identifier]) for identifier in required):
        _fail("incomplete_commitment", "Required commitments are not complete", goal.id)
    for criterion in goal.criteria:
        if criterion.required:
            results = [
                r
                for r in records.values()
                if isinstance(r, VerificationResult) and r.criterion_id == criterion.id
            ]
            if not results or any(r.disposition != "accept" for r in results):
                _fail(
                    "unverified_criterion",
                    "Required criterion lacks accepted verification",
                    criterion.id,
                )
    if any(
        isinstance(r, ActionRecord) and r.goal_id == goal.id and r.status in _UNSETTLED_ACTIONS
        for r in records.values()
    ):
        _fail("pending_actions", "Goal has unsettled effects", goal.id)


def reduce_event(state: StateView, event: JournalEvent) -> StateView:
    """Apply one contiguous committed event without modifying its inputs."""
    if event.sequence != state.sequence + 1:
        _fail("invalid_sequence", "Journal sequence must be contiguous")
    records = {item.id: item for item in state.records}
    if len(records) != len(state.records):
        _fail("duplicate_record", "State contains duplicate record identities")
    before = dict(records)
    payload = event.model_dump(mode="json")["payload"]
    try:
        if event.event_type == "state.changed":
            if (
                set(payload) != {"records"}
                or not isinstance(payload["records"], list)
                or not payload["records"]
            ):
                _fail("invalid_event", "state.changed requires a nonempty records list")
            _explicit_records(payload["records"], record=True)
            updates = [_ADAPTER.validate_python(raw) for raw in payload["records"]]
            if len({r.id for r in updates}) != len(updates):
                _fail("duplicate_record", "Event writes each record at most once")
            # Promotion changes representation, not logical identity/history.
            # Snapshot before all writes so post-image order cannot invent an
            # eligible embedded revision inside this same event.
            embedded_definitions = tuple(
                c
                for record in records.values()
                if isinstance(record, Goal)
                for c in (*record.constraints, *record.criteria)
            )
            for item in updates:
                if isinstance(item, JournalEvent):
                    _fail("invalid_event", "Journal events cannot be materialized as records")
                previous = records.get(item.id)
                if previous is None:
                    matches = tuple(c for c in embedded_definitions if c.id == item.id)
                    if (
                        isinstance(item, Criterion) and matches and any(c != item for c in matches)
                    ) or (
                        item.revision != 1
                        and (
                            not isinstance(item, (Constraint, Criterion))
                            or not matches
                            or any(c != item for c in matches)
                        )
                    ):
                        _fail(
                            "invalid_revision",
                            "New records start at revision one; promotion preserves "
                            "an existing embedded definition exactly",
                            item.id,
                        )
                else:
                    if (
                        item.kind != previous.kind
                        or item.created_at != previous.created_at
                        or item.revision != previous.revision + 1
                    ):
                        _fail(
                            "invalid_revision",
                            "Updates preserve identity and advance one revision",
                            item.id,
                        )
                    if (
                        isinstance(item, Goal)
                        and isinstance(previous, Goal)
                        and item.status != previous.status
                        and (item.status in {"paused", "cancelled"} or previous.status == "paused")
                    ):
                        _fail(
                            "invalid_transition",
                            "Pause, resume and cancel require explicit goal control events",
                            item.id,
                        )
                    if isinstance(item, ArtifactRef) and isinstance(previous, ArtifactRef):
                        immutable = (
                            "digest",
                            "storage_location",
                            "media_type",
                            "origin",
                            "scope",
                            "size_bytes",
                            "valid",
                        )
                        if any(
                            getattr(item, name) != getattr(previous, name) for name in immutable
                        ):
                            _fail(
                                "invalid_transition",
                                "Artifact identity is immutable; validity requires invalidation",
                                item.id,
                            )
                    if hasattr(item, "status"):
                        validate_transition(previous, item.status)
                    if isinstance(item, TaskFrame) and isinstance(previous, TaskFrame):
                        if item.fencing_token < previous.fencing_token or (
                            previous.status == "running"
                            and item.status in {"suspended", "blocked", "failed", "cancelled"}
                            and item.fencing_token <= previous.fencing_token
                        ):
                            _fail(
                                "invalid_fence",
                                "Stopping a running task must advance its fence",
                                item.id,
                            )
                records[item.id] = item
        elif event.event_type in {"goal.paused", "goal.resumed", "goal.cancelled"}:
            if payload:
                _fail("invalid_event", "Goal controls have an empty payload")
            _control(records, event)
        elif event.event_type == "artifact_invalidated":
            artifact = records.get(event.subject.id)
            if payload:
                _fail("invalid_event", "Artifact invalidation has an empty payload")
            if (
                not isinstance(artifact, ArtifactRef)
                or not artifact.valid
                or event.subject.revision != artifact.revision
                or event.subject.kind not in {"record", "artifact"}
            ):
                _fail(
                    "invalid_transition",
                    "Invalidation requires a current valid artifact",
                    event.subject.id,
                )
            records[artifact.id] = _replace(artifact, valid=False)
        elif event.event_type == "completion.invalidated":
            if (
                set(payload) != {"record_ids", "reason"}
                or not isinstance(payload["reason"], str)
                or not payload["reason"].strip()
                or not isinstance(payload["record_ids"], list)
                or not payload["record_ids"]
            ):
                _fail("invalid_event", "Invalidation requires record IDs and a reason")
            identifiers = payload["record_ids"]
            if any(not isinstance(i, str) for i in identifiers) or len(set(identifiers)) != len(
                identifiers
            ):
                _fail("invalid_event", "Invalidation IDs must be unique strings")
            for identifier in identifiers:
                invalidated = records.get(identifier)
                if not isinstance(invalidated, (Goal, TaskFrame, Commitment)) or not _complete(
                    invalidated
                ):
                    _fail(
                        "invalid_transition",
                        "Invalidation reopens completed goals, tasks or commitments only",
                        identifier,
                    )
            # Apply goals first so the record_ids order cannot change resumption.
            for identifier in identifiers:
                invalidated = records[identifier]
                if isinstance(invalidated, Goal):
                    records[identifier] = _replace(invalidated, status="active")
            # Reopen the entire bundle before testing dependencies. A dependent
            # must see an invalidated prerequisite's new state in either ID order.
            for identifier in identifiers:
                invalidated = records[identifier]
                if isinstance(invalidated, (TaskFrame, Commitment)):
                    goal_record = records[invalidated.goal_id]
                    status = (
                        "ready" if getattr(goal_record, "status", None) == "active" else "suspended"
                    )
                    records[identifier] = _replace(invalidated, status=status)
            for identifier in identifiers:
                reopened = records[identifier]
                if (
                    isinstance(reopened, TaskFrame)
                    and reopened.status == "ready"
                    and not _eligible(reopened, records)
                ):
                    # Eligibility belongs to this same event's one revision.
                    records[identifier] = reopened.model_copy(
                        update={
                            "status": "blocked",
                            "wake_condition": WakeCondition(
                                kind="dependency_change",
                                refs=reopened.dependencies,
                                reason_code="completion_invalidated",
                            ),
                        }
                    )
        elif event.event_type == "audit.recorded":
            if (
                set(payload) != {"reason_code", "details"}
                or not isinstance(payload["reason_code"], str)
                or not payload["reason_code"].strip()
                or not isinstance(payload["details"], dict)
            ):
                _fail("invalid_event", "Audit event requires reason_code and details")
            if len(event.model_dump_json().encode("utf-8")) > 65536:
                _fail("invalid_event", "Audit metadata exceeds 65536 bytes")
        else:
            _fail("unknown_event", "Unsupported executive event type")
    except (ValidationError, TypeError, KeyError) as exc:
        raise InvalidTransition("invalid_event", "Invalid executive event payload") from exc
    _validate_state(records)
    for identifier, item in records.items():
        previous = before.get(identifier)
        if (
            isinstance(item, Goal)
            and item.status == "completed"
            and (previous is None or not _complete(previous))
        ):
            _completion(item, records)
        if (
            isinstance(item, ActionRecord)
            and item.status == "dispatching"
            and (previous is None or getattr(previous, "status", None) != "dispatching")
        ):
            g = records[item.goal_id]
            if isinstance(g, Goal) and g.status != "active":
                _fail("inactive_goal", "Dispatch requires an active goal", item.id)
        if isinstance(item, TaskFrame) and (previous is None or item != previous):
            g = records[item.goal_id]
            if item.status in {"ready", "running", "verifying"}:
                if isinstance(g, Goal) and g.status != "active":
                    _fail("inactive_goal", "Runnable tasks require an active goal", item.id)
                if (
                    item.status == "ready"
                    or (
                        item.status == "running"
                        and previous is not None
                        and getattr(previous, "status", None) != "running"
                    )
                ) and not _eligible(item, records):
                    _fail(
                        "blocked_task",
                        "Runnable tasks require settled dependencies and effects",
                        item.id,
                    )
    return StateView(
        sequence=event.sequence, records=tuple(sorted(records.values(), key=lambda r: r.id))
    )


def rebuild(events: Iterable[JournalEvent]) -> StateView:
    """Reconstruct materialized records from the complete retained journal."""
    state = StateView()
    seen: set[str] = set()
    for event in events:
        if event.id in seen:
            _fail("duplicate_event", "Journal event IDs must be unique", event.id)
        seen.add(event.id)
        state = reduce_event(state, event)
    return state
