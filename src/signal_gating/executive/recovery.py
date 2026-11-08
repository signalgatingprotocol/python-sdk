"""Prepare conservative restart fencing on the single owning writer."""

from __future__ import annotations

from datetime import datetime

from .actions import CapabilityRegistry
from .authority import reference
from .budgets import BudgetPolicy
from .messages import CommitBatch
from .records import (
    ActionRecord,
    Episode,
    JournalEvent,
    PersistentRecord,
    Principal,
    Reservation,
    TaskFrame,
)
from .reducers import reduce_event
from .store import Store


def recover(store: Store, *, registry: CapabilityRegistry, now: datetime) -> CommitBatch:
    """Never infer a cancelled provider call or a successful external effect.

    Binding-specific lookup is scheduled later by Dispatcher, outside this batch.
    Paused/cancelled goals are preserved. A recovered lease cannot accept proposals.
    """
    store.check_writable()
    state = store.view()
    owner = next(
        (r for r in state.records if isinstance(r, Principal) and "owner" in r.roles), None
    )
    if owner is None:
        return CommitBatch()
    writes: list[PersistentRecord] = []
    uncertain: set[str] = set()
    for record in state.records:
        if isinstance(record, Episode) and record.status == "running":
            writes.append(
                record.model_copy(update={"revision": record.revision + 1, "status": "fenced"})
            )
            if record.reservation_id:
                uncertain.add(record.reservation_id)
        elif isinstance(record, TaskFrame) and record.status == "running":
            writes.append(
                record.model_copy(
                    update={
                        "revision": record.revision + 1,
                        "status": "suspended",
                        "fencing_token": record.fencing_token + 1,
                    }
                )
            )
        elif isinstance(record, ActionRecord) and record.status in {"dispatching", "reconciling"}:
            writes.append(
                record.model_copy(
                    update={
                        "revision": record.revision + 1,
                        "status": "outcome_unknown"
                        if record.status == "dispatching"
                        else "needs_human",
                    }
                )
            )
            if record.reservation_id:
                uncertain.add(record.reservation_id)
    events: list[JournalEvent] = []
    if writes:
        events.append(
            JournalEvent(
                created_at=now,
                sequence=1,
                event_type="state.changed",
                producer_id="kernel-recovery",
                principal_id=owner.id,
                subject=reference(owner),
                payload={"records": [r.model_dump(mode="json") for r in writes]},
            )
        )
    projected = state
    for event in events:
        projected = reduce_event(
            projected, event.model_copy(update={"sequence": projected.sequence + 1})
        )
    policy = BudgetPolicy(clock=lambda: now)
    for identifier in sorted(uncertain):
        reservation = next((r for r in projected.records if r.id == identifier), None)
        if not isinstance(reservation, Reservation) or reservation.status not in {
            "reserved",
            "uncertain",
        }:
            continue
        batch = policy.mark_unknown(projected, identifier)
        for event in batch.events:
            projected = reduce_event(
                projected, event.model_copy(update={"sequence": projected.sequence + 1})
            )
            events.append(event)
    return CommitBatch(
        events=tuple(events), expected_revisions=tuple(reference(r) for r in state.records)
    )
