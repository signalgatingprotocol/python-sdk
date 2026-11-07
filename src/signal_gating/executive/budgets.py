"""Prepare event-only hierarchical accounting; the coordinator commits atomically.

Combine reservation events with episode/action authorization in the same Store
transaction (merge post-images when creating their subject in the same event).
A trusted quote is an operator/adapter assertion, never a worker assertion.
Recorded actual quantities include estimates; unresolved exposure remains held
in reserved, with accounting_uncertain distinguishing it from final accounting.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import cast
from uuid import uuid4

from .errors import BudgetError
from .messages import CommitBatch, ResolveUsage
from .records import (
    ArtifactRef,
    Budget,
    Goal,
    JournalEvent,
    Limits,
    PersistentRecord,
    Ref,
    Reservation,
    StateView,
    Usage,
    utc_now,
)

_QUANTITIES = ("input_tokens", "output_tokens", "calls", "elapsed_seconds", "cost_microunits")


def _quantity(value: Usage | Limits, name: str) -> int | float | None:
    return cast(int | float | None, getattr(value, name))


def _ref(record: PersistentRecord) -> Ref:
    return Ref(id=record.id, revision=record.revision)


def _chain(state: StateView, budget_id: str) -> tuple[Budget, ...]:
    records = {record.id: record for record in state.records}
    result: list[Budget] = []
    seen: set[str] = set()
    while budget_id:
        budget = records.get(budget_id)
        if not isinstance(budget, Budget) or budget_id in seen:
            raise BudgetError("invalid_budget_tree", "Missing budget or cyclic ancestry")
        if result:
            child = result[-1]
            if child.limits.currency != budget.limits.currency:
                raise BudgetError("currency_mismatch", "Ancestor budget currency differs")
            if any(
                getattr(budget.scope, name) is not None
                and getattr(child.scope, name) != getattr(budget.scope, name)
                for name in ("owner_id", "goal_id")
            ):
                raise BudgetError("budget_scope_mismatch", "Child escapes ancestor scope")
        seen.add(budget_id)
        result.append(budget)
        budget_id = budget.parent_id or ""
    return tuple(result)


def _reservation(state: StateView, identifier: str) -> Reservation:
    result = next((r for r in state.records if r.id == identifier), None)
    if not isinstance(result, Reservation):
        raise BudgetError("unknown_reservation", "Reservation does not exist")
    return result


def _bound(reservation: Reservation, name: str) -> int | float | None:
    if name == "cost_microunits" and not reservation.trusted_maximum_quote:
        return None
    return _quantity(reservation.maximum, name)


def _accounted(reservation: Reservation, name: str) -> int | float | None:
    """Conservative cumulative floor, without replacing a final settlement."""
    quantities = [_quantity(reservation.usage, name)]
    if reservation.observed_usage is not None:
        quantities.append(_quantity(reservation.observed_usage, name))
    present = [q for q in quantities if q is not None]
    return max(present) if present else None


def _overage(reservation: Reservation) -> bool:
    return any(
        _bound(reservation, name) is not None
        and _accounted(reservation, name) is not None
        and cast(int | float, _accounted(reservation, name))
        > cast(int | float, _bound(reservation, name))
        for name in _QUANTITIES
    )


def _hold(reservation: Reservation, name: str) -> int | float | None:
    used = _accounted(reservation, name)
    if reservation.status in ("settled", "released") or (
        reservation.usage.status == "known" and used is not None
    ):
        return 0
    maximum = _bound(reservation, name)
    return None if maximum is None else max(0, maximum - (used or 0))


class BudgetPolicy:
    """Pure admission/accounting preparation against a versioned StateView.

    The default money policy requires a finite trusted charge bound, including
    zero-price operations. Non-strict mode permits unknown exposure only where
    all ancestor ceilings for that quantity are unbounded. It never turns None
    into zero. Every affected ancestor is fenced by the returned CAS read set.
    """

    def __init__(
        self, *, strict_money: bool = True, clock: Callable[[], datetime] = utc_now
    ) -> None:
        self.strict_money = strict_money
        self.clock = clock

    def _members(self, state: StateView, budget_id: str) -> tuple[Reservation, ...]:
        return tuple(
            r
            for r in state.records
            if isinstance(r, Reservation)
            and any(b.id == budget_id for b in _chain(state, r.budget_id))
        )

    def _batch(
        self,
        state: StateView,
        chain: tuple[Budget, ...],
        reservation: Reservation,
        writes: tuple[PersistentRecord, ...],
        *,
        reason: str = "budget_idempotent",
    ) -> CommitBatch:
        budget = chain[0]
        goal = next(
            (r for r in state.records if isinstance(r, Goal) and r.id == budget.scope.goal_id), None
        )
        principal_id = budget.scope.owner_id or (goal.owner_id if goal else None)
        if principal_id is None:
            raise BudgetError("unowned_budget", "Budget requires an owner scope or goal")
        event = JournalEvent(
            created_at=self.clock(),
            sequence=state.sequence + 1,
            event_type="state.changed" if writes else "audit.recorded",
            producer_id="budget-policy-v1",
            principal_id=principal_id,
            subject=_ref(reservation),
            payload={"records": [r.model_dump(mode="json") for r in writes]}
            if writes
            else {"reason_code": reason, "details": {}},
        )
        previous = next((r for r in state.records if r.id == reservation.id), None)
        return CommitBatch(
            events=(event,),
            expected_revisions=(
                *(_ref(b) for b in chain),
                *((_ref(previous),) if previous else ()),
            ),
        )

    def reserve(
        self,
        state: StateView,
        budget_id: str,
        limits: Limits,
        reservation_id: str,
        *,
        subject_id: str | None = None,
        trusted_maximum_quote: bool = False,
    ) -> CommitBatch:
        chain = _chain(state, budget_id)
        subject_id = subject_id or budget_id
        if subject_id == reservation_id:
            raise BudgetError(
                "invalid_reservation_subject", "Reservation requires a separate subject binding"
            )
        if limits.currency != chain[0].limits.currency:
            raise BudgetError("currency_mismatch", "Reservation currency differs from budget")
        if self.strict_money and (limits.cost_microunits is None or not trusted_maximum_quote):
            raise BudgetError(
                "unknown_maximum_quote", "Strict money requires a trusted finite quote"
            )
        if any(_overage(r) for b in chain for r in self._members(state, b.id)):
            raise BudgetError("adapter_bound_exceeded", "Adapter exceeded its promised bound")
        if any(
            r.accounting_conflict is not None for b in chain for r in self._members(state, b.id)
        ):
            raise BudgetError(
                "accounting_conflict", "Unresolved observed usage blocks budget ancestry"
            )
        previous = next((r for r in state.records if r.id == reservation_id), None)
        if previous is not None:
            if not isinstance(previous, Reservation) or (
                previous.budget_id != budget_id
                or previous.subject_id != subject_id
                or previous.maximum != limits
                or previous.trusted_maximum_quote != trusted_maximum_quote
            ):
                raise BudgetError("reservation_conflict", "Reservation ID has different binding")
            if previous.status != "reserved":
                raise BudgetError("reservation_consumed", "Reservation is already reconciled")
            return self._batch(state, chain, previous, ())
        for budget in chain:
            for name in _QUANTITIES:
                ceiling = _quantity(budget.limits, name)
                amount = (
                    None
                    if name == "cost_microunits" and not trusted_maximum_quote
                    else _quantity(limits, name)
                )
                actual = _quantity(budget.actual, name)
                held = _quantity(budget.reserved, name)
                if ceiling is not None and (
                    amount is None
                    or actual is None
                    or held is None
                    or actual + held + amount > ceiling
                ):
                    raise BudgetError(
                        "budget_exhausted",
                        f"Cannot bound {name} within budget",
                        record_id=budget.id,
                    )
        reservation = Reservation(
            id=reservation_id,
            created_at=self.clock(),
            budget_id=budget_id,
            subject_id=subject_id,
            maximum=limits,
            trusted_maximum_quote=trusted_maximum_quote,
            usage=Usage(currency=limits.currency),
        )
        updates = self._balances(state, chain, None, reservation)
        return self._batch(state, chain, reservation, (*updates, reservation))

    def _balances(
        self,
        state: StateView,
        chain: tuple[Budget, ...],
        previous: Reservation | None,
        reservation: Reservation,
    ) -> tuple[Budget, ...]:
        updates: list[Budget] = []
        for budget in chain:
            members = tuple(r for r in self._members(state, budget.id) if r.id != reservation.id)
            members = (*members, reservation)
            held: dict[str, int | float | None] = {}
            actual: dict[str, int | float | None] = {}
            for name in _QUANTITIES:
                old = _accounted(previous, name) if previous else 0
                new = _accounted(reservation, name)
                balance = _quantity(budget.actual, name)
                if balance is None:
                    raise BudgetError(
                        "unknown_actual_balance", "Actual balance needs reconciliation"
                    )
                actual[name] = balance - (old or 0) + (new or 0)
                before_hold = _hold(previous, name) if previous else 0
                after_hold = _hold(reservation, name)
                held_balance = _quantity(budget.reserved, name)
                if after_hold is None:
                    if held_balance is not None:
                        existing = [_hold(r, name) for r in self._members(state, budget.id)]
                        if None in existing or held_balance != sum(
                            cast(int | float, value) for value in existing
                        ):
                            raise BudgetError(
                                "untracked_hold",
                                "Cannot hide an untracked hold in unknown exposure",
                            )
                    held[name] = None
                elif before_hold is not None and held_balance is not None:
                    held[name] = held_balance - before_hold + after_hold
                else:
                    # A formerly unbounded hold can become known only after
                    # reconciliation. Rebuild that quantity from its ledger.
                    quantities = [_hold(r, name) for r in members]
                    held[name] = (
                        None
                        if None in quantities
                        else sum(cast(int | float, q) for q in quantities)
                    )
            uncertain = any(
                r.status == "uncertain"
                or r.accounting_conflict is not None
                or _overage(r)
                or any(_hold(r, name) is None for name in _QUANTITIES)
                for r in members
            )
            actual_status = "known"
            uncertain_statuses = {r.usage.status for r in members if r.status == "uncertain"}
            if (
                any(r.accounting_conflict is not None for r in members)
                or "unknown" in uncertain_statuses
            ):
                actual_status = "unknown"
            elif "estimated" in uncertain_statuses:
                actual_status = "estimated"
            updates.append(
                budget.model_copy(
                    update={
                        "revision": budget.revision + 1,
                        "actual": Usage.model_validate(
                            dict(status=actual_status, currency=budget.limits.currency, **actual)
                        ),
                        "reserved": Usage.model_validate(
                            dict(
                                status="unknown" if None in held.values() else "known",
                                currency=budget.limits.currency,
                                **held,
                            )
                        ),
                        "accounting_uncertain": uncertain,
                    }
                )
            )
        return tuple(updates)

    def settle(self, state: StateView, reservation_id: str, usage: Usage) -> CommitBatch:
        return self._settle(state, reservation_id, usage)

    def observe_usage(self, state: StateView, reservation_id: str, usage: Usage) -> CommitBatch:
        """Retain measured usage even after an immutable final/owner settlement.

        Ordinary settle still rejects conflicting finals. This observation path
        preserves that settlement, journals the new report, raises its measured
        accounting floor monotonically and durably blocks affected ancestry.
        A bound violation remains a violation after any later owner resolution.
        """
        previous = _reservation(state, reservation_id)
        reason = previous.accounting_conflict
        if reason is None:
            try:
                return self.settle(state, reservation_id, usage)
            except BudgetError as exc:
                if exc.code not in {"settlement_conflict", "currency_mismatch"}:
                    raise
                reason = exc.code
        chain = _chain(state, previous.budget_id)
        observed = previous.observed_usage
        quantities: dict[str, int | float | None] = {}
        for name in _QUANTITIES:
            old = _quantity(observed, name) if observed else None
            # Foreign money cannot be added to a budget without a trusted FX
            # conversion. Retain the exact raw report in the conflict audit;
            # the unresolved conflict blocks all new admission in its ancestry.
            new = (
                None
                if name == "cost_microunits" and usage.currency != previous.maximum.currency
                else _quantity(usage, name)
            )
            values = [q for q in (old, new) if q is not None]
            quantities[name] = max(values) if values else None
        envelope = Usage.model_validate(
            dict(
                status="unknown",
                currency=previous.maximum.currency,
                source="conservative-conflicting-observations",
                **quantities,
            )
        )
        reservation = previous.model_copy(
            update={
                "revision": previous.revision + 1,
                "observed_usage": envelope,
                "accounting_conflict": reason,
            }
        )
        updates = self._balances(state, chain, previous, reservation)
        result = self._batch(state, chain, reservation, (*updates, reservation))
        details = {
            "reservation_id": reservation.id,
            "previous_revision": previous.revision,
            "prior_settlement": previous.usage.model_dump(mode="json"),
            "observed_usage": usage.model_dump(mode="json"),
        }
        audit = result.events[0].model_copy(
            update={
                "id": uuid4().hex,
                "event_type": "audit.recorded",
                "payload": {"reason_code": reason, "details": details},
            }
        )
        events = (*result.events, audit)
        if _overage(reservation):
            events = (
                *events,
                audit.model_copy(
                    update={
                        "id": uuid4().hex,
                        "payload": {"reason_code": "adapter_bound_exceeded", "details": details},
                    }
                ),
            )
        return result.model_copy(update={"events": events})

    def _settle(
        self,
        state: StateView,
        reservation_id: str,
        usage: Usage,
        evidence: tuple[Ref, ...] | None = None,
    ) -> CommitBatch:
        previous = _reservation(state, reservation_id)
        chain = _chain(state, previous.budget_id)
        if usage.currency != previous.maximum.currency:
            raise BudgetError("currency_mismatch", "Usage currency differs from reservation")
        if previous.status in ("settled", "released"):
            if previous.status != "settled" or previous.usage != usage:
                raise BudgetError("settlement_conflict", "Final settlement cannot be replaced")
            return self._batch(state, chain, previous, ())
        complete = usage.status == "known"
        if not complete:
            # Incomplete cumulative observations cannot retract an earlier
            # figure. A complete known settlement can reconcile estimates.
            usage = usage.model_copy(
                update={
                    name: max(
                        cast(int | float, _quantity(previous.usage, name)),
                        _quantity(usage, name) or 0,
                    )
                    for name in _QUANTITIES
                    if _quantity(previous.usage, name) is not None
                }
            )
        reservation = previous.model_copy(
            update={
                "revision": previous.revision + 1,
                "usage": usage,
                "status": "settled" if complete else "uncertain",
                "evidence": previous.evidence if evidence is None else evidence,
            }
        )
        updates = self._balances(state, chain, previous, reservation)
        result = self._batch(state, chain, reservation, (*updates, reservation))
        if _overage(reservation):
            breach = result.events[0].model_copy(
                update={
                    "id": uuid4().hex,
                    "sequence": state.sequence + 2,
                    "event_type": "audit.recorded",
                    "payload": {
                        "reason_code": "adapter_bound_exceeded",
                        "details": {"reservation_id": reservation.id},
                    },
                }
            )
            result = result.model_copy(update={"events": (*result.events, breach)})
        return result

    def mark_unknown(self, state: StateView, reservation_id: str) -> CommitBatch:
        previous = _reservation(state, reservation_id)
        chain = _chain(state, previous.budget_id)
        if previous.status == "uncertain":
            return self._batch(state, chain, previous, ())
        if previous.status != "reserved":
            raise BudgetError("settlement_conflict", "Settled reservation cannot become unknown")
        reservation = previous.model_copy(
            update={"revision": previous.revision + 1, "status": "uncertain"}
        )
        updates = self._balances(state, chain, previous, reservation)
        return self._batch(state, chain, reservation, (*updates, reservation))

    def resolve_usage(self, command: ResolveUsage, state: StateView) -> CommitBatch:
        """Intake callback. Intake must authenticate the owner before invoking it."""
        reservation = _reservation(state, command.reservation_id)
        if reservation.revision != command.expected_revision:
            raise BudgetError("stale_usage_resolution", "Reservation revision changed")
        if _overage(reservation) and (
            reservation.accounting_conflict is not None
            or command.accept_reserved_maximum
            or command.usage is None
            or any(
                _accounted(reservation, name) is not None
                and cast(int | float, _accounted(reservation, name))
                > cast(int | float, _quantity(command.usage, name))
                for name in _QUANTITIES
            )
        ):
            raise BudgetError(
                "adapter_bound_exceeded", "Usage resolution cannot erase observed overage"
            )
        if reservation.accounting_conflict is not None:
            raise BudgetError(
                "accounting_conflict", "Usage resolution cannot erase conflicting observations"
            )
        budget = _chain(state, reservation.budget_id)[0]
        records = {r.id: r for r in state.records}
        goal = records.get(budget.scope.goal_id or "")
        owner_id = budget.scope.owner_id or (goal.owner_id if isinstance(goal, Goal) else None)
        for ref in command.evidence:
            current = records.get(ref.id)
            if (
                current is None
                or current.revision != ref.revision
                or (ref.kind != "record" and current.kind != ref.kind)
            ):
                raise BudgetError("stale_evidence", "Evidence must name a current record")
            if isinstance(current, ArtifactRef) and (
                not current.valid
                or current.scope.goal_id != budget.scope.goal_id
                or (owner_id != current.scope.owner_id and owner_id not in current.shared_with)
            ):
                raise BudgetError("evidence_access_denied", "Usage evidence access denied")
        usage = command.usage
        if command.accept_reserved_maximum:
            if _overage(reservation):
                raise BudgetError(
                    "adapter_bound_exceeded", "Maximum acceptance cannot erase overage"
                )
            if any(_bound(reservation, name) is None for name in _QUANTITIES):
                raise BudgetError("unknown_reserved_maximum", "Cannot charge an unknown maximum")
            usage = Usage(
                status="known",
                source="owner-accepted-reserved-maximum",
                **reservation.maximum.model_dump(),
            )
        assert usage is not None
        result = self._settle(state, reservation.id, usage, command.evidence)
        reads = {ref.id: ref for ref in (*result.expected_revisions, *command.evidence)}
        return result.model_copy(update={"expected_revisions": tuple(reads.values())})
