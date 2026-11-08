"""Durable, fail-closed action dispatch and uncertain-effect reconciliation.

Only the Store's owning coordinator may call this component. Synchronous prepare,
recheck and commit sections contain no awaits. All adapter calls follow committed
state and run outside transactions. Cancellation never proves absence of effects.
Adapters and their resource extractors, schemas and quotes are trusted operator
configuration; this module is not a sandbox against its own host or adapters.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, Protocol, cast
from uuid import uuid4

from signal_gating._immutable import deep_thaw

from .artifacts import ArtifactStore
from .authority import Authority, ResourceExtractor, decision, reference
from .budgets import BudgetPolicy
from .errors import ExecutiveError
from .memory import records_by_id
from .messages import CommitBatch, OutboxEntry
from .records import (
    ActionObservation,
    ActionRecord,
    ArtifactRef,
    CapabilitySpec,
    Commitment,
    Criterion,
    DecisionReceipt,
    Disposition,
    Goal,
    JournalEvent,
    Limits,
    PersistentRecord,
    PolicyDecision,
    Principal,
    Ref,
    TaskFrame,
    Usage,
    VerificationResult,
    utc_now,
)
from .store import Store
from .workers import _wait_until


class ExecutionAdapter(Protocol):
    async def execute(self, action: ActionRecord) -> ActionObservation: ...
    async def reconcile(self, action: ActionRecord) -> ActionObservation: ...
    async def cancel(self, action: ActionRecord) -> ActionObservation: ...


# Deliberately closed JSON Schema subset. Unsupported validation keywords fail at
# registration; a trusted validator can implement a larger schema dialect.
_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "description",
    "title",
    "$schema",
}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _check_schema(schema: Mapping[str, Any]) -> None:
    if set(schema) - _SCHEMA_KEYS or ("type" in schema and schema["type"] not in _TYPES):
        raise ValueError("Unsupported capability input schema; configure a trusted validator")
    for child in schema.get("properties", {}).values():
        _check_schema(child)
    if "items" in schema:
        _check_schema(schema["items"])
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        raise ValueError("additionalProperties must be boolean")


def _json_equal(left: Any, right: Any) -> bool:
    # JSON booleans and numbers are distinct even though Python True == 1.
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    if (
        isinstance(left, Sequence)
        and not isinstance(left, str)
        and isinstance(right, Sequence)
        and not isinstance(right, str)
    ):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _valid(value: Any, schema: Mapping[str, Any]) -> bool:
    kind = schema.get("type")
    matches = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, Sequence) and not isinstance(value, str),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if kind is not None and not matches[kind]:
        return False
    if "const" in schema and not _json_equal(value, schema["const"]):
        return False
    if "enum" in schema and not any(_json_equal(value, option) for option in schema["enum"]):
        return False
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= value.keys():
            return False
        if schema.get("additionalProperties") is False and set(value) - properties.keys():
            return False
        if any(not _valid(v, properties[k]) for k, v in value.items() if k in properties):
            return False
    if isinstance(value, Sequence) and not isinstance(value, str):
        if any(not _valid(v, schema.get("items", {})) for v in value):
            return False
    for lower, upper, quantity in (
        ("minimum", "maximum", value if type(value) in (int, float) else None),
        ("minLength", "maxLength", len(value) if isinstance(value, str) else None),
        (
            "minItems",
            "maxItems",
            len(value) if isinstance(value, Sequence) and not isinstance(value, str) else None,
        ),
    ):
        if quantity is not None and (
            (lower in schema and quantity < schema[lower])
            or (upper in schema and quantity > schema[upper])
        ):
            return False
    return True


@dataclass(frozen=True)
class _Binding:
    spec: CapabilitySpec
    adapter: ExecutionAdapter
    resource_scope: ResourceExtractor | None
    quote: Limits | None
    quote_source: str | None
    quote_version: str | None
    input_validator: Callable[[Mapping[str, Any]], bool] | None
    require_approval: bool


class CapabilityRegistry:
    """Own trusted bindings. Each registration gets a fresh identity.

    Use registry.specs in ContextAssembler/ContextPacket. Re-registering even the
    same object/version invalidates pending actions. Reconstructing configuration
    on restart accepts spec.binding_id ONLY from operator-persisted configuration
    on its first registration. Never restore configuration from worker/model JSON.
    Replacing a registration always rotates that identity, including after restore.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, _Binding] = {}

    @property
    def specs(self) -> Mapping[str, CapabilitySpec]:
        return MappingProxyType({k: v.spec for k, v in self._bindings.items()})

    def register(
        self,
        spec: CapabilitySpec,
        adapter: ExecutionAdapter,
        *,
        resource_scope: ResourceExtractor | None = None,
        quote: Limits | None = None,
        quote_source: str | None = None,
        quote_version: str | None = None,
        input_validator: Callable[[Mapping[str, Any]], bool] | None = None,
        require_approval: bool | None = None,
    ) -> None:
        if spec.effect_class == "human_only":
            raise ValueError("Human-only capabilities have no executable binding")
        if any(
            not callable(getattr(adapter, method, None))
            for method in ("execute", "reconcile", "cancel")
        ):
            raise TypeError("Adapter must implement execute, reconcile and cancel")
        if resource_scope is not None and not callable(resource_scope):
            raise TypeError("Resource extractor must be callable")
        if input_validator is None:
            _check_schema(spec.input_schema)
        elif not callable(input_validator):
            raise TypeError("Input validator must be callable")
        if require_approval is not None and type(require_approval) is not bool:
            raise TypeError("require_approval must be boolean")
        self._bindings[spec.id] = _Binding(
            spec.model_copy(
                update={
                    "binding_id": spec.binding_id
                    if spec.id not in self._bindings and spec.binding_id is not None
                    else uuid4().hex
                }
            ),
            adapter,
            resource_scope,
            quote,
            quote_source,
            quote_version,
            input_validator,
            spec.effect_class != "read_only" if require_approval is None else require_approval,
        )


class Dispatcher:
    """One-writer dispatch. A receipt never substitutes for current admission.

    Uncooperative calls consume max_active until completion. Late observations
    stay in a bounded queue and must be drained/committed by the owning coordinator
    using record_observation. No background callback commits state or starts work.
    """

    def __init__(
        self,
        store: Store,
        *,
        authority: Authority,
        budgets: BudgetPolicy,
        registry: CapabilityRegistry,
        clock: Callable[[], datetime] = utc_now,
        wait_until: Callable[[datetime], Awaitable[None]] = _wait_until,
        max_active: int = 2,
        max_retries: int = 1,
    ) -> None:
        if type(max_active) is not int or max_active < 1:
            raise ValueError("max_active must be a positive integer")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.store, self.authority, self.budgets, self.registry = (
            store,
            authority,
            budgets,
            registry,
        )
        self.clock, self.wait_until = clock, wait_until
        self.max_active, self.max_retries = max_active, max_retries
        self._pending: set[asyncio.Task[ActionObservation]] = set()
        self._active_ids: set[str] = set()
        self._detached: dict[
            asyncio.Task[ActionObservation], tuple[ActionRecord, ActionObservation]
        ] = {}
        self._late: list[tuple[ActionRecord, ActionObservation]] = []
        self._cancelled: set[asyncio.Task[ActionObservation]] = set()
        self._closed = False
        self._shutdown = asyncio.Event()
        self.late_observations_ready = asyncio.Event()

    def _action(self, identifier: str) -> ActionRecord:
        item = next((r for r in self.store.view().records if r.id == identifier), None)
        if not isinstance(item, ActionRecord):
            raise ValueError("Unknown action")
        return item

    def _commit(
        self,
        action: ActionRecord,
        disposition: Disposition,
        reason: str,
        *,
        writes: tuple[PersistentRecord, ...] = (),
        policy: CommitBatch | None = None,
        queue: bool = False,
        details: Mapping[str, Any] | None = None,
        evidence: tuple[Ref, ...] = (),
        receipt_reads: tuple[Ref, ...] | None = None,
    ) -> DecisionReceipt:
        state = self.store.view()
        goal = next(r for r in state.records if r.id == action.goal_id)
        assert isinstance(goal, Goal)
        items = {r.id: r for r in writes}
        audits = []
        if policy is not None:
            for event in policy.events:
                if event.event_type == "state.changed":
                    # Obtain typed post-images through the canonical reducer.
                    from .reducers import reduce_event

                    reduced = reduce_event(
                        state, event.model_copy(update={"sequence": state.sequence + 1})
                    )
                    before = {r.id: r for r in state.records}
                    for r in reduced.records:
                        if before.get(r.id) != r:
                            items[r.id] = r
                else:
                    audits.append(event)
        # A newly-created reservation can refer to the proposed action (already
        # present). Both authorization and balances share this canonical bundle.
        event = JournalEvent(
            created_at=self.clock(),
            sequence=state.sequence + 1,
            event_type="state.changed",
            producer_id="action-dispatcher-v1",
            principal_id=goal.owner_id,
            subject=reference(action),
            payload={"records": []},
        )
        receipt = DecisionReceipt(
            created_at=self.clock(),
            subject_id=action.id,
            disposition=disposition,
            reason_code=reason,
            state_revisions=tuple(reference(r) for r in state.records)
            if receipt_reads is None
            else receipt_reads,
            evidence=evidence,
            event_ids=(event.id,),
        )
        items[receipt.id] = receipt
        event = event.model_copy(
            update={"payload": {"records": [r.model_dump(mode="json") for r in items.values()]}}
        )
        if details is not None:
            audits.append(
                JournalEvent(
                    created_at=self.clock(),
                    sequence=1,
                    event_type="audit.recorded",
                    producer_id="action-dispatcher-v1",
                    principal_id=goal.owner_id,
                    subject=reference(action),
                    payload={"reason_code": reason, "details": dict(details)},
                )
            )
        self.store.commit(
            CommitBatch(
                events=(event, *audits),
                expected_revisions=tuple(reference(r) for r in state.records),
                outbox=(OutboxEntry(action_id=action.id, available_at=self.clock()),)
                if queue
                else (),
            )
        )
        return cast(
            DecisionReceipt, next(r for r in self.store.view().records if r.id == receipt.id)
        )

    @staticmethod
    def _review_binding(action: ActionRecord) -> tuple[Any, ...]:
        return (
            action.id,
            action.arguments_digest,
            action.arguments,
            action.scope,
            action.constraint_revision,
            action.binding_id,
            action.binding_version,
            action.episode_id,
            action.worker_id,
            action.origin_principal_id,
            action.accepted_task_revision,
            action.accepted_episode_revision,
            action.accepted_fencing_token,
            action.read_set,
            action.idempotency_key,
            action.expires_at,
        )

    def _approved(self, action: ActionRecord) -> bool:
        if action.authorization_revision < 1:
            return False
        goal = next(r for r in self.store.view().records if r.id == action.goal_id)
        assert isinstance(goal, Goal)
        for event in self.store.events():
            if event.event_type != "state.changed" or event.principal_id != goal.owner_id:
                continue
            images = event.payload.get("records", ())
            if not isinstance(images, Sequence) or isinstance(images, str):
                continue
            for image in images:
                if not isinstance(image, Mapping) or image.get("id") != action.id:
                    continue
                if (
                    image.get("status") == "authorized"
                    and image.get("authorization_revision") == action.authorization_revision
                ):
                    reviewed = ActionRecord.model_validate(deep_thaw(image))
                    return self._review_binding(reviewed) == self._review_binding(action)
        return False

    def _admission(self, action: ActionRecord) -> tuple[PolicyDecision, _Binding | None]:
        binding = self.registry._bindings.get(action.capability_id)
        if (
            binding is None
            or action.binding_id != binding.spec.binding_id
            or action.binding_version != binding.spec.binding_version
        ):
            return decision("stale", "capability_binding_changed"), binding
        if action.expires_at is not None and self.clock() >= action.expires_at:
            return decision("stale", "action_expired"), binding
        if action.dispatch_attempts:
            original = None
            for event in self.store.events():
                images = event.model_dump(mode="json")["payload"].get("records", ())
                for image in images:
                    if image.get("id") == action.id and image.get("status") == "dispatching":
                        original = ActionRecord.model_validate(image)
                        break
                if original is not None:
                    break
            if original is None or self._review_binding(original) != self._review_binding(action):
                return decision("stale", "issued_action_binding_changed"), binding
        raw = action.model_dump(mode="json")["arguments"]
        digest = hashlib.sha256(
            json.dumps(
                raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        if digest != action.arguments_digest:
            return decision("deny", "arguments_digest_mismatch"), binding
        try:
            valid = (
                binding.input_validator(action.arguments)
                if binding.input_validator
                else _valid(action.arguments, binding.spec.input_schema)
            )
            if valid is not True:
                return decision("deny", "invalid_action_arguments"), binding
        except Exception:
            return decision("deny", "invalid_action_arguments"), binding
        if (
            action.origin_principal_id is None
            or self.authority.worker_principals.get(action.worker_id or "")
            != action.origin_principal_id
        ):
            return decision("deny", "action_principal_binding_changed"), binding
        authority = Authority(
            rules=self.authority.rules,
            worker_principals=self.authority.worker_principals,
            capabilities=self.registry.specs,
            resource_scopes={
                k: b.resource_scope
                for k, b in self.registry._bindings.items()
                if b.resource_scope is not None
            },
        )
        verdict = authority.check_dispatch(action, self.store.view(), self.clock())
        if verdict.disposition != "authorize":
            return verdict, binding
        if (
            binding.require_approval
            and action.status not in {"proposed", "awaiting_review"}
            and not self._approved(action)
        ):
            return decision("stale", "action_approval_binding_changed"), binding
        if action.constraint_revision != next(
            r.revision for r in self.store.view().records if r.id == action.goal_id
        ):
            return decision("stale", "action_constraints_changed"), binding
        # Read referenced bytes through the real ACL/integrity/version boundary.
        state = self.store.view()
        indexed = records_by_id(state)
        principal = indexed.get(action.origin_principal_id)
        assert isinstance(principal, Principal)
        from .authority import _proof_roots

        task = indexed.get(action.task_id)
        assert isinstance(task, TaskFrame)
        pending = [
            *_proof_roots(task, action.read_set, indexed),
            *(ref for ref in action.read_set if isinstance(indexed.get(ref.id), ArtifactRef)),
        ]
        seen: set[str] = set()
        try:
            while pending:
                ref = pending.pop()
                if ref.id in seen:
                    continue
                seen.add(ref.id)
                item = indexed.get(ref.id)
                if isinstance(item, ArtifactRef):
                    ArtifactStore(self.store).read(
                        item, principal=principal, goal_id=action.goal_id
                    )
                if isinstance(item, Criterion):
                    pending.extend(
                        reference(r)
                        for r in state.records
                        if isinstance(r, VerificationResult)
                        and r.criterion_id == item.id
                        and r.disposition == "accept"
                    )
                if item is not None:
                    for name in ("evidence", "inputs", "required_evidence", "read_set"):
                        pending.extend(getattr(item, name, ()))
        except ExecutiveError:
            return decision("stale", "inaccessible_action_artifact"), binding
        if binding.quote is None or not binding.quote_source or not binding.quote_version:
            return decision("deny", "missing_action_quote"), binding
        quote = binding.quote
        if (
            quote.calls is None
            or quote.calls < 1
            or quote.elapsed_seconds is None
            or quote.elapsed_seconds <= 0
        ):
            return decision("deny", "invalid_action_quote"), binding
        return verdict, binding

    def _reserve(
        self, action: ActionRecord, binding: _Binding, *, retry: bool = False
    ) -> tuple[ActionRecord, CommitBatch]:
        quote = binding.quote
        if (
            quote is None
            or not binding.quote_source
            or not binding.quote_version
            or quote.calls is None
            or quote.calls < 1
            or quote.input_tokens is None
            or quote.output_tokens is None
            or quote.elapsed_seconds is None
            or quote.elapsed_seconds <= 0
        ):
            raise ExecutiveError(
                "invalid_action_quote", "Every call requires a finite versioned maximum"
            )
        state = self.store.view()
        indexed = records_by_id(state)
        goal = indexed[action.goal_id]
        task = indexed[action.task_id]
        assert isinstance(goal, Goal) and isinstance(task, TaskFrame)
        commitment = indexed.get(task.commitment_id)
        budget_id = (
            commitment.budget_id
            if isinstance(commitment, Commitment) and commitment.budget_id
            else goal.budget_id
        )
        if budget_id is None or binding.quote is None:
            raise ExecutiveError(
                "missing_action_budget", "Action requires an owned budget and quote"
            )
        reservation_id = (
            uuid4().hex if retry or action.reservation_id is None else action.reservation_id
        )
        batch = self.budgets.reserve(
            state,
            budget_id,
            binding.quote,
            reservation_id,
            subject_id=action.id,
            trusted_maximum_quote=True,
        )
        updated = action.model_copy(
            update={
                "revision": action.revision + 1,
                "reservation_id": reservation_id,
                "prior_reservation_ids": (*action.prior_reservation_ids, action.reservation_id)
                if retry and action.reservation_id
                else action.prior_reservation_ids,
            }
        )
        return updated, batch

    def _quote_details(self, binding: _Binding) -> Mapping[str, Any]:
        return {
            "quote_source": binding.quote_source,
            "quote_version": binding.quote_version,
            "maximum": binding.quote.model_dump(mode="json") if binding.quote else None,
            "binding_id": binding.spec.binding_id,
            "binding_version": binding.spec.binding_version,
        }

    def _reject(self, action: ActionRecord, verdict: PolicyDecision) -> DecisionReceipt:
        writes: list[PersistentRecord] = []
        policy = None
        if action.status in {"proposed", "awaiting_review", "authorized"}:
            status = (
                "awaiting_review"
                if verdict.disposition == "require_review" and action.status != "authorized"
                else "cancelled"
                if action.status == "authorized"
                else "denied"
            )
            if action.status != status:
                writes.append(
                    action.model_copy(update={"revision": action.revision + 1, "status": status})
                )
            if status == "cancelled" and action.reservation_id and not action.dispatch_attempts:
                policy = self.budgets.settle(
                    self.store.view(),
                    action.reservation_id,
                    Usage.zero(currency=self._currency(action)),
                )
        receipt_reads = None
        if verdict.reason_code == "constraint_conflict":
            from .intake import Intake

            state = self.store.view()
            goal = next(r for r in state.records if r.id == action.goal_id)
            assert isinstance(goal, Goal)
            writes.append(
                goal.model_copy(update={"revision": goal.revision + 1, "status": "blocked"})
            )
            Intake._fence(goal, state, writes, conflict_refs=verdict.evidence)
            receipt_reads = (reference(goal), *verdict.evidence)
        return self._commit(
            action,
            verdict.disposition,
            verdict.reason_code,
            writes=tuple(writes),
            policy=policy,
            evidence=verdict.evidence,
            receipt_reads=receipt_reads,
        )

    async def dispatch_one(self, action_id: str) -> DecisionReceipt:
        action = self._action(action_id)
        if action.status not in {"proposed", "awaiting_review", "authorized"}:
            return self._commit(action, "defer", "action_not_dispatchable")
        if (
            self._closed
            or action_id in self._active_ids
            or len(self._pending) + len(self._late) >= self.max_active
        ):
            return self._commit(action, "defer", "dispatcher_capacity")
        verdict, binding = self._admission(action)
        if verdict.disposition != "authorize":
            return self._reject(action, verdict)
        assert binding is not None
        if binding.require_approval and (
            action.status != "authorized" or action.authorization_revision < 1
        ):
            writes = (
                ()
                if action.status == "awaiting_review"
                else (
                    action.model_copy(
                        update={"revision": action.revision + 1, "status": "awaiting_review"}
                    ),
                )
            )
            return self._commit(action, "require_review", "owner_approval_required", writes=writes)
        try:
            authorized, reservation = self._reserve(action, binding)
        except ExecutiveError as exc:
            return self._reject(action, decision("deny", exc.code))
        selected_grant = next(
            (
                r.id
                for r in verdict.evidence
                if any(
                    g.id == r.id and getattr(g, "kind", None) == "grant"
                    for g in self.store.view().records
                )
            ),
            None,
        )
        authorized = authorized.model_copy(
            update={"status": "authorized", "grant_id": selected_grant}
        )
        self._commit(
            action,
            "authorize",
            "action_authorized",
            writes=(authorized,),
            policy=reservation,
            queue=True,
            details=self._quote_details(binding),
        )
        action = self._action(action_id)
        verdict, binding = self._admission(action)
        if verdict.disposition != "authorize":
            return self._reject(action, verdict)
        assert binding is not None
        try:
            # Reusing a reservation rechecks all ancestors and observed overages.
            _, reservation = self._reserve(action, binding)
        except ExecutiveError as exc:
            return self._reject(action, decision("deny", exc.code))
        issued = action.model_copy(
            update={
                "revision": action.revision + 1,
                "status": "dispatching",
                "dispatched_at": self.clock(),
                "dispatch_attempts": 1,
            }
        )
        self._commit(
            action, "authorize", "action_dispatching", writes=(issued,), policy=reservation
        )
        return await self._call(issued, binding, "execute")

    def _unknown(self, issued: ActionRecord) -> ActionObservation:
        return ActionObservation(
            action_id=issued.id,
            expected_revision=issued.revision,
            outcome="outcome_unknown",
            usage=Usage(currency=self._currency(issued)),
        )

    def _currency(self, action: ActionRecord) -> str:
        reservation = next(
            (r for r in self.store.view().records if r.id == action.reservation_id), None
        )
        return getattr(getattr(reservation, "maximum", None), "currency", "USD")

    def _cancel(self, task: asyncio.Task[ActionObservation]) -> None:
        if not task.done() and task not in self._cancelled:
            self._cancelled.add(task)
            task.cancel()

    def _finished(self, task: asyncio.Task[ActionObservation]) -> None:
        self._pending.discard(task)
        self._cancelled.discard(task)
        detached = self._detached.pop(task, None)
        if detached is not None:
            issued, fallback = detached
            self._late.append((issued, self._observed(task, issued, fallback)))
            self.late_observations_ready.set()
        if not task.cancelled():
            task.exception()

    def _observed(
        self,
        task: asyncio.Task[ActionObservation],
        issued: ActionRecord,
        fallback: ActionObservation | None = None,
    ) -> ActionObservation:
        if task.done() and not task.cancelled():
            try:
                observation = task.result()
                if (
                    isinstance(observation, ActionObservation)
                    and observation.action_id == issued.id
                    and observation.expected_revision == issued.revision
                ):
                    return observation
                if isinstance(observation, ActionObservation):
                    return (fallback or self._unknown(issued)).model_copy(
                        update={
                            "usage": observation.usage.model_copy(update={"status": "unknown"}),
                        }
                    )
            except Exception:
                pass
        return fallback or self._unknown(issued)

    async def _call(self, issued: ActionRecord, binding: _Binding, method: str) -> DecisionReceipt:
        self._active_ids.add(issued.id)
        fallback = self._unknown(issued)
        # Deadline tasks are independent of child cancellation completion.
        timeout = binding.quote.elapsed_seconds if binding.quote else None
        assert timeout is not None
        deadline = self.clock() + timedelta(seconds=timeout)
        if issued.expires_at is not None and method == "execute":
            deadline = min(deadline, issued.expires_at)
        if (
            method == "execute"
            and issued.dispatch_attempts > 1
            and issued.dispatched_at is not None
            and binding.spec.idempotency_retention_seconds is not None
        ):
            deadline = min(
                deadline,
                issued.dispatched_at
                + timedelta(seconds=binding.spec.idempotency_retention_seconds),
            )

        async def invoke() -> ActionObservation:
            # Registration changes between task creation and actual entry never
            # invoke a replaced adapter. The durable dispatch remains uncertain.
            if (
                self.registry._bindings.get(issued.capability_id) is not binding
                or self.clock() >= deadline
            ):
                return self._unknown(issued)
            if method == "execute":
                return await binding.adapter.execute(issued)
            return await binding.adapter.reconcile(issued)

        task = asyncio.create_task(invoke())
        self._pending.add(task)
        task.add_done_callback(self._finished)

        async def wait_deadline() -> None:
            await self.wait_until(deadline)

        timer = asyncio.create_task(wait_deadline())

        async def wait_shutdown() -> None:
            await self._shutdown.wait()

        shutdown = asyncio.create_task(wait_shutdown())
        try:
            completed, _ = await asyncio.wait(
                (task, timer, shutdown), return_when=asyncio.FIRST_COMPLETED
            )
            if task in completed and not self._closed:
                observation = self._observed(task, issued)
            else:
                observation = self._unknown(issued)
                if not task.done():
                    self._detached[task] = (issued, fallback)
                    self._cancel(task)
                else:
                    # Even deadline/completion races preserve actual usage/effect.
                    observation = self._observed(task, issued)
            return self.record_observation(issued, observation)
        except asyncio.CancelledError:
            observation = self._observed(task, issued)
            if not task.done():
                self._detached[task] = (issued, fallback)
                self._cancel(task)
            self.record_observation(issued, observation)
            raise
        finally:
            timer.cancel()
            shutdown.cancel()
            timer.add_done_callback(_drain)
            shutdown.add_done_callback(_drain)
            self._active_ids.discard(issued.id)

    def record_observation(
        self, issued: ActionRecord, observation: ActionObservation
    ) -> DecisionReceipt:
        """Commit an adapter/late observation on the owning writer, never resume work."""
        current = self._action(issued.id)
        if observation.action_id != issued.id or observation.expected_revision != issued.revision:
            observation = self._unknown(issued).model_copy(
                update={
                    "usage": observation.usage.model_copy(update={"status": "unknown"}),
                }
            )
        # Preserve accounting for old attempts without overwriting a final outcome.
        policy = None
        accounting_reason = None
        if issued.reservation_id:
            policy = self.budgets.observe_usage(
                self.store.view(), issued.reservation_id, observation.usage
            )
            accounting_reason = next(
                (
                    str(event.payload["reason_code"])
                    for event in policy.events
                    if event.event_type == "audit.recorded"
                    and event.payload.get("reason_code")
                    in {"settlement_conflict", "currency_mismatch", "accounting_conflict"}
                ),
                None,
            )
        terminal = current.status in {"succeeded", "failed"}
        older_attempt = issued.reservation_id != current.reservation_id
        outcome = observation.outcome
        if outcome == "needs_human" and current.status == "dispatching":
            outcome = "outcome_unknown"
        if outcome == "outcome_unknown" and current.status in {"reconciling", "needs_human"}:
            outcome = "needs_human"
        if current.status == "outcome_unknown" and outcome == "needs_human":
            outcome = "outcome_unknown"
        writes: tuple[PersistentRecord, ...] = ()
        if not terminal and not older_attempt:
            writes = (
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "status": outcome,
                        "external_receipt": observation.external_receipt
                        or current.external_receipt,
                        "evidence": tuple(
                            dict.fromkeys((*current.evidence, *observation.evidence))
                        ),
                    }
                ),
            )
        return self._commit(
            current,
            "require_review"
            if accounting_reason
            else "accept"
            if outcome in {"succeeded", "failed"} and not older_attempt
            else "defer",
            accounting_reason
            or (
                "late_attempt_observation"
                if older_attempt
                else "action_observed"
                if not terminal
                else "late_action_observation"
            ),
            writes=writes,
            policy=policy,
            details={
                "outcome": observation.outcome,
                "issued_revision": issued.revision,
                "accounting_reason": accounting_reason,
                "usage": observation.usage.model_dump(mode="json"),
                "external_receipt": observation.external_receipt,
            },
        )

    async def reconcile_one(self, action_id: str) -> DecisionReceipt:
        action = self._action(action_id)
        if (
            action_id in self._active_ids
            or any(a.id == action_id for a, _ in self._detached.values())
            or any(a.id == action_id for a, _ in self._late)
        ):
            return self._commit(action, "defer", "action_in_flight")
        if action.status not in {"dispatching", "outcome_unknown", "reconciling", "needs_human"}:
            return self._commit(action, "defer", "action_not_unresolved")
        if action.status == "dispatching":
            self.record_observation(action, self._unknown(action))
            action = self._action(action_id)
        if self._closed or len(self._pending) + len(self._late) >= self.max_active:
            return self._commit(action, "defer", "dispatcher_capacity")
        binding = self.registry._bindings.get(action.capability_id)
        same_binding = (
            binding is not None
            and action.binding_id == binding.spec.binding_id
            and action.binding_version == binding.spec.binding_version
        )
        if action.status == "outcome_unknown":
            reconciling = action.model_copy(
                update={"revision": action.revision + 1, "status": "reconciling"}
            )
            self._commit(action, "defer", "action_reconciling", writes=(reconciling,))
            action = self._action(action_id)
        if same_binding and binding is not None and binding.spec.reconciliation == "lookup":
            # A no-effect lookup preserves paused/cancelled goals. Its own call
            # still consumes a separately admitted maximum; old uncertainty stays.
            try:
                lookup, reservation = self._reserve(action, binding, retry=True)
                self._commit(
                    action,
                    "authorize",
                    "action_lookup_reserved",
                    writes=(lookup,),
                    policy=reservation,
                    details=self._quote_details(binding),
                )
                return await self._call(lookup, binding, "reconcile")
            except ExecutiveError:
                pass
        if (
            same_binding
            and binding is not None
            and action.status == "reconciling"
            and binding.spec.idempotency == "durable"
            and action.dispatched_at is not None
            and binding.spec.idempotency_retention_seconds is not None
            and self.clock()
            < action.dispatched_at + timedelta(seconds=binding.spec.idempotency_retention_seconds)
            and action.dispatch_attempts <= self.max_retries
        ):
            verdict, _ = self._admission(action)
            if verdict.disposition == "authorize":
                try:
                    retry, reservation = self._reserve(action, binding, retry=True)
                    retry = retry.model_copy(
                        update={"dispatch_attempts": action.dispatch_attempts + 1}
                    )
                    self._commit(
                        action,
                        "authorize",
                        "durable_idempotent_retry",
                        writes=(retry,),
                        policy=reservation,
                        queue=True,
                        details=self._quote_details(binding),
                    )
                    return await self._call(retry, binding, "execute")
                except ExecutiveError:
                    pass
        human = action.model_copy(update={"revision": action.revision + 1, "status": "needs_human"})
        return self._commit(
            action, "require_review", "effect_requires_human_reconciliation", writes=(human,)
        )

    def take_late_observations(self) -> tuple[tuple[ActionRecord, ActionObservation], ...]:
        result = tuple(self._late)
        self._late.clear()
        self.late_observations_ready.clear()
        return result

    async def close(self) -> None:
        self._closed = True
        self._shutdown.set()
        for task in self._pending:
            self._cancel(task)


def _drain(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
