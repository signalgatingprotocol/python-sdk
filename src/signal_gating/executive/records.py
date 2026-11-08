"""Versioned executive state and supporting value objects.

Only declared JSON boundaries accept arbitrary payloads. All containers are copied
and frozen after validation; model_copy validates updates just like construction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from math import isfinite
from re import fullmatch
from typing import Annotated, Any, Literal, TypeVar
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    model_validator,
)

from signal_gating._immutable import deep_freeze, deep_thaw


def _version(value: object) -> object:
    if type(value) is not int or value != 1:
        raise ValueError("unsupported schema version; expected integer 1")
    return value


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must have a UTC offset")
    return value.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


SchemaVersion = Annotated[Literal[1], BeforeValidator(_version)]
Identifier = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{32}$")]
NonEmpty = Annotated[str, Field(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegativeNumber = Annotated[FiniteNumber, Field(ge=0)]
PositiveNumber = Annotated[FiniteNumber, Field(gt=0)]
Timestamp = Annotated[datetime, AfterValidator(_utc)]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Currency = Annotated[str, Field(strict=True, pattern=r"^[A-Z]{3}$")]
JsonObject = Mapping[str, JsonValue]
GoalStatus = Literal["active", "paused", "blocked", "completed", "cancelled", "failed"]
TaskStatus = Literal[
    "pending",
    "ready",
    "running",
    "verifying",
    "completed",
    "suspended",
    "blocked",
    "failed",
    "cancelled",
]
ActionStatus = Literal[
    "proposed",
    "denied",
    "awaiting_review",
    "authorized",
    "dispatching",
    "succeeded",
    "failed",
    "outcome_unknown",
    "reconciling",
    "needs_human",
    "cancelled",
]
Disposition = Literal[
    "process",
    "defer",
    "coalesce",
    "archive",
    "admit",
    "revise",
    "dispute",
    "reject",
    "request_evidence",
    "authorize",
    "deny",
    "require_review",
    "stale",
    "accept",
    "insufficient_evidence",
    "contradicted",
    "needs_human",
]
M = TypeVar("M", bound="Model")


def _check_finite(value: object) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("non-finite numbers are not JSON values")
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_finite(item)


class Model(BaseModel):
    """Immutable, versioned data with closed schemas and validated copies."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True, allow_inf_nan=False
    )
    schema_version: SchemaVersion = 1

    @field_serializer("*", mode="wrap")
    def _serialize(self, value: Any, handler: Any) -> Any:
        return handler(deep_thaw(value))

    @model_validator(mode="after")
    def _freeze(self) -> Model:
        for name in type(self).model_fields:
            value = getattr(self, name)
            _check_finite(value)
            object.__setattr__(self, name, deep_freeze(value))
        return self

    def model_copy(self: M, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> M:
        # Every field is immutable; a validated reconstruction also prevents the
        # usual Pydantic model_copy(update=...) validation bypass.
        values = self.model_dump(mode="python")
        values.update(update or {})
        return type(self).model_validate(values)


class Record(Model):
    id: Identifier = Field(default_factory=lambda: uuid4().hex)
    revision: PositiveInt = 1
    created_at: Timestamp = Field(default_factory=utc_now)


class Ref(Model):
    id: NonEmpty
    revision: PositiveInt = 1
    kind: Literal[
        "record",
        "goal",
        "task",
        "belief",
        "artifact",
        "criterion",
        "grant",
        "budget",
        "action",
        "draft",
    ] = "record"

    @model_validator(mode="after")
    def _reference_identity(self) -> Ref:
        if self.id.startswith("draft:"):
            if fullmatch(r"draft:[A-Za-z0-9_.-]+", self.id) is None:
                raise ValueError("invalid episode-local draft reference")
        elif self.kind == "draft" or fullmatch(r"[0-9a-f]{32}", self.id) is None:
            raise ValueError("record references require UUID hex IDs")
        return self


class Scope(Model):
    owner_id: Identifier | None = None
    goal_id: Identifier | None = None
    principal_ids: tuple[Identifier, ...] = ()
    resource_paths: tuple[NonEmpty, ...] = ()


class Constraint(Record):
    kind: Literal["constraint"] = "constraint"
    description: NonEmpty
    enforcement: Literal["rule", "verifier", "human_review", "unresolved"] = "unresolved"
    enforcement_id: NonEmpty | None = None
    hard: bool = True

    @model_validator(mode="after")
    def _mechanism(self) -> Constraint:
        if self.enforcement in ("rule", "verifier") and self.enforcement_id is None:
            raise ValueError("rule and verifier constraints require enforcement_id")
        return self


class Criterion(Record):
    kind: Literal["criterion"] = "criterion"
    description: NonEmpty
    evaluator_type: Literal["deterministic", "model", "human"] = "deterministic"
    verifier_id: NonEmpty
    inputs: tuple[Ref, ...] = ()
    acceptance_rule: NonEmpty
    # Owner-declared local-check binding; workers cannot choose a smaller input set.
    check_id: NonEmpty | None = None
    workspace_inputs: tuple[NonEmpty, ...] = ()
    required_evidence: tuple[Ref, ...] = ()
    required: bool = True


class Principal(Record):
    kind: Literal["principal"] = "principal"
    name: NonEmpty
    trust_domain: NonEmpty
    authenticated_origin: NonEmpty
    roles: tuple[Literal["owner", "operator", "worker", "verifier", "producer"], ...] = ()


class Limits(Model):
    """Explicit resource maxima. A missing maximum is unbounded, never zero."""

    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    calls: NonNegativeInt | None = None
    elapsed_seconds: NonNegativeNumber | None = None
    cost_microunits: NonNegativeInt | None = None
    currency: Currency = "USD"


class Usage(Model):
    """Unknown quantities stay unknown until reconciled, even with partial data."""

    status: Literal["known", "estimated", "unknown"] = "unknown"
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    calls: NonNegativeInt | None = None
    elapsed_seconds: NonNegativeNumber | None = None
    cost_microunits: NonNegativeInt | None = None
    currency: Currency = "USD"
    source: NonEmpty | None = None

    @classmethod
    def zero(cls, *, currency: str = "USD") -> Usage:
        """Known zero use, only for a scope in which no work has occurred."""
        return cls(
            status="known",
            input_tokens=0,
            output_tokens=0,
            calls=0,
            elapsed_seconds=0,
            cost_microunits=0,
            currency=currency,
        )

    @model_validator(mode="after")
    def _known_quantities(self) -> Usage:
        if self.status == "known" and any(
            value is None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.calls,
                self.elapsed_seconds,
                self.cost_microunits,
            )
        ):
            raise ValueError("known usage requires all resource quantities")
        return self


class Goal(Record):
    kind: Literal["goal"] = "goal"
    owner_id: Identifier
    outcome: NonEmpty
    worker_id: NonEmpty
    limits: Limits
    constraints: tuple[Constraint, ...] = ()
    criteria: tuple[Criterion, ...] = ()
    required_commitment_ids: tuple[Identifier, ...] = ()
    status: GoalStatus = "active"
    parent_id: Identifier | None = None
    budget_id: Identifier | None = None
    priority: NonNegativeInt = 0
    deadline: Timestamp | None = None


def goal_definition_digest(goal: Goal) -> str:
    """Versioned owner intent fingerprint; lifecycle bookkeeping is excluded."""
    value = goal.model_dump(mode="json", exclude={"revision", "status"})
    encoded = json.dumps(
        {"policy": "goal-definition-v1", "goal": value},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CriterionApproval(Record):
    """Explicit owner control, never inferred from artifact or signal contents."""

    kind: Literal["criterion_approval"] = "criterion_approval"
    owner_id: Identifier
    goal: Ref
    criterion: Ref
    goal_definition_digest: Digest
    artifacts: tuple[Ref, ...] = Field(min_length=1)
    explanation: NonEmpty


class Commitment(Record):
    kind: Literal["commitment"] = "commitment"
    goal_id: Identifier
    responsible_principal_id: Identifier
    required: bool = True
    dependencies: tuple[Ref, ...] = ()
    deadline: Timestamp | None = None
    budget_id: Identifier | None = None
    status: TaskStatus = "pending"


class Belief(Record):
    kind: Literal["belief"] = "belief"
    claim: NonEmpty
    scope: Scope
    evidence: tuple[Ref, ...] = ()
    epistemic_status: Literal["supported", "hypothesis", "disputed", "retracted"] = "hypothesis"
    valid_from: Timestamp | None = None
    valid_until: Timestamp | None = None
    supersedes: tuple[Ref, ...] = ()
    confidence: Annotated[FiniteNumber, Field(ge=0, le=1)] | None = None
    confidence_source: NonEmpty | None = None
    calibration_status: Literal["uncalibrated", "calibrated", "unknown"] = "unknown"

    @model_validator(mode="after")
    def _interval_and_confidence(self) -> Belief:
        if self.valid_from and self.valid_until and self.valid_until < self.valid_from:
            raise ValueError("valid_until precedes valid_from")
        if self.confidence is not None and self.confidence_source is None:
            raise ValueError("confidence requires its source")
        return self


class WakeCondition(Model):
    kind: Literal["time", "dependency_change", "new_evidence", "human_decision", "budget"]
    at: Timestamp | None = None
    refs: tuple[Ref, ...] = ()
    reason_code: NonEmpty

    @model_validator(mode="after")
    def _wake_time(self) -> WakeCondition:
        if self.kind == "time" and self.at is None:
            raise ValueError("time wake requires at")
        return self


class TaskFrame(Record):
    kind: Literal["task"] = "task"
    goal_id: Identifier
    commitment_id: Identifier
    subgoal: NonEmpty
    status: TaskStatus = "ready"
    progress_artifacts: tuple[Ref, ...] = ()
    open_questions: tuple[NonEmpty, ...] = ()
    dependencies: tuple[Ref, ...] = ()
    pending_action_ids: tuple[Identifier, ...] = ()
    resume_brief: str = ""
    wake_condition: WakeCondition | None = None
    fencing_token: NonNegativeInt = 0
    failure_count: NonNegativeInt = 0
    priority: NonNegativeInt = 0
    deadline: Timestamp | None = None
    urgent: bool = False
    eligible_since: Timestamp | None = None
    last_scheduled_at: Timestamp | None = None
    failure_signature: Digest | None = None
    failure_reason: NonEmpty | None = None


class Grant(Record):
    kind: Literal["grant"] = "grant"
    issuer_id: Identifier
    subject_id: Identifier
    capabilities: tuple[NonEmpty, ...]
    scope: Scope
    restrictions: tuple[Constraint, ...] = ()
    expires_at: Timestamp | None = None
    revoked: bool = False
    revocation_revision: NonNegativeInt = 0


class Budget(Record):
    kind: Literal["budget"] = "budget"
    scope: Scope
    limits: Limits
    reserved: Usage = Field(default_factory=Usage.zero)
    actual: Usage = Field(default_factory=Usage.zero)
    accounting_uncertain: bool = False
    parent_id: Identifier | None = None

    @model_validator(mode="before")
    @classmethod
    def _initial_balances(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            values = dict(value)
            limits = values.get("limits")
            currency = (
                limits.currency
                if isinstance(limits, Limits)
                else limits.get("currency", "USD")
                if isinstance(limits, Mapping)
                else "USD"
            )
            for name in ("reserved", "actual"):
                if name not in values:
                    values[name] = Usage.zero(currency=currency)
            return values
        return value

    @model_validator(mode="after")
    def _accounting_currency(self) -> Budget:
        if (
            self.reserved.currency != self.limits.currency
            or self.actual.currency != self.limits.currency
        ):
            raise ValueError("budget balances must use the budget currency")
        if (
            "unknown" in (self.reserved.status, self.actual.status)
            and not self.accounting_uncertain
        ):
            raise ValueError("unknown balances require explicit accounting uncertainty")
        return self


class Reservation(Record):
    kind: Literal["reservation"] = "reservation"
    budget_id: Identifier
    subject_id: Identifier
    maximum: Limits
    trusted_maximum_quote: bool = False
    usage: Usage = Field(default_factory=Usage)
    # Preserve final settlement while retaining a later conservative measured floor.
    observed_usage: Usage | None = None
    accounting_conflict: NonEmpty | None = None
    status: Literal["reserved", "settled", "uncertain", "released"] = "reserved"
    evidence: tuple[Ref, ...] = ()


class ArtifactRef(Record):
    kind: Literal["artifact"] = "artifact"
    digest: Digest
    storage_location: NonEmpty
    media_type: NonEmpty
    origin: NonEmpty
    scope: Scope
    retention_class: Literal["persistent", "ephemeral", "audit"] = "persistent"
    size_bytes: NonNegativeInt
    shared_with: tuple[Identifier, ...] = ()
    valid: bool = True


class ExecutiveSignal(Record):
    kind: Literal["signal"] = "signal"
    signal_kind: NonEmpty
    principal_id: Identifier
    producer_id: NonEmpty
    external_id: NonEmpty | None = None
    producer_sequence: NonNegativeInt | None = None
    received_at: Timestamp = Field(default_factory=utc_now)
    observed_at: Timestamp | None = None
    causal_ids: tuple[Identifier, ...] = ()
    scope: Scope
    payload: JsonObject = Field(default_factory=dict)
    artifacts: tuple[Ref, ...] = ()


class DecisionReceipt(Record):
    kind: Literal["receipt"] = "receipt"
    subject_id: NonEmpty
    state_revisions: tuple[Ref, ...] = ()
    policy_version: NonEmpty = "deterministic-v1"
    disposition: Disposition
    reason_code: NonEmpty
    evidence: tuple[Ref, ...] = ()
    event_ids: tuple[Identifier, ...] = ()
    wake_condition: WakeCondition | None = None


class CapabilitySpec(Model):
    id: NonEmpty
    binding_version: NonEmpty
    description: str = ""
    # Operator registration identity changes even if a version string is reused.
    binding_id: Identifier | None = None
    input_schema: JsonObject
    effect_class: Literal["read_only", "reversible", "consequential", "human_only"]
    resource_scope_rules: tuple[NonEmpty, ...] = ()
    cancellation: Literal["unsupported", "best_effort", "guaranteed"] = "unsupported"
    idempotency: Literal["none", "durable"] = "none"
    idempotency_scope: NonEmpty | None = None
    idempotency_retention_seconds: PositiveNumber | None = None
    reconciliation: Literal["none", "lookup", "human"] = "none"
    usage_accounting: Literal["known", "estimated", "unknown"] = "unknown"

    @model_validator(mode="after")
    def _durable_scope(self) -> CapabilitySpec:
        if self.idempotency == "durable" and (
            self.idempotency_scope is None or self.idempotency_retention_seconds is None
        ):
            raise ValueError("durable idempotency requires scope and retention")
        return self


class ActionRecord(Record):
    kind: Literal["action"] = "action"
    goal_id: Identifier
    task_id: Identifier
    capability_id: NonEmpty
    binding_version: NonEmpty
    binding_id: Identifier | None = None
    # Trusted acceptance lineage, never inferred from a worker name or payload role.
    episode_id: Identifier | None = None
    worker_id: NonEmpty | None = None
    origin_principal_id: Identifier | None = None
    accepted_task_revision: PositiveInt | None = None
    accepted_episode_revision: PositiveInt | None = None
    accepted_fencing_token: NonNegativeInt | None = None
    dispatched_at: Timestamp | None = None
    dispatch_attempts: NonNegativeInt = 0
    prior_reservation_ids: tuple[Identifier, ...] = ()
    arguments_digest: Digest
    arguments: JsonObject = Field(default_factory=dict)
    arguments_ref: Ref | None = None
    scope: Scope
    authorization_revision: NonNegativeInt = 0
    constraint_revision: PositiveInt = 1
    grant_id: Identifier | None = None
    reservation_id: Identifier | None = None
    status: ActionStatus = "proposed"
    idempotency_key: NonEmpty
    read_set: tuple[Ref, ...] = ()
    expires_at: Timestamp | None = None
    external_receipt: NonEmpty | None = None
    evidence: tuple[Ref, ...] = ()


class Episode(Record):
    kind: Literal["episode"] = "episode"
    task_id: Identifier
    fencing_token: PositiveInt
    context_packet_id: Identifier
    worker_id: NonEmpty
    provider: NonEmpty
    model: NonEmpty
    allowed_proposal_types: tuple[
        Literal[
            "belief_change",
            "task_decomposition",
            "action_intent",
            "evidence_request",
            "completion_claim",
        ],
        ...,
    ]
    deadline: Timestamp
    reservation_id: Identifier | None = None
    usage: Usage = Field(default_factory=Usage)
    status: Literal["running", "accepted", "rejected", "expired", "fenced"] = "running"


class PolicyDecision(Model):
    gate: Literal["attention", "memory", "execution", "completion"]
    disposition: Disposition
    reason_code: NonEmpty
    policy_version: NonEmpty = "deterministic-v1"
    evidence: tuple[Ref, ...] = ()
    wake_condition: WakeCondition | None = None

    @model_validator(mode="after")
    def _gate_disposition(self) -> PolicyDecision:
        allowed = {
            "attention": {"process", "defer", "coalesce", "archive"},
            "memory": {"admit", "revise", "dispute", "reject", "request_evidence"},
            "execution": {"authorize", "deny", "require_review", "stale"},
            "completion": {"accept", "insufficient_evidence", "contradicted", "needs_human"},
        }
        if self.disposition not in allowed[self.gate]:
            raise ValueError("disposition does not belong to the selected gate")
        if self.disposition == "defer" and self.wake_condition is None:
            raise ValueError("deferred items require a wake condition")
        return self


class ContextManifest(Model):
    required_ids: tuple[Identifier, ...] = ()
    visible_ids: tuple[Identifier, ...] = ()
    context_token_limit: PositiveInt | None = None
    included: tuple[Ref, ...] = ()
    omitted: tuple[Ref, ...] = ()
    # Read-only status snapshots fenced at packet admission, never action proof.
    observed_refs: tuple[Ref, ...] = ()
    selection_reasons: tuple[NonEmpty, ...] = ()
    required_tokens: NonNegativeInt = 0
    input_tokens: NonNegativeInt = 0
    output_tokens_reserved: NonNegativeInt = 2048
    framing_tokens_reserved: NonNegativeInt = 0
    counter_id: NonEmpty = "utf8-bytes-v1"
    estimated: bool = True


class ContextPacket(Record):
    kind: Literal["context_packet"] = "context_packet"
    goal: Goal
    task: TaskFrame
    episode_id: Identifier
    fencing_token: PositiveInt
    commitments: tuple[Commitment, ...] = ()
    beliefs: tuple[Belief, ...] = ()
    changes: tuple[Ref, ...] = ()
    contradictions: tuple[Ref, ...] = ()
    pending_effects: tuple[ActionRecord, ...] = ()
    capabilities: tuple[CapabilitySpec, ...] = ()
    remaining_limits: Limits
    mandatory_read_set: tuple[Ref, ...] = ()
    manifest: ContextManifest
    rendered_input: str
    rendered_artifact: ArtifactRef | None = None
    worker_instructions: str | None = None


class ArtifactDraft(Model):
    key: Annotated[str, Field(strict=True, min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")]
    media_type: NonEmpty
    text: str

    @model_validator(mode="after")
    def _utf8(self) -> ArtifactDraft:
        try:
            self.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("draft must be valid UTF-8") from exc
        # Staging enforces the operator-configured per-artifact and aggregate
        # byte ceilings before persistence. A worker cannot choose those limits.
        return self


class ActionObservation(Model):
    action_id: Identifier
    expected_revision: PositiveInt
    outcome: Literal["succeeded", "failed", "outcome_unknown", "needs_human"]
    evidence: tuple[Ref, ...] = ()
    external_receipt: NonEmpty | None = None
    usage: Usage = Field(default_factory=Usage)
    explanation: str = ""


class CriterionGap(Model):
    criterion_id: Identifier
    reason_code: NonEmpty
    description: NonEmpty
    evidence_needed: tuple[Ref, ...] = ()


class VerificationResult(Record):
    kind: Literal["verification"] = "verification"
    criterion_id: Identifier
    verifier_id: NonEmpty
    disposition: Literal["accept", "insufficient_evidence", "contradicted", "needs_human"]
    reason_code: NonEmpty
    read_set: tuple[Ref, ...] = ()
    evidence: tuple[Ref, ...] = ()
    gaps: tuple[CriterionGap, ...] = ()
    policy_version: NonEmpty = "deterministic-v1"
    usage: Usage = Field(default_factory=Usage)


class JournalEvent(Record):
    kind: Literal["event"] = "event"
    sequence: PositiveInt
    event_type: NonEmpty
    producer_id: NonEmpty
    principal_id: Identifier
    causal_ids: tuple[Identifier, ...] = ()
    subject: Ref
    # The versioned reducer for event_type validates this declared wire boundary.
    payload: JsonObject = Field(default_factory=dict)


class SchedulerState(Record):
    """One journalled fairness cursor and bounded control-backlog observation."""

    kind: Literal["scheduler"] = "scheduler"
    id: Literal["d3f108b20fa446218be896319d57243e"] = "d3f108b20fa446218be896319d57243e"
    cursor: NonNegativeInt = 0
    control_pending: NonNegativeInt = 0
    control_since: Timestamp | None = None
    overload_reported: bool = False


PersistentRecord = Annotated[
    Principal
    | Goal
    | Commitment
    | Belief
    | TaskFrame
    | Grant
    | Budget
    | Reservation
    | ArtifactRef
    | ExecutiveSignal
    | DecisionReceipt
    | ActionRecord
    | Episode
    | Constraint
    | Criterion
    | CriterionApproval
    | VerificationResult
    | JournalEvent
    | ContextPacket
    | SchedulerState,
    Field(discriminator="kind"),
]


class StateView(Model):
    sequence: NonNegativeInt = 0
    records: tuple[PersistentRecord, ...] = ()


class RunSummary(Model):
    completed_goal_ids: tuple[Identifier, ...] = ()
    blocked_task_ids: tuple[Identifier, ...] = ()
    pending_review_ids: tuple[Identifier, ...] = ()
    unknown_action_ids: tuple[Identifier, ...] = ()
    usage: Usage = Field(default_factory=Usage)
    stop_reason: Literal["idle", "stopped", "eof"] = "idle"


class KernelView(Model):
    state: StateView = Field(default_factory=StateView)
    summary: RunSummary = Field(default_factory=RunSummary)
    goal_id: Identifier | None = None


class EvaluationCase(Record):
    kind: Literal["evaluation_case"] = "evaluation_case"
    name: NonEmpty
    description: str = ""
    initial_state: StateView
    inputs: tuple[ExecutiveSignal, ...] = ()
    expected_dispositions: tuple[Disposition, ...] = ()
    expected_completed_goal_ids: tuple[Identifier, ...] = ()


class EvaluationResult(Model):
    # Optional measured fixture projection. passed/summary retain their meanings.
    metrics: dict[str, int] = Field(default_factory=dict)
    scope: Literal["synthetic_fixture", "observed_run"] = "synthetic_fixture"
    trial: NonNegativeInt = 0
    arm: str = ""
    store_path: str | None = None
    measurements: dict[str, NonNegativeInt | NonNegativeNumber | None] = Field(default_factory=dict)
    usage_observations: tuple[Usage, ...] = ()

    case_id: Identifier
    policy_version: NonEmpty
    passed: bool
    receipts: tuple[DecisionReceipt, ...] = ()
    summary: RunSummary
    failures: tuple[NonEmpty, ...] = ()

    @property
    def completed(self) -> bool:
        return self.passed and bool(self.summary.completed_goal_ids)


class EvaluationReport(Model):
    results: tuple[EvaluationResult, ...]
    scope: Literal["synthetic_fixture", "observed_run"] = "synthetic_fixture"
    policy_versions: tuple[NonEmpty, ...] = ()
    arm: str = ""
    configuration_digest: str = ""
    manifest_digest: str = ""
    evaluation_kind: Literal["fresh", "replay"] = "fresh"
    case_names: dict[str, str] = Field(default_factory=dict)

    @property
    def attempts(self) -> int:
        return len(self.results)

    @property
    def accepted(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def worker_calls(self) -> int:
        return sum(r.metrics.get("worker_calls", 0) for r in self.results)

    @property
    def control_calls(self) -> int:
        return sum(r.metrics.get("control_calls", 0) for r in self.results)

    @property
    def verifier_calls(self) -> int:
        return sum(r.metrics.get("verifier_calls", 0) for r in self.results)

    @property
    def total_calls(self) -> int:
        return self.worker_calls + self.control_calls + self.verifier_calls

    @property
    def cost_status(self) -> Literal["known", "unknown"]:
        return (
            "known"
            if self.results
            and all(r.measurements.get("cost_microunits") is not None for r in self.results)
            else "unknown"
        )
