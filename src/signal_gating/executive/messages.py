"""Closed command and proposal unions for executive state transitions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .records import (
    ArtifactDraft,
    Belief,
    Commitment,
    Constraint,
    Criterion,
    DecisionReceipt,
    Digest,
    ExecutiveSignal,
    Goal,
    Grant,
    Identifier,
    JournalEvent,
    JsonObject,
    Limits,
    Model,
    NonEmpty,
    PersistentRecord,
    PositiveInt,
    Principal,
    Record,
    Ref,
    Scope,
    TaskFrame,
    Timestamp,
    Usage,
    WakeCondition,
    utc_now,
)


class CommandBase(Record):
    producer_id: NonEmpty = "local"
    external_id: NonEmpty | None = None
    causal_ids: tuple[Identifier, ...] = ()


class CreateGoal(CommandBase):
    kind: Literal["create_goal"] = "create_goal"
    goal: Goal
    commitments: tuple[Commitment, ...] = ()
    tasks: tuple[TaskFrame, ...] = ()


class ReviseGoal(CommandBase):
    kind: Literal["revise_goal"] = "revise_goal"
    goal_id: Identifier
    expected_revision: PositiveInt
    outcome: NonEmpty | None = None
    constraints: tuple[Constraint, ...] | None = None
    criteria: tuple[Criterion, ...] | None = None
    limits: Limits | None = None
    worker_id: NonEmpty | None = None
    explanation: str = ""


class GoalControl(CommandBase):
    goal_id: Identifier
    expected_revision: PositiveInt
    explanation: str = ""


class PauseGoal(GoalControl):
    kind: Literal["pause_goal"] = "pause_goal"


class ResumeGoal(GoalControl):
    kind: Literal["resume_goal"] = "resume_goal"


class CancelGoal(GoalControl):
    kind: Literal["cancel_goal"] = "cancel_goal"


class ResolveConflict(CommandBase):
    kind: Literal["resolve_conflict"] = "resolve_conflict"
    belief_ids: tuple[Identifier, ...] = ()
    accepted_belief_id: Identifier | None = None
    goal_id: Identifier | None = None
    expected_goal_revision: PositiveInt | None = None
    conflicting_revisions: tuple[Ref, ...] = ()
    resolved_constraints: tuple[Constraint, ...] | None = None
    evidence: tuple[Ref, ...] = ()
    explanation: NonEmpty

    @model_validator(mode="after")
    def _resolution_mode(self) -> ResolveConflict:
        refs = {ref.id for ref in self.conflicting_revisions}
        if len(refs) != len(self.conflicting_revisions) or not refs:
            raise ValueError("conflict resolution requires unique current revision references")
        if self.goal_id is None:
            if (
                not self.belief_ids
                or len(set(self.belief_ids)) != len(self.belief_ids)
                or refs != set(self.belief_ids)
            ):
                raise ValueError("belief resolution must name all conflicting belief revisions")
            if self.expected_goal_revision is not None or self.resolved_constraints is not None:
                raise ValueError("belief and constraint resolution modes cannot be mixed")
            if (
                self.accepted_belief_id is not None
                and self.accepted_belief_id not in self.belief_ids
            ):
                raise ValueError("accepted belief must belong to the conflict")
        elif (
            self.belief_ids
            or self.accepted_belief_id is not None
            or self.expected_goal_revision is None
            or self.resolved_constraints is None
        ):
            raise ValueError(
                "constraint resolution requires goal revision and replacement constraints only"
            )
        return self


class DecideAction(CommandBase):
    kind: Literal["decide_action"] = "decide_action"
    action_id: Identifier
    expected_revision: PositiveInt
    decision: Literal["approve", "deny"]
    arguments_digest: Digest
    scope: Scope = Field(default_factory=Scope)
    constraint_revision: PositiveInt
    explanation: str = ""


class ResolveAction(CommandBase):
    kind: Literal["resolve_action"] = "resolve_action"
    action_id: Identifier
    expected_revision: PositiveInt
    outcome: Literal["succeeded", "failed"]
    evidence: tuple[Ref, ...] = Field(min_length=1)
    explanation: NonEmpty


class ResolveUsage(CommandBase):
    kind: Literal["resolve_usage"] = "resolve_usage"
    reservation_id: Identifier
    expected_revision: PositiveInt
    usage: Usage | None = None
    evidence: tuple[Ref, ...] = ()
    accept_reserved_maximum: bool = False
    explanation: NonEmpty

    @model_validator(mode="after")
    def _explicit_resolution(self) -> ResolveUsage:
        if self.accept_reserved_maximum:
            if self.usage is not None:
                raise ValueError("choose known usage or reserved maximum, not both")
        elif self.usage is None or self.usage.status != "known" or not self.evidence:
            raise ValueError("known usage with evidence or explicit maximum acceptance required")
        return self


class AttachEvidence(CommandBase):
    kind: Literal["attach_evidence"] = "attach_evidence"
    goal_id: Identifier
    evidence: tuple[Ref, ...] = Field(min_length=1)
    task_id: Identifier | None = None
    criterion_id: Identifier | None = None


class Reverify(CommandBase):
    kind: Literal["reverify"] = "reverify"
    goal_id: Identifier
    criterion_ids: tuple[Identifier, ...] = ()


class ShareArtifact(CommandBase):
    kind: Literal["share_artifact"] = "share_artifact"
    artifact_id: Identifier
    principal_ids: tuple[Identifier, ...] = Field(min_length=1)
    expected_revision: PositiveInt = 1


class SetGrant(CommandBase):
    kind: Literal["set_grant"] = "set_grant"
    grant: Grant
    expected_revision: PositiveInt | None = None


Command = Annotated[
    CreateGoal
    | ReviseGoal
    | PauseGoal
    | ResumeGoal
    | CancelGoal
    | ResolveConflict
    | DecideAction
    | ResolveAction
    | ResolveUsage
    | AttachEvidence
    | Reverify
    | ShareArtifact
    | SetGrant,
    Field(discriminator="kind"),
]


class AuthenticatedCommand(Model):
    command: Command
    principal: Principal

    @property
    def principal_id(self) -> str:
        return self.principal.id


class ProposalBase(Record):
    episode_id: Identifier
    fencing_token: PositiveInt
    affected_records: tuple[Ref, ...] = ()
    read_set: tuple[Ref, ...] = ()
    evidence: tuple[Ref, ...] = ()
    justification: str = ""


class BeliefChange(ProposalBase):
    kind: Literal["belief_change"] = "belief_change"
    belief: Belief


class TaskDecomposition(ProposalBase):
    kind: Literal["task_decomposition"] = "task_decomposition"
    tasks: tuple[TaskFrame, ...] = Field(min_length=1)
    commitments: tuple[Commitment, ...] = ()


class ActionIntent(ProposalBase):
    kind: Literal["action_intent"] = "action_intent"
    capability_id: NonEmpty
    arguments: JsonObject
    scope: Scope = Field(default_factory=Scope)
    required: bool = True


class EvidenceRequest(ProposalBase):
    kind: Literal["evidence_request"] = "evidence_request"
    query: NonEmpty
    refs: tuple[Ref, ...] = ()
    wake_condition: WakeCondition | None = None


class CompletionClaim(ProposalBase):
    kind: Literal["completion_claim"] = "completion_claim"
    criterion_ids: tuple[Identifier, ...] = Field(min_length=1)
    artifacts: tuple[Ref, ...] = ()


Proposal = Annotated[
    BeliefChange | TaskDecomposition | ActionIntent | EvidenceRequest | CompletionClaim,
    Field(discriminator="kind"),
]


class EpisodeResult(Model):
    proposals: tuple[Proposal, ...] = ()
    artifact_drafts: tuple[ArtifactDraft, ...] = ()
    usage: Usage = Field(default_factory=Usage)
    stop_reason: Literal["proposed", "needs_evidence", "blocked", "exhausted"]
    blocker: WakeCondition | None = None
    resume_brief: str = ""

    @model_validator(mode="after")
    def _progression(self) -> EpisodeResult:
        if self.stop_reason == "blocked" and self.blocker is None:
            raise ValueError("blocked result requires a typed blocker")
        if self.stop_reason == "needs_evidence" and not any(
            isinstance(proposal, EvidenceRequest) for proposal in self.proposals
        ):
            raise ValueError("needs_evidence requires an evidence request")
        keys = [draft.key for draft in self.artifact_drafts]
        if len(keys) != len(set(keys)):
            raise ValueError("artifact draft keys must be unique within an episode")
        return self


class InboxAcknowledgment(Model):
    """Durable identity binding for an authenticated admitted input."""

    principal_id: Identifier
    producer_id: NonEmpty
    external_id: NonEmpty
    payload_digest: Digest
    receipt_id: Identifier


class OutboxEntry(Model):
    """Queue metadata; the referenced ActionRecord owns the effect lifecycle."""

    action_id: Identifier
    available_at: Timestamp = Field(default_factory=utc_now)


class CommitBatch(Model):
    # Response metadata only. The coordinator commits first, then resolves this
    # ID to the canonical stored receipt. Backpressure has no durable receipt.
    response_receipt: DecisionReceipt | None = None
    events: tuple[JournalEvent, ...] = ()
    records: tuple[PersistentRecord, ...] = ()
    receipts: tuple[DecisionReceipt, ...] = ()
    expected_revisions: tuple[Ref, ...] = ()
    signals: tuple[ExecutiveSignal, ...] = ()
    inbox_acknowledgments: tuple[InboxAcknowledgment, ...] = ()
    outbox: tuple[OutboxEntry, ...] = ()
