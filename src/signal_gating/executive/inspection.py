"""Read-only projections. Idle, successful proof and unsettled spending differ."""

from __future__ import annotations

import re
from typing import Any

from pydantic import TypeAdapter

from . import messages
from . import records as contracts
from .artifacts import AccessDenied, ArtifactStore
from .authority import registered
from .errors import ConflictError
from .records import (
    ActionRecord,
    ArtifactRef,
    Budget,
    ContextManifest,
    ContextPacket,
    CriterionApproval,
    DecisionReceipt,
    Episode,
    Goal,
    JournalEvent,
    KernelView,
    Model,
    PersistentRecord,
    Principal,
    Ref,
    Reservation,
    RunSummary,
    Scope,
    StateView,
    TaskFrame,
    Usage,
    VerificationResult,
)
from .store import Store

_RECORD_ADAPTER: TypeAdapter[PersistentRecord] = TypeAdapter(PersistentRecord)


def export_contract_schemas() -> dict[str, object]:
    """Return experimental, complete v1 DTO JSON Schemas, without providers.

    These are validation-mode schemas, including the closed discriminated
    command/proposal/record unions. They are deliberately separate from the
    compact worker prompt contract and the redacted goal observation format.
    Pydantic's cross-field semantic checks still require the local DTO validator;
    JSON Schema alone does not establish freshness, authority or conformance of
    an execution. No import, federation, delegation or remote enforcement exists.
    """
    schemas = {
        name: value.model_json_schema()
        for module in (contracts, messages)
        for name, value in vars(module).items()
        if isinstance(value, type)
        and issubclass(value, Model)
        and value.__module__ == module.__name__
    }
    command: TypeAdapter[messages.Command] = TypeAdapter(messages.Command)
    proposal: TypeAdapter[messages.Proposal] = TypeAdapter(messages.Proposal)
    schemas.update(
        PersistentRecord=_RECORD_ADAPTER.json_schema(),
        Command=command.json_schema(),
        Proposal=proposal.json_schema(),
    )
    for schema in schemas.values():
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return {"schema_version": 1, "experimental": True, "schemas": schemas}


# Positive allowlist: no opaque JSON, free-form task/prompt/claim bodies,
# authentication origins, machine paths, external receipts or executable inputs.
# Missing/new DTO fields stay private until explicitly reviewed for this format.
_OBSERVED_FIELDS = frozenset(
    "schema_version id revision kind created_at owner_id goal_id principal_id principal_ids "
    "parent_id budget_id task_id commitment_id episode_id context_packet_id criterion_id "
    "issuer_id subject_id responsible_principal_id origin_principal_id worker_id roles "
    "status required hard enforcement enforcement_id verifier_id check_id criteria constraints "
    "required_commitment_ids dependencies inputs required_evidence scope restrictions "
    "revoked revocation_revision expires_at deadline priority valid_from valid_until "
    "epistemic_status evidence supersedes confidence confidence_source calibration_status "
    "progress_artifacts pending_action_ids wake_condition fencing_token failure_count "
    "failure_signature failure_reason urgent eligible_since last_scheduled_at "
    "limits reserved actual accounting_uncertain maximum trusted_maximum_quote usage "
    "observed_usage accounting_conflict input_tokens output_tokens calls elapsed_seconds "
    "cost_microunits currency source binding_version binding_id accepted_task_revision "
    "accepted_episode_revision accepted_fencing_token dispatched_at dispatch_attempts "
    "prior_reservation_ids arguments_digest arguments_ref authorization_revision "
    "constraint_revision grant_id reservation_id read_set disposition reason_code "
    "state_revisions policy_version event_ids artifacts goal criterion goal_definition_digest "
    "gaps evidence_needed signal_kind producer_sequence received_at observed_at causal_ids "
    "allowed_proposal_types commitments beliefs changes contradictions pending_effects "
    "remaining_limits mandatory_read_set manifest required_ids visible_ids context_token_limit "
    "included omitted observed_refs required_tokens output_tokens_reserved framing_tokens_reserved "
    "counter_id estimated sequence event_type subject at refs".split()
)
_ID_LISTS = frozenset(
    {
        "required_ids",
        "visible_ids",
        "event_ids",
        "causal_ids",
        "required_commitment_ids",
        "pending_action_ids",
    }
)
_SCALAR_REFERENCES = frozenset(
    "owner_id goal_id principal_id parent_id budget_id task_id commitment_id episode_id "
    "context_packet_id criterion_id issuer_id subject_id responsible_principal_id "
    "origin_principal_id grant_id reservation_id".split()
)


def _observation(value: Any, visible: set[str], hidden: set[str]) -> Any:
    if isinstance(value, Ref):
        return value.model_dump(mode="json") if value.id in visible else None
    if (
        isinstance(value, ArtifactRef)
        or isinstance(value, contracts.Record)
        and value.id not in visible
    ):
        return None
    if isinstance(value, Model):
        result = {}
        for key in type(value).model_fields:
            if key not in _OBSERVED_FIELDS:
                continue
            item = getattr(value, key)
            # Scalar record links need the same privacy boundary as Ref values.
            # Unknown external subjects remain inert lineage, not permissions.
            if key in _SCALAR_REFERENCES and item in hidden:
                continue
            if key in _ID_LISTS:
                item = tuple(identity for identity in item if identity in visible)
            projected = _observation(item, visible, hidden)
            if projected is not None or item is None:
                result[key] = projected
        if isinstance(value, ContextManifest):
            result["observed_refs_authority"] = "advisory_only"
        if isinstance(value, contracts.Record):
            result["authority"] = "observation_only"
        return result
    if isinstance(value, (tuple, list)):
        return [
            projected
            for item in value
            if (projected := _observation(item, visible, hidden)) is not None
        ]
    if isinstance(value, str):
        # IDs and operator code/version labels survive; prose, URLs, filesystem
        # paths and multi-line content have no place in this metadata projection.
        return value if re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value) else "[redacted]"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _postimages(event: JournalEvent) -> tuple[PersistentRecord, ...]:
    images = (
        event.model_dump(mode="python")["payload"].get("records", ())
        if event.event_type == "state.changed"
        else ()
    )
    if not isinstance(images, (tuple, list)):
        return ()
    return tuple(_RECORD_ADAPTER.validate_python(image) for image in images)


def _goal_records(state: StateView, goal: Goal, events: tuple[JournalEvent, ...]) -> set[str]:
    selected = {goal.id, *(c.id for c in (*goal.criteria, *goal.constraints))}
    known = {record.id for record in state.records}
    for event in events:
        for image in _postimages(event):
            if isinstance(image, Goal) and image.id == goal.id:
                selected.update(c.id for c in (*image.criteria, *image.constraints))
    for record in state.records:
        scope = getattr(record, "scope", None)
        if (
            getattr(record, "goal_id", None) == goal.id
            or isinstance(scope, Scope)
            and scope.goal_id == goal.id
            and scope.owner_id in (None, goal.owner_id)
            or isinstance(record, ContextPacket)
            and record.goal.id == goal.id
            or isinstance(record, CriterionApproval)
            and record.goal.id == goal.id
        ):
            selected.add(record.id)
    # Follow only typed ownership relationships, never arbitrary evidence refs
    # into another goal. Receipts inherit scope from their subject/read history.
    changed = True
    while changed:
        previous = set(selected)
        scoped_events = {
            event.id
            for event in events
            if event.subject.id in selected
            or any(image.id in selected for image in _postimages(event))
        }
        for record in state.records:
            if (
                isinstance(record, Episode)
                and record.task_id in selected
                or isinstance(record, Reservation)
                and record.budget_id in selected
                or isinstance(record, VerificationResult)
                and record.criterion_id in selected
                or isinstance(record, DecisionReceipt)
                and (record.subject_id not in known or record.subject_id in selected)
                and (
                    record.subject_id in selected
                    or any(
                        ref.id in selected for ref in (*record.state_revisions, *record.evidence)
                    )
                    or bool(set(record.event_ids) & scoped_events)
                )
            ):
                selected.add(record.id)
        changed = previous != selected
    return selected


def export_goal(store: Store, goal_id: str, *, principal: Principal) -> dict[str, object]:
    """Export a current owner-only, redacted observation, never restorable state.

    Reopen a read-only snapshot even for a stale reader. Only the exact current
    registered Principal DTO of the goal owner may inspect the graph. Artifact
    sharing alone does not authorize goal/history access. Manifests require current
    owner/goal scope, validity, ACL and verified bytes. Private context artifacts
    are excluded even for the owner. No bytes, prompts, opaque payloads, machine
    paths, credentials or transferable grants are exported. Removed/private
    references are omitted, so this metadata is intentionally not a full audit
    replay, an import format, a permission, or a remote enforcement guarantee.
    Manifest integrity labels describe bytes/checksums only, never acceptance
    of evidence, successful work, or an evaluation/comparison benefit claim.
    """
    with Store.open_readonly(store.root) as snapshot:
        state, events = snapshot.view(), snapshot.events()
        goal = next((r for r in state.records if isinstance(r, Goal) and r.id == goal_id), None)
        if (
            goal is None
            or registered(principal, state) != principal
            or goal.owner_id != principal.id
        ):
            raise AccessDenied("export_access_denied", "Goal export access denied")
        selected = _goal_records(state, goal, events)
        manifests = []
        private = {
            record.rendered_artifact.id
            for record in (*state.records, *(image for e in events for image in _postimages(e)))
            if isinstance(record, ContextPacket) and record.rendered_artifact is not None
        }
        for record in state.records:
            if not isinstance(record, ArtifactRef) or record.id not in selected:
                continue
            if (
                not record.valid
                or record.scope.owner_id != goal.owner_id
                or record.id in private
                or record.media_type.partition(";")[0].strip().lower()
                == "application/vnd.sgp.executive-context+json"
            ):
                private.add(record.id)
                continue
            # Read() performs current authorization plus complete digest/size
            # verification. No artifact content enters the exported value.
            ArtifactStore(snapshot).read(record, principal=principal, goal_id=goal.id)
            with Store.open_readonly(store.root) as current:
                latest = next((r for r in current.view().records if r.id == record.id), None)
                if latest != record:
                    raise ConflictError("stale_export_artifact", "Artifact changed during export")
            manifests.append(
                {
                    "schema_version": 1,
                    "id": record.id,
                    "revision": record.revision,
                    "digest": record.digest,
                    "media_type": record.media_type
                    if re.fullmatch(
                        r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", record.media_type
                    )
                    else "[redacted]",
                    "size_bytes": record.size_bytes,
                    "valid": True,
                    "scope": {"owner_id": goal.owner_id, "goal_id": goal.id},
                    "shared_with": list(record.shared_with),
                    "integrity": "verified",
                    "authority": "observation_only",
                }
            )
        selected -= private
        scoped_events = tuple(
            event
            for event in events
            if event.subject.id in selected
            or any(image.id in selected for image in _postimages(event))
        )
        visible = selected | {event.id for event in scoped_events} | {principal.id}
        # Preserve external causal identities as inert lineage, without exposing
        # known private/cross-goal records or context-artifact identities.
        known = {r.id for r in state.records} | private
        known.update(
            definition.id
            for record in (*state.records, *(image for e in events for image in _postimages(e)))
            if isinstance(record, Goal)
            for definition in (*record.criteria, *record.constraints)
        )
        lineage = {event.subject.id for event in scoped_events}
        lineage.update(identity for event in scoped_events for identity in event.causal_ids)
        lineage.update(
            identity
            for record in state.records
            if isinstance(record, contracts.ExecutiveSignal) and record.id in selected
            for identity in record.causal_ids
        )
        visible.update(
            identity for identity in lineage if identity not in known or identity in visible
        )
        visible -= private
        hidden = known - visible
        history = []
        for event in scoped_events:
            observation = _observation(event, visible, hidden)
            observation["records"] = [
                _observation(image, visible, hidden)
                for image in _postimages(event)
                if image.id in selected and not isinstance(image, ArtifactRef)
            ]
            history.append(observation)
        logical = [
            _observation(record, visible, hidden)
            for record in state.records
            if record.id in selected and not isinstance(record, (ArtifactRef, JournalEvent))
        ]
        policies = {
            _observation(record.policy_version, visible, hidden)
            for record in (
                *state.records,
                *(image for e in scoped_events for image in _postimages(e)),
            )
            if record.id in selected and isinstance(record, (DecisionReceipt, VerificationResult))
        }
        return {
            "schema_version": 1,
            "experimental": True,
            "format": "goal-observation-v1",
            "authority": "observation_only",
            "redaction": "metadata_only_private_references_omitted",
            "goal_id": goal.id,
            "principal_id": principal.id,
            "sequence": state.sequence,
            "policy_versions": sorted(policies),
            "records": logical,
            "artifacts": manifests,
            "history": history,
        }


def inspect_state(state: StateView, goal_id: str | None = None) -> KernelView:
    goals = {r.id for r in state.records if isinstance(r, Goal) and goal_id in (None, r.id)}
    tasks = tuple(r for r in state.records if isinstance(r, TaskFrame) and r.goal_id in goals)
    actions = tuple(r for r in state.records if isinstance(r, ActionRecord) and r.goal_id in goals)
    budgets = tuple(
        r
        for r in state.records
        if isinstance(r, Budget)
        and r.scope.goal_id in goals
        and (goal_id is not None or r.parent_id is None)
    )
    quantities = ("input_tokens", "output_tokens", "calls", "elapsed_seconds", "cost_microunits")
    currencies = {r.actual.currency for r in budgets}
    uncertain = any(r.accounting_uncertain or r.actual.status != "known" for r in budgets)
    usage = Usage.model_validate(
        {
            key: None
            if len(currencies) > 1 or any(getattr(r.actual, key) is None for r in budgets)
            else sum(getattr(r.actual, key) for r in budgets)
            for key in quantities
        }
        | {
            "status": "unknown" if uncertain or len(currencies) > 1 else "known",
            "currency": next(iter(currencies), "USD"),
            "source": "kernel-budget-summary",
        }
    )
    summary = RunSummary(
        completed_goal_ids=tuple(
            r.id
            for r in state.records
            if isinstance(r, Goal) and r.id in goals and r.status == "completed"
        ),
        blocked_task_ids=tuple(r.id for r in tasks if r.status == "blocked"),
        pending_review_ids=tuple(
            r.id for r in actions if r.status in {"awaiting_review", "needs_human"}
        ),
        unknown_action_ids=tuple(
            r.id
            for r in actions
            if r.status in {"dispatching", "outcome_unknown", "reconciling", "needs_human"}
        ),
        usage=usage,
    )
    # Keep the full typed proof graph available for inspection; goal_id filters
    # the summary, not authoritative references required to explain a finding.
    return KernelView(state=state, summary=summary, goal_id=goal_id)


def operator_view(view: KernelView) -> dict[str, Any]:
    """Current operator proof metadata, without context or artifact content.

    Keep reference identities explicit. Publication is not sharing and an idle
    finding is not settled spending. The typed KernelView remains authoritative.
    """
    records = view.state.records
    references = {r.id: Ref(id=r.id, revision=r.revision) for r in records}
    for record in records:
        if isinstance(record, Goal):
            for definition in (*record.criteria, *record.constraints):
                references[definition.id] = Ref(id=definition.id, revision=definition.revision)
    groups = {
        "goals": "goal",
        "tasks": "task",
        "commitments": "commitment",
        "actions": "action",
        "budgets": "budget",
        "reservations": "reservation",
        "findings": "verification",
        "approvals": "criterion_approval",
        "receipts": "receipt",
        "principals": "principal",
    }
    result: dict[str, Any] = {
        name: [r.model_dump(mode="json") for r in records if r.kind == kind]
        for name, kind in groups.items()
    }
    result.update(
        sequence=view.state.sequence,
        goal_id=view.goal_id,
        summary=view.summary.model_dump(mode="json"),
        current_refs=[ref.model_dump(mode="json") for ref in references.values()],
        artifacts=[
            r.model_dump(
                mode="json",
                include={
                    "id",
                    "revision",
                    "digest",
                    "media_type",
                    "size_bytes",
                    "shared_with",
                    "valid",
                },
            )
            for r in records
            if r.kind == "artifact"
        ],
        pending_reviews=[
            r.model_dump(mode="json") for r in records if r.id in view.summary.pending_review_ids
        ],
    )
    selected = [r for r in records if isinstance(r, Goal) and r.id == view.goal_id]
    result["goal_status"] = selected[0].status if selected else None
    return result
