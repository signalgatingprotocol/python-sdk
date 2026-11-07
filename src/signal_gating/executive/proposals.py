"""Validate a whole worker result before preparing any authoritative write.

Drafts are staged by the coordinator using ArtifactStore.stage_drafts outside
transactions. Supply its mapping here, then release_staged in finally after
commit/rejection/error. This module neither writes bytes nor dispatches effects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any

from .errors import ExecutiveError, InvalidTransition
from .memory import MemoryPolicy, invalidate_dependents, records_by_id, reference
from .messages import (
    ActionIntent,
    BeliefChange,
    CommitBatch,
    CompletionClaim,
    EpisodeResult,
    EvidenceRequest,
    TaskDecomposition,
)
from .records import (
    ActionRecord,
    ArtifactRef,
    Belief,
    Commitment,
    ContextPacket,
    DecisionReceipt,
    Disposition,
    Episode,
    ExecutiveSignal,
    Goal,
    Grant,
    JournalEvent,
    Model,
    PersistentRecord,
    Ref,
    Scope,
    StateView,
    TaskFrame,
    utc_now,
)
from .reducers import reduce_event


class ProposalValidator:
    def __init__(
        self,
        *,
        memory: MemoryPolicy | None = None,
        staged_drafts: Mapping[str, ArtifactRef] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.clock = clock
        self.memory = memory or MemoryPolicy(clock=clock)
        self.staged_drafts = MappingProxyType(dict(staged_drafts or {}))

    def validate(self, result: EpisodeResult, episode: Episode, state: StateView) -> CommitBatch:
        records = records_by_id(state)
        task = records.get(episode.task_id)
        goal = records.get(task.goal_id) if isinstance(task, TaskFrame) else None
        principal_id = (
            goal.owner_id
            if isinstance(goal, Goal)
            else next(
                (r.id for r in state.records if getattr(r, "kind", "") == "principal"), episode.id
            )
        )
        now = self.clock()
        reads: dict[str, Ref] = {}

        def require(ref: Ref) -> None:
            found = records.get(ref.id)
            if (
                found is None
                or found.revision != ref.revision
                or (ref.kind != "record" and ref.kind != getattr(found, "kind", ""))
            ):
                raise InvalidTransition("stale_dependency", "Proposal dependency changed")
            # Embedded constraints/criteria are fenced by their owning Goal.
            if any(r.id == ref.id for r in state.records):
                reads[ref.id] = reference(found)
            else:
                for owner in state.records:
                    if isinstance(owner, Goal) and any(
                        c.id == ref.id for c in (*owner.constraints, *owner.criteria)
                    ):
                        reads[owner.id] = reference(owner)

        def changed(items: tuple[PersistentRecord, ...], subject: Ref) -> JournalEvent:
            return JournalEvent(
                created_at=now,
                sequence=1,
                event_type="state.changed",
                producer_id="proposal-validator",
                principal_id=principal_id,
                subject=subject,
                payload={"records": [r.model_dump(mode="json") for r in items]},
            )

        def finish(
            disposition: Disposition, reason: str, events: tuple[JournalEvent, ...] = ()
        ) -> CommitBatch:
            receipt = DecisionReceipt(
                created_at=now,
                subject_id=episode.id,
                disposition=disposition,
                reason_code=reason,
                state_revisions=tuple(reads.values()),
                event_ids=tuple(e.id for e in events),
            )
            receipt_event = changed((receipt,), reference(episode))
            return CommitBatch(
                response_receipt=receipt,
                events=(*events, receipt_event),
                expected_revisions=tuple(reads.values()),
            )

        try:
            current = records.get(episode.id)
            if (
                current != episode
                or not isinstance(task, TaskFrame)
                or not isinstance(goal, Goal)
                or episode.status != "running"
                or task.status != "running"
                or goal.status != "active"
                or episode.fencing_token != task.fencing_token
                or episode.deadline <= now
            ):
                raise InvalidTransition("stale_dependency", "Episode is stale, fenced or expired")
            context = records.get(episode.context_packet_id)
            if (
                not isinstance(context, ContextPacket)
                or context.episode_id != episode.id
                or context.task.id != task.id
                or context.goal.id != goal.id
                or context.fencing_token != episode.fencing_token
                or goal.worker_id != episode.worker_id
            ):
                raise InvalidTransition("stale_dependency", "Episode context binding changed")
            for binding in (episode, context, task, goal):
                require(reference(binding))
            mandatory = {
                *context.mandatory_read_set,
                reference(context.goal),
                reference(context.task),
                *(reference(b) for b in context.beliefs),
                *(reference(c) for c in context.goal.constraints),
            }
            mandatory.update(
                reference(g)
                for g in state.records
                if isinstance(g, Grant)
                and not g.revoked
                and g.scope.goal_id in (None, goal.id)
                and g.scope.owner_id in (None, goal.owner_id)
            )
            for ref in mandatory:
                require(ref)
            if len({p.id for p in result.proposals}) != len(result.proposals):
                raise InvalidTransition("duplicate_proposal", "Proposal IDs must be unique")
            if any(p.id in records for p in result.proposals):
                raise InvalidTransition("duplicate_proposal", "Proposal ID already exists")
            for proposal in result.proposals:
                if (
                    proposal.episode_id != episode.id
                    or proposal.fencing_token != episode.fencing_token
                ):
                    raise InvalidTransition("stale_dependency", "Proposal episode fence changed")
                if proposal.kind not in episode.allowed_proposal_types:
                    raise InvalidTransition("proposal_type_denied", "Proposal type is not allowed")
                supplied: dict[str, Ref] = {}
                for ref in proposal.read_set:
                    require(ref)
                    if ref.id in supplied:
                        raise InvalidTransition("duplicate_read", "Each dependency is named once")
                    supplied[ref.id] = ref
                if any(
                    ref.id not in supplied or supplied[ref.id].revision != ref.revision
                    for ref in mandatory
                ):
                    raise InvalidTransition(
                        "missing_mandatory_read_set", "Worker omitted a mandatory dependency"
                    )
                for ref in proposal.affected_records:
                    require(ref)
                    if ref.id not in supplied:
                        raise InvalidTransition(
                            "missing_affected_read", "Affected records need a read dependency"
                        )
            # Staging owns byte limits. Check the injected mapping matches exactly
            # this episode's declared bytes and scope; never stage from this call.
            drafts = {d.key: d for d in result.artifact_drafts}
            if drafts.keys() != self.staged_drafts.keys():
                raise InvalidTransition(
                    "draft_not_staged", "Draft mapping differs from episode drafts"
                )
            for key, artifact in self.staged_drafts.items():
                draft = drafts[key]
                data = draft.text.encode("utf-8")
                if (
                    artifact.id in records
                    or artifact.revision != 1
                    or not artifact.valid
                    or artifact.scope != Scope(owner_id=goal.owner_id, goal_id=goal.id)
                    or artifact.media_type != draft.media_type
                    or artifact.size_bytes != len(data)
                    or artifact.digest != hashlib.sha256(data).hexdigest()
                ):
                    raise InvalidTransition("invalid_staged_draft", "Staged draft binding differs")

            def resolve(value: Any) -> Any:
                if isinstance(value, Ref) and value.id.startswith("draft:"):
                    artifact = self.staged_drafts.get(value.id[6:])
                    if (
                        artifact is None
                        or value.revision != 1
                        or value.kind not in {"record", "draft", "artifact"}
                    ):
                        raise InvalidTransition("missing_draft", "Unknown draft reference")
                    return reference(artifact)
                if isinstance(value, Model):
                    return value.model_copy(
                        update={
                            name: resolve(getattr(value, name)) for name in type(value).model_fields
                        }
                    )
                if isinstance(value, (list, tuple)):
                    return tuple(resolve(item) for item in value)
                # Declared JSON objects (such as action arguments) are opaque.
                return value

            resolved: EpisodeResult = resolve(result)
            staged_ids = {a.id for a in self.staged_drafts.values()}
            evaluation_state = StateView(
                sequence=state.sequence, records=(*state.records, *self.staged_drafts.values())
            )
            evaluation_records = records_by_id(evaluation_state)
            writes: dict[str, PersistentRecord] = dict(
                (a.id, a) for a in self.staged_drafts.values()
            )
            materially_changed: list[Ref] = []
            completion = False

            def write(item: PersistentRecord) -> None:
                if item.id in writes:
                    raise InvalidTransition("duplicate_write", "Bundle writes each identity once")
                writes[item.id] = item
                if item.id in records:
                    require(reference(records[item.id]))
                    materially_changed.append(reference(item))

            def evidence(refs: tuple[Ref, ...]) -> None:
                for ref in refs:
                    item = evaluation_records.get(ref.id)
                    if (
                        item is None
                        or item.revision != ref.revision
                        or (ref.kind != "record" and ref.kind != getattr(item, "kind", ""))
                    ):
                        raise InvalidTransition("stale_dependency", "Evidence is missing or stale")
                    if ref.id not in staged_ids:
                        require(ref)
                    if isinstance(item, ArtifactRef) and (
                        not item.valid
                        or item.scope.goal_id != goal.id
                        or (
                            item.scope.owner_id != goal.owner_id
                            and goal.owner_id not in item.shared_with
                        )
                    ):
                        raise InvalidTransition(
                            "invalid_evidence", "Artifact evidence is inaccessible"
                        )

            for proposal in resolved.proposals:
                evidence(proposal.evidence)
                if isinstance(proposal, BeliefChange):
                    belief = proposal.belief
                    if belief.scope.goal_id != goal.id or belief.scope.owner_id not in (
                        None,
                        goal.owner_id,
                    ):
                        raise InvalidTransition(
                            "belief_scope_mismatch", "Belief belongs to another goal"
                        )
                    old = records.get(belief.id)
                    if old is not None:
                        if (
                            not isinstance(old, Belief)
                            or belief.revision != old.revision + 1
                            or belief.created_at != old.created_at
                        ):
                            raise InvalidTransition(
                                "invalid_revision", "Belief revision must advance once"
                            )
                        if not any(ref.id == old.id for ref in proposal.read_set):
                            raise InvalidTransition(
                                "missing_affected_read", "Belief update needs its prior revision"
                            )
                    elif belief.revision != 1:
                        raise InvalidTransition(
                            "invalid_revision", "New beliefs start at revision one"
                        )
                    all_evidence = tuple(dict.fromkeys((*belief.evidence, *proposal.evidence)))
                    evidence(all_evidence)
                    evidence(belief.supersedes)
                    belief = belief.model_copy(update={"evidence": all_evidence})
                    memory_change = proposal.model_copy(update={"belief": belief})
                    for consulted in self.memory.inputs(memory_change, evaluation_state).records:
                        if consulted.id not in staged_ids:
                            require(reference(consulted))
                    policy = self.memory.evaluate(memory_change, evaluation_state)
                    if policy.disposition == "reject":
                        raise InvalidTransition(policy.reason_code, "Memory rejected belief")
                    status = (
                        "disputed"
                        if policy.disposition == "dispute"
                        else (
                            "supported"
                            if policy.disposition in {"admit", "revise"}
                            else "hypothesis"
                        )
                    )
                    if any(ref.id in staged_ids for ref in all_evidence):
                        status = "hypothesis"
                    write(belief.model_copy(update={"epistemic_status": status}))
                elif isinstance(proposal, TaskDecomposition):
                    parent = records.get(task.commitment_id)
                    if not isinstance(parent, Commitment):
                        raise InvalidTransition("missing_reference", "Parent commitment is missing")
                    require(reference(parent))
                    inherited_budget = parent.budget_id or goal.budget_id
                    for commitment in proposal.commitments:
                        if (
                            commitment.id in records
                            or commitment.revision != 1
                            or commitment.goal_id != goal.id
                            or commitment.responsible_principal_id
                            != parent.responsible_principal_id
                            or commitment.budget_id not in (None, inherited_budget)
                            or commitment.status != "pending"
                        ):
                            raise InvalidTransition(
                                "invalid_decomposition",
                                "Decomposition cannot widen responsibility, budget or lifecycle",
                            )
                        write(
                            commitment.model_copy(
                                update={
                                    "budget_id": inherited_budget,
                                    "dependencies": tuple(
                                        dict.fromkeys(
                                            (*parent.dependencies, *commitment.dependencies)
                                        )
                                    ),
                                }
                            )
                        )
                    permitted = {parent.id, *(c.id for c in proposal.commitments)}
                    for child in proposal.tasks:
                        if (
                            child.id in records
                            or child.revision != 1
                            or child.goal_id != goal.id
                            or child.commitment_id not in permitted
                            or child.status not in {"pending", "ready"}
                            or child.fencing_token != 0
                            or child.pending_action_ids
                        ):
                            raise InvalidTransition(
                                "invalid_decomposition", "Child work must inherit its parent goal"
                            )
                        write(
                            child.model_copy(
                                update={
                                    "dependencies": tuple(
                                        dict.fromkeys((*task.dependencies, *child.dependencies))
                                    )
                                }
                            )
                        )
                elif isinstance(proposal, ActionIntent):
                    capability = next(
                        (c for c in context.capabilities if c.id == proposal.capability_id), None
                    )
                    if capability is None or capability.effect_class == "human_only":
                        raise InvalidTransition(
                            "capability_denied", "No permitted capability binding"
                        )
                    if proposal.scope.owner_id not in (
                        None,
                        goal.owner_id,
                    ) or proposal.scope.goal_id not in (None, goal.id):
                        raise InvalidTransition(
                            "action_scope_mismatch", "Action scope differs from goal"
                        )
                    write(
                        ActionRecord(
                            id=proposal.id,
                            created_at=now,
                            goal_id=goal.id,
                            task_id=task.id,
                            capability_id=capability.id,
                            binding_version=capability.binding_version,
                            arguments_digest=hashlib.sha256(
                                json.dumps(
                                    proposal.model_dump(mode="json")["arguments"],
                                    sort_keys=True,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                ).encode("utf-8")
                            ).hexdigest(),
                            arguments=proposal.arguments,
                            scope=proposal.scope,
                            idempotency_key=proposal.id,
                            constraint_revision=goal.revision,
                            read_set=proposal.read_set,
                            evidence=proposal.evidence,
                        )
                    )
                elif isinstance(proposal, CompletionClaim):
                    if not set(proposal.criterion_ids) <= {c.id for c in goal.criteria}:
                        raise InvalidTransition(
                            "unknown_criterion", "Completion claim names another goal criterion"
                        )
                    evidence(proposal.artifacts)
                    completion = True
                    write(
                        ExecutiveSignal(
                            id=proposal.id,
                            created_at=now,
                            received_at=now,
                            signal_kind="completion_claim",
                            principal_id=goal.owner_id,
                            producer_id="worker-proposal",
                            scope=Scope(owner_id=goal.owner_id, goal_id=goal.id),
                            artifacts=proposal.artifacts,
                            payload={"proposal": proposal.model_dump(mode="json")},
                        )
                    )
                elif isinstance(proposal, EvidenceRequest):
                    evidence(proposal.refs)
                    write(
                        ExecutiveSignal(
                            id=proposal.id,
                            created_at=now,
                            received_at=now,
                            signal_kind="evidence_request",
                            principal_id=goal.owner_id,
                            producer_id="worker-proposal",
                            scope=Scope(owner_id=goal.owner_id, goal_id=goal.id),
                            payload={"proposal": proposal.model_dump(mode="json")},
                        )
                    )
            # Detect conflicts against the complete bundle, so two new claims
            # cannot both gain support merely because neither existed before it.
            candidates = {record.id: record for record in evaluation_state.records}
            candidates.update(writes)
            candidate_state = StateView(records=tuple(candidates.values()))
            for candidate_belief in tuple(writes.values()):
                if not isinstance(
                    candidate_belief, Belief
                ) or candidate_belief.epistemic_status not in {
                    "supported",
                    "disputed",
                }:
                    continue
                try:
                    conflicts = self.memory.conflicts(
                        candidate_belief,
                        self.memory.inputs(
                            BeliefChange(
                                episode_id=episode.id,
                                fencing_token=episode.fencing_token,
                                belief=candidate_belief,
                            ),
                            candidate_state,
                        ),
                    )
                except Exception as exc:
                    raise InvalidTransition(
                        "conflict_detector_failed", "Trusted conflict detector failed"
                    ) from exc
                if conflicts:
                    writes[candidate_belief.id] = candidate_belief.model_copy(
                        update={"epistemic_status": "disputed"}
                    )
                for ref in conflicts:
                    conflict = candidates[ref.id]
                    if not isinstance(conflict, Belief):
                        continue
                    if conflict.id in writes:
                        writes[conflict.id] = writes[conflict.id].model_copy(
                            update={"epistemic_status": "disputed"}
                        )
                    elif conflict.epistemic_status != "disputed":
                        write(
                            conflict.model_copy(
                                update={
                                    "revision": conflict.revision + 1,
                                    "epistemic_status": "disputed",
                                }
                            )
                        )
            # A simultaneous revision can withdraw support that passed against
            # pre-batch state. Check every proposed support chain against final
            # post-images, including unchanged intermediate beliefs.
            candidates.update(writes)
            candidate_state = StateView(records=tuple(candidates.values()))
            for support_change in resolved.proposals:
                if not isinstance(support_change, BeliefChange):
                    continue
                supported = writes[support_change.belief.id]
                if not isinstance(supported, Belief) or supported.epistemic_status != "supported":
                    continue
                final_policy = self.memory.evaluate(
                    support_change.model_copy(update={"belief": supported}), candidate_state
                )
                if final_policy.disposition not in {"admit", "revise", "dispute"}:
                    raise InvalidTransition("stale_dependency", "Bundle withdrew belief support")
                pending = [supported]
                checked: set[str] = set()
                while pending:
                    supported_input = pending.pop()
                    if supported_input.id in checked:
                        continue
                    checked.add(supported_input.id)
                    for support_ref in supported_input.evidence:
                        dependency = candidates.get(support_ref.id)
                        if (
                            dependency is None
                            or dependency.revision != support_ref.revision
                            or (
                                support_ref.kind != "record" and support_ref.kind != dependency.kind
                            )
                            or isinstance(dependency, Belief)
                            and dependency.epistemic_status != "supported"
                            or isinstance(dependency, ArtifactRef)
                            and not dependency.valid
                        ):
                            raise InvalidTransition(
                                "stale_dependency", "Bundle withdrew belief support"
                            )
                        if dependency.id in records:
                            require(reference(records[dependency.id]))
                        if isinstance(dependency, Belief):
                            pending.append(dependency)
            # No worker status can finish a task, fulfill a criterion or dispatch.
            writes[episode.id] = episode.model_copy(
                update={"revision": episode.revision + 1, "status": "accepted"}
            )
            writes[task.id] = task.model_copy(
                update={
                    "revision": task.revision + 1,
                    "status": "verifying" if completion else "suspended",
                    "fencing_token": task.fencing_token + (0 if completion else 1),
                    "resume_brief": resolved.resume_brief,
                    "wake_condition": resolved.blocker
                    if resolved.stop_reason == "blocked"
                    else None,
                }
            )
            events = (changed(tuple(writes.values()), reference(episode)),)
            after = reduce_event(
                state, events[0].model_copy(update={"sequence": state.sequence + 1})
            )
            cascade = invalidate_dependents(after, tuple(materially_changed))
            for cascade_event in cascade:
                after = reduce_event(
                    after, cascade_event.model_copy(update={"sequence": after.sequence + 1})
                )
            before = {r.id: r for r in state.records}
            for item in after.records:
                if item.id in before and item != before[item.id]:
                    reads[item.id] = reference(before[item.id])
            return finish("accept", "bundle_validated", (*events, *cascade))
        except ExecutiveError as exc:
            # Keep only current pre-batch refs on a negative receipt. A stale
            # worker read is a reason to reject, never a stale CAS for the receipt.
            reads = {
                identifier: ref
                for identifier, ref in reads.items()
                if identifier in records and records[identifier].revision == ref.revision
            }
            return finish("stale" if exc.code == "stale_dependency" else "reject", exc.code)
