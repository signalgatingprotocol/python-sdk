"""Fail-closed authority from registered state and operator-supplied functions.

Principals come from authenticated adapters. Their IDs/origins identify callers;
roles, grants and current revisions come only from the authoritative StateView.
Rule and resource extractor registries are trusted Python configuration, never
imports or executable expressions selected by model-produced text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import PurePosixPath
from types import MappingProxyType

from signal_gating._immutable import deep_thaw

from .memory import records_by_id
from .messages import (
    ActionIntent,
    Command,
    CreateGoal,
    ResolveConflict,
    ResolveUsage,
    SetGrant,
)
from .records import (
    ActionRecord,
    ArtifactRef,
    Belief,
    Budget,
    CapabilitySpec,
    Constraint,
    ContextPacket,
    Criterion,
    Episode,
    Goal,
    Grant,
    PolicyDecision,
    Principal,
    Record,
    Ref,
    Reservation,
    Scope,
    StateView,
    TaskFrame,
    VerificationResult,
)

Rule = Callable[[Mapping[str, Record]], PolicyDecision]
ResourceExtractor = Callable[[ActionIntent], Scope]


def _proof_roots(
    task: TaskFrame, reads: tuple[Ref, ...], indexed: Mapping[str, Record]
) -> tuple[Ref, ...]:
    """Definition/data reads fence versions; only accepted support implies proof.

    Explicit task dependencies always retain satisfaction checks. Supplied unknown
    beliefs and criterion definitions confer no authority just by being read.
    """
    return tuple(
        dict.fromkeys(
            (
                *task.dependencies,
                *(
                    ref
                    for ref in reads
                    if (
                        isinstance(item := indexed.get(ref.id), Belief)
                        and item.epistemic_status == "supported"
                    )
                    or (isinstance(item, VerificationResult) and item.disposition == "accept")
                ),
            )
        )
    )


def decision(disposition: str, reason: str, *, evidence: tuple[Ref, ...] = ()) -> PolicyDecision:
    return PolicyDecision.model_validate(
        {
            "gate": "execution",
            "disposition": disposition,
            "reason_code": reason,
            "evidence": evidence,
        }
    )


def reference(record: Record) -> Ref:
    return Ref(id=record.id, revision=record.revision)


def registered(principal: Principal, state: StateView) -> Principal | None:
    current = next((r for r in state.records if r.id == principal.id), None)
    if (
        isinstance(current, Principal)
        and current.authenticated_origin == principal.authenticated_origin
        and current.trust_domain == principal.trust_domain
    ):
        return current
    return None


class RuleRegistry:
    def __init__(self, rules: Mapping[str, Rule] | None = None) -> None:
        if any(not callable(rule) for rule in (rules or {}).values()):
            raise TypeError("Rules must be trusted Python callables")
        self._rules = MappingProxyType(dict(rules or {}))

    def evaluate(self, rule_id: str, inputs: Mapping[str, Record]) -> PolicyDecision:
        rule = self._rules.get(rule_id)
        if rule is None:
            return decision("require_review", "unknown_rule")
        try:
            result = rule(MappingProxyType(dict(inputs)))
            if not isinstance(result, PolicyDecision) or result.gate != "execution":
                return decision("require_review", "invalid_rule_result")
            return result
        except Exception:
            # Rule failures never grant permission or disclose sensitive inputs.
            return decision("require_review", "rule_failed")


def _path(value: str) -> PurePosixPath | None:
    if not value or "\x00" in value or "\\" in value or ".." in value.split("/"):
        return None
    return PurePosixPath(value)


def _scope_allows(
    grant: Scope, requested: Scope, *, owner_id: str, goal_id: str, principal_id: str
) -> bool:
    if grant.owner_id not in (None, owner_id) or grant.goal_id not in (None, goal_id):
        return False
    if requested.owner_id not in (None, owner_id) or requested.goal_id not in (None, goal_id):
        return False
    if grant.principal_ids and principal_id not in grant.principal_ids:
        return False
    if requested.principal_ids and requested.principal_ids != (principal_id,):
        return False
    for resource in requested.resource_paths:
        path = _path(resource)
        if path is None:
            return False
        if not any(
            root is not None and (path == root or root in path.parents)
            for root in (_path(value) for value in grant.resource_paths)
        ):
            return False
    return True


class Authority:
    def __init__(
        self,
        *,
        rules: RuleRegistry | None = None,
        worker_principals: Mapping[str, str] | None = None,
        capabilities: Mapping[str, CapabilitySpec] | None = None,
        resource_scopes: Mapping[str, ResourceExtractor] | None = None,
    ) -> None:
        self.rules = rules or RuleRegistry()
        self.worker_principals = MappingProxyType(dict(worker_principals or {}))
        self.capabilities = MappingProxyType(dict(capabilities or {}))
        if any(not callable(extractor) for extractor in (resource_scopes or {}).values()):
            raise TypeError("Resource extractors must be trusted Python callables")
        self.resource_scopes = MappingProxyType(dict(resource_scopes or {}))

    def check_command(
        self, principal: Principal, command: Command, state: StateView
    ) -> PolicyDecision:
        current = registered(principal, state)
        if current is None or "owner" not in current.roles:
            return decision("deny", "owner_required")
        records = {r.id: r for r in state.records}
        if isinstance(command, CreateGoal):
            if command.goal.owner_id != current.id:
                return decision("deny", "owner_scope_mismatch")
            return decision("authorize", "owner_command", evidence=(reference(current),))
        goal_id: str | None
        if isinstance(command, SetGrant):
            grant = command.grant
            if (
                grant.issuer_id != current.id
                or grant.scope.owner_id not in (None, current.id)
                or not isinstance(records.get(grant.subject_id), Principal)
            ):
                return decision("deny", "grant_scope_mismatch")
            existing = records.get(grant.id)
            if isinstance(existing, Grant) and existing.issuer_id != current.id:
                return decision("deny", "grant_scope_mismatch")
            if grant.scope.goal_id is None:
                return decision("authorize", "owner_command", evidence=(reference(current),))
            goal_id = grant.scope.goal_id
        elif isinstance(command, ResolveConflict) and command.goal_id is None:
            beliefs = [records.get(identifier) for identifier in command.belief_ids]
            if not beliefs or any(not isinstance(b, Belief) for b in beliefs):
                return decision("deny", "unknown_conflict")
            goals = [records.get(b.scope.goal_id or "") for b in beliefs if isinstance(b, Belief)]
            if any(not isinstance(g, Goal) or g.owner_id != current.id for g in goals):
                return decision("deny", "owner_scope_mismatch")
            return decision("authorize", "owner_command", evidence=(reference(current),))
        elif isinstance(command, ResolveUsage):
            reservation = records.get(command.reservation_id)
            budget = (
                records.get(reservation.budget_id) if isinstance(reservation, Reservation) else None
            )
            if not isinstance(budget, Budget):
                return decision("deny", "unknown_reservation")
            goal_id = budget.scope.goal_id
            if goal_id is None and budget.scope.owner_id == current.id:
                return decision(
                    "authorize", "owner_command", evidence=(reference(current), reference(budget))
                )
        else:
            goal_id = getattr(command, "goal_id", None)
            if goal_id is None:
                target = records.get(
                    getattr(command, "action_id", getattr(command, "artifact_id", ""))
                )
                if isinstance(target, ActionRecord):
                    goal_id = target.goal_id
                elif isinstance(target, ArtifactRef):
                    goal_id = target.scope.goal_id
        target_goal = records.get(goal_id or "")
        if not isinstance(target_goal, Goal) or target_goal.owner_id != current.id:
            return decision("deny", "owner_scope_mismatch")
        return decision(
            "authorize", "owner_command", evidence=(reference(current), reference(target_goal))
        )

    def check_constraints(
        self, constraints: tuple[Constraint, ...], inputs: Mapping[str, Record]
    ) -> PolicyDecision:
        pending: PolicyDecision | None = None
        constraint_refs = {reference(c) for c in constraints if c.hard}
        for constraint in constraints:
            if not constraint.hard:
                continue
            if constraint.enforcement != "rule":
                result = decision("require_review", "unresolved_hard_constraint")
            else:
                result = self.rules.evaluate(
                    constraint.enforcement_id or "", {**inputs, "constraint": constraint}
                )
            if result.reason_code == "constraint_conflict":
                conflict_refs = result.evidence or (reference(constraint),)
                if not set(conflict_refs) <= constraint_refs:
                    return decision(
                        "require_review",
                        "invalid_conflict_evidence",
                        evidence=(reference(constraint),),
                    )
                return decision("deny", "constraint_conflict", evidence=conflict_refs)
            if result.disposition != "authorize":
                result = result.model_copy(update={"evidence": (reference(constraint),)})
                if result.disposition == "deny":
                    return result
                pending = result
        return pending or decision("authorize", "constraints_satisfied")

    def check_dispatch(
        self, action: ActionRecord, state: StateView, now: datetime
    ) -> PolicyDecision:
        """Repeat current admission using exact, trusted proposal-acceptance lineage."""
        if action.episode_id is None or action.accepted_task_revision is None:
            return decision("deny", "missing_action_lineage")
        episode = next((r for r in state.records if r.id == action.episode_id), None)
        if not isinstance(episode, Episode) or episode.worker_id != action.worker_id:
            return decision("deny", "action_worker_binding_changed")
        intent = ActionIntent(
            id=action.id,
            episode_id=episode.id,
            fencing_token=episode.fencing_token,
            capability_id=action.capability_id,
            arguments=deep_thaw(action.arguments),
            scope=action.scope,
            read_set=action.read_set,
        )
        return self._check_action(intent, state, now, accepted=action)

    def check_action(self, intent: ActionIntent, state: StateView, now: datetime) -> PolicyDecision:
        return self._check_action(intent, state, now)

    def _check_action(
        self,
        intent: ActionIntent,
        state: StateView,
        now: datetime,
        *,
        accepted: ActionRecord | None = None,
    ) -> PolicyDecision:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Action check requires an aware timestamp")
        records = records_by_id(state)
        episode = records.get(intent.episode_id)
        if not isinstance(episode, Episode):
            return decision("deny", "unknown_episode")
        task = records.get(episode.task_id)
        packet = records.get(episode.context_packet_id)
        target = records.get(task.goal_id) if isinstance(task, TaskFrame) else None
        if (
            not isinstance(task, TaskFrame)
            or not isinstance(packet, ContextPacket)
            or not isinstance(target, Goal)
        ):
            return decision("deny", "unknown_action_context")
        worker = records.get(self.worker_principals.get(episode.worker_id, ""))
        owner = records.get(target.owner_id)
        if (
            not isinstance(worker, Principal)
            or "worker" not in worker.roles
            or not isinstance(owner, Principal)
            or "owner" not in owner.roles
        ):
            return decision("deny", "unknown_worker_or_owner")
        accepted_current = accepted is not None and (
            episode.status == "accepted"
            and episode.revision == accepted.accepted_episode_revision
            and task.status in {"suspended", "verifying"}
            and task.revision == accepted.accepted_task_revision
            and task.fencing_token == accepted.accepted_fencing_token
        )
        if accepted is not None and not accepted_current:
            return decision("stale", "stale_accepted_action_context")
        if target.status != "active" or (task.status != "running" and not accepted_current):
            return decision("deny", "inactive_goal_or_task")
        if (
            (episode.status != "running" and not accepted_current)
            or now >= episode.deadline
            or "action_intent" not in episode.allowed_proposal_types
            or intent.fencing_token != episode.fencing_token
            or (intent.fencing_token != task.fencing_token and not accepted_current)
            or (packet.fencing_token != task.fencing_token and not accepted_current)
            or packet.episode_id != episode.id
            or packet.task.id != task.id
            or packet.goal.id != target.id
            or packet.goal.revision != target.revision
            or (packet.task.revision != task.revision and not accepted_current)
        ):
            return decision("stale", "stale_action_context")
        for ref in (
            *(packet.mandatory_read_set if accepted is None else ()),
            *((reference(b) for b in packet.beliefs) if accepted is None else ()),
            *intent.read_set,
            *intent.affected_records,
        ):
            record = records.get(ref.id)
            if (
                record is None
                or record.revision != ref.revision
                or (ref.kind != "record" and getattr(record, "kind", None) != ref.kind)
            ):
                return decision("stale", "stale_action_read_set")
        from .scheduler import _fresh

        indexed = records_by_id(state)
        dependencies = _proof_roots(task, intent.read_set, indexed)
        if not _fresh(task.model_copy(update={"dependencies": dependencies}), indexed, now):
            return decision("stale", "expired_action_support")
        from .reducers import _dependency_satisfied

        materialized = {r.id: r for r in state.records}
        for ref in task.dependencies:
            item = indexed.get(ref.id)
            if item is None or not _dependency_satisfied(item, materialized):
                return decision("stale", "unsettled_action_dependency")
        pending = list(dependencies)
        seen: set[str] = set()
        while pending:
            dependency = pending.pop()
            item = indexed.get(dependency.id)
            if (
                item is None
                or item.revision != dependency.revision
                or (dependency.kind != "record" and dependency.kind != getattr(item, "kind", None))
            ):
                return decision("stale", "stale_action_support")
            if item.id in seen:
                continue
            if isinstance(item, Criterion):
                finding = next(
                    (
                        r
                        for r in state.records
                        if isinstance(r, VerificationResult)
                        and r.criterion_id == item.id
                        and r.disposition == "accept"
                    ),
                    None,
                )
                if finding is None:
                    return decision("stale", "unsupported_action_criterion")
                pending.extend((*item.inputs, *item.required_evidence, reference(finding)))
            if isinstance(item, VerificationResult):
                if item.disposition != "accept":
                    return decision("stale", "unsupported_action_finding")
                pending.extend((*item.read_set, *item.evidence))
            if isinstance(item, Belief):
                if item.epistemic_status != "supported":
                    return decision("stale", "unsupported_action_belief")
                if item.id not in seen:
                    pending.extend(item.evidence)
            if isinstance(item, ArtifactRef) and not item.valid:
                return decision("stale", "invalid_action_artifact")
            seen.add(dependency.id)
        capability = self.capabilities.get(intent.capability_id)
        extractor = self.resource_scopes.get(intent.capability_id)
        if capability is None or capability.id != intent.capability_id or extractor is None:
            return decision("deny", "unknown_capability_binding")
        if capability.effect_class == "human_only":
            return decision("deny", "human_only_capability")
        try:
            resources = extractor(intent)
            if not isinstance(resources, Scope):
                return decision("deny", "invalid_resource_scope")
        except Exception:
            return decision("deny", "invalid_resource_scope")
        if (
            resources.resource_paths != intent.scope.resource_paths
            or resources.owner_id not in (None, target.owner_id)
            or resources.goal_id not in (None, target.id)
            or not _scope_allows(
                intent.scope,
                resources,
                owner_id=target.owner_id,
                goal_id=target.id,
                principal_id=worker.id,
            )
        ):
            return decision("deny", "argument_scope_mismatch")
        constraints = self.check_constraints(
            target.constraints, {"intent": intent, "goal": target, "task": task}
        )
        if constraints.disposition != "authorize":
            return constraints
        review: PolicyDecision | None = None
        for grant in (r for r in state.records if isinstance(r, Grant)):
            if (
                grant.subject_id != worker.id
                or grant.issuer_id != target.owner_id
                or grant.revoked
                or (grant.expires_at is not None and now >= grant.expires_at)
                or intent.capability_id not in grant.capabilities
                or not _scope_allows(
                    grant.scope,
                    intent.scope,
                    owner_id=target.owner_id,
                    goal_id=target.id,
                    principal_id=worker.id,
                )
            ):
                continue
            restrictions = self.check_constraints(
                grant.restrictions, {"intent": intent, "goal": target, "grant": grant}
            )
            if restrictions.disposition == "authorize":
                return decision(
                    "authorize",
                    "scoped_grant",
                    evidence=(
                        reference(grant),
                        reference(owner),
                        reference(worker),
                        reference(target),
                        reference(task),
                        reference(episode),
                    ),
                )
            review = restrictions
        return review or decision("deny", "no_current_grant")
