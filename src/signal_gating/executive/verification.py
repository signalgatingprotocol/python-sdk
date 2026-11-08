"""Evidence-backed completion prepared for the sole owning writer.

evaluate/reverify/consume_request return event-only CommitBatch writes. Commit
before reporting completion. They perform local reads, never execute commands or
call models. Registered checks must already have been authorized, reserved,
dispatched and accounted by Dispatcher. There is deliberately no arbitrary async
verifier callback that could spend without that boundary. Unconfigured model
criteria remain visible as unconfigured judgments requiring human attention.

Use the same current ArtifactStore, authenticated verifier principal, Authority
and trusted LocalWorkspace/CapabilityRegistry configuration as the coordinator.
All original materialized records participate in final CAS. Local reads happen
outside transactions; no library can prevent a hostile host changing bytes after
the final read. A subsequent reverify rechecks actual bytes and workspace inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Protocol

from .actions import CapabilityRegistry
from .adapters.local_tools import LocalWorkspace, StoreArtifacts
from .artifacts import ArtifactStore
from .authority import Authority, registered
from .budgets import _overage
from .errors import ConflictError, ExecutiveError
from .memory import invalidate_dependents, records_by_id, reference
from .messages import AttachEvidence, CommitBatch, CompletionClaim, Reverify
from .records import (
    ActionRecord,
    ArtifactRef,
    Belief,
    Commitment,
    ContextPacket,
    Criterion,
    CriterionApproval,
    CriterionGap,
    DecisionReceipt,
    Disposition,
    Episode,
    ExecutiveSignal,
    Goal,
    JournalEvent,
    PersistentRecord,
    Principal,
    Record,
    Ref,
    Reservation,
    StateView,
    TaskFrame,
    Usage,
    VerificationResult,
    WakeCondition,
    goal_definition_digest,
    utc_now,
)
from .reducers import _dependency_satisfied, reduce_event
from .scheduler import _fresh


class Verifier(Protocol):
    async def verify(self, criterion: Criterion, state: StateView) -> VerificationResult: ...


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _goal(criterion: Criterion, state: StateView) -> Goal:
    for item in state.records:
        if isinstance(item, Goal) and criterion in item.criteria:
            return item
    raise ConflictError("stale_criterion", "Verification requires the current embedded criterion")


class ArtifactVerifier:
    """The all_exist_and_intact rule establishes availability and integrity only."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        principal: Principal,
        evidence: tuple[Ref, ...] = (),
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.artifacts, self.principal, self.evidence, self.clock = (
            artifacts,
            principal,
            evidence,
            clock,
        )

    def _result(
        self,
        criterion: Criterion,
        state: StateView,
        disposition: Disposition,
        reason: str,
        *,
        evidence: tuple[Ref, ...] = (),
        reads: tuple[Ref, ...] = (),
        policy: str = "verification-v1",
    ) -> VerificationResult:
        goal = _goal(criterion, state)
        return VerificationResult.model_validate(
            {
                "created_at": self.clock(),
                "criterion_id": criterion.id,
                "verifier_id": criterion.verifier_id,
                "disposition": disposition,
                "reason_code": reason,
                "policy_version": policy,
                "read_set": tuple(
                    dict.fromkeys(
                        (
                            reference(goal),
                            reference(criterion),
                            *(reference(c) for c in goal.constraints),
                            *reads,
                        )
                    )
                ),
                "evidence": evidence,
                "gaps": ()
                if disposition == "accept"
                else (
                    CriterionGap(
                        criterion_id=criterion.id,
                        reason_code=reason,
                        description=f"{criterion.description}: {reason}",
                        evidence_needed=criterion.required_evidence,
                    ),
                ),
                "usage": Usage.zero(),
            }
        )

    def _support(self, criterion: Criterion, state: StateView) -> tuple[Ref, ...]:
        """Definition reads fence versions; explicit dependencies require proof."""
        goal = _goal(criterion, state)
        indexed = records_by_id(state)
        pending = list((*criterion.inputs, *criterion.required_evidence))
        seen: set[str] = set()
        reads: list[Ref] = []
        while pending:
            ref = pending.pop()
            item = indexed.get(ref.id)
            if (
                item is None
                or item.revision != ref.revision
                or (ref.kind != "record" and ref.kind != getattr(item, "kind", None))
            ):
                raise ValueError("stale_verification_input")
            if item.id in seen:
                continue
            seen.add(item.id)
            reads.append(reference(item))
            if isinstance(item, ArtifactRef):
                self.artifacts.read(item, principal=self.principal, goal_id=goal.id)
            elif isinstance(item, Belief):
                now = self.clock()
                if (
                    item.epistemic_status != "supported"
                    or (item.valid_from and now < item.valid_from)
                    or (item.valid_until and now >= item.valid_until)
                ):
                    raise ValueError("invalid_verification_support")
                pending.extend(item.evidence)
            elif isinstance(item, Criterion):
                pending.extend((*item.inputs, *item.required_evidence))
            elif isinstance(item, VerificationResult):
                if item.disposition != "accept":
                    raise ValueError("invalid_verification_support")
                pending.extend((*item.read_set, *item.evidence))
        return tuple(reads)

    async def verify(self, criterion: Criterion, state: StateView) -> VerificationResult:
        goal = _goal(criterion, state)
        refs = self.evidence
        try:
            reads = self._support(criterion, state)
            declared = tuple(
                r for r in reads if isinstance(records_by_id(state).get(r.id), ArtifactRef)
            )
            refs = declared or self.evidence
            if (
                criterion.evaluator_type != "deterministic"
                or criterion.acceptance_rule != "all_exist_and_intact"
            ):
                return self._result(criterion, state, "needs_human", "unsupported_acceptance_rule")
            if not refs:
                return self._result(criterion, state, "insufficient_evidence", "missing_artifact")
            for ref in refs:
                current = records_by_id(state).get(ref.id)
                if (
                    not isinstance(current, ArtifactRef)
                    or current.revision != ref.revision
                    or ref.kind not in {"record", "artifact"}
                ):
                    raise ValueError("stale_artifact")
                self.artifacts.read(current, principal=self.principal, goal_id=goal.id)
            return self._result(
                criterion,
                state,
                "accept",
                "artifacts_intact",
                evidence=refs,
                reads=(*reads, *refs),
                policy="artifact-integrity-v1",
            )
        except (ExecutiveError, ValueError, OSError):
            return self._result(
                criterion,
                state,
                "insufficient_evidence",
                "artifact_unavailable",
                evidence=refs,
                policy="artifact-integrity-v1",
            )


class HumanApprovalVerifier(ArtifactVerifier):
    async def verify(self, criterion: Criterion, state: StateView) -> VerificationResult:
        goal = _goal(criterion, state)
        try:
            reads = self._support(criterion, state)
            if (
                criterion.evaluator_type != "human"
                or criterion.acceptance_rule != "explicit_owner_approval"
            ):
                raise ValueError("unsupported_acceptance_rule")
            owner = next((r for r in state.records if r.id == goal.owner_id), None)
            if not isinstance(owner, Principal) or "owner" not in owner.roles:
                raise ValueError("owner_required")
            for item in reversed(state.records):
                if (
                    not isinstance(item, CriterionApproval)
                    or item.owner_id != owner.id
                    or item.goal.id != goal.id
                    or item.criterion != reference(criterion)
                    or item.goal_definition_digest != goal_definition_digest(goal)
                ):
                    continue
                declared = {
                    r
                    for r in (*criterion.inputs, *criterion.required_evidence)
                    if isinstance(records_by_id(state).get(r.id), ArtifactRef)
                }
                if not {(r.id, r.revision) for r in declared} <= {
                    (r.id, r.revision) for r in item.artifacts
                }:
                    continue
                for ref in item.artifacts:
                    current = records_by_id(state).get(ref.id)
                    if (
                        not isinstance(current, ArtifactRef)
                        or current.revision != ref.revision
                        or ref.kind not in {"record", "artifact"}
                    ):
                        raise ValueError("stale_approval_artifact")
                    self.artifacts.read(current, principal=self.principal, goal_id=goal.id)
                return self._result(
                    criterion,
                    state,
                    "accept",
                    "owner_approved",
                    evidence=item.artifacts,
                    reads=(*reads, reference(item), reference(owner), *item.artifacts),
                    policy="owner-approval-v1",
                )
        except (ExecutiveError, ValueError, OSError):
            pass
        return self._result(
            criterion,
            state,
            "needs_human",
            "explicit_owner_approval_required",
            policy="owner-approval-v1",
        )


def _historical(artifacts: ArtifactStore, ref: Ref) -> Record | None:
    """Canonical post-images, including embedded definitions, for lineage only."""
    current = records_by_id(artifacts.store.view()).get(ref.id)
    if current is not None and current.revision == ref.revision:
        return current
    for event in artifacts.store.events():
        if event.event_type != "state.changed":
            continue
        for image in event.model_dump(mode="json")["payload"]["records"]:
            candidates = (image, *image.get("criteria", ()), *image.get("constraints", ()))
            for candidate in candidates:
                if candidate.get("id") == ref.id and candidate.get("revision") == ref.revision:
                    from pydantic import TypeAdapter

                    return TypeAdapter(PersistentRecord).validate_python(candidate)
    # Goal controls/invalidation derive revisions instead of carrying explicit
    # state.changed images. Replay is the authoritative source for those images.
    historical = StateView()
    for event in artifacts.store.events():
        historical = reduce_event(historical, event)
        candidate_record = records_by_id(historical).get(ref.id)
        if candidate_record is not None and candidate_record.revision == ref.revision:
            return candidate_record
    return None


def _lineage_current(
    artifacts: ArtifactStore,
    ref: Ref,
    state: StateView,
    *,
    lifecycle_ids: frozenset[str] = frozenset(),
) -> bool:
    """Allow only identified lifecycle bookkeeping, never changed action inputs."""
    current = records_by_id(state).get(ref.id)
    if current is None or (ref.kind != "record" and ref.kind != getattr(current, "kind", None)):
        return False
    if current.revision == ref.revision:
        return True
    if current.id not in lifecycle_ids:
        return False
    old = _historical(artifacts, ref)
    if isinstance(current, Goal) and isinstance(old, Goal):
        return goal_definition_digest(current) == goal_definition_digest(old)
    if isinstance(current, TaskFrame) and isinstance(old, TaskFrame):
        excluded = {
            "revision",
            "status",
            "fencing_token",
            "pending_action_ids",
            "resume_brief",
            "wake_condition",
            "failure_count",
            "failure_signature",
            "failure_reason",
            "eligible_since",
            "last_scheduled_at",
        }
        return old.model_dump(exclude=excluded) == current.model_dump(exclude=excluded)
    if isinstance(current, Commitment) and isinstance(old, Commitment):
        return old.model_dump(exclude={"revision", "status"}) == current.model_dump(
            exclude={"revision", "status"}
        )
    if isinstance(current, Episode) and isinstance(old, Episode):
        excluded = {"revision", "status", "usage"}
        return old.model_dump(exclude=excluded) == current.model_dump(exclude=excluded)
    return False


class LocalCheckVerifier(ArtifactVerifier):
    """Authenticate existing execution evidence without launching another check."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        principal: Principal,
        workspace: LocalWorkspace,
        registry: CapabilityRegistry,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        super().__init__(artifacts, principal=principal, clock=clock)
        self.workspace, self.registry = workspace, registry
        if workspace.artifacts is None or workspace.artifacts.store is not artifacts.store:
            raise ValueError("Local check and verification must use the same owning Store")

    async def verify(self, criterion: Criterion, state: StateView) -> VerificationResult:
        goal = _goal(criterion, state)
        try:
            reads = self._support(criterion, state)
            if (
                criterion.evaluator_type != "deterministic"
                or criterion.acceptance_rule != "exit_zero"
                or criterion.check_id is None
                or not criterion.workspace_inputs
            ):
                raise ValueError("unsupported_check_configuration")
            config = self.workspace.checks.get(criterion.check_id)
            spec = self.registry.specs.get("run_check")
            if config is None or spec is None:
                raise ValueError("missing_registered_check")
            bridge = StoreArtifacts(self.artifacts.store)
            current_hashes = self.workspace.input_digests(criterion)
            for action in state.records:
                if (
                    not isinstance(action, ActionRecord)
                    or action.goal_id != goal.id
                    or action.capability_id != "run_check"
                    or action.status != "succeeded"
                ):
                    continue
                try:
                    episode = records_by_id(state).get(action.episode_id or "")
                    packet = (
                        records_by_id(state).get(episode.context_packet_id)
                        if isinstance(episode, Episode)
                        else None
                    )
                    worker = records_by_id(state).get(action.origin_principal_id or "")
                    if (
                        not isinstance(episode, Episode)
                        or episode.worker_id != action.worker_id
                        or episode.status != "accepted"
                        or not isinstance(packet, ContextPacket)
                        or packet.goal.id != goal.id
                        or packet.task.id != action.task_id
                        or not isinstance(worker, Principal)
                        or "worker" not in worker.roles
                        or not {goal.id, criterion.id, action.task_id, episode.id, packet.id}
                        <= {r.id for r in action.read_set}
                    ):
                        continue
                    if (
                        action.binding_id != spec.binding_id
                        or action.binding_version != spec.binding_version
                        or action.dispatch_attempts < 1
                        or action.dispatched_at is None
                    ):
                        continue
                    arguments = action.model_dump(mode="json")["arguments"]
                    if (
                        action.arguments_digest != _digest(arguments)
                        or set(arguments) != {"command_id", "criterion_ref", "expected_sha256"}
                        or arguments["command_id"] != criterion.check_id
                        or Ref.model_validate(arguments["criterion_ref"]).id != criterion.id
                        or Ref.model_validate(arguments["criterion_ref"]).revision
                        != criterion.revision
                        or Ref.model_validate(arguments["criterion_ref"]).kind
                        not in {"record", "criterion"}
                        or arguments["expected_sha256"] != current_hashes
                    ):
                        continue
                    if not all(
                        _lineage_current(
                            self.artifacts,
                            ref,
                            state,
                            lifecycle_ids=frozenset(
                                (
                                    action.goal_id,
                                    action.task_id,
                                    action.episode_id or "",
                                    packet.task.commitment_id,
                                )
                            ),
                        )
                        for ref in action.read_set
                    ):
                        continue
                    old_goal = _historical(
                        self.artifacts, Ref(id=goal.id, revision=action.constraint_revision)
                    )
                    if not isinstance(old_goal, Goal) or goal_definition_digest(
                        old_goal
                    ) != goal_definition_digest(goal):
                        continue
                    reservation = next(
                        (r for r in state.records if r.id == action.reservation_id), None
                    )
                    if (
                        not isinstance(reservation, Reservation)
                        or reservation.subject_id != action.id
                    ):
                        continue
                    matches = bridge.receipts(action)
                    identity = {
                        "schema": "workspace-receipt-v1",
                        "action_id": action.id,
                        "idempotency_key": action.idempotency_key,
                        "arguments_digest": action.arguments_digest,
                        "arguments": arguments,
                        "goal_id": action.goal_id,
                        "origin_principal_id": action.origin_principal_id,
                        "binding_id": spec.binding_id,
                        "binding_version": spec.binding_version,
                        "capability_id": "run_check",
                        "configuration_digest": self.workspace.configuration_digest,
                    }
                    if any(
                        any(data.get(k) != v for k, v in identity.items()) for _, data in matches
                    ):
                        continue
                    completed = [(ref, d) for ref, d in matches if d.get("phase") == "completed"]
                    if len(completed) != 1:
                        continue
                    receipt, data = completed[0]
                    if action.external_receipt != receipt.id or receipt.id not in {
                        r.id for r in action.evidence
                    }:
                        continue
                    prepared_ref = Ref.model_validate(data["prepared_ref"])
                    prepared, _ = bridge.receipt_evidence(
                        action, prepared_ref, principal=self.principal
                    )
                    prepared_data = next(
                        (
                            d
                            for r, d in matches
                            if r.id == prepared.id and d.get("phase") == "prepared"
                        ),
                        None,
                    )
                    expected = {
                        "criterion_ref": arguments["criterion_ref"],
                        "argv": list(config.argv),
                        "check_configuration": config.model_dump(mode="json"),
                        "launcher": self.workspace.launcher_version,
                        "before_sha256": current_hashes,
                    }
                    if prepared_data is None or any(
                        data.get(k) != v or prepared_data.get(k) != v for k, v in expected.items()
                    ):
                        continue
                    if (
                        data.get("outcome") != "succeeded"
                        or data.get("reason") != ""
                        or type(data.get("returncode")) is not int
                        or data["returncode"] != 0
                        or data.get("output_truncated") is not False
                        or data.get("process_group_cleanup") != "group_absent"
                        or data.get("after_sha256") != current_hashes
                    ):
                        continue
                    output_refs = tuple(Ref.model_validate(r) for r in data["outputs"])
                    if len(output_refs) != 1:
                        continue
                    output, output_bytes = bridge.receipt_evidence(
                        action, output_refs[0], principal=self.principal
                    )
                    receipt_artifact, _ = bridge.receipt_evidence(
                        action, receipt, principal=self.principal
                    )
                    evidence = (reference(output), reference(prepared), reference(receipt_artifact))
                    if len(output_bytes) > config.max_output_bytes or not {
                        r.id for r in evidence
                    } <= {r.id for r in action.evidence}:
                        continue
                    # All observed evidence and the settled action join final freshness/CAS.
                    return self._result(
                        criterion,
                        state,
                        "accept",
                        "registered_check_passed",
                        evidence=evidence,
                        reads=(
                            *reads,
                            reference(action),
                            reference(reservation),
                            *(reference(records_by_id(state)[r.id]) for r in action.read_set),
                            *evidence,
                        ),
                        policy="registered-check-v1:" + self.workspace.configuration_digest,
                    )
                except (ExecutiveError, ValueError, OSError, KeyError, TypeError):
                    continue
        except (ExecutiveError, ValueError, OSError, KeyError, TypeError):
            pass
        return self._result(
            criterion,
            state,
            "insufficient_evidence",
            "registered_check_unproven",
            policy="registered-check-v1",
        )


class VerificationService:
    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        principal: Principal,
        authority: Authority | None = None,
        local_checks: Mapping[str, tuple[LocalWorkspace, CapabilityRegistry]] | None = None,
        verifiers: Mapping[str, Verifier] | None = None,
        repair_rounds: int = 2,
        model_checks: Mapping[str, tuple[CapabilityRegistry, str, str]] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.artifacts, self.principal, self.authority, self.clock = (
            artifacts,
            principal,
            authority or Authority(),
            clock,
        )
        if type(repair_rounds) is not int or repair_rounds < 0:
            raise ValueError("repair_rounds must be a nonnegative integer")
        self.repair_rounds = repair_rounds
        self.verifiers = MappingProxyType(dict(verifiers or {}))
        self.local_checks = MappingProxyType(dict(local_checks or {}))
        self.model_checks = MappingProxyType(dict(model_checks or {}))

    def _configuration(self) -> str:
        local = {
            key: {
                "workspace": workspace.configuration_digest,
                "checks": {
                    name: check.model_dump(mode="json") for name, check in workspace.checks.items()
                },
                "binding": registry.specs["run_check"].model_dump(mode="json")
                if "run_check" in registry.specs
                else None,
            }
            for key, (workspace, registry) in self.local_checks.items()
        }
        models = {
            key: {
                "capability": capability,
                "policy": policy,
                "binding": registry.specs[capability].model_dump(mode="json")
                if capability in registry.specs
                else None,
            }
            for key, (registry, capability, policy) in self.model_checks.items()
        }
        return _digest(
            {
                "local": local,
                "models": models,
                "injected": sorted(self.verifiers),
                "repair_rounds": self.repair_rounds,
            }
        )

    def judgment_arguments(self, criterion: Criterion, state: StateView) -> dict[str, Any]:
        """Build exact model action arguments from trusted selected configuration.

        Kernel14 submits these as a normal ActionIntent. Dispatcher must commit
        its subject reservation before calling the bounded configured adapter.
        This helper does not admit, reserve, execute or confer any grant.
        """
        self._current(state)
        goal = _goal(criterion, state)
        if criterion.evaluator_type != "model" or criterion.verifier_id not in self.model_checks:
            raise ValueError("unconfigured_model_judgment")
        _, _, policy = self.model_checks[criterion.verifier_id]
        reads = ArtifactVerifier(
            self.artifacts, principal=self.principal, clock=self.clock
        )._support(criterion, state)
        return {
            "criterion_ref": reference(criterion).model_dump(mode="json"),
            "goal_definition_digest": goal_definition_digest(goal),
            "verifier_id": criterion.verifier_id,
            "policy_version": policy,
            "verification_inputs": [
                r.model_dump(mode="json") for r in sorted(reads, key=lambda r: (r.id, r.revision))
            ],
        }

    def _model_finding(
        self,
        criterion: Criterion,
        result: VerificationResult,
        action: ActionRecord,
        state: StateView,
    ) -> VerificationResult:
        """Trusted coordinator observation, never an executable model callback.

        The configured adapter parses its bounded output into this typed result.
        Dispatcher.record_observation must already have committed the same Usage
        through BudgetPolicy.observe_usage, including late conflicts. Worker
        signals/proposals and approval-shaped output cannot invoke this host API.
        """
        goal = _goal(criterion, state)
        registry, capability, policy = self.model_checks[criterion.verifier_id]
        spec = registry.specs.get(capability)
        reservation = records_by_id(state).get(action.reservation_id or "")
        expected = self.judgment_arguments(criterion, state)
        if (
            spec is None
            or action.capability_id != capability
            or action.binding_id != spec.binding_id
            or action.binding_version != spec.binding_version
            or action.status != "succeeded"
            or action.goal_id != goal.id
            or action.dispatched_at is None
            or action.dispatch_attempts < 1
            or action.model_dump(mode="json")["arguments"] != expected
            or action.arguments_digest != _digest(expected)
            or not isinstance(reservation, Reservation)
            or reservation.subject_id != action.id
            or reservation.status not in {"settled", "uncertain"}
            or result.usage != (reservation.observed_usage or reservation.usage)
            or result.criterion_id != criterion.id
            or result.verifier_id != criterion.verifier_id
            or not result.evidence
            or not set(result.evidence) <= set(action.evidence)
            or not all(
                _lineage_current(
                    self.artifacts,
                    r,
                    state,
                    lifecycle_ids=frozenset(
                        (
                            action.goal_id,
                            action.task_id,
                            action.episode_id or "",
                        )
                    ),
                )
                for r in (*action.read_set, *result.read_set)
            )
        ):
            raise ValueError("unbound_model_judgment")
        for ref in result.evidence:
            artifact = records_by_id(state).get(ref.id)
            if (
                not isinstance(artifact, ArtifactRef)
                or artifact.revision != ref.revision
                or ref.kind not in {"record", "artifact"}
            ):
                raise ValueError("stale_judgment_evidence")
            self.artifacts.read(artifact, principal=self.principal, goal_id=goal.id)
        base = ArtifactVerifier(self.artifacts, principal=self.principal, clock=self.clock)
        supported = base._support(criterion, state)
        material = (
            *supported,
            reference(goal),
            reference(criterion),
            *(reference(c) for c in goal.constraints),
        )
        if not {r.id for r in material} <= {r.id for r in action.read_set}:
            raise ValueError("missing_judgment_reads")
        finding = base._result(
            criterion,
            state,
            result.disposition,
            "model_judgment_" + result.disposition,
            evidence=result.evidence,
            reads=tuple(
                reference(records_by_id(state)[r.id])
                for r in (
                    *material,
                    *action.read_set,
                    reference(action),
                    reference(reservation),
                    *result.evidence,
                )
            ),
            policy="model-judgment-v1:" + policy,
        )
        return finding.model_copy(update={"usage": result.usage})

    async def evaluate_judgment(
        self, action_id: str, result: VerificationResult, state: StateView
    ) -> CommitBatch:
        """Consume an already-dispatched/accounted trusted adapter observation.

        Late observations go to Dispatcher.record_observation first, even if
        this freshness check refuses them. No secondary accounting ledger exists.
        """
        self._current(state)
        criterion = records_by_id(state).get(result.criterion_id)
        action = records_by_id(state).get(action_id)
        try:
            if not isinstance(criterion, Criterion) or not isinstance(action, ActionRecord):
                raise ValueError("unknown_judgment_context")
            accepted = self._model_finding(criterion, result, action, state)
            return await self._evaluate(
                _goal(criterion, state), state, action.id, None, {}, {criterion.id: accepted}
            )
        except (ExecutiveError, ValueError, OSError, KeyError, TypeError):
            # A measured conflict may have changed the ledger even when its
            # conservative envelope cannot authenticate this exact raw report.
            # Recheck the existing proof so completed work cannot retain success
            # across its now-unsettled accounting. Never edit or drop that bill.
            if isinstance(action, ActionRecord):
                goal = records_by_id(state).get(action.goal_id)
                if isinstance(goal, Goal) and self._unsettled(goal, state):
                    return await self._evaluate(goal, state, action.id, None, {})
            return self._finish(state, action_id, "insufficient_evidence", "unbound_model_judgment")

    def _current(self, state: StateView) -> None:
        if self.artifacts.store.view() != state:
            raise ConflictError(
                "stale_verification_state", "Verification requires the current Store snapshot"
            )
        principal = registered(self.principal, state)
        if principal is None or not set(principal.roles) & {"owner", "verifier"}:
            raise ConflictError("verifier_required", "Verification requires a registered verifier")

    def _event(self, records: tuple[PersistentRecord, ...], subject: Ref) -> JournalEvent:
        return JournalEvent(
            sequence=1,
            created_at=self.clock(),
            event_type="state.changed",
            producer_id="verification",
            principal_id=self.principal.id,
            subject=subject,
            payload={"records": [r.model_dump(mode="json") for r in records]},
        )

    def _finish(
        self,
        state: StateView,
        subject: str,
        disposition: Disposition,
        reason: str,
        events: tuple[JournalEvent, ...] = (),
        proof: tuple[Ref, ...] = (),
    ) -> CommitBatch:
        receipt = DecisionReceipt(
            created_at=self.clock(),
            subject_id=subject,
            disposition=disposition,
            reason_code=reason,
            policy_version="verification-v1",
            event_ids=tuple(e.id for e in events),
            state_revisions=proof,
        )
        event = self._event((receipt,), Ref(id=self.principal.id))
        return CommitBatch(
            events=(*events, event),
            expected_revisions=tuple(reference(r) for r in state.records),
            response_receipt=receipt,
        )

    async def evaluate(self, claim: CompletionClaim, state: StateView) -> CommitBatch:
        self._current(state)
        records = records_by_id(state)
        episode = records.get(claim.episode_id)
        task = records.get(episode.task_id) if isinstance(episode, Episode) else None
        goal = records.get(task.goal_id) if isinstance(task, TaskFrame) else None
        signal = records.get(claim.id)
        if (
            not isinstance(goal, Goal)
            or not isinstance(task, TaskFrame)
            or not isinstance(episode, Episode)
        ):
            return self._finish(state, claim.id, "stale", "missing_completion_context")
        if (
            not isinstance(signal, ExecutiveSignal)
            or signal.signal_kind != "completion_claim"
            or signal.producer_id != "worker-proposal"
            or signal.payload.get("proposal") != claim.model_dump(mode="json")
            or signal.scope.goal_id != goal.id
            or episode.status != "accepted"
            or task.status != "verifying"
            or goal.status != "active"
            or task.fencing_token != claim.fencing_token
        ):
            return self._finish(state, claim.id, "stale", "stale_completion_claim")
        if not set(claim.criterion_ids) <= {c.id for c in goal.criteria}:
            return self._finish(state, claim.id, "stale", "unknown_criterion")
        for ref in (*claim.read_set, *claim.affected_records):
            if ref.id == task.id and task.revision == ref.revision + 1:
                continue  # ProposalValidator's own verifying transition.
            if not _lineage_current(self.artifacts, ref, state):
                return self._finish(state, claim.id, "stale", "stale_completion_inputs")
        return await self._evaluate(
            goal, state, claim.id, task, {c.id: claim.artifacts for c in goal.criteria}
        )

    async def reverify(self, goal_id: str, state: StateView) -> CommitBatch:
        self._current(state)
        goal = records_by_id(state).get(goal_id)
        if not isinstance(goal, Goal):
            raise ConflictError("unknown_goal", "Reverification requires a persisted goal")
        return await self._evaluate(goal, state, goal.id, None, {})

    async def consume_request(self, signal: ExecutiveSignal, state: StateView) -> CommitBatch:
        self._current(state)
        if signal not in state.records or signal.signal_kind not in {"attach_evidence", "reverify"}:
            return self._finish(state, signal.id, "stale", "unknown_verification_request")
        principal = records_by_id(state).get(signal.principal_id)
        goal = records_by_id(state).get(signal.scope.goal_id or "")
        if (
            not isinstance(principal, Principal)
            or "owner" not in principal.roles
            or not isinstance(goal, Goal)
            or principal.id != goal.owner_id
        ):
            return self._finish(state, signal.id, "deny", "owner_required")
        try:
            command = (
                AttachEvidence if signal.signal_kind == "attach_evidence" else Reverify
            ).model_validate(signal.payload["command"])
            if command.goal_id != goal.id:
                raise ValueError("request_scope_mismatch")
            ids = (
                (
                    (command.criterion_id,)
                    if command.criterion_id
                    else tuple(c.id for c in goal.criteria)
                )
                if isinstance(command, AttachEvidence)
                else command.criterion_ids or tuple(c.id for c in goal.criteria)
            )
            if not set(ids) <= {c.id for c in goal.criteria}:
                raise ValueError("unknown_criterion")
            extras = (
                {identifier: command.evidence for identifier in ids}
                if isinstance(command, AttachEvidence)
                else {}
            )
            return await self._evaluate(goal, state, signal.id, None, extras)
        except (ValueError, KeyError, TypeError):
            return self._finish(state, signal.id, "stale", "invalid_verification_request")

    def _unsettled(self, goal: Goal, state: StateView) -> bool:
        indexed = records_by_id(state)
        for item in state.records:
            if (
                isinstance(item, ActionRecord)
                and item.goal_id == goal.id
                and item.status in {"dispatching", "outcome_unknown", "reconciling", "needs_human"}
            ):
                return True
            if isinstance(item, Reservation):
                subject = indexed.get(item.subject_id)
                if isinstance(subject, Episode):
                    subject = indexed.get(subject.task_id)
                related = (
                    getattr(subject, "goal_id", None) == goal.id
                    or getattr(subject, "id", None) == goal.id
                )
                if related and (
                    item.status in {"reserved", "uncertain"}
                    or item.accounting_conflict is not None
                    or _overage(item)
                ):
                    return True
        return False

    def _selected_evidence(
        self, criterion: Criterion, goal: Goal, state: StateView
    ) -> tuple[Ref, ...]:
        """Reuse a stable selection, never discover an arbitrary latest artifact.

        Finding creation is a trusted typed journal boundary. Its named Goal
        image binds the full material criterion/configuration; lifecycle-only
        revisions preserve that intent. Every selected ref is re-read afterward.
        Failed rechecks retain the selection so absence cannot switch identities.
        """
        for finding in state.records:
            if (
                not isinstance(finding, VerificationResult)
                or finding.criterion_id != criterion.id
                or finding.verifier_id != criterion.verifier_id
                or finding.policy_version != "artifact-integrity-v1"
                or criterion.verifier_id != "artifact-integrity-v1"
            ):
                continue
            goal_ref = next((r for r in finding.read_set if r.id == goal.id), None)
            original = _historical(self.artifacts, goal_ref) if goal_ref is not None else None
            if (
                isinstance(original, Goal)
                and goal_definition_digest(original) == goal_definition_digest(goal)
                and criterion in original.criteria
            ):
                return finding.evidence
        return ()

    def _prerequisites(
        self,
        roots: tuple[Ref, ...],
        goal: Goal,
        state: StateView,
        findings: Mapping[str, VerificationResult],
    ) -> tuple[bool, tuple[Ref, ...]]:
        """Revalidate explicit proof, including nested support and real bytes.

        Only explicit dependencies demand completion/acceptance. Definition/data
        reads still fence versions; an unrelated hypothesis or unverified
        criterion definition cannot become a prerequisite merely by being read.
        """
        indexed = records_by_id(state)
        materialized = {r.id: r for r in state.records}
        pending = [(r, True) for r in roots]
        seen: set[tuple[str, bool]] = set()
        reads: list[Ref] = []
        valid = True

        def finding_support(finding: VerificationResult) -> None:
            for support in finding.read_set:
                current = indexed.get(support.id)
                if isinstance(current, (Goal, TaskFrame, Episode)) and _lineage_current(
                    self.artifacts, support, state, lifecycle_ids=frozenset((support.id,))
                ):
                    reads.append(support)
                    support = reference(current)
                pending.append((support, False))
            pending.extend(
                (r, isinstance(indexed.get(r.id), (Belief, VerificationResult)))
                for r in finding.evidence
            )

        while pending:
            ref, explicit = pending.pop()
            item = indexed.get(ref.id)
            if item is None:
                valid = False
                continue
            reads.append(ref)
            if item.revision != ref.revision or (
                ref.kind != "record" and ref.kind != getattr(item, "kind", None)
            ):
                valid = False
                reads.append(reference(item))
            if (item.id, explicit) in seen:
                continue
            seen.add((item.id, explicit))
            if isinstance(item, ArtifactRef):
                try:
                    self.artifacts.read(item, principal=self.principal, goal_id=goal.id)
                except (ExecutiveError, OSError, ValueError):
                    valid = False
            elif isinstance(item, Belief):
                if explicit and item.epistemic_status != "supported":
                    valid = False
                if explicit or item.epistemic_status == "supported":
                    now = self.clock()
                    if (item.valid_from is not None and now < item.valid_from) or (
                        item.valid_until is not None and now >= item.valid_until
                    ):
                        valid = False
                    pending.extend(
                        (r, isinstance(indexed.get(r.id), (Belief, VerificationResult)))
                        for r in item.evidence
                    )
            elif isinstance(item, Criterion):
                if explicit:
                    finding = findings.get(item.id) or next(
                        (
                            r
                            for r in state.records
                            if isinstance(r, VerificationResult) and r.criterion_id == item.id
                        ),
                        None,
                    )
                    if finding is None or finding.disposition != "accept":
                        valid = False
                    elif item.id in findings:
                        finding_support(finding)
                    else:
                        pending.append((reference(finding), True))
                pending.extend(
                    (r, explicit and isinstance(indexed.get(r.id), (Belief, VerificationResult)))
                    for r in (*item.inputs, *item.required_evidence)
                )
            elif isinstance(item, VerificationResult):
                if explicit and item.disposition != "accept":
                    valid = False
                if item.disposition == "accept":
                    replacement = findings.get(item.criterion_id, item)
                    if replacement.disposition != "accept":
                        valid = False
                    else:
                        finding_support(replacement)
            elif isinstance(item, (TaskFrame, Commitment, Goal, ActionRecord)):
                if explicit and not _dependency_satisfied(item, materialized):
                    valid = False
                if isinstance(item, (TaskFrame, Commitment)):
                    if explicit:
                        pending.extend((r, True) for r in item.dependencies)
                    if isinstance(item, Commitment) and explicit:
                        pending.extend(
                            (reference(r), True)
                            for r in state.records
                            if isinstance(r, TaskFrame) and r.commitment_id == item.id
                        )
                elif isinstance(item, Goal) and explicit:
                    pending.extend((reference(c), True) for c in item.criteria if c.required)
                    pending.extend(
                        (reference(r), True)
                        for r in state.records
                        if isinstance(r, Commitment)
                        and r.goal_id == item.id
                        and (r.required or r.id in item.required_commitment_ids)
                    )
                elif isinstance(item, ActionRecord) and explicit:
                    pending.extend((r, False) for r in (*item.read_set, *item.evidence))
            elif explicit:
                valid = False
        return valid, tuple(dict.fromkeys(reads))

    def _shared_selection(self, refs: tuple[Ref, ...], state: StateView) -> tuple[Ref, ...]:
        """Refresh selected identity across ACL-only metadata revisions.

        The original claim/owner request remains immutable. Current bytes/access
        are checked by the verifier; replacement content and revocation stay stale.
        """
        indexed = records_by_id(state)
        selected = []
        for ref in refs:
            current = indexed.get(ref.id)
            old = _historical(self.artifacts, ref)
            if (
                isinstance(current, ArtifactRef)
                and isinstance(old, ArtifactRef)
                and current.revision >= old.revision
                and current.model_dump(exclude={"revision", "shared_with"})
                == old.model_dump(exclude={"revision", "shared_with"})
            ):
                selected.append(reference(current))
            else:
                selected.append(ref)
        return tuple(selected)

    async def _evaluate(
        self,
        goal: Goal,
        state: StateView,
        subject: str,
        task: TaskFrame | None,
        extras: Mapping[str, tuple[Ref, ...]],
        judgments: Mapping[str, VerificationResult] | None = None,
    ) -> CommitBatch:
        if goal.status not in {"active", "completed", "blocked", "paused"}:
            return self._finish(state, subject, "stale", "inactive_goal")
        configuration = self._configuration()
        # Intake requests are durable evidence selection, not success. Preserve
        # their selection across later rechecks, while every byte/version/access
        # check still happens below against current authoritative state.
        attached: dict[str, tuple[Ref, ...]] = {
            c.id: extras.get(c.id) or self._selected_evidence(c, goal, state) for c in goal.criteria
        }
        for signal in state.records:
            if (
                not isinstance(signal, ExecutiveSignal)
                or signal.signal_kind != "attach_evidence"
                or signal.principal_id != goal.owner_id
                or signal.scope.goal_id != goal.id
            ):
                continue
            try:
                command = AttachEvidence.model_validate(signal.payload["command"])
                if command.goal_id != goal.id or command.evidence != signal.artifacts:
                    continue
                for criterion in goal.criteria:
                    if command.criterion_id in (None, criterion.id):
                        attached[criterion.id] = tuple(
                            dict.fromkeys((*attached.get(criterion.id, ()), *command.evidence))
                        )
            except (ValueError, KeyError, TypeError):
                continue
        extras = {key: self._shared_selection(refs, state) for key, refs in attached.items()}
        results: list[VerificationResult] = []
        for criterion in goal.criteria:
            base = ArtifactVerifier(
                self.artifacts,
                principal=self.principal,
                evidence=extras.get(criterion.id, ()),
                clock=self.clock,
            )
            verifier: Verifier
            if criterion.evaluator_type == "human" and criterion.verifier_id == "owner-approval-v1":
                verifier = HumanApprovalVerifier(
                    self.artifacts, principal=self.principal, clock=self.clock
                )
            elif (
                criterion.evaluator_type == "deterministic"
                and criterion.verifier_id == "artifact-integrity-v1"
            ):
                verifier = base
            elif (
                criterion.evaluator_type == "deterministic"
                and criterion.verifier_id in self.local_checks
            ):
                workspace, registry = self.local_checks[criterion.verifier_id]
                verifier = LocalCheckVerifier(
                    self.artifacts,
                    principal=self.principal,
                    workspace=workspace,
                    registry=registry,
                    clock=self.clock,
                )
            elif criterion.evaluator_type == "model" and criterion.verifier_id in self.model_checks:
                supplied = (judgments or {}).get(criterion.id)
                old = supplied or next(
                    (
                        r
                        for r in state.records
                        if isinstance(r, VerificationResult)
                        and r.criterion_id == criterion.id
                        and r.policy_version.startswith("model-judgment-v1:")
                    ),
                    None,
                )
                try:
                    if old is None:
                        raise ValueError("missing_model_judgment")
                    if supplied is not None:
                        result = supplied
                    else:
                        model_action = next(
                            (
                                a
                                for r in old.read_set
                                if isinstance((a := records_by_id(state).get(r.id)), ActionRecord)
                                and a.capability_id == self.model_checks[criterion.verifier_id][1]
                            ),
                            None,
                        )
                        if model_action is None:
                            raise ValueError("missing_model_action")
                        result = self._model_finding(criterion, old, model_action, state)
                except (ExecutiveError, ValueError, OSError, KeyError, TypeError):
                    result = base._result(
                        criterion,
                        state,
                        "needs_human",
                        "model_judgment_unproven",
                        policy="model-judgment-v1:" + self.model_checks[criterion.verifier_id][2],
                    )
                results.append(result)
                continue
            elif (
                criterion.evaluator_type == "deterministic"
                and criterion.verifier_id in self.verifiers
            ):
                # Trusted free, local computation only. Metered judgments use Dispatcher.
                verifier = self.verifiers[criterion.verifier_id]
            else:
                result = base._result(
                    criterion,
                    state,
                    "needs_human"
                    if criterion.evaluator_type in {"human", "model"}
                    else "insufficient_evidence",
                    "model_judgment_unconfigured"
                    if criterion.evaluator_type == "model"
                    else "unconfigured_verifier",
                    policy="model-judgment-unconfigured-v1"
                    if criterion.evaluator_type == "model"
                    else "verification-v1",
                )
                results.append(result)
                continue
            result = await verifier.verify(criterion, state)
            if (
                criterion.evaluator_type == "deterministic"
                and criterion.verifier_id in self.verifiers
            ):
                try:
                    support = base._support(criterion, state)
                    required_local = {reference(goal), reference(criterion), *support}
                    if (
                        result.criterion_id != criterion.id
                        or result.verifier_id != criterion.verifier_id
                        or not required_local <= set(result.read_set)
                    ):
                        raise ValueError("unbound_local_verifier_result")
                    if result.usage.status != "known" or any(
                        getattr(result.usage, key) != 0
                        for key in (
                            "calls",
                            "input_tokens",
                            "output_tokens",
                            "elapsed_seconds",
                            "cost_microunits",
                        )
                    ):
                        raise ValueError("local_verifier_requires_zero_usage")
                    for ref in (*result.read_set, *result.evidence):
                        local_record = records_by_id(state).get(ref.id)
                        if local_record is None or local_record.revision != ref.revision:
                            raise ValueError("stale_local_verifier_evidence")
                        if isinstance(local_record, ArtifactRef):
                            self.artifacts.read(
                                local_record, principal=self.principal, goal_id=goal.id
                            )
                except (ExecutiveError, ValueError, OSError):
                    result = base._result(
                        criterion, state, "insufficient_evidence", "unbound_local_verifier_result"
                    )
            results.append(result)
        if self.artifacts.store.view() != state:
            return self._finish(
                self.artifacts.store.view(), subject, "stale", "late_verification_result"
            )
        if self._configuration() != configuration:
            return self._finish(state, subject, "stale", "verification_configuration_changed")
        # Clock-aware support must still be valid after any asynchronous verifier.
        indexed = records_by_id(state)
        support_task = task or next(
            (r for r in state.records if isinstance(r, TaskFrame) and r.goal_id == goal.id), None
        )
        if support_task is not None:
            for i, result in enumerate(results):
                if not _fresh(
                    support_task.model_copy(update={"dependencies": result.read_set}),
                    indexed,
                    self.clock(),
                ):
                    results[i] = result.model_copy(
                        update={
                            "disposition": "insufficient_evidence",
                            "reason_code": "expired_verification_support",
                            "gaps": (
                                CriterionGap(
                                    criterion_id=result.criterion_id,
                                    reason_code="expired_verification_support",
                                    description="Verification support expired",
                                ),
                            ),
                        }
                    )
        constraints = self.authority.check_constraints(goal.constraints, indexed)
        required = {c.id for c in goal.criteria if c.required}
        passed = all(
            r.disposition == "accept" for r in results if r.criterion_id in required
        ) and bool(required)
        findings = {r.criterion_id: r for r in results}
        commitments = {
            r.id: r for r in state.records if isinstance(r, Commitment) and r.goal_id == goal.id
        }
        needed = set(goal.required_commitment_ids) | {
            r.id for r in commitments.values() if r.required
        }
        prerequisites: dict[str, tuple[bool, tuple[Ref, ...]]] = {
            c.id: self._prerequisites(c.dependencies, goal, state, findings)
            for c in commitments.values()
        }
        for frame in state.records:
            if isinstance(frame, TaskFrame) and frame.goal_id == goal.id:
                associated = commitments.get(frame.commitment_id)
                roots = (*frame.dependencies, *(associated.dependencies if associated else ()))
                prerequisites[frame.id] = self._prerequisites(roots, goal, state, findings)
        required_support = all(prerequisites[identifier][0] for identifier in needed) and all(
            prerequisites[r.id][0]
            for r in state.records
            if isinstance(r, TaskFrame) and r.goal_id == goal.id and r.commitment_id in needed
        )
        proof = tuple(dict.fromkeys(r for _, reads in prerequisites.values() for r in reads))
        clear = (
            constraints.disposition == "authorize"
            and not self._unsettled(goal, state)
            and required_support
        )
        events: list[JournalEvent] = []
        working = state
        if (goal.status == "completed" and not (passed and clear)) or (
            not required_support
            and any(
                isinstance(r, (TaskFrame, Commitment))
                and r.status == "completed"
                and r.id in prerequisites
                and not prerequisites[r.id][0]
                for r in state.records
            )
        ):
            for event in invalidate_dependents(state, (reference(goal),)):
                event = event.model_copy(update={"sequence": working.sequence + 1})
                working = reduce_event(working, event)
                events.append(event)
        current = records_by_id(working)
        current_goal = current[goal.id]
        assert isinstance(current_goal, Goal)
        writes: dict[str, PersistentRecord] = {}
        for result in results:
            old = next(
                (
                    r
                    for r in working.records
                    if isinstance(r, VerificationResult) and r.criterion_id == result.criterion_id
                ),
                None,
            )
            if old is not None:
                result = result.model_copy(
                    update={
                        "id": old.id,
                        "revision": old.revision + 1,
                        "created_at": old.created_at,
                    }
                )
            writes[result.id] = result
        verifying = [
            r
            for r in working.records
            if isinstance(r, TaskFrame)
            and r.goal_id == goal.id
            and r.status == "verifying"
            and (task is None or r.id == task.id)
        ]
        for frame in verifying:
            associated = commitments.get(frame.commitment_id)
            dependencies = prerequisites[frame.id][0] and all(
                (item := current.get(ref.id)) is not None
                and item.revision == ref.revision
                and _dependency_satisfied(
                    item, {r.id: r for r in (*working.records, *writes.values())}
                )
                for ref in (*frame.dependencies, *(associated.dependencies if associated else ()))
            )
            effects = all(
                getattr(current.get(identifier), "status", None) == "succeeded"
                for identifier in frame.pending_action_ids
            )
            if passed and clear and dependencies and effects and current_goal.status == "active":
                writes[frame.id] = frame.model_copy(
                    update={
                        "revision": frame.revision + 1,
                        "status": "completed",
                        "failure_count": 0,
                        "failure_signature": None,
                        "failure_reason": None,
                        "wake_condition": None,
                    }
                )
            elif not passed and current_goal.status == "active":
                signature = _digest(
                    {
                        "goal": goal_definition_digest(goal),
                        "findings": [
                            {
                                "criterion": c.model_dump(mode="json"),
                                "policy": r.policy_version,
                                "evidence": [e.model_dump(mode="json") for e in r.evidence],
                                "workspace": self._repair_inputs(c),
                            }
                            for c, r in zip(goal.criteria, results)
                        ],
                    }
                )
                count = frame.failure_count + 1 if frame.failure_signature == signature else 1
                repair_ready = count <= self.repair_rounds and dependencies and effects
                writes[frame.id] = frame.model_copy(
                    update={
                        "revision": frame.revision + 1,
                        "status": "ready" if repair_ready else "blocked",
                        "failure_count": count,
                        "failure_signature": signature,
                        "failure_reason": "verification_failed",
                        "wake_condition": None
                        if repair_ready
                        else WakeCondition(
                            kind="new_evidence"
                            if count > self.repair_rounds
                            else "dependency_change",
                            refs=frame.dependencies,
                            reason_code="verification_repair_limit"
                            if count > self.repair_rounds
                            else "verification_dependencies_unsettled",
                        ),
                    }
                )
            elif current_goal.status == "active":
                writes[frame.id] = frame.model_copy(
                    update={
                        "revision": frame.revision + 1,
                        "status": "blocked",
                        "wake_condition": WakeCondition(
                            kind="dependency_change" if not dependencies else "human_decision",
                            refs=prerequisites[frame.id][1],
                            reason_code="completion_prerequisites_unsettled"
                            if not dependencies
                            else "completion_dependencies_unsettled",
                        ),
                    }
                )
        # Reopened completed work and pending frames must not remain runnable
        # merely because a now-expired belief still says supported in its DTO.
        for candidate in working.records:
            if (
                not isinstance(candidate, (TaskFrame, Commitment))
                or candidate.id not in prerequisites
                or prerequisites[candidate.id][0]
                or candidate.id in writes
            ):
                continue
            if (
                candidate.status not in {"ready", "running", "verifying", "suspended", "blocked"}
                or current_goal.status != "active"
            ):
                continue
            updates: dict[str, Any] = {"revision": candidate.revision + 1, "status": "blocked"}
            if isinstance(candidate, TaskFrame):
                updates["wake_condition"] = WakeCondition(
                    kind="dependency_change",
                    refs=prerequisites[candidate.id][1],
                    reason_code="completion_prerequisites_unsettled",
                )
                if candidate.status == "running":
                    updates["fencing_token"] = candidate.fencing_token + 1
            writes[candidate.id] = candidate.model_copy(update=updates)
        # Material evidence can release a verification-only block; scheduler owns
        # readiness and corrected-frame resumption after this suspended handoff.
        if passed:
            for candidate_frame in working.records:
                if (
                    isinstance(candidate_frame, TaskFrame)
                    and candidate_frame.goal_id == goal.id
                    and (
                        candidate_frame.failure_reason == "verification_failed"
                        or (
                            current_goal.status == "active"
                            and candidate_frame.status == "blocked"
                            and candidate_frame.wake_condition is not None
                            and candidate_frame.wake_condition.reason_code
                            == "completion_prerequisites_unsettled"
                            and prerequisites[candidate_frame.id][0]
                        )
                    )
                    and candidate_frame.id not in writes
                ):
                    writes[candidate_frame.id] = candidate_frame.model_copy(
                        update={
                            "revision": candidate_frame.revision + 1,
                            "status": "suspended"
                            if candidate_frame.status == "blocked"
                            else candidate_frame.status,
                            "failure_count": 0,
                            "failure_signature": None,
                            "failure_reason": None,
                            "wake_condition": None,
                        }
                    )
        combined = {r.id: r for r in working.records}
        combined.update(writes)
        for commitment in working.records:
            if (
                not isinstance(commitment, Commitment)
                or commitment.goal_id != goal.id
                or commitment.status in {"completed", "cancelled", "failed"}
            ):
                continue
            frames = [
                r
                for r in combined.values()
                if isinstance(r, TaskFrame) and r.commitment_id == commitment.id
            ]
            if (
                frames
                and prerequisites[commitment.id][0]
                and all(r.status == "completed" for r in frames)
                and all(
                    (item := current.get(ref.id)) is not None
                    and item.revision == ref.revision
                    and _dependency_satisfied(item, combined)
                    for ref in commitment.dependencies
                )
            ):
                # Preserve lifecycle rules using explicit intermediate transitions.
                value = commitment
                path = {
                    "pending": ("ready", "running", "verifying", "completed"),
                    "ready": ("running", "verifying", "completed"),
                    "running": ("verifying", "completed"),
                    "verifying": ("completed",),
                    "blocked": ("ready", "running", "verifying", "completed"),
                    "suspended": ("ready", "running", "verifying", "completed"),
                }[value.status]
                for status in path[:-1]:
                    value = value.model_copy(
                        update={"revision": value.revision + 1, "status": status}
                    )
                    event = self._event((value,), reference(value)).model_copy(
                        update={"sequence": working.sequence + 1}
                    )
                    working = reduce_event(working, event)
                    events.append(event)
                writes[value.id] = value.model_copy(
                    update={"revision": value.revision + 1, "status": "completed"}
                )
                combined[value.id] = writes[value.id]
        if (
            passed
            and clear
            and current_goal.status == "active"
            and all(
                getattr(combined.get(identifier), "status", None) == "completed"
                for identifier in needed
            )
        ):
            updated_goal = current_goal.model_copy(
                update={"revision": current_goal.revision + 1, "status": "completed"}
            )
            writes[goal.id] = updated_goal
        # Normalize only this batch's own lifecycle changes. Immutable action
        # inputs were already checked above; they never receive relaxed lookup.
        resulting = {r.id: r for r in working.records}
        resulting.update(writes)
        for identifier, written in tuple(writes.items()):
            if isinstance(written, VerificationResult):
                writes[identifier] = written.model_copy(
                    update={
                        "read_set": tuple(
                            reference(resulting[ref.id])
                            if isinstance(resulting.get(ref.id), (Goal, TaskFrame, Commitment))
                            else ref
                            for ref in written.read_set
                        )
                    }
                )
        if writes:
            event = self._event(tuple(writes.values()), reference(goal)).model_copy(
                update={"sequence": working.sequence + 1}
            )
            reduce_event(working, event)
            events.append(event)
        reason = (
            "completion_verified"
            if goal.id in writes and getattr(writes[goal.id], "status", None) == "completed"
            else "verification_passed"
            if passed and clear
            else "constraints_unresolved"
            if constraints.disposition != "authorize"
            else "completion_dependencies_unsettled"
            if passed
            else "verification_gaps"
        )
        return self._finish(
            state,
            subject,
            "accept" if passed and clear else "insufficient_evidence",
            reason,
            tuple(events),
            proof=tuple(
                dict.fromkeys(
                    (
                        *proof,
                        *(
                            reference(r)
                            for r in writes.values()
                            if isinstance(r, VerificationResult)
                        ),
                    )
                )
            ),
        )

    def _repair_inputs(self, criterion: Criterion) -> dict[str, str] | None:
        configured = self.local_checks.get(criterion.verifier_id)
        if configured is None:
            return None
        try:
            return configured[0].input_digests(criterion)
        except (ExecutiveError, ValueError, OSError):
            return None
