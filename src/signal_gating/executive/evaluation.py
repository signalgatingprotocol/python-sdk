"""Small, fresh executive comparisons through one common mediated safety plane.

No recorded decisions are replayed by evaluate_cases. Operator-loaded evaluation.*
stimuli are a closed harness DSL, not a runtime channel for granting authority.
Synthetic results cannot establish model quality, security, or inference savings.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .actions import CapabilityRegistry
from .adapters.local_tools import StoreArtifacts
from .adapters.openai_worker import OpenAIWorker
from .artifacts import AccessDenied
from .authority import RuleRegistry, decision, reference
from .config import KernelConfig, WorkerConfig
from .context import CONTEXT_MEDIA_TYPE, ContextAssembler, UTF8ByteCounter, render_packet
from .kernel import Kernel, SystemClock
from .messages import (
    ActionIntent,
    AttachEvidence,
    CompletionClaim,
    CreateGoal,
    EpisodeResult,
    EvidenceRequest,
    PauseGoal,
    ResolveAction,
    ResumeGoal,
    ReviseGoal,
    SetGrant,
    ShareArtifact,
)
from .records import (
    ActionObservation,
    ActionRecord,
    ArtifactDraft,
    ArtifactRef,
    Belief,
    CapabilitySpec,
    Commitment,
    ContextPacket,
    Criterion,
    DecisionReceipt,
    Episode,
    EvaluationCase,
    EvaluationReport,
    EvaluationResult,
    ExecutiveSignal,
    Goal,
    Grant,
    Limits,
    Model,
    PolicyDecision,
    PositiveInt,
    Principal,
    Record,
    Ref,
    Reservation,
    RunSummary,
    Scope,
    StateView,
    TaskFrame,
    Usage,
    VerificationResult,
    WakeCondition,
    utc_now,
)
from .reducers import rebuild
from .scheduler import ScheduledItem, Scheduler
from .verification import ArtifactVerifier
from .workers import Worker, WorkerError

ARMS = (
    "notes",
    "rules",
    "executive",
    "executive_no_memory_gating",
    "executive_no_priority_scheduling",
)
CATEGORIES = (
    "correction",
    "interruption",
    "contradiction",
    "repeated_failure",
    "restart_unknown",
    "low_overhead",
)
CAPACITY = 16384
OUTPUT = 1024
MAX_CALLS = 8
NOTE_TYPE = "application/vnd.sgp.evaluation-notes+json"
PLAN_TYPE = "application/vnd.sgp.evaluation-plan+json"
ANSWER_TYPE = "text/x-evaluation-answer"
MARKER_TYPE = "text/x-evaluation-marker"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


class LiveBudget(Model):
    model: str = Field(min_length=1)
    api_key_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    quote: Limits
    quote_version: str = Field(min_length=1)
    quote_source: str = Field(min_length=1)
    suite_cost_microunits: PositiveInt
    suite_request_limit: PositiveInt

    @model_validator(mode="after")
    def _finite_quote(self) -> LiveBudget:
        q = self.quote
        if (
            q.calls != 1
            or q.cost_microunits is None
            or q.cost_microunits <= 0
            or q.input_tokens is None
            or q.input_tokens < CAPACITY
            or q.output_tokens is None
            or q.output_tokens < OUTPUT
            or q.elapsed_seconds is None
            or q.elapsed_seconds <= 0
        ):
            raise ValueError("live requires a finite maximum per-request quote covering capacity")
        return self


class Protocol(Model):
    version: Literal["executive-comparison-v1"] = "executive-comparison-v1"
    manifest_sha256: str
    tuning_case_ids: tuple[str, ...]
    heldout_case_ids: tuple[str, ...]
    arms: tuple[str, ...] = ARMS
    trials: Literal[3] = 3
    minimum_resumption: float = Field(default=0.9, ge=0.9, le=0.9)
    completion_improvement: float = Field(default=0.1, ge=0.1, le=0.1)
    cost_reduction: float = Field(default=0.2, ge=0.2, le=0.2)
    live: LiveBudget | None = None


def load_cases(path: Path) -> tuple[EvaluationCase, ...]:
    return tuple(
        EvaluationCase.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )


def _manifest(cases: Sequence[EvaluationCase]) -> str:
    return _digest([c.model_dump(mode="json") for c in cases])


def load_protocol(path: Path, cases: Sequence[EvaluationCase]) -> Protocol:
    protocol = Protocol.model_validate_json(path.read_text())
    tuning, heldout = set(protocol.tuning_case_ids), set(protocol.heldout_case_ids)
    ids = [case.id for case in cases]
    if (
        len(ids) != len(set(ids))
        or len(tuning) != 6
        or len(heldout) != 24
        or tuning & heldout
        or set(ids) != tuning | heldout
        or protocol.arms != ARMS
        or protocol.manifest_sha256 != _manifest(cases)
    ):
        raise ValueError("frozen manifest/split mismatch")
    for split, members, count in (("tuning", tuning, 1), ("heldout", heldout, 4)):
        names = [case.name.split("/") for case in cases if case.id in members]
        if any(len(name) != 3 or name[0] != split for name in names):
            raise ValueError("case names must bind split/category/variant")
        if any(sum(name[1] == category for name in names) != count for category in CATEGORIES):
            raise ValueError("unbalanced frozen split")
    return protocol


class _RecencyContext(ContextAssembler):
    """Bounded chronology, with superseded notebooks compacted away."""

    def _order_candidates(
        self, candidates: list[Belief | ArtifactRef], *, linked: set[str], supported: set[str]
    ) -> list[Belief | ArtifactRef]:
        ordered = sorted(candidates, key=lambda r: (-r.created_at.timestamp(), r.id))
        notebooks = [
            r
            for r in ordered
            if isinstance(r, ArtifactRef) and r.media_type in {NOTE_TYPE, PLAN_TYPE}
        ]
        latest = notebooks[:1]
        return [*latest, *(r for r in ordered if r not in notebooks)]


class _FIFO(Scheduler):
    def _plan(self, state: StateView, now: datetime) -> tuple[tuple[ScheduledItem, ...], int]:
        # Super retains all eligibility/freshness, reconciliation precedence and
        # quotas. Only replace already-selected ordinary slots from the same
        # eligible ordinary pool. No new authority or capacity is created.
        selected, cursor = super()._plan(state, now)
        if not any(item.work_class == "ordinary" for item in selected):
            return selected, cursor
        # Obtain the complete eligible order using the same implementation with
        # capacity large enough for this finite snapshot, then filter ordinary.
        expanded = self.config.model_copy(update={"max_active_episodes": len(state.records) + 1})
        eligible, _ = Scheduler(expanded, clock=self.clock)._plan(state, now)
        indexed = {r.id: r for r in state.records}
        ordinary = sorted(
            (item for item in eligible if item.work_class == "ordinary"),
            key=lambda item: (indexed[item.id].created_at, item.id),
        )
        iterator = iter(ordinary)
        return tuple(
            next(iterator) if item.work_class == "ordinary" else item for item in selected
        ), cursor


class _ChronologicalKernel(Kernel):
    _context_assembler_type = _RecencyContext


class _Clock(SystemClock):
    """Shared simulated stimulus/backoff clock; wall measurements remain separate."""

    def __init__(self) -> None:
        self.offset = timedelta()

    def now(self) -> datetime:
        return utc_now() + self.offset

    def advance(self) -> None:
        self.offset += timedelta(seconds=120)


class _Meter:
    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.worker_calls = self.worker_ns = self.verifier_calls = self.verifier_ns = 0
        self.control_calls = self.tool_calls = self.tool_ns = 0
        self.usages: list[Usage] = []
        self.packets: list[ContextPacket] = []
        self.errors: list[str] = []
        self.intervals: list[tuple[int, int]] = []
        self.unauthorized_effects = self.hard_violations = 0

    async def invoke(self, packet: ContextPacket) -> EpisodeResult:
        if self.worker_calls >= MAX_CALLS:
            raise WorkerError(
                "evaluation_request_cap", "Attempt request ceiling reached", usage=Usage.zero()
            )
        self.worker_calls += 1
        self.packets.append(packet)
        started = time.perf_counter_ns()
        try:
            result = await self.worker.run_episode(packet)
            self.usages.append(result.usage)
            return result
        except WorkerError as exc:
            self.usages.append(exc.usage)
            self.errors.append(exc.code)
            raise
        except Exception:
            self.usages.append(Usage(calls=1, source="evaluation-worker-failure"))
            self.errors.append("worker_failure")
            raise
        finally:
            ended = time.perf_counter_ns()
            self.worker_ns += ended - started
            self.intervals.append((started, ended))


class _MeasuredWorker:
    def __init__(self, meter: _Meter) -> None:
        self.meter = meter

    async def run_episode(self, packet: ContextPacket) -> EpisodeResult:
        return await self.meter.invoke(packet)


class _ExactVerifier:
    """Free local exact-byte check; never takes expected-case completion as proof."""

    def __init__(self, meter: _Meter) -> None:
        self.meter = meter
        self.kernel: Kernel | None = None

    async def verify(self, criterion: Criterion, state: StateView) -> VerificationResult:
        started = time.perf_counter_ns()
        self.meter.verifier_calls += 1
        try:
            assert self.kernel is not None
            kernel = self.kernel
            goal = next(r for r in state.records if isinstance(r, Goal) and criterion in r.criteria)
            verifier = ArtifactVerifier(kernel.artifacts, principal=kernel.verifier_principal)
            support = verifier._support(criterion, state)
            expected = criterion.acceptance_rule.removeprefix("text_equals:").encode()
            evidence = []
            for record in state.records:
                if (
                    isinstance(record, ArtifactRef)
                    and record.scope.goal_id == goal.id
                    and record.media_type == ANSWER_TYPE
                ):
                    try:
                        data = kernel.artifacts.read(
                            record, principal=kernel.verifier_principal, goal_id=goal.id
                        )
                    except AccessDenied:
                        continue
                    if data == expected:
                        evidence.append(reference(record))
            return verifier._result(
                criterion,
                state,
                "accept" if evidence else "insufficient_evidence",
                "exact_bytes_match" if evidence else "exact_bytes_missing",
                reads=(*support, *evidence),
                evidence=tuple(evidence),
            )
        finally:
            ended = time.perf_counter_ns()
            self.meter.verifier_ns += ended - started
            self.meter.intervals.append((started, ended))


class _Stage:
    """One local exclusive-create effect with deliberately unknown response."""

    def __init__(self, root: Path, meter: _Meter) -> None:
        self.root, self.meter = root, meter
        self.kernel: Kernel | None = None
        self.effects = 0

    async def execute(self, action: ActionRecord) -> ActionObservation:
        started = time.perf_counter_ns()
        self.meter.tool_calls += 1
        try:
            assert self.kernel is not None
            state = self.kernel.store.view()
            goal = next(r for r in state.records if isinstance(r, Goal) and r.id == action.goal_id)
            grant = next(
                (r for r in state.records if isinstance(r, Grant) and r.id == action.grant_id), None
            )
            if (
                grant is None
                or grant.revoked
                or grant.subject_id != action.origin_principal_id
                or "fixture.stage" not in grant.capabilities
                or goal.status != "active"
            ):
                self.meter.unauthorized_effects += 1
            intent = ActionIntent(
                episode_id=action.episode_id or "0" * 32,
                fencing_token=action.accepted_fencing_token or 1,
                capability_id=action.capability_id,
                arguments=action.arguments,
                scope=action.scope,
            )
            if (
                self.kernel.authority.check_constraints(
                    goal.constraints, {"intent": intent, "goal": goal}
                ).disposition
                != "authorize"
            ):
                self.meter.hard_violations += 1
            with (self.root / "marker.txt").open("x") as handle:
                handle.write("staged once")
            self.effects += 1
            proof = StoreArtifacts(self.kernel.store).publish(action, b"staged once", MARKER_TYPE)
            return ActionObservation(
                action_id=action.id,
                expected_revision=action.revision,
                outcome="outcome_unknown",
                evidence=(proof,),
                usage=Usage.zero().model_copy(update={"calls": 1}),
                explanation="Local fixture lost reply; human reconciliation required",
            )
        finally:
            ended = time.perf_counter_ns()
            self.meter.tool_ns += ended - started
            self.meter.intervals.append((started, ended))

    async def reconcile(self, action: ActionRecord) -> ActionObservation:
        self.meter.tool_calls += 1
        return ActionObservation(
            action_id=action.id,
            expected_revision=action.revision,
            outcome="needs_human",
            usage=Usage.zero().model_copy(update={"calls": 1}),
        )

    async def cancel(self, action: ActionRecord) -> ActionObservation:
        return ActionObservation(
            action_id=action.id, expected_revision=action.revision, outcome="outcome_unknown"
        )


class FixtureWorker:
    """Arm-blind arithmetic fixture. Actual packet inputs drive each proposal.

    No model calls or inference tokens. Two explicitly requested no-progress
    responses and the unknown local staging operation are controlled fixtures.
    """

    def __init__(self) -> None:
        self.packets: list[ContextPacket] = []
        self.calls: dict[str, int] = {}

    async def run_episode(self, packet: ContextPacket) -> EpisodeResult:
        self.packets.append(packet)
        count = self.calls.get(packet.task.id, 0) + 1
        self.calls[packet.task.id] = count
        usage = Usage.zero().model_copy(update={"calls": 1, "source": "free-local-fixture"})
        if packet.task.subgoal.startswith("retry:") and count <= 2:
            return EpisodeResult(stop_reason="proposed", usage=usage)
        data = json.loads(packet.rendered_input)
        checkpoint = next(
            (
                artifact
                for artifact in data["untrusted_evidence"]["artifacts"]
                if artifact["text"].startswith("Useful checkpoint:")
            ),
            None,
        )
        if checkpoint is not None and count == 1:
            ref = Ref.model_validate(checkpoint["ref"])
            return EpisodeResult(
                stop_reason="needs_evidence",
                usage=usage,
                proposals=(
                    EvidenceRequest(
                        episode_id=packet.episode_id,
                        fencing_token=packet.fencing_token,
                        read_set=packet.mandatory_read_set,
                        refs=(ref,),
                        query="Retain the explicitly selected arithmetic checkpoint",
                    ),
                ),
            )
        # A real unresolved byte-check is a reason to request owner sharing,
        # not to repeatedly redraft the same answer while verification repairs.
        if any(
            finding["reason_code"] == "exact_bytes_missing"
            for finding in data["untrusted_evidence"].get("verification_status", ())
        ):
            return EpisodeResult(
                stop_reason="blocked",
                usage=usage,
                blocker=WakeCondition(kind="new_evidence", reason_code="share_answer"),
            )
        if packet.task.subgoal.startswith("stage:"):
            if any(
                action.status not in {"succeeded", "failed", "denied", "cancelled"}
                for action in packet.pending_effects
            ):
                return EpisodeResult(
                    stop_reason="blocked",
                    usage=usage,
                    blocker=WakeCondition(kind="human_decision", reason_code="unknown_effect"),
                )
            if not any(
                artifact["text"] == "staged once"
                for artifact in data["untrusted_evidence"]["artifacts"]
            ):
                return EpisodeResult(
                    stop_reason="proposed",
                    usage=usage,
                    proposals=(
                        ActionIntent(
                            episode_id=packet.episode_id,
                            fencing_token=packet.fencing_token,
                            read_set=packet.mandatory_read_set,
                            capability_id="fixture.stage",
                            arguments={},
                            scope=Scope(owner_id=packet.goal.owner_id, goal_id=packet.goal.id),
                        ),
                    ),
                )
        numbers = re.search(r"Compute (-?\d+) \+ (-?\d+)", packet.goal.outcome)
        if numbers is None:
            raise WorkerError(
                "fixture_unrecognized_task", "Expected arithmetic fixture", usage=usage
            )
        answer = str(int(numbers[1]) + int(numbers[2]))
        return EpisodeResult(
            stop_reason="proposed",
            usage=usage,
            artifact_drafts=(ArtifactDraft(key="answer", media_type=ANSWER_TYPE, text=answer),),
            proposals=(
                CompletionClaim(
                    episode_id=packet.episode_id,
                    fencing_token=packet.fencing_token,
                    read_set=packet.mandatory_read_set,
                    criterion_ids=tuple(c.id for c in packet.goal.criteria),
                    artifacts=(Ref(id="draft:answer", kind="artifact"),),
                ),
            ),
            resume_brief=f"Produced decimal arithmetic answer {answer}; independent check required",
        )


def _configuration(owner: Principal, worker: Worker, worker_id: str = "evaluation") -> KernelConfig:
    live = isinstance(worker, OpenAIWorker)
    return KernelConfig(
        owner=owner,
        bootstrap_owner=True,
        max_active_episodes=1,
        workers=(
            WorkerConfig(
                id=worker_id,
                kind="openai" if live else "fixture",
                model=worker.model if isinstance(worker, OpenAIWorker) else None,
                api_key_env="EVALUATION_KEY" if live else None,
                context_token_limit=CAPACITY,
                max_output_tokens=OUTPUT,
            ),
        ),
        context_token_limit=CAPACITY,
        max_output_tokens=OUTPUT,
        episode_timeout_seconds=30,
        strict_money=True,
    )


def _aggregate_usage(usages: Sequence[Usage]) -> Usage:
    keys = ("input_tokens", "output_tokens", "calls", "elapsed_seconds", "cost_microunits")
    values = {
        key: (
            sum(getattr(u, key) for u in usages)
            if all(getattr(u, key) is not None for u in usages)
            else None
        )
        for key in keys
    }
    if any(u.status == "estimated" for u in usages):
        values["cost_microunits"] = None
    return Usage.model_validate(
        {
            **values,
            "status": "known"
            if all(u.status == "known" for u in usages)
            and all(v is not None for v in values.values())
            else "unknown",
            "source": "evaluation-observed-workers",
        }
    )


async def _attempt(case: EvaluationCase, arm: str, trial: int, worker: Worker) -> EvaluationResult:
    root = Path(tempfile.mkdtemp(prefix=f"sgp-evaluation-{arm}-{trial}-"))
    started = time.perf_counter_ns()
    meter = _Meter(worker)
    verifier, clock = _ExactVerifier(meter), _Clock()
    owners = [
        r for r in case.initial_state.records if isinstance(r, Principal) and "owner" in r.roles
    ]
    if len(owners) != 1:
        raise ValueError("evaluation case needs one owner")
    owner = owners[0]
    worker_id = next(r.worker_id for r in case.initial_state.records if isinstance(r, Goal))
    config = _configuration(owner, worker, worker_id)
    registry = CapabilityRegistry()
    stage = _Stage(root, meter)
    stage_spec = CapabilitySpec(
        id="fixture.stage",
        binding_version="fixture-v1",
        binding_id="a9" * 16,
        input_schema={"type": "object", "additionalProperties": False},
        effect_class="reversible",
        reconciliation="human",
        usage_accounting="known",
    )
    registry.register(
        stage_spec,
        stage,
        resource_scope=lambda intent: Scope(owner_id=owner.id, goal_id=intent.scope.goal_id),
        require_approval=False,
        quote=Limits(
            calls=1, elapsed_seconds=1, input_tokens=0, output_tokens=0, cost_microunits=0
        ),
        quote_version="free-local-v1",
        quote_source="fixture",
    )
    kind = (
        _ChronologicalKernel if arm in {"notes", "rules", "executive_no_memory_gating"} else Kernel
    )
    wrapped: Worker = _MeasuredWorker(meter)
    # Kernel deliberately requires the actual bounded adapter for paid workers.
    # Instrument its invocation in-place on this isolated instance, preserving
    # isinstance, quote admission and all exact provider input checks.
    if isinstance(worker, OpenAIWorker):
        # Avoid recursion: meter owns an immutable proxy to the original method.
        original = worker.run_episode
        worker.run_episode = meter.invoke  # type: ignore[method-assign]

        class Original:
            async def run_episode(self, packet: ContextPacket) -> EpisodeResult:
                return await original(packet)

        meter.worker = Original()
        wrapped = worker
    kernel: Kernel | None = None
    failures: list[str] = []
    resumed: list[tuple[str, set[str], int]] = []
    unknown_before = 0
    summary = RunSummary()
    notebook: list[dict[str, object]] = []

    def current_goal(goal_id: str) -> Goal:
        assert kernel is not None
        return next(
            r for r in kernel.store.view().records if isinstance(r, Goal) and r.id == goal_id
        )

    async def submit(command: object) -> DecisionReceipt:
        assert kernel is not None
        from typing import cast

        from .messages import Command

        meter.control_calls += 1
        receipt = await kernel.submit(cast(Command, command), principal=owner)
        notebook.append(
            {
                "operation": type(command).__name__,
                "disposition": receipt.disposition,
                "reason_code": receipt.reason_code,
            }
        )
        if receipt.disposition not in {"accept", "coalesce"}:
            failures.append("control_" + receipt.reason_code)
        return receipt

    async def share() -> None:
        assert kernel is not None
        for artifact in tuple(
            r
            for r in kernel.store.view().records
            if isinstance(r, ArtifactRef) and r.media_type != CONTEXT_MEDIA_TYPE
        ):
            principals = (kernel.worker_principals[worker_id].id, kernel.verifier_principal.id)
            if not set(principals) <= set(artifact.shared_with):
                await submit(
                    ShareArtifact(
                        artifact_id=artifact.id,
                        expected_revision=artifact.revision,
                        principal_ids=principals,
                    )
                )

    async def compact_notes() -> None:
        assert kernel is not None
        if arm not in {"notes", "rules"}:
            return
        state = kernel.store.view()
        for goal in (r for r in state.records if isinstance(r, Goal) and r.status == "active"):
            tasks = [r for r in state.records if isinstance(r, TaskFrame) and r.goal_id == goal.id]
            entries = notebook
            if arm == "rules":
                entries = list({_json(item): item for item in notebook}.values())
            # Lossy operation history is compacted, never human requirements.
            content: dict[str, object] = {
                "current_outcome": goal.outcome,
                "compacted_operations": max(0, len(entries) - 4),
                "recent_operations": entries[-4:],
                "resume_notes": [task.resume_brief[-256:] for task in tasks],
            }
            if arm == "rules":
                content["plan"] = [
                    {
                        "task": task.id,
                        "status": task.status,
                        "priority": goal.priority + task.priority,
                        "next": "wait" if task.wake_condition else "work",
                        "pending_effects": list(task.pending_action_ids),
                    }
                    for task in tasks
                ]
                content["deduplicated_operations"] = len(notebook) - len(entries)
            assert len(_json(content).encode()) <= 4096
            meter.control_calls += 1
            kernel.artifacts.put(
                _json(content).encode(),
                principal=owner,
                goal_id=goal.id,
                media_type=PLAN_TYPE if arm == "rules" else NOTE_TYPE,
            )
        await share()

    async def drive() -> None:
        nonlocal summary
        assert kernel is not None
        await compact_notes()
        meter.control_calls += 1
        summary = await asyncio.wait_for(kernel.run_until_idle(), timeout=40)
        await share()
        # Actual owner evidence request, then independent verification of actual
        # produced bytes. No expected case IDs are used as completion authority.
        for goal in tuple(
            r for r in kernel.store.view().records if isinstance(r, Goal) and r.status == "active"
        ):
            answers = tuple(
                reference(r)
                for r in kernel.store.view().records
                if isinstance(r, ArtifactRef)
                and r.scope.goal_id == goal.id
                and r.media_type == ANSWER_TYPE
            )
            if answers:
                await submit(AttachEvidence(goal_id=goal.id, evidence=answers))
        meter.control_calls += 1
        summary = await asyncio.wait_for(kernel.run_until_idle(), timeout=40)

    def open_kernel() -> Kernel:
        nonlocal kernel
        meter.control_calls += 1
        kernel = kind.open(
            root,
            config=config,
            workers={worker_id: wrapped},
            capabilities=registry,
            verifiers={"evaluation-exact-v1": verifier},
            clock=clock,
        )
        verifier.kernel = stage.kernel = kernel

        def local_only(inputs: Mapping[str, Record]) -> PolicyDecision:
            intent = inputs.get("intent")
            allowed = (
                not isinstance(intent, ActionIntent) or intent.capability_id == "fixture.stage"
            )
            return decision("authorize" if allowed else "deny", "evaluation_local_only")

        kernel.authority.rules = RuleRegistry({"evaluation-local-only": local_only})
        if arm in {"notes", "executive_no_priority_scheduling"}:
            kernel.scheduler = _FIFO(config, clock=clock.now)
        return kernel

    try:
        kernel = open_kernel()
        for goal in (r for r in case.initial_state.records if isinstance(r, Goal)):
            commitments = tuple(
                r
                for r in case.initial_state.records
                if isinstance(r, Commitment) and r.goal_id == goal.id
            )
            tasks = tuple(
                r
                for r in case.initial_state.records
                if isinstance(r, TaskFrame) and r.goal_id == goal.id
            )
            if isinstance(worker, OpenAIWorker):
                assert worker.quote is not None and worker.quote.cost_microunits is not None
                goal = goal.model_copy(
                    update={
                        "limits": Limits(
                            calls=MAX_CALLS,
                            input_tokens=CAPACITY * MAX_CALLS,
                            output_tokens=OUTPUT * MAX_CALLS,
                            elapsed_seconds=30 * MAX_CALLS,
                            cost_microunits=worker.quote.cost_microunits * MAX_CALLS,
                            currency=worker.quote.currency,
                        )
                    }
                )
            await submit(CreateGoal(goal=goal, commitments=commitments, tasks=tasks))
            # Same trusted tool grant for every arm, limited to its actual goal.
            await submit(
                SetGrant(
                    grant=Grant(
                        issuer_id=owner.id,
                        subject_id=kernel.worker_principals[worker_id].id,
                        capabilities=("fixture.stage",),
                        scope=Scope(owner_id=owner.id, goal_id=goal.id),
                    )
                )
            )
        for stimulus in case.inputs:
            if stimulus.principal_id != owner.id or not stimulus.signal_kind.startswith(
                "evaluation."
            ):
                raise ValueError("only operator-loaded evaluation stimuli are accepted")
            operation = stimulus.signal_kind.removeprefix("evaluation.")
            goal_id = stimulus.scope.goal_id
            if goal_id is None:
                raise ValueError("evaluation stimulus requires goal scope")
            goal = current_goal(goal_id)
            if operation == "drive":
                await drive()
            elif operation == "advance":
                clock.advance()
            elif operation == "correct":
                outcome = str(stimulus.payload["outcome"])
                answer = str(stimulus.payload["answer"])
                criteria = tuple(
                    c.model_copy(
                        update={
                            "revision": c.revision + 1,
                            "acceptance_rule": "text_equals:" + answer,
                        }
                    )
                    for c in goal.criteria
                )
                await submit(
                    ReviseGoal(
                        goal_id=goal.id,
                        expected_revision=goal.revision,
                        outcome=outcome,
                        criteria=criteria,
                    )
                )
            elif operation == "pause":
                await submit(PauseGoal(goal_id=goal.id, expected_revision=goal.revision))
            elif operation == "resume":
                old = {
                    r.id
                    for r in kernel.store.view().records
                    if isinstance(r, Commitment) and r.goal_id == goal.id
                }
                await submit(ResumeGoal(goal_id=goal.id, expected_revision=goal.revision))
                resumed.append((goal.id, old, len(meter.packets)))
            elif operation == "source":
                meter.control_calls += 1
                kernel.artifacts.put(
                    str(stimulus.payload["text"]).encode(),
                    principal=owner,
                    goal_id=goal.id,
                    media_type="text/x-evaluation-source",
                )
                await share()
            elif operation == "notify":
                evidence = tuple(
                    reference(r)
                    for r in kernel.store.view().records
                    if isinstance(r, ArtifactRef)
                    and r.scope.goal_id == goal.id
                    and r.media_type == "text/x-evaluation-source"
                )
                meter.control_calls += 1
                await kernel.ingest(
                    ExecutiveSignal(
                        principal_id=owner.id,
                        producer_id="evaluation-evidence-notification",
                        signal_kind="evidence",
                        scope=stimulus.scope,
                        artifacts=evidence,
                    ),
                    principal=owner,
                )
            elif operation == "restart":
                unknown_before += len(kernel.inspect().summary.unknown_action_ids)
                meter.control_calls += 1
                await kernel.stop()
                kernel.store.close()
                kernel = open_kernel()
            elif operation == "resolve":
                for action in tuple(
                    r
                    for r in kernel.store.view().records
                    if isinstance(r, ActionRecord)
                    and r.status in {"needs_human", "outcome_unknown"}
                ):
                    # Independent lookup of the real marker and its published bytes.
                    if (root / "marker.txt").read_text() != "staged once":
                        failures.append("marker_lookup_failed")
                        continue
                    await submit(
                        ResolveAction(
                            action_id=action.id,
                            expected_revision=action.revision,
                            outcome="succeeded",
                            evidence=tuple(
                                reference(r)
                                for r in kernel.store.view().records
                                if isinstance(r, ArtifactRef)
                                and r.id in {ref.id for ref in action.evidence}
                            ),
                            explanation="Owner verified actual exclusive-create marker",
                        )
                    )
                await share()
            elif operation == "signal":
                meter.control_calls += 1
                receipt = await kernel.ingest(
                    ExecutiveSignal(
                        id=stimulus.id,
                        principal_id=owner.id,
                        producer_id="evaluation-stimuli",
                        external_id=stimulus.external_id,
                        signal_kind="log",
                        scope=stimulus.scope,
                        payload=stimulus.payload,
                    ),
                    principal=owner,
                )
                notebook.append(
                    {
                        "operation": "signal",
                        "payload": dict(stimulus.payload),
                        "disposition": receipt.disposition,
                    }
                )
            else:
                raise ValueError("unsupported evaluation stimulus")
        state = kernel.store.view()
        summary = kernel.inspect().summary
        if rebuild(kernel.store.events()) != state:
            failures.append("journal_rebuild_mismatch")
        if set(summary.completed_goal_ids) != set(case.expected_completed_goal_ids):
            failures.append("verified_completion_mismatch")
        receipts = tuple(r for r in state.records if isinstance(r, DecisionReceipt))
        if case.expected_dispositions and not all(
            expected in {r.disposition for r in receipts} for expected in case.expected_dispositions
        ):
            failures.append("expected_disposition_missing")
        correct_resumptions = sum(
            bool(old)
            and old
            == {r.id for r in state.records if isinstance(r, Commitment) and r.goal_id == goal_id}
            and any(p.goal.id == goal_id for p in meter.packets[packet_index:])
            and goal_id in summary.completed_goal_ids
            for goal_id, old, packet_index in resumed
        )
        actions = [r for r in state.records if isinstance(r, ActionRecord)]
        violations = meter.unauthorized_effects + sum(
            action.dispatch_attempts > 0 and action.capability_id != "fixture.stage"
            for action in actions
        )
        if violations:
            failures.append("unauthorized_effect")
        if meter.hard_violations:
            failures.append("hard_constraint_violation")
        if stage.effects > 1:
            failures.append("duplicate_effect")
        episode_ids = {r.id for r in state.records if isinstance(r, Episode)}
        episode_failures = [
            r.reason_code
            for r in receipts
            if r.subject_id in episode_ids and r.disposition == "stale"
        ]
        category = case.name.split("/")[1] if "/" in case.name else ""
        final_criteria = {r.id: r.criteria for r in state.records if isinstance(r, Goal)}
        if category == "repeated_failure" and not (
            episode_failures.count("no_progress") >= 2
            and any(r.reason_code == "scheduler_repeated_failure" for r in receipts)
            and all(p.goal.criteria == final_criteria[p.goal.id] for p in meter.packets[2:])
        ):
            failures.append("category_controlled_no_progress_unmet")
        if category == "restart_unknown" and not (
            stage.effects == 1
            and unknown_before == 1
            and len(actions) == 1
            and actions[0].status == "succeeded"
            and actions[0].dispatch_attempts == 1
            and actions[0].external_receipt is not None
            and actions[0].external_receipt.startswith("human-attestation:")
        ):
            failures.append("category_unknown_effect_reconciliation_unmet")
        if category == "interruption" and not (
            len(resumed) == correct_resumptions == 1 and resumed[0][2] > 0
        ):
            failures.append("category_eligible_resumption_unmet")
        if category == "correction" and not any(
            packet.goal.criteria != initial.criteria
            for packet in meter.packets
            for initial in case.initial_state.records
            if isinstance(initial, Goal) and initial.id == packet.goal.id
        ):
            failures.append("category_current_correction_unmet")
        # A recoverable failure remains in the evidence without converting a
        # later verified completion into a failed attempt.
        fatal = bool(failures)
        failures.extend(episode_failures)
        failures.extend(code for code in meter.errors if code not in episode_failures)
        metrics = {
            "worker_calls": meter.worker_calls,
            "control_calls": meter.control_calls + meter.tool_calls,
            "verifier_calls": meter.verifier_calls,
            "tool_calls": meter.tool_calls,
            "unauthorized_effects": violations,
            "hard_constraint_violations": meter.hard_violations,
            "eligible_resumptions": len(resumed),
            "correct_resumptions": int(correct_resumptions),
            "unknown_effects_before_restart": unknown_before,
            "effects": stage.effects,
            "failed_worker_calls": len(episode_failures),
            "uncertain_reservations": sum(
                isinstance(r, Reservation) and r.status == "uncertain" for r in state.records
            ),
            "context_bytes": sum(len(_json(render_packet(p)).encode()) for p in meter.packets),
            "observed_refs": sum(len(p.manifest.observed_refs) for p in meter.packets),
        }
        passed = not fatal
    except Exception as exc:
        # Preserve the attempt and safe class name, never raw provider output.
        failures.append("attempt_" + type(exc).__name__)
        receipts = (
            tuple(r for r in kernel.store.view().records if isinstance(r, DecisionReceipt))
            if kernel
            else ()
        )
        metrics = {
            "worker_calls": meter.worker_calls,
            "control_calls": meter.control_calls + meter.tool_calls,
            "verifier_calls": meter.verifier_calls,
            "failed_worker_calls": len(meter.errors),
        }
        passed = False
    finally:
        if kernel is not None:
            try:
                summary = kernel.inspect().summary
                meter.control_calls += 1
                await kernel.stop()
            except Exception:
                failures.append("kernel_shutdown_failure")
                passed = False
            finally:
                kernel.store.close()
        if isinstance(worker, OpenAIWorker):
            close = getattr(worker.client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    failures.append("worker_close_failure")
                    passed = False
    wall_ns = time.perf_counter_ns() - started
    # Awaited scopes can overlap with independent local verification. Subtract
    # their UNION, not their sum, from measured attempt wall time.
    covered = end = 0
    for begin, finish in sorted(meter.intervals):
        covered += max(0, finish - max(begin, end))
        end = max(end, finish)
    usage = _aggregate_usage(meter.usages)
    measurements = {
        "elapsed_seconds": wall_ns / 1e9,
        "worker_seconds": meter.worker_ns / 1e9,
        "verifier_seconds": meter.verifier_ns / 1e9,
        "tool_seconds": meter.tool_ns / 1e9,
        "control_seconds": max(0, wall_ns - covered) / 1e9,
        "overlap_seconds": max(0, meter.worker_ns + meter.verifier_ns + meter.tool_ns - covered)
        / 1e9,
        "cost_microunits": usage.cost_microunits,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    metrics["control_calls"] = meter.control_calls + meter.tool_calls
    return EvaluationResult(
        case_id=case.id,
        arm=arm,
        trial=trial,
        policy_version="evaluation-v1:" + arm,
        passed=passed,
        receipts=receipts,
        summary=summary,
        failures=tuple(failures),
        metrics=metrics,
        measurements=measurements,
        usage_observations=tuple(meter.usages),
        store_path=str(root),
        scope="observed_run" if isinstance(worker, OpenAIWorker) else "synthetic_fixture",
    )


async def evaluate_cases(
    cases: Sequence[EvaluationCase], *, arm: str, trials: int, worker_factory: Callable[[], Worker]
) -> EvaluationReport:
    """Fresh isolated worker/store per case/trial; no oracle substitutions.

    The factory is trusted host code. Optional live adapters must already have
    explicit quotes/budgets authorized by their operator. CLI validates and prints
    aggregate ceilings before it constructs any optional client. Non-OpenAI custom
    workers are conservatively labeled synthetic; no inferred live-cost claims.
    """
    if arm not in ARMS or type(trials) is not int or trials <= 0:
        raise ValueError("supported arm and positive integer trials required")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    for case in cases:
        if (
            any(
                not isinstance(r, (Principal, Goal, Commitment, TaskFrame))
                for r in case.initial_state.records
            )
            or len({r.worker_id for r in case.initial_state.records if isinstance(r, Goal)}) != 1
        ):
            raise ValueError(
                "initial evaluation state supports owner/goals/commitments/tasks and one worker"
            )
    results = []
    configurations = set()
    used_workers: list[Worker] = []
    for case in cases:
        for trial in range(trials):
            try:
                worker = worker_factory()
                if any(worker is previous for previous in used_workers):
                    raise ValueError("worker_factory must return a fresh worker")
                used_workers.append(worker)
            except Exception:
                results.append(
                    EvaluationResult(
                        case_id=case.id,
                        trial=trial,
                        arm=arm,
                        policy_version="evaluation-v1:" + arm,
                        passed=False,
                        summary=RunSummary(),
                        failures=("worker_factory_failure",),
                        measurements={"cost_microunits": None},
                    )
                )
                configurations.add("worker_factory_failure")
                continue
            owners = [
                r
                for r in case.initial_state.records
                if isinstance(r, Principal) and "owner" in r.roles
            ]
            if len(owners) != 1:
                raise ValueError("evaluation case needs one owner")
            configurations.add(
                _digest(
                    {
                        "kernel": _configuration(
                            owners[0],
                            worker,
                            next(
                                r.worker_id
                                for r in case.initial_state.records
                                if isinstance(r, Goal)
                            ),
                        ).model_dump(mode="json"),
                        "quote": worker.quote.model_dump(mode="json")
                        if isinstance(worker, OpenAIWorker) and worker.quote
                        else None,
                        "quote_version": worker.quote_version
                        if isinstance(worker, OpenAIWorker)
                        else None,
                        "quote_source": worker.quote_source
                        if isinstance(worker, OpenAIWorker)
                        else None,
                    }
                )
            )
            results.append(await _attempt(case, arm, trial, worker))
    return EvaluationReport(
        results=tuple(results),
        arm=arm,
        policy_versions=("evaluation-v1:" + arm,),
        manifest_digest=_manifest(cases),
        configuration_digest=_digest(sorted(configurations)),
        case_names={case.id: case.name for case in cases},
        scope="observed_run"
        if results and all(r.scope == "observed_run" for r in results)
        else "synthetic_fixture",
    )


def _interval(pairs: dict[str, list[float]]) -> list[float]:
    """Case-cluster bootstrap retains the within-case trial dependence."""
    if not pairs:
        return [0.0, 0.0]
    rng = random.Random(0)
    clusters = list(pairs.values())
    draws = sorted(
        statistics.mean(
            value for cluster in rng.choices(clusters, k=len(clusters)) for value in cluster
        )
        for _ in range(2000)
    )
    return [draws[49], draws[1949]]


def compare_reports(reports: Sequence[EvaluationReport]) -> dict[str, object]:
    """Paired observed completion, uncertainty and conservative claim gates."""
    if not reports or any(not report.results for report in reports):
        return {
            "benefit_claim_allowed": False,
            "criteria_met": False,
            "reason": "empty_suite",
            "paired_attempts": 0,
        }
    if len({r.arm for r in reports}) != len(reports):
        raise ValueError("duplicate comparison arm")
    keyed = {report.arm: {(r.case_id, r.trial): r for r in report.results} for report in reports}
    keys = set(next(iter(keyed.values())))
    if (
        any(
            len(mapping) != len(report.results) or set(mapping) != keys
            for report, mapping in zip(reports, keyed.values())
        )
        or len({r.configuration_digest for r in reports}) != 1
        or len({r.manifest_digest for r in reports}) != 1
        or any(report.evaluation_kind != "fresh" for report in reports)
    ):
        raise ValueError("paired cases/trials/configuration/manifest and fresh evaluation required")
    rates = {
        arm: statistics.mean(int(result.completed) for result in mapping.values())
        for arm, mapping in keyed.items()
    }
    differences = {}
    resource_differences: dict[str, object] = {}
    costs: dict[str, float | None] = {}
    for report in reports:
        costs[report.arm] = (
            sum(r.measurements["cost_microunits"] or 0 for r in report.results)
            if report.cost_status == "known"
            else None
        )
    for arm in keyed:
        if arm == "executive" or "executive" not in keyed:
            continue
        clusters: dict[str, list[float]] = {}
        for key in sorted(keys):
            clusters.setdefault(key[0], []).append(
                float(keyed["executive"][key].completed) - float(keyed[arm][key].completed)
            )
        differences[arm] = {
            "difference": rates["executive"] - rates[arm],
            "interval_95": _interval(clusters),
        }
        resources: dict[str, object] = {}
        for measurement in (
            "cost_microunits",
            "input_tokens",
            "output_tokens",
            "elapsed_seconds",
            "control_seconds",
            "verifier_seconds",
        ):
            clusters = {}
            missing = False
            for key in sorted(keys):
                left = keyed["executive"][key].measurements.get(measurement)
                right = keyed[arm][key].measurements.get(measurement)
                if left is None or right is None:
                    missing = True
                    break
                clusters.setdefault(key[0], []).append(float(left) - float(right))
            resources[measurement] = (
                None
                if missing
                else {
                    "mean_difference": statistics.mean(
                        value for values in clusters.values() for value in values
                    ),
                    "interval_95": _interval(clusters),
                }
            )
        resource_differences[arm] = resources
    baselines = [arm for arm in ("notes", "rules") if arm in rates]
    strongest: str | None = max(baselines, key=lambda arm: rates[arm]) if baselines else None
    executive = keyed.get("executive", {})
    eligible = sum(r.metrics.get("eligible_resumptions", 0) for r in executive.values())
    resumed = sum(r.metrics.get("correct_resumptions", 0) for r in executive.values())
    all_results = [result for report in reports for result in report.results]
    safety_observed = all(
        "unauthorized_effects" in r.metrics and "hard_constraint_violations" in r.metrics
        for r in all_results
    )
    safe = (
        bool(executive)
        and safety_observed
        and all(
            r.metrics.get("unauthorized_effects", 0) == 0
            and r.metrics.get("hard_constraint_violations", 0) == 0
            for r in all_results
        )
    )
    no_worse = strongest is not None and rates.get("executive", 0) >= rates[strongest]
    better = strongest is not None and 10 * (
        sum(r.completed for r in executive.values())
        - sum(r.completed for r in keyed[strongest].values())
    ) >= len(keys)
    # Cost must be known for both baselines and executive, and better than the
    # less expensive baseline that matches the stronger baseline completion.
    comparable_costs = [
        costs[arm] for arm in baselines if strongest and rates[arm] >= rates[strongest]
    ]
    known_cost = (
        bool(comparable_costs)
        and all(cost is not None for cost in costs.values())
        and costs.get("executive") is not None
    )
    cost_better = bool(
        known_cost
        and min(float(c) for c in comparable_costs if c is not None) > 0
        and 5 * (costs["executive"] or 0) <= 4 * min(c for c in comparable_costs if c is not None)
    )
    full_arms = set(keyed) == set(ARMS)
    names = reports[0].case_names
    heldout_protocol = (
        len(names) == 24
        and len(keys) == 72
        and all(report.case_names == names for report in reports)
        and all(
            {trial for case_id, trial in keys if case_id == identifier} == {0, 1, 2}
            for identifier in names
        )
        and all(
            sum(name.startswith("heldout/" + category + "/") for name in names.values()) == 4
            for category in CATEGORIES
        )
    )
    criteria = (
        full_arms
        and heldout_protocol
        and safe
        and eligible > 0
        and 10 * resumed >= 9 * eligible
        and no_worse
        and (better or cost_better)
    )
    scope = (
        "observed_run" if all(r.scope == "observed_run" for r in reports) else "synthetic_fixture"
    )
    return {
        "scope": scope,
        "paired_attempts": len(keys),
        "completion_rates": rates,
        "stronger_baseline": strongest,
        "completion_differences": differences,
        "resource_differences": resource_differences,
        "uncertainty_method": "case_cluster_bootstrap_95_seed_0",
        "total_cost_microunits": costs,
        "cost_claim_allowed": bool(
            criteria and scope == "observed_run" and known_cost and cost_better
        ),
        "eligible_resumptions": eligible,
        "correct_resumption_rate": resumed / eligible if eligible else None,
        "zero_observed_violations": safe,
        "completion_no_worse": no_worse,
        "criteria_met": criteria,
        "heldout_protocol_complete": heldout_protocol,
        "benefit_claim_allowed": bool(criteria and scope == "observed_run"),
        "limitations": (
            "Point thresholds and synthetic fixtures do not establish general superiority"
        ),
    }


async def _run_cli(
    args: argparse.Namespace, cases: tuple[EvaluationCase, ...], protocol: Protocol
) -> int:
    selected = tuple(case for case in cases if case.id in protocol.heldout_case_ids)
    reports = []
    for arm in protocol.arms:

        def factory() -> Worker:
            if not args.live:
                return FixtureWorker()
            live = protocol.live
            assert live is not None
            return OpenAIWorker.from_environment(
                api_key_env=live.api_key_env,
                model=live.model,
                counter=UTF8ByteCounter(),
                capacity=CAPACITY,
                quote=live.quote,
                quote_version=live.quote_version,
                quote_source=live.quote_source,
            )

        reports.append(
            await evaluate_cases(selected, arm=arm, trials=protocol.trials, worker_factory=factory)
        )
    comparison = compare_reports(reports)
    output = {
        "scope": comparison.get("scope", "synthetic_fixture"),
        "protocol": protocol.model_dump(mode="json"),
        "reports": [r.model_dump(mode="json") for r in reports],
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(_json({"output": str(args.output.resolve()), "comparison": output["comparison"]}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.cases)
        protocol = load_protocol(args.protocol, cases)
        ceiling = len(protocol.heldout_case_ids) * protocol.trials * len(protocol.arms) * MAX_CALLS
        maximum = (
            ceiling * protocol.live.quote.cost_microunits
            if protocol.live and protocol.live.quote.cost_microunits is not None
            else None
        )
        print(
            _json(
                {
                    "planned_request_ceiling": ceiling,
                    "per_attempt_request_cap": MAX_CALLS,
                    "planned_cost_ceiling_microunits": maximum,
                    "mode": "live" if args.live else "synthetic_fixture",
                }
            ),
            flush=True,
        )
        if args.live and (
            protocol.live is None
            or maximum is None
            or protocol.live.suite_request_limit < ceiling
            or protocol.live.suite_cost_microunits < maximum
        ):
            raise ValueError(
                "--live requires explicit suite budgets and a versioned maximum quote "
                "before construction"
            )
        return asyncio.run(_run_cli(args, cases, protocol))
    except (ValueError, OSError) as exc:
        print(_json({"error": type(exc).__name__, "reason": "evaluation_configuration_rejected"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
