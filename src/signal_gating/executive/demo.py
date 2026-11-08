"""Free deterministic release fixture, using the real kernel and local checks.

The embedded project makes the installed demo independent of checkout examples.
Static 3.10 parsing is fixture evidence, not an interpreter compatibility matrix.
The optional review-staging adapter models a lost reply to a local effect. It
never deploys, publishes externally, sends messages, or retries an unknown effect.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, TypeVar

from .actions import CapabilityRegistry
from .adapters.local_tools import StoreArtifacts
from .authority import RuleRegistry, decision, reference
from .config import ArgvCheck, KernelConfig, WorkerConfig, WorkspaceConfig
from .kernel import Kernel
from .messages import (
    ActionIntent,
    ApproveCriterion,
    Command,
    CompletionClaim,
    CreateGoal,
    DecideAction,
    EpisodeResult,
    PauseGoal,
    ResumeGoal,
    Reverify,
    ReviseGoal,
    SetGrant,
    ShareArtifact,
)
from .records import (
    ActionObservation,
    ActionRecord,
    ArtifactDraft,
    ArtifactRef,
    Budget,
    CapabilitySpec,
    Commitment,
    Constraint,
    ContextPacket,
    Criterion,
    DecisionReceipt,
    EvaluationResult,
    ExecutiveSignal,
    Goal,
    Grant,
    Limits,
    PersistentRecord,
    PolicyDecision,
    Principal,
    Record,
    Ref,
    Scope,
    TaskFrame,
    Usage,
    VerificationResult,
    WakeCondition,
)
from .reducers import rebuild

RELEASE_SOURCE = 'def release_name():\n    return "fixture-release"\n'
INCOMPATIBLE_SOURCE = 'type ReleaseName = str\ndef release_name():\n    return "fixture-release"\n'
REQUIREMENT = 'requires-python = ">=3.10"\n'
DROPPED_REQUIREMENT = 'requires-python = ">=3.12"\n'
# Isolated current-interpreter execution plus a separate static 3.10 grammar check.
# No import of checkout files, optional SDK, environment references or network.
COMPATIBILITY_CHECK = (
    "import ast,pathlib; "
    "source=pathlib.Path('release.py').read_text(); "
    "assert pathlib.Path('requires-python.txt').read_text() == " + repr(REQUIREMENT) + "; "
    "ast.parse(source, feature_version=(3,10)); "
    "namespace={}; exec(compile(source,'release.py','exec'),namespace); "
    "assert namespace['release_name']() == 'fixture-release'; "
    "print('synthetic_fixture: static Python 3.10 grammar and current-interpreter test PASS')"
)
OWNER = Principal(
    id="1f" * 16,
    created_at=datetime(2026, 11, 7, tzinfo=timezone.utc),
    name="Synthetic fixture owner",
    trust_domain="local-fixture",
    authenticated_origin="deterministic-demo",
    roles=("owner",),
)
GOAL_ID = "2f" * 16
CASE_ID = "3f" * 16
T = TypeVar("T", bound=PersistentRecord)


class _Clock:
    def now(self) -> datetime:
        return OWNER.created_at

    def monotonic(self) -> float:
        return 0.0

    async def wait_until(self, deadline: datetime) -> None:
        await asyncio.Event().wait()


def _records(kernel: Kernel, kind: type[T]) -> list[T]:
    return [r for r in kernel.store.view().records if isinstance(r, kind)]


def _scope(paths: tuple[str, ...] = ()) -> Scope:
    return Scope(owner_id=OWNER.id, goal_id=GOAL_ID, resource_paths=paths)


class _ReviewStaging:
    """Trusted synthetic local adapter with a deliberately lost execution reply."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bridge: StoreArtifacts | None = None
        self.lookup_available = False

    async def execute(self, action: ActionRecord) -> ActionObservation:
        # Exclusive creation makes an accidental duplicate effect observable.
        with (self.root / "review-staged.txt").open("x") as handle:
            handle.write("Local review staging complete. No deployment.\n")
            handle.flush()
            os.fsync(handle.fileno())
        assert self.bridge is not None
        proof = self.bridge.publish(
            action, (self.root / "review-staged.txt").read_bytes(), "text/plain"
        )
        return ActionObservation(
            action_id=action.id,
            expected_revision=action.revision,
            outcome="outcome_unknown",
            evidence=(proof,),
            usage=Usage.zero().model_copy(update={"calls": 1}),
            explanation="Synthetic lost reply after local review staging; lookup required",
        )

    async def reconcile(self, action: ActionRecord) -> ActionObservation:
        assert self.bridge is not None
        if not self.lookup_available:
            return ActionObservation(
                action_id=action.id,
                expected_revision=action.revision,
                outcome="needs_human",
                usage=Usage.zero().model_copy(update={"calls": 1}),
                explanation="Synthetic lookup temporarily unavailable; do not retry effect",
            )
        expected = b"Local review staging complete. No deployment.\n"
        matches: list[Ref] = []
        # Evidence is a plain local artifact, read through the real ArtifactStore.
        for ref in action.evidence:
            artifact = next(r for r in self.bridge.store.view().records if r.id == ref.id)
            if isinstance(artifact, ArtifactRef):
                data = self.bridge.artifacts.read(artifact, principal=OWNER, goal_id=GOAL_ID)
                if data == expected and (self.root / "review-staged.txt").read_bytes() == data:
                    matches.append(reference(artifact))
        return ActionObservation(
            action_id=action.id,
            expected_revision=action.revision,
            outcome="succeeded" if matches else "needs_human",
            evidence=tuple(matches),
            usage=Usage.zero().model_copy(update={"calls": 1}),
            explanation="Independent local lookup; the staging effect was not repeated",
        )

    async def cancel(self, action: ActionRecord) -> ActionObservation:
        return ActionObservation(
            action_id=action.id, expected_revision=action.revision, outcome="outcome_unknown"
        )


def _registry(adapter: _ReviewStaging) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        CapabilitySpec(
            id="fixture_prepare_review",
            binding_id="4f" * 16,
            binding_version="synthetic-fixture-v1",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect_class="reversible",
            reconciliation="lookup",
            usage_accounting="known",
        ),
        adapter,
        resource_scope=lambda intent: _scope((".",)),
        quote=Limits(
            input_tokens=0, output_tokens=0, calls=1, elapsed_seconds=1, cost_microunits=0
        ),
        quote_source="synthetic_fixture",
        quote_version="v1",
    )
    return registry


def _compatibility_policy(inputs: Mapping[str, Record], config: KernelConfig) -> PolicyDecision:
    """Trusted versioned structural restriction, never compatibility proof."""
    workspace = config.workspace
    if workspace is None or workspace.named_checks != (
        ArgvCheck(
            id="static-python310",
            argv=(sys.executable, "-I", "-S", "-c", COMPATIBILITY_CHECK),
            maximum_charge_microunits=0,
        ),
    ):
        return decision("deny", "fixture_compatibility_check_configuration_changed")
    goal = next((r for r in inputs.values() if isinstance(r, Goal)), None)
    if goal is None or not any(
        c.required
        and c.verifier_id == "local-check-v1"
        and c.check_id == "static-python310"
        and c.workspace_inputs == ("release.py", "requires-python.txt")
        and c.acceptance_rule == "exit_zero"
        and c.revision == 2
        for c in goal.criteria
    ):
        return decision("deny", "fixture_compatibility_requirement_changed")
    intent = inputs.get("intent")
    if isinstance(intent, ActionIntent):
        allowed = (
            (intent.capability_id == "fixture_prepare_review" and not intent.arguments)
            or (
                intent.capability_id == "write_text"
                and intent.arguments.get("path") == "release.py"
            )
            or (
                intent.capability_id == "run_check"
                and intent.arguments.get("command_id") == "static-python310"
                and intent.arguments.get("criterion_ref")
                == reference(goal.criteria[0]).model_dump(mode="json")
            )
        )
        if not allowed:
            return decision("deny", "fixture_compatibility_operation_denied")
    return decision("authorize", "fixture_compatibility_policy_intact")


def _has_resume_context(packet: ContextPacket) -> bool:
    """Require actual current inputs before the deterministic worker can repair."""
    if not any(
        c.hard and c.enforcement_id == "fixture-python310-v1" for c in packet.goal.constraints
    ):
        return False
    criterion = next(
        (
            c
            for c in packet.goal.criteria
            if c.required and c.check_id == "static-python310" and c.revision == 2
        ),
        None,
    )
    if criterion is None or not any(
        c.id == packet.task.commitment_id and c.goal_id == packet.goal.id
        for c in packet.commitments
    ):
        return False
    try:
        evidence = json.loads(packet.rendered_input)["untrusted_evidence"]
        reads = set(packet.mandatory_read_set)
        retained = any(
            a["text"] == "Unrelated fixture arithmetic test: 2+2 == 4 PASS\n"
            and Ref.model_validate(a["ref"]) in reads
            for a in evidence.get("artifacts", ())
        )
        unresolved = any(
            Ref.model_validate(status["criterion"]) == reference(criterion)
            and status["semantics"] == "observation_as_of_admission"
            and Ref.model_validate(status["ref"]) in packet.manifest.observed_refs
            and reference(criterion) in reads
            and status["disposition"] == "insufficient_evidence"
            and status["reason_code"] == "registered_check_unproven"
            and any(gap["reason_code"] == "registered_check_unproven" for gap in status["gaps"])
            for status in evidence.get("verification_status", ())
        )
        return retained and unresolved
    except (ValueError, KeyError, TypeError):
        return False


class _ProposalContext(TypedDict):
    episode_id: str
    fencing_token: int
    read_set: tuple[Ref, ...]


class _Worker:
    def __init__(self) -> None:
        self.phase = "delayed"
        self.started: asyncio.Queue[ContextPacket] = asyncio.Queue()
        self.release = asyncio.Event()
        self.packets: list[ContextPacket] = []
        self.delayed_proposal: ActionIntent | None = None

    async def run_episode(self, packet: ContextPacket) -> EpisodeResult:
        self.packets.append(packet)
        common: _ProposalContext = {
            "episode_id": packet.episode_id,
            "fencing_token": packet.fencing_token,
            "read_set": packet.mandatory_read_set,
        }
        usage = Usage.zero().model_copy(update={"calls": 1})
        scope = _scope((".",))
        if self.phase == "delayed":
            await self.started.put(packet)
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                pass  # Return actual late usage; the runner must fence the proposal.
            visible = json.loads(packet.rendered_input)["untrusted_evidence"]["artifacts"]
            selected = next(a["ref"] for a in visible if a["text"] == DROPPED_REQUIREMENT)
            self.delayed_proposal = ActionIntent(
                **common,
                capability_id="write_text",
                arguments={
                    "path": "requires-python.txt",
                    "expected_sha256": hashlib.sha256(REQUIREMENT.encode()).hexdigest(),
                    "content_ref": selected,
                },
                scope=_scope(("requires-python.txt",)),
                justification="Incompatible delayed proposal drops Python 3.10",
            )
            return EpisodeResult(
                proposals=(self.delayed_proposal,), stop_reason="proposed", usage=usage
            )
        if self.phase == "stage":
            self.phase = "wait"
            return EpisodeResult(
                stop_reason="proposed",
                usage=usage,
                proposals=(
                    ActionIntent(
                        **common, capability_id="fixture_prepare_review", arguments={}, scope=scope
                    ),
                ),
            )
        if self.phase == "repair":
            if not _has_resume_context(packet):
                return EpisodeResult(
                    stop_reason="blocked",
                    usage=usage,
                    blocker=WakeCondition(
                        kind="human_decision", reason_code="resume_context_incomplete"
                    ),
                )
            self.phase = "write"
            return EpisodeResult(
                stop_reason="blocked",
                blocker=WakeCondition(kind="new_evidence", reason_code="share_release_draft"),
                artifact_drafts=(
                    ArtifactDraft(key="release", media_type="text/x-python", text=RELEASE_SOURCE),
                ),
                usage=usage,
                resume_brief="Keep Python 3.10; share the new release draft before writing it.",
            )
        if self.phase == "write":
            artifacts = json.loads(packet.rendered_input)["untrusted_evidence"]["artifacts"]
            selected = next(a["ref"] for a in artifacts if a["text"] == RELEASE_SOURCE)
            self.phase = "check"
            proposal = ActionIntent(
                **common,
                capability_id="write_text",
                arguments={
                    "path": "release.py",
                    "expected_sha256": hashlib.sha256(INCOMPATIBLE_SOURCE.encode()).hexdigest(),
                    "content_ref": selected,
                },
                scope=_scope(("release.py",)),
            )
        elif self.phase == "check":
            self.phase = "claim"
            proposal = ActionIntent(
                **common,
                capability_id="run_check",
                arguments={
                    "command_id": "static-python310",
                    "criterion_ref": reference(packet.goal.criteria[0]).model_dump(mode="json"),
                    "expected_sha256": {
                        "release.py": hashlib.sha256(RELEASE_SOURCE.encode()).hexdigest(),
                        "requires-python.txt": hashlib.sha256(REQUIREMENT.encode()).hexdigest(),
                    },
                },
                scope=scope,
            )
        elif self.phase == "claim":
            self.phase = "wait"
            return EpisodeResult(
                stop_reason="proposed",
                proposals=(
                    CompletionClaim(
                        **common, criterion_ids=tuple(c.id for c in packet.goal.criteria)
                    ),
                ),
                usage=usage,
            )
        else:
            return self._blocked(usage)
        return EpisodeResult(stop_reason="proposed", proposals=(proposal,), usage=usage)

    @staticmethod
    def _blocked(usage: Usage) -> EpisodeResult:
        return EpisodeResult(
            stop_reason="blocked",
            blocker=WakeCondition(kind="new_evidence", reason_code="fixture_owner_step"),
            usage=usage,
        )


async def _submit(kernel: Kernel, command: Command) -> DecisionReceipt:
    receipt = await kernel.submit(command, principal=OWNER)
    if receipt.disposition != "accept":
        raise RuntimeError("Demo owner step rejected: " + receipt.reason_code)
    return receipt


async def _approve(kernel: Kernel) -> None:
    actions = [a for a in _records(kernel, ActionRecord) if a.status == "awaiting_review"]
    if len(actions) != 1:
        raise RuntimeError("Demo expected one reviewable action")
    action = actions[0]
    await _submit(
        kernel,
        DecideAction(
            action_id=action.id,
            expected_revision=action.revision,
            decision="approve",
            arguments_digest=action.arguments_digest,
            scope=action.scope,
            constraint_revision=action.constraint_revision,
            explanation="Explicit synthetic owner authorization for this local operation only",
        ),
    )


async def _share(kernel: Kernel, artifacts: list[ArtifactRef], *, worker: bool = False) -> None:
    principals: tuple[str, ...] = (kernel.verifier_principal.id,)
    if worker:
        principals += (kernel.worker_principals["fixture"].id,)
    for artifact in artifacts:
        selected = tuple(p for p in principals if p not in artifact.shared_with)
        if selected:
            await _submit(
                kernel,
                ShareArtifact(
                    artifact_id=artifact.id,
                    expected_revision=artifact.revision,
                    principal_ids=selected,
                ),
            )


async def _notify_evidence(kernel: Kernel, artifact: ArtifactRef, *, purpose: str) -> None:
    receipt = await kernel.ingest(
        ExecutiveSignal(
            signal_kind="evidence",
            principal_id=OWNER.id,
            producer_id="fixture-owner-evidence",
            scope=_scope(),
            artifacts=(reference(artifact),),
            payload={"purpose": purpose},
        ),
        principal=OWNER,
    )
    if receipt.disposition != "admit":
        raise RuntimeError("Demo evidence notification rejected: " + receipt.reason_code)


async def run_demo(root: Path) -> EvaluationResult:
    """Run once in an empty directory. Output is measured synthetic fixture data."""
    root = root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Demo requires an empty directory")
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "release.py").write_text(INCOMPATIBLE_SOURCE)
    (workspace / "requires-python.txt").write_text(REQUIREMENT)
    worker = _Worker()
    config = KernelConfig(
        owner=OWNER,
        bootstrap_owner=True,
        workers=(
            WorkerConfig(
                id="fixture", kind="fixture", context_token_limit=65536, max_output_tokens=256
            ),
        ),
        workspace=WorkspaceConfig(
            root=workspace,
            writable_paths=(".",),
            named_checks=(
                ArgvCheck(
                    id="static-python310",
                    argv=(sys.executable, "-I", "-S", "-c", COMPATIBILITY_CHECK),
                    maximum_charge_microunits=0,
                ),
            ),
        ),
        verifier_ids=("local-check-v1",),
        verification_repair_rounds=0,
    )
    clock = _Clock()
    adapter = _ReviewStaging(workspace)

    def open_kernel() -> Kernel:
        kernel = Kernel.open(
            root / "store",
            config=config,
            workers={"fixture": worker},
            capabilities=_registry(adapter),
            verifiers={},
            clock=clock,
        )
        adapter.bridge = StoreArtifacts(kernel.store)
        kernel.authority.rules = RuleRegistry(
            {"fixture-python310-v1": lambda inputs: _compatibility_policy(inputs, config)}
        )
        return kernel

    async with open_kernel() as kernel:
        criteria = (
            Criterion(
                description="Release artifact is intact",
                verifier_id="artifact-integrity-v1",
                acceptance_rule="all_exist_and_intact",
            ),
            Criterion(
                description="Owner reviews release artifact; deployment excluded",
                evaluator_type="human",
                verifier_id="owner-approval-v1",
                acceptance_rule="explicit_owner_approval",
            ),
        )
        goal = Goal(
            id=GOAL_ID,
            owner_id=OWNER.id,
            outcome="Prepare release",
            worker_id="fixture",
            limits=Limits(
                input_tokens=1000000,
                output_tokens=100000,
                calls=50,
                elapsed_seconds=1000,
                cost_microunits=0,
            ),
            criteria=criteria,
        )
        await _submit(kernel, CreateGoal(goal=goal))
        for revision, outcome in enumerate(
            (
                "Prepare tested release",
                "Prepare reviewed release",
                "Prepare release; deployment requires separate authorization",
            ),
            1,
        ):
            await _submit(
                kernel, ReviseGoal(goal_id=GOAL_ID, expected_revision=revision, outcome=outcome)
            )
        await _submit(
            kernel,
            SetGrant(
                grant=Grant(
                    issuer_id=OWNER.id,
                    subject_id=kernel.worker_principals["fixture"].id,
                    capabilities=("write_text", "run_check", "fixture_prepare_review"),
                    scope=_scope((".",)),
                )
            ),
        )
        incompatible = kernel.artifacts.put(
            DROPPED_REQUIREMENT.encode(),
            principal=OWNER,
            goal_id=GOAL_ID,
            media_type="text/plain",
        )
        await _submit(
            kernel,
            ShareArtifact(
                artifact_id=incompatible.id,
                expected_revision=incompatible.revision,
                principal_ids=(kernel.worker_principals["fixture"].id,),
            ),
        )
        unrelated_check = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-c",
            "assert 2+2 == 4; print('Unrelated fixture arithmetic test: 2+2 == 4 PASS')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            unrelated_output, unrelated_error = await asyncio.wait_for(
                unrelated_check.communicate(), 5
            )
        finally:
            if unrelated_check.returncode is None:
                unrelated_check.kill()
                await unrelated_check.communicate()
        if unrelated_check.returncode != 0 or unrelated_error:
            raise RuntimeError("Unrelated fixture check failed")
        unrelated = kernel.artifacts.put(
            unrelated_output,
            principal=OWNER,
            goal_id=GOAL_ID,
            media_type="text/plain",
        )
        original_commitments = tuple(c.id for c in _records(kernel, Commitment))
        running = asyncio.create_task(kernel.run_until_idle())
        try:
            old_packet = await asyncio.wait_for(worker.started.get(), 5)
            assert old_packet.goal.revision == 4
            worker.phase = "wait"
            corrected = criteria[0].model_copy(
                update={
                    "revision": 2,
                    "verifier_id": "local-check-v1",
                    "acceptance_rule": "exit_zero",
                    "check_id": "static-python310",
                    "workspace_inputs": ("release.py", "requires-python.txt"),
                    "description": (
                        "Keep Python 3.10: static grammar and current-interpreter fixture test"
                    ),
                }
            )
            await _submit(
                kernel,
                ReviseGoal(
                    goal_id=GOAL_ID,
                    expected_revision=4,
                    outcome=(
                        "Prepare reviewed release preserving Python 3.10; "
                        "deployment requires separate authorization"
                    ),
                    constraints=(
                        Constraint(
                            description="Preserve Python 3.10 compatibility",
                            enforcement="rule",
                            enforcement_id="fixture-python310-v1",
                        ),
                    ),
                    criteria=(corrected, criteria[1]),
                    explanation="Owner adds Python 3.10 requirement",
                ),
            )
            await asyncio.wait_for(running, 5)
        finally:
            worker.release.set()
            await asyncio.gather(running, return_exceptions=True)
        assert worker.delayed_proposal is not None
        await kernel.run_until_idle()  # Drain the actual late usage, never its stale work.
        await _submit(kernel, Reverify(goal_id=GOAL_ID))
        await kernel.run_until_idle()
        compatibility_gap = any(
            r.criterion_id == corrected.id and r.disposition != "accept"
            for r in _records(kernel, VerificationResult)
        )
        for index in range(3):
            await kernel.ingest(
                ExecutiveSignal(
                    signal_kind="log",
                    principal_id=OWNER.id,
                    producer_id="fixture-log-" + str(index),
                    scope=_scope(),
                    payload={"message": "same pending release log"},
                ),
                principal=OWNER,
            )
        await kernel.run_until_idle()
        # A real local effect loses its reply and survives pause/restart unresolved.
        worker.phase = "stage"
        await _submit(
            kernel,
            ShareArtifact(
                artifact_id=unrelated.id,
                expected_revision=unrelated.revision,
                principal_ids=(kernel.worker_principals["fixture"].id,),
            ),
        )
        retained = next(a for a in _records(kernel, ArtifactRef) if a.id == unrelated.id)
        await _notify_evidence(kernel, retained, purpose="prepare_local_review_staging")
        await kernel.run_until_idle()
        await _approve(kernel)
        await kernel.run_until_idle()
        unknown = next(
            a for a in _records(kernel, ActionRecord) if a.capability_id == "fixture_prepare_review"
        )
        assert unknown.status in {"outcome_unknown", "needs_human"}
        interrupted = _records(kernel, TaskFrame)[0]
        current = _records(kernel, Goal)[0]
        await _submit(
            kernel,
            ReviseGoal(
                goal_id=GOAL_ID,
                expected_revision=current.revision,
                outcome=(
                    "Urgent correction: prepare a local Python 3.10 release "
                    "for review only; no deployment"
                ),
                explanation="Suspend the existing work with its compatibility requirement intact",
            ),
        )
        current = _records(kernel, Goal)[0]
        await _submit(
            kernel,
            PauseGoal(
                goal_id=GOAL_ID,
                expected_revision=current.revision,
                explanation="Urgent owner correction; retain the task frame",
            ),
        )
        await kernel.run_until_idle()
        paused = _records(kernel, TaskFrame)[0]
        budget = _records(kernel, Budget)[0]
        limits = _records(kernel, Goal)[0].limits
        assert kernel.store.view() == rebuild(kernel.store.events())

    worker.phase = "repair"
    adapter.lookup_available = True
    async with open_kernel() as kernel:
        assert _records(kernel, Goal)[0].status == "paused"
        assert _records(kernel, TaskFrame)[0] == paused
        assert _records(kernel, Budget)[0] == budget
        assert _records(kernel, Goal)[0].limits == limits
        assert kernel.inspect().summary.unknown_action_ids == (unknown.id,)
        before = (workspace / "review-staged.txt").stat()
        blocked_retry = await kernel.dispatcher.dispatch_one(unknown.id)
        assert blocked_retry.reason_code == "action_not_dispatchable"
        await kernel.dispatcher.reconcile_one(unknown.id)
        after = (workspace / "review-staged.txt").stat()
        assert (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        assert _records(kernel, Goal)[0].status == "paused"
        resume_packet_index = len(worker.packets)
        current = _records(kernel, Goal)[0]
        await _submit(kernel, ResumeGoal(goal_id=GOAL_ID, expected_revision=current.revision))
        retained = next(a for a in _records(kernel, ArtifactRef) if a.id == unrelated.id)
        await _notify_evidence(kernel, retained, purpose="resume_release_with_retained_evidence")
        await kernel.run_until_idle()
        draft = next(a for a in _records(kernel, ArtifactRef) if a.media_type == "text/x-python")
        assert not draft.shared_with
        resumed_packet = worker.packets[resume_packet_index]
        assert _has_resume_context(resumed_packet)
        assert resumed_packet.goal.constraints and resumed_packet.goal.criteria[0] == corrected
        await _share(kernel, [draft], worker=True)
        shared_draft = next(a for a in _records(kernel, ArtifactRef) if a.id == draft.id)
        await _notify_evidence(kernel, shared_draft, purpose="apply_shared_release_repair")
        await kernel.run_until_idle()
        await _approve(kernel)  # Exact current write binding, not the delayed proposal.
        await kernel.run_until_idle()
        await _approve(kernel)  # Owner-selected independent local check.
        await kernel.run_until_idle()
        check = next(a for a in _records(kernel, ActionRecord) if a.capability_id == "run_check")
        assert check.status == "succeeded"
        await _share(
            kernel,
            [
                a
                for a in _records(kernel, ArtifactRef)
                if a.media_type != "application/vnd.sgp.executive-context+json"
                and a.id not in {unrelated.id, incompatible.id}
            ],
        )
        current = _records(kernel, Goal)[0]
        review_artifact = next(a for a in _records(kernel, ArtifactRef) if a.id == draft.id)
        await _submit(
            kernel,
            ApproveCriterion(
                goal_id=GOAL_ID,
                expected_revision=current.revision,
                criterion=reference(current.criteria[1]),
                artifacts=(reference(review_artifact),),
                explanation="Reviewed synthetic artifact; authorize local release preparation only",
            ),
        )
        await kernel.run_until_idle()  # Settle owner-requested verification before a new claim.
        worker.phase = "claim"
        await _notify_evidence(kernel, review_artifact, purpose="complete_owner_reviewed_release")
        summary = await kernel.run_until_idle()
        state = kernel.store.view()
        receipts = tuple(_records(kernel, DecisionReceipt))
        actions = _records(kernel, ActionRecord)
        findings = _records(kernel, VerificationResult)
        retained = next(a for a in _records(kernel, ArtifactRef) if a.id == unrelated.id)
        intact = (
            kernel.artifacts.read(retained, principal=OWNER, goal_id=GOAL_ID)
            == b"Unrelated fixture arithmetic test: 2+2 == 4 PASS\n"
        )
        granted = {
            capability
            for grant in _records(kernel, Grant)
            if grant.subject_id == kernel.worker_principals["fixture"].id
            and not grant.revoked
            and grant.scope.goal_id == GOAL_ID
            for capability in grant.capabilities
        }
        metrics = {
            "stale_proposals_rejected": sum(
                r.disposition == "stale" and r.subject_id == old_packet.episode_id for r in receipts
            ),
            "forbidden_dispatches": sum(
                a.dispatch_attempts
                for a in actions
                if a.capability_id not in granted or a.episode_id == old_packet.episode_id
            ),
            "resumed_commitments": len(
                set(original_commitments)
                & {c.id for c in _records(kernel, Commitment) if c.status == "completed"}
                & {c.id for c in resumed_packet.commitments}
            ),
            "coalesced_events": sum(r.reason_code == "attention_coalesced" for r in receipts),
            "compatibility_checks_passed": sum(
                r.criterion_id == corrected.id and r.disposition == "accept" for r in findings
            ),
            "unrelated_artifacts_retained": int(intact and compatibility_gap),
            "reconciled_effects": sum(
                a.id == unknown.id and a.status == "succeeded" and a.dispatch_attempts == 1
                for a in actions
            ),
            "journal_rebuild_matches": int(state == rebuild(kernel.store.events())),
            "resumed_context_supplied": int(_has_resume_context(resumed_packet)),
            "interrupted_frames_retained": int(
                paused.id == interrupted.id and paused.fencing_token > interrupted.fencing_token
            ),
        }
        expected = {
            "stale_proposals_rejected": 1,
            "forbidden_dispatches": 0,
            "resumed_commitments": 1,
            "coalesced_events": 2,
            "compatibility_checks_passed": 1,
            "unrelated_artifacts_retained": 1,
            "reconciled_effects": 1,
            "journal_rebuild_matches": 1,
            "interrupted_frames_retained": 1,
            "resumed_context_supplied": 1,
        }
        failures = tuple(key for key, value in expected.items() if metrics[key] != value)
        if GOAL_ID not in summary.completed_goal_ids:
            failures += ("release_preparation_incomplete",)
        return EvaluationResult(
            case_id=CASE_ID,
            policy_version="deterministic-v1",
            passed=not failures,
            receipts=receipts,
            summary=summary,
            failures=failures,
            metrics=metrics,
        )
