"""Completion requires current, authenticated and actually readable evidence."""

import asyncio
import hashlib
import sys
from datetime import timedelta
from uuid import uuid4

import pytest

from signal_gating.executive.artifacts import ArtifactStore
from signal_gating.executive.authority import reference
from signal_gating.executive.intake import Intake
from signal_gating.executive.messages import (
    ApproveCriterion,
    AttachEvidence,
    CommitBatch,
    CompletionClaim,
    EpisodeResult,
    Reverify,
    ReviseGoal,
    ShareArtifact,
)
from signal_gating.executive.proposals import ProposalValidator
from signal_gating.executive.records import (
    Belief,
    Commitment,
    Constraint,
    CriterionApproval,
    Episode,
    ExecutiveSignal,
    Limits,
    Scope,
    VerificationResult,
)
from signal_gating.executive.reducers import rebuild
from signal_gating.executive.store import Store
from signal_gating.executive.verification import ArtifactVerifier, VerificationService

from .factories import (
    COMMITMENT_ID,
    EPISODE_ID,
    GOAL_ID,
    NOW,
    OWNER_ID,
    TASK_ID,
    criterion,
    frame,
    goal,
    grant,
    owner,
    packet,
)
from .test_memory import event


def get(store, identifier):
    return next(r for r in store.view().records if r.id == identifier)


def change(store, record, **updates):
    store.commit(
        CommitBatch(
            events=(
                event((record.model_copy(update={"revision": record.revision + 1, **updates}),)),
            ),
            expected_revisions=(reference(record),),
        )
    )


@pytest.fixture
def setup(tmp_path):
    with Store.open(tmp_path / "store") as store:
        g = goal(
            criteria=(
                criterion(
                    verifier_id="artifact-integrity-v1", acceptance_rule="all_exist_and_intact"
                ),
            )
        )
        c = Commitment(id=COMMITMENT_ID, goal_id=g.id, responsible_principal_id=OWNER_ID)
        store.commit(CommitBatch(events=(event((owner(), g, c, frame())),)))
        artifacts = ArtifactStore(store)
        a = artifacts.put(
            b"reviewable output", principal=owner(), goal_id=g.id, media_type="text/plain"
        )
        g = get(store, g.id)
        change(
            store,
            g,
            criteria=(
                g.criteria[0].model_copy(
                    update={"revision": 2, "required_evidence": (reference(a),)}
                ),
            ),
        )
        yield store, artifacts, a


def service(setup, **kwargs):
    store, artifacts, _ = setup
    return VerificationService(artifacts, principal=owner(), clock=lambda: NOW, **kwargs)


def claim(store, artifacts=(), *, drafts=(), staged=None):
    """A real accepted worker claim, never a hand-forged success finding."""
    task = get(store, TASK_ID)
    if task.status in {"suspended", "blocked"}:
        change(store, task, status="ready", wake_condition=None)
        task = get(store, TASK_ID)
    if task.status != "running":
        change(store, task, status="running", fencing_token=task.fencing_token + 1)
    task = get(store, TASK_ID)
    g = get(store, GOAL_ID)
    context = packet(goal=g, task=task, episode_id=EPISODE_ID)
    # Each repair has a new context and episode; old accepted lineage stays immutable.
    context = context.model_copy(update={"id": uuid4().hex})
    episode = Episode(
        task_id=task.id,
        context_packet_id=context.id,
        fencing_token=task.fencing_token,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        allowed_proposal_types=("completion_claim",),
        deadline=NOW + timedelta(minutes=5),
    )
    context = context.model_copy(
        update={
            "episode_id": episode.id,
            "fencing_token": task.fencing_token,
            "mandatory_read_set": (
                reference(g),
                reference(task),
                *(reference(c) for c in g.constraints),
                *(reference(r) for r in store.view().records if r.kind == "grant"),
            ),
        }
    )
    store.commit(CommitBatch(events=(event((context, episode)),)))
    proposal = CompletionClaim(
        episode_id=episode.id,
        fencing_token=episode.fencing_token,
        criterion_ids=tuple(c.id for c in g.criteria),
        artifacts=artifacts,
        read_set=context.mandatory_read_set,
    )
    batch = ProposalValidator(clock=lambda: NOW, staged_drafts=staged).validate(
        EpisodeResult(proposals=(proposal,), artifact_drafts=drafts, stop_reason="proposed"),
        episode,
        store.view(),
    )
    assert batch.response_receipt.disposition == "accept", batch.response_receipt
    store.commit(batch)
    return proposal


def finding(store):
    return next(r for r in store.view().records if isinstance(r, VerificationResult))


@pytest.mark.asyncio
async def test_complete_only_after_current_criteria_and_required_commitments(setup):
    store, _, a = setup
    proposal = claim(store, (reference(a),))
    batch = await service(setup).evaluate(proposal, store.view())
    assert get(store, GOAL_ID).status == "active"
    store.commit(batch)
    assert finding(store).disposition == "accept"
    assert get(store, TASK_ID).status == "completed"
    assert get(store, COMMITMENT_ID).status == "completed"
    assert get(store, GOAL_ID).status == "completed"
    assert rebuild(store.events()) == store.view()
    with Store.open_readonly(store.root) as read:
        assert read.view() == store.view()


@pytest.mark.asyncio
async def test_optional_commitment_does_not_block_but_required_work_does(setup):
    store, _, a = setup
    optional = Commitment(goal_id=GOAL_ID, responsible_principal_id=OWNER_ID, required=False)
    store.commit(CommitBatch(events=(event((optional,)),)))
    proposal = claim(store, (reference(a),))
    store.commit(await service(setup).evaluate(proposal, store.view()))
    assert get(store, GOAL_ID).status == "completed"
    assert get(store, optional.id).status == "pending"


@pytest.mark.asyncio
async def test_missing_criterion_and_unrelated_pass_never_complete(setup):
    store, _, a = setup
    g = get(store, GOAL_ID)
    missing = criterion(
        id="e" * 32,
        verifier_id="registered-check-v1",
        acceptance_rule="exit_zero",
        check_id="missing",
        workspace_inputs=("source",),
    )
    change(store, g, criteria=(*g.criteria, missing))
    proposal = claim(store, (reference(a),))
    store.commit(await service(setup).evaluate(proposal, store.view()))
    unrelated_pass = next(
        r
        for r in store.view().records
        if isinstance(r, VerificationResult) and r.criterion_id == missing.id
    )
    assert unrelated_pass.disposition == "insufficient_evidence"
    assert unrelated_pass.gaps[0].criterion_id == missing.id
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_two_repairs_then_block_and_material_evidence_resets(setup):
    store, artifacts, a = setup
    g = get(store, GOAL_ID)
    change(
        store,
        g,
        criteria=(g.criteria[0].model_copy(update={"revision": 3, "required_evidence": ()}),),
    )
    for expected in ("ready", "ready", "blocked"):
        proposal = claim(store)
        store.commit(await service(setup).evaluate(proposal, store.view()))
        assert get(store, TASK_ID).status == expected
    old = finding(store)
    assert old.revision == 3
    assert len([r for r in store.view().records if isinstance(r, VerificationResult)]) == 1
    # An actual owner evidence attachment supplies a new material input.
    batch = Intake(store).accept(
        AttachEvidence(goal_id=GOAL_ID, criterion_id=g.criteria[0].id, evidence=(reference(a),)),
        principal=owner(),
        state=store.view(),
    )
    store.commit(batch)
    request = next(
        r
        for r in store.view().records
        if isinstance(r, ExecutiveSignal) and r.signal_kind == "attach_evidence"
    )
    store.commit(await service(setup).consume_request(request, store.view()))
    assert finding(store).id == old.id
    assert finding(store).disposition == "accept"
    assert get(store, TASK_ID).failure_count == 0
    assert artifacts.read(a, principal=owner(), goal_id=GOAL_ID)
    store.commit(await service(setup).reverify(GOAL_ID, store.view()))
    assert finding(store).disposition == "accept"


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "corrupt", "invalid", "revoked"])
async def test_invalidated_evidence_reopens_completion(setup, damage):
    store, artifacts, a = setup
    verifier = owner(id="f" * 32, roles=("verifier",))
    store.commit(CommitBatch(events=(event((verifier,)),)))
    share = Intake(store).accept(
        ShareArtifact(artifact_id=a.id, principal_ids=(verifier.id,)),
        principal=owner(),
        state=store.view(),
    )
    store.commit(share)
    a = get(store, a.id)
    g = get(store, GOAL_ID)
    change(
        store,
        g,
        criteria=(
            g.criteria[0].model_copy(update={"revision": 3, "required_evidence": (reference(a),)}),
        ),
    )
    svc = VerificationService(artifacts, principal=verifier, clock=lambda: NOW)
    store.commit(await svc.evaluate(claim(store, (reference(a),)), store.view()))
    assert get(store, GOAL_ID).status == "completed"
    if damage in {"missing", "corrupt"}:
        path = store.root / a.storage_location
        if damage == "missing":
            path.unlink()
        else:
            path.write_bytes(b"broken")
    elif damage == "invalid":
        artifacts.delete(a, principal=owner())
    else:
        change(store, a, shared_with=())
    store.commit(await svc.reverify(GOAL_ID, store.view()))
    assert get(store, GOAL_ID).status == "active"
    assert finding(store).disposition != "accept"


@pytest.mark.asyncio
async def test_late_verifier_cannot_accept_corrected_goal(setup, monkeypatch):
    store, _, a = setup
    proposal = claim(store, (reference(a),))
    original = ArtifactVerifier.verify

    async def delayed(self, c, state):
        result = await original(self, c, state)
        command = ReviseGoal(
            goal_id=GOAL_ID,
            expected_revision=get(store, GOAL_ID).revision,
            outcome="Corrected requirement",
        )
        store.commit(Intake(store).accept(command, principal=owner(), state=store.view()))
        await asyncio.sleep(0)
        return result

    monkeypatch.setattr(ArtifactVerifier, "verify", delayed)
    batch = await service(setup).evaluate(proposal, store.view())
    store.commit(batch)
    assert batch.response_receipt.disposition == "stale"
    assert get(store, GOAL_ID).status == "active"
    assert not any(
        isinstance(r, VerificationResult) and r.disposition == "accept"
        for r in store.view().records
    )


def human(setup):
    store, _, a = setup
    g = get(store, GOAL_ID)
    c = g.criteria[0].model_copy(
        update={
            "revision": 3,
            "evaluator_type": "human",
            "verifier_id": "owner-approval-v1",
            "acceptance_rule": "explicit_owner_approval",
        }
    )
    change(store, g, criteria=(c,))
    return ApproveCriterion(
        goal_id=GOAL_ID,
        expected_revision=get(store, GOAL_ID).revision,
        criterion=reference(c),
        artifacts=(reference(a).model_copy(update={"kind": "artifact"}),),
        explanation="Reviewed these bytes",
    )


@pytest.mark.asyncio
async def test_human_criterion_requires_explicit_authenticated_owner_act(setup):
    store, _, a = setup
    command = human(setup)
    proposal = claim(store, (reference(a),))
    store.commit(await service(setup).evaluate(proposal, store.view()))
    assert finding(store).disposition == "needs_human"
    command = command.model_copy(update={"expected_revision": get(store, GOAL_ID).revision})
    batch = Intake(store).accept(command, principal=owner(), state=store.view())
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)
    assert any(isinstance(r, CriterionApproval) for r in store.view().records)
    store.commit(await service(setup).evaluate(claim(store, (reference(a),)), store.view()))
    assert get(store, GOAL_ID).status == "completed"
    stable = finding(store).id
    # Completing the goal only advances lifecycle revisions, not owner intent.
    store.commit(await service(setup).reverify(GOAL_ID, store.view()))
    assert finding(store).id == stable
    assert finding(store).disposition == "accept"
    assert get(store, GOAL_ID).status == "completed"


@pytest.mark.parametrize(
    "source", ["worker", "forged_origin", "stale_goal", "stale_artifact", "stale_criterion"]
)
def test_owner_approval_role_and_revision_controls(setup, source):
    store, _, a = setup
    command = human(setup)
    principal = owner()
    if source in {"worker", "forged_origin"}:
        worker = owner(id="f" * 32, roles=("worker",))
        store.commit(CommitBatch(events=(event((worker,)),)))
        principal = (
            worker.model_copy(update={"roles": ("owner",)})
            if source == "worker"
            else owner(authenticated_origin="forged")
        )
    elif source == "stale_goal":
        command = command.model_copy(update={"expected_revision": 1})
    elif source == "stale_artifact":
        change(store, a, shared_with=(OWNER_ID,))
    else:
        command = command.model_copy(
            update={"criterion": command.criterion.model_copy(update={"revision": 1})}
        )
    batch = Intake(store).accept(command, principal=principal, state=store.view())
    assert batch.response_receipt.disposition in {"deny", "stale"}
    store.commit(batch)
    assert not any(isinstance(r, CriterionApproval) for r in store.view().records)


@pytest.mark.asyncio
async def test_spoofed_signal_metadata_and_approval_shaped_draft_cannot_approve(setup):
    store, artifacts, a = setup
    command = human(setup)
    worker = owner(id="f" * 32, roles=("worker",))
    store.commit(CommitBatch(events=(event((worker,)),)))
    fake = ExecutiveSignal(
        signal_kind="criterion_approved",
        producer_id="owner-criterion-approval",
        principal_id=OWNER_ID,
        scope=Scope(goal_id=GOAL_ID),
        payload={"command": command.model_dump(mode="json")},
    )
    store.commit(Intake(store).accept(fake, principal=worker, state=store.view()))
    # Owner origin and approval-like JSON are data, even after genuine staging.
    from signal_gating.executive.records import ArtifactDraft

    draft_specs = (
        ArtifactDraft(
            key="approval",
            text=command.model_dump_json(),
            media_type="application/vnd.sgp.approval+json",
        ),
    )
    drafts = artifacts.stage_drafts(draft_specs, principal=owner(), goal_id=GOAL_ID)
    try:
        proposal = claim(store, (reference(a),), drafts=draft_specs, staged=drafts)
    finally:
        artifacts.release_staged(drafts.values())
    assert all(get(store, ref.id).origin == OWNER_ID for ref in drafts.values())
    store.commit(await service(setup).evaluate(proposal, store.view()))
    assert finding(store).disposition == "needs_human"
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_reverify_request_is_consumed_without_implying_success(setup):
    store, _, _ = setup
    store.commit(
        Intake(store).accept(Reverify(goal_id=GOAL_ID), principal=owner(), state=store.view())
    )
    request = next(
        r
        for r in store.view().records
        if isinstance(r, ExecutiveSignal) and r.signal_kind == "reverify"
    )
    assert not any(isinstance(r, VerificationResult) for r in store.view().records)
    store.commit(await service(setup).consume_request(request, store.view()))
    assert finding(store).disposition == "accept"
    assert get(store, GOAL_ID).status == "active"  # A ready task has not claimed completion.


async def checked(setup, tmp_path, *, program="print('checked')", extra_criterion=False):
    """Real registered action with owner review, reserve, dispatch and accounting."""
    from signal_gating.executive.actions import CapabilityRegistry, Dispatcher
    from signal_gating.executive.adapters.local_tools import LocalWorkspace, StoreArtifacts
    from signal_gating.executive.authority import Authority
    from signal_gating.executive.budgets import BudgetPolicy
    from signal_gating.executive.config import ArgvCheck
    from signal_gating.executive.messages import ActionIntent, DecideAction
    from signal_gating.executive.records import Budget

    store, artifacts, _ = setup
    root = tmp_path / "workspace"
    root.mkdir()
    root.joinpath("source").write_text("current input")
    g = get(store, GOAL_ID)
    c = g.criteria[0].model_copy(
        update={
            "revision": 3,
            "verifier_id": "registered-check-v1",
            "acceptance_rule": "exit_zero",
            "required_evidence": (),
            "check_id": "check",
            "workspace_inputs": ("source",),
        }
    )
    worker = owner(id="e" * 32, roles=("worker",))
    budget = Budget(
        id="f" * 32,
        scope=Scope(goal_id=g.id),
        limits=Limits(
            input_tokens=0, output_tokens=0, calls=10, elapsed_seconds=120, cost_microunits=0
        ),
    )
    authorization = grant(
        subject_id=worker.id,
        capabilities=("run_check",),
        scope=Scope(owner_id=OWNER_ID, goal_id=g.id, resource_paths=(".",)),
    )
    task = get(store, TASK_ID)
    store.commit(
        CommitBatch(
            events=(
                event(
                    (
                        worker,
                        budget,
                        authorization,
                        g.model_copy(
                            update={
                                "revision": g.revision + 1,
                                "budget_id": budget.id,
                                "criteria": (
                                    c,
                                    *(
                                        (
                                            c.model_copy(
                                                update={"id": "d" * 32, "check_id": "other"}
                                            ),
                                        )
                                        if extra_criterion
                                        else ()
                                    ),
                                ),
                            }
                        ),
                        task.model_copy(
                            update={
                                "revision": task.revision + 1,
                                "status": "running",
                                "fencing_token": 1,
                            }
                        ),
                    )
                ),
            ),
            expected_revisions=(reference(g), reference(task)),
        )
    )
    g, task = get(store, g.id), get(store, task.id)
    argv = (sys.executable, "-c", program)
    workspace = LocalWorkspace(
        root,
        readable=(".",),
        writable=(),
        commands={"check": argv},
        artifacts=StoreArtifacts(store),
        check_options={
            "check": ArgvCheck(
                id="check",
                argv=argv,
                timeout_seconds=3,
                max_output_bytes=4096,
                maximum_charge_microunits=0,
            )
        },
    )
    registry = CapabilityRegistry()
    workspace.register(registry)
    context = packet(
        goal=g,
        task=task,
        capabilities=tuple(registry.specs.values()),
        episode_id=EPISODE_ID,
        fencing_token=1,
        mandatory_read_set=(reference(g), reference(task), reference(c), reference(authorization)),
    )
    episode = Episode(
        id=EPISODE_ID,
        task_id=task.id,
        context_packet_id=context.id,
        fencing_token=1,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        allowed_proposal_types=("action_intent",),
        deadline=NOW + timedelta(minutes=5),
    )
    store.commit(CommitBatch(events=(event((context, episode)),)))
    intent = ActionIntent(
        episode_id=episode.id,
        fencing_token=1,
        capability_id="run_check",
        arguments={
            "command_id": "check",
            "criterion_ref": reference(c).model_dump(mode="json"),
            "expected_sha256": {
                "source": hashlib.sha256(root.joinpath("source").read_bytes()).hexdigest()
            },
        },
        scope=Scope(owner_id=OWNER_ID, goal_id=g.id, resource_paths=(".",)),
        read_set=context.mandatory_read_set,
    )
    authority = Authority(
        worker_principals={"fixture": worker.id},
        capabilities=registry.specs,
        resource_scopes={name: workspace.resource_scope for name in registry.specs},
    )
    batch = ProposalValidator(clock=lambda: NOW, worker_principals={"fixture": worker.id}).validate(
        EpisodeResult(proposals=(intent,), stop_reason="proposed"), episode, store.view()
    )
    assert batch.response_receipt.disposition == "accept", batch.response_receipt
    store.commit(batch)
    dispatcher = Dispatcher(
        store,
        authority=authority,
        budgets=BudgetPolicy(clock=lambda: NOW),
        registry=registry,
        clock=lambda: NOW,
        wait_until=lambda deadline: asyncio.sleep((deadline - NOW).total_seconds()),
    )
    try:
        assert (await dispatcher.dispatch_one(intent.id)).disposition == "require_review"
        action = get(store, intent.id)
        batch = Intake(store, clock=lambda: NOW).accept(
            DecideAction(
                action_id=action.id,
                expected_revision=action.revision,
                decision="approve",
                arguments_digest=action.arguments_digest,
                scope=action.scope,
                constraint_revision=action.constraint_revision,
            ),
            principal=owner(),
            state=store.view(),
        )
        assert batch.response_receipt.disposition == "accept", batch.response_receipt
        store.commit(batch)
        dispatched = await dispatcher.dispatch_one(intent.id)
        assert dispatched.disposition == "accept", dispatched
    finally:
        await dispatcher.close()
    action = get(store, intent.id)
    assert action.status == "succeeded"
    assert get(store, action.reservation_id).status == "settled"
    return workspace, registry, action


@pytest.mark.asyncio
async def test_registered_check_real_dispatch_and_sharing_survive_verification(setup, tmp_path):
    from signal_gating.executive.adapters.local_tools import StoreArtifacts

    store, artifacts, _ = setup
    workspace, registry, action = await checked(setup, tmp_path)
    verifier = owner(id="a" * 32, roles=("verifier",))
    store.commit(CommitBatch(events=(event((verifier,)),)))
    # Share all three refs. Prepared/output immutable identities retain provenance.
    for ref in action.evidence:
        current = get(store, ref.id)
        batch = Intake(store).accept(
            ShareArtifact(
                artifact_id=ref.id, expected_revision=current.revision, principal_ids=(verifier.id,)
            ),
            principal=owner(),
            state=store.view(),
        )
        assert batch.response_receipt.disposition == "accept"
        store.commit(batch)
    adapter = next(a for s, a in workspace.capabilities() if s.id == "run_check")
    reconciled = await adapter.reconcile(action)
    assert reconciled.outcome == "succeeded", reconciled
    assert all(ref.revision == get(store, ref.id).revision for ref in reconciled.evidence)
    assert StoreArtifacts(store).receipts(action)
    svc = VerificationService(
        artifacts,
        principal=verifier,
        clock=lambda: NOW,
        local_checks={"registered-check-v1": (workspace, registry)},
    )
    proposal = claim(store)
    batch = await svc.evaluate(proposal, store.view())
    store.commit(batch)
    assert finding(store).disposition == "accept", finding(store)
    assert get(store, GOAL_ID).status == "completed"
    from signal_gating.executive.memory import records_by_id

    indexed = records_by_id(store.view())
    assert all(indexed[r.id].revision == r.revision for r in finding(store).read_set)
    store.commit(await svc.reverify(GOAL_ID, store.view()))
    assert finding(store).disposition == "accept"
    assert get(store, GOAL_ID).status == "completed"
    assert rebuild(store.events()) == store.view()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["input", "revoked", "missing", "corrupt", "criterion", "configuration"]
)
async def test_registered_check_requires_current_input_and_all_receipt_bytes(
    setup, tmp_path, damage
):
    store, artifacts, _ = setup
    workspace, registry, action = await checked(setup, tmp_path)
    svc = VerificationService(
        artifacts,
        principal=owner(),
        clock=lambda: NOW,
        local_checks={"registered-check-v1": (workspace, registry)},
    )
    store.commit(await svc.reverify(GOAL_ID, store.view()))
    assert finding(store).disposition == "accept", finding(store)
    if damage == "input":
        workspace.root.joinpath("source").write_text("changed input")
    elif damage in {"revoked", "missing", "corrupt"}:
        output = get(store, action.evidence[0].id)
        if damage == "revoked":
            artifacts.delete(output, principal=owner())
        elif damage == "missing":
            (store.root / output.storage_location).unlink()
        else:
            (store.root / output.storage_location).write_bytes(b"corrupt output")
    elif damage == "criterion":
        g = get(store, GOAL_ID)
        c = g.criteria[0].model_copy(update={"revision": 4, "acceptance_rule": "stronger_rule"})
        store.commit(
            Intake(store).accept(
                ReviseGoal(goal_id=g.id, expected_revision=g.revision, criteria=(c,)),
                principal=owner(),
                state=store.view(),
            )
        )
    else:
        # Registration identity is part of the current operator-selected binding.
        workspace.register(registry)
    store.commit(await svc.reverify(GOAL_ID, store.view()))
    assert finding(store).disposition == "insufficient_evidence"
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_pass_from_unrelated_command_never_satisfies_another_criterion(setup, tmp_path):
    store, artifacts, _ = setup
    workspace, registry, _ = await checked(setup, tmp_path, extra_criterion=True)
    svc = VerificationService(
        artifacts,
        principal=owner(),
        clock=lambda: NOW,
        local_checks={"registered-check-v1": (workspace, registry)},
    )
    store.commit(await svc.evaluate(claim(store), store.view()))
    results = {r.criterion_id: r for r in store.view().records if isinstance(r, VerificationResult)}
    assert results["6" * 32].disposition == "accept"
    assert results["d" * 32].disposition == "insufficient_evidence"
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocker", ["required_commitment", "unsettled_effect", "constraint", "explicit_dependency"]
)
async def test_proof_does_not_override_required_work_effects_or_constraints(setup, blocker):
    store, _, a = setup
    if blocker == "required_commitment":
        store.commit(
            CommitBatch(
                events=(event((Commitment(goal_id=GOAL_ID, responsible_principal_id=OWNER_ID),)),)
            )
        )
    elif blocker == "unsettled_effect":
        from .factories import action

        store.commit(CommitBatch(events=(event((action(status="outcome_unknown"),)),)))
    elif blocker == "constraint":
        g = get(store, GOAL_ID)
        change(
            store,
            g,
            constraints=(
                Constraint(description="Owner decision needed", enforcement="human_review"),
            ),
        )
    else:
        g = get(store, GOAL_ID)
        # The definition can be read, but an explicit task prerequisite must pass.
        prerequisite = criterion(
            id="d" * 32, verifier_id="unconfigured", acceptance_rule="unknown", required=False
        )
        change(store, g, criteria=(*g.criteria, prerequisite))
        task = get(store, TASK_ID)
        change(store, task, status="blocked", dependencies=(reference(prerequisite),))
        # Bypass the test helper's running edge: explicit prerequisites prevent a
        # fresh worker at the reducer too. Reverify cannot finish that work.
        store.commit(await service(setup).reverify(GOAL_ID, store.view()))
        assert get(store, TASK_ID).status == "blocked"
        assert get(store, GOAL_ID).status == "active"
        return
    store.commit(await service(setup).evaluate(claim(store, (reference(a),)), store.view()))
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_human_approval_material_correction_and_exact_artifact_binding(setup):
    store, _, a = setup
    command = human(setup)
    store.commit(Intake(store).accept(command, principal=owner(), state=store.view()))
    store.commit(await service(setup).evaluate(claim(store, (reference(a),)), store.view()))
    assert get(store, GOAL_ID).status == "completed"
    g = get(store, GOAL_ID)
    store.commit(
        Intake(store).accept(
            ReviseGoal(
                goal_id=g.id, expected_revision=g.revision, outcome="Different owner intent"
            ),
            principal=owner(),
            state=store.view(),
        )
    )
    assert finding(store).disposition == "insufficient_evidence"
    store.commit(await service(setup).reverify(GOAL_ID, store.view()))
    assert finding(store).disposition == "needs_human"
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_expired_belief_support_is_rechecked_without_revision_change(setup):
    store, _, a = setup
    belief = Belief(
        claim="Time limited evidence",
        scope=Scope(goal_id=GOAL_ID),
        evidence=(reference(a),),
        epistemic_status="supported",
        valid_until=NOW + timedelta(seconds=1),
    )
    g = get(store, GOAL_ID)
    store.commit(
        CommitBatch(
            events=(
                event(
                    (
                        belief,
                        g.model_copy(
                            update={
                                "revision": g.revision + 1,
                                "criteria": (
                                    g.criteria[0].model_copy(
                                        update={"revision": 3, "inputs": (reference(belief),)}
                                    ),
                                ),
                            }
                        ),
                    )
                ),
            ),
            expected_revisions=(reference(g),),
        )
    )
    store.commit(await service(setup).evaluate(claim(store, (reference(a),)), store.view()))
    assert get(store, GOAL_ID).status == "completed"
    svc = VerificationService(setup[1], principal=owner(), clock=lambda: NOW + timedelta(seconds=2))
    store.commit(await svc.reverify(GOAL_ID, store.view()))
    assert finding(store).disposition == "insufficient_evidence"
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_model_criterion_is_a_visible_unconfigured_judgment(setup):
    store, _, a = setup
    g = get(store, GOAL_ID)
    change(
        store,
        g,
        criteria=(
            g.criteria[0].model_copy(
                update={"revision": 3, "evaluator_type": "model", "verifier_id": "model-worker"}
            ),
        ),
    )
    store.commit(await service(setup).evaluate(claim(store, (reference(a),)), store.view()))
    assert finding(store).disposition == "needs_human"
    assert finding(store).policy_version == "model-judgment-unconfigured-v1"
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_final_batch_cas_preserves_owner_pause(setup):
    from signal_gating.executive.errors import ConflictError
    from signal_gating.executive.messages import PauseGoal

    store, _, a = setup
    batch = await service(setup).evaluate(claim(store, (reference(a),)), store.view())
    g = get(store, GOAL_ID)
    store.commit(
        Intake(store).accept(
            PauseGoal(goal_id=g.id, expected_revision=g.revision),
            principal=owner(),
            state=store.view(),
        )
    )
    with pytest.raises(ConflictError):
        store.commit(batch)
    assert get(store, GOAL_ID).status == "paused"
    store.commit(await service(setup).reverify(GOAL_ID, store.view()))
    assert get(store, GOAL_ID).status == "paused"


@pytest.mark.asyncio
async def test_configured_model_judgment_uses_real_accounted_dispatch(setup):
    from signal_gating.executive.actions import CapabilityRegistry, Dispatcher
    from signal_gating.executive.authority import Authority
    from signal_gating.executive.budgets import BudgetPolicy
    from signal_gating.executive.messages import ActionIntent
    from signal_gating.executive.records import ActionObservation, Budget, CapabilitySpec, Usage

    store, artifacts, a = setup
    g, task = get(store, GOAL_ID), get(store, TASK_ID)
    c = g.criteria[0].model_copy(
        update={"revision": 3, "evaluator_type": "model", "verifier_id": "review-model"}
    )
    budget = Budget(
        id="f" * 32,
        scope=Scope(goal_id=g.id),
        limits=Limits(
            input_tokens=0, output_tokens=0, calls=10, elapsed_seconds=60, cost_microunits=0
        ),
    )
    worker = owner(id="e" * 32, roles=("worker",))
    permission = grant(
        subject_id=worker.id,
        capabilities=("model-review",),
        scope=Scope(owner_id=OWNER_ID, goal_id=g.id),
    )
    store.commit(
        CommitBatch(
            events=(
                event(
                    (
                        budget,
                        worker,
                        permission,
                        g.model_copy(
                            update={
                                "revision": g.revision + 1,
                                "budget_id": budget.id,
                                "criteria": (c,),
                            }
                        ),
                        task.model_copy(
                            update={
                                "revision": task.revision + 1,
                                "status": "running",
                                "fencing_token": 1,
                            }
                        ),
                    )
                ),
            ),
            expected_revisions=(reference(g), reference(task)),
        )
    )
    store.commit(
        Intake(store).accept(
            ShareArtifact(artifact_id=a.id, principal_ids=(worker.id,)),
            principal=owner(),
            state=store.view(),
        )
    )
    a = get(store, a.id)
    g = get(store, GOAL_ID)
    c = c.model_copy(update={"revision": 4, "required_evidence": (reference(a),)})
    change(store, g, criteria=(c,))
    g, task = get(store, GOAL_ID), get(store, TASK_ID)
    usage = Usage.zero().model_copy(update={"calls": 1})
    issued = []

    class ModelFixture:
        async def execute(self, action):
            assert not store._connection.in_transaction
            assert get(store, action.id).status == "dispatching"
            assert get(store, action.reservation_id).subject_id == action.id
            issued.append(action)
            output = artifacts.put(
                b"bounded fixture judgment",
                principal=owner(),
                goal_id=g.id,
                media_type="application/json",
            )
            return ActionObservation(
                action_id=action.id,
                expected_revision=action.revision,
                outcome="succeeded",
                evidence=(reference(output),),
                usage=usage,
            )

        async def reconcile(self, action):
            raise AssertionError("not needed")

        async def cancel(self, action):
            raise AssertionError("not needed")

    registry = CapabilityRegistry()
    registry.register(
        CapabilitySpec(
            id="model-review",
            binding_version="1",
            input_schema={"type": "object"},
            effect_class="read_only",
        ),
        ModelFixture(),
        resource_scope=lambda intent: Scope(),
        quote=Limits(
            input_tokens=0, output_tokens=0, calls=1, elapsed_seconds=1, cost_microunits=0
        ),
        quote_source="fixture-model-bound",
        quote_version="review-v1",
    )
    svc = VerificationService(
        artifacts,
        principal=owner(),
        clock=lambda: NOW,
        model_checks={"review-model": (registry, "model-review", "review-v1")},
    )
    arguments = svc.judgment_arguments(c, store.view())
    context = packet(
        goal=g,
        task=task,
        fencing_token=1,
        capabilities=tuple(registry.specs.values()),
        mandatory_read_set=(
            reference(g),
            reference(task),
            reference(c),
            reference(a),
            reference(permission),
        ),
    )
    episode = Episode(
        id=EPISODE_ID,
        task_id=task.id,
        context_packet_id=context.id,
        fencing_token=1,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        allowed_proposal_types=("action_intent",),
        deadline=NOW + timedelta(minutes=1),
    )
    store.commit(CommitBatch(events=(event((context, episode)),)))
    intent = ActionIntent(
        episode_id=episode.id,
        fencing_token=1,
        capability_id="model-review",
        arguments=arguments,
        read_set=context.mandatory_read_set,
        scope=Scope(owner_id=OWNER_ID, goal_id=g.id),
    )
    batch = ProposalValidator(clock=lambda: NOW, worker_principals={"fixture": worker.id}).validate(
        EpisodeResult(proposals=(intent,), stop_reason="proposed"), episode, store.view()
    )
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)
    premature = VerificationResult(
        criterion_id=c.id,
        verifier_id=c.verifier_id,
        disposition="accept",
        reason_code="fixture",
        usage=usage,
    )
    rejected = await svc.evaluate_judgment(intent.id, premature, store.view())
    assert rejected.response_receipt.disposition == "insufficient_evidence"
    dispatcher = Dispatcher(
        store,
        authority=Authority(
            worker_principals={"fixture": worker.id},
            capabilities=registry.specs,
            resource_scopes={"model-review": lambda intent: Scope()},
        ),
        budgets=BudgetPolicy(clock=lambda: NOW),
        registry=registry,
        clock=lambda: NOW,
    )
    try:
        dispatched = await dispatcher.dispatch_one(intent.id)
        assert dispatched.disposition == "accept", dispatched
    finally:
        await dispatcher.close()
    action = get(store, intent.id)
    judgment = premature.model_copy(update={"evidence": action.evidence})
    store.commit(await svc.evaluate_judgment(action.id, judgment, store.view()))
    assert finding(store).disposition == "accept"
    assert finding(store).policy_version == "model-judgment-v1:review-v1"
    assert get(store, action.reservation_id).usage == usage
    store.commit(await svc.evaluate(claim(store), store.view()))
    assert get(store, GOAL_ID).status == "completed"
    assert finding(store).disposition == "accept"
    store.commit(await svc.reverify(GOAL_ID, store.view()))
    assert finding(store).disposition == "accept"
    # Unaccounted/mismatched observations do not reuse a settled bill.
    wrong_usage = judgment.model_copy(update={"usage": usage.model_copy(update={"calls": 2})})
    refused = await svc.evaluate_judgment(action.id, wrong_usage, store.view())
    assert refused.response_receipt.disposition == "insufficient_evidence"
    observed = dispatcher.record_observation(
        issued[0],
        ActionObservation(
            action_id=action.id,
            expected_revision=issued[0].revision,
            outcome="succeeded",
            usage=wrong_usage.usage,
            evidence=action.evidence,
        ),
    )
    assert observed.disposition == "require_review"
    store.commit(await svc.evaluate_judgment(action.id, wrong_usage, store.view()))
    assert get(store, action.reservation_id).observed_usage.calls == 2
    assert get(store, budget.id).accounting_uncertain
    assert get(store, GOAL_ID).status == "active"
    assert rebuild(store.events()) == store.view()


@pytest.mark.asyncio
async def test_spoofed_worker_proposal_signal_is_not_admitted_claim(setup):
    store, _, a = setup
    proposal = claim(store, (reference(a),))
    forged = proposal.model_copy(update={"id": uuid4().hex})
    worker = owner(id="e" * 32, roles=("worker",))
    store.commit(CommitBatch(events=(event((worker,)),)))
    signal = ExecutiveSignal(
        id=forged.id,
        signal_kind="completion_claim",
        producer_id="worker-proposal",
        principal_id=OWNER_ID,
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
        payload={"proposal": forged.model_dump(mode="json")},
    )
    store.commit(Intake(store).accept(signal, principal=worker, state=store.view()))
    batch = await service(setup).evaluate(forged, store.view())
    assert batch.response_receipt.disposition == "stale"
    store.commit(batch)
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_late_registered_verifier_cannot_accept_rotated_configuration(
    setup, tmp_path, monkeypatch
):
    from signal_gating.executive.verification import LocalCheckVerifier

    store, artifacts, _ = setup
    workspace, registry, _ = await checked(setup, tmp_path)
    svc = VerificationService(
        artifacts,
        principal=owner(),
        clock=lambda: NOW,
        local_checks={"registered-check-v1": (workspace, registry)},
    )
    original = LocalCheckVerifier.verify

    async def rotate(self, c, state):
        result = await original(self, c, state)
        workspace.register(registry)
        await asyncio.sleep(0)
        return result

    monkeypatch.setattr(LocalCheckVerifier, "verify", rotate)
    batch = await svc.evaluate(claim(store), store.view())
    assert batch.response_receipt.disposition == "stale"
    store.commit(batch)
    assert get(store, GOAL_ID).status == "active"


@pytest.mark.asyncio
async def test_failed_recheck_blocks_explicit_prerequisite_instead_of_invalid_ready(setup):
    store, _, a = setup
    store.commit(await service(setup).reverify(GOAL_ID, store.view()))
    task = get(store, TASK_ID)
    c = get(store, GOAL_ID).criteria[0]
    change(store, task, dependencies=(reference(c),))
    proposal = claim(store, (reference(a),))
    (store.root / a.storage_location).write_bytes(b"corrupt")
    store.commit(await service(setup).evaluate(proposal, store.view()))
    assert get(store, TASK_ID).status == "blocked"
    assert finding(store).disposition == "insufficient_evidence"


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency_owner", ["task", "commitment"])
@pytest.mark.parametrize("after_completion", [False, True])
async def test_explicit_prerequisite_expiry_after_claim_blocks_completion(
    setup, dependency_owner, after_completion
):
    store, artifacts, a = setup
    prerequisite = Belief(
        claim="Explicit prerequisite",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
        valid_until=NOW + timedelta(seconds=1),
    )
    store.commit(CommitBatch(events=(event((prerequisite,)),)))
    target = get(store, TASK_ID if dependency_owner == "task" else COMMITMENT_ID)
    change(store, target, dependencies=(reference(prerequisite),))
    proposal = claim(store, (reference(a),))
    if after_completion:
        store.commit(await service(setup).evaluate(proposal, store.view()))
        assert get(store, GOAL_ID).status == "completed"
    svc = VerificationService(
        artifacts, principal=owner(), clock=lambda: NOW + timedelta(seconds=2)
    )
    batch = await (
        svc.reverify(GOAL_ID, store.view())
        if after_completion
        else svc.evaluate(proposal, store.view())
    )
    store.commit(batch)
    assert get(store, GOAL_ID).status == "active"
    assert get(store, COMMITMENT_ID).status != "completed"
    assert get(store, TASK_ID).status == "blocked"
    assert reference(prerequisite) in batch.response_receipt.state_revisions
    assert reference(prerequisite) in batch.expected_revisions
    assert rebuild(store.events()) == store.view()


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency_owner", ["task", "commitment"])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "revoked"])
@pytest.mark.parametrize("after_completion", [False, True])
async def test_explicit_prerequisite_transitive_artifact_support_rechecked(
    setup, dependency_owner, damage, after_completion
):
    store, artifacts, a = setup
    verifier = owner(id="a" * 32, roles=("verifier",))
    store.commit(CommitBatch(events=(event((verifier,)),)))
    prerequisite_artifact = artifacts.put(
        b"prerequisite bytes", principal=owner(), goal_id=GOAL_ID, media_type="text/plain"
    )
    for artifact in (a, prerequisite_artifact):
        store.commit(
            Intake(store).accept(
                ShareArtifact(artifact_id=artifact.id, principal_ids=(verifier.id,)),
                principal=owner(),
                state=store.view(),
            )
        )
    a, prerequisite_artifact = get(store, a.id), get(store, prerequisite_artifact.id)
    g = get(store, GOAL_ID)
    change(
        store,
        g,
        criteria=(
            g.criteria[0].model_copy(update={"revision": 3, "required_evidence": (reference(a),)}),
        ),
    )
    inner = Belief(
        claim="Underlying support",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
        evidence=(reference(prerequisite_artifact),),
    )
    outer = Belief(
        claim="Explicit transitive prerequisite",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
        evidence=(reference(inner),),
    )
    store.commit(CommitBatch(events=(event((inner, outer)),)))
    target = get(store, TASK_ID if dependency_owner == "task" else COMMITMENT_ID)
    change(store, target, dependencies=(reference(outer),))
    svc = VerificationService(artifacts, principal=verifier, clock=lambda: NOW)
    proposal = claim(store, (reference(a),))
    if after_completion:
        store.commit(await svc.evaluate(proposal, store.view()))
        assert get(store, GOAL_ID).status == "completed"
    if damage == "missing":
        (store.root / prerequisite_artifact.storage_location).unlink()
    elif damage == "corrupt":
        (store.root / prerequisite_artifact.storage_location).write_bytes(b"corrupt")
    else:
        change(store, prerequisite_artifact, shared_with=())
    batch = await (
        svc.reverify(GOAL_ID, store.view())
        if after_completion
        else svc.evaluate(proposal, store.view())
    )
    store.commit(batch)
    assert get(store, GOAL_ID).status == "active"
    assert get(store, TASK_ID).status == "blocked"
    assert get(store, COMMITMENT_ID).status != "completed"
    assert {outer.id, inner.id, prerequisite_artifact.id} <= {
        r.id for r in batch.response_receipt.state_revisions
    }
    if damage == "missing":
        (store.root / prerequisite_artifact.storage_location).write_bytes(b"prerequisite bytes")
        store.commit(await svc.reverify(GOAL_ID, store.view()))
        assert get(store, TASK_ID).status == "suspended"
        assert get(store, TASK_ID).wake_condition is None
        assert get(store, GOAL_ID).status == "active"
    assert rebuild(store.events()) == store.view()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["invalid", "missing", "corrupt"])
async def test_claim_only_selection_survives_reverify_and_real_restart(setup, damage):
    store, artifacts, a = setup
    g = get(store, GOAL_ID)
    change(
        store,
        g,
        criteria=(g.criteria[0].model_copy(update={"revision": 3, "required_evidence": ()}),),
    )
    store.commit(await service(setup).evaluate(claim(store, (reference(a),)), store.view()))
    initial = finding(store)
    assert initial.disposition == "accept"
    assert get(store, GOAL_ID).status == "completed"
    store.commit(await service(setup).reverify(GOAL_ID, store.view()))
    assert get(store, GOAL_ID).status == "completed"
    assert finding(store).disposition == "accept"
    root = store.root
    store.close()
    with Store.open(root) as reopened:
        svc = VerificationService(ArtifactStore(reopened), principal=owner(), clock=lambda: NOW)
        reopened.commit(await svc.reverify(GOAL_ID, reopened.view()))
        assert get(reopened, GOAL_ID).status == "completed"
        assert finding(reopened).id == initial.id
        assert finding(reopened).evidence == (reference(a),)
        if damage == "invalid":
            ArtifactStore(reopened).delete(get(reopened, a.id), principal=owner())
        elif damage == "missing":
            (root / a.storage_location).unlink()
        else:
            (root / a.storage_location).write_bytes(b"corrupt")
        ArtifactStore(reopened).put(
            b"newer unselected content", principal=owner(), goal_id=GOAL_ID, media_type="text/plain"
        )
        reopened.commit(await svc.reverify(GOAL_ID, reopened.view()))
        assert get(reopened, GOAL_ID).status == "active"
        assert finding(reopened).disposition == "insufficient_evidence"
        assert finding(reopened).evidence == (reference(a),)
        reopened.commit(await svc.reverify(GOAL_ID, reopened.view()))
        assert finding(reopened).evidence == (reference(a),)
        assert finding(reopened).disposition == "insufficient_evidence"
        assert rebuild(reopened.events()) == reopened.view()


@pytest.mark.asyncio
@pytest.mark.parametrize("correction", ["outcome", "criterion"])
async def test_claim_only_selection_does_not_survive_material_correction(setup, correction):
    store, _, a = setup
    g = get(store, GOAL_ID)
    change(
        store,
        g,
        criteria=(g.criteria[0].model_copy(update={"revision": 3, "required_evidence": ()}),),
    )
    store.commit(await service(setup).evaluate(claim(store, (reference(a),)), store.view()))
    g = get(store, GOAL_ID)
    changes = (
        {"outcome": "Different material goal"}
        if correction == "outcome"
        else {
            "criteria": (
                g.criteria[0].model_copy(
                    update={"revision": 4, "description": "Different artifact selection"}
                ),
            )
        }
    )
    store.commit(
        Intake(store).accept(
            ReviseGoal(goal_id=g.id, expected_revision=g.revision, **changes),
            principal=owner(),
            state=store.view(),
        )
    )
    # The existence of any newer artifact cannot stand in for a selected input.
    setup[1].put(b"unselected", principal=owner(), goal_id=GOAL_ID, media_type="text/plain")
    store.commit(await service(setup).reverify(GOAL_ID, store.view()))
    assert get(store, GOAL_ID).status == "active"
    assert finding(store).disposition == "insufficient_evidence"
    assert finding(store).evidence == ()


@pytest.mark.asyncio
async def test_prerequisite_definition_evidence_is_not_an_unverified_prerequisite(setup):
    store, _, a = setup
    data = Belief(claim="Uncertain definition input", scope=Scope(goal_id=GOAL_ID))
    store.commit(CommitBatch(events=(event((data,)),)))
    definition = criterion(
        id="e" * 32,
        verifier_id="unconfigured-definition",
        required=False,
        inputs=(reference(data),),
    )
    g = get(store, GOAL_ID)
    change(store, g, criteria=(*g.criteria, definition))
    supported = Belief(
        claim="Support names a current definition",
        scope=Scope(goal_id=GOAL_ID),
        epistemic_status="supported",
        evidence=(reference(definition),),
    )
    store.commit(CommitBatch(events=(event((supported,)),)))
    change(store, get(store, TASK_ID), dependencies=(reference(supported),))
    store.commit(await service(setup).evaluate(claim(store, (reference(a),)), store.view()))
    assert get(store, TASK_ID).status == "completed"
    assert get(store, GOAL_ID).status == "completed"
