"""Atomic worker bundles cannot replace trusted authority or evidence policy."""

from datetime import timedelta

import pytest

from signal_gating.executive import (
    ActionIntent,
    ArtifactDraft,
    Belief,
    CommitBatch,
    Commitment,
    CompletionClaim,
    Episode,
    EpisodeResult,
    Ref,
    TaskDecomposition,
)
from signal_gating.executive.artifacts import ArtifactStore
from signal_gating.executive.memory import MemoryPolicy
from signal_gating.executive.proposals import ProposalValidator
from signal_gating.executive.reducers import rebuild
from signal_gating.executive.store import Store

from .factories import (
    COMMITMENT_ID,
    EPISODE_ID,
    GOAL_ID,
    NOW,
    OWNER_ID,
    TASK_ID,
    frame,
    goal,
    owner,
    packet,
    proposal,
)
from .test_memory import artifact, event, ref


@pytest.fixture
def running(tmp_path):
    with Store.open(tmp_path / "store") as store:
        task = frame(status="running", fencing_token=1)
        commitment = Commitment(
            id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID
        )
        context = packet(task=task, mandatory_read_set=(ref(goal()), ref(task)))
        episode = Episode(
            id=EPISODE_ID,
            task_id=TASK_ID,
            fencing_token=1,
            context_packet_id=context.id,
            worker_id="fixture",
            provider="fixture",
            model="fixture",
            allowed_proposal_types=(
                "belief_change",
                "task_decomposition",
                "action_intent",
                "evidence_request",
                "completion_claim",
            ),
            deadline=NOW + timedelta(minutes=5),
        )
        store.commit(
            CommitBatch(events=(event((owner(), goal(), commitment, task, context, episode)),))
        )
        yield store, episode, (ref(goal()), ref(task))


def result(*proposals, drafts=()):
    return EpisodeResult(proposals=proposals, artifact_drafts=drafts, stop_reason="proposed")


def validator(**kwargs):
    return ProposalValidator(clock=lambda: NOW, **kwargs)


def test_missing_mandatory_read_set_rejects_whole_bundle(running):
    store, episode, reads = running
    batch = validator().validate(result(proposal()), episode, store.view())
    assert batch.response_receipt.reason_code == "missing_mandatory_read_set"
    store.commit(batch)
    assert not any(isinstance(r, Belief) for r in store.view().records)


@pytest.mark.parametrize("mode", ["revision", "fence", "expired"])
def test_stale_and_fenced_proposals_are_durable_without_state_writes(running, mode):
    store, episode, reads = running
    before = store.view()
    if mode == "revision":
        reads = (reads[0].model_copy(update={"revision": 2}), reads[1])
    elif mode == "fence":
        episode = episode.model_copy(update={"fencing_token": 2})
    else:
        episode = episode.model_copy(update={"deadline": NOW})
    batch = validator().validate(result(proposal(read_set=reads)), episode, before)
    assert batch.response_receipt.disposition == "stale"
    assert batch.response_receipt.reason_code == "stale_dependency"
    store.commit(batch)
    assert not any(isinstance(r, Belief) for r in store.view().records)


def test_one_invalid_member_prevents_every_belief_and_action_write(running):
    store, episode, reads = running
    action = ActionIntent(
        episode_id=EPISODE_ID,
        fencing_token=1,
        read_set=reads,
        capability_id="email.send",
        arguments={},
    )
    batch = validator().validate(result(proposal(read_set=reads), action), episode, store.view())
    assert batch.response_receipt.disposition == "reject"
    assert not batch.outbox
    store.commit(batch)
    assert not any(r.kind in {"belief", "action"} for r in store.view().records)


def test_hypothesis_ignores_worker_supported_label(running):
    store, episode, reads = running
    change = proposal(read_set=reads)
    change = change.model_copy(
        update={"belief": change.belief.model_copy(update={"epistemic_status": "supported"})}
    )
    batch = validator().validate(result(change), episode, store.view())
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)
    belief = next(r for r in store.view().records if isinstance(r, Belief))
    assert belief.epistemic_status == "hypothesis"
    assert rebuild(store.events()) == store.view()


def test_trusted_conflict_marks_both_beliefs_disputed(running):
    store, episode, reads = running
    a = artifact()
    old = Belief(
        id="e" * 32, claim="Passed", scope=a.scope, evidence=(ref(a),), epistemic_status="supported"
    )
    store.commit(CommitBatch(events=(event((a, old)),)))
    change = proposal(
        read_set=reads,
        belief=proposal().belief.model_copy(
            update={"claim": "Failed", "scope": a.scope, "evidence": (ref(a),)}
        ),
    )
    memory = MemoryPolicy(
        evidence_validators={OWNER_ID: lambda c, s: True},
        conflict_detectors={OWNER_ID: lambda a, b, s: True},
        clock=lambda: NOW,
    )
    batch = validator(memory=memory).validate(result(change), episode, store.view())
    store.commit(batch)
    assert {r.epistemic_status for r in store.view().records if isinstance(r, Belief)} == {
        "disputed"
    }


def test_draft_resolution_does_not_establish_supported_evidence(running):
    store, episode, reads = running
    draft = ArtifactDraft(key="report", media_type="text/plain", text="unverified")
    artifacts = ArtifactStore(store, max_artifact_bytes=20, max_draft_bytes=20)
    staged = artifacts.stage_drafts((draft,), principal=owner(), goal_id=GOAL_ID)
    try:
        change = proposal(
            read_set=reads,
            belief=proposal().belief.model_copy(
                update={
                    "evidence": (Ref(id="draft:report", kind="draft"),),
                    "epistemic_status": "supported",
                }
            ),
        )
        batch = validator(
            staged_drafts=staged,
            memory=MemoryPolicy(
                evidence_validators={OWNER_ID: lambda c, s: True}, clock=lambda: NOW
            ),
        ).validate(result(change, drafts=(draft,)), episode, store.view())
        assert batch.response_receipt.disposition == "accept"
        store.commit(batch)
        b = next(r for r in store.view().records if isinstance(r, Belief))
        assert b.epistemic_status == "hypothesis"
        assert b.evidence == (ref(staged["report"]),)
        assert artifacts.read(staged["report"], principal=owner(), goal_id=GOAL_ID) == b"unverified"
    finally:
        artifacts.release_staged(staged.values())


def test_unstaged_or_missing_draft_rejects_bundle(running):
    store, episode, reads = running
    change = proposal(
        read_set=reads,
        belief=proposal().belief.model_copy(
            update={"evidence": (Ref(id="draft:missing", kind="draft"),)}
        ),
    )
    batch = validator().validate(result(change), episode, store.view())
    assert batch.response_receipt.disposition == "reject"
    store.commit(batch)
    assert not any(isinstance(r, Belief) for r in store.view().records)


def test_configured_draft_size_limits_fail_before_staging(running):
    store, _, _ = running
    from signal_gating.executive.errors import StoreError

    artifacts = ArtifactStore(store, max_artifact_bytes=2, max_draft_bytes=3)
    before = store.view()
    with pytest.raises(StoreError, match="limit"):
        artifacts.stage_drafts(
            (ArtifactDraft(key="large", media_type="text/plain", text="€"),),
            principal=owner(),
            goal_id=GOAL_ID,
        )
    assert store.view() == before


def test_decomposition_cycle_rejects_all_members(running):
    store, episode, reads = running
    first = frame(id="f" * 32, dependencies=(Ref(id="b" * 32),))
    second = frame(id="b" * 32, dependencies=(Ref(id="f" * 32),))
    decomposition = TaskDecomposition(
        episode_id=EPISODE_ID, fencing_token=1, read_set=reads, tasks=(first, second)
    )
    batch = validator().validate(
        result(proposal(read_set=reads), decomposition), episode, store.view()
    )
    assert batch.response_receipt.reason_code == "dependency_cycle"
    store.commit(batch)
    assert not any(isinstance(r, Belief) for r in store.view().records)


def test_decomposition_inherits_parent_scope_budget_and_dependencies(running):
    store, episode, reads = running
    child = frame(id="a" * 32)
    decomposition = TaskDecomposition(
        episode_id=EPISODE_ID, fencing_token=1, read_set=reads, tasks=(child,)
    )
    batch = validator().validate(result(decomposition), episode, store.view())
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)
    assert next(r for r in store.view().records if r.id == child.id).goal_id == GOAL_ID


def test_completion_claim_is_request_not_verification_success(running):
    store, episode, reads = running
    claim = CompletionClaim(
        episode_id=EPISODE_ID,
        fencing_token=1,
        read_set=reads,
        criterion_ids=(goal().criteria[0].id,),
    )
    batch = validator().validate(result(claim), episode, store.view())
    store.commit(batch)
    current = {r.id: r for r in store.view().records}
    assert current[GOAL_ID].status == "active"
    assert current[TASK_ID].status == "verifying"
    assert not any(r.kind == "verification" for r in current.values())


def test_known_action_is_proposed_with_canonical_digest_and_no_dispatch(running):
    import hashlib

    from signal_gating.executive import CapabilitySpec

    store, episode, reads = running
    context = next(r for r in store.view().records if r.kind == "context_packet")
    capability = CapabilitySpec(
        id="workspace.read", binding_version="1", input_schema={}, effect_class="read_only"
    )
    context = context.model_copy(update={"revision": 2, "capabilities": (capability,)})
    store.commit(CommitBatch(events=(event((context,)),), expected_revisions=(Ref(id=context.id),)))
    intent = ActionIntent(
        episode_id=EPISODE_ID,
        fencing_token=1,
        read_set=reads,
        capability_id=capability.id,
        arguments={"path": "a.txt"},
    )
    batch = validator().validate(result(intent), episode, store.view())
    assert batch.response_receipt.disposition == "accept"
    assert batch.outbox == ()
    store.commit(batch)
    action = next(r for r in store.view().records if r.kind == "action")
    assert action.status == "proposed"
    assert action.arguments_digest == hashlib.sha256(b'{"path":"a.txt"}').hexdigest()


def test_declared_human_only_capability_cannot_be_proposed(running):
    from signal_gating.executive import CapabilitySpec

    store, episode, reads = running
    context = next(r for r in store.view().records if r.kind == "context_packet")
    context = context.model_copy(
        update={
            "revision": 2,
            "capabilities": (
                CapabilitySpec(
                    id="email.send", binding_version="1", input_schema={}, effect_class="human_only"
                ),
            ),
        }
    )
    store.commit(CommitBatch(events=(event((context,)),), expected_revisions=(Ref(id=context.id),)))
    intent = ActionIntent(
        episode_id=EPISODE_ID,
        fencing_token=1,
        read_set=reads,
        capability_id="email.send",
        arguments={},
    )
    batch = validator().validate(result(intent), episode, store.view())
    assert batch.response_receipt.reason_code == "capability_denied"


def test_prepared_bundle_cannot_commit_after_dependency_changes(running):
    from signal_gating.executive import ReviseGoal
    from signal_gating.executive.errors import ConflictError
    from signal_gating.executive.intake import Intake

    store, episode, reads = running
    prepared = validator().validate(result(proposal(read_set=reads)), episode, store.view())
    store.commit(
        Intake(store, clock=lambda: NOW).accept(
            ReviseGoal(goal_id=GOAL_ID, expected_revision=1, outcome="New"),
            principal=owner(),
            state=store.view(),
        )
    )
    before = store.view()
    with pytest.raises(ConflictError):
        store.commit(prepared)
    assert store.view() == before


def test_invalid_bundle_leaves_staged_draft_uncommitted(running):
    store, episode, reads = running
    draft = ArtifactDraft(key="report", media_type="text/plain", text="report")
    artifacts = ArtifactStore(store)
    staged = artifacts.stage_drafts((draft,), principal=owner(), goal_id=GOAL_ID)
    try:
        intent = ActionIntent(
            episode_id=EPISODE_ID,
            fencing_token=1,
            read_set=reads,
            capability_id="email.send",
            arguments={},
        )
        batch = validator(staged_drafts=staged).validate(
            result(proposal(read_set=reads), intent, drafts=(draft,)), episode, store.view()
        )
        assert batch.response_receipt.disposition == "reject"
        store.commit(batch)
        assert not any(r.id == staged["report"].id for r in store.view().records)
    finally:
        artifacts.release_staged(staged.values())
    assert artifacts.collect_orphans() == 1


def test_unrelated_state_revision_does_not_stale_prepared_result(running):
    store, episode, reads = running
    a = artifact()
    store.commit(CommitBatch(events=(event((a,)),)))
    batch = validator().validate(result(proposal(read_set=reads)), episode, store.view())
    newer = a.model_copy(update={"revision": 2, "shared_with": (OWNER_ID,)})
    store.commit(CommitBatch(events=(event((newer,)),), expected_revisions=(ref(a),)))
    store.commit(batch)
    assert (
        next(r for r in store.view().records if isinstance(r, Belief)).epistemic_status
        == "hypothesis"
    )


def test_action_argument_objects_are_not_treated_as_typed_draft_refs(running):
    from signal_gating.executive import CapabilitySpec

    store, episode, reads = running
    context = next(r for r in store.view().records if r.kind == "context_packet")
    context = context.model_copy(
        update={
            "revision": 2,
            "capabilities": (
                CapabilitySpec(
                    id="workspace.read",
                    binding_version="1",
                    input_schema={},
                    effect_class="read_only",
                ),
            ),
        }
    )
    store.commit(CommitBatch(events=(event((context,)),), expected_revisions=(Ref(id=context.id),)))
    intent = ActionIntent(
        episode_id=EPISODE_ID,
        fencing_token=1,
        read_set=reads,
        capability_id="workspace.read",
        arguments={"id": "draft:business-id", "revision": 1},
    )
    batch = validator().validate(result(intent), episode, store.view())
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)
    assert next(r for r in store.view().records if r.kind == "action").arguments == intent.arguments


def test_contradictory_supported_claims_inside_one_bundle_are_both_disputed(running):
    store, episode, reads = running
    a = artifact()
    store.commit(CommitBatch(events=(event((a,)),)))
    first = proposal(
        read_set=reads,
        belief=proposal().belief.model_copy(
            update={"claim": "Pass", "scope": a.scope, "evidence": (ref(a),)}
        ),
    )
    second = first.model_copy(
        update={
            "id": "f" * 32,
            "belief": first.belief.model_copy(update={"id": "e" * 32, "claim": "Fail"}),
        }
    )
    memory = MemoryPolicy(
        evidence_validators={OWNER_ID: lambda c, s: True},
        conflict_detectors={OWNER_ID: lambda a, b, s: a.claim != b.claim},
        clock=lambda: NOW,
    )
    batch = validator(memory=memory).validate(result(first, second), episode, store.view())
    store.commit(batch)
    assert {r.epistemic_status for r in store.view().records if isinstance(r, Belief)} == {
        "disputed"
    }


@pytest.mark.parametrize("dependent_first", [False, True])
@pytest.mark.parametrize("transitive", [False, True])
def test_simultaneous_belief_revisions_reject_withdrawn_support_in_either_order(
    running, dependent_first, transitive
):
    store, episode, reads = running
    source_artifact = artifact()
    source = Belief(
        id="a" * 32,
        created_at=NOW,
        claim="Source is checked",
        scope=source_artifact.scope,
        evidence=(ref(source_artifact),),
        epistemic_status="supported",
    )
    intermediate = Belief(
        id="b" * 32,
        created_at=NOW,
        claim="Dependent is checked",
        scope=source.scope,
        evidence=(ref(source),),
        epistemic_status="supported",
    )
    dependent = (
        Belief(
            id="e" * 32,
            created_at=NOW,
            claim="Transitive dependent is checked",
            scope=source.scope,
            evidence=(ref(intermediate),),
            epistemic_status="supported",
        )
        if transitive
        else intermediate
    )
    seed = (source_artifact, source, intermediate) + ((dependent,) if transitive else ())
    store.commit(CommitBatch(events=(event(seed),)))
    before = store.view()
    withdraw = proposal(
        read_set=(*reads, ref(source)),
        belief=source.model_copy(update={"revision": 2, "evidence": ()}),
    )
    retain = proposal(
        id="f" * 32,
        read_set=(*reads, ref(dependent)),
        belief=dependent.model_copy(update={"revision": 2}),
    )
    proposals = (retain, withdraw) if dependent_first else (withdraw, retain)
    memory = MemoryPolicy(
        evidence_validators={OWNER_ID: lambda change, state: True}, clock=lambda: NOW
    )
    batch = validator(memory=memory).validate(result(*proposals), episode, before)
    assert batch.response_receipt.disposition == "stale"
    assert batch.response_receipt.reason_code == "stale_dependency"
    store.commit(batch)
    # Rejection may journal its receipt but must preserve every pre-batch record.
    current = {record.id: record for record in store.view().records}
    assert all(current[record.id] == record for record in before.records)
    assert rebuild(store.events()) == store.view()
    with Store.open_readonly(store.root) as snapshot:
        assert snapshot.view() == store.view()
