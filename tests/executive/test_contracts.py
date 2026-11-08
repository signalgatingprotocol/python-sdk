"""Versioned, immutable executive wire contracts."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from signal_gating.executive import config, messages, records
from tests.executive.factories import (
    action,
    criterion,
    frame,
    goal,
    grant,
    owner,
    packet,
    proposal,
    signal,
)


def test_contract_roundtrip_and_invalid_values():
    samples = [
        owner(),
        goal(),
        frame(),
        grant(),
        signal(),
        proposal(),
        action(),
        packet(),
        criterion(),
    ]
    for value in samples:
        assert type(value).model_validate_json(value.model_dump_json()) == value
        assert value.schema_version == 1
        with pytest.raises(ValidationError):
            type(value).model_validate({**value.model_dump(), "schema_version": 2})
        with pytest.raises(ValidationError):
            type(value).model_validate({**value.model_dump(), "schema_version": True})
        with pytest.raises(ValidationError):
            type(value).model_validate({**value.model_dump(), "surprise": "value"})
        with pytest.raises(ValidationError):
            value.schema_version = 2
    for field in (
        "max_active_episodes",
        "context_token_limit",
        "max_signal_bytes",
        "max_output_tokens",
    ):
        for invalid in (True, -1, 0, 1.5):
            with pytest.raises(ValidationError):
                config.KernelConfig(**{field: invalid})
    for invalid in (True, float("nan"), float("inf"), -1, 0):
        with pytest.raises(ValidationError):
            config.KernelConfig(episode_timeout_seconds=invalid)
    with pytest.raises(ValidationError):
        goal(created_at=datetime(2026, 1, 1))
    with pytest.raises(ValidationError):
        goal(revision=True)
    with pytest.raises(ValidationError):
        goal(id="external-id")
    with pytest.raises(ValidationError):
        records.Usage(input_tokens=-1)
    with pytest.raises(ValidationError):
        records.Limits(cost_microunits=True)
    with pytest.raises(ValidationError):
        records.Belief(claim="x", scope=records.Scope(), confidence=float("nan"))


def test_every_command_and_proposal_tag_roundtrips():
    command_values = [
        messages.CreateGoal(goal=goal()),
        messages.ReviseGoal(goal_id=goal().id, expected_revision=1, outcome="revised"),
        messages.PauseGoal(goal_id=goal().id, expected_revision=1),
        messages.ResumeGoal(goal_id=goal().id, expected_revision=1),
        messages.CancelGoal(goal_id=goal().id, expected_revision=1),
        messages.ResolveConflict(
            belief_ids=("1" * 32,),
            conflicting_revisions=(records.Ref(id="1" * 32),),
            explanation="checked",
        ),
        messages.DecideAction(
            action_id=action().id,
            expected_revision=1,
            decision="approve",
            arguments_digest="a" * 64,
            constraint_revision=1,
        ),
        messages.ResolveAction(
            action_id=action().id,
            expected_revision=1,
            outcome="succeeded",
            evidence=(records.Ref(id="1" * 32),),
            explanation="observed",
        ),
        messages.ResolveUsage(
            reservation_id="1" * 32,
            expected_revision=1,
            accept_reserved_maximum=True,
            explanation="accepted",
        ),
        messages.AttachEvidence(goal_id=goal().id, evidence=(records.Ref(id="1" * 32),)),
        messages.Reverify(goal_id=goal().id),
        messages.ShareArtifact(artifact_id="1" * 32, principal_ids=(owner().id,)),
        messages.SetGrant(grant=grant()),
    ]
    proposal_values = [
        proposal(),
        messages.TaskDecomposition(episode_id="1" * 32, fencing_token=1, tasks=(frame(),)),
        messages.ActionIntent(
            episode_id="1" * 32,
            fencing_token=1,
            capability_id="workspace.read",
            arguments={"path": "a"},
        ),
        messages.EvidenceRequest(episode_id="1" * 32, fencing_token=1, query="source"),
        messages.CompletionClaim(
            episode_id="1" * 32, fencing_token=1, criterion_ids=(criterion().id,)
        ),
    ]
    for union, values in ((messages.Command, command_values), (messages.Proposal, proposal_values)):
        adapter = TypeAdapter(union)
        for value in values:
            assert adapter.validate_json(value.model_dump_json()) == value
            for change in ({"kind": "future"}, {"schema_version": 2}, {"schema_version": True}):
                with pytest.raises(ValidationError):
                    adapter.validate_python({**value.model_dump(), **change})


def test_nested_containers_are_unaliased_immutable_and_serializable():
    payload = {"nested": [{"n": 1}]}
    value = signal(payload=payload)
    payload["nested"][0]["n"] = 2
    assert value.payload["nested"][0]["n"] == 1
    with pytest.raises(TypeError):
        value.payload["nested"][0]["n"] = 3
    assert json.loads(value.model_dump_json())["payload"] == {"nested": [{"n": 1}]}
    with pytest.raises(ValidationError):
        value.model_copy(update={"revision": True})
    with pytest.raises(ValidationError):
        signal(payload={"bad": float("nan")})
    with pytest.raises(ValidationError):
        signal(payload={"bad": object()})


def test_defaults_and_explicit_accounting_uncertainty():
    value = config.KernelConfig()
    assert (value.max_active_episodes, value.episode_timeout_seconds) == (2, 120)
    assert (value.max_output_tokens, value.context_token_limit) == (2048, 8192)
    assert (value.max_signal_bytes, value.max_artifact_bytes) == (65536, 8388608)
    assert (value.identical_failure_limit, value.verification_repair_rounds) == (2, 2)
    assert value.max_queued_ordinary_items == 10000
    assert value.bootstrap_grants == ()
    assert value.bootstrap_owner is False
    assert records.Usage().status == "unknown"
    assert records.Usage().cost_microunits is None
    assert records.Usage().input_tokens is None
    assert records.Usage().output_tokens is None
    assert goal().status == "active"
    assert records.Commitment(goal_id=goal().id, responsible_principal_id=owner().id).required
    assert records.RunSummary().completed_goal_ids == ()
    assert (
        datetime.fromisoformat(goal().model_dump()["created_at"].isoformat()).tzinfo == timezone.utc
    )


def test_usage_resolution_and_structured_errors():
    with pytest.raises(ValidationError):
        messages.ResolveUsage(reservation_id="1" * 32, expected_revision=1, explanation="free")
    with pytest.raises(ValidationError):
        messages.ResolveUsage(
            reservation_id="1" * 32,
            expected_revision=1,
            usage=records.Usage(),
            explanation="unknown",
        )
    with pytest.raises(ValidationError):
        messages.ResolveAction(
            action_id=action().id,
            expected_revision=1,
            outcome="succeeded",
            explanation="no evidence",
        )
    with pytest.raises(ValidationError):
        messages.EpisodeResult(stop_reason="blocked")
    from signal_gating.executive.errors import ContractError

    error = ContractError("unsupported_schema", "Unsupported schema", record_id="1" * 32)
    assert error.code == "unsupported_schema"
    assert error.record_id == "1" * 32


def test_all_persistent_tags_and_supporting_dtos_roundtrip():
    ref = records.Ref(id="1" * 32)
    scope = records.Scope(owner_id=owner().id, goal_id=goal().id)
    limits = goal().limits
    usage = records.Usage(
        status="known",
        input_tokens=0,
        output_tokens=0,
        calls=0,
        elapsed_seconds=0,
        cost_microunits=0,
    )
    commitment = records.Commitment(goal_id=goal().id, responsible_principal_id=owner().id)
    constraint = records.Constraint(description="Stay in workspace")
    budget = records.Budget(scope=scope, limits=limits)
    reservation = records.Reservation(budget_id=budget.id, subject_id=frame().id, maximum=limits)
    artifact = records.ArtifactRef(
        digest="a" * 64,
        storage_location="blobs/a",
        media_type="text/plain",
        origin="fixture",
        scope=scope,
        size_bytes=1,
    )
    receipt = records.DecisionReceipt(subject_id=goal().id, disposition="accept", reason_code="ok")
    episode = records.Episode(
        task_id=frame().id,
        fencing_token=1,
        context_packet_id=packet().id,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        deadline=goal().created_at,
        allowed_proposal_types=("belief_change",),
    )
    verification = records.VerificationResult(
        criterion_id=criterion().id,
        verifier_id="check",
        disposition="accept",
        reason_code="checked",
    )
    event = records.JournalEvent(
        sequence=1,
        event_type="goal_created",
        producer_id="local",
        principal_id=owner().id,
        subject=ref,
    )
    persistent = [
        owner(),
        goal(),
        commitment,
        proposal().belief,
        frame(),
        grant(),
        budget,
        reservation,
        artifact,
        signal(),
        receipt,
        action(),
        episode,
        constraint,
        criterion(),
        verification,
        event,
        packet(),
    ]
    adapter = TypeAdapter(records.PersistentRecord)
    assert {item.kind for item in persistent} == {
        "principal",
        "goal",
        "commitment",
        "belief",
        "task",
        "grant",
        "budget",
        "reservation",
        "artifact",
        "signal",
        "receipt",
        "action",
        "episode",
        "constraint",
        "criterion",
        "verification",
        "event",
        "context_packet",
    }
    state = records.StateView(sequence=1, records=tuple(persistent))
    summary = records.RunSummary(usage=usage)
    evaluation = records.EvaluationCase(name="case", initial_state=state)
    result = records.EvaluationResult(
        case_id=evaluation.id, policy_version="deterministic-v1", passed=True, summary=summary
    )
    supporting = [
        ref,
        scope,
        limits,
        usage,
        records.WakeCondition(kind="time", at=goal().created_at, reason_code="timer"),
        records.PolicyDecision(
            gate="attention",
            disposition="defer",
            reason_code="wait",
            wake_condition=records.WakeCondition(kind="human_decision", reason_code="review"),
        ),
        records.CapabilitySpec(
            id="read",
            binding_version="1",
            input_schema={"type": "object"},
            effect_class="read_only",
        ),
        records.ArtifactDraft(key="a", media_type="text/plain", text="hello"),
        records.ActionObservation(action_id=action().id, expected_revision=1, outcome="succeeded"),
        records.CriterionGap(
            criterion_id=criterion().id, reason_code="missing", description="Need file"
        ),
        records.ContextManifest(),
        packet(),
        state,
        summary,
        records.KernelView(state=state),
        evaluation,
        result,
        records.EvaluationReport(results=(result,)),
        messages.CommitBatch(events=(event,), records=tuple(persistent), receipts=(receipt,)),
        messages.AuthenticatedCommand(command=messages.CreateGoal(goal=goal()), principal=owner()),
        messages.EpisodeResult(proposals=(proposal(),), stop_reason="proposed"),
        config.SchedulerConfig(),
        config.KernelConfig(),
        config.WorkerConfig(id="fixture", kind="fixture"),
        config.PriceConfig(
            version="1",
            input_microunits_per_million_tokens=0,
            output_microunits_per_million_tokens=0,
        ),
        config.ArgvCheck(id="check", argv=("python", "-m", "pytest")),
        config.WorkspaceConfig(root="/tmp/workspace"),
    ]
    for item in persistent:
        assert adapter.validate_json(item.model_dump_json()) == item
    for item in persistent + supporting:
        assert type(item).model_validate_json(item.model_dump_json()) == item
        with pytest.raises(ValidationError):
            type(item).model_validate({**item.model_dump(), "schema_version": 2})
        with pytest.raises(ValidationError):
            type(item).model_validate({**item.model_dump(), "schema_version": True})
        with pytest.raises(ValidationError):
            type(item).model_validate({**item.model_dump(), "unexpected": "value"})
        with pytest.raises(ValidationError):
            item.schema_version = 2


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "calls", "cost_microunits"])
@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_usage_rejects_non_integer_or_negative_quantities(field, value):
    with pytest.raises(ValidationError):
        records.Usage(**{field: value})


def test_invalid_time_and_context_and_draft_boundaries():
    with pytest.raises(ValidationError):
        records.Usage(status="known")
    with pytest.raises(ValidationError):
        records.WakeCondition(kind="time", reason_code="timer")
    with pytest.raises(ValidationError):
        records.ArtifactDraft(key="a", media_type="text/plain", text="\ud800")
    assert records.ArtifactDraft(key="a", media_type="text/plain", text="é").text == "é"
    with pytest.raises(ValidationError):
        messages.EpisodeResult(stop_reason="needs_evidence")
    with pytest.raises(ValidationError):
        config.WorkerConfig(id="real", kind="openai")
    with pytest.raises(ValidationError):
        config.KernelConfig(bootstrap_owner=True)
    with pytest.raises(ValidationError):
        config.KernelConfig(context_token_limit=2048)
    assert (
        goal(created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)).created_at.tzinfo == timezone.utc
    )


def test_refs_and_policy_dispositions_are_typed():
    with pytest.raises(ValidationError):
        records.Ref(id="unscoped-external-id")
    assert records.Ref(id="draft:report").id == "draft:report"
    with pytest.raises(ValidationError):
        records.Ref(id="draft:../escape")
    with pytest.raises(ValidationError):
        records.PolicyDecision(gate="execution", disposition="archive", reason_code="wrong_gate")
    with pytest.raises(ValidationError):
        records.PolicyDecision(gate="attention", disposition="defer", reason_code="no_wake")


def test_new_budget_has_known_zero_balances_and_consistent_currency():
    budget = records.Budget(scope=records.Scope(), limits=records.Limits(currency="EUR"))
    assert budget.reserved.status == budget.actual.status == "known"
    assert budget.reserved.cost_microunits == budget.actual.cost_microunits == 0
    assert budget.reserved.currency == budget.actual.currency == "EUR"
    with pytest.raises(ValidationError):
        records.Budget(
            scope=records.Scope(),
            limits=records.Limits(currency="EUR"),
            actual=records.Usage(currency="USD"),
        )
