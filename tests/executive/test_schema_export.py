"""Experimental full DTO schemas and redacted local observations."""

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from signal_gating.executive import (
    Budget,
    CommitBatch,
    Commitment,
    ContextManifest,
    DecisionReceipt,
    Episode,
    EpisodeResult,
    ExecutiveSignal,
    Grant,
    JournalEvent,
    PersistentRecord,
    Ref,
    Reservation,
    StoreError,
    Usage,
    messages,
    records,
)
from signal_gating.executive.actions import CapabilityRegistry, Dispatcher
from signal_gating.executive.artifacts import AccessDenied, ArtifactStore
from signal_gating.executive.authority import Authority
from signal_gating.executive.budgets import BudgetPolicy
from signal_gating.executive.inspection import export_contract_schemas, export_goal
from signal_gating.executive.intake import Intake
from signal_gating.executive.proposals import ProposalValidator
from signal_gating.executive.store import Store
from signal_gating.executive.workers import EpisodeRunner, WorkerError

from .factories import (
    COMMITMENT_ID,
    GOAL_ID,
    NOW,
    OWNER_ID,
    action,
    frame,
    goal,
    grant,
    owner,
    packet,
)


def write(store, *values):
    current = {r.id: r for r in store.view().records}
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    created_at=NOW,
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=values[0].id),
                    payload={"records": [r.model_dump(mode="json") for r in values]},
                ),
            ),
            expected_revisions=tuple(
                Ref(id=r.id, revision=current[r.id].revision) for r in values if r.id in current
            ),
        )
    )


def test_full_contract_schemas_are_json_portable_closed_and_provider_independent():
    exported = json.loads(json.dumps(export_contract_schemas()))
    assert exported["schema_version"] == 1
    assert exported["experimental"] is True
    schemas = exported["schemas"]
    for module in (records, messages):
        for name, cls in vars(module).items():
            if (
                isinstance(cls, type)
                and issubclass(cls, records.Model)
                and cls.__module__ == module.__name__
            ):
                assert schemas[name] == cls.model_json_schema() | {
                    "$schema": "https://json-schema.org/draft/2020-12/schema"
                }
    assert "arguments" in schemas["ActionRecord"]["properties"]
    assert "observed_refs" in schemas["ContextManifest"]["properties"]
    assert schemas["EpisodeResult"]["additionalProperties"] is False
    assert "oneOf" in schemas["Command"]
    assert "oneOf" in schemas["Proposal"]
    assert schemas["Goal"]["properties"]["schema_version"]["const"] == 1
    code = (
        "from signal_gating.executive import export_contract_schemas; import sys; "
        "export_contract_schemas(); assert 'openai' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def check_refs(value, root):
        if isinstance(value, dict):
            if "$ref" in value:
                target = root
                for key in value["$ref"].removeprefix("#/").split("/"):
                    target = target[key]
            for item in value.values():
                check_refs(item, root)
        elif isinstance(value, list):
            for item in value:
                check_refs(item, root)

    for schema in schemas.values():
        check_refs(schema, schema)


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parent / "fixtures/contracts").glob("*.json"))
)
def test_actual_contract_vectors_roundtrip_reject_future_versions(path):
    vector = json.loads(path.read_text())
    contract = getattr(records, vector["contract"], None) or getattr(messages, vector["contract"])
    adapter = TypeAdapter(contract)
    value = adapter.validate_python(vector["value"])
    assert json.loads(adapter.dump_json(value)) == vector["value"]
    with pytest.raises(ValidationError):
        adapter.validate_python(vector["value"] | {"schema_version": 2})
    assert vector["case"] in {"duplicate", "stale_proposal", "interrupted", "unknown_effect"}


def test_goal_export_scopes_history_receipts_bindings_usage_and_redacts(tmp_path):
    with Store.open(tmp_path / "store") as store:
        worker = owner(id="e" * 32, roles=("worker",))
        other = goal(id="f" * 32, outcome="OTHER-GOAL-PRIVATE", criteria=())
        write(
            store,
            owner(),
            worker,
            goal(outcome="PRIVATE-PROMPT"),
            other,
            Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID),
            frame(),
            grant(),
            Budget(
                id="9" * 32,
                scope=records.Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
                limits=goal().limits,
                accounting_uncertain=True,
                actual=Usage(calls=1, source="interrupted-invocation"),
            ),
            Reservation(
                id="a" * 32,
                budget_id="9" * 32,
                subject_id=action().id,
                maximum=goal().limits,
                status="uncertain",
                usage=Usage(calls=1, source="interrupted-invocation"),
            ),
            action(
                status="outcome_unknown",
                arguments={"credentials": "SECRET", "path": "/private/machine"},
                reservation_id="a" * 32,
            ),
        )
        artifacts = ArtifactStore(store)
        evidence = artifacts.put(
            b"PRIVATE-EVIDENCE", principal=owner(), goal_id=GOAL_ID, media_type="text/plain"
        )
        evidence = evidence.model_copy(update={"revision": 2, "shared_with": (worker.id,)})
        write(store, evidence)
        assert artifacts.read(evidence, principal=worker, goal_id=GOAL_ID) == b"PRIVATE-EVIDENCE"
        private = artifacts.put(
            b"RAW-PRIVATE-CONTEXT",
            principal=owner(),
            goal_id=GOAL_ID,
            media_type="application/vnd.sgp.executive-context+json",
        )
        other_artifact = artifacts.put(
            b"OTHER", principal=owner(), goal_id=other.id, media_type="text/plain"
        )
        receipt = DecisionReceipt(
            subject_id=action().id,
            disposition="stale",
            reason_code="stale_dependency",
            state_revisions=(Ref(id=goal().id),),
            evidence=(Ref(id=evidence.id), Ref(id=private.id), Ref(id=other_artifact.id)),
        )
        context = packet(
            rendered_input="PRIVATE-PROMPT",
            worker_instructions="PRIVATE-INSTRUCTIONS",
            rendered_artifact=private,
            manifest=ContextManifest(
                observed_refs=(Ref(id=receipt.id),),
                included=(Ref(id=evidence.id), Ref(id=private.id)),
            ),
        )
        write(store, receipt, context)
        before = store.view()
        exported = export_goal(store, GOAL_ID, principal=owner())
        assert store.view() == before
        assert exported["schema_version"] == 1
        assert exported["authority"] == "observation_only"
        assert exported["goal_id"] == GOAL_ID
        assert exported["principal_id"] == owner().id
        assert exported["policy_versions"] == ["deterministic-v1"]
        text = json.dumps(exported)
        for forbidden in (
            "credentials",
            "SECRET",
            "/private/machine",
            "PRIVATE-PROMPT",
            "PRIVATE-INSTRUCTIONS",
            "RAW-PRIVATE-CONTEXT",
            "PRIVATE-EVIDENCE",
            "OTHER-GOAL-PRIVATE",
            private.id,
            other_artifact.id,
            str(store.root),
            "storage_location",
            "idempotency_key",
        ):
            assert forbidden not in text
        projected = {r["id"]: r for r in exported["records"]}
        assert projected[action().id]["status"] == "outcome_unknown"
        assert projected[action().id]["binding_version"] == "1"
        assert projected["9" * 32]["actual"]["status"] == "unknown"
        assert projected["9" * 32]["actual"]["cost_microunits"] is None
        assert projected["a" * 32]["usage"]["source"] == "interrupted-invocation"
        assert projected["a" * 32]["status"] == "uncertain"
        assert projected[context.id]["manifest"]["observed_refs"] == [
            Ref(id=receipt.id).model_dump(mode="json")
        ]
        assert projected[receipt.id]["evidence"] == [Ref(id=evidence.id).model_dump(mode="json")]
        assert projected[grant().id]["authority"] == "observation_only"
        with pytest.raises(ValidationError):
            Grant.model_validate(projected[grant().id])
        assert exported["artifacts"][0]["id"] == evidence.id
        assert exported["artifacts"][0]["integrity"] == "verified"
        assert exported["history"]
        assert all(event["authority"] == "observation_only" for event in exported["history"])
        with pytest.raises(AccessDenied):
            export_goal(store, GOAL_ID, principal=worker)
        with pytest.raises(AccessDenied):
            export_goal(store, GOAL_ID, principal=owner(authenticated_origin="forged"))
        with pytest.raises(AccessDenied):
            export_goal(store, "0" * 32, principal=owner())
        assert (
            TypeAdapter(PersistentRecord).validate_python(action().model_dump(mode="json"))
            == action()
        )


def test_export_uses_current_snapshot_and_integrity_even_with_old_reader(tmp_path):
    with Store.open(tmp_path / "store") as store:
        write(store, owner(), goal())
        artifacts = ArtifactStore(store)
        evidence = artifacts.put(
            b"proof", principal=owner(), goal_id=GOAL_ID, media_type="text/plain"
        )
        with Store.open_readonly(store.root) as stale:
            artifacts.delete(evidence, principal=owner())
            assert export_goal(stale, GOAL_ID, principal=owner())["artifacts"] == []
        current = artifacts.put(
            b"proof2", principal=owner(), goal_id=GOAL_ID, media_type="text/plain"
        )
        (store.root / "artifacts" / current.digest).write_bytes(b"broken")
        with pytest.raises(StoreError) as caught:
            export_goal(store, GOAL_ID, principal=owner())
        assert caught.value.code == "artifact_corrupt"


def load_vector(name):
    vector = json.loads((Path(__file__).parent / f"fixtures/contracts/{name}.json").read_text())
    cls = getattr(records, vector["contract"], None) or getattr(messages, vector["contract"])
    return cls.model_validate(vector["value"])


async def test_vectors_exercise_duplicate_stale_interrupted_and_unknown_boundaries(tmp_path):
    """These vectors are local contract behavior, not remote enforcement proof."""
    with Store.open(tmp_path / "store") as store:
        task = frame(status="running", fencing_token=1)
        context = packet(task=task)
        episode = Episode(
            id=context.episode_id,
            created_at=NOW,
            task_id=task.id,
            fencing_token=1,
            context_packet_id=context.id,
            worker_id="fixture",
            provider="fixture",
            model="fixture",
            allowed_proposal_types=("belief_change",),
            deadline=NOW + timedelta(minutes=5),
        )
        write(
            store,
            owner(),
            goal(),
            Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID),
            task,
            context,
            episode,
        )
        duplicate = load_vector("duplicate")
        first = Intake(store).accept(duplicate, principal=owner(), state=store.view())
        store.commit(first)
        again = Intake(store).accept(duplicate, principal=owner(), state=store.view())
        assert again.response_receipt == first.response_receipt and not again.events

        stale = ProposalValidator(clock=lambda: NOW).validate(
            EpisodeResult(proposals=(load_vector("stale_proposal"),), stop_reason="proposed"),
            episode,
            store.view(),
        )
        assert stale.response_receipt.disposition == "stale"

        class NeverCalled:
            async def run_episode(self, context):
                pytest.fail("Fenced episode must not invoke a worker")

        interrupted = load_vector("interrupted").model_copy(
            update={"deadline": NOW + timedelta(minutes=5)}
        )
        runner = EpisodeRunner(is_current=lambda *args: True, clock=lambda: NOW)
        with pytest.raises(WorkerError) as caught:
            await runner.run(interrupted, context, NeverCalled())
        assert caught.value.code == "episode_fenced"
        assert interrupted.usage.status == "unknown" and interrupted.usage.calls == 1
        await runner.close()

        unknown = load_vector("unknown_effect")
        write(store, unknown)
        dispatcher = Dispatcher(
            store,
            authority=Authority(),
            budgets=BudgetPolicy(),
            registry=CapabilityRegistry(),
            clock=lambda: NOW,
        )
        receipt = await dispatcher.dispatch_one(unknown.id)
        assert receipt.reason_code == "action_not_dispatchable"
        assert next(r for r in store.view().records if r.id == unknown.id).dispatch_attempts == 1
        await dispatcher.close()


def test_export_preserves_scoped_causal_and_historical_policy_links(tmp_path):
    with Store.open(tmp_path / "store") as store:
        lineage = "d" * 32
        signal = ExecutiveSignal(
            signal_kind="observation",
            principal_id=OWNER_ID,
            producer_id="fixture",
            causal_ids=(lineage,),
            scope=records.Scope(goal_id=GOAL_ID),
        )
        receipt = DecisionReceipt(
            subject_id=signal.id,
            disposition="admit",
            reason_code="fixture",
            policy_version="historical-v1",
        )
        write(store, owner(), goal(), signal, receipt)
        write(store, receipt.model_copy(update={"revision": 2, "policy_version": "current-v2"}))
        exported = export_goal(store, GOAL_ID, principal=owner())
        assert exported["policy_versions"] == ["current-v2", "historical-v1"]
        row = next(r for r in exported["records"] if r["id"] == signal.id)
        assert row["causal_ids"] == [lineage]
        versions = [
            image["revision"]
            for event in exported["history"]
            for image in event["records"]
            if image["id"] == receipt.id
        ]
        assert versions == [1, 2]


def test_cross_goal_receipt_does_not_inherit_scope_from_shared_evidence(tmp_path):
    with Store.open(tmp_path / "store") as store:
        foreign_goal = goal(id="f" * 32, criteria=())
        receipt = DecisionReceipt(
            subject_id=foreign_goal.id,
            disposition="deny",
            reason_code="PRIVATE-OTHER-GOAL-STATUS",
            evidence=(Ref(id=GOAL_ID),),
        )
        write(store, owner(), goal(), foreign_goal, receipt)
        exported = export_goal(store, GOAL_ID, principal=owner())
        assert receipt.id not in json.dumps(exported)
        assert foreign_goal.id not in json.dumps(exported)


def test_private_and_cross_goal_scalar_references_are_omitted_everywhere(tmp_path):
    with Store.open(tmp_path / "store") as store:
        foreign_goal = goal(id="f" * 32, criteria=())
        foreign_grant = grant(id="e" * 32, scope=records.Scope(goal_id=foreign_goal.id))
        write(
            store,
            owner(),
            goal(parent_id=foreign_goal.id),
            foreign_goal,
            foreign_grant,
            Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID),
            frame(),
            action(grant_id=foreign_grant.id),
        )
        private = ArtifactStore(store).put(
            b"private context",
            principal=owner(),
            goal_id=GOAL_ID,
            media_type="application/vnd.sgp.executive-context+json",
        )
        receipt = DecisionReceipt(subject_id=private.id, disposition="admit", reason_code="fixture")
        external = DecisionReceipt(
            subject_id="external-subject",
            disposition="admit",
            reason_code="fixture",
            evidence=(Ref(id=GOAL_ID),),
        )
        write(store, receipt, external)
        exported = export_goal(store, GOAL_ID, principal=owner())
        serialized = json.dumps(exported)
        for hidden in (private.id, foreign_goal.id, foreign_grant.id):
            assert hidden not in serialized
        logical = {record["id"]: record for record in exported["records"]}
        assert "subject_id" not in logical[receipt.id]
        assert "parent_id" not in logical[GOAL_ID]
        assert "grant_id" not in logical[action().id]
        assert logical[external.id]["subject_id"] == "external-subject"
        historical = [image for event in exported["history"] for image in event["records"]]
        assert any(image["id"] == receipt.id for image in historical)
        assert all("subject_id" not in image for image in historical if image["id"] == receipt.id)


@pytest.mark.parametrize("mismatch", [{"revision": 1}, {"roles": ("owner",)}, {"name": "Altered"}])
def test_export_requires_complete_current_principal_not_partial_identity(tmp_path, mismatch):
    with Store.open(tmp_path / "store") as store:
        write(store, owner(), goal())
        current = owner(revision=2, name="Current owner", roles=("operator",))
        write(store, current)
        assert export_goal(store, GOAL_ID, principal=current)["principal_id"] == OWNER_ID
        with pytest.raises(AccessDenied) as mismatch_error:
            export_goal(store, GOAL_ID, principal=current.model_copy(update=mismatch))
        with pytest.raises(AccessDenied) as origin_error:
            export_goal(
                store,
                GOAL_ID,
                principal=current.model_copy(update={"authenticated_origin": "forged"}),
            )
        assert mismatch_error.value.code == origin_error.value.code == "export_access_denied"
        assert str(mismatch_error.value) == str(origin_error.value) == "Goal export access denied"
