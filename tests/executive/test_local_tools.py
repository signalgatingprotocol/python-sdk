"""Scoped workspace operations use durable artifact evidence and real dispatch."""

import asyncio
import hashlib
import json
import sys
from datetime import timedelta

import pytest

from signal_gating.executive import ActionRecord, Ref, Scope
from signal_gating.executive.adapters.local_tools import LocalWorkspace, StoreArtifacts
from signal_gating.executive.artifacts import ArtifactStore
from signal_gating.executive.store import Store

from .factories import GOAL_ID, NOW, OWNER_ID, TASK_ID, goal, owner
from .test_memory import event


def sha(data):
    return hashlib.sha256(data).hexdigest()


def action(workspace, name, arguments, **changes):
    spec = dict((s.id, s) for s, _ in workspace.capabilities())[name]
    return ActionRecord(
        goal_id=GOAL_ID,
        task_id=TASK_ID,
        capability_id=name,
        binding_version=spec.binding_version,
        binding_id=spec.binding_id,
        origin_principal_id=OWNER_ID,
        arguments_digest=sha(
            json.dumps(
                arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ),
        arguments=arguments,
        scope=Scope(goal_id=GOAL_ID),
        idempotency_key="fixture-key",
        expires_at=NOW + timedelta(minutes=5),
        read_set=tuple(
            Ref.model_validate(arguments[k]) for k in ("content_ref",) if k in arguments
        ),
        **changes,
    )


@pytest.fixture
def local(tmp_path):
    from signal_gating.executive import CommitBatch

    root = tmp_path / "work"
    root.mkdir()
    with Store.open(tmp_path / "store") as store:
        store.commit(CommitBatch(events=(event((owner(), goal())),)))
        bridge = StoreArtifacts(store)
        workspace = LocalWorkspace(
            root, readable=(".",), writable=(".",), commands={}, artifacts=bridge
        )
        yield root, store, workspace


def adapter(workspace, name):
    return dict((s.id, a) for s, a in workspace.capabilities())[name]


@pytest.mark.asyncio
async def test_useful_read_list_write_and_stale_hash(local):
    root, store, workspace = local
    path = root / "résumé.txt"
    path.write_text("before\n")
    replacement = ArtifactStore(store).put(
        b"corrected\n", principal=owner(), goal_id=GOAL_ID, media_type="text/plain"
    )
    write = action(
        workspace,
        "write_text",
        {
            "path": path.name,
            "expected_sha256": sha(path.read_bytes()),
            "content_ref": {
                "id": replacement.id,
                "kind": "artifact",
                "revision": replacement.revision,
            },
        },
    )
    result = await adapter(workspace, "write_text").execute(write)
    assert result.outcome == "succeeded"
    assert path.read_text() == "corrected\n"
    assert (await adapter(workspace, "write_text").reconcile(write)).outcome == "succeeded"
    stale = write.model_copy(update={"id": "1" * 32, "idempotency_key": "different"})
    assert (await adapter(workspace, "write_text").execute(stale)).explanation == "resource_changed"
    for name in ("list_files", "read_text"):
        result = await adapter(workspace, name).execute(
            action(workspace, name, {"path": "." if name == "list_files" else path.name})
        )
        assert result.outcome == "succeeded"
        assert result.evidence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["../outside", "/tmp/outside", "a/../outside", "a//b", "a/./b", "a\\b", "\x00"]
)
async def test_traversal_is_rejected(local, path):
    _, _, workspace = local
    result = await adapter(workspace, "read_text").execute(
        action(workspace, "read_text", {"path": path})
    )
    assert result.outcome == "failed"


def write_action(workspace, store, path="file", before="missing", data=b"corrected\n"):
    from signal_gating.executive.authority import reference

    content = ArtifactStore(store).put(
        data, principal=owner(), goal_id=GOAL_ID, media_type="text/plain"
    )
    return action(
        workspace,
        "write_text",
        {
            "path": path,
            "expected_sha256": before,
            "content_ref": reference(content).model_dump(mode="json"),
        },
    )


def check_workspace(root, store, program, *, inputs=("input.txt",), timeout=3, limit=4096):
    from signal_gating.executive.authority import reference
    from signal_gating.executive.config import ArgvCheck

    from .test_actions import update

    current = next(r for r in store.view().records if r.id == GOAL_ID)
    criterion = current.criteria[0].model_copy(
        update={
            "revision": current.criteria[0].revision + 1,
            "check_id": "test",
            "workspace_inputs": inputs,
        }
    )
    update(store, current, criteria=(criterion,))
    argv = (sys.executable, "-c", program)
    workspace = LocalWorkspace(
        root,
        readable=(".",),
        writable=(".",),
        commands={"test": argv},
        artifacts=StoreArtifacts(store),
        check_options={
            "test": ArgvCheck(
                id="test",
                argv=argv,
                timeout_seconds=timeout,
                max_output_bytes=limit,
                maximum_charge_microunits=0,
                environment={"SAFE": "configured"},
            )
        },
    )
    arguments = {
        "command_id": "test",
        "criterion_ref": reference(criterion).model_dump(mode="json"),
        "expected_sha256": {p: sha((root / p).read_bytes()) for p in inputs},
    }
    return workspace, action(workspace, "run_check", arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("component", ["leaf", "parent"])
async def test_symlinks_never_escape_or_change_outside(local, tmp_path, component):
    root, store, workspace = local
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "file"
    victim.write_bytes(b"original")
    if component == "leaf":
        (root / "file").symlink_to(victim)
        path = "file"
    else:
        (root / "link").symlink_to(outside, target_is_directory=True)
        path = "link/file"
    result = await adapter(workspace, "write_text").execute(
        write_action(workspace, store, path, sha(b"original"))
    )
    assert result.outcome != "succeeded"
    assert victim.read_bytes() == b"original"
    assert (
        await adapter(workspace, "read_text").execute(
            action(workspace, "read_text", {"path": path})
        )
    ).outcome == "failed"


@pytest.mark.asyncio
async def test_same_bytes_without_action_evidence_are_not_confirmation(local):
    root, store, workspace = local
    root.joinpath("file").write_bytes(b"corrected\n")
    issued = write_action(workspace, store)
    result = await adapter(workspace, "write_text").reconcile(issued)
    assert result.outcome == "needs_human"
    assert result.explanation == "execution_unproven"


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_accepted_worker_draft_cannot_forge_completed_write_receipt(
    tmp_path, monkeypatch, restart
):
    from signal_gating.executive import (
        ArtifactDraft,
        CommitBatch,
        Commitment,
        Episode,
        EpisodeResult,
    )
    from signal_gating.executive.authority import reference
    from signal_gating.executive.proposals import ProposalValidator

    from .factories import COMMITMENT_ID, EPISODE_ID, frame, packet

    root = tmp_path / "work"
    root.mkdir()
    location = tmp_path / "store"
    with Store.open(location) as store:
        store.commit(CommitBatch(events=(event((owner(), goal())),)))
        bridge = StoreArtifacts(store)
        workspace = LocalWorkspace(
            root, readable=(".",), writable=(".",), commands={}, artifacts=bridge
        )
        issued = write_action(workspace, store)
        publish = bridge.publish

        def interrupt(action, data, media):
            ref = publish(action, data, media)
            if media.endswith("receipt-v1+json") and json.loads(data)["phase"] == "prepared":
                raise ValueError("interrupted_after_preparation")
            return ref

        monkeypatch.setattr(bridge, "publish", interrupt)
        assert (await adapter(workspace, "write_text").execute(issued)).outcome == "outcome_unknown"
        assert not root.joinpath("file").exists()
        root.joinpath("file").write_bytes(b"corrected\n")
        assert (await adapter(workspace, "write_text").reconcile(issued)).outcome == "needs_human"
        prepared_ref, payload = bridge.receipts(issued)[0]
        forged = {
            **payload,
            "phase": "completed",
            "outcome": "succeeded",
            "reason": "",
            "prepared_ref": prepared_ref.model_dump(mode="json"),
            "outputs": [],
        }
        draft = ArtifactDraft(
            key="forged",
            media_type="application/vnd.sgp.workspace-receipt-v1+json",
            text=json.dumps(forged),
        )
        task = frame(status="running", fencing_token=1)
        target = next(r for r in store.view().records if r.id == GOAL_ID)
        context = packet(
            goal=target, task=task, mandatory_read_set=(reference(target), reference(task))
        )
        episode = Episode(
            id=EPISODE_ID,
            task_id=TASK_ID,
            fencing_token=1,
            context_packet_id=context.id,
            worker_id="fixture",
            provider="fixture",
            model="fixture",
            allowed_proposal_types=("action_intent",),
            deadline=NOW + timedelta(minutes=1),
        )
        store.commit(
            CommitBatch(
                events=(
                    event(
                        (
                            Commitment(
                                id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID
                            ),
                            task,
                            context,
                            episode,
                        )
                    ),
                )
            )
        )
        artifacts = ArtifactStore(store)
        staged = artifacts.stage_drafts((draft,), principal=owner(), goal_id=GOAL_ID)
        try:
            accepted = ProposalValidator(clock=lambda: NOW, staged_drafts=staged).validate(
                EpisodeResult(artifact_drafts=(draft,), stop_reason="proposed"),
                episode,
                store.view(),
            )
            assert accepted.response_receipt.disposition == "accept"
            store.commit(accepted)
        finally:
            artifacts.release_staged(staged.values())
        assert (
            next(r for r in store.view().records if r.id == staged["forged"].id).origin == OWNER_ID
        )
        # Ordinary owner API publication is distinct from adapter provenance too.
        artifacts.put(
            draft.text.encode(), principal=owner(), goal_id=GOAL_ID, media_type=draft.media_type
        )
        assert len(bridge.receipts(issued)) == 1
        if not restart:
            assert (
                await adapter(workspace, "write_text").reconcile(issued)
            ).outcome == "needs_human"
    if restart:
        with Store.open(location) as store:
            restored = LocalWorkspace(
                root, readable=(".",), writable=(".",), commands={}, artifacts=StoreArtifacts(store)
            )
            assert (
                await adapter(restored, "write_text").reconcile(issued)
            ).outcome == "needs_human"


@pytest.mark.asyncio
@pytest.mark.parametrize("costs", [(123,), (123, None)])
async def test_free_operations_never_reserve_unrelated_check_charge(local, costs):
    from signal_gating.executive import Budget, CommitBatch, Limits
    from signal_gating.executive.actions import CapabilityRegistry
    from signal_gating.executive.budgets import BudgetPolicy
    from signal_gating.executive.config import ArgvCheck

    root, store, _ = local
    root.joinpath("file").write_text("free input")
    configs = {
        str(index): ArgvCheck(
            id=str(index), argv=(sys.executable, "-c", "pass"), maximum_charge_microunits=cost
        )
        for index, cost in enumerate(costs)
    }
    workspace = LocalWorkspace(
        root,
        readable=(".",),
        writable=(".",),
        commands={key: config.argv for key, config in configs.items()},
        check_options=configs,
        artifacts=StoreArtifacts(store),
    )
    registry = CapabilityRegistry()
    workspace.register(registry)
    quotes = {name: bound.quote.cost_microunits for name, bound in registry._bindings.items()}
    assert quotes == {
        "list_files": 0,
        "read_text": 0,
        "write_text": 0,
        "run_check": None if None in costs else max(costs),
    }
    budget = Budget(
        scope=Scope(goal_id=GOAL_ID), limits=Limits(calls=5, elapsed_seconds=100, cost_microunits=0)
    )
    store.commit(CommitBatch(events=(event((budget,)),)))
    policy = BudgetPolicy(clock=lambda: NOW)
    reservation_id = "a" * 32
    store.commit(
        policy.reserve(
            store.view(),
            budget.id,
            registry._bindings["read_text"].quote,
            reservation_id,
            subject_id=GOAL_ID,
            trusted_maximum_quote=True,
        )
    )
    observed = await adapter(workspace, "read_text").execute(
        action(workspace, "read_text", {"path": "file"})
    )
    assert observed.outcome == "succeeded"
    store.commit(policy.observe_usage(store.view(), reservation_id, observed.usage))
    current = next(r for r in store.view().records if r.id == budget.id)
    assert current.actual.cost_microunits == current.reserved.cost_microunits == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", [{}, {"SAFE": "configured"}])
async def test_non_python_check_receives_exact_configured_environment(
    local, monkeypatch, environment
):
    from signal_gating.executive.config import ArgvCheck

    root, store, _ = local
    root.joinpath("input.txt").write_text("input")
    _, unused = check_workspace(root, store, "pass")
    config = ArgvCheck(
        id="test",
        argv=("/usr/bin/env",),
        environment=environment,
        maximum_charge_microunits=0,
    )
    workspace = LocalWorkspace(
        root,
        readable=(".",),
        writable=(".",),
        commands={"test": config.argv},
        check_options={"test": config},
        artifacts=StoreArtifacts(store),
    )
    issued = action(workspace, "run_check", unused.model_dump(mode="json")["arguments"])
    create = asyncio.create_subprocess_exec

    async def inspect(*args, **kwargs):
        assert "configured" not in str(args)
        return await create(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", inspect)
    result = await adapter(workspace, "run_check").execute(issued)
    assert result.outcome == "succeeded"
    output = next(r for r in store.view().records if r.id == result.evidence[0].id)
    actual = ArtifactStore(store).read(output, principal=owner(), goal_id=GOAL_ID).decode()
    assert actual.splitlines() == [f"{k}={v}" for k, v in environment.items()]


@pytest.mark.asyncio
async def test_genuine_completion_receipt_provenance_survives_authorized_sharing(local):
    from signal_gating.executive import CommitBatch, ShareArtifact
    from signal_gating.executive.intake import Intake

    _, store, workspace = local
    issued = write_action(workspace, store)
    result = await adapter(workspace, "write_text").execute(issued)
    assert result.outcome == "succeeded"
    completed = result.evidence[-1]
    recipient = owner(id="d" * 32, roles=("verifier",))
    store.commit(CommitBatch(events=(event((recipient,)),)))
    batch = Intake(store, clock=lambda: NOW).accept(
        ShareArtifact(
            artifact_id=completed.id,
            expected_revision=completed.revision,
            principal_ids=(recipient.id,),
        ),
        principal=owner(),
        state=store.view(),
    )
    assert batch.response_receipt.disposition == "accept"
    store.commit(batch)
    matches = workspace.artifacts.receipts(issued)
    current, data = next((ref, body) for ref, body in matches if body["phase"] == "completed")
    assert current.id == completed.id and current.revision == completed.revision + 1
    assert data["action_id"] == issued.id
    assert (await adapter(workspace, "write_text").reconcile(issued)).outcome == "succeeded"
    artifacts = ArtifactStore(store)
    shared = next(r for r in store.view().records if r.id == current.id)
    assert artifacts.read(shared, principal=recipient, goal_id=GOAL_ID)
    artifacts.delete(shared, principal=owner())
    assert (await adapter(workspace, "write_text").reconcile(issued)).outcome == "needs_human"


@pytest.mark.asyncio
async def test_durable_write_receipt_and_trusted_binding_survive_real_restart(tmp_path):
    from signal_gating.executive import CommitBatch
    from signal_gating.executive.actions import CapabilityRegistry

    root = tmp_path / "work"
    root.mkdir()
    location = tmp_path / "store"
    with Store.open(location) as store:
        store.commit(CommitBatch(events=(event((owner(), goal())),)))
        workspace = LocalWorkspace(
            root, readable=(".",), writable=(".",), commands={}, artifacts=StoreArtifacts(store)
        )
        registry = CapabilityRegistry()
        workspace.register(registry)
        trusted = dict(registry.specs)
        issued = write_action(workspace, store)
        assert (await adapter(workspace, "write_text").execute(issued)).outcome == "succeeded"
    with Store.open(location) as store:
        restored = LocalWorkspace(
            root, readable=(".",), writable=(".",), commands={}, artifacts=StoreArtifacts(store)
        )
        registry = CapabilityRegistry()
        restored.register(registry, restored_specs=trusted)
        assert dict(registry.specs) == trusted
        assert (await adapter(restored, "write_text").reconcile(issued)).outcome == "succeeded"
        # Even unchanged same-registry replacement rotates current authority.
        restored.register(registry)
        assert registry.specs["write_text"].binding_id != trusted["write_text"].binding_id
        assert (await adapter(restored, "write_text").execute(issued)).outcome == "failed"
        changed = LocalWorkspace(
            root,
            readable=(".",),
            writable=(".",),
            commands={},
            artifacts=StoreArtifacts(store),
            max_file_bytes=50,
        )
        with pytest.raises(ValueError, match="configuration changed"):
            changed.register(CapabilityRegistry(), restored_specs=trusted)


def test_unsupported_platform_refuses_before_workspace_mutation(tmp_path, monkeypatch):
    from signal_gating.executive.errors import StoreError

    monkeypatch.setattr(sys, "platform", "win32")
    target = tmp_path / "absent"
    with pytest.raises(StoreError, match="Linux or macOS"):
        LocalWorkspace(target, readable=(".",), writable=(".",), commands={})
    assert not target.exists()


def test_root_symlink_is_rejected_without_mutation(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        LocalWorkspace(alias, readable=(".",), writable=(".",), commands={})
    assert not list(root.iterdir())


@pytest.mark.asyncio
async def test_check_launch_stays_on_verified_descriptor_during_root_rename(local, monkeypatch):
    root, store, _ = local
    root.joinpath("input.txt").write_text("input")
    workspace, issued = check_workspace(root, store, "open('marker','w').write('bound')")
    create = asyncio.create_subprocess_exec
    renamed = root.with_name("original")

    async def race(*args, **kwargs):
        root.rename(renamed)
        root.mkdir()
        root.joinpath("input.txt").write_text("foreign")
        return await create(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", race)
    result = await adapter(workspace, "run_check").execute(issued)
    assert result.outcome == "outcome_unknown"
    assert renamed.joinpath("marker").read_text() == "bound"
    assert not root.joinpath("marker").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["timeout", "cancel", "leader_exit"])
async def test_check_cleans_descendant_process_group(local, stop):
    root, store, _ = local
    root.joinpath("input.txt").write_text("input")
    child = (
        "import pathlib,time; pathlib.Path('ready').touch(); "
        "time.sleep(1.5); pathlib.Path('escaped').touch()"
    )
    program = (
        "import subprocess,sys,time,pathlib; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "\nwhile not pathlib.Path('ready').exists(): time.sleep(.005)\n"
        + ("" if stop == "leader_exit" else "time.sleep(30)")
    )
    workspace, issued = check_workspace(root, store, program, timeout=0.5)
    task = asyncio.create_task(adapter(workspace, "run_check").execute(issued))
    if stop == "cancel":

        async def ready():
            while not root.joinpath("ready").exists():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(ready(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await adapter(workspace, "run_check").reconcile(issued)).outcome == "needs_human"
    else:
        result = await asyncio.wait_for(task, timeout=3)
        assert result.outcome == "outcome_unknown"
        assert result.explanation == (
            "check_descendants_terminated" if stop == "leader_exit" else "check_timeout"
        )
    assert root.joinpath("ready").exists()
    await asyncio.sleep(1.6)
    assert not root.joinpath("escaped").exists()


@pytest.mark.asyncio
async def test_crash_between_replace_and_completed_receipt_stays_unknown(local, monkeypatch):
    root, store, workspace = local
    issued = write_action(workspace, store)
    bridge = workspace.artifacts
    publish = bridge.publish

    def fail_completion(action, data, media):
        if media.endswith("receipt-v1+json") and json.loads(data)["phase"] == "completed":
            raise ValueError("publication_failed")
        return publish(action, data, media)

    monkeypatch.setattr(bridge, "publish", fail_completion)
    assert (await adapter(workspace, "write_text").execute(issued)).outcome == "outcome_unknown"
    assert root.joinpath("file").read_bytes() == b"corrected\n"
    monkeypatch.setattr(bridge, "publish", publish)
    assert (await adapter(workspace, "write_text").reconcile(issued)).outcome == "needs_human"
    assert (await adapter(workspace, "write_text").execute(issued)).outcome == "needs_human"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["target_inode", "parent", "temporary"])
async def test_detectable_replacement_race_is_refused(local, monkeypatch, change):
    root, store, workspace = local
    root.joinpath("sub").mkdir()
    root.joinpath("sub/file").write_bytes(b"before")
    issued = write_action(workspace, store, "sub/file", sha(b"before"))
    bridge = workspace.artifacts
    publish = bridge.publish

    def race(action, data, media):
        result = publish(action, data, media)
        if media.endswith("receipt-v1+json") and json.loads(data)["phase"] == "prepared":
            if change == "target_inode":
                root.joinpath("other").write_bytes(b"before")
                root.joinpath("other").replace(root / "sub/file")
            elif change == "parent":
                root.joinpath("sub").rename(root / "old")
                root.joinpath("sub").mkdir()
                root.joinpath("sub/file").write_bytes(b"foreign")
            else:
                next(root.joinpath("sub").glob(".sgp-write-*")).write_bytes(b"tampered")
        return result

    monkeypatch.setattr(bridge, "publish", race)
    result = await adapter(workspace, "write_text").execute(issued)
    assert result.outcome != "succeeded"
    assert root.joinpath("sub/file").read_bytes() == (
        b"foreign" if change == "parent" else b"before"
    )
    assert not list(root.rglob(".sgp-write-*"))


@pytest.mark.asyncio
async def test_same_byte_foreign_replacement_after_rename_is_detected(local, monkeypatch):
    import os

    root, store, workspace = local
    issued = write_action(workspace, store)
    replace = os.replace

    def raced(*args, **kwargs):
        result = replace(*args, **kwargs)
        if str(args[0]).startswith(".sgp-write-"):
            root.joinpath("foreign").write_bytes(b"corrected\n")
            root.joinpath("foreign").replace(root / "file")
        return result

    monkeypatch.setattr(os, "replace", raced)
    result = await adapter(workspace, "write_text").execute(issued)
    assert result.outcome == "outcome_unknown"
    assert result.explanation == "resource_changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["delete", "corrupt", "prepared_delete", "replacement_revoke"])
async def test_reconciliation_requires_current_valid_intact_receipt_and_content(local, damage):
    from signal_gating.executive import ArtifactRef

    root, store, workspace = local
    issued = write_action(workspace, store)
    result = await adapter(workspace, "write_text").execute(issued)
    assert result.outcome == "succeeded"
    target_id = (
        issued.arguments["content_ref"]["id"]
        if damage == "replacement_revoke"
        else result.evidence[0 if damage == "prepared_delete" else -1].id
    )
    target = next(r for r in store.view().records if r.id == target_id)
    assert isinstance(target, ArtifactRef)
    if damage == "corrupt":
        store.root.joinpath(target.storage_location).write_bytes(b"corrupt")
    else:
        ArtifactStore(store).delete(target, principal=owner())
    assert (await adapter(workspace, "write_text").reconcile(issued)).outcome == "needs_human"


@pytest.mark.asyncio
async def test_named_check_exact_argv_environment_cwd_and_inputs(local):
    root, store, _ = local
    root.joinpath("input.txt").write_text("input")
    workspace, issued = check_workspace(
        root, store, "import os; print(os.getcwd()); print(dict(os.environ)); print('ok')"
    )
    result = await adapter(workspace, "run_check").execute(issued)
    assert result.outcome == "succeeded"
    receipts = workspace.artifacts.receipts(issued)
    completed = next(d for _, d in receipts if d["phase"] == "completed")
    assert completed["before_sha256"] == completed["after_sha256"] == {"input.txt": sha(b"input")}
    assert completed["argv"] == list(workspace.checks["test"].argv)
    assert completed["criterion_ref"] == dict(issued.arguments["criterion_ref"])
    current = next(r for r in store.view().records if r.id == result.evidence[0].id)
    output = ArtifactStore(store).read(current, principal=owner(), goal_id=GOAL_ID).decode()
    assert str(root) in output
    assert "configured" in output
    assert "HOME" not in output and "SECRET" not in output
    root.joinpath("input.txt").write_text("changed")
    assert (await adapter(workspace, "run_check").reconcile(issued)).outcome == "needs_human"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["subset", "unknown", "extra_args", "stale", "changed"])
async def test_check_cannot_shrink_inputs_add_args_or_reuse_stale_results(local, case):
    root, store, _ = local
    root.joinpath("input.txt").write_text("input")
    root.joinpath("other.txt").write_text("other")
    program = (
        "open('marker','w').write('ran')"
        if case != "changed"
        else "open('input.txt','w').write('changed')"
    )
    workspace, issued = check_workspace(root, store, program, inputs=("input.txt", "other.txt"))
    arguments = issued.model_dump(mode="json")["arguments"]
    if case == "subset":
        arguments["expected_sha256"].pop("other.txt")
    elif case == "unknown":
        arguments["command_id"] = "unregistered"
    elif case == "extra_args":
        arguments["args"] = ["--extra"]
    elif case == "stale":
        arguments["expected_sha256"]["input.txt"] = "0" * 64
    issued = action(workspace, "run_check", arguments)
    result = await adapter(workspace, "run_check").execute(issued)
    assert result.outcome != "succeeded"
    assert not root.joinpath("marker").exists()
    if case == "changed":
        assert result.explanation == "check_inputs_changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["revision", "retired"])
async def test_check_requires_current_goal_embedded_criterion_not_standalone_history(local, change):
    from signal_gating.executive import CommitBatch
    from signal_gating.executive.authority import reference

    from .test_actions import get, update

    root, store, _ = local
    root.joinpath("input.txt").write_text("input")
    workspace, issued = check_workspace(root, store, "open('marker','w').write('bad')")
    target = get(store, GOAL_ID)
    previous = target.criteria[0]
    # Retain the previously promoted standalone definition as historical state.
    store.commit(CommitBatch(events=(event((previous,)),), expected_revisions=(reference(target),)))
    revised = previous.model_copy(update={"revision": previous.revision + 1})
    if change == "retired":
        from .factories import criterion

        revised = criterion(id="9" * 32)
    update(store, target, criteria=(revised,))
    assert any(r.id == previous.id for r in store.view().records)
    result = await adapter(workspace, "run_check").execute(issued)
    assert result.outcome != "succeeded"
    assert result.explanation == "stale_criterion"
    assert not root.joinpath("marker").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["output", "timeout"])
async def test_check_output_and_deadline_are_bounded_and_never_blindly_rerun(local, case):
    root, store, _ = local
    root.joinpath("input.txt").write_text("input")
    program = "print('x'*100000)" if case == "output" else "import time; time.sleep(30)"
    workspace, issued = check_workspace(root, store, program, timeout=0.3, limit=100)
    result = await adapter(workspace, "run_check").execute(issued)
    assert result.outcome == "outcome_unknown"
    assert result.explanation == ("output_limit_exceeded" if case == "output" else "check_timeout")
    current = next(r for r in store.view().records if r.id == result.evidence[0].id)
    assert len(ArtifactStore(store).read(current, principal=owner(), goal_id=GOAL_ID)) <= 100
    assert (await adapter(workspace, "run_check").execute(issued)).outcome == "outcome_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("belief_status", [None, "hypothesis", "disputed"])
@pytest.mark.parametrize("prerequisite", [False, True])
async def test_first_check_from_real_context_keeps_reads_without_assuming_proof(
    local, belief_status, prerequisite
):
    from signal_gating.executive import (
        ActionIntent,
        Belief,
        Budget,
        CommitBatch,
        Commitment,
        Episode,
        EpisodeResult,
        Limits,
    )
    from signal_gating.executive.actions import CapabilityRegistry, Dispatcher
    from signal_gating.executive.authority import Authority, reference
    from signal_gating.executive.budgets import BudgetPolicy
    from signal_gating.executive.context import ContextAssembler
    from signal_gating.executive.proposals import ProposalValidator
    from signal_gating.executive.reducers import rebuild

    from .factories import COMMITMENT_ID, EPISODE_ID, frame, grant
    from .test_actions import BUDGET, WORKER, get
    from .test_context import config

    root, store, _ = local
    root.joinpath("input.txt").write_text("input")
    workspace, direct = check_workspace(root, store, "print('first passing check')")
    target = get(store, GOAL_ID)
    target = target.model_copy(
        update={
            "revision": target.revision + 1,
            "budget_id": BUDGET,
            "limits": Limits(input_tokens=200_000),
        }
    )
    worker = owner(id=WORKER, roles=("worker",))
    task = frame(
        status="running",
        fencing_token=1,
        dependencies=(reference(target.criteria[0]),) if prerequisite else (),
    )
    belief = Belief(
        claim="Unresolved supplied data",
        epistemic_status=belief_status or "hypothesis",
        scope=Scope(goal_id=GOAL_ID),
        created_at=NOW,
    )
    records = (
        target,
        worker,
        task,
        Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID),
        grant(
            subject_id=WORKER,
            capabilities=("run_check",),
            scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID, resource_paths=(".",)),
        ),
        Budget(
            id=BUDGET,
            scope=Scope(goal_id=GOAL_ID),
            limits=Limits(
                input_tokens=0, output_tokens=0, calls=10, elapsed_seconds=100, cost_microunits=0
            ),
        ),
        *((belief,) if belief_status else ()),
    )
    store.commit(
        CommitBatch(
            events=(event(records),),
            expected_revisions=tuple(reference(r) for r in store.view().records),
        )
    )
    registry = CapabilityRegistry()
    workspace.register(registry)
    artifacts = ArtifactStore(store)
    packet = ContextAssembler(
        artifacts, worker=config(), capabilities=tuple(registry.specs.values()), clock=lambda: NOW
    ).build(task, store.view(), principal=worker, episode_id=EPISODE_ID)
    assert reference(target.criteria[0]) in packet.mandatory_read_set
    if belief_status:
        assert reference(belief) in packet.mandatory_read_set
        assert packet.beliefs[0].epistemic_status == belief_status
    episode = Episode(
        id=EPISODE_ID,
        task_id=TASK_ID,
        fencing_token=1,
        context_packet_id=packet.id,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        allowed_proposal_types=("action_intent",),
        deadline=NOW + timedelta(minutes=1),
    )
    try:
        store.commit(CommitBatch(events=(event((packet.rendered_artifact, packet, episode)),)))
    finally:
        artifacts.release_staged((packet.rendered_artifact,))
    intent = ActionIntent(
        episode_id=episode.id,
        fencing_token=1,
        read_set=packet.mandatory_read_set,
        capability_id="run_check",
        arguments=direct.model_dump(mode="json")["arguments"],
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID, resource_paths=(".",)),
    )
    authority = Authority(
        worker_principals={"fixture": WORKER},
        capabilities=registry.specs,
        resource_scopes={name: workspace.resource_scope for name in registry.specs},
    )
    # The same actual Authority gate is used before and immediately at dispatch.
    verdict = authority.check_action(intent, store.view(), NOW)
    if prerequisite:
        assert verdict.disposition == "stale"
        assert verdict.reason_code == "unsettled_action_dependency"
        return
    assert verdict.disposition == "authorize", verdict
    batch = ProposalValidator(clock=lambda: NOW, worker_principals={"fixture": WORKER}).validate(
        EpisodeResult(proposals=(intent,), stop_reason="proposed"), episode, store.view()
    )
    assert batch.response_receipt.disposition == "accept"
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
        from signal_gating.executive import DecideAction
        from signal_gating.executive.intake import Intake

        assert (await dispatcher.dispatch_one(intent.id)).disposition == "require_review"
        pending = get(store, intent.id)
        approval = Intake(store, clock=lambda: NOW).accept(
            DecideAction(
                action_id=pending.id,
                expected_revision=pending.revision,
                decision="approve",
                arguments_digest=pending.arguments_digest,
                scope=pending.scope,
                constraint_revision=pending.constraint_revision,
            ),
            principal=owner(),
            state=store.view(),
        )
        assert approval.response_receipt.disposition == "accept"
        store.commit(approval)
        receipt = await dispatcher.dispatch_one(intent.id)
        assert receipt.disposition == "accept", receipt
        saved = get(store, intent.id)
        assert saved.status == "succeeded"
        assert {r.id for r in packet.mandatory_read_set} <= {r.id for r in saved.read_set}
        # Proposal acceptance advances only task/episode bookkeeping references.
        assert {r for r in packet.mandatory_read_set if r.id != TASK_ID} <= set(saved.read_set)
        assert get(store, saved.reservation_id).status == "settled"
        assert store.view().records == rebuild(store.events()).records
    finally:
        await dispatcher.close()
