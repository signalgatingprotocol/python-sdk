"""Real foreground processes, using fixture workers and no provider credentials."""

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from signal_gating.executive import CreateGoal, KernelConfig, WorkerConfig

from .factories import goal, owner

ROOT = Path(__file__).resolve().parents[2]


def files(tmp_path, **updates):
    config = KernelConfig(
        owner=owner(),
        bootstrap_owner=True,
        workers=(
            WorkerConfig(
                id="fixture", kind="fixture", context_token_limit=32768, max_output_tokens=128
            ),
        ),
        **updates,
    )
    config_file, goal_file = tmp_path / "config.json", tmp_path / "goal.json"
    config_file.write_text(config.model_dump_json())
    goal_file.write_text(CreateGoal(goal=goal()).model_dump_json())
    return config_file, goal_file


def arguments(tmp_path, config, goal_file):
    return [
        "run",
        "--store",
        str(tmp_path / "store"),
        "--config",
        str(config),
        "--goal",
        str(goal_file),
    ]


def invoke(args, *, cwd, data=""):
    return subprocess.run(
        [sys.executable, "-m", "signal_gating.executive", *args],
        input=data,
        text=True,
        capture_output=True,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=15,
    )


class Session:
    def __init__(self, args, cwd, *, script=None):
        launch = (
            [sys.executable, str(script)]
            if script
            else [sys.executable, "-m", "signal_gating.executive"]
        )
        self.process = subprocess.Popen(
            [*launch, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        self.lines = queue.Queue()

        def read():
            for line in self.process.stdout:
                self.lines.put(json.loads(line))

        threading.Thread(target=read, daemon=True).start()

    def wait(self, status=None, command_id=None):
        for _ in range(50):
            value = self.lines.get(timeout=10)
            if (status is None or value["status"] == status) and (
                command_id is None or value["command_id"] == command_id
            ):
                return value
        raise AssertionError("response not observed")

    def send(self, value):
        self.process.stdin.write(value if isinstance(value, str) else json.dumps(value) + "\n")
        self.process.stdin.flush()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        assert self.process.returncode == 0, self.process.stderr.read()


def test_module_runs_from_another_cwd_and_batch_idle_is_not_success(tmp_path):
    config, goal_file = files(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    result = invoke([*arguments(tmp_path, config, goal_file), "--batch"], cwd=other)
    assert result.returncode == 3, result.stderr
    output = [json.loads(line) for line in result.stdout.splitlines()]
    created = next(row for row in output if row["receipt"] is not None)
    assert created["receipt"]["disposition"] == "accept"
    assert created["goal_id"] == goal().id
    from signal_gating.executive import Ref

    refs = [Ref.model_validate(value) for value in created["current_refs"]]
    assert any(ref.id == goal().criteria[0].id for ref in refs)
    assert output[-1]["tasks"][0]["wake_condition"]["reason_code"] == "fixture_owner_needed"
    assert all(set(("command_id", "receipt", "status")) <= row.keys() for row in output)


def test_malformed_recovery_owner_binding_pause_inspect_and_eof(tmp_path):
    config, goal_file = files(tmp_path)
    session = Session(arguments(tmp_path, config, goal_file), tmp_path)
    try:
        ready = session.wait("idle")
        session.send(
            {
                "kind": "revise_goal",
                "id": "a" * 32,
                "goal_id": goal().id,
                "expected_revision": ready["goals"][0]["revision"],
                "outcome": "Streamed correction of the synthetic fixture",
            }
        )
        ready = session.wait(command_id="a" * 32)
        assert ready["goals"][0]["outcome"] == "Streamed correction of the synthetic fixture"
        session.send("API_KEY_VALUE invalid json\n")
        error = session.wait("invalid_command")
        assert "API_KEY_VALUE" not in json.dumps(error)
        session.send(
            {
                "kind": "pause_goal",
                "goal_id": goal().id,
                "expected_revision": ready["goals"][0]["revision"],
                "principal": {"roles": ["owner"]},
            }
        )
        assert session.wait("invalid_command")["receipt"] is None
        command_id = "e" * 32
        session.send(
            {
                "id": command_id,
                "kind": "pause_goal",
                "goal_id": goal().id,
                "expected_revision": ready["goals"][0]["revision"],
            }
        )
        response = session.wait(command_id=command_id)
        assert response["goal_status"] == "paused"
        assert response["receipt"]["subject_id"] == command_id
        assert response["status"] == "accept"
        assert response["receipt"] == next(
            r for r in response["receipts"] if r["id"] == response["receipt"]["id"]
        )
        inspected = invoke(
            ["inspect", "--store", str(tmp_path / "store"), "--goal", goal().id, "--json"],
            cwd=tmp_path,
        )
        assert inspected.returncode == 0, inspected.stderr
        assert json.loads(inspected.stdout)["goal_status"] == "paused"
        second = invoke(arguments(tmp_path, config, goal_file) + ["--batch"], cwd=tmp_path)
        assert second.returncode == 2
        assert json.loads(second.stdout)["reason_code"] == "startup_error"
    finally:
        session.close()
    assert session.process.returncode == 0
    persisted = invoke(["inspect", "--store", str(tmp_path / "store"), "--json"], cwd=tmp_path)
    assert json.loads(persisted.stdout)["goals"][0]["status"] == "paused"


def test_resume_is_explicit_and_does_not_duplicate_commitments(tmp_path):
    config, goal_file = files(tmp_path)
    initial = Session(arguments(tmp_path, config, goal_file), tmp_path)
    try:
        before = initial.wait("idle")
        initial.send(
            {
                "kind": "pause_goal",
                "goal_id": goal().id,
                "expected_revision": before["goals"][0]["revision"],
            }
        )
        initial.wait("accept")
    finally:
        initial.close()
    session = Session(
        ["run", "--store", str(tmp_path / "store"), "--config", str(config), "--resume", goal().id],
        tmp_path,
    )
    try:
        resumed = session.wait("accept")
        assert resumed["receipt"]["reason_code"] == "input_committed"
        assert len(resumed["commitments"]) == len(before["commitments"]) == 1
        assert resumed["goals"][0]["id"] == before["goals"][0]["id"]
    finally:
        session.close()


def test_export_missing_store_fails_without_creating_store_or_output(tmp_path):
    args = [
        "export",
        "--store",
        str(tmp_path / "store"),
        "--goal",
        goal().id,
        "--output",
        str(tmp_path / "export"),
    ]
    result = invoke(args, cwd=tmp_path)
    assert result.returncode == 2
    assert json.loads(result.stdout)["reason_code"] == "startup_error"
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "export").exists()


def test_export_reads_snapshot_during_writer_without_loading_config_or_credentials(tmp_path):
    config, goal_file = files(tmp_path)
    session = Session(arguments(tmp_path, config, goal_file), tmp_path)
    try:
        before = session.wait("idle")
        config.unlink()
        output = tmp_path / "portable.json"
        result = invoke(
            [
                "export",
                "--store",
                str(tmp_path / "store"),
                "--goal",
                goal().id,
                "--output",
                str(output),
            ],
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_text())
        assert value["schema_version"] == 1
        assert value["authority"] == "observation_only"
        assert value["principal_id"] == owner().id
        assert "API_KEY_VALUE" not in output.read_text()
        assert json.loads(result.stdout)["status"] == "exported"
        session.send({"kind": "inspect"})
        after = session.wait("inspect")
        assert after["sequence"] == before["sequence"]
    finally:
        session.close()


def test_live_requires_opt_in_and_explicit_budgets_before_client(tmp_path):
    config, goal_file = files(tmp_path)
    value = json.loads(config.read_text())
    value["workers"][0].update(kind="openai", model="fake", api_key_env="FAKE_KEY")
    config.write_text(json.dumps(value))
    for live in ([], ["--live"]):
        result = invoke(arguments(tmp_path, config, goal_file) + live + ["--batch"], cwd=tmp_path)
        assert result.returncode == 2
        assert "API_KEY_VALUE" not in result.stdout + result.stderr
        assert not (tmp_path / "store").exists()


def test_active_restart_loads_current_goal_without_fake_ack_or_new_commitment(tmp_path):
    config, goal_file = files(tmp_path)
    first = invoke(arguments(tmp_path, config, goal_file) + ["--batch"], cwd=tmp_path)
    assert first.returncode == 3
    before = json.loads(first.stdout.splitlines()[-1])
    again = invoke(
        [
            "run",
            "--store",
            str(tmp_path / "store"),
            "--config",
            str(config),
            "--resume",
            goal().id,
            "--batch",
        ],
        cwd=tmp_path,
    )
    assert again.returncode == 3, again.stdout + again.stderr
    loaded = json.loads(again.stdout.splitlines()[0])
    assert loaded["status"] == "loaded"
    assert loaded["receipt"] is None
    assert loaded["command_id"] is None
    assert loaded["commitments"] == before["commitments"]
    assert loaded["goal_status"] == "active"


def test_config_errors_do_not_echo_credentials_and_paths_are_config_relative(tmp_path):
    from signal_gating.executive.cli import load_config

    config, goal_file = files(tmp_path)
    value = json.loads(config.read_text())
    value["workspace"] = {"root": "workspace"}
    config.write_text(json.dumps(value))
    assert load_config(config).workspace.root == tmp_path / "workspace"
    value["workers"][0]["api_key"] = "API_KEY_VALUE"
    value["API_KEY_VALUE"] = "invalid secret-shaped mapping key"
    config.write_text(json.dumps(value))
    result = invoke(arguments(tmp_path, config, goal_file), cwd=tmp_path)
    assert result.returncode == 2
    assert "API_KEY_VALUE" not in result.stdout + result.stderr
    assert not (tmp_path / "store").exists()


def test_init_is_idempotent_and_corruption_exits_four(tmp_path):
    import sqlite3

    store = tmp_path / "store"
    assert invoke(["init", "--store", str(store)], cwd=tmp_path).returncode == 0
    assert invoke(["init", "--store", str(store)], cwd=tmp_path).returncode == 0
    config, goal_file = files(tmp_path)
    assert (
        invoke(arguments(tmp_path, config, goal_file) + ["--batch"], cwd=tmp_path).returncode == 3
    )
    with sqlite3.connect(store / "executive.sqlite3") as db:
        db.execute("UPDATE records SET checksum = 'invalid'")
    result = invoke(["inspect", "--store", str(store), "--json"], cwd=tmp_path)
    assert result.returncode == 4
    assert json.loads(result.stdout)["error_code"] == "corrupt_store"


def test_fake_provider_in_flight_correction_is_responsive_and_eof_recovers(tmp_path, monkeypatch):
    import time

    config, goal_file = files(tmp_path)
    value = json.loads(config.read_text())
    value["workers"][0].update(
        kind="openai",
        model="fake",
        api_key_env="FAKE_KEY",
        prices={
            "version": "fake-max-v1",
            "input_microunits_per_million_tokens": 0,
            "output_microunits_per_million_tokens": 0,
            "maximum_charge_microunits": 0,
        },
    )
    config.write_text(json.dumps(value))
    creation = json.loads(goal_file.read_text())
    creation["goal"]["limits"].update(
        input_tokens=100000, output_tokens=10000, elapsed_seconds=1000
    )
    goal_file.write_text(json.dumps(creation))
    marker = tmp_path / "fake-call-started"
    script = tmp_path / "fake_client.py"
    script.write_text(
        """import asyncio
from pathlib import Path
from types import SimpleNamespace
from signal_gating.executive.adapters.openai_worker import OpenAIWorker
from signal_gating.executive.cli import main
class Client:
    max_retries = 0
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
    async def create(self, **kwargs):
        Path(MARKER).write_text("started")
        await asyncio.Event().wait()
    async def close(self):
        pass
def construct(**kwargs):
    kwargs.pop("api_key_env")
    return OpenAIWorker(Client(), **kwargs)
OpenAIWorker.from_environment = staticmethod(construct)
raise SystemExit(main())
""".replace("MARKER", repr(str(marker)))
    )
    monkeypatch.setenv("FAKE_KEY", "API_KEY_VALUE")  # synthetic sentinel only
    session = Session(arguments(tmp_path, config, goal_file) + ["--live"], tmp_path, script=script)
    try:
        started = session.wait("accept")
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "accounted fake provider call was not dispatched"
        command_id = "f" * 32
        session.send(
            {
                "kind": "revise_goal",
                "id": command_id,
                "goal_id": goal().id,
                "expected_revision": started["goals"][0]["revision"],
                "outcome": "Corrected while fake provider waits API_KEY_VALUE",
            }
        )
        assert session.wait("invalid_command")["receipt"] is None
        session.send(
            {
                "kind": "revise_goal",
                "id": command_id,
                "goal_id": goal().id,
                "expected_revision": started["goals"][0]["revision"],
                "outcome": "Corrected while fake provider waits",
            }
        )
        corrected = session.wait(command_id=command_id)
        assert corrected["receipt"]["disposition"] == "accept"
        assert corrected["goals"][0]["outcome"] == "Corrected while fake provider waits"
        session.send(
            {
                "kind": "pause_goal",
                "id": "d" * 32,
                "goal_id": goal().id,
                "expected_revision": corrected["goals"][0]["revision"],
            }
        )
        paused = session.wait(command_id="d" * 32)
        assert paused["goal_status"] == "paused"
        assert "API_KEY_VALUE" not in json.dumps(paused)
    finally:
        session.close()
    inspected = invoke(["inspect", "--store", str(tmp_path / "store"), "--json"], cwd=tmp_path)
    assert "API_KEY_VALUE" not in inspected.stdout + inspected.stderr
    final = json.loads(inspected.stdout)
    assert final["goals"][0]["status"] == "paused"
    assert len(final["commitments"]) == 1
    assert final["budgets"][0]["accounting_uncertain"]


async def test_subprocess_exposes_review_and_routes_owner_approval_and_action_decision(tmp_path):
    import hashlib

    from signal_gating.executive import (
        ActionIntent,
        ArgvCheck,
        Criterion,
        EpisodeResult,
        Grant,
        Kernel,
        Limits,
        Principal,
        Scope,
        SetGrant,
        ShareArtifact,
        Usage,
        WorkspaceConfig,
    )
    from signal_gating.executive.actions import CapabilityRegistry
    from signal_gating.executive.authority import reference
    from signal_gating.executive.kernel import SystemClock

    from .factories import criterion

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.txt").write_text("input")
    worker_principal = Principal(
        id="e" * 32,
        name="Fixture",
        trust_domain="local",
        authenticated_origin="fixture",
        roles=("worker",),
    )
    check = criterion(
        verifier_id="local-check-v1",
        acceptance_rule="exit_zero",
        check_id="check",
        workspace_inputs=("input.txt",),
        required=False,
    )
    human = Criterion(
        id="a" * 32,
        description="Owner review",
        evaluator_type="human",
        verifier_id="owner-approval-v1",
        acceptance_rule="explicit_owner_approval",
    )
    g = goal(
        criteria=(check, human),
        limits=Limits(
            input_tokens=100000,
            output_tokens=10000,
            calls=20,
            elapsed_seconds=1000,
            cost_microunits=0,
        ),
    )
    config = KernelConfig(
        owner=owner(),
        bootstrap_owner=True,
        workers=(
            WorkerConfig(
                id="fixture",
                kind="fixture",
                principal=worker_principal,
                context_token_limit=32768,
                max_output_tokens=128,
            ),
        ),
        workspace=WorkspaceConfig(
            root=workspace,
            named_checks=(
                ArgvCheck(
                    id="check", argv=(sys.executable, "-c", "pass"), maximum_charge_microunits=0
                ),
            ),
        ),
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(config.model_dump_json())

    class Proposer:
        async def run_episode(self, packet):
            return EpisodeResult(
                stop_reason="proposed",
                usage=Usage.zero(),
                proposals=(
                    ActionIntent(
                        episode_id=packet.episode_id,
                        fencing_token=packet.fencing_token,
                        read_set=packet.mandatory_read_set,
                        capability_id="run_check",
                        arguments={
                            "command_id": "check",
                            "criterion_ref": reference(check).model_dump(mode="json"),
                            "expected_sha256": {"input.txt": hashlib.sha256(b"input").hexdigest()},
                        },
                        scope=Scope(owner_id=g.owner_id, goal_id=g.id, resource_paths=(".",)),
                    ),
                ),
            )

    async with Kernel.open(
        tmp_path / "store",
        config=config,
        workers={"fixture": Proposer()},
        capabilities=CapabilityRegistry(),
        verifiers={},
        clock=SystemClock(),
    ) as k:
        await k.submit(CreateGoal(goal=g), principal=k.owner)
        await k.submit(
            SetGrant(
                grant=Grant(
                    issuer_id=owner().id,
                    subject_id=worker_principal.id,
                    capabilities=("run_check",),
                    scope=Scope(owner_id=owner().id, goal_id=g.id, resource_paths=(".",)),
                )
            ),
            principal=k.owner,
        )
        summary = await k.run_until_idle()
        assert len(summary.pending_review_ids) == 1
        artifact = k.artifacts.put(
            b"reviewed bytes", principal=k.owner, goal_id=g.id, media_type="text/plain"
        )
        await k.submit(
            ShareArtifact(
                artifact_id=artifact.id,
                expected_revision=artifact.revision,
                principal_ids=(k.verifier_principal.id,),
            ),
            principal=k.owner,
        )
        artifact = next(r for r in k.inspect().state.records if r.id == artifact.id)
    session = Session(
        ["run", "--store", str(tmp_path / "store"), "--config", str(config_file), "--resume", g.id],
        tmp_path,
    )
    try:
        ready = session.wait("idle")
        assert len(ready["pending_reviews"]) == 1
        selected = next(a for a in ready["artifacts"] if a["id"] == artifact.id)
        assert selected["revision"] == artifact.revision
        approval_id = "b" * 32
        session.send(
            {
                "kind": "approve_criterion",
                "id": approval_id,
                "goal_id": g.id,
                "expected_revision": ready["goals"][0]["revision"],
                "criterion": reference(human).model_dump(mode="json"),
                "artifacts": [reference(artifact).model_dump(mode="json")],
                "explanation": "Reviewed these exact fixture bytes",
            }
        )
        approved = session.wait(command_id=approval_id)
        assert approved["receipt"]["disposition"] == "accept"
        assert approved["approvals"][0]["artifacts"][0]["revision"] == artifact.revision
        action = approved["pending_reviews"][0]
        session.send(
            {
                "kind": "decide_action",
                "id": "c" * 32,
                "action_id": action["id"],
                "expected_revision": action["revision"],
                "decision": "deny",
                "arguments_digest": action["arguments_digest"],
                "scope": action["scope"],
                "constraint_revision": action["constraint_revision"],
            }
        )
        denied = session.wait(command_id="c" * 32)
        assert denied["receipt"]["disposition"] == "accept", json.dumps(denied["receipt"])
        assert denied["actions"][0]["status"] == "denied"
        assert not denied["pending_reviews"]
        # Even well-formed resolution controls use shared current-state policy.
        session.send(
            {
                "kind": "resolve_action",
                "id": "d" * 32,
                "action_id": action["id"],
                "expected_revision": denied["actions"][0]["revision"],
                "outcome": "succeeded",
                "evidence": [reference(artifact).model_dump(mode="json")],
                "explanation": "Synthetic selected evidence",
            }
        )
        invalid_resolution = session.wait(command_id="d" * 32)
        assert invalid_resolution["receipt"]["disposition"] == "deny"
        assert invalid_resolution["receipt"]["reason_code"] == "invalid_action_resolution"
    finally:
        session.close()


async def test_serve_callbacks_return_actual_transient_ack_and_settled_view(tmp_path):
    from signal_gating.executive import AuthenticatedCommand, PauseGoal

    from .test_kernel import FunctionWorker, blocked, new_goal, open_kernel

    async with open_kernel(tmp_path, FunctionWorker(blocked)) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        await kernel.run_until_idle()
        current = next(r for r in kernel.inspect().state.records if r.kind == "goal")
        command = PauseGoal(goal_id=current.id, expected_revision=current.revision)
        kernel.intake.max_queued_control_items = 0
        responses, idles = [], []
        observed = asyncio.Event()

        async def stream():
            yield AuthenticatedCommand(command=command, principal=owner())
            await observed.wait()

        def response(item, receipt):
            responses.append((item, receipt))
            assert receipt.reason_code == "control_queue_full"
            assert receipt.id not in {r.id for r in kernel.inspect().state.records}

        def idle(view):
            idles.append(view)
            observed.set()

        await asyncio.wait_for(kernel.serve(stream(), on_response=response, on_idle=idle), 5)
        assert responses[0][0].command == command
        assert responses[0][1].disposition == "reject"
        assert any(
            r.kind == "task" and r.wake_condition is not None for r in idles[0].state.records
        )


async def test_subprocess_resolves_unknown_action_and_usage_with_selected_evidence(tmp_path):
    from signal_gating.executive import (
        ActionIntent,
        ActionObservation,
        CapabilitySpec,
        EpisodeResult,
        Grant,
        Kernel,
        Limits,
        Principal,
        Scope,
        SetGrant,
        Usage,
    )
    from signal_gating.executive.actions import CapabilityRegistry
    from signal_gating.executive.authority import reference
    from signal_gating.executive.kernel import SystemClock

    from .test_kernel import new_goal

    principal = Principal(
        id="e" * 32,
        name="Fixture",
        trust_domain="local",
        authenticated_origin="fixture",
        roles=("worker",),
    )
    config = KernelConfig(
        owner=owner(),
        bootstrap_owner=True,
        workers=(
            WorkerConfig(
                id="fixture",
                kind="fixture",
                principal=principal,
                context_token_limit=32768,
                max_output_tokens=128,
            ),
        ),
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(config.model_dump_json())
    registry = CapabilityRegistry()

    class UnknownRead:
        async def execute(self, action):
            return ActionObservation(
                action_id=action.id,
                expected_revision=action.revision,
                outcome="outcome_unknown",
                usage=Usage(input_tokens=0, output_tokens=0, calls=1, elapsed_seconds=0),
            )

        async def reconcile(self, action):
            raise AssertionError("Human reconciliation only")

        async def cancel(self, action):
            return await self.execute(action)

    registry.register(
        CapabilitySpec(
            id="fake.read",
            binding_version="v1",
            input_schema={"type": "object", "additionalProperties": False},
            effect_class="read_only",
            reconciliation="human",
        ),
        UnknownRead(),
        quote=Limits(
            input_tokens=0, output_tokens=0, calls=1, elapsed_seconds=10, cost_microunits=0
        ),
        quote_source="free-local-fixture",
        quote_version="v1",
        resource_scope=lambda _: Scope(),
    )

    class Proposer:
        async def run_episode(self, packet):
            return EpisodeResult(
                stop_reason="proposed",
                usage=Usage.zero(),
                proposals=(
                    ActionIntent(
                        episode_id=packet.episode_id,
                        fencing_token=packet.fencing_token,
                        read_set=packet.mandatory_read_set,
                        capability_id="fake.read",
                        arguments={},
                        scope=Scope(owner_id=owner().id, goal_id=packet.goal.id),
                    ),
                ),
            )

    g = new_goal()
    async with Kernel.open(
        tmp_path / "store",
        config=config,
        workers={"fixture": Proposer()},
        capabilities=registry,
        verifiers={},
        clock=SystemClock(),
    ) as k:
        await k.submit(CreateGoal(goal=g), principal=k.owner)
        await k.submit(
            SetGrant(
                grant=Grant(
                    issuer_id=owner().id,
                    subject_id=principal.id,
                    capabilities=("fake.read",),
                    scope=Scope(owner_id=owner().id, goal_id=g.id),
                )
            ),
            principal=k.owner,
        )
        summary = await k.run_until_idle()
        assert len(summary.unknown_action_ids) == 1
        artifact = k.artifacts.put(
            b"synthetic external observation",
            principal=k.owner,
            goal_id=g.id,
            media_type="text/plain",
        )
    session = Session(
        ["run", "--store", str(tmp_path / "store"), "--config", str(config_file), "--resume", g.id],
        tmp_path,
    )
    try:
        ready = session.wait("idle")
        action = next(
            a for a in ready["actions"] if a["id"] in ready["summary"]["unknown_action_ids"]
        )
        session.send(
            {
                "kind": "resolve_action",
                "id": "b" * 32,
                "action_id": action["id"],
                "expected_revision": action["revision"],
                "outcome": "succeeded",
                "evidence": [reference(artifact).model_dump(mode="json")],
                "explanation": "Selected synthetic external observation",
            }
        )
        resolved = session.wait(command_id="b" * 32)
        assert resolved["receipt"]["disposition"] == "accept"
        assert resolved["actions"][0]["status"] == "succeeded"
        reservation = next(r for r in resolved["reservations"] if r["status"] == "uncertain")
        session.send(
            {
                "kind": "resolve_usage",
                "id": "c" * 32,
                "reservation_id": reservation["id"],
                "expected_revision": reservation["revision"],
                "usage": Usage.zero().model_copy(update={"calls": 1}).model_dump(mode="json"),
                "evidence": [reference(artifact).model_dump(mode="json")],
                "explanation": "Evidence-backed synthetic zero-cost observation",
            }
        )
        settled = session.wait(command_id="c" * 32)
        assert settled["receipt"]["disposition"] == "accept"
        assert not settled["summary"]["unknown_action_ids"]
        assert (
            next(r for r in settled["reservations"] if r["id"] == reservation["id"])["status"]
            == "settled"
        )
        assert not any(r["status"] == "uncertain" for r in settled["reservations"])
        assert all(not b["accounting_uncertain"] for b in settled["budgets"])
    finally:
        session.close()


@pytest.mark.parametrize("phase", ["idle", "response"])
async def test_observer_failure_stops_service_including_before_reader_runs(tmp_path, phase):
    from signal_gating.executive import AuthenticatedCommand, CreateGoal

    from .test_kernel import FunctionWorker, blocked, new_goal, open_kernel

    async with open_kernel(tmp_path, FunctionWorker(blocked)) as kernel:

        async def stream():
            if phase == "response":
                yield AuthenticatedCommand(command=CreateGoal(goal=new_goal()), principal=owner())
            await asyncio.Event().wait()

        def fail(_):
            raise ValueError("observer failed")

        with pytest.raises(ValueError, match="observer failed"):
            await kernel.serve(
                stream(),
                on_idle=fail if phase == "idle" else None,
                on_response=(lambda _, ack: fail(ack)) if phase == "response" else None,
            )
        with pytest.raises(RuntimeError, match="stopped"):
            await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())


def test_cancelled_resume_requires_new_goal_and_preserves_existing_work(tmp_path):
    config, goal_file = files(tmp_path)
    session = Session(arguments(tmp_path, config, goal_file), tmp_path)
    try:
        ready = session.wait("idle")
        session.send(
            {
                "kind": "cancel_goal",
                "id": "d" * 32,
                "goal_id": goal().id,
                "expected_revision": ready["goals"][0]["revision"],
            }
        )
        assert session.wait(command_id="d" * 32)["goal_status"] == "cancelled"
    finally:
        session.close()
    before = invoke(["inspect", "--store", str(tmp_path / "store"), "--json"], cwd=tmp_path)
    resumed = invoke(
        [
            "run",
            "--store",
            str(tmp_path / "store"),
            "--config",
            str(config),
            "--resume",
            goal().id,
            "--batch",
        ],
        cwd=tmp_path,
    )
    assert resumed.returncode == 2
    assert "new goal" in json.loads(resumed.stdout)["detail"]
    after = invoke(["inspect", "--store", str(tmp_path / "store"), "--json"], cwd=tmp_path)
    assert before.stdout == after.stdout


@pytest.mark.parametrize("operation", ["init", "inspect", "run"])
def test_physical_sqlite_corruption_exits_four_without_echoing_data(tmp_path, operation):
    store = tmp_path / "store"
    assert invoke(["init", "--store", str(store)], cwd=tmp_path).returncode == 0
    with (store / "executive.sqlite3").open("r+b") as stream:
        stream.write(b"API_KEY_VALUE broken SQLite header")
    if operation == "run":
        config, goal_file = files(tmp_path)
        args = arguments(tmp_path, config, goal_file) + ["--batch"]
    else:
        args = [operation, "--store", str(store)]
    result = invoke(args, cwd=tmp_path)
    assert result.returncode == 4, result.stdout + result.stderr
    response = json.loads(result.stdout)
    assert response["reason_code"] == "runtime_error"
    assert response["error_code"] == "store_open_failed"
    assert "API_KEY_VALUE" not in result.stdout + result.stderr


@pytest.mark.parametrize("failure", ["missing", "unsupported_schema", "permission"])
def test_ordinary_store_startup_failures_stay_exit_two(tmp_path, failure, monkeypatch, capsys):
    import sqlite3

    from signal_gating.executive.cli import main
    from signal_gating.executive.errors import StoreError
    from signal_gating.executive.store import Store

    store = tmp_path / "store"
    if failure == "unsupported_schema":
        assert invoke(["init", "--store", str(store)], cwd=tmp_path).returncode == 0
        with sqlite3.connect(store / "executive.sqlite3") as db:
            db.execute("PRAGMA user_version=999")
    elif failure == "permission":

        def denied(_):
            raise StoreError("store_open_failed", "Could not open executive snapshot") from (
                PermissionError("API_KEY_VALUE sensitive permission failure")
            )

        monkeypatch.setattr(Store, "open_readonly", denied)
    assert main(["inspect", "--store", str(store), "--json"]) == 2
    output = capsys.readouterr()
    response = json.loads(output.out)
    assert response["reason_code"] == "startup_error"
    assert "API_KEY_VALUE" not in output.out + output.err
    if failure == "unsupported_schema":
        assert response["error_code"] == "unsupported_schema"
    elif failure == "missing":
        assert not store.exists()


@pytest.mark.parametrize("message", ["database disk image is malformed", "file is not a database"])
def test_legacy_sqlite_corruption_without_errorcode_still_exits_four(message, monkeypatch, capsys):
    import sqlite3

    from signal_gating.executive.cli import main
    from signal_gating.executive.errors import StoreError
    from signal_gating.executive.store import Store

    # sqlite_errorcode/sqlite_errorname were added in Python 3.11. A constructed
    # sqlite exception has neither attribute, exercising the Python 3.10 path.
    error = sqlite3.DatabaseError(message)
    assert not hasattr(error, "sqlite_errorcode")

    def corrupt(_):
        raise StoreError("store_open_failed", "Could not open executive snapshot") from error

    monkeypatch.setattr(Store, "open_readonly", corrupt)
    assert main(["inspect", "--store", "/unused", "--json"]) == 4
    response = json.loads(capsys.readouterr().out)
    assert response["reason_code"] == "runtime_error"
    assert message not in json.dumps(response)


@pytest.mark.parametrize("sqlite_code,expected", [(11, 4), (26, 4), (267, 4), (14, 2), (1, 2)])
def test_sqlite_primary_and_extended_error_codes_are_classified_without_echo(
    sqlite_code,
    expected,
    monkeypatch,
    capsys,
):
    import sqlite3

    from signal_gating.executive.cli import main
    from signal_gating.executive.errors import StoreError
    from signal_gating.executive.store import Store

    # Stable SQLite primary codes: CORRUPT=11, NOTADB=26, CANTOPEN=14,
    # ERROR=1; extended CORRUPT_VTAB=267. Prefer codes over message inference.
    error = sqlite3.DatabaseError("API_KEY_VALUE file is not a database")
    error.sqlite_errorcode = sqlite_code

    def fail(_):
        raise StoreError("store_open_failed", "Could not open executive snapshot") from error

    monkeypatch.setattr(Store, "open_readonly", fail)
    assert main(["inspect", "--store", "/unused", "--json"]) == expected
    captured = capsys.readouterr()
    assert "API_KEY_VALUE" not in captured.out + captured.err
    assert json.loads(captured.out)["error_code"] == "store_open_failed"


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("close_failure", [False, True])
async def test_partial_worker_startup_closes_all_transports_with_redacted_exit_precedence(
    tmp_path,
    monkeypatch,
    capsys,
    close_failure,
    cancelled,
):
    import argparse

    from signal_gating.executive import cli
    from signal_gating.executive.adapters.openai_worker import OpenAIWorker
    from signal_gating.executive.context import UTF8ByteCounter

    config, goal_file = files(tmp_path)
    closed = []

    class Client:
        max_retries = 0

        def __init__(self, name):
            self.name = name

        async def close(self):
            closed.append(self.name)
            if close_failure and self.name == "first":
                raise RuntimeError("API_KEY_VALUE sensitive close failure")

    def partial_construction(_, workers):
        for name in ("first", "second"):
            workers[name] = OpenAIWorker(
                Client(name), model="fake", counter=UTF8ByteCounter(), capacity=32768, quote=None
            )
        if cancelled:
            raise asyncio.CancelledError("API_KEY_VALUE sensitive cancellation")
        raise ValueError("API_KEY_VALUE sensitive startup failure")

    monkeypatch.setattr(cli, "_workers", partial_construction)
    args = argparse.Namespace(
        config=config, goal=goal_file, store=tmp_path / "store", live=False, batch=True
    )
    if cancelled and not close_failure:
        with pytest.raises(asyncio.CancelledError):
            await cli._run(args)
    else:
        assert await cli._run(args) == (4 if close_failure else 2)
    assert closed == ["first", "second"]
    captured = capsys.readouterr()
    assert "API_KEY_VALUE" not in captured.out + captured.err
    responses = [json.loads(line) for line in captured.out.splitlines()]
    if not cancelled:
        assert responses[0]["reason_code"] == "startup_error"
    if close_failure:
        assert responses[-1]["reason_code"] == "runtime_error"
    assert not args.store.exists()
