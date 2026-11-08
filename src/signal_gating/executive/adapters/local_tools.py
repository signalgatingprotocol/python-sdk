"""Trusted local tools, not a sandbox against configured programs or host writers.

The coordinator injects StoreArtifacts and calls register before building contexts.
All artifact access/publication runs on that owning thread outside transactions.
Completed receipts are ordinary scoped artifacts, not a second action ledger.
Imports are portable; constructing a workspace refuses unsupported writer hosts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import stat
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from ..actions import CapabilityRegistry, ExecutionAdapter
from ..artifacts import ArtifactStore
from ..authority import reference
from ..config import ArgvCheck
from ..errors import ExecutiveError
from ..locking import require_writer_platform
from ..messages import ActionIntent
from ..records import (
    ActionObservation,
    ActionRecord,
    ArtifactRef,
    CapabilitySpec,
    Criterion,
    Goal,
    Limits,
    Principal,
    Ref,
    Scope,
    Usage,
)
from ..store import Store

_RECEIPT = "application/vnd.sgp.workspace-receipt-v1+json"
_PUBLICATION_SOURCE = "local-workspace-v2"
_MISSING = "missing"
# Static package-owned launcher: no shell, interpolation, imports from cwd or
# checkout-relative executable. -I/-S isolates this plumbing from host startup.
# Closing the sole inherited directory descriptor prevents leaking it to checks.
_LAUNCHER_VERSION = "fchdir-exec-env-v2"
_LAUNCHER = (
    "import json,os,sys; env=json.load(sys.stdin); "
    "fd=int(sys.argv[1]); os.fchdir(fd); os.close(fd); "
    "null=os.open(os.devnull,os.O_RDONLY); os.dup2(null,0); os.close(null); "
    "os.execve(sys.argv[2],sys.argv[2:],env)"
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _parts(path: Any, *, directory: bool = False) -> tuple[str, ...]:
    if directory and path == ".":
        return ()
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("invalid_path")
    path.encode("utf-8")
    parts = tuple(path.split("/"))
    if any(p in {"", ".", ".."} for p in parts):
        raise ValueError("invalid_path")
    return parts


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class StoreArtifacts:
    """Single-writer bridge. Never use from a pool/thread or inside a transaction.

    Receipt provenance comes from adapter-owned creation journal events, not owner
    origin, MIME or content claims (worker drafts also have owner origin). Task13
    consumes receipts() then checks action lineage/current ACLs/inputs independently.
    """

    def __init__(self, store: Store) -> None:
        self.store = store
        self.artifacts = ArtifactStore(store)

    def _ready(self) -> None:
        self.store.check_writable()
        if self.store._connection.in_transaction:
            raise RuntimeError("Local tool calls cannot run inside a transaction")

    def _owner(self, action: ActionRecord) -> Principal:
        self._ready()
        records = {r.id: r for r in self.store.view().records}
        goal = records.get(action.goal_id)
        if not isinstance(goal, Goal):
            raise ValueError("missing_goal")
        principal = records.get(goal.owner_id)
        if not isinstance(principal, Principal):
            raise ValueError("missing_owner")
        return principal

    def content(self, action: ActionRecord, ref: Ref) -> bytes:
        self._ready()
        records = {r.id: r for r in self.store.view().records}
        current = records.get(ref.id)
        principal = records.get(action.origin_principal_id or "")
        if (
            not isinstance(current, ArtifactRef)
            or (current.id, current.revision) != (ref.id, ref.revision)
            or ref.kind not in {"record", "artifact"}
            or not any((r.id, r.revision) == (ref.id, ref.revision) for r in action.read_set)
            or not isinstance(principal, Principal)
        ):
            raise ValueError("inaccessible_content")
        return self.artifacts.read(current, principal=principal, goal_id=action.goal_id)

    def criterion(self, ref: Ref, goal_id: str) -> Criterion:
        self._ready()
        goal = next((r for r in self.store.view().records if r.id == goal_id), None)
        if not isinstance(goal, Goal):
            raise ValueError("missing_goal")
        criterion = next((c for c in goal.criteria if c.id == ref.id), None)
        if (
            criterion is None
            or (criterion.id, criterion.revision) != (ref.id, ref.revision)
            or ref.kind not in {"record", "criterion"}
        ):
            raise ValueError("stale_criterion")
        return criterion

    def publish(self, action: ActionRecord, data: bytes, media_type: str) -> Ref:
        principal = self._owner(action)
        return reference(
            self.artifacts.put(
                data,
                principal=principal,
                goal_id=action.goal_id,
                media_type=media_type,
                publication_source=_PUBLICATION_SOURCE,
            )
        )

    def receipts(self, action: ActionRecord) -> tuple[tuple[Ref, dict[str, Any]], ...]:
        """Authenticate adapter publication, then read current scoped receipt bytes.

        Public verifier hook: creation-event provenance survives ACL revisions,
        but validity/access/integrity use current ArtifactStore reads. The caller
        must additionally bind the receipt body to the action/check/configuration.
        Generic put(), accepted drafts and body claims cannot supply provenance.
        """
        principal = self._owner(action)
        published: set[str] = set()
        for event in self.store.events():
            if (
                event.event_type != "state.changed"
                or event.producer_id != _PUBLICATION_SOURCE
                or event.principal_id != principal.id
                or event.subject is None
                or event.subject.revision != 1
                or event.subject.kind != "artifact"
            ):
                continue
            for image in event.model_dump(mode="json")["payload"]["records"]:
                if (
                    image.get("kind") == "artifact"
                    and image.get("id") == event.subject.id
                    and image.get("revision") == 1
                    and image.get("origin") == principal.id
                    and image.get("scope", {}).get("goal_id") == action.goal_id
                ):
                    published.add(image["id"])
        found = []
        for current in self.store.view().records:
            if (
                isinstance(current, ArtifactRef)
                and current.id in published
                and current.valid
                and current.origin == principal.id
                and current.media_type == _RECEIPT
                and current.scope.goal_id == action.goal_id
            ):
                data = json.loads(
                    self.artifacts.read(current, principal=principal, goal_id=action.goal_id)
                )
                if isinstance(data, dict) and data.get("action_id") == action.id:
                    found.append((reference(current), data))
        return tuple(found)


class LocalWorkspace:
    """Relative prefix scopes; '.' covers the root, never symlink components.

    commands contains exact trusted argv. check_options adds timeout/output/env/cost
    metadata for those same IDs/argv, never additional commands. Unknown spending
    bounds default closed under strict-money policy. Root and trusted configuration
    determine a stable first-registration identity; replacement in one registry
    still rotates it. A changed root inode/configuration requires new authority.
    """

    def __init__(
        self,
        root: Path,
        *,
        readable: tuple[str, ...],
        writable: tuple[str, ...],
        commands: Mapping[str, tuple[str, ...]],
        artifacts: StoreArtifacts | None = None,
        check_options: Mapping[str, ArgvCheck] | None = None,
        max_file_bytes: int = 1_048_576,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        require_writer_platform()
        for value in (max_file_bytes, max_entries):
            if type(value) is not int or value <= 0:
                raise ValueError("Workspace limits must be positive integers")
        self.root = Path(os.path.abspath(root))
        self.readable = tuple(_parts(p, directory=True) for p in readable)
        self.writable = tuple(_parts(p, directory=True) for p in writable)
        self.artifacts, self.clock = artifacts, clock
        self.max_file_bytes, self.max_entries = max_file_bytes, max_entries
        options = dict(check_options or {})
        if set(options) - commands.keys():
            raise ValueError("Check options cannot add commands")
        self.checks = {
            key: options.get(key, ArgvCheck(id=key, argv=argv)) for key, argv in commands.items()
        }
        for key, config in self.checks.items():
            if (
                config.id != key
                or config.argv != commands[key]
                or not Path(config.argv[0]).is_absolute()
            ):
                raise ValueError("Checks require exact configured argv with an absolute executable")
            if any("\x00" in arg for arg in config.argv) or any(
                "\x00" in v for v in config.environment.values()
            ):
                raise ValueError("Invalid command configuration")
        with self._root() as descriptor:
            self._identity = (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
        self._lock = asyncio.Lock()
        config_identity = {
            "root": str(self.root),
            "inode": self._identity,
            "readable": readable,
            "writable": writable,
            "checks": {k: c.model_dump(mode="json") for k, c in sorted(self.checks.items())},
            "max_file_bytes": max_file_bytes,
            "max_entries": max_entries,
            "artifact_store": str(artifacts.store.root) if artifacts else None,
            "implementation": "local-workspace-v1",
            "launcher": _LAUNCHER_VERSION,
            "publication_source": _PUBLICATION_SOURCE,
        }
        self.configuration_digest = _digest(_json(config_identity))
        self._pairs: tuple[tuple[CapabilitySpec, ExecutionAdapter], ...] = tuple(
            self._capability(name)
            for name in ("list_files", "read_text", "write_text", "run_check")
        )

    @contextmanager
    def _root(self) -> Iterator[int]:
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in self.root.parts[1:]:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            info = os.fstat(descriptor)
            if hasattr(self, "_identity") and (info.st_dev, info.st_ino) != self._identity:
                raise ValueError("resource_changed")
            yield descriptor
        finally:
            os.close(descriptor)

    @contextmanager
    def _directory(self, parts: tuple[str, ...]) -> Iterator[int]:
        with self._root() as root:
            descriptor = os.dup(root)
            try:
                for part in parts:
                    child = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=descriptor,
                    )
                    os.close(descriptor)
                    descriptor = child
                yield descriptor
            finally:
                os.close(descriptor)

    def _anchor(self, parts: tuple[str, ...], descriptor: int) -> None:
        with self._directory(parts) as current:
            if (os.fstat(current).st_dev, os.fstat(current).st_ino) != (
                os.fstat(descriptor).st_dev,
                os.fstat(descriptor).st_ino,
            ):
                raise ValueError("resource_changed")

    def _allowed(
        self, path: str, *, write: bool = False, directory: bool = False
    ) -> tuple[str, ...]:
        parts = _parts(path, directory=directory)
        allowed = self.writable if write else self.readable
        if not any(parts[: len(prefix)] == prefix for prefix in allowed):
            raise ValueError("path_denied")
        return parts

    def _read_at(self, parent: int, name: str) -> tuple[bytes, tuple[int, ...]]:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > self.max_file_bytes:
                raise ValueError("invalid_file")
            chunks = []
            size = 0
            while chunk := os.read(descriptor, min(65_536, self.max_file_bytes + 1 - size)):
                chunks.append(chunk)
                size += len(chunk)
                if size > self.max_file_bytes:
                    raise ValueError("file_too_large")
            signature = _signature(before)
            if signature != _signature(os.fstat(descriptor)) or signature != _signature(
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            ):
                raise ValueError("resource_changed")
            return b"".join(chunks), signature
        finally:
            os.close(descriptor)

    def _read(self, path: str, *, write: bool = False) -> bytes:
        parts = self._allowed(path, write=write)
        with self._directory(parts[:-1]) as parent:
            data, _ = self._read_at(parent, parts[-1])
            self._anchor(parts[:-1], parent)
            return data

    def _bridge(self) -> StoreArtifacts:
        if self.artifacts is None:
            raise ValueError("missing_artifact_bridge")
        return self.artifacts

    def _criterion(self, arguments: Mapping[str, Any], goal_id: str) -> Criterion:
        ref = Ref.model_validate(arguments["criterion_ref"])
        criterion = self._bridge().criterion(ref, goal_id)
        if criterion.check_id != arguments["command_id"] or not criterion.workspace_inputs:
            raise ValueError("invalid_check_criterion")
        if len(set(criterion.workspace_inputs)) != len(criterion.workspace_inputs):
            raise ValueError("invalid_check_criterion")
        for path in criterion.workspace_inputs:
            self._allowed(path)
        if set(arguments["expected_sha256"]) != set(criterion.workspace_inputs):
            raise ValueError("invalid_check_inputs")
        return criterion

    def _validate(self, name: str, arguments: Mapping[str, Any]) -> bool:
        try:
            required = (
                {"path"}
                if name in {"list_files", "read_text"}
                else {"path", "expected_sha256", "content_ref"}
                if name == "write_text"
                else {"command_id", "criterion_ref", "expected_sha256"}
            )
            if set(arguments) != required:
                return False
            if name == "run_check":
                if arguments["command_id"] not in self.checks:
                    return False
                Ref.model_validate(arguments["criterion_ref"])
                digests = arguments["expected_sha256"]
                return isinstance(digests, Mapping) and all(
                    isinstance(k, str)
                    and self._allowed(k)
                    and isinstance(v, str)
                    and re.fullmatch(r"[0-9a-f]{64}", v)
                    for k, v in digests.items()
                )
            self._allowed(
                arguments["path"], write=name == "write_text", directory=name == "list_files"
            )
            if name == "write_text":
                Ref.model_validate(arguments["content_ref"])
                expected = arguments["expected_sha256"]
                return isinstance(expected, str) and bool(
                    expected == _MISSING or re.fullmatch(r"[0-9a-f]{64}", expected)
                )
            return True
        except (ValueError, TypeError, KeyError, UnicodeError):
            return False

    def resource_scope(self, intent: ActionIntent) -> Scope:
        if not self._validate(intent.capability_id, intent.arguments):
            raise ValueError("invalid_action_arguments")
        if intent.capability_id == "run_check":
            if intent.scope.goal_id is None:
                raise ValueError("missing_goal")
            self._criterion(intent.arguments, intent.scope.goal_id)
            # Trusted programs can affect the whole root; exact-file scope would lie.
            paths = (".",)
        else:
            paths = (cast(str, intent.arguments["path"]),)
        return Scope(resource_paths=paths)

    def _capability(self, name: str) -> tuple[CapabilitySpec, ExecutionAdapter]:
        ref_schema = {
            "type": "object",
            "required": ["id", "revision"],
            "properties": {
                "schema_version": {"type": "integer", "enum": [1]},
                "id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
                "revision": {"type": "integer", "minimum": 1},
                "kind": {"type": "string", "enum": ["record", "artifact", "criterion"]},
            },
            "additionalProperties": False,
        }
        properties: dict[str, Any] = {"path": {"type": "string", "minLength": 1}}
        if name == "write_text":
            properties.update(
                {
                    "expected_sha256": {"type": "string", "pattern": "^(missing|[0-9a-f]{64})$"},
                    "content_ref": ref_schema,
                }
            )
        elif name == "run_check":
            properties = {
                "command_id": {
                    "type": "string",
                    **({"enum": sorted(self.checks)} if self.checks else {}),
                },
                "criterion_ref": ref_schema,
                "expected_sha256": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                },
            }
        spec = CapabilitySpec(
            id=name,
            description=self.configuration_digest,
            binding_version="local-workspace-v1",
            binding_id=_digest(_json([self.configuration_digest, name]))[:32],
            input_schema={
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
                "description": "Relative paths; checks require the exact owner-declared input set",
            },
            effect_class="read_only"
            if name in {"list_files", "read_text"}
            else "reversible"
            if name == "write_text"
            else "consequential",
            resource_scope_rules=("descriptor-relative-prefix-v1",),
            reconciliation="lookup",
            cancellation="best_effort",
            usage_accounting="known" if name != "run_check" else "unknown",
        )
        return spec, _LocalAdapter(self, spec)

    def capabilities(self) -> tuple[tuple[CapabilitySpec, ExecutionAdapter], ...]:
        return self._pairs

    def register(
        self,
        registry: CapabilityRegistry,
        *,
        restored_specs: Mapping[str, CapabilitySpec] | None = None,
    ) -> None:
        """Mandatory extractor/validator and versioned per-call maximum configuration."""
        if restored_specs is not None and set(restored_specs) != {s.id for s, _ in self._pairs}:
            raise ValueError("Restore requires the complete trusted workspace specs")
        for spec, adapter in self._pairs:
            if restored_specs is not None:
                restored = restored_specs[spec.id]
                if restored.binding_id is None or restored.model_copy(
                    update={"binding_id": None}
                ) != spec.model_copy(update={"binding_id": None}):
                    raise ValueError("Restored binding configuration changed")
                spec = restored

            def validate(args: Mapping[str, Any], name: str = spec.id) -> bool:
                return self._validate(name, args)

            check = spec.id == "run_check"
            costs = [c.maximum_charge_microunits for c in self.checks.values()]
            cost = (
                0
                if not check
                else None
                if not costs or any(c is None for c in costs)
                else max((c for c in costs if c is not None), default=0)
            )
            duration = (
                max((c.timeout_seconds for c in self.checks.values()), default=30) + 2
                if check
                else 30
            )
            registry.register(
                spec,
                adapter,
                resource_scope=self.resource_scope,
                input_validator=validate,
                quote=Limits(
                    input_tokens=0,
                    output_tokens=0,
                    calls=1,
                    elapsed_seconds=duration,
                    cost_microunits=cost,
                ),
                quote_source="trusted-local-workspace-config",
                quote_version=self.configuration_digest,
            )
            assert isinstance(adapter, _LocalAdapter)
            adapter.spec = registry.specs[spec.id]


class _LocalAdapter:
    def __init__(self, workspace: LocalWorkspace, spec: CapabilitySpec) -> None:
        self.workspace, self.spec = workspace, spec

    def _identity(self, action: ActionRecord) -> dict[str, Any]:
        return {
            "schema": "workspace-receipt-v1",
            "action_id": action.id,
            "idempotency_key": action.idempotency_key,
            "arguments_digest": action.arguments_digest,
            "arguments": action.model_dump(mode="json")["arguments"],
            "goal_id": action.goal_id,
            "origin_principal_id": action.origin_principal_id,
            "binding_id": action.binding_id,
            "binding_version": action.binding_version,
            "capability_id": action.capability_id,
            "configuration_digest": self.workspace.configuration_digest,
        }

    def _observation(
        self,
        action: ActionRecord,
        outcome: str,
        *,
        reason: str = "",
        evidence: tuple[Ref, ...] = (),
        started: float | None = None,
        uncertain: bool = False,
    ) -> ActionObservation:
        elapsed = max(0.0, self.workspace.clock() - started) if started is not None else 0.0
        # Configured programs may incur unknown spending. Lookup is always local.
        cost = None if uncertain else 0
        return ActionObservation(
            action_id=action.id,
            expected_revision=action.revision,
            outcome=cast(Any, outcome),
            evidence=evidence,
            external_receipt=evidence[-1].id if evidence else None,
            explanation=reason,
            usage=Usage(
                status="unknown" if uncertain else "known",
                input_tokens=0,
                output_tokens=0,
                calls=1,
                elapsed_seconds=elapsed,
                cost_microunits=cost,
                source="local-workspace-v1",
            ),
        )

    def _valid_action(self, action: ActionRecord) -> bool:
        return (
            action.capability_id == self.spec.id
            and action.binding_version == self.spec.binding_version
            and action.binding_id == self.spec.binding_id
            and _digest(_json(action.model_dump(mode="json")["arguments"]))
            == action.arguments_digest
            and self.workspace._validate(self.spec.id, action.arguments)
        )

    def _receipts(self, action: ActionRecord) -> tuple[tuple[Ref, dict[str, Any]], ...]:
        matches = self.workspace._bridge().receipts(action)
        identity = self._identity(action)
        if any(any(data.get(k) != v for k, v in identity.items()) for _, data in matches):
            raise ValueError("receipt_binding_changed")
        return matches

    def _publish(self, action: ActionRecord, payload: dict[str, Any]) -> Ref:
        return self.workspace._bridge().publish(
            action, _json({**self._identity(action), **payload}), _RECEIPT
        )

    def _completed(
        self, action: ActionRecord, matches: tuple[tuple[Ref, dict[str, Any]], ...]
    ) -> ActionObservation:
        completed = [(ref, data) for ref, data in matches if data.get("phase") == "completed"]
        if len(completed) != 1:
            return self._observation(action, "needs_human", reason="execution_unproven")
        receipt, data = completed[0]
        # Read every referenced output through its current validity, ACL and digest.
        bridge = self.workspace._bridge()
        evidence = tuple(Ref.model_validate(r) for r in data.get("outputs", ()))
        if self.spec.id in {"write_text", "run_check"}:
            prepared_ref = Ref.model_validate(data["prepared_ref"])
            prepared = next(
                (d for r, d in matches if r == prepared_ref and d.get("phase") == "prepared"), None
            )
            if prepared is None:
                raise ValueError("incomplete_evidence")
            evidence = (*evidence, prepared_ref)
        principal = bridge._owner(action)
        indexed = {r.id: r for r in bridge.store.view().records}
        for ref in evidence:
            current = indexed.get(ref.id)
            if not isinstance(current, ArtifactRef) or reference(current) != ref:
                raise ValueError("stale_evidence")
            bridge.artifacts.read(current, principal=principal, goal_id=action.goal_id)
        if self.spec.id == "write_text" and data["outcome"] == "succeeded":
            replacement = bridge.content(
                action, Ref.model_validate(action.arguments["content_ref"])
            )
            if data["before_sha256"] != action.arguments["expected_sha256"] or data[
                "after_sha256"
            ] != _digest(replacement):
                raise ValueError("receipt_binding_changed")
            if (
                _digest(self.workspace._read(cast(str, action.arguments["path"]), write=True))
                != data["after_sha256"]
            ):
                return self._observation(
                    action, "needs_human", reason="resource_changed", evidence=(receipt,)
                )
        if self.spec.id == "run_check":
            criterion = self.workspace._criterion(action.arguments, action.goal_id)
            after = {p: _digest(self.workspace._read(p)) for p in criterion.workspace_inputs}
            if after != data["after_sha256"] or after != data["before_sha256"]:
                return self._observation(
                    action, "needs_human", reason="check_inputs_changed", evidence=(receipt,)
                )
        return self._observation(
            action, data["outcome"], reason=data["reason"], evidence=(*evidence, receipt)
        )

    async def reconcile(self, action: ActionRecord) -> ActionObservation:
        started = self.workspace.clock()
        async with self.workspace._lock:
            try:
                if not self._valid_action(action):
                    raise ValueError("invalid_action_arguments")
                return self._completed(action, self._receipts(action))
            except (ValueError, OSError, ExecutiveError, KeyError, TypeError):
                return self._observation(
                    action, "needs_human", reason="evidence_unavailable", started=started
                )

    async def cancel(self, action: ActionRecord) -> ActionObservation:
        # execute owns process-group cleanup. Cancellation alone proves no outcome.
        return self._observation(
            action, "outcome_unknown", reason="cancellation_unproven", uncertain=True
        )

    async def execute(self, action: ActionRecord) -> ActionObservation:
        started = self.workspace.clock()
        async with self.workspace._lock:
            if not self._valid_action(action):
                return self._observation(
                    action, "failed", reason="invalid_action_arguments", started=started
                )
            try:
                matches = self._receipts(action)
                if matches:
                    return self._completed(action, matches)
                if self.spec.id == "write_text":
                    return self._write(action, started)
                if self.spec.id == "run_check":
                    return await self._check(action, started)
                if self.spec.id == "read_text":
                    data = self.workspace._read(cast(str, action.arguments["path"]))
                    data.decode("utf-8")
                    output = self.workspace._bridge().publish(
                        action, data, "text/plain; charset=utf-8"
                    )
                else:
                    parts = self.workspace._allowed(
                        cast(str, action.arguments["path"]), directory=True
                    )
                    with self.workspace._directory(parts) as directory:
                        before = _signature(os.fstat(directory))
                        names: list[str] = []
                        # scandir is incremental; listdir would allocate an unbounded list.
                        with os.scandir(directory) as entries:
                            for entry in entries:
                                if len(names) >= self.workspace.max_entries:
                                    raise ValueError("too_many_entries")
                                info = entry.stat(follow_symlinks=False)
                                if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                                    raise ValueError("invalid_file")
                                entry.name.encode("utf-8")
                                names.append(entry.name)
                        self.workspace._anchor(parts, directory)
                        if before != _signature(os.fstat(directory)):
                            raise ValueError("resource_changed")
                    data = _json(sorted(names))
                    output = self.workspace._bridge().publish(action, data, "application/json")
                receipt = self._publish(
                    action,
                    {
                        "phase": "completed",
                        "outcome": "succeeded",
                        "reason": "",
                        "outputs": [output.model_dump(mode="json")],
                    },
                )
                return self._observation(
                    action, "succeeded", evidence=(output, receipt), started=started
                )
            except (ValueError, OSError, ExecutiveError, UnicodeError) as exc:
                # Never expose raw OS paths, command diagnostics or artifact content.
                reason = (
                    str(exc)
                    if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc))
                    else "workspace_unavailable"
                )
                return self._observation(
                    action,
                    "failed" if self.spec.effect_class == "read_only" else "outcome_unknown",
                    reason=reason,
                    started=started,
                    uncertain=self.spec.effect_class != "read_only",
                )

    def _preimage(self, parent: int, name: str) -> tuple[str, tuple[int, ...] | None, int]:
        try:
            data, signature = self.workspace._read_at(parent, name)
            return _digest(data), signature, stat.S_IMODE(signature[2])
        except FileNotFoundError:
            return _MISSING, None, 0o600

    def _write(self, action: ActionRecord, started: float) -> ActionObservation:
        workspace = self.workspace
        parts = workspace._allowed(cast(str, action.arguments["path"]), write=True)
        replacement = workspace._bridge().content(
            action, Ref.model_validate(action.arguments["content_ref"])
        )
        replacement.decode("utf-8")
        if len(replacement) > workspace.max_file_bytes:
            raise ValueError("file_too_large")
        with workspace._directory(parts[:-1]) as parent:
            expected, signature, mode = self._preimage(parent, parts[-1])
            if expected != action.arguments["expected_sha256"]:
                return self._observation(
                    action, "failed", reason="resource_changed", started=started
                )
            temporary = ".sgp-write-" + uuid4().hex
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=parent,
            )
            replaced = False
            try:
                with os.fdopen(descriptor, "wb") as file:
                    file.write(replacement)
                    file.flush()
                    # Preserve ordinary access bits, never setuid/setgid/sticky bits.
                    os.fchmod(file.fileno(), mode & 0o777)
                    os.fsync(file.fileno())
                    temporary_signature = _signature(os.fstat(file.fileno()))
                prepared = self._publish(
                    action,
                    {
                        "phase": "prepared",
                        "before_sha256": expected,
                        "after_sha256": _digest(replacement),
                    },
                )
                workspace._anchor(parts[:-1], parent)
                current, current_signature, _ = self._preimage(parent, parts[-1])
                if current != expected or current_signature != signature:
                    return self._observation(
                        action,
                        "failed",
                        reason="resource_changed",
                        evidence=(prepared,),
                        started=started,
                    )
                staged, staged_signature = workspace._read_at(parent, temporary)
                if staged != replacement or staged_signature != temporary_signature:
                    raise ValueError("resource_changed")
                os.replace(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
                replaced = True
                os.fsync(parent)
                workspace._anchor(parts[:-1], parent)
                after, after_signature = workspace._read_at(parent, parts[-1])
                # Rename may change ctime, but a foreign same-byte replacement
                # still has a different descriptor identity or other metadata.
                if after != replacement or after_signature[:-1] != temporary_signature[:-1]:
                    raise ValueError("resource_changed")
                receipt = self._publish(
                    action,
                    {
                        "phase": "completed",
                        "outcome": "succeeded",
                        "reason": "",
                        "before_sha256": expected,
                        "after_sha256": _digest(after),
                        "prepared_ref": prepared.model_dump(mode="json"),
                        "outputs": [],
                    },
                )
                return self._observation(
                    action, "succeeded", evidence=(prepared, receipt), started=started
                )
            finally:
                if not replaced:
                    try:
                        os.unlink(temporary, dir_fd=parent)
                    except FileNotFoundError:
                        pass

    async def _check(self, action: ActionRecord, started: float) -> ActionObservation:
        workspace = self.workspace
        criterion_ref = Ref.model_validate(action.arguments["criterion_ref"])
        criterion = workspace._criterion(action.arguments, action.goal_id)
        config = workspace.checks[cast(str, action.arguments["command_id"])]
        before = {p: _digest(workspace._read(p)) for p in criterion.workspace_inputs}
        if before != action.arguments["expected_sha256"]:
            return self._observation(action, "failed", reason="resource_changed", started=started)
        prepared = self._publish(
            action,
            {
                "phase": "prepared",
                "criterion_ref": criterion_ref.model_dump(mode="json"),
                "argv": list(config.argv),
                "check_configuration": config.model_dump(mode="json"),
                "launcher": _LAUNCHER_VERSION,
                "before_sha256": before,
            },
        )
        if {p: _digest(workspace._read(p)) for p in criterion.workspace_inputs} != before:
            return self._observation(
                action, "failed", reason="resource_changed", evidence=(prepared,), started=started
            )
        workspace._criterion(action.arguments, action.goal_id)
        output = bytearray()
        overflow = asyncio.Event()
        process = None
        tasks: list[asyncio.Task[Any]] = []
        reason = ""
        returncode = None
        group_stopped = False
        group_cleanup = "not_started"

        def kill() -> bool:
            nonlocal group_stopped, group_cleanup
            if group_stopped:
                return True
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    group_cleanup = "signal_delivered"
                except ProcessLookupError:
                    group_cleanup = "group_absent"
                except OSError:
                    # macOS can report EPERM while a dead leader awaits reaping.
                    # Recheck after wait; persistent cleanup failure stays unknown.
                    group_cleanup = "unknown"
                    return False
                group_stopped = True
            return True

        async def pump(stream: asyncio.StreamReader | None) -> None:
            assert stream is not None
            while chunk := await stream.read(8192):
                remaining = config.max_output_bytes - len(output)
                output.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()
                    kill()

        try:
            with workspace._root() as root:
                # cwd points to the verified open directory, not a raced pathname.
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _LAUNCHER,
                    str(root),
                    *config.argv,
                    pass_fds=(root,),
                    env={},
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )

            async def send_environment() -> None:
                assert process is not None and process.stdin is not None
                # Pass exact original values privately as pipe data, not argv or
                # the launcher's Python-startup-mutated os.environ. EOF frames it.
                process.stdin.write(_json(dict(config.environment)))
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()

            async def wait_process() -> int:
                nonlocal reason
                assert process is not None
                # asyncio's wait may also wait for descendant-held pipe EOF.
                # Observe leader exit independently, then kill its remaining group.
                while process.returncode is None:
                    await asyncio.sleep(0.005)
                code = process.returncode
                kill()  # Reap descendants retaining pipes after the leader exits.
                if group_cleanup == "signal_delivered":
                    reason = "check_descendants_terminated"
                await process.wait()
                return code

            tasks = [
                asyncio.create_task(wait_process()),
                asyncio.create_task(pump(process.stdout)),
                asyncio.create_task(pump(process.stderr)),
                asyncio.create_task(send_environment()),
            ]
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=config.timeout_seconds)
            returncode = results[0]
            if overflow.is_set():
                reason = "output_limit_exceeded"
        except asyncio.TimeoutError:
            reason = "check_timeout"
        except asyncio.CancelledError:
            kill()
            raise
        except OSError:
            reason = "check_start_or_cleanup_unknown"
        finally:
            kill()
            for task in tasks:
                if not task.done():
                    task.cancel()
            if process is not None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    reason = "check_cleanup_unknown"
                if not kill():
                    reason = "check_cleanup_unknown"
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        after = {p: _digest(workspace._read(p)) for p in criterion.workspace_inputs}
        workspace._criterion(action.arguments, action.goal_id)
        unknown = bool(reason)
        if not reason and after != before:
            reason = "check_inputs_changed"
        outcome = (
            "outcome_unknown" if unknown else "failed" if reason or returncode != 0 else "succeeded"
        )
        output_ref = workspace._bridge().publish(action, bytes(output), "application/octet-stream")
        receipt = self._publish(
            action,
            {
                "phase": "completed",
                "outcome": outcome,
                "reason": reason,
                "prepared_ref": prepared.model_dump(mode="json"),
                "criterion_ref": criterion_ref.model_dump(mode="json"),
                "argv": list(config.argv),
                "check_configuration": config.model_dump(mode="json"),
                "launcher": _LAUNCHER_VERSION,
                "before_sha256": before,
                "after_sha256": after,
                "returncode": returncode,
                "output_truncated": overflow.is_set(),
                "process_group_cleanup": group_cleanup,
                "outputs": [output_ref.model_dump(mode="json")],
            },
        )
        observation = self._observation(
            action,
            outcome,
            reason=reason,
            evidence=(output_ref, prepared, receipt),
            started=started,
            uncertain=unknown or config.maximum_charge_microunits != 0,
        )
        return observation
