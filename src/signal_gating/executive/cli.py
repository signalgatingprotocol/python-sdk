"""Foreground JSONL operator controls. No dynamic imports or implicit live calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import threading
from collections.abc import AsyncIterator, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from .actions import CapabilityRegistry
from .adapters.openai_worker import OpenAIWorker
from .config import KernelConfig
from .context import UTF8ByteCounter
from .errors import ExecutiveError, StoreError
from .inspection import inspect_state, operator_view
from .kernel import Kernel, SystemClock
from .messages import AuthenticatedCommand, Command, CreateGoal, EpisodeResult, ResumeGoal
from .records import ContextPacket, DecisionReceipt, Goal, KernelView, Limits, Usage, WakeCondition
from .store import DATABASE_NAME, Store
from .workers import Worker

_COMMAND = TypeAdapter(Command)
_DIMENSIONS = ("input_tokens", "output_tokens", "calls", "elapsed_seconds", "cost_microunits")
_SECRETS: ContextVar[tuple[str, ...]] = ContextVar("executive_credentials", default=())


class OperatorInputError(ValueError):
    """Safe predefined diagnostic, never an input or provider exception echo."""


def _error(exc: Exception, *, started: bool = False) -> int:
    corrupt = isinstance(exc, StoreError) and exc.code in {"corrupt_store", "invalid_journal"}
    cause = exc.__cause__
    if isinstance(exc, StoreError) and isinstance(cause, sqlite3.DatabaseError):
        sqlite_code = getattr(cause, "sqlite_errorcode", None)
        if isinstance(sqlite_code, int):
            # Extended SQLite result codes retain their primary code in one byte.
            corrupt = corrupt or (sqlite_code & 0xFF) in {
                getattr(sqlite3, "SQLITE_CORRUPT", 11),
                getattr(sqlite3, "SQLITE_NOTADB", 26),
            }
        else:
            # Python 3.10 has no sqlite_errorcode. Match only SQLite's exact
            # corruption diagnostics, never generic I/O errors or arbitrary text.
            corrupt = corrupt or str(cause) in {
                "database disk image is malformed",
                "file is not a database",
            }
    code = 4 if started or corrupt else 2
    details: dict[str, Any] = {"detail": str(exc)} if isinstance(exc, OperatorInputError) else {}
    if isinstance(exc, ExecutiveError):
        details["error_code"] = exc.code
    if isinstance(exc, ValidationError):
        # Locations can themselves contain arbitrary user-supplied mapping keys.
        details["validation_errors"] = exc.error_count()
    _emit("error", reason_code="runtime_error" if code == 4 else "startup_error", **details)
    return code


def load_config(path: Path) -> KernelConfig:
    """Read closed JSON config; relative workspace roots resolve beside that file."""
    config = KernelConfig.model_validate_json(path.read_text(encoding="utf-8"))
    if config.workspace is not None:
        workspace = config.workspace
        config = config.model_copy(
            update={
                "workspace": workspace.model_copy(
                    update={
                        "root": (path.resolve().parent / workspace.root).resolve(),
                    }
                )
            }
        )
    return config


class FixtureWorker:
    """Synthetic, free diagnostic worker. It deliberately requests owner input."""

    async def run_episode(self, packet: ContextPacket) -> EpisodeResult:
        return EpisodeResult(
            stop_reason="blocked",
            blocker=WakeCondition(kind="human_decision", reason_code="fixture_owner_needed"),
            resume_brief="Synthetic fixture stopped. No task outcome was produced.",
            usage=Usage.zero(currency=packet.remaining_limits.currency),
        )


def _emit(
    status: str,
    *,
    command_id: str | None = None,
    receipt: DecisionReceipt | None = None,
    view: KernelView | None = None,
    **details: Any,
) -> None:
    value = operator_view(view) if view is not None else {}
    value.update(
        command_id=command_id,
        receipt=receipt.model_dump(mode="json") if receipt is not None else None,
        status=status,
        **details,
    )
    print(json.dumps(_redact(value), sort_keys=True, ensure_ascii=True), flush=True)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        for secret in _SECRETS.get():
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {_redact(key): _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _goal(path: Path) -> CreateGoal:
    value = json.loads(path.read_text(encoding="utf-8"))
    # A complete CreateGoal preserves explicit task/commitment IDs. A bare Goal
    # is convenient for operators and receives its generated ID exactly once.
    return (
        CreateGoal.model_validate(value)
        if value.get("kind") == "create_goal"
        else CreateGoal(goal=Goal.model_validate(value))
    )


def _explicit(limits: Limits) -> None:
    if any(getattr(limits, name) is None for name in _DIMENSIONS):
        raise OperatorInputError(
            "Live execution requires input_tokens, output_tokens, calls, "
            "elapsed_seconds and cost_microunits ceilings."
        )


def _preflight(config: KernelConfig, live: bool, goal: Goal) -> None:
    remote = [worker for worker in config.workers if worker.kind == "openai"]
    if not remote:
        return
    if not live:
        raise OperatorInputError("OpenAI workers require --live.")
    _explicit(goal.limits)
    for worker in remote:
        if worker.prices is None or worker.prices.maximum_charge_microunits is None:
            raise OperatorInputError(
                "Each OpenAI worker requires prices.version and prices.maximum_charge_microunits."
            )
        if worker.prices.currency != goal.limits.currency:
            raise OperatorInputError("Live quote and budget currencies must agree.")


def _workers(config: KernelConfig, result: dict[str, Worker]) -> None:
    for worker in config.workers:
        if worker.kind == "fixture":
            result[worker.id] = FixtureWorker()
            continue
        assert worker.prices is not None and worker.api_key_env is not None
        assert worker.model is not None
        result[worker.id] = OpenAIWorker.from_environment(
            api_key_env=worker.api_key_env,
            model=worker.model,
            counter=UTF8ByteCounter(),
            capacity=worker.context_token_limit,
            quote=Limits(
                input_tokens=worker.context_token_limit,
                output_tokens=worker.max_output_tokens,
                calls=1,
                elapsed_seconds=config.episode_timeout_seconds,
                cost_microunits=worker.prices.maximum_charge_microunits,
                currency=worker.prices.currency,
            ),
            quote_version=worker.prices.version,
            quote_source="operator-configuration",
        )


async def _close_workers(workers: dict[str, Worker]) -> None:
    failed = False
    for worker in workers.values():
        if isinstance(worker, OpenAIWorker):
            close = getattr(worker.client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    failed = True
    if failed:
        raise RuntimeError("Provider transport close failed")


async def _commands(
    kernel: Kernel, goal_id: str, config: KernelConfig, live: bool
) -> AsyncIterator[AuthenticatedCommand]:
    loop = asyncio.get_running_loop()
    lines: asyncio.Queue[str | None] = asyncio.Queue()

    def receive() -> None:
        try:
            for line in sys.stdin:
                loop.call_soon_threadsafe(lines.put_nowait, line)
        finally:
            try:
                loop.call_soon_threadsafe(lines.put_nowait, None)
            except RuntimeError:
                pass  # foreground service already stopped

    threading.Thread(target=receive, name="executive-stdin", daemon=True).start()
    while True:
        line = await lines.get()
        if line is None:
            return
        command_id = None
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Expected tagged object")
            if _redact(value) != value:
                raise OperatorInputError("Credential values must not enter operator commands.")
            candidate = value.get("id")
            if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{32}", candidate):
                command_id = candidate
            if value.get("kind") in {"inspect", "stop"}:
                if set(value) - {"kind", "id"}:
                    raise ValueError("Unknown control field")
                command_id = command_id or uuid4().hex
                if value["kind"] == "stop":
                    _emit("stopping", command_id=command_id, view=kernel.inspect(goal_id))
                    return
                _emit("inspect", command_id=command_id, view=kernel.inspect(goal_id))
                continue
            command = _COMMAND.validate_python(value)
            command_id = command.id
            if isinstance(command, CreateGoal):
                _preflight(config, live, command.goal)
            elif command.kind == "revise_goal" and command.limits is not None:
                if any(w.kind == "openai" for w in config.workers):
                    _explicit(command.limits)
            yield AuthenticatedCommand(command=command, principal=kernel.owner)
        except (ValueError, ValidationError, TypeError):
            _emit(
                "invalid_command",
                command_id=command_id,
                reason_code="invalid_command",
                view=kernel.inspect(goal_id),
            )


async def _run(args: argparse.Namespace) -> int:
    workers: dict[str, Worker] = {}
    failure: BaseException | None = None
    code = 4
    try:
        code = await _run_session(args, workers)
    except BaseException as exc:
        failure = exc
    try:
        await _close_workers(workers)
    except Exception:
        _emit("error", reason_code="runtime_error", detail="Provider transport close failed.")
        return 4
    if failure is not None:
        raise failure
    return code


async def _run_session(args: argparse.Namespace, workers: dict[str, Worker]) -> int:
    started = False
    try:
        config = load_config(args.config)
        creation = _goal(args.goal) if args.goal else None
        if creation is not None:
            target = creation.goal
        else:
            with Store.open_readonly(args.store) as store:
                persisted = next(
                    (
                        r
                        for r in store.view().records
                        if isinstance(r, Goal) and r.id == args.resume
                    ),
                    None,
                )
            if persisted is None or persisted.status in {"cancelled", "failed"}:
                raise OperatorInputError(
                    "Resume requires an existing goal. Cancelled or failed goals "
                    "require a new goal."
                )
            target = persisted
        _preflight(config, args.live, target)
        if (args.store / DATABASE_NAME).exists():
            with Store.open_readonly(args.store) as snapshot:
                for record in snapshot.view().records:
                    if isinstance(record, Goal) and record.status not in {"cancelled", "failed"}:
                        _preflight(config, args.live, record)
        # Only explicit live execution may read the named credential references.
        # Keep them out of all JSONL fields, including unexpected echoed content.
        _SECRETS.set(
            tuple(
                value
                for worker in config.workers
                if worker.kind == "openai" and worker.api_key_env is not None
                if (value := os.environ.get(worker.api_key_env))
            )
        )
        definition = (creation or target).model_dump(mode="json")
        if _redact(definition) != definition:
            raise OperatorInputError("Credential values must not enter goal definitions.")
        _workers(config, workers)
        async with Kernel.open(
            args.store,
            config=config,
            workers=workers,
            capabilities=CapabilityRegistry(),
            verifiers={},
            clock=SystemClock(),
        ) as kernel:
            # Re-read the goal from this writer after recovery, not the preflight
            # snapshot. Resume is always the explicit owner control transition.
            if creation is None:
                target = next(
                    r
                    for r in kernel.inspect().state.records
                    if isinstance(r, Goal) and r.id == target.id
                )
            started = True
            command: Command | None = creation
            if creation is None and target.status == "paused":
                command = ResumeGoal(goal_id=target.id, expected_revision=target.revision)
            if command is not None:
                receipt = await kernel.submit(command, principal=kernel.owner)
                _emit(
                    receipt.disposition,
                    command_id=command.id,
                    receipt=receipt,
                    view=kernel.inspect(target.id),
                    worker_principals={
                        name: principal.id for name, principal in kernel.worker_principals.items()
                    },
                    verifier_principal_id=kernel.verifier_principal.id,
                )
                if receipt.disposition != "accept":
                    return 2
            else:
                _emit(
                    "loaded",
                    view=kernel.inspect(target.id),
                    worker_principals={
                        name: principal.id for name, principal in kernel.worker_principals.items()
                    },
                    verifier_principal_id=kernel.verifier_principal.id,
                )
            if args.batch:
                await kernel.run_until_idle()
                view = kernel.inspect(target.id)
                _emit("idle", view=view)
                return 0 if target.id in view.summary.completed_goal_ids else 3

            def response(item: AuthenticatedCommand, ack: DecisionReceipt) -> None:
                _emit(
                    ack.disposition,
                    command_id=item.command.id,
                    receipt=ack,
                    view=kernel.inspect(target.id),
                )

            await kernel.serve(
                _commands(kernel, target.id, config, args.live),
                on_response=response,
                on_idle=lambda _: _emit("idle", view=kernel.inspect(target.id)),
            )
            _emit("stopped", view=kernel.inspect(target.id))
            return 0
    except Exception as exc:
        return _error(exc, started=started)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("init", "run", "inspect", "export", "demo"):
        command = subparsers.add_parser(name)
        command.add_argument("--store", required=True, type=Path)
        if name == "run":
            command.add_argument("--config", required=True, type=Path)
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--goal", type=Path)
            source.add_argument("--resume")
            command.add_argument("--live", action="store_true")
            command.add_argument("--batch", action="store_true")
        if name in {"inspect", "export"}:
            command.add_argument("--goal", required=name == "export")
        if name == "inspect":
            command.add_argument("--json", action="store_true")
        if name == "export":
            command.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a foreground operator command, with stable process exit codes."""
    try:
        args = _parser().parse_args(argv)
        if args.operation in {"export", "demo"}:
            _emit(
                "unavailable",
                reason_code=f"{args.operation}_unavailable",
                detail="Integration is supplied by Task "
                + ("18" if args.operation == "export" else "16"),
            )
            return 2
        if args.operation == "run":
            return asyncio.run(_run(args))
        if args.operation == "init":
            with Store.open(args.store) as store:
                _emit("initialized", view=inspect_state(store.view()))
        else:
            with Store.open_readonly(args.store) as store:
                view = inspect_state(store.view(), args.goal)
                if args.goal is not None and not any(
                    isinstance(r, Goal) and r.id == args.goal for r in view.state.records
                ):
                    raise ValueError("Unknown goal")
                _emit("inspect", view=view)
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        return _error(exc)
