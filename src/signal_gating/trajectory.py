"""Trajectory capture: verifiable structured records of mesh execution."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from inspect import isawaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from signal_gating.errors import MeshError, SignalSerializationError
from signal_gating.signal import Signal

if TYPE_CHECKING:
    from signal_gating.mesh import Mesh

ReceiptFilter: TypeAlias = str | Iterable[str] | None
SignalTypeSelector: TypeAlias = str | type[Signal]
SignalTypeFilter: TypeAlias = SignalTypeSelector | Iterable[SignalTypeSelector] | None

# Base Signal envelope fields -- excluded from a Receipt's domain payload because
# they are already represented as typed Receipt fields.
_ENVELOPE_FIELDS = frozenset(
    {
        "id",
        "source",
        "timestamp",
        "priority",
        "trace_id",
        "correlation_id",
        "parent_id",
        "metadata",
    }
)


def _domain_payload(signal: Signal) -> dict[str, Any]:
    return {k: v for k, v in signal.model_dump().items() if k not in _ENVELOPE_FIELDS}


def _digest(core: dict[str, Any]) -> str:
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _receipt_mismatch_error(receipt: Receipt) -> SignalSerializationError:
    return SignalSerializationError(
        "trajectory receipt digest mismatch for "
        f"{receipt.signal_type} {receipt.signal_id!r} "
        f"on trace {receipt.trace_id!r}"
    )


def _filter_values(values: ReceiptFilter) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return set(values)


def _signal_type_value(value: SignalTypeSelector) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, type) and issubclass(value, Signal):
        return value.wire_type()
    raise TypeError("signal type filters must be strings or Signal subclasses")


def _signal_type_filter_values(values: SignalTypeFilter) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    if isinstance(values, type) and issubclass(values, Signal):
        return {values.wire_type()}
    return {_signal_type_value(value) for value in values}


def _receipt_matches(
    receipt: Receipt,
    *,
    event_kind_filter: set[str] | None = None,
    action_filter: set[str] | None = None,
    signal_type_filter: set[str] | None = None,
) -> bool:
    return (
        (event_kind_filter is None or receipt.event_kind in event_kind_filter)
        and (action_filter is None or receipt.action in action_filter)
        and (signal_type_filter is None or receipt.signal_type in signal_type_filter)
    )


@dataclass(frozen=True, slots=True)
class Receipt:
    """A verifiable, structured record of one mesh execution event.

    A Receipt serves two purposes that used to pull in different directions:

    * **Audit** — ``event_kind``/``action`` plus ``signal_type`` and ``payload``
      are a human-readable projection of what happened.
    * **Replay** — ``wire`` is the full, self-describing wire envelope, so
      ``to_signal()`` reconstructs the *exact* original typed signal. Without it
      a trajectory could be read but never re-run; with it a persisted
      trajectory replays as typed signals after a restart, the same way the
      dead-letter queue does.

    The ``digest`` covers both, so a tampered payload *or* a tampered wire
    envelope is detectable — reconstruction is as trustworthy as the audit view.
    """

    trace_id: str
    signal_id: str
    parent_id: str
    signal_type: str
    source: str
    target: str
    priority: int
    timestamp: float
    payload: dict[str, Any]
    wire: dict[str, Any]
    digest: str
    event_kind: str = "signal"
    action: str = "hop"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_signal(
        cls,
        signal: Signal,
        source: str,
        target: str,
        *,
        event_kind: str = "signal",
        action: str = "hop",
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> Receipt:
        core: dict[str, Any] = {
            "trace_id": signal.trace_id,
            "signal_id": signal.id,
            "parent_id": signal.parent_id,
            "signal_type": signal.wire_type(),
            "source": source,
            "target": target,
            "priority": signal.priority,
            "timestamp": time.time() if timestamp is None else timestamp,
            "payload": _domain_payload(signal),
            "wire": signal.to_wire(),
            "event_kind": event_kind,
            "action": action,
            "metadata": metadata or {},
        }
        return cls(
            trace_id=core["trace_id"],
            signal_id=core["signal_id"],
            parent_id=core["parent_id"],
            signal_type=core["signal_type"],
            source=core["source"],
            target=core["target"],
            priority=core["priority"],
            timestamp=core["timestamp"],
            payload=core["payload"],
            wire=core["wire"],
            digest=_digest(core),
            event_kind=core["event_kind"],
            action=core["action"],
            metadata=core["metadata"],
        )

    @classmethod
    def from_event(cls, event: Any) -> Receipt:
        """Build a Receipt from a mesh event object.

        ``MeshEvent`` lives in ``mesh.py`` to avoid making mesh depend on
        trajectory capture. This method accepts any object with the same
        attributes, which keeps recorder sinks lightweight and testable.
        """
        return cls.from_signal(
            event.signal,
            source=event.source,
            target=event.target,
            event_kind=event.event_kind,
            action=event.action,
            metadata=dict(event.metadata),
            timestamp=event.timestamp,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Receipt:
        """Rebuild a Receipt from a :meth:`to_dict` mapping (e.g. a JSONL line).

        The inverse of ``to_dict``; the reconstructed Receipt carries the same
        ``digest`` and so still ``verify()``-ies.
        """
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def to_signal(self, *, strict: bool = True) -> Signal:
        """Reconstruct the original typed signal from the stored wire envelope.

        Same contract as :meth:`Signal.from_wire`: the signal's class must be
        imported (and therefore registered). With ``strict=False`` an unknown
        type degrades to a best-effort base ``Signal`` instead of raising
        ``UnknownSignalType``.
        """
        return Signal.from_wire(self.wire, strict=strict)

    def verify(self) -> bool:
        core = {k: v for k, v in asdict(self).items() if k != "digest"}
        if _digest(core) == self.digest:
            return True
        if self.event_kind == "signal" and self.action == "hop" and not self.metadata:
            legacy_core = {
                k: v for k, v in core.items()
                if k not in {"event_kind", "action", "metadata"}
            }
            return _digest(legacy_core) == self.digest
        return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrajectoryRecorder:
    """Records verifiable Receipts for mesh events and signal hops.

    Attach with ``mesh.record(recorder)`` to capture direct orchestration events
    and connected routes. ``mesh.intercept(recorder)`` remains supported for the
    old edge-hop-only observer path.

        recorder = TrajectoryRecorder()
        mesh.record(recorder)
        ...
        recorder.export_jsonl("runs.jsonl")
    """

    def __init__(self) -> None:
        self._receipts: list[Receipt] = []

    def __call__(self, signal: Signal, source: str, target: str) -> Signal:
        self._receipts.append(Receipt.from_signal(signal, source, target))
        return signal

    def record_event(self, event: Any) -> None:
        """Record a structured mesh event emitted by ``Mesh.record()``."""
        self._receipts.append(Receipt.from_event(event))

    @property
    def receipts(self) -> list[Receipt]:
        return list(self._receipts)

    def filter_receipts(
        self,
        *,
        event_kinds: ReceiptFilter = None,
        actions: ReceiptFilter = None,
        signal_types: SignalTypeFilter = None,
        verify: bool = False,
    ) -> list[Receipt]:
        """Return retained receipts matching all supplied audit filters.

        Each filter accepts either one string or an iterable of strings. With
        ``verify=True``, matched receipts are digest-checked before they are
        returned.
        """
        event_kind_filter = _filter_values(event_kinds)
        action_filter = _filter_values(actions)
        signal_type_filter = _signal_type_filter_values(signal_types)
        receipts = [
            receipt
            for receipt in self._receipts
            if _receipt_matches(
                receipt,
                event_kind_filter=event_kind_filter,
                action_filter=action_filter,
                signal_type_filter=signal_type_filter,
            )
        ]
        if verify:
            for receipt in receipts:
                if not receipt.verify():
                    raise _receipt_mismatch_error(receipt)
        return receipts

    def trajectories(self) -> dict[str, list[Receipt]]:
        """Receipts grouped by trace_id, preserving capture order within each run."""
        grouped: dict[str, list[Receipt]] = {}
        for r in self._receipts:
            grouped.setdefault(r.trace_id, []).append(r)
        return grouped

    def export_jsonl(
        self,
        path: str | Path,
        *,
        event_kinds: ReceiptFilter = None,
        actions: ReceiptFilter = None,
        signal_types: SignalTypeFilter = None,
        verify: bool = False,
    ) -> int:
        """Write all receipts as JSON Lines. Returns the number written.

        Each line carries the full wire envelope, so :meth:`load_jsonl` reloads a
        verifiable trajectory and :meth:`replay` reconstructs the typed signals.
        Optional filters export only matching receipts. With ``verify=True``,
        selected receipts are digest-checked before the output file is opened.
        """
        receipts = self.filter_receipts(
            event_kinds=event_kinds,
            actions=actions,
            signal_types=signal_types,
            verify=verify,
        )
        lines = [json.dumps(r.to_dict(), default=str) + "\n" for r in receipts]
        out = Path(path)
        with out.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        return len(lines)

    def load_jsonl(self, path: str | Path, *, verify: bool = True) -> int:
        """Load receipts from a file written by :meth:`export_jsonl`, appending.

        The durability half of the trajectory story: persist a run, survive a
        restart, then reload it as verifiable receipts you can inspect or
        ``replay`` as typed signals. By default, each receipt's digest is
        verified before it is accepted. Returns the number of receipts loaded.
        """
        src = Path(path)
        receipts: list[Receipt] = []
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                receipt = Receipt.from_dict(json.loads(line))
                if verify and not receipt.verify():
                    raise _receipt_mismatch_error(receipt)
                receipts.append(receipt)
        self._receipts.extend(receipts)
        return len(receipts)

    def replay(
        self,
        *,
        strict: bool = True,
        verify: bool = True,
        event_kinds: ReceiptFilter = None,
        actions: ReceiptFilter = None,
        signal_types: SignalTypeFilter = None,
    ) -> list[Signal]:
        """Reconstruct retained signals as their original types, in capture order.

        The faithful counterpart to :meth:`export_jsonl`: a trajectory read off
        disk comes back as the exact typed signals that produced it, ready to
        re-run, audit, or learn from. Import the modules defining your signal
        types first so they are registered (see :meth:`Signal.from_wire`); with
        ``strict=False`` an unknown type degrades to a base ``Signal`` rather
        than raising. Optional filters restrict reconstruction to matching
        receipts. By default, every selected receipt digest is verified before
        any signal is reconstructed.
        """
        receipts = self.filter_receipts(
            event_kinds=event_kinds,
            actions=actions,
            signal_types=signal_types,
            verify=verify,
        )
        return [r.to_signal(strict=strict) for r in receipts]

    async def replay_into(
        self,
        mesh: Mesh,
        *,
        actions: ReceiptFilter = None,
        event_kinds: ReceiptFilter = None,
        signal_types: SignalTypeFilter = None,
        strict: bool = True,
        verify: bool = True,
        strict_targets: bool = True,
    ) -> ReplayResult:
        """Replay retained delivery entries into a mesh."""
        return await TrajectoryReplayRunner.from_recorder(self).replay_into(
            mesh,
            actions=actions,
            event_kinds=event_kinds,
            signal_types=signal_types,
            strict=strict,
            verify=verify,
            strict_targets=strict_targets,
        )

    def clear(self) -> None:
        self._receipts.clear()


@dataclass(frozen=True, slots=True)
class ReplayDelivery:
    """Per-receipt replay disposition."""

    receipt_index: int
    action: str
    trace_id: str
    signal_id: str
    signal_type: str
    target: str
    status: Literal["delivered", "skipped", "failed"]
    reason: str = ""


@dataclass(slots=True)
class ReplayResult:
    """Summary of a delivery replay into a mesh."""

    attempted: int = 0
    delivered: int = 0
    skipped: int = 0
    failed: int = 0
    missing_targets: list[str] = field(default_factory=list)
    receipts: list[Receipt] = field(default_factory=list)
    deliveries: list[ReplayDelivery] = field(default_factory=list)

    @property
    def actions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for delivery in self.deliveries:
            counts[delivery.action] = counts.get(delivery.action, 0) + 1
        return counts

    @property
    def trace_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(delivery.trace_id for delivery in self.deliveries))


class TrajectoryReplayRunner:
    """Replay recorded mesh delivery entries into a fresh mesh.

    This is execution replay's first honest layer: it re-delivers recorded entry
    signals (`inject`, `request_sent`, fan-out sends, race sends, and pub/sub
    deliveries) so handlers run again. It does not recreate pending request
    futures, workflow control flow, or external LLM sessions.
    """

    replayable_actions = frozenset(
        {
            "inject",
            "request_sent",
            "scatter_sent",
            "race_sent",
            "published",
        }
    )

    def __init__(self, receipts: Sequence[Receipt]) -> None:
        self._receipts = list(receipts)

    @classmethod
    def from_recorder(cls, recorder: TrajectoryRecorder) -> TrajectoryReplayRunner:
        """Build a runner from a recorder's retained receipts."""
        return cls(recorder.receipts)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        verify: bool = True,
    ) -> TrajectoryReplayRunner:
        """Load receipts from JSONL and build a replay runner."""
        recorder = TrajectoryRecorder()
        recorder.load_jsonl(path, verify=verify)
        return cls.from_recorder(recorder)

    @property
    def receipts(self) -> list[Receipt]:
        return list(self._receipts)

    def replayable_receipts(
        self,
        *,
        actions: ReceiptFilter = None,
        event_kinds: ReceiptFilter = None,
        signal_types: SignalTypeFilter = None,
    ) -> list[Receipt]:
        """Return receipts this runner can deliver into a mesh."""
        action_filter = _filter_values(actions)
        allowed = set(self.replayable_actions) if action_filter is None else action_filter
        event_kind_filter = _filter_values(event_kinds)
        signal_type_filter = _signal_type_filter_values(signal_types)
        return [
            receipt
            for receipt in self._receipts
            if receipt.action in allowed
            and _receipt_matches(
                receipt,
                event_kind_filter=event_kind_filter,
                signal_type_filter=signal_type_filter,
            )
        ]

    async def replay_into(
        self,
        mesh: Mesh,
        *,
        actions: ReceiptFilter = None,
        event_kinds: ReceiptFilter = None,
        signal_types: SignalTypeFilter = None,
        strict: bool = True,
        verify: bool = True,
        strict_targets: bool = True,
    ) -> ReplayResult:
        """Deliver replayable receipt signals into ``mesh``.

        Args:
            mesh: The mesh with agents already registered and running.
            actions: Optional subset of replayable actions to deliver.
            event_kinds: Optional event namespaces to include.
            signal_types: Optional stable signal wire types to include.
            strict: Passed to :meth:`Receipt.to_signal`.
            verify: Verify all retained receipts before any delivery.
            strict_targets: Raise ``MeshError`` for missing targets. With
                ``False``, missing targets are recorded in the result and skipped.
        """
        if verify:
            for receipt in self._receipts:
                if not receipt.verify():
                    raise _receipt_mismatch_error(receipt)

        action_filter = _filter_values(actions)
        allowed_actions = (
            set(self.replayable_actions) if action_filter is None else action_filter
        )
        event_kind_filter = _filter_values(event_kinds)
        signal_type_filter = _signal_type_filter_values(signal_types)
        result = ReplayResult()
        for index, receipt in enumerate(self._receipts):
            if not _receipt_matches(
                receipt,
                event_kind_filter=event_kind_filter,
                signal_type_filter=signal_type_filter,
            ):
                result.skipped += 1
                result.deliveries.append(
                    ReplayDelivery(
                        receipt_index=index,
                        action=receipt.action,
                        trace_id=receipt.trace_id,
                        signal_id=receipt.signal_id,
                        signal_type=receipt.signal_type,
                        target=receipt.target,
                        status="skipped",
                        reason="filtered",
                    )
                )
                continue
            if receipt.action not in allowed_actions:
                result.skipped += 1
                result.deliveries.append(
                    ReplayDelivery(
                        receipt_index=index,
                        action=receipt.action,
                        trace_id=receipt.trace_id,
                        signal_id=receipt.signal_id,
                        signal_type=receipt.signal_type,
                        target=receipt.target,
                        status="skipped",
                        reason="action_not_replayable",
                    )
                )
                continue

            result.attempted += 1
            if not receipt.target:
                if strict_targets:
                    raise MeshError(
                        f"Cannot replay {receipt.action!r}: receipt has no target"
                    )
                result.missing_targets.append("")
                result.skipped += 1
                result.failed += 1
                result.deliveries.append(
                    ReplayDelivery(
                        receipt_index=index,
                        action=receipt.action,
                        trace_id=receipt.trace_id,
                        signal_id=receipt.signal_id,
                        signal_type=receipt.signal_type,
                        target=receipt.target,
                        status="failed",
                        reason="missing_target",
                    )
                )
                continue

            try:
                target = mesh.get(receipt.target)
            except MeshError:
                if strict_targets:
                    raise
                result.missing_targets.append(receipt.target)
                result.skipped += 1
                result.failed += 1
                result.deliveries.append(
                    ReplayDelivery(
                        receipt_index=index,
                        action=receipt.action,
                        trace_id=receipt.trace_id,
                        signal_id=receipt.signal_id,
                        signal_type=receipt.signal_type,
                        target=receipt.target,
                        status="failed",
                        reason="missing_target",
                    )
                )
                continue

            signal = receipt.to_signal(strict=strict)
            record_event = getattr(mesh, "_record_event", None)
            if record_event is not None:
                replay_event = record_event(
                    "replay_delivered",
                    signal,
                    source="mesh",
                    target=receipt.target,
                    original_action=receipt.action,
                    original_source=receipt.source,
                    original_signal_id=receipt.signal_id,
                )
                if isawaitable(replay_event):
                    await replay_event
            await target.inbox.send(signal)
            result.delivered += 1
            result.receipts.append(receipt)
            result.deliveries.append(
                ReplayDelivery(
                    receipt_index=index,
                    action=receipt.action,
                    trace_id=receipt.trace_id,
                    signal_id=receipt.signal_id,
                    signal_type=receipt.signal_type,
                    target=receipt.target,
                    status="delivered",
                )
            )

        return result
