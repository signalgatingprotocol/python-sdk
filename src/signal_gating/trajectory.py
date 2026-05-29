"""Trajectory capture: a verifiable, structured record of every signal hop."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from signal_gating.signal import Signal

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


@dataclass(frozen=True, slots=True)
class Receipt:
    """A verifiable, structured record of one signal crossing the mesh.

    A Receipt serves two purposes that used to pull in different directions:

    * **Audit** — ``signal_type`` and ``payload`` are a human-readable projection
      (domain fields only; the envelope is denormalized into the typed fields).
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

    @classmethod
    def from_signal(cls, signal: Signal, source: str, target: str) -> Receipt:
        core: dict[str, Any] = {
            "trace_id": signal.trace_id,
            "signal_id": signal.id,
            "parent_id": signal.parent_id,
            "signal_type": type(signal).__name__,
            "source": source,
            "target": target,
            "priority": signal.priority,
            "timestamp": time.time(),
            "payload": _domain_payload(signal),
            "wire": signal.to_wire(),
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
        return _digest(core) == self.digest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrajectoryRecorder:
    """A mesh interceptor that records a Receipt for every signal hop.

    Attach with ``mesh.intercept(recorder)``. It is a pure observer: it returns
    every signal unchanged and never blocks delivery.

        recorder = TrajectoryRecorder()
        mesh.intercept(recorder)
        ...
        recorder.export_jsonl("runs.jsonl")
    """

    def __init__(self) -> None:
        self._receipts: list[Receipt] = []

    def __call__(self, signal: Signal, source: str, target: str) -> Signal:
        self._receipts.append(Receipt.from_signal(signal, source, target))
        return signal

    @property
    def receipts(self) -> list[Receipt]:
        return list(self._receipts)

    def trajectories(self) -> dict[str, list[Receipt]]:
        """Receipts grouped by trace_id, preserving capture order within each run."""
        grouped: dict[str, list[Receipt]] = {}
        for r in self._receipts:
            grouped.setdefault(r.trace_id, []).append(r)
        return grouped

    def export_jsonl(self, path: str | Path) -> int:
        """Write all receipts as JSON Lines. Returns the number written.

        Each line carries the full wire envelope, so :meth:`load_jsonl` reloads a
        verifiable trajectory and :meth:`replay` reconstructs the typed signals.
        """
        out = Path(path)
        with out.open("w", encoding="utf-8") as f:
            for r in self._receipts:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")
        return len(self._receipts)

    def load_jsonl(self, path: str | Path) -> int:
        """Load receipts from a file written by :meth:`export_jsonl`, appending.

        The durability half of the trajectory story: persist a run, survive a
        restart, then reload it as verifiable receipts you can inspect or
        ``replay`` as typed signals. Returns the number of receipts loaded.
        """
        src = Path(path)
        loaded = 0
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._receipts.append(Receipt.from_dict(json.loads(line)))
                loaded += 1
        return loaded

    def replay(self, *, strict: bool = True) -> list[Signal]:
        """Reconstruct every captured signal as its original type, in capture order.

        The faithful counterpart to :meth:`export_jsonl`: a trajectory read off
        disk comes back as the exact typed signals that produced it, ready to
        re-run, audit, or learn from. Import the modules defining your signal
        types first so they are registered (see :meth:`Signal.from_wire`); with
        ``strict=False`` an unknown type degrades to a base ``Signal`` rather
        than raising.
        """
        return [r.to_signal(strict=strict) for r in self._receipts]

    def clear(self) -> None:
        self._receipts.clear()
