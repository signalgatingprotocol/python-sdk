# Trajectory Replay via the Wire Format

- **Date:** 2026-05-29
- **Status:** Built
- **Repo:** `signalgatingprotocol/python-sdk` (branch `claude/signal-wire-format-LelTt`)
- **Scope:** Close the loop on trajectories. A `Receipt` could be read but never
  re-run; this makes a persisted trajectory replay as the exact typed signals
  that produced it — the same durability the dead-letter queue already has.

## Problem

The wire format (`2026-05-29-signal-wire-format-design.md`) made signals
serialize to a self-describing envelope and reconstruct as their original
subclass. The dead-letter queue uses it: persist on shutdown, reload and replay
after a restart as real types.

Trajectories did not. `Receipt.from_signal` stored `type(signal).__name__` and a
lossy domain `payload` (envelope fields stripped out). So `export_jsonl` produced
an audit log you could read but not replay — there was no path back to a typed
signal. The DLQ could round-trip; the trajectory could not. That asymmetry was
the most visible gap left by the wire-format work, and trajectories are the
substrate that persistent memory, skill distillation, and RL/eval pipelines all
build on — none of which can consume a flat, untyped log.

## Goal

A trajectory read off disk comes back as the exact typed signals that produced
it, ready to re-run, audit, or learn from — without changing the audit view or
the one-line capture story.

## Design

Additive, not a rewrite. A `Receipt` now serves two purposes side by side:

- **Audit** — `signal_type` and `payload` are the existing human-readable
  projection (domain fields only; envelope denormalized into typed fields).
  Unchanged, so every existing test and the JSONL shape still hold.
- **Replay** — a new `wire` field carries the full `signal.to_wire()` envelope.
  `Receipt.to_signal()` reconstructs the exact original typed signal via
  `Signal.from_wire`.

The `digest` now covers `wire` as well, so a tampered replay envelope is as
detectable as a tampered payload — reconstruction is as trustworthy as the audit
view.

```python
@dataclass(frozen=True, slots=True)
class Receipt:
    ...
    payload: dict[str, Any]   # audit projection (domain fields only)
    wire: dict[str, Any]      # full self-describing envelope (replay)
    digest: str               # sha256 over everything but itself

    @classmethod
    def from_dict(cls, data) -> Receipt: ...   # inverse of to_dict(); still verifies
    def to_signal(self, *, strict=True) -> Signal: ...  # -> original subclass
```

`TrajectoryRecorder` gains the durability half, mirroring `DeadLetterQueue`:

```python
recorder.export_jsonl(path)            # now carries the wire envelope per line
reloaded.load_jsonl(path) -> int       # append verifiable Receipts after a restart
reloaded.replay(*, strict=True) -> list[Signal]  # typed signals, capture order
```

## Decisions / taste

- **Additive over clean-slate.** Keeping `payload` *and* `wire` is mild
  redundancy (`payload` ⊂ `wire["data"]`), but it preserves the established
  verifiable audit record and the one-line capture path, and keeps the change
  low-risk. Audit-read and replay are both first-class rather than one paying a
  reconstruction cost.
- **`signal_type` stays `type(signal).__name__`** (the human class name, per the
  v1 spec). `wire["type"]` is the authoritative addressing key for
  reconstruction; the two coincide unless a class pins `__signal_type__`.
- **Same registration contract as `Signal.from_wire`.** Import the modules that
  define your signal types before `replay`; `strict=True` raises
  `UnknownSignalType`, `strict=False` degrades to a base `Signal` with the
  payload preserved in `metadata`.
- **No new dependencies, no base-class changes.** Lives entirely in
  `trajectory.py`; core stays stdlib-only (`json`, `hashlib`, `dataclasses`).

## Testing

`tests/test_trajectory.py` (existing tests unchanged; new ones added):

- Receipt carries the wire envelope and `to_signal()` reconstructs the exact
  subclass — identity (`id`, `trace_id`), priority, and full equality.
- Full lineage (`parent_id` / `trace_id`) survives reconstruction.
- `digest` covers `wire`: tampering with the replay envelope fails `verify()`.
- `from_dict(to_dict())` round-trips and stays verifiable.
- Export → fresh recorder `load_jsonl` → `replay()` yields the original typed
  signals, in capture order, all receipts still verifying (the restart story).
- Unknown type: strict replay raises `UnknownSignalType`; lenient degrades.

## Success criteria

1. A persisted trajectory reloads and replays as its original typed signals.
2. Reconstruction is tamper-evident (digest covers the wire envelope).
3. Existing audit behavior, JSONL shape, and one-line capture are unchanged.
4. `ruff` + `mypy --strict` + full `pytest` green; no new dependency.
