"""Atomic storage, history compatibility, and process ownership regressions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal as process_signal
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from signal_gating.executive import (
    Budget,
    CommitBatch,
    Commitment,
    ConflictError,
    DecisionReceipt,
    InboxAcknowledgment,
    JournalEvent,
    Limits,
    OutboxEntry,
    Ref,
    Reservation,
    Scope,
    StoreError,
)
from signal_gating.executive.store import Store

from .factories import COMMITMENT_ID, GOAL_ID, NOW, OWNER_ID, action, frame, goal, owner, signal


def event(**overrides: object) -> JournalEvent:
    return JournalEvent.model_validate(
        {
            "id": "d" * 32,
            "created_at": NOW,
            "sequence": 1,
            "event_type": "audit.recorded",
            "producer_id": "fixture",
            "principal_id": OWNER_ID,
            "subject": Ref(id=GOAL_ID),
            "payload": {
                "reason_code": "storage_test",
                "details": {"z": 1, "a": {"z": "雪", "a": 2}},
            },
            **overrides,
        }
    )


def batch(**overrides: object) -> CommitBatch:
    """Store fixtures carry the exact typed writes in canonical journal events."""
    values = {"events": (event(),), "records": (owner(), goal()), **overrides}
    records = list(values["records"])
    identifiers = {r.id for r in records}
    if any(r.kind == "goal" and r.revision == 1 for r in records) and OWNER_ID not in identifiers:
        records.append(owner())
    if any(r.kind == "action" for r in records):
        records.extend(
            (
                Commitment(
                    id=COMMITMENT_ID,
                    created_at=NOW,
                    goal_id=GOAL_ID,
                    responsible_principal_id=OWNER_ID,
                ),
                frame(),
            )
        )
    if any(r.kind == "reservation" for r in records):
        records.append(
            Budget(
                id="0" * 32, created_at=NOW, scope=Scope(goal_id=GOAL_ID), limits=Limits(calls=1)
            )
        )
    values["records"] = tuple(records)
    writes = (*records, *values.get("receipts", ()), *values.get("signals", ()))
    events = tuple(values["events"])
    if writes and events:
        events = (
            events[0].model_copy(
                update={
                    "event_type": "state.changed",
                    "payload": {"records": [r.model_dump(mode="json") for r in writes]},
                }
            ),
            *events[1:],
        )
    values["events"] = events
    return CommitBatch.model_validate(values)


@pytest.fixture
def store(tmp_path: Path):
    result = Store.open(tmp_path / "store")
    try:
        yield result
    finally:
        result.close()


def test_complete_batch_is_atomic_and_survives_reopen(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = Store.open(root)
    receipt = DecisionReceipt(
        id="e" * 32,
        subject_id=GOAL_ID,
        disposition="admit",
        reason_code="accepted",
        event_ids=(event().id,),
    )
    acknowledgment = InboxAcknowledgment(
        principal_id=OWNER_ID,
        producer_id="operator",
        external_id="command-1",
        payload_digest="a" * 64,
        receipt_id=receipt.id,
    )
    queued = OutboxEntry(action_id=action().id, available_at=NOW)
    reservation = Reservation(
        id="f" * 32,
        budget_id="0" * 32,
        subject_id=action().id,
        maximum=Limits(calls=1),
    )
    committed = store.commit(
        batch(
            records=(owner(), goal(), action(), reservation),
            receipts=(receipt,),
            signals=(signal(external_id="signal-1", producer_sequence=1),),
            inbox_acknowledgments=(acknowledgment,),
            outbox=(queued,),
        )
    )
    before = store.view()
    store.close()
    with Store.open(root) as reopened:
        assert reopened.view() == before
        assert len(before.records) == 9
        assert reopened.events() == committed
        assert reopened.acknowledgment(OWNER_ID, "operator", "command-1") == acknowledgment
        assert reopened.acknowledgment(OWNER_ID, "operator", "absent") is None
        assert reopened.inbox_acknowledgments() == (acknowledgment,)
        assert reopened.outbox() == (queued,)


def test_failure_between_events_and_view_writes_rolls_back(store: Store) -> None:
    before = store.view()
    # The trigger runs only once journal insertion has succeeded.
    with sqlite3.connect(store.root / "executive.sqlite3") as db:
        db.execute(
            "CREATE TRIGGER fail_view BEFORE INSERT ON records "
            "BEGIN SELECT RAISE(ABORT, 'injected view failure'); END"
        )
    with pytest.raises(ConflictError):
        store.commit(batch())
    assert store.view() == before
    assert store.events() == ()
    with sqlite3.connect(store.root / "executive.sqlite3") as db:
        db.execute("DROP TRIGGER fail_view")
    assert store.commit(batch())[0].sequence == 1


def test_stale_compare_and_swap_rolls_back_whole_batch(store: Store) -> None:
    store.commit(batch())
    changed = goal(revision=2, outcome="Revised")
    store.commit(
        batch(
            events=(event(id="e" * 32),),
            records=(changed,),
            expected_revisions=(Ref(id=GOAL_ID, revision=1),),
        )
    )
    before = store.view()
    with pytest.raises(ConflictError):
        store.commit(
            batch(
                events=(event(id="f" * 32),),
                records=(goal(revision=3),),
                expected_revisions=(Ref(id=GOAL_ID, revision=1),),
            )
        )
    assert store.view() == before
    assert [item.sequence for item in store.events()] == [1, 2]


@pytest.mark.parametrize("revision,expected", [(2, ()), (3, (Ref(id=GOAL_ID),))])
def test_updates_require_cas_and_next_revision(
    store: Store, revision: int, expected: tuple[Ref, ...]
) -> None:
    store.commit(batch())
    with pytest.raises(ConflictError):
        store.commit(
            batch(
                events=(event(id="e" * 32),),
                records=(goal(revision=revision),),
                expected_revisions=expected,
            )
        )
    assert store.view().sequence == 1


def test_read_set_is_checked_even_without_writing_record(store: Store) -> None:
    with pytest.raises(ConflictError):
        store.commit(batch(expected_revisions=(Ref(id=OWNER_ID),)))
    assert store.events() == ()
    store.commit(batch())
    with pytest.raises(ConflictError):
        store.commit(
            batch(
                events=(event(id="e" * 32),),
                records=(),
                expected_revisions=(Ref(id=GOAL_ID, kind="task"),),
            )
        )


def test_sequence_is_assigned_by_store_and_cursor_is_exclusive(store: Store) -> None:
    first = store.commit(batch(events=(event(sequence=999), event(id="e" * 32, sequence=99))))
    assert [item.sequence for item in first] == [1, 2]
    second = store.commit(batch(events=(event(id="f" * 32),), records=()))
    assert second[0].sequence == 3
    assert store.events(after=2) == second
    assert store.view().sequence == 3
    assert store.commit(CommitBatch()) == ()


@pytest.mark.parametrize("variant", ["event", "record", "signal_external", "signal_sequence"])
def test_duplicate_identities_fail_without_partial_writes(store: Store, variant: str) -> None:
    store.commit(batch(signals=(signal(external_id="same", producer_sequence=3),)))
    kwargs: dict[str, object] = {"events": (event(id="e" * 32),), "records": ()}
    if variant == "event":
        kwargs["events"] = (event(),)
    elif variant == "record":
        kwargs["records"] = (goal(),)
    else:
        kwargs["signals"] = (
            signal(
                id="a" * 32,
                payload={"changed": True},
                external_id="same" if variant == "signal_external" else "different",
                producer_sequence=4 if variant == "signal_external" else 3,
            ),
        )
    before = store.view()
    with pytest.raises(ConflictError):
        store.commit(batch(**kwargs))
    assert store.view() == before


def test_canonical_json_and_checksum_detect_corruption(store: Store) -> None:
    store.commit(batch())
    with sqlite3.connect(store.root / "executive.sqlite3") as db:
        payload, checksum = db.execute("SELECT payload, checksum FROM journal").fetchone()
        assert payload == json.dumps(
            json.loads(payload),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        assert checksum == hashlib.sha256(payload.encode("utf-8")).hexdigest()
        db.execute(
            "UPDATE journal SET payload = replace(payload, 'state.changed', 'state.corrupted')"
        )
    with pytest.raises(StoreError, match="checksum"):
        store.events()


def test_readonly_snapshot_is_stable_while_writer_advances(store: Store) -> None:
    store.commit(batch())
    with Store.open_readonly(store.root) as snapshot:
        before = snapshot.view()
        store.commit(
            batch(
                events=(event(id="e" * 32),),
                records=(goal(revision=2),),
                expected_revisions=(Ref(id=GOAL_ID),),
            )
        )
        assert snapshot.view() == before
        assert len(snapshot.events()) == 1
        with pytest.raises(StoreError):
            snapshot.commit(CommitBatch(events=(event(id="f" * 32),)))
    with Store.open_readonly(store.root) as snapshot:
        assert snapshot.view().sequence == 2


def test_readonly_missing_store_never_creates_files(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    with pytest.raises(StoreError):
        Store.open_readonly(root)
    assert not root.exists()


def test_future_schema_is_rejected_before_mutation(tmp_path: Path) -> None:
    root = tmp_path / "future"
    root.mkdir()
    path = root / "executive.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=999")
    before = path.read_bytes()
    with pytest.raises(StoreError, match="schema"):
        Store.open(root)
    assert path.read_bytes() == before
    assert sorted(item.name for item in root.iterdir()) == ["executive.sqlite3"]


def test_migration_preserves_interpretable_event_bytes(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    raw = batch().events[0].model_dump_json()
    records = batch().records
    with sqlite3.connect(root / "executive.sqlite3") as db:
        db.execute(
            "CREATE TABLE journal(sequence INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, "
            "payload TEXT NOT NULL, checksum TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE records(id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
            "revision INTEGER NOT NULL CHECK(revision>0), payload TEXT NOT NULL, "
            "checksum TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO journal VALUES(1,?,?,?)",
            (event().id, raw, hashlib.sha256(raw.encode()).hexdigest()),
        )
        for record in records:
            encoded = record.model_dump_json()
            db.execute(
                "INSERT INTO records VALUES(?,?,?,?,?)",
                (record.id, record.kind, 1, encoded, hashlib.sha256(encoded.encode()).hexdigest()),
            )
        db.execute("PRAGMA user_version=1")
    with Store.open(root) as store:
        assert store.events() == batch().events
        assert store.view().records == batch().records
        assert store.outbox() == ()
        assert store.commit(batch(events=(event(id="e" * 32),), records=()))[0].sequence == 2
    with sqlite3.connect(root / "executive.sqlite3") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert db.execute("SELECT payload FROM journal WHERE sequence=1").fetchone()[0] == raw


def test_writer_ownership_is_refused_in_another_process(store: Store) -> None:
    code = """from pathlib import Path
import sys
from signal_gating.executive import StoreError
from signal_gating.executive.store import Store
try:
    Store.open(Path(sys.argv[1]))
except StoreError as exc:
    print(exc.code)
else:
    raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(store.root)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "store_owned"


def test_kill_between_journal_and_view_writes_releases_lock_and_rolls_back(tmp_path: Path) -> None:
    root = tmp_path / "crash"
    code = """from pathlib import Path
import os, signal, sys
from signal_gating.executive.store import Store
from tests.executive.test_store import batch
store = Store.open(Path(sys.argv[1]))
def crash(sql):
    if sql.startswith('INSERT INTO records'):
        os.kill(os.getpid(), signal.SIGKILL)
store._connection.set_trace_callback(crash)
store.commit(batch())
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == -process_signal.SIGKILL, result.stderr
    with Store.open(root) as reopened:
        assert reopened.view().sequence == 0
        assert reopened.view().records == ()
        assert reopened.events() == ()
        assert reopened.commit(batch())[0].sequence == 1


def test_closed_store_fails_clearly_and_releases_ownership(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = Store.open(root)
    store.close()
    store.close()
    with pytest.raises(StoreError):
        store.view()
    with Store.open(root) as reopened:
        assert reopened.events() == ()


def test_unsupported_writer_platform_fails_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    root = tmp_path / "unsupported"
    with pytest.raises(StoreError):
        Store.open(root)
    assert not root.exists()


def test_batch_requires_journal_for_authoritative_changes(store: Store) -> None:
    with pytest.raises(StoreError):
        store.commit(CommitBatch(records=(goal(),)))
    assert store.view().records == ()


def test_acknowledgment_and_outbox_references_are_atomic(store: Store) -> None:
    ack = InboxAcknowledgment(
        principal_id=OWNER_ID,
        producer_id="operator",
        external_id="x",
        payload_digest="a" * 64,
        receipt_id="a" * 32,
    )
    for kwargs in (
        {"inbox_acknowledgments": (ack,)},
        {"outbox": (OutboxEntry(action_id="b" * 32),)},
    ):
        with pytest.raises(StoreError):
            store.commit(batch(**kwargs))
        assert store.events() == ()
        assert store.view().records == ()


def test_batch_extensions_are_strict_immutable_and_roundtrip() -> None:
    ack = InboxAcknowledgment(
        principal_id=OWNER_ID,
        producer_id="operator",
        external_id="x",
        payload_digest="a" * 64,
        receipt_id="a" * 32,
    )
    queued = OutboxEntry(action_id="b" * 32, available_at=NOW)
    value = CommitBatch(inbox_acknowledgments=(ack,), outbox=(queued,))
    assert CommitBatch.model_validate_json(value.model_dump_json()) == value
    with pytest.raises(ValidationError):
        ack.payload_digest = "b" * 64
    with pytest.raises(ValidationError):
        InboxAcknowledgment.model_validate({**ack.model_dump(), "payload_digest": "bad"})
    with pytest.raises(ValidationError):
        OutboxEntry.model_validate({"action_id": "b" * 32, "status": "authorized"})


@pytest.mark.parametrize("changed", [False, True])
def test_duplicate_inbox_identity_preserves_original_binding(store: Store, changed: bool) -> None:
    receipt = DecisionReceipt(
        id="e" * 32, subject_id=GOAL_ID, disposition="admit", reason_code="accepted"
    )
    ack = InboxAcknowledgment(
        principal_id=OWNER_ID,
        producer_id="operator",
        external_id="x",
        payload_digest="a" * 64,
        receipt_id=receipt.id,
    )
    store.commit(batch(receipts=(receipt,), inbox_acknowledgments=(ack,)))
    before = store.view()
    duplicate = ack.model_copy(update={"payload_digest": "b" * 64}) if changed else ack
    with pytest.raises(ConflictError):
        store.commit(
            batch(events=(event(id="f" * 32),), records=(), inbox_acknowledgments=(duplicate,))
        )
    assert store.view() == before
    assert store.acknowledgment(OWNER_ID, "operator", "x") == ack


def test_future_record_schema_rejected_before_writes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    with Store.open(root) as store:
        store.commit(batch())
    path = root / "executive.sqlite3"
    with sqlite3.connect(path) as db:
        raw = event().model_dump(mode="json")
        raw["schema_version"] = 999
        payload = json.dumps(raw)
        db.execute(
            "UPDATE journal SET payload=?,checksum=?",
            (payload, hashlib.sha256(payload.encode()).hexdigest()),
        )
    before = path.read_bytes()
    with pytest.raises(StoreError, match="schema"):
        Store.open(root)
    assert path.read_bytes() == before


def test_failed_migration_rolls_back_schema_and_releases_lock(tmp_path: Path) -> None:
    root = tmp_path / "migration"
    with Store.open(root) as store:
        store.commit(batch(signals=(signal(external_id="same"),)))
    path = root / "executive.sqlite3"
    with sqlite3.connect(path) as db:
        for table in ("signal_keys", "receipt_events", "inbox_acknowledgments", "outbox"):
            db.execute(f"DROP TABLE {table}")
        db.execute("PRAGMA user_version=1")
        # An old store with duplicate identities cannot silently lose one in migration.
        duplicate = signal(id="a" * 32, external_id="same").model_dump_json()
        db.execute(
            "INSERT INTO records VALUES(?,?,?,?,?)",
            ("a" * 32, "signal", 1, duplicate, hashlib.sha256(duplicate.encode()).hexdigest()),
        )
        # Keep replay consistent so this specifically exercises migration's
        # unique signal-index backfill, rather than the replay preflight gate.
        duplicate_event = event(
            id="e" * 32,
            sequence=2,
            event_type="state.changed",
            payload={"records": [json.loads(duplicate)]},
        ).model_dump_json()
        db.execute(
            "INSERT INTO journal VALUES(2,?,?,?)",
            ("e" * 32, duplicate_event, hashlib.sha256(duplicate_event.encode()).hexdigest()),
        )
    with pytest.raises(StoreError):
        Store.open(root)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='signal_keys'").fetchone() is None
        db.execute("DELETE FROM records WHERE id=?", ("a" * 32,))
        db.execute("DELETE FROM journal WHERE sequence=2")
    with Store.open(root) as migrated:
        with pytest.raises(ConflictError):
            migrated.commit(
                batch(
                    events=(event(id="e" * 32),),
                    records=(),
                    signals=(signal(id="b" * 32, external_id="same"),),
                )
            )
        assert migrated.view().sequence == 1


def test_kill_after_committed_batch_preserves_records(tmp_path: Path) -> None:
    root = tmp_path / "committed-crash"
    code = """from pathlib import Path
import os, signal, sys
from signal_gating.executive.store import Store
from tests.executive.test_store import batch
store = Store.open(Path(sys.argv[1]))
store.commit(batch())
os.kill(os.getpid(), signal.SIGKILL)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == -process_signal.SIGKILL, result.stderr
    with Store.open(root) as store:
        assert store.events() == batch().events
        assert store.view().records == batch().records


def test_readonly_side_indexes_reject_envelope_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "corrupt-index"
    receipt = DecisionReceipt(
        id="e" * 32, subject_id=GOAL_ID, disposition="admit", reason_code="accepted"
    )
    ack = InboxAcknowledgment(
        principal_id=OWNER_ID,
        producer_id="operator",
        external_id="x",
        payload_digest="a" * 64,
        receipt_id=receipt.id,
    )
    with Store.open(root) as store:
        store.commit(
            batch(
                receipts=(receipt,),
                inbox_acknowledgments=(ack,),
                records=(goal(), action()),
                outbox=(OutboxEntry(action_id=action().id),),
            )
        )
    with sqlite3.connect(root / "executive.sqlite3") as db:
        db.execute("UPDATE inbox_acknowledgments SET external_id='wrong'")
    with pytest.raises(StoreError, match="index"):
        Store.open_readonly(root)


def test_sqlite_durability_and_contention_fail_without_partial_state(store: Store) -> None:
    assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert 0 < store._connection.execute("PRAGMA busy_timeout").fetchone()[0] <= 1000
    with sqlite3.connect(store.root / "executive.sqlite3") as other:
        other.execute("BEGIN IMMEDIATE")
        with pytest.raises(StoreError) as error:
            store.commit(batch())
        assert error.value.code == "store_commit_failed"
    assert store.view().sequence == 0
    assert store.events() == ()
    assert store.commit(batch())[0].sequence == 1


def test_unknown_receipt_event_fails_foreign_key_and_rolls_back(store: Store) -> None:
    receipt = DecisionReceipt(
        subject_id=GOAL_ID, disposition="admit", reason_code="accepted", event_ids=("a" * 32,)
    )
    with pytest.raises(ConflictError):
        store.commit(batch(receipts=(receipt,)))
    assert store.events() == ()
    assert store.view().records == ()


def test_late_outbox_failure_rolls_back_inbox_receipt_and_reservation(store: Store) -> None:
    receipt = DecisionReceipt(
        id="e" * 32, subject_id=GOAL_ID, disposition="admit", reason_code="accepted"
    )
    ack = InboxAcknowledgment(
        principal_id=OWNER_ID,
        producer_id="operator",
        external_id="x",
        payload_digest="a" * 64,
        receipt_id=receipt.id,
    )
    reservation = Reservation(
        id="f" * 32, budget_id="0" * 32, subject_id=action().id, maximum=Limits(calls=1)
    )
    with sqlite3.connect(store.root / "executive.sqlite3") as db:
        db.execute(
            "CREATE TRIGGER fail_outbox BEFORE INSERT ON outbox "
            "BEGIN SELECT RAISE(ABORT, 'injected outbox failure'); END"
        )
    with pytest.raises(ConflictError):
        store.commit(
            batch(
                records=(goal(), action(), reservation),
                receipts=(receipt,),
                inbox_acknowledgments=(ack,),
                outbox=(OutboxEntry(action_id=action().id),),
            )
        )
    assert store.view().records == ()
    assert store.events() == ()
    assert store.inbox_acknowledgments() == ()
    assert store.outbox() == ()


def test_unversioned_nonempty_database_is_refused_before_writes(tmp_path: Path) -> None:
    root = tmp_path / "unversioned"
    root.mkdir()
    path = root / "executive.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated(value TEXT)")
    before = path.read_bytes()
    with pytest.raises(StoreError, match="Unversioned"):
        Store.open(root)
    assert path.read_bytes() == before
    assert sorted(item.name for item in root.iterdir()) == ["executive.sqlite3"]


def test_duplicate_outbox_keys_in_one_batch_do_not_silently_overwrite(store: Store) -> None:
    with pytest.raises(ConflictError):
        store.commit(
            batch(
                records=(goal(), action()),
                outbox=(
                    OutboxEntry(action_id=action().id, available_at=NOW),
                    OutboxEntry(action_id=action().id),
                ),
            )
        )
    assert store.events() == ()
    assert store.outbox() == ()


def test_fork_survivor_does_not_keep_dead_writer_ownership(tmp_path: Path) -> None:
    root = tmp_path / "fork-survivor"
    read_fd, write_fd = os.pipe()
    code = """from pathlib import Path
import os, signal, sys
from signal_gating.executive.store import Store
store = Store.open(Path(sys.argv[1]))
child = os.fork()
if child == 0:
    os.close(1)
    os.close(2)
    os.read(int(sys.argv[2]), 1)
    os._exit(0)
print(child, flush=True)
os.kill(os.getpid(), signal.SIGKILL)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(root), str(read_fd)],
            pass_fds=(read_fd,),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == -process_signal.SIGKILL, result.stderr
        child_pid = int(result.stdout.strip())
        os.kill(child_pid, 0)
        with Store.open(root) as reopened:
            assert reopened.view().sequence == 0
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_fork_cleanup_does_not_unlock_living_writer(tmp_path: Path) -> None:
    code = """from pathlib import Path
import os, sys
from signal_gating.executive import StoreError
from signal_gating.executive.store import Store
store = Store.open(Path(sys.argv[1]))
child = os.fork()
if child == 0:
    os._exit(0)
os.waitpid(child, 0)
try:
    Store.open(Path(sys.argv[1]))
except StoreError as exc:
    assert exc.code == 'store_owned'
else:
    raise SystemExit('fork cleanup released the living parent ownership')
store.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "living-parent")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("opener_kind", ["writer", "snapshot"])
@pytest.mark.parametrize("uncheckpointed", [False, True])
def test_future_wal_schema_rejected_without_source_mutation(
    tmp_path: Path,
    opener_kind: str,
    uncheckpointed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "preflight-temp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    root = tmp_path / "future-wal"
    root.mkdir()
    path = root / "executive.sqlite3"
    database = sqlite3.connect(path)
    try:
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA wal_autocheckpoint=0")
        database.execute("PRAGMA user_version=999")
        if not uncheckpointed:
            database.close()
            assert not Path(str(path) + "-wal").exists()
        else:
            assert Path(str(path) + "-wal").stat().st_size > 0
        before = {item.name: item.read_bytes() for item in root.iterdir()}
        opener = Store.open if opener_kind == "writer" else Store.open_readonly
        with pytest.raises(StoreError) as error:
            opener(root)
        assert error.value.code == "unsupported_schema"
        assert {item.name: item.read_bytes() for item in root.iterdir()} == before
        assert list(scratch.iterdir()) == []
    finally:
        database.close()


def test_preflight_honors_supported_schema_committed_only_in_wal(tmp_path: Path) -> None:
    root = tmp_path / "wal-schema-override"
    with Store.open(root):
        pass
    path = root / "executive.sqlite3"
    database = sqlite3.connect(path)
    try:
        database.execute("PRAGMA wal_autocheckpoint=0")
        database.execute("PRAGMA user_version=999")
        database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert int.from_bytes(path.read_bytes()[60:64], "big") == 999
        database.execute("PRAGMA user_version=2")
        assert Path(str(path) + "-wal").stat().st_size > 0
        with Store.open(root) as store:
            assert store.commit(batch())[0].sequence == 1
        with Store.open_readonly(root) as snapshot:
            assert snapshot.view().records == batch().records
    finally:
        database.close()


def test_unstable_preflight_is_bounded_and_cleans_temporary_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "changing-source"
    with Store.open(root):
        pass
    scratch = tmp_path / "preflight-temp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    path = root / "executive.sqlite3"
    database = sqlite3.connect(path)
    real_copy = shutil.copyfile
    copies = 0

    def copy_while_source_changes(source: Path, destination: Path) -> str:
        nonlocal copies
        result = real_copy(source, destination)
        if source == path:
            copies += 1
            database.execute(f"PRAGMA user_version={1 if copies % 2 else 2}")
        return str(result)

    try:
        database.execute("PRAGMA wal_autocheckpoint=0")
        monkeypatch.setattr(shutil, "copyfile", copy_while_source_changes)
        with pytest.raises(StoreError) as error:
            Store.open(root)
        assert error.value.code == "unstable_snapshot"
        assert 0 < copies <= 3
        assert list(scratch.iterdir()) == []
    finally:
        database.close()


@pytest.mark.parametrize("explicit_records", [False, True])
def test_embedded_constraint_promotion_requires_owning_goal_cas(store, explicit_records):
    from signal_gating.executive import Constraint
    from signal_gating.executive.reducers import rebuild, reduce_event

    original = Constraint(description="Original embedded constraint")
    revised = original.model_copy(
        update={"revision": 2, "description": "Revised embedded constraint"}
    )
    store.commit(batch(records=(owner(), goal(constraints=(original,)))))
    store.commit(
        batch(
            events=(event(id="e" * 32),),
            records=(goal(revision=2, constraints=(revised,)),),
            expected_revisions=(Ref(id=GOAL_ID),),
        )
    )
    promotion = event(
        id="f" * 32,
        event_type="state.changed",
        payload={"records": [revised.model_dump(mode="json")]},
    )
    before = store.view()
    # Promotion is a valid representation change, with no invented revision 1.
    reduce_event(before, promotion.model_copy(update={"sequence": before.sequence + 1}))
    with pytest.raises(ConflictError) as error:
        store.commit(
            CommitBatch(events=(promotion,), records=(revised,) if explicit_records else ())
        )
    assert error.value.code == "missing_promotion_read"
    assert store.view() == before
    committed = store.commit(
        CommitBatch(
            events=(promotion,), expected_revisions=(Ref(id=GOAL_ID, revision=2, kind="goal"),)
        )
    )
    assert committed
    assert next(r for r in store.view().records if r.id == revised.id) == revised
    assert rebuild(store.events()) == store.view()
    with Store.open_readonly(store.root) as snapshot:
        assert snapshot.view() == store.view()


@pytest.mark.parametrize("explicit_records", [False, True])
def test_embedded_criterion_promotion_requires_owning_goal_cas(store, explicit_records):
    from signal_gating.executive.reducers import rebuild, reduce_event

    from .factories import criterion

    original = criterion()
    revised = original.model_copy(update={"revision": 2, "description": "Revised"})
    store.commit(batch(records=(owner(), goal(criteria=(original,)))))
    store.commit(
        batch(
            events=(event(id="e" * 32),),
            records=(goal(revision=2, criteria=(revised,)),),
            expected_revisions=(Ref(id=GOAL_ID),),
        )
    )
    promotion = event(
        id="f" * 32,
        event_type="state.changed",
        payload={"records": [revised.model_dump(mode="json")]},
    )
    before = store.view()
    reduce_event(before, promotion.model_copy(update={"sequence": before.sequence + 1}))
    with pytest.raises(ConflictError) as error:
        store.commit(
            CommitBatch(events=(promotion,), records=(revised,) if explicit_records else ())
        )
    assert error.value.code == "missing_promotion_read"
    assert store.view() == before
    store.commit(
        CommitBatch(events=(promotion,), expected_revisions=(Ref(id=GOAL_ID, revision=2),))
    )
    assert next(r for r in store.view().records if r.id == revised.id) == revised
    assert rebuild(store.events()) == store.view()
    with Store.open_readonly(store.root) as snapshot:
        assert snapshot.view() == store.view()
