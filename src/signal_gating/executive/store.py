"""Single-writer SQLite persistence for already-decided executive batches.

Only deterministic validation and database work run in transactions. Callers
stage artifacts and obtain model/tool results before constructing CommitBatch.
Local filesystems only: SQLite WAL and advisory locking are not a remote lease.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from .errors import ConflictError, InvalidTransition, StoreError
from .locking import OwnershipLock, require_writer_platform
from .messages import CommitBatch, InboxAcknowledgment, OutboxEntry
from .migrations import SCHEMA_VERSION, migrate, schema_version
from .records import (
    DecisionReceipt,
    ExecutiveSignal,
    JournalEvent,
    Model,
    PersistentRecord,
    StateView,
)
from .reducers import rebuild, reduce_event

DATABASE_NAME = "executive.sqlite3"
BUSY_TIMEOUT_MS = 1000
_PREFLIGHT_ATTEMPTS = 3
_RECORD_ADAPTER: TypeAdapter[PersistentRecord] = TypeAdapter(PersistentRecord)
_EVENT_ADAPTER = TypeAdapter(JournalEvent)
_ACK_ADAPTER = TypeAdapter(InboxAcknowledgment)
_OUTBOX_ADAPTER = TypeAdapter(OutboxEntry)
T = TypeVar("T", bound=Model)


def canonical_json(value: Model) -> str:
    """Stable UTF-8 JSON used for persisted envelopes and input digests."""
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _checksum(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode(payload: str, checksum: str, adapter: TypeAdapter[T]) -> T:
    if _checksum(payload) != checksum:
        raise StoreError("corrupt_store", "Executive payload checksum mismatch")
    try:
        return adapter.validate_json(payload)
    except ValidationError as exc:
        raise StoreError(
            "unsupported_record", "Invalid or unsupported executive record schema"
        ) from exc


def _events(connection: sqlite3.Connection, after: int = 0) -> tuple[JournalEvent, ...]:
    result = []
    for sequence, identifier, payload, checksum in connection.execute(
        "SELECT sequence,id,payload,checksum FROM journal WHERE sequence>? ORDER BY sequence",
        (after,),
    ):
        event = _decode(payload, checksum, _EVENT_ADAPTER)
        if (event.sequence, event.id) != (sequence, identifier):
            raise StoreError("corrupt_store", "Journal envelope disagrees with its index")
        result.append(event)
    return tuple(result)


def _records(connection: sqlite3.Connection) -> tuple[PersistentRecord, ...]:
    result = []
    for identifier, kind, revision, payload, checksum in connection.execute(
        "SELECT id,kind,revision,payload,checksum FROM records ORDER BY id"
    ):
        record = _decode(payload, checksum, _RECORD_ADAPTER)
        if (record.id, record.kind, record.revision) != (identifier, kind, revision):
            raise StoreError("corrupt_store", "Record envelope disagrees with its index")
        result.append(record)
    return tuple(result)


def _acknowledgments(
    connection: sqlite3.Connection,
    identity: tuple[str, str, str] | None = None,
) -> tuple[InboxAcknowledgment, ...]:
    query = (
        "SELECT principal_id,producer_id,external_id,receipt_id,payload,checksum "
        "FROM inbox_acknowledgments"
    )
    if identity is not None:
        query += " WHERE principal_id=? AND producer_id=? AND external_id=?"
    result = []
    for principal, producer, external, receipt, payload, checksum in connection.execute(
        query + " ORDER BY principal_id,producer_id,external_id",
        identity or (),
    ):
        ack = _decode(payload, checksum, _ACK_ADAPTER)
        if (ack.principal_id, ack.producer_id, ack.external_id, ack.receipt_id) != (
            principal,
            producer,
            external,
            receipt,
        ):
            raise StoreError("corrupt_store", "Inbox envelope disagrees with its index")
        result.append(ack)
    return tuple(result)


def _outbox(connection: sqlite3.Connection) -> tuple[OutboxEntry, ...]:
    result = []
    for action_id, payload, checksum in connection.execute(
        "SELECT action_id,payload,checksum FROM outbox"
    ):
        queued = _decode(payload, checksum, _OUTBOX_ADAPTER)
        if queued.action_id != action_id:
            raise StoreError("corrupt_store", "Outbox envelope disagrees with its index")
        result.append(queued)
    return tuple(sorted(result, key=lambda item: (item.available_at, item.action_id)))


def _validate(connection: sqlite3.Connection) -> int:
    version = schema_version(connection)
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise StoreError("corrupt_store", "SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise StoreError("corrupt_store", "Executive database contains invalid references")
    if version:
        events = _events(connection)
        records = _records(connection)
        try:
            reconstructed = rebuild(events)
        except InvalidTransition as exc:
            raise StoreError("invalid_journal", "Executive journal cannot be replayed") from exc
        if reconstructed.records != records:
            raise StoreError("corrupt_store", "Materialized state disagrees with the journal")
    if version >= 2:
        _outbox(connection)
        _acknowledgments(connection)
    return version


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.as_uri() + "?mode=ro",
        uri=True,
        isolation_level=None,
        timeout=BUSY_TIMEOUT_MS / 1000,
    )
    connection.execute("PRAGMA query_only=ON")
    connection.execute("BEGIN")
    return connection


def _fingerprint(path: Path) -> tuple[int, int, int, int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    # Reads may update atime. Only identity/content-change metadata participates.
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _preflight(path: Path) -> None:
    """Validate a stable DB/WAL copy without letting SQLite touch source sidecars.

    SQLite mode=ro may create WAL/SHM files or change SHM read marks. Run it only
    on a private temporary copy until compatibility is established. Copy the WAL
    as well as the main file: committed schema/history can exist only in WAL.
    Local-filesystem metadata detects concurrent append/checkpoint/replacement;
    unstable sources fail closed after bounded retries. Hostile writers capable
    of defeating filesystem metadata remain outside the host trust boundary.
    """
    sources = (path, Path(str(path) + "-wal"))
    for _ in range(_PREFLIGHT_ATTEMPTS):
        before = tuple(_fingerprint(source) for source in sources)
        with TemporaryDirectory(prefix="sgp-preflight-") as temporary:
            target = Path(temporary) / path.name
            try:
                for source, fingerprint in zip(sources, before):
                    if fingerprint is not None:
                        shutil.copyfile(source, Path(temporary) / source.name)
            except FileNotFoundError:
                # Checkpoint/rename raced the copy; do not interpret a mixture.
                continue
            if before[0] is None or before != tuple(_fingerprint(source) for source in sources):
                continue
            connection = _connect_readonly(target)
            try:
                _validate(connection)
            finally:
                connection.close()
            return
    raise StoreError("unstable_snapshot", "Executive database changed during preflight snapshot")


class Store:
    """Own one writer or a pinned read-only snapshot; close explicitly."""

    def __init__(
        self,
        root: Path,
        connection: sqlite3.Connection,
        ownership: OwnershipLock | None,
        *,
        readonly: bool,
    ) -> None:
        self.root = root
        self._connection = connection
        self._ownership = ownership
        self._readonly = readonly
        self._closed = False
        self._pid = os.getpid()

    @classmethod
    def open(cls, root: Path) -> Store:
        require_writer_platform()
        root = root.resolve()
        ownership = None
        connection = None
        try:
            path = root / DATABASE_NAME
            # Unsupported schema/envelope versions fail before lock, WAL, DDL,
            # or data writes. The owned connection checks the version again.
            if path.exists():
                _preflight(path)
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            ownership = OwnershipLock.acquire(root)
            connection = sqlite3.connect(
                path,
                isolation_level=None,
                timeout=BUSY_TIMEOUT_MS / 1000,
            )
            schema_version(connection)
            connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA foreign_keys=ON")
            if connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise StoreError("unsupported_store", "Executive store requires SQLite WAL mode")
            connection.execute("PRAGMA synchronous=FULL")
            migrate(connection)
            _validate(connection)
            return cls(root, connection, ownership, readonly=False)
        except BaseException as exc:
            if connection is not None:
                connection.close()
            if ownership is not None:
                ownership.close()
            if isinstance(exc, (sqlite3.Error, OSError)):
                raise StoreError("store_open_failed", "Could not open executive store") from exc
            raise

    @classmethod
    def open_readonly(cls, root: Path) -> Store:
        """Pin one consistent committed snapshot without acquiring writer ownership."""
        root = root.resolve()
        connection = None
        try:
            _preflight(root / DATABASE_NAME)
            connection = _connect_readonly(root / DATABASE_NAME)
            if _validate(connection) != SCHEMA_VERSION:
                raise StoreError("migration_required", "Open a writer to migrate this store first")
            return cls(root, connection, None, readonly=True)
        except BaseException as exc:
            if connection is not None:
                connection.close()
            if isinstance(exc, (sqlite3.Error, OSError)):
                raise StoreError("store_open_failed", "Could not open executive snapshot") from exc
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise StoreError("store_closed", "Executive store is closed")
        if self._pid != os.getpid():
            raise StoreError("inherited_store", "Reopen the executive store after process fork")

    @contextmanager
    def _read(self) -> Iterator[None]:
        self._check_open()
        try:
            if not self._readonly:
                self._connection.execute("BEGIN")
            yield
        except sqlite3.Error as exc:
            raise StoreError("store_read_failed", "Could not read executive state") from exc
        finally:
            if not self._readonly:
                self._connection.rollback()

    def view(self) -> StateView:
        with self._read():
            sequence = self._connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM journal"
            ).fetchone()[0]
            return StateView(sequence=sequence, records=_records(self._connection))

    def events(self, after: int = 0) -> tuple[JournalEvent, ...]:
        if type(after) is not int or after < 0:
            raise ValueError("event cursor must be a nonnegative integer")
        with self._read():
            return _events(self._connection, after)

    def acknowledgment(
        self,
        principal_id: str,
        producer_id: str,
        external_id: str,
    ) -> InboxAcknowledgment | None:
        """Look up the digest/receipt binding before acknowledging duplicate input."""
        with self._read():
            rows = _acknowledgments(self._connection, (principal_id, producer_id, external_id))
            return rows[0] if rows else None

    def inbox_acknowledgments(self) -> tuple[InboxAcknowledgment, ...]:
        with self._read():
            return _acknowledgments(self._connection)

    def outbox(self) -> tuple[OutboxEntry, ...]:
        """Queue metadata; inspect ActionRecord status before selecting work."""
        with self._read():
            return _outbox(self._connection)

    def commit(self, batch: CommitBatch) -> tuple[JournalEvent, ...]:
        self._check_open()
        if self._readonly:
            raise StoreError("readonly_store", "A snapshot cannot commit executive state")
        records = batch.records + batch.receipts + batch.signals
        if not batch.events and (records or batch.inbox_acknowledgments or batch.outbox):
            raise StoreError("missing_events", "Authoritative writes require journal events")
        if any(isinstance(record, JournalEvent) for record in records):
            raise StoreError("invalid_batch", "Journal events belong in the batch events field")
        if len({record.id for record in records}) != len(records):
            raise ConflictError("duplicate_record", "A batch writes each record at most once")
        if len({queued.action_id for queued in batch.outbox}) != len(batch.outbox):
            raise ConflictError("duplicate_outbox", "A batch queues each action at most once")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            expected = {ref.id: ref for ref in batch.expected_revisions}
            if len(expected) != len(batch.expected_revisions):
                raise ConflictError("duplicate_read", "A read set names each record at most once")
            current = {record.id: record for record in _records(self._connection)}
            for ref in batch.expected_revisions:
                previous = current.get(ref.id)
                if (
                    previous is None
                    or previous.revision != ref.revision
                    or (ref.kind != "record" and previous.kind != ref.kind)
                ):
                    raise ConflictError(
                        "stale_revision", "Expected record revision is stale", record_id=ref.id
                    )
            for record in records:
                previous = current.get(record.id)
                if previous is None:
                    valid = record.revision == 1
                else:
                    valid = (
                        record.id in expected
                        and record.kind == previous.kind
                        and record.revision == previous.revision + 1
                    )
                if not valid:
                    raise ConflictError(
                        "invalid_revision",
                        "Write requires the next CAS revision",
                        record_id=record.id,
                    )
            sequence = self._connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM journal"
            ).fetchone()[0]
            assigned = tuple(
                item.model_copy(update={"sequence": sequence + index})
                for index, item in enumerate(batch.events, 1)
            )
            reduced = StateView(sequence=sequence, records=tuple(current.values()))
            for item in assigned:
                reduced = reduce_event(reduced, item)
            derived = tuple(
                record for record in reduced.records if current.get(record.id) != record
            )
            if records and {record.id: record for record in records} != {
                record.id: record for record in derived
            }:
                raise InvalidTransition(
                    "materialization_mismatch", "Caller writes disagree with journal reduction"
                )
            # Derived cascades are ordinary authoritative writes: require CAS for
            # every existing changed record, just as for caller-supplied writes.
            for record in derived:
                previous = current.get(record.id)
                if previous is not None and record.id not in expected:
                    raise ConflictError(
                        "invalid_revision",
                        "Derived writes require a CAS read set",
                        record_id=record.id,
                    )
            records = derived
            for item in assigned:
                payload = canonical_json(item)
                self._connection.execute(
                    "INSERT INTO journal VALUES(?,?,?,?)",
                    (item.sequence, item.id, payload, _checksum(payload)),
                )
            for record in records:
                payload = canonical_json(record)
                self._connection.execute(
                    "INSERT INTO records VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "revision=excluded.revision,payload=excluded.payload,checksum=excluded.checksum",
                    (record.id, record.kind, record.revision, payload, _checksum(payload)),
                )
                if isinstance(record, ExecutiveSignal):
                    self._connection.execute(
                        "INSERT INTO signal_keys VALUES(?,?,?,?,?) ON CONFLICT(signal_id) "
                        "DO UPDATE SET principal_id=excluded.principal_id,"
                        "producer_id=excluded.producer_id,external_id=excluded.external_id,"
                        "producer_sequence=excluded.producer_sequence",
                        (
                            record.id,
                            record.principal_id,
                            record.producer_id,
                            record.external_id,
                            record.producer_sequence,
                        ),
                    )
                if isinstance(record, DecisionReceipt):
                    self._connection.execute(
                        "DELETE FROM receipt_events WHERE receipt_id=?", (record.id,)
                    )
                    self._connection.executemany(
                        "INSERT INTO receipt_events VALUES(?,?)",
                        ((record.id, identifier) for identifier in record.event_ids),
                    )
            kinds = {identifier: item.kind for identifier, item in current.items()}
            kinds.update((item.id, item.kind) for item in records)
            for ack in batch.inbox_acknowledgments:
                if kinds.get(ack.receipt_id) != "receipt":
                    raise StoreError("missing_receipt", "Inbox acknowledgment requires a receipt")
                payload = canonical_json(ack)
                self._connection.execute(
                    "INSERT INTO inbox_acknowledgments VALUES(?,?,?,?,?,?)",
                    (
                        ack.principal_id,
                        ack.producer_id,
                        ack.external_id,
                        ack.receipt_id,
                        payload,
                        _checksum(payload),
                    ),
                )
            for queued in batch.outbox:
                if kinds.get(queued.action_id) != "action":
                    raise StoreError("missing_action", "Outbox entry requires an action record")
                payload = canonical_json(queued)
                self._connection.execute(
                    "INSERT INTO outbox VALUES(?,?,?) ON CONFLICT(action_id) DO UPDATE SET "
                    "payload=excluded.payload,checksum=excluded.checksum",
                    (queued.action_id, payload, _checksum(payload)),
                )
            self._connection.commit()
            return assigned
        except BaseException as exc:
            self._connection.rollback()
            if isinstance(exc, sqlite3.IntegrityError):
                raise ConflictError(
                    "identity_conflict", "Batch identity or reference conflicts"
                ) from exc
            if isinstance(exc, sqlite3.Error):
                raise StoreError("store_commit_failed", "Executive transaction failed") from exc
            raise

    def close(self) -> None:
        if not self._closed:
            try:
                self._connection.close()
            finally:
                if self._ownership is not None:
                    self._ownership.close()
                self._closed = True

    def __enter__(self) -> Store:
        self._check_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
