"""Transactional schema evolution; journal bytes remain unchanged."""

from __future__ import annotations

import json
import sqlite3

from .errors import StoreError

SCHEMA_VERSION = 2

# Version 1 is the journal and materialized view. Version 2 adds durable admission
# and dispatch indexes without rewriting versioned historical event envelopes.
_MIGRATIONS = (
    (
        "CREATE TABLE journal (sequence INTEGER PRIMARY KEY CHECK(sequence>0), "
        "id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, checksum TEXT NOT NULL)",
        "CREATE TABLE records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
        "revision INTEGER NOT NULL CHECK(revision>0), payload TEXT NOT NULL, "
        "checksum TEXT NOT NULL)",
    ),
    (
        "CREATE TABLE signal_keys (signal_id TEXT PRIMARY KEY REFERENCES records(id), "
        "principal_id TEXT NOT NULL, producer_id TEXT NOT NULL, external_id TEXT, "
        "producer_sequence INTEGER, UNIQUE(principal_id,producer_id,external_id), "
        "UNIQUE(principal_id,producer_id,producer_sequence))",
        "CREATE TABLE receipt_events (receipt_id TEXT NOT NULL REFERENCES records(id), "
        "event_id TEXT NOT NULL REFERENCES journal(id), PRIMARY KEY(receipt_id,event_id))",
        "CREATE TABLE inbox_acknowledgments (principal_id TEXT NOT NULL, "
        "producer_id TEXT NOT NULL, external_id TEXT NOT NULL, "
        "receipt_id TEXT NOT NULL REFERENCES records(id), payload TEXT NOT NULL, "
        "checksum TEXT NOT NULL, PRIMARY KEY(principal_id,producer_id,external_id))",
        "CREATE TABLE outbox (action_id TEXT PRIMARY KEY REFERENCES records(id), "
        "payload TEXT NOT NULL, checksum TEXT NOT NULL)",
    ),
)


def schema_version(connection: sqlite3.Connection) -> int:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if not 0 <= version <= SCHEMA_VERSION:
        raise StoreError("unsupported_schema", "Unsupported executive database schema")
    if (
        version == 0
        and connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
    ):
        raise StoreError("unsupported_schema", "Unversioned nonempty executive database")
    return version


def migrate(connection: sqlite3.Connection) -> None:
    """Apply all pending DDL atomically, leaving committed history interpretable."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        version = schema_version(connection)
        for target in range(version + 1, SCHEMA_VERSION + 1):
            for statement in _MIGRATIONS[target - 1]:
                connection.execute(statement)
            if target == 2:
                # Backfill indexes from v1 records; JSON is read, never rewritten.
                for identifier, kind, payload in connection.execute(
                    "SELECT id,kind,payload FROM records WHERE kind IN ('signal','receipt')"
                ):
                    data = json.loads(payload)
                    if kind == "signal":
                        connection.execute(
                            "INSERT INTO signal_keys VALUES(?,?,?,?,?)",
                            (
                                identifier,
                                data["principal_id"],
                                data["producer_id"],
                                data.get("external_id"),
                                data.get("producer_sequence"),
                            ),
                        )
                    else:
                        connection.executemany(
                            "INSERT INTO receipt_events VALUES(?,?)",
                            ((identifier, event_id) for event_id in data["event_ids"]),
                        )
            connection.execute(f"PRAGMA user_version={target}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
