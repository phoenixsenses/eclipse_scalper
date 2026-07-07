"""Read-only source access layer (Phase 6). Every connection opened here
is `mode=ro` with `query_only=ON` and an authorizer that denies every
write-capable SQLite action -- proven by dedicated rejection tests using
disposable fixtures, never exercised against the live database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_SOURCE_PATH = Path(__file__).resolve().parents[2] / "data" / "microstructure.db"

_WRITE_ACTIONS = {
    sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE, sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_VIEW, sqlite3.SQLITE_DROP_VIEW, sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_TRANSACTION,
}
# Writable PRAGMAs to deny; read-only PRAGMAs (e.g. table_info, index_list) are permitted.
_WRITABLE_PRAGMA_PREFIXES = ("journal_mode", "synchronous", "wal_checkpoint", "auto_vacuum",
                             "user_version", "application_id", "foreign_keys", "locking_mode")


class SourceMutationRejected(Exception):
    """Raised (via the authorizer, translated to a Python exception at
    the call site) whenever a write-capable action is attempted through
    a connection opened by this module."""


class RejectionLog(list):
    """A plain list subclass so tests can assert both emptiness and
    exact denied-action tuples without a separate accessor."""


def _make_authorizer(log: RejectionLog):
    def authorizer(action, arg1, arg2, dbname, trigger):
        if action in _WRITE_ACTIONS:
            log.append((action, arg1, arg2))
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA and arg1 and arg1.lower() in _WRITABLE_PRAGMA_PREFIXES:
            log.append((action, arg1, arg2))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return authorizer


def open_read_only(path: str | Path = DEFAULT_SOURCE_PATH, *, timeout: float = 10.0
                    ) -> tuple[sqlite3.Connection, RejectionLog]:
    """Opens `path` as `mode=ro`, sets `query_only=ON` (a second,
    independent layer of write-rejection beyond the authorizer -- SQLite
    itself refuses any write attempt at the connection level under this
    pragma), and installs a denial-logging authorizer. Returns the
    connection and the (initially empty) rejection log; callers should
    assert the log stays empty for the lifetime of a legitimate read-only
    session."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    conn.execute("PRAGMA query_only=ON")
    log = RejectionLog()
    conn.set_authorizer(_make_authorizer(log))
    return conn, log


def assert_read_only_session_clean(log: RejectionLog) -> None:
    """Raises SourceMutationRejected if any write was attempted (and
    denied) during the session -- a legitimate read-only exporter should
    never trigger even one entry."""
    if log:
        raise SourceMutationRejected(f"{len(log)} write attempt(s) denied during a read-only session: {log}")
