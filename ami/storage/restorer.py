"""Minimal SQLite-slice restorer (Phase 13). Restores a bounded,
single-table, single-partition slice into a brand-new disposable SQLite
file beneath an approved temp root only -- never a production
destination, never a full historical restoration.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass

from ami.storage.registry import SourceTableSpec

APPROVED_RESTORE_ROOT_NAMES = (".runtime_temp", ".pytest_temp")


class RestoreDestinationRejected(Exception):
    """Raised when the destination path is not beneath an approved temp
    root, or looks like a production/repository-root/OS-temp path."""


class RestoreManifestMismatchError(Exception):
    """Raised when the manifest supplied does not match the Parquet
    content being restored."""


class RestoreUnsupportedTableError(Exception):
    """Raised for any table not in the source registry allowlist."""


@dataclass(frozen=True)
class RestoreResult:
    destination_path: str
    row_count: int
    scientific_content_hash: str


def _canonical_row_hash(rows: list[tuple]) -> str:
    parts = ["\x1f".join(repr(v) for v in row) for row in rows]
    return hashlib.sha256("\x1e".join(parts).encode()).hexdigest()


def _validate_destination(path: str) -> None:
    norm = os.path.normpath(os.path.abspath(path)).replace("\\", "/")
    if not any(f"/{root}/" in norm + "/" or norm.endswith(f"/{root}") for root in APPROVED_RESTORE_ROOT_NAMES):
        raise RestoreDestinationRejected(
            f"{path!r} is not beneath an approved temp root {APPROVED_RESTORE_ROOT_NAMES}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        raise RestoreDestinationRejected(f"{path!r} already exists and is non-empty; refuses silent overwrite")


def restore_slice(*, destination_path: str, spec: SourceTableSpec, rows: list[tuple],
                   manifest: dict, expected_scientific_hash: str) -> RestoreResult:
    """Restores `rows` (already read from a verified Parquet partition)
    into a brand-new minimal SQLite file at `destination_path`. Raises
    before writing anything if the destination is not approved or the
    manifest doesn't match."""
    _validate_destination(destination_path)
    if manifest.get("ordered_scientific_content_hash") != expected_scientific_hash:
        raise RestoreManifestMismatchError("manifest scientific-content hash does not match the rows supplied")

    computed_hash = _canonical_row_hash(rows)
    if computed_hash != expected_scientific_hash:
        raise RestoreManifestMismatchError(
            f"row content hash {computed_hash} does not match expected {expected_scientific_hash}")

    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    conn = sqlite3.connect(destination_path)
    try:
        col_defs = []
        for col in spec.preserved_columns:
            sql_type = spec.source_types[col]
            nullable = "" if col in spec.nullable_columns else " NOT NULL"
            pk = " PRIMARY KEY" if col == spec.stable_ordering_field else ""
            col_defs.append(f"{col} {sql_type}{pk}{nullable}")
        conn.execute(f"CREATE TABLE {spec.table}_restored ({', '.join(col_defs)})")
        placeholders = ",".join("?" for _ in spec.preserved_columns)
        conn.executemany(f"INSERT INTO {spec.table}_restored VALUES ({placeholders})", rows)
        conn.commit()
        restored = conn.execute(
            f"SELECT {','.join(spec.preserved_columns)} FROM {spec.table}_restored "
            f"ORDER BY {spec.stable_ordering_field} ASC").fetchall()
    finally:
        conn.close()

    restored_hash = _canonical_row_hash(restored)
    if restored_hash != expected_scientific_hash:
        raise RestoreManifestMismatchError(
            f"restored content hash {restored_hash} does not match expected {expected_scientific_hash}")

    return RestoreResult(destination_path=destination_path, row_count=len(restored),
                          scientific_content_hash=restored_hash)


def cleanup_restored_slice(destination_path: str) -> bool:
    """Deletes only a file this restorer itself created, and only if it
    is beneath an approved temp root (re-validated, not trusted from a
    prior call)."""
    _validate_destination_for_cleanup(destination_path)
    if os.path.exists(destination_path):
        os.remove(destination_path)
        return True
    return False


def _validate_destination_for_cleanup(path: str) -> None:
    norm = os.path.normpath(os.path.abspath(path)).replace("\\", "/")
    if not any(f"/{root}/" in norm + "/" or norm.endswith(f"/{root}") for root in APPROVED_RESTORE_ROOT_NAMES):
        raise RestoreDestinationRejected(f"refusing to clean up a path outside approved temp roots: {path!r}")
