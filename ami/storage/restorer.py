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


def stream_restore_slice(*, destination_path: str, spec: SourceTableSpec, parquet_path: str,
                          manifest: dict, expected_scientific_hash: str,
                          batch_size: int = 1_000_000) -> RestoreResult:
    """RAM-bounded restore for arbitrarily large partitions: streams the
    Parquet in batches straight into a new minimal SQLite file, computing
    the ordered scientific-content hash incrementally as it inserts.
    Raises before touching anything if the destination is not approved.
    Verifies the manifest hash matches the expected hash up front, and the
    accumulated insert hash at the end -- no separate full re-read."""
    import pyarrow.parquet as pq

    _validate_destination(destination_path)
    if manifest.get("ordered_scientific_content_hash") != expected_scientific_hash:
        raise RestoreManifestMismatchError("manifest scientific-content hash does not match expected")

    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    conn = sqlite3.connect(destination_path)
    hasher = hashlib.sha256()
    total = 0
    try:
        col_defs = []
        for col in spec.preserved_columns:
            sql_type = spec.source_types[col]
            nullable = "" if col in spec.nullable_columns else " NOT NULL"
            pk = " PRIMARY KEY" if col == spec.stable_ordering_field else ""
            col_defs.append(f"{col} {sql_type}{pk}{nullable}")
        conn.execute(f"CREATE TABLE {spec.table}_restored ({', '.join(col_defs)})")
        placeholders = ",".join("?" for _ in spec.preserved_columns)
        insert_sql = f"INSERT INTO {spec.table}_restored VALUES ({placeholders})"
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=batch_size):
            d = batch.to_pydict()
            cols = [d[c] for c in spec.preserved_columns]
            n = len(cols[0]) if cols else 0
            batch_rows = [tuple(cols[j][i] for j in range(len(spec.preserved_columns))) for i in range(n)]
            conn.executemany(insert_sql, batch_rows)
            for r in batch_rows:
                serialized = "\x1f".join(repr(v) for v in r)
                hasher.update((("" if total == 0 else "\x1e") + serialized).encode())
                total += 1
        conn.commit()
        restored_count = conn.execute(f"SELECT COUNT(*) FROM {spec.table}_restored").fetchone()[0]
    finally:
        conn.close()

    restored_hash = hasher.hexdigest()
    if restored_hash != expected_scientific_hash:
        raise RestoreManifestMismatchError(
            f"restored content hash {restored_hash} does not match expected {expected_scientific_hash}")
    if restored_count != total:
        raise RestoreManifestMismatchError(f"restored count {restored_count} != streamed {total}")
    return RestoreResult(destination_path=destination_path, row_count=restored_count,
                          scientific_content_hash=restored_hash)


def stream_restore_slice_multi(*, destination_path: str, spec: SourceTableSpec, parquet_paths: list[str],
                                manifest: dict, expected_scientific_hash: str,
                                batch_size: int = 1_000_000) -> RestoreResult:
    """Multi-shard RAM-bounded restore (for partitions published by
    `ami.storage.sharded_archive`): streams each shard's Parquet, in the
    given order, into ONE new minimal SQLite file, maintaining a single
    running hash/count across shard boundaries -- never holds more than
    one batch (from one shard at a time) in memory, and never
    concatenates shards into one in-memory structure first. Identical
    validation and hash convention to `stream_restore_slice`; a
    single-shard call (`parquet_paths` of length 1) produces the exact
    same result as that function."""
    import pyarrow.parquet as pq

    _validate_destination(destination_path)
    if manifest.get("ordered_scientific_content_hash") != expected_scientific_hash:
        raise RestoreManifestMismatchError("manifest scientific-content hash does not match expected")

    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    conn = sqlite3.connect(destination_path)
    hasher = hashlib.sha256()
    total = 0
    try:
        col_defs = []
        for col in spec.preserved_columns:
            sql_type = spec.source_types[col]
            nullable = "" if col in spec.nullable_columns else " NOT NULL"
            pk = " PRIMARY KEY" if col == spec.stable_ordering_field else ""
            col_defs.append(f"{col} {sql_type}{pk}{nullable}")
        conn.execute(f"CREATE TABLE {spec.table}_restored ({', '.join(col_defs)})")
        placeholders = ",".join("?" for _ in spec.preserved_columns)
        insert_sql = f"INSERT INTO {spec.table}_restored VALUES ({placeholders})"
        for parquet_path in parquet_paths:
            pf = pq.ParquetFile(parquet_path)
            for batch in pf.iter_batches(batch_size=batch_size):
                d = batch.to_pydict()
                cols = [d[c] for c in spec.preserved_columns]
                n = len(cols[0]) if cols else 0
                batch_rows = [tuple(cols[j][i] for j in range(len(spec.preserved_columns))) for i in range(n)]
                conn.executemany(insert_sql, batch_rows)
                for r in batch_rows:
                    serialized = "\x1f".join(repr(v) for v in r)
                    hasher.update((("" if total == 0 else "\x1e") + serialized).encode())
                    total += 1
        conn.commit()
        restored_count = conn.execute(f"SELECT COUNT(*) FROM {spec.table}_restored").fetchone()[0]
    finally:
        conn.close()

    restored_hash = hasher.hexdigest()
    if restored_hash != expected_scientific_hash:
        raise RestoreManifestMismatchError(
            f"restored content hash {restored_hash} does not match expected {expected_scientific_hash}")
    if restored_count != total:
        raise RestoreManifestMismatchError(f"restored count {restored_count} != streamed {total}")
    return RestoreResult(destination_path=destination_path, row_count=restored_count,
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
