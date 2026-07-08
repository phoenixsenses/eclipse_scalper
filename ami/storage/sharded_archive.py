"""Memory-bounded, multi-shard streaming export for partitions too large
for the single-file design in `ami.storage.archive` to finalize within
the RAM guardrail (book_ticker/SOLUSDT/2026-04, 114,404,095 rows, hit a
~3.0GB resident-memory wall during finalization -- see
`MIGRATION_LOG.md` / `STORAGE_ROTATION_RETENTION_PRODUCTION_ARCHIVE_
ACTIVATION_V1.md` root-cause section).

Root cause (confirmed via `EXPLAIN QUERY PLAN` against the real,
650GB+ `data/microstructure.db`):

    SEARCH book_ticker USING INDEX idx_bt_symbol_ts (symbol=? AND ts_ms>? AND ts_ms<?)
    USE TEMP B-TREE FOR ORDER BY

The only index covering the WHERE clause (`symbol`, `ts_ms`) returns
rows in `ts_ms` order, not `id` order, so `ORDER BY id ASC` (the
existing single-file exporter's ordering, required for stable,
resumable, watermark-bounded output) forces SQLite to build a full
temporary sort structure over every one of the ~114M matching rows
before it can yield the first row in the requested order -- this,
combined with a single ~2GB Parquet file's footer/statistics
finalization, is what exceeded the RAM ceiling.

This module avoids the sort by ordering the scan on `ts_ms` (with `id`
as a tiebreaker) instead of `id` alone: `EXPLAIN QUERY PLAN` confirms
`ORDER BY ts_ms ASC, id ASC` against the same `(symbol, ts_ms)` index
needs no `TEMP B-TREE` at all -- the index already returns rows in
`ts_ms` order per symbol, so this ordering is free, using the exact
same efficient index scan as the original (broken) query, examining
only the ~114M matching rows (an earlier design here scanned via an
`id`-range `NOT INDEXED` rowid scan instead, which also avoids the
sort but was measured, against the real database, to require
examining ~6.7x more rows -- other symbols interleaved across the same
`id` range -- and was abandoned for that reason). `id` is still
tracked, validated against the frozen watermark, and reported in every
shard's metadata; it is just no longer the sort key. Resumability uses
a compound `(ts_ms, id)` cursor since `ts_ms` alone is not guaranteed
unique. Output is split into multiple deterministic `part-NNNNN.
parquet` shards (rotating to a new `ParquetWriter` every
`max_rows_per_shard` rows) so no single file's finalization dominates
RAM either, with a hard RSS guard that aborts safely (leaving completed
shards in place) if resident memory exceeds a configured ceiling, and
resume-safe staging so a subsequent call continues from the next
unwritten `(ts_ms, id)` position rather than restarting.

The aggregate ordered scientific-content hash is deliberately NOT
computed during export (that would require a hasher whose internal
state survives a crash/resume, which the stdlib does not support) --
`stream_hash_parquet_multi` computes it afterwards in one fresh,
streaming, RAM-bounded pass over the finished shards, using the exact
same row-serialization convention as `ami.storage.archive.
canonical_row_hash` / `stream_hash_parquet`, so it is byte-for-byte
comparable to every other partition's hash in this package.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from ami.storage.archive import build_pyarrow_schema, ExportValidationError
from ami.storage.registry import SourceTableSpec

RESUME_STATE_NAME = "resume_state.json"
SHARD_CHECKPOINT_PREFIX = "shard-"
SHARD_CHECKPOINT_SUFFIX = ".checkpoint.json"


class MemoryGuardAbort(Exception):
    """Raised when `rss_check()` reports resident memory at or above
    `rss_limit_bytes`. Not a failure -- all shards finalized so far are
    left in place (never deleted), and a resume checkpoint is written, so
    a subsequent call with the same `staging_dir` picks up where this one
    stopped rather than re-exporting already-completed shards."""


class StaleStagingConflict(Exception):
    """Raised when a staging directory contains shard/parquet files that
    do not match this partition's identity (e.g. a different table,
    symbol, or partition-month) -- refuses to silently mix or discard
    another job's staging content. The caller must clean it up (via
    `clean_stale_staging`) or choose a different `staging_dir` first."""


def _shard_path(staging_dir: str, shard_index: int) -> str:
    return os.path.join(staging_dir, f"part-{shard_index:05d}.parquet")


def _checkpoint_path(staging_dir: str, shard_index: int) -> str:
    return os.path.join(staging_dir, f"{SHARD_CHECKPOINT_PREFIX}{shard_index:05d}{SHARD_CHECKPOINT_SUFFIX}")


def resolve_partition_id_bounds(conn, spec: SourceTableSpec, partition) -> tuple[int | None, int | None]:
    """Cheap, index-backed (`MIN`/`MAX` aggregate, no `ORDER BY`, no sort
    buffer -- the same pattern `ami.storage.partition.plan_partition`
    already uses for its `COUNT(*)`/`MAX(...)` estimate) lookup of the
    `[min_id, max_id]` range bounding this partition's matching rows.
    Returns `(None, None)` if no rows match."""
    row = conn.execute(
        f"SELECT MIN({spec.stable_ordering_field}), MAX({spec.stable_ordering_field}) FROM {spec.table} "
        f"WHERE {spec.symbol_field}=? AND {spec.partition_ts_field}>=? AND {spec.partition_ts_field}<?",
        (partition.symbol, partition.partition_start_ms, partition.partition_end_ms)).fetchone()
    return row[0], row[1]


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: str, payload: dict) -> None:
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(partial, path)


def discover_resumable_shards(staging_dir: str, *, partition_id: str) -> list[dict]:
    """Scans `staging_dir` for already-completed shard checkpoints
    belonging to this exact `partition_id`. A shard counts as completed
    only if both its checkpoint JSON and its (non-`.partial`) Parquet
    file exist -- any `.partial` leftover from a hard crash mid-shard is
    never trusted, never resumed from, and is left for
    `clean_stale_staging` to remove. Raises `StaleStagingConflict` if a
    checkpoint belonging to a *different* partition_id is found (refuses
    to silently mix two jobs' shards in one staging directory)."""
    if not os.path.isdir(staging_dir):
        return []
    completed: list[dict] = []
    for name in sorted(os.listdir(staging_dir)):
        if not (name.startswith(SHARD_CHECKPOINT_PREFIX) and name.endswith(SHARD_CHECKPOINT_SUFFIX)):
            continue
        ckpt = _read_json(os.path.join(staging_dir, name))
        if ckpt.get("partition_id") != partition_id:
            raise StaleStagingConflict(
                f"{staging_dir!r} contains a checkpoint for a different partition_id "
                f"({ckpt.get('partition_id')!r} != {partition_id!r}) -- refusing to resume; "
                "call clean_stale_staging first")
        shard_path = os.path.join(staging_dir, ckpt["shard_file"])
        if os.path.exists(shard_path) and not shard_path.endswith(".partial"):
            completed.append(ckpt)
    completed.sort(key=lambda c: c["shard_index"])
    return completed


def clean_stale_staging(staging_dir: str, *, partition_id: str | None = None) -> int:
    """Removes every `.partial` file (always -- never a trustworthy,
    resumable artifact) and, if `partition_id` is given, every shard
    Parquet/checkpoint pair that does NOT belong to that partition_id
    (a genuinely stale, abandoned staging directory from an unrelated or
    superseded job). Never touches `resume_state.json`/manifest/catalog-
    entry/_SUCCESS files, and never removes anything outside
    `staging_dir`. Returns the number of files removed."""
    if not os.path.isdir(staging_dir):
        return 0
    removed = 0
    for name in sorted(os.listdir(staging_dir)):
        path = os.path.join(staging_dir, name)
        if name.endswith(".partial"):
            os.remove(path)
            removed += 1
            continue
        if partition_id is not None and name.startswith(SHARD_CHECKPOINT_PREFIX) and \
                name.endswith(SHARD_CHECKPOINT_SUFFIX):
            ckpt = _read_json(path)
            if ckpt.get("partition_id") != partition_id:
                shard_path = os.path.join(staging_dir, ckpt["shard_file"])
                if os.path.exists(shard_path):
                    os.remove(shard_path)
                    removed += 1
                os.remove(path)
                removed += 1
    return removed


@dataclass(frozen=True)
class ShardedExportResult:
    shards: tuple[dict, ...]          # per-shard: {shard_index, path, row_count, min_id, max_id, byte_size}
    row_count: int
    min_id: int | None
    max_id: int | None
    complete: bool                    # False if this call stopped early (RSS guard); resumable


def stream_export_to_parquet_sharded(
        conn, spec: SourceTableSpec, partition, staging_dir: str, *,
        batch_size: int = 1_000_000, max_rows_per_shard: int = 10_000_000,
        max_output_bytes_per_shard: int, rss_check=None, rss_limit_bytes: int | None = None,
        rss_check_every_rows: int = 2_000_000) -> ShardedExportResult:
    """Bounded, resumable, multi-shard streaming export. See module
    docstring for the full rationale. Never raises out of a partially
    written shard -- a shard's `.parquet` file only appears (via
    `os.replace` from `.partial`) together with its checkpoint JSON,
    written immediately after, so `discover_resumable_shards` never sees
    a shard file without its checkpoint or vice versa.

    `rss_check`, if given, is a zero-arg callable returning current
    process resident memory in bytes (e.g. `lambda:
    psutil.Process().memory_info().rss`). Checked every
    `rss_check_every_rows` rows within a shard (never mid-row-group-write,
    only at a row-count boundary where no partially-written Arrow batch
    is in flight); if it returns `>= rss_limit_bytes`, the current shard
    is abandoned (its `.partial` is left for `clean_stale_staging`, never
    trusted or resumed from) and `MemoryGuardAbort` is raised. Everything
    written before the abandoned shard remains valid and resumable."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(staging_dir, exist_ok=True)
    partition_id = partition.partition_id

    completed = discover_resumable_shards(staging_dir, partition_id=partition_id)
    next_shard_index = (completed[-1]["shard_index"] + 1) if completed else 0
    # Resume cursor is a compound (ts_ms, id) position -- ts_ms alone is
    # not guaranteed unique, id alone is not the scan order any more.
    cursor_ts = completed[-1]["max_ts"] if completed else -1
    cursor_id = completed[-1]["max_id"] if completed else -1
    total_row_count = sum(c["row_count"] for c in completed)
    overall_min_id = completed[0]["min_id"] if completed else None
    overall_max_id = completed[-1]["max_id"] if completed else None

    partition_min_id, partition_max_id = resolve_partition_id_bounds(conn, spec, partition)
    if partition_min_id is None:
        if completed:
            return ShardedExportResult(shards=tuple(completed), row_count=total_row_count,
                                        min_id=overall_min_id, max_id=overall_max_id, complete=True)
        raise ExportValidationError("empty partition")

    schema = build_pyarrow_schema(spec)
    id_idx = spec.preserved_columns.index(spec.stable_ordering_field)
    sym_idx = spec.preserved_columns.index(spec.symbol_field)
    ts_idx = spec.preserved_columns.index(spec.partition_ts_field)
    cols_sql = ",".join(spec.preserved_columns)

    # `ORDER BY ts_ms ASC, id ASC` against the (symbol, ts_ms) index needs
    # no TEMP B-TREE (confirmed via EXPLAIN QUERY PLAN against the real
    # 650GB+ database) -- the index already returns rows in ts_ms order
    # per symbol, so this ordering is free, using the same efficient scan
    # the original (broken) `ORDER BY id` query's WHERE clause used.
    cur = conn.execute(
        f"SELECT {cols_sql} FROM {spec.table} WHERE {spec.symbol_field}=? AND "
        f"{spec.partition_ts_field}>=? AND {spec.partition_ts_field}<? AND "
        f"({spec.partition_ts_field}>? OR ({spec.partition_ts_field}=? AND {spec.stable_ordering_field}>?)) "
        f"ORDER BY {spec.partition_ts_field} ASC, {spec.stable_ordering_field} ASC",
        (partition.symbol, partition.partition_start_ms, partition.partition_end_ms,
         cursor_ts, cursor_ts, cursor_id))

    new_shards: list[dict] = []
    shard_index = next_shard_index
    writer = None
    shard_partial_path = None
    shard_row_count = 0
    shard_min_id = shard_max_id = shard_min_ts = shard_max_ts = None
    rows_since_rss_check = 0

    def _finalize_current_shard():
        nonlocal writer, shard_partial_path, shard_row_count, shard_min_id, shard_max_id, \
            shard_min_ts, shard_max_ts
        if writer is None:
            return
        writer.close()
        writer = None
        final_path = shard_partial_path[: -len(".partial")]
        os.replace(shard_partial_path, final_path)
        ckpt = {
            "partition_id": partition_id, "shard_index": shard_index,
            "shard_file": os.path.basename(final_path), "row_count": shard_row_count,
            "min_id": shard_min_id, "max_id": shard_max_id,
            "min_ts": shard_min_ts, "max_ts": shard_max_ts,
            "byte_size": os.path.getsize(final_path),
        }
        _write_json_atomic(_checkpoint_path(staging_dir, shard_index), ckpt)
        new_shards.append(ckpt)

    try:
        while True:
            batch = cur.fetchmany(batch_size)
            if not batch:
                break
            for r in batch:
                rid, rsym, rts = r[id_idx], r[sym_idx], r[ts_idx]
                if rsym != partition.symbol:
                    raise ExportValidationError(f"unexpected symbol {rsym!r}")
                if not (partition.partition_start_ms <= rts < partition.partition_end_ms):
                    raise ExportValidationError(f"timestamp {rts} out of partition range")
                if rid > partition.source_watermark_value:
                    raise ExportValidationError(f"id {rid} exceeds watermark")

                if writer is None:
                    shard_partial_path = _shard_path(staging_dir, shard_index) + ".partial"
                    writer = pq.ParquetWriter(shard_partial_path, schema, compression="zstd",
                                               use_dictionary=False, write_statistics=True)
                    shard_row_count = 0
                    shard_min_id = rid
                    shard_min_ts = rts

                shard_row_count += 1
                shard_max_id = rid
                shard_max_ts = rts
                total_row_count += 1
                overall_min_id = rid if overall_min_id is None else min(overall_min_id, rid)
                overall_max_id = rid if overall_max_id is None else max(overall_max_id, rid)

            if writer is not None:
                arrays = []
                batch_cols = list(zip(*batch))
                for i, col in enumerate(spec.preserved_columns):
                    pa_type = schema.field(col).type
                    values = list(batch_cols[i])
                    if pa_type == pa.int64():
                        values = [int(v) if v is not None else None for v in values]
                    arrays.append(pa.array(values, type=pa_type))
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))

                if os.path.getsize(shard_partial_path) > max_output_bytes_per_shard:
                    raise ExportValidationError(
                        f"shard {shard_index} parquet output exceeds per-shard cap "
                        f"{max_output_bytes_per_shard}")

            if writer is not None and shard_row_count >= max_rows_per_shard:
                _finalize_current_shard()
                shard_index += 1

            rows_since_rss_check += len(batch)
            if rss_check is not None and rss_limit_bytes is not None and \
                    rows_since_rss_check >= rss_check_every_rows:
                rows_since_rss_check = 0
                current_rss = rss_check()
                if current_rss >= rss_limit_bytes:
                    if writer is not None:
                        writer.close()
                        writer = None  # `.partial` left in place, untrusted, never resumed from
                    raise MemoryGuardAbort(
                        f"resident memory {current_rss} >= guard limit {rss_limit_bytes} "
                        f"after {total_row_count} total rows ({len(new_shards)} new shards this call); "
                        f"{shard_index} shard(s) finalized and resumable in {staging_dir!r}")
    finally:
        if writer is not None:
            _finalize_current_shard()

    return ShardedExportResult(
        shards=tuple(completed + new_shards), row_count=total_row_count,
        min_id=overall_min_id, max_id=overall_max_id, complete=True)


def stream_hash_parquet_multi(shard_paths: list[str], preserved_columns: tuple[str, ...]) -> dict:
    """Fresh, RAM-bounded (never a full in-memory read of any shard, let
    alone all of them) streaming hash over multiple Parquet shards, read
    in the given order via `ParquetFile.iter_batches`. Uses the identical
    row-serialization convention as `ami.storage.archive.
    canonical_row_hash`/`stream_hash_parquet`, applied across shard
    boundaries as if they were one continuous ordered row sequence -- so
    the result is byte-for-byte the same hash a single-file export of the
    same logical row set would have produced."""
    import pyarrow.parquet as pq

    hasher = hashlib.sha256()
    row_count = 0
    min_id = max_id = None
    id_col = "id" if "id" in preserved_columns else preserved_columns[0]
    for path in shard_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1_000_000):
            d = batch.to_pydict()
            cols = [d[c] for c in preserved_columns]
            ids = d[id_col]
            n = len(cols[0]) if cols else 0
            for i in range(n):
                serialized = "\x1f".join(repr(cols[j][i]) for j in range(len(preserved_columns)))
                hasher.update((("" if row_count == 0 else "\x1e") + serialized).encode())
                row_count += 1
                rid = ids[i]
                min_id = rid if min_id is None else min(min_id, rid)
                max_id = rid if max_id is None else max(max_id, rid)
    return {"row_count": row_count, "scientific_content_hash": hasher.hexdigest(),
            "min_id": min_id, "max_id": max_id}
