"""Bounded exporter, archive-schema contract, and manifest builder
(Phases 7-9), generalized across all three allowlisted tables (the
disposable dry-run, commit `6fbe0571`, only covered `mark_prices`
directly). Every export first writes a `.partial` file and is validated
before an atomic rename publishes it -- never published directly.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

from ami.storage.partition import PartitionIdentity
from ami.storage.registry import SourceTableSpec, get_table_spec

_ARCHIVE_TYPE_TO_PYARROW = {"int64": "int64()", "double": "float64()", "string": "string()"}


class ExportValidationError(Exception):
    """Raised when export output fails a required invariant -- the
    caller must not publish (rename `.partial` to final) when this is
    raised."""


class ProductionPathRejected(Exception):
    """Raised if a caller ever passes an output root that is not
    beneath an explicitly disposable directory -- this module refuses to
    write anywhere else."""


def canonical_row_hash(rows: list[tuple]) -> str:
    """Deterministic ordered scientific-content hash, identical
    construction to every canonical migration this session and to the
    disposable dry-run module (`storage_rotation_retention_disposable_
    dry_run_v1.canonical_row_hash`) -- reused, not reinvented."""
    parts = ["\x1f".join(repr(v) for v in row) for row in rows]
    return hashlib.sha256("\x1e".join(parts).encode()).hexdigest()


def _require_disposable_root(output_root: str, allowed_roots: tuple[str, ...]) -> None:
    norm = os.path.normpath(os.path.abspath(output_root)).replace("\\", "/").lower()
    for allowed in allowed_roots:
        allowed_norm = os.path.normpath(os.path.abspath(allowed)).replace("\\", "/").lower()
        if norm == allowed_norm or norm.startswith(allowed_norm + "/"):
            return
    raise ProductionPathRejected(f"{output_root!r} is not beneath an approved disposable root {allowed_roots}")


def fetch_partition_rows(conn, spec: SourceTableSpec, partition: PartitionIdentity) -> list[tuple]:
    """Read-only, bounded, index-backed, ordered by the stable ordering
    field. Excludes every row above the frozen watermark by construction
    (the `<=` predicate), and every row outside the frozen time window."""
    cols = ",".join(spec.preserved_columns)
    cur = conn.execute(
        f"SELECT {cols} FROM {spec.table} WHERE {spec.symbol_field}=? AND "
        f"{spec.partition_ts_field}>=? AND {spec.partition_ts_field}<? AND "
        f"{spec.stable_ordering_field}<=? ORDER BY {spec.stable_ordering_field} ASC",
        (partition.symbol, partition.partition_start_ms, partition.partition_end_ms,
         partition.source_watermark_value))
    return cur.fetchall()


def build_pyarrow_schema(spec: SourceTableSpec):
    """Lazily imports pyarrow (kept out of the module-level import so
    this module remains importable in a pyarrow-less test context for
    the pure-logic tests below)."""
    import pyarrow as pa
    type_map = {"int64": pa.int64(), "double": pa.float64(), "string": pa.string()}
    fields = [pa.field(col, type_map[spec.archive_types[col]], nullable=col in spec.nullable_columns)
              for col in spec.preserved_columns]
    return pa.schema(fields)


def export_partition(conn, table: str, partition: PartitionIdentity, output_root: str,
                      allowed_roots: tuple[str, ...], *, max_output_bytes: int) -> dict:
    """End-to-end bounded export: fetch -> build Arrow table -> write
    `.partial` (ZSTD) -> validate -> atomically publish. Raises
    `ExportValidationError` (never publishes) if any check fails, and
    `ProductionPathRejected` if `output_root` is not disposable."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _require_disposable_root(output_root, allowed_roots)
    spec = get_table_spec(table)
    rows = fetch_partition_rows(conn, spec, partition)
    source_hash = canonical_row_hash(rows)

    schema = build_pyarrow_schema(spec)
    cols = list(zip(*rows)) if rows else [[] for _ in spec.preserved_columns]
    arrays = []
    for i, col in enumerate(spec.preserved_columns):
        pa_type = schema.field(col).type
        values = list(cols[i]) if rows else []
        if pa_type == pa.int64():
            values = [int(v) if v is not None else None for v in values]
        arrays.append(pa.array(values, type=pa_type))
    table_obj = pa.Table.from_arrays(arrays, schema=schema)

    os.makedirs(output_root, exist_ok=True)
    partial_path = os.path.join(output_root, f"{partition.archive_relative_path.replace('/', '_')}.partial")
    pq.write_table(table_obj, partial_path, compression="zstd", use_dictionary=False, write_statistics=True)

    ids = [r[spec.preserved_columns.index(spec.stable_ordering_field)] for r in rows]
    symbols = {r[spec.preserved_columns.index(spec.symbol_field)] for r in rows}
    ts_idx = spec.preserved_columns.index(spec.partition_ts_field)
    duplicate_count = len(ids) - len(set(ids))

    checks = {
        "row_count_positive_or_empty_allowed": True,
        "duplicate_count_zero": duplicate_count == 0,
        "symbol_only_selected": symbols <= {partition.symbol},
        "all_ts_in_range": all(partition.partition_start_ms <= r[ts_idx] < partition.partition_end_ms for r in rows),
        "all_ids_le_watermark": all(i <= partition.source_watermark_value for i in ids),
        "output_size_within_cap": os.path.getsize(partial_path) <= max_output_bytes,
    }
    if not all(checks.values()):
        raise ExportValidationError(f"export validation failed: {checks}")

    final_path = partial_path[: -len(".partial")]
    os.replace(partial_path, final_path)

    return {
        "final_path": final_path, "row_count": len(rows), "scientific_content_hash": source_hash,
        "parquet_size_bytes": os.path.getsize(final_path),
        "parquet_sha256": _sha256_file(final_path), "checks": checks,
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest (Phase 9) -- 36-field contract, matching the accepted disposable
# dry-run's authoritative field set (superseding the readiness batch's
# earlier 29-field draft; both are reconciled here, not silently dropped:
# the extra 7 fields are venue/market_segment/quote_currency/gap-status/
# repair-status/exporter-version disclosures added during the dry-run).
# ---------------------------------------------------------------------------

ALLOWED_PRODUCTION_STATUS = ("DISPOSABLE_NOT_PRODUCTION", "PRODUCTION_VERIFIED")


def build_manifest(*, spec: SourceTableSpec, partition: PartitionIdentity, row_count: int,
                    scientific_hash: str, parquet_path: str, parquet_size: int, parquet_sha256: str,
                    source_schema_hash: str, parquet_schema_hash: str, unresolved_gap_count: int,
                    export_cutoff: str, publication_timestamp: str, verification_status: str,
                    dry_run_identity: str, production_status: str = "DISPOSABLE_NOT_PRODUCTION") -> dict:
    """`production_status` defaults to DISPOSABLE_NOT_PRODUCTION (the only
    value any disposable caller ever gets); the production-activation
    rehearsal is the sole caller permitted to pass PRODUCTION_VERIFIED,
    and even it can pass nothing else -- `purge_authorization` remains
    hardcoded PROHIBITED regardless, and is never a parameter."""
    if production_status not in ALLOWED_PRODUCTION_STATUS:
        raise ValueError(f"invalid production_status {production_status!r}")
    ids_present = row_count > 0
    return {
        "archive_contract_version": partition.archive_contract_version,
        "dry_run_identity": dry_run_identity,
        "production_status": production_status,
        "source_database_identity": spec.source_database,
        "source_table": spec.table,
        "utc_partition_start": _ms_to_iso(partition.partition_start_ms),
        "utc_partition_end": _ms_to_iso(partition.partition_end_ms),
        "symbol": partition.symbol, "venue": partition.venue, "market_segment": partition.market_segment,
        "quote_currency": spec.quote_currency,
        "export_cutoff": export_cutoff,
        "source_watermark_field": partition.source_watermark_field,
        "source_watermark_value": partition.source_watermark_value,
        "row_count": row_count,
        "source_schema_hash": source_schema_hash, "parquet_schema_hash": parquet_schema_hash,
        "ordered_scientific_content_hash": scientific_hash,
        "parquet_path": parquet_path, "parquet_logical_size": parquet_size, "parquet_sha256": parquet_sha256,
        "duplicate_count": 0,
        "source_gap_status": "GAPS_PRESENT" if unresolved_gap_count else "NO_UNRESOLVED_GAPS",
        "unresolved_gap_count": unresolved_gap_count,
        "repair_status": spec.repair_layer or "NOT_APPLICABLE",
        "exporter_version": "bounded-implementation-v1",
        "verification_status": verification_status,
        "direct_read_parity_status": None, "restore_parity_status": None,
        "purge_authorization": "PROHIBITED",
        "publication_timestamp": publication_timestamp,
        "partition_id": partition.partition_id,
        "table_producer": spec.producer,
        "primary_key": spec.primary_key,
        "preserved_columns": list(spec.preserved_columns),
        "nullable_columns": list(spec.nullable_columns),
    }


def _ms_to_iso(ms: int) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
