"""AMI FAM_CASCADE_ABSORPTION_IMPACT -- controlled canonical migration/backfill
entry point (schema 12->13, M-0035).

Composes the already-validated rehearsal DDL (folded verbatim, plus FK/CHECK
additions only, into ami.warehouse.schema's init_schema() as
_SCHEMA_PHASE_ABSORPTION_IMPACT) with a frozen-source-package backfill: every
row this function writes is copied, verbatim and order-preserving, from the
retained disposable rehearsal database produced by
BATCH-CASCADE-ABSORPTION-IMPACT-DISPOSABLE-REHEARSAL-V1 (commit fc43e972) and
row-accounted by BATCH-CASCADE-ABSORPTION-IMPACT-ROW-ACCOUNTING-FREEZE-V1
(commit 931cd3dd):
D:/eclipse_scalper/.runtime_temp/absorption_impact_rehearsal_v1/rehearsal_run1.sqlite.

This function performs NO network call and NO recomputation -- it is a
content-identical copy (source table names differ from the canonical target
names only by the operator's `ami_absorption_impact_*` naming ruling; no
column, value, or row-identity change), so post-migration content hashes are
provably byte-identical (modulo the frozen bookkeeping-column exclusion
already established in the rehearsal/freeze) to the frozen rehearsal values.

NOT_CALLED_AUTOMATICALLY: `run_canonical_migration()` takes explicit
connections -- it is never invoked as an import side effect (CVD/geometry
migration precedent).

conn: writable connection to canonical.sqlite (or its disposable copy),
already schema-migrated (caller runs ami.warehouse.schema.init_schema(conn)
first -- schema application and data backfill stay two separately-auditable
steps, same precedent).
source_ro: READ-ONLY connection to the frozen retained rehearsal database.
"""
from __future__ import annotations

import hashlib
import sqlite3


class FrozenSourceRowConflict(Exception):
    """A row copied from the frozen source package collided with an existing
    canonical row under different content. This can only happen if the
    migration is run twice against two DIFFERENT frozen packages -- a hard
    stop, never silently resolved."""


# (disposable_source_table [fc43e972 naming], canonical_target_table
#  [operator ami_absorption_impact_* naming ruling], columns in order)
_TABLE_COPY_PLAN = (
    ("absorption_impact_windowed_flow", "ami_absorption_impact_windowed_flow", (
        "feature_id", "feature_definition_version", "signal_id", "source_event_id",
        "independent_cycle_id", "symbol", "direction", "signal_birth_ts", "window_id",
        "window_start_ts_ms", "window_end_ts_ms", "trade_count", "native_rows_used",
        "repaired_rows_used", "signed_notional", "total_notional", "mark_price_start",
        "mark_price_end", "mark_return_bps", "floor_usd_m_applied", "floor_usd_m_value",
        "price_response_per_signed_notional", "evidence_layer", "feature_available_ts_ms",
        "known_at_classification", "created_ms")),
    ("absorption_impact_window_quality_v1", "ami_absorption_impact_window_quality_v1", (
        "quality_id", "quality_contract_version", "signal_id", "symbol", "window_id",
        "window_start_ts_ms", "window_end_ts_ms", "evidence_layer", "quality_status",
        "confirmed_gap_overlap", "unresolved_gap_overlap", "before_collection_began",
        "repaired_rows_used", "native_rows_used", "assessed_at_ms")),
    ("absorption_impact_exclusions", "ami_absorption_impact_exclusions", (
        "exclusion_id", "signal_id", "symbol", "window_id", "reason_code", "created_ms")),
)

_PK_COLUMNS = {
    "ami_absorption_impact_windowed_flow": ("feature_id",),
    "ami_absorption_impact_window_quality_v1": ("quality_id",),
    "ami_absorption_impact_exclusions": ("exclusion_id",),
}

# Bookkeeping-only columns excluded from content hashing -- identical
# discipline to the rehearsal's own content_hash_of_disposable() /
# ami/warehouse/experiment_ledger.py's _VOLATILE_BOOKKEEPING_COLUMNS.
_CONTENT_COLUMNS = {
    "ami_absorption_impact_windowed_flow": (
        "feature_id, feature_definition_version, signal_id, source_event_id, "
        "independent_cycle_id, symbol, direction, signal_birth_ts, window_id, "
        "window_start_ts_ms, window_end_ts_ms, trade_count, native_rows_used, "
        "repaired_rows_used, signed_notional, total_notional, mark_price_start, "
        "mark_price_end, mark_return_bps, floor_usd_m_applied, floor_usd_m_value, "
        "price_response_per_signed_notional, evidence_layer, feature_available_ts_ms, "
        "known_at_classification"
    ),
    "ami_absorption_impact_window_quality_v1": (
        "quality_id, quality_contract_version, signal_id, symbol, window_id, "
        "window_start_ts_ms, window_end_ts_ms, evidence_layer, quality_status, "
        "confirmed_gap_overlap, unresolved_gap_overlap, before_collection_began, "
        "repaired_rows_used, native_rows_used"
    ),
    "ami_absorption_impact_exclusions": "exclusion_id, signal_id, symbol, window_id, reason_code",
}


def run_canonical_migration(conn: sqlite3.Connection, source_ro: sqlite3.Connection,
                             *, provenance: str = "absorption-impact-canonical-migration-v1") -> dict:
    """Idempotent, content-identical copy from the frozen retained rehearsal
    database into the (already schema-migrated) canonical connection.

    Idempotency: rerunning against the SAME frozen source is a content-compare
    NOOP for every row (never a duplicate insert, never an overwrite). A
    same-identity row with DIFFERENT content raises FrozenSourceRowConflict
    (fail-closed -- never silently overwritten).
    """
    del provenance  # rows are copied verbatim; no provenance column exists in this family's schema
    counts = {}
    for src_table, dst_table, cols in _TABLE_COPY_PLAN:
        col_list = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        pk_cols = _PK_COLUMNS[dst_table]
        src_rows = source_ro.execute(f"SELECT {col_list} FROM {src_table}").fetchall()
        inserted = 0
        noop = 0
        for raw in src_rows:
            row = tuple(raw)
            existing = conn.execute(
                f"SELECT {col_list} FROM {dst_table} WHERE " +
                " AND ".join(f"{c}=?" for c in pk_cols),
                tuple(row[cols.index(c)] for c in pk_cols)).fetchone()
            if existing is not None:
                if tuple(existing) != row:
                    raise FrozenSourceRowConflict(
                        f"{dst_table}: {dict(zip(pk_cols, (row[cols.index(c)] for c in pk_cols)))} "
                        "collision with different content")
                noop += 1
                continue
            conn.execute(f"INSERT INTO {dst_table} ({col_list}) VALUES ({placeholders})", row)
            inserted += 1
        counts[dst_table] = {"inserted": inserted, "noop_identical": noop, "source_rows": len(src_rows)}
    conn.commit()
    return counts


def content_hashes(conn: sqlite3.Connection) -> dict:
    """The three frozen content hashes the row-accounting freeze requires to
    byte-compare against the rehearsal's own values (windowed_flow/quality/
    exclusions), bookkeeping timestamp columns excluded -- same computation
    as the rehearsal's content_hash_of_disposable()."""
    hashes = {}
    for _, dst_table, _ in _TABLE_COPY_PLAN:
        cols = _CONTENT_COLUMNS[dst_table]
        rows = conn.execute(f"SELECT {cols} FROM {dst_table} ORDER BY 1").fetchall()
        hashes[dst_table] = hashlib.sha256("|".join(str(r) for r in rows).encode()).hexdigest()
    return hashes


def row_counts(conn: sqlite3.Connection) -> dict:
    return {dst: conn.execute(f"SELECT COUNT(*) FROM {dst}").fetchone()[0]
            for _, dst, _ in _TABLE_COPY_PLAN}
