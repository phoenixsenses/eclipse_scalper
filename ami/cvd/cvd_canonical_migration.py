"""AMI CVD REPAIR + WINDOWED TAKER-FLOW -- controlled canonical migration/
backfill entry point (schema 11->12).

Composes the already-validated rehearsal DDL (folded verbatim into
ami.warehouse.schema's init_schema() as _SCHEMA_PHASE_CVD) with a
frozen-source-package backfill: every row this function writes is copied,
verbatim and order-preserving, from the disposable rehearsal database
produced by BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1
(data/ami/cvd_rehearsal_disposable_20260705/cvd_rehearsal_disposable.sqlite).
This function performs NO network call and NO recomputation -- it is a
content-identical copy, so post-migration content hashes are provably
byte-identical to the frozen rehearsal values (see
S34_CVD_SCHEMA_11_TO_12_MIGRATION_PROPOSAL_2026-07-05.md §2 step 4).

NOT_CALLED_AUTOMATICALLY: `run_canonical_migration()` takes explicit
connections -- it is never invoked as an import side effect (geometry
migration precedent, ami/geometry/birth_truncated_geometry_canonical_
migration.py).

conn: writable connection to canonical.sqlite (or its disposable copy),
already schema-migrated (caller runs ami.warehouse.schema.init_schema(conn)
first -- schema application and data backfill stay two separately-auditable
steps, same precedent).
source_ro: READ-ONLY connection to the frozen disposable rehearsal database.
"""
from __future__ import annotations

import sqlite3

from ami.cvd import cvd_source_quality_contract_v1 as quality
from ami.cvd import windowed_taker_flow as wtf


class FrozenSourceRowConflict(Exception):
    """A row copied from the frozen source package collided with an existing
    canonical row under different content. This can only happen if the
    migration is run twice against two DIFFERENT frozen packages -- a hard
    stop, never silently resolved."""


_TABLE_COPY_PLAN = (
    # (disposable_source_table, canonical_target_table, columns_in_order)
    ("ami_agg_trades_repaired_stage", "ami_agg_trades_repaired", (
        "symbol", "agg_trade_id", "ts_ms", "retrieved_at_ms", "price", "quantity",
        "notional", "signed_quantity", "signed_notional", "is_buyer_maker",
        "taker_side", "first_trade_id", "last_trade_id", "source_regime_id",
        "retrieval_batch_id", "retrieval_page_index", "source_provenance",
        "source_quality_status", "legacy_match_status", "legacy_match_fingerprint",
        "superseded_by_batch_id", "data_version_id", "created_ms")),
    ("ami_cvd_repair_batch_ledger", "ami_cvd_repair_batch_ledger", (
        "retrieval_batch_id", "symbol", "requested_start_ms", "requested_end_ms",
        "pagination_method", "page_count", "row_count", "first_agg_trade_id",
        "last_agg_trade_id", "earliest_trade_ts_ms", "latest_trade_ts_ms",
        "page_overlap_rows", "missing_id_ranges", "request_errors", "truncation_flag",
        "content_sha256", "gap_manifest_sha256", "duplicate_manifest_sha256",
        "exact_reconstruction_verdict", "data_version_id", "created_ms")),
    ("ami_cvd_windowed_flow", "ami_cvd_windowed_flow", (
        "feature_id", "feature_definition_version", "raw_interpretation_version",
        "quality_contract_version", "signal_id", "source_event_id",
        "independent_cycle_id", "symbol", "signal_birth_ts", "window_id",
        "window_start_ts_ms", "window_end_ts_ms", "evidence_layer",
        "source_row_count", "legacy_row_count", "repair_row_count", "cvd_qty",
        "cvd_notional", "total_notional", "taker_buy_qty", "taker_sell_qty",
        "taker_buy_notional", "taker_sell_notional", "normalized_cvd",
        "source_row_manifest_sha256", "source_regime_ids", "repair_method",
        "repair_population_version", "feature_available_ts_ms",
        "known_at_classification", "schema_version", "provenance", "created_ms")),
    ("ami_cvd_windowed_flow_proxy", "ami_cvd_windowed_flow_proxy", (
        "feature_id", "feature_definition_version", "quality_contract_version",
        "signal_id", "source_event_id", "independent_cycle_id", "symbol",
        "signal_birth_ts", "window_id", "window_start_ts_ms", "window_end_ts_ms",
        "evidence_layer", "candle_timeframe", "contained_candle_count",
        "last_contained_close_ts_ms", "proxy_cvd_qty", "proxy_taker_buy_qty",
        "proxy_taker_sell_qty", "candle_versions", "source_row_manifest_sha256",
        "feature_available_ts_ms", "known_at_classification", "descriptive_only",
        "schema_version", "provenance", "created_ms")),
    ("ami_cvd_bucket_exclusions", "ami_cvd_bucket_exclusions", (
        "exclusion_id", "feature_definition_version", "signal_id", "direction",
        "reason", "schema_version", "created_ms")),
    ("ami_cvd_window_quality_v1", "ami_cvd_window_quality_v1", (
        "quality_id", "quality_contract_version", "assessment_version", "signal_id",
        "independent_cycle_id", "symbol", "signal_birth_ts", "window_id",
        "window_start_ts_ms", "window_end_ts_ms", "evidence_layer",
        "source_regime_ids", "regime_spanning", "legacy_row_count", "repair_row_count",
        "total_row_count", "duplicate_count", "collision_count", "unresolved_match_count",
        "missing_minute_count", "repaired_minute_count", "cadence_proof",
        "completeness_proof", "quality_status", "feature_available_ts_ms",
        "source_provenance", "data_version_id", "feature_definition_version",
        "assessed_at_ms")),
)

_PK_COLUMNS = {
    "ami_agg_trades_repaired": ("symbol", "agg_trade_id", "retrieval_batch_id"),
    "ami_cvd_repair_batch_ledger": ("retrieval_batch_id",),
    "ami_cvd_windowed_flow": ("feature_id",),
    "ami_cvd_windowed_flow_proxy": ("feature_id",),
    "ami_cvd_bucket_exclusions": ("exclusion_id",),
    "ami_cvd_window_quality_v1": ("quality_id",),
}


def run_canonical_migration(conn: sqlite3.Connection, source_ro: sqlite3.Connection,
                             *, provenance: str = "cvd-repair-rehearsal-canonical-migration-v1") -> dict:
    """Idempotent, content-identical copy from the frozen disposable rehearsal
    database into the (already schema-migrated) canonical connection.

    Idempotency: rerunning against the SAME frozen source is a content-compare
    NOOP for every row (never a duplicate insert, never an overwrite). A
    same-identity row with DIFFERENT content raises FrozenSourceRowConflict
    (fail-closed -- never silently overwritten).
    """
    del provenance  # rows are copied verbatim; provenance already lives in each row
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
    """The three frozen content hashes the migration proposal requires to
    byte-compare against the rehearsal's own values (exact/proxy/quality)."""
    return {
        "exact": wtf.content_hash_exact(conn),
        "proxy": wtf.content_hash_proxy(conn),
        "quality": quality.content_hash(conn),
    }


def row_counts(conn: sqlite3.Connection) -> dict:
    return {dst: conn.execute(f"SELECT COUNT(*) FROM {dst}").fetchone()[0]
            for _, dst, _ in _TABLE_COPY_PLAN}
