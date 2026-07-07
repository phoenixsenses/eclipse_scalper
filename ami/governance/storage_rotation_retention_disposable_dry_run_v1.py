"""BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1.

Reusable exporter/verifier logic for a bounded, non-production, non-
destructive archive rehearsal. This module is the narrowly-required
implementation surfaced by that batch -- it was exercised for real
against the live `microstructure.db` (`mark_prices`/ETHUSDT/2026-05,
260,657 rows) with a disposable driver script kept outside version
control (`.runtime_temp/storage_rotation_dry_run_v1/driver.py`); the
real run's results are recorded verbatim in
`reports/research/s34/../STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1.json`.

Every function here is either pure (hashing, validation, manifest
construction, verdict determination) or takes an already-open read-only
connection/already-fetched rows -- none of it independently decides to
open a live database, and none of it ever issues a write-capable SQL
statement (`INSERT`/`UPDATE`/`DELETE`/`CREATE`/`DROP`/`ALTER`/`VACUUM`
never appear in this module's source, enforced by a dedicated test).
"""
from __future__ import annotations

import hashlib

# ---------------------------------------------------------------------------
# Frozen selection identity (Phase 2) -- from the real dry-run
# ---------------------------------------------------------------------------

SELECTED_TABLE = "mark_prices"
SELECTED_SYMBOL = "ETHUSDT"
SELECTED_VENUE = "BINANCE_USDM_PERP"
SELECTED_MARKET_SEGMENT = "PERPETUAL_FUTURES"
PARTITION_START_MS = 1777593600000  # 2026-05-01T00:00:00Z
PARTITION_END_MS = 1780272000000    # 2026-06-01T00:00:00Z (half-open)
COLUMNS = ("id", "ts_ms", "symbol", "mark_price", "funding_rate", "next_funding_time_ms")

# Rejected candidates (Phase 2 -- recorded, not silently switched)
REJECTED_CANDIDATES = (
    {"table": "agg_trades", "reason": "larger row count (~393.6M total, ~2-3M/month estimated) than "
                                       "necessary for a first rehearsal; mark_prices provides the smallest "
                                       "safe end-to-end test per the operator's stated preference order"},
    {"table": "book_ticker", "reason": "largest table (~4.79B rows total); reserved for a later, "
                                        "larger-scale rehearsal once the mark_prices contract is proven"},
)

# Real dry-run results (frozen, from the actual run against the live DB)
FROZEN_RESULT = {
    "watermark_id": 13265132,
    "row_count": 260657,
    "min_id": 12483110,
    "max_id": 13265132,
    "min_ts": 1777593604001,
    "max_ts": 1780271978008,
    "duplicate_id_count": 0,
    "scientific_content_hash": "228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999",
    "parquet_sha256": "6f91914400dcbe84b662c9260a24f9e5eb7f56b2d9db34adfde55a53af8e900f",
    "parquet_size_bytes": 2476313,
    "compression_ratio": 5.052485691429153,
    "run_a_run_b_byte_identical": True,
}


def canonical_row_hash(rows: list[tuple]) -> str:
    """Deterministic ordered scientific-content hash over already-sorted
    (by id) row tuples. The same construction used by every canonical
    migration this session (per-field repr() joined by U+001F, records
    joined by U+001E, sha256) -- reused, not reinvented."""
    parts = ["\x1f".join(repr(v) for v in row) for row in rows]
    blob = "\x1e".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()


def build_parquet_schema_dict() -> dict:
    """Frozen archive schema contract (Phase 4), as a plain dict (no
    pyarrow import required to inspect it -- keeps this module importable
    even in a context where pyarrow is unavailable, e.g. for the pure
    classification/verdict tests)."""
    return {
        "id": {"sql_type": "INTEGER", "parquet_type": "int64", "nullable": False},
        "ts_ms": {"sql_type": "INTEGER", "parquet_type": "int64", "nullable": False},
        "symbol": {"sql_type": "TEXT", "parquet_type": "string", "nullable": False},
        "mark_price": {"sql_type": "REAL", "parquet_type": "double", "nullable": False},
        "funding_rate": {"sql_type": "REAL", "parquet_type": "double", "nullable": True},
        "next_funding_time_ms": {"sql_type": "INTEGER", "parquet_type": "int64", "nullable": True},
    }


def validate_partition_rows(rows: list[tuple], *, watermark_id: int,
                             expected_symbol: str = SELECTED_SYMBOL,
                             partition_start_ms: int = PARTITION_START_MS,
                             partition_end_ms: int = PARTITION_END_MS) -> dict:
    """Phase 7 validation, pure -- takes already-fetched rows (id, ts_ms,
    symbol, ...) and checks every required invariant before a caller
    would be permitted to publish. Never itself decides to publish."""
    ids = [r[0] for r in rows]
    ts_vals = [r[1] for r in rows]
    symbols = {r[2] for r in rows}
    dup_count = len(ids) - len(set(ids))
    checks = {
        "row_count": len(rows),
        "duplicate_count": dup_count,
        "symbol_only_selected": symbols <= {expected_symbol},
        "all_ts_in_range": all(partition_start_ms <= t < partition_end_ms for t in ts_vals),
        "all_ids_le_watermark": all(i <= watermark_id for i in ids),
        "min_id": min(ids) if ids else None,
        "max_id": max(ids) if ids else None,
    }
    checks["all_pass"] = (
        checks["duplicate_count"] == 0 and checks["symbol_only_selected"]
        and checks["all_ts_in_range"] and checks["all_ids_le_watermark"] and checks["row_count"] > 0
    )
    return checks


def build_manifest(*, row_count: int, watermark_id: int, min_id: int, max_id: int,
                    min_ts: int, max_ts: int, scientific_hash: str, parquet_path: str,
                    parquet_size: int, parquet_sha256: str, source_schema_hash: str,
                    parquet_schema_hash: str, export_cutoff: str, publication_timestamp: str,
                    verification_status: str, dry_run_identity: str) -> dict:
    """Phase 8 manifest, matching the accepted 29+-field storage contract
    (see the accepted readiness batch's Phase 6). `production_status` and
    `purge_authorization` are hardcoded -- a disposable rehearsal manifest
    can never claim production status or purge eligibility."""
    return {
        "archive_contract_version": "v1",
        "dry_run_identity": dry_run_identity,
        "production_status": "DISPOSABLE_NOT_PRODUCTION",
        "source_database_identity": "microstructure.db",
        "source_table": SELECTED_TABLE,
        "utc_partition_start": "2026-05-01T00:00:00Z",
        "utc_partition_end": "2026-06-01T00:00:00Z",
        "symbol": SELECTED_SYMBOL, "venue": SELECTED_VENUE, "market_segment": SELECTED_MARKET_SEGMENT,
        "export_cutoff": export_cutoff,
        "source_watermark_field": "id", "source_watermark_value": watermark_id,
        "row_count": row_count, "min_source_id": min_id, "max_source_id": max_id,
        "min_receipt_ts": min_ts, "max_receipt_ts": max_ts,
        "source_schema_hash": source_schema_hash, "parquet_schema_hash": parquet_schema_hash,
        "ordered_scientific_content_hash": scientific_hash,
        "parquet_path": parquet_path, "parquet_logical_size": parquet_size, "parquet_sha256": parquet_sha256,
        "duplicate_count": 0,
        "exporter_version": "dry-run-v1", "verification_status": verification_status,
        "purge_authorization": "PROHIBITED",
        "publication_timestamp": publication_timestamp,
    }


def detect_corruption(*, original_sha256: str, candidate_sha256: str,
                       original_content_hash: str, candidate_content_hash: str | None) -> dict:
    """Phase 13, pure comparison. `candidate_content_hash=None` models an
    unreadable/corrupt Parquet file (content hash could not even be
    computed) -- treated as a mismatch, fail-closed."""
    hash_differs = candidate_sha256 != original_sha256
    content_differs = candidate_content_hash != original_content_hash
    return {
        "file_hash_differs": hash_differs,
        "content_hash_differs": content_differs,
        "corruption_detected": hash_differs or content_differs,
        "eligible_for_verified_status": not (hash_differs or content_differs),
    }


def determine_dry_run_verdict(*, tooling_available: bool, all_validations_passed: bool,
                               direct_read_parity: bool, restore_parity: bool,
                               run_a_run_b_scientific_match: bool, interruption_safe: bool,
                               corruption_detected_correctly: bool, live_source_writes: int,
                               source_rows_deleted: int, resource_limits_respected: bool) -> str:
    """Phase 18 verdict, fail-closed precedence: tooling absence is
    checked first (a hard blocker independent of everything else);
    any live-source write or source-row deletion is an automatic
    non-COMPLETE regardless of how good every other signal looks."""
    if not tooling_available:
        return "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_BLOCKED_BY_TOOLING"
    if live_source_writes > 0 or source_rows_deleted > 0:
        return "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_INCOMPLETE"
    if not resource_limits_respected:
        return "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_BLOCKED_BY_RESOURCE_LIMIT"
    if all((all_validations_passed, direct_read_parity, restore_parity,
            run_a_run_b_scientific_match, interruption_safe, corruption_detected_correctly)):
        return "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_COMPLETE"
    return "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_INCOMPLETE"


FORBIDDEN_SQL_STATEMENTS = ("INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE", "DROP TABLE",
                            "ALTER TABLE", "VACUUM", "REINDEX")
