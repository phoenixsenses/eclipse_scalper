"""BATCH-STORAGE-ROTATION-RETENTION-READINESS-AND-CONTRACT-V1.

Operational-readiness design module only. Defines the table-retention
classification registry, the archive/manifest contract constants, and a
pure, deterministic storage-health state function.

This module performs NO destructive action and NO database mutation of
any kind: it never calls VACUUM, DELETE, UPDATE, DROP, or any table-write
statement, and it never opens a live database connection at all (every
function here is pure -- it takes already-collected metrics as plain
Python values, never a database path or connection). Any future archive/
purge implementation is explicitly out of scope (see
`FUTURE_COMPONENTS` at the bottom, which is metadata only, not code).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Retention classes (Phase 3)
# ---------------------------------------------------------------------------

RETENTION_CLASSES = frozenset({
    "CANONICAL_IMMUTABLE",
    "CONTINUITY_CRITICAL_ACTIVE",
    "RESEARCH_CRITICAL_COMPACT",
    "RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE",
    "DERIVED_REBUILDABLE",
    "TEMPORARY_DISPOSABLE",
})

# One row per database/table root actually found in this batch's read-only
# inventory (Phase 1/2). `evidence` cites the exact bounded query or
# accepted report this batch used -- nothing here is guessed.
TABLE_REGISTRY: tuple[dict, ...] = (
    # --- data/ami/canonical.sqlite (governed research warehouse) ---
    {"db": "canonical.sqlite", "table": "ami_signal_lifecycle", "class": "CANONICAL_IMMUTABLE",
     "reason": "canonical anchor identity; every research family depends on it",
     "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "ami_events", "class": "CANONICAL_IMMUTABLE",
     "reason": "canonical cascade-event identity", "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "ami_cycles", "class": "CANONICAL_IMMUTABLE",
     "reason": "canonical independent-cycle identity", "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "ami_cvd_windowed_flow", "class": "CANONICAL_IMMUTABLE",
     "reason": "canonical feature family (CVD), immutable per experiment_ledger contract",
     "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "ami_absorption_impact_windowed_flow", "class": "CANONICAL_IMMUTABLE",
     "reason": "canonical feature family (absorption)", "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "ami_book_spread_change_windowed_flow", "class": "CANONICAL_IMMUTABLE",
     "reason": "canonical feature family (book-spread), M-0036", "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "ami_lifecycle_path_observations", "class": "CANONICAL_IMMUTABLE",
     "reason": "canonical outcome (endpoint_return_bps/mfe_bps), all experiments depend on it",
     "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "experiment_registry", "class": "CANONICAL_IMMUTABLE",
     "reason": "governance record of every experiment ever registered", "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "experiment_results", "class": "CANONICAL_IMMUTABLE",
     "reason": "governance record of every scientific result", "purge_eligible": False},
    {"db": "canonical.sqlite", "table": "researcher_exposure_ledger", "class": "CANONICAL_IMMUTABLE",
     "reason": "TEST-exposure accounting, never deletable", "purge_eligible": False},
    # --- data/ami/knowledge.sqlite ---
    {"db": "knowledge.sqlite", "table": "failure_archive", "class": "CANONICAL_IMMUTABLE",
     "reason": "graveyard of record; deleting it defeats the epistemic-gate discipline",
     "purge_eligible": False},
    {"db": "knowledge.sqlite", "table": "epistemic_test_nullifiers", "class": "CANONICAL_IMMUTABLE",
     "reason": "single-use TEST-evidence ledger, append-only by contract", "purge_eligible": False},
    {"db": "knowledge.sqlite", "table": "experiment_gate_receipts", "class": "CANONICAL_IMMUTABLE",
     "reason": "preregistration/execution receipts", "purge_eligible": False},
    {"db": "knowledge.sqlite", "table": "graveyard_slash_fingerprints", "class": "CANONICAL_IMMUTABLE",
     "reason": "graveyard enforcement seed data", "purge_eligible": False},
    {"db": "knowledge.sqlite", "table": "epistemic_authorization_tokens", "class": "CANONICAL_IMMUTABLE",
     "reason": "retry/supersession authorization audit trail", "purge_eligible": False},
    # --- data/microstructure.db (raw collector estate) ---
    {"db": "microstructure.db", "table": "agg_trades", "class": "RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE",
     "reason": "raw aggregate-trade tick stream, ~393.6M rows, 142.1 day span; source for the repaired "
               "ami_agg_trades_repaired canonical table (already materialized separately)",
     "purge_eligible": "CONDITIONAL -- only after the CVD/absorption canonical repair layer is proven "
                        "to no longer require raw replay for the same window"},
    {"db": "microstructure.db", "table": "book_ticker", "class": "RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE",
     "reason": "raw L1 best-bid/ask stream, ~4.79B rows (all symbols, id-space upper bound), 87.0 day span; "
               "source for FAM_BOOK_SPREAD_DYNAMICS (M-0036 already materialized the canonical 196/324/128 "
               "feature rows -- raw replay is not required for the already-migrated population, but IS "
               "required for the still-growing LONG-sample-accrual path)",
     "purge_eligible": "CONDITIONAL -- blocked while FAM_BOOK_SPREAD_DYNAMICS LONG is parked for sample "
                        "growth (needs future anchors' pre-birth windows, which requires the raw stream "
                        "to remain queryable up to each future anchor's birth time)"},
    {"db": "microstructure.db", "table": "mark_prices", "class": "RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE",
     "reason": "raw mark-price/funding-rate stream, ~21.1M rows, 142.1 day span",
     "purge_eligible": "CONDITIONAL -- funding-rate history is otherwise dead (funding_rates table stale "
                        "since 2026-04-13); mark_prices is the only remaining funding-adjacent raw source"},
    {"db": "microstructure.db", "table": "liquidations", "class": "RESEARCH_CRITICAL_COMPACT",
     "reason": "raw liquidation events, ~1.33M rows, 140.8 day span -- the direct anchor source for "
               "ami_events/ami_signal_lifecycle (cascade detection); compact enough to keep active "
               "indefinitely, not a high-frequency-archive candidate by volume",
     "purge_eligible": False},
    {"db": "microstructure.db", "table": "open_interest", "class": "RESEARCH_CRITICAL_COMPACT",
     "reason": "OI raw series, small (~row count not independently counted this batch, PK has no "
               "AUTOINCREMENT id; date range 101.2 days), live collector; blocks OPEN_INTEREST family "
               "eligibility until coverage improves -- must not be purged while that recheck is pending",
     "purge_eligible": False},
    {"db": "microstructure.db", "table": "funding_rates", "class": "RESEARCH_CRITICAL_COMPACT",
     "reason": "small (178 rows per prior accepted audit), source dead since 2026-04-13 but still the "
               "only historical funding-level record", "purge_eligible": False},
    {"db": "microstructure.db", "table": "spot_prices", "class": "RESEARCH_CRITICAL_COMPACT",
     "reason": "spot-price series, live, 122.1 day span; FAM_SPOT_PERP_BASIS_REVERSAL depends on its "
               "continued forward accrual to clear the coverage block", "purge_eligible": False},
    {"db": "microstructure.db", "table": "event_diary", "class": "RESEARCH_CRITICAL_COMPACT",
     "reason": "compact (~134K rows), narrative/event audit trail", "purge_eligible": False},
    {"db": "microstructure.db", "table": "gaps", "class": "CONTINUITY_CRITICAL_ACTIVE",
     "reason": "source-gap ledger (812 rows: 20 agg_trades, 741 liquidations, 51 mark_prices) -- "
               "required by every repair/quality-contract module", "purge_eligible": False},
    {"db": "microstructure.db", "table": "detector_heartbeat", "class": "CONTINUITY_CRITICAL_ACTIVE",
     "reason": "collector liveness state", "purge_eligible": False},
    {"db": "microstructure.db", "table": "detector_log", "class": "DERIVED_REBUILDABLE",
     "reason": "detector diagnostic log, not a research dependency", "purge_eligible": "FUTURE -- age-based, low priority"},
    {"db": "microstructure.db", "table": "detector_signals", "class": "DERIVED_REBUILDABLE",
     "reason": "legacy detector signal output, superseded by ami_signal_lifecycle",
     "purge_eligible": "FUTURE -- requires confirming no active consumer remains"},
    {"db": "microstructure.db", "table": "basis_reversion_candidates", "class": "DERIVED_REBUILDABLE",
     "reason": "ungoverned exploratory candidate table, not a canonical dependency",
     "purge_eligible": "FUTURE -- requires confirming no active consumer remains"},
    {"db": "microstructure.db", "table": "liq_heatmap", "class": "DERIVED_REBUILDABLE",
     "reason": "derived visualization aggregate, reproducible from liquidations",
     "purge_eligible": "FUTURE -- requires confirming no active consumer remains"},
    {"db": "microstructure.db", "table": "sol_s35_candidates", "class": "DERIVED_REBUILDABLE",
     "reason": "ungoverned exploratory candidate table, not a canonical dependency",
     "purge_eligible": "FUTURE -- requires confirming no active consumer remains"},
    {"db": "microstructure.db", "table": "vol_state", "class": "CONTINUITY_CRITICAL_ACTIVE",
     "reason": "live volatility-regime state used by active detectors", "purge_eligible": False},
    # --- ancillary raw databases ---
    {"db": "funding_history.db", "table": "*", "class": "RESEARCH_CRITICAL_COMPACT",
     "reason": "48MB, dead collector since 2026-05-12, still the only historical record of this source",
     "purge_eligible": False},
    {"db": "oi_history.db", "table": "*", "class": "RESEARCH_CRITICAL_COMPACT",
     "reason": "9MB, dead collector since 2026-05-14", "purge_eligible": False},
    {"db": "paper_trades.db", "table": "*", "class": "RESEARCH_CRITICAL_COMPACT",
     "reason": "70KB, paper-trade evidence", "purge_eligible": False},
    {"db": "risk_state.db", "table": "*", "class": "CONTINUITY_CRITICAL_ACTIVE",
     "reason": "24KB, live risk-engine state", "purge_eligible": False},
    {"db": "s34_feature_factory.db", "table": "*", "class": "DERIVED_REBUILDABLE",
     "reason": "1.5MB, ungoverned derived feature cache", "purge_eligible": "FUTURE"},
    {"db": "s34_intelligence.db", "table": "*", "class": "CONTINUITY_CRITICAL_ACTIVE",
     "reason": "28MB, live shadow/dashboard state, actively written", "purge_eligible": False},
    # --- stray test scratch files (data/test_*.db, 416 files, ~13MB total) ---
    {"db": "data/test_s34_gates_micro_*.db", "table": "*", "class": "TEMPORARY_DISPOSABLE",
     "reason": "pytest-created scratch DB, real-shaped fixture written directly under data/ instead of "
               "--basetemp (pre-existing test-hygiene gap, disclosed not fixed this batch); 51 files",
     "purge_eligible": "FUTURE -- separate, low-risk cleanup batch, not this one"},
    {"db": "data/test_s34_gates_trades_*.db", "table": "*", "class": "TEMPORARY_DISPOSABLE",
     "reason": "same as above; 50 files", "purge_eligible": "FUTURE"},
    {"db": "data/test_s34_micro_*.db", "table": "*", "class": "TEMPORARY_DISPOSABLE",
     "reason": "same as above; 169 files", "purge_eligible": "FUTURE"},
    {"db": "data/test_s34_old_micro_*.db", "table": "*", "class": "TEMPORARY_DISPOSABLE",
     "reason": "same as above; 44 files", "purge_eligible": "FUTURE"},
    {"db": "data/test_s34_report_micro_*.db", "table": "*", "class": "TEMPORARY_DISPOSABLE",
     "reason": "same as above; 50 files", "purge_eligible": "FUTURE"},
    {"db": "data/test_s34_report_trades_*.db", "table": "*", "class": "TEMPORARY_DISPOSABLE",
     "reason": "same as above; 50 files", "purge_eligible": "FUTURE"},
    {"db": "data/test_status_snapshot_*.db", "table": "*", "class": "TEMPORARY_DISPOSABLE",
     "reason": "same as above; 2 files", "purge_eligible": "FUTURE"},
    {"db": "data/test_tmp_logger.db", "table": "*", "class": "TEMPORARY_DISPOSABLE",
     "reason": "same as above; 1 file", "purge_eligible": "FUTURE"},
)


def classify(db: str, table: str) -> dict | None:
    for row in TABLE_REGISTRY:
        if row["db"] == db and row["table"] == table:
            return row
    return None


def unclassified_tables(known_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Fail-closed helper: given a list of (db, table) pairs actually
    observed in the estate, returns any pair NOT present in
    TABLE_REGISTRY -- an unclassified table must never be silently
    treated as purge-eligible."""
    registered = {(r["db"], r["table"]) for r in TABLE_REGISTRY}
    return [p for p in known_pairs if p not in registered]


# ---------------------------------------------------------------------------
# Storage policy constants (operator ruling, this batch)
# ---------------------------------------------------------------------------

ACTIVE_RAW_HORIZON_DAYS_INITIAL = 30  # operator ruling: prefer the safer 30-day window
ACTIVE_RAW_HORIZON_DAYS_MIN_FUTURE_RANGE = (14, 30)  # future amendment may narrow within this range
ARCHIVE_FORMAT = "PARQUET"
ARCHIVE_COMPRESSION = "ZSTD"
FORBIDDEN_ARCHIVE_FORMATS = frozenset({"CSV", "JSONL", "PICKLE", "COMPRESSED_SQLITE_COPY"})
PARTITION_GRANULARITY = "CLOSED_UTC_CALENDAR_MONTH"

STORAGE_HEALTH_STATES = frozenset({
    "STORAGE_HEALTHY", "STORAGE_WARNING", "STORAGE_CRITICAL", "STORAGE_EMERGENCY",
    "ARCHIVE_LAGGING", "ARCHIVE_VERIFICATION_FAILED",
    "PURGE_BLOCKED_BY_RESEARCH_DEPENDENCY", "PURGE_BLOCKED_BY_SOURCE_REPAIR",
    "PURGE_BLOCKED_BY_CONTINUITY", "STORAGE_STATE_UNKNOWN",
})

# Proposed thresholds (design only; not wired to any automation in this batch)
THRESHOLD_WARNING_PCT_FREE = 20.0
THRESHOLD_CRITICAL_PCT_FREE = 10.0
THRESHOLD_EMERGENCY_PCT_FREE = 5.0
THRESHOLD_WARNING_ABS_FREE_GB = 200.0
THRESHOLD_CRITICAL_ABS_FREE_GB = 100.0
THRESHOLD_EMERGENCY_ABS_FREE_GB = 50.0


def storage_health_state(pct_free: float, abs_free_gb: float) -> str:
    """Pure function -- no I/O, no side effects. Determines the coarse
    drive-capacity health state from two already-measured numbers. Never
    triggers or authorizes any deletion by itself; a caller (a future,
    separately-authorized runtime component) decides what to DO with the
    returned state. Fails toward the more severe state when the two
    inputs disagree (fail-closed, never fail-open toward HEALTHY)."""
    if pct_free <= THRESHOLD_EMERGENCY_PCT_FREE or abs_free_gb <= THRESHOLD_EMERGENCY_ABS_FREE_GB:
        return "STORAGE_EMERGENCY"
    if pct_free <= THRESHOLD_CRITICAL_PCT_FREE or abs_free_gb <= THRESHOLD_CRITICAL_ABS_FREE_GB:
        return "STORAGE_CRITICAL"
    if pct_free <= THRESHOLD_WARNING_PCT_FREE or abs_free_gb <= THRESHOLD_WARNING_ABS_FREE_GB:
        return "STORAGE_WARNING"
    return "STORAGE_HEALTHY"


# Permitted / prohibited automated responses per state (design only)
PERMITTED_AUTOMATED_RESPONSE = {
    "STORAGE_HEALTHY": ("none required",),
    "STORAGE_WARNING": ("operator notification",),
    "STORAGE_CRITICAL": ("operator notification", "block optional expensive batch jobs"),
    "STORAGE_EMERGENCY": ("operator notification", "block optional expensive batch jobs",
                           "pause nonessential derived-data builds under a separately authorized runtime policy"),
}
PROHIBITED_AUTOMATED_RESPONSE = (
    "automatic deletion of any row", "automatic archival purge without a verified manifest",
    "automatic VACUUM", "automatic collector stop/restart", "silent overwrite of raw/canonical/governance evidence",
)


# ---------------------------------------------------------------------------
# Failure-mode table (Phase 13) -- design metadata, not executable logic
# ---------------------------------------------------------------------------

FAILURE_MODES: tuple[dict, ...] = tuple(
    {"failure": f, "response": "FAIL_CLOSED", "deletion_permitted": False} for f in (
        "parquet_write_failure", "partial_file", "manifest_write_failure", "checksum_mismatch",
        "row_count_mismatch", "schema_mismatch", "duplicate_rows", "missing_rows", "late_arriving_rows",
        "source_gap_discovered_after_archive", "repair_discovered_after_archive",
        "collector_restart_during_export", "wal_growth_during_export", "disk_full_during_export",
        "disk_full_during_purge", "interrupted_chunk_delete", "archive_file_missing",
        "archive_corruption", "restore_mismatch", "incompatible_archive_schema",
        "unsupported_parquet_reader", "active_research_dependency", "unknown_table_classification",
        "unknown_timestamp_semantics",
    )
)


# ---------------------------------------------------------------------------
# Future implementation components (Phase 14) -- metadata only, not built
# ---------------------------------------------------------------------------

FUTURE_COMPONENTS: tuple[str, ...] = (
    "storage_policy_configuration", "table_registry", "archive_planner",
    "read_only_source_snapshot_exporter", "parquet_writer", "manifest_writer", "verifier",
    "restore_tester", "purge_planner", "chunked_purger", "storage_health_reporter",
    "archive_catalog", "cli_dry_run_command", "cli_apply_command", "scheduled_runner_integration",
    "parquet_research_reader", "minimal_slice_restorer",
)
