"""PHASE 7B-0 / 7B-0.1: Disposable migration rehearsal harness for
ami_lifecycle_path_observations + its field-level provenance closure.

DISPOSABLE_DB_ONLY / NO_CANONICAL_DB_WRITE: mirrors
ami/lifecycle/migration_rehearsal.py's exact discipline -- the real
canonical.sqlite is opened ONLY `mode=ro` (fingerprint + row counts) and via
`shutil.copy2` (a read of the source, write only to the new disposable path).
All schema/compute/rollback operations run exclusively against the
disposable copy.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from ami.lifecycle.canonical_field_provenance import init_field_provenance_schema
from ami.lifecycle.migration_rehearsal import schema_fingerprint
from ami.lifecycle.path_field_provenance import (
    PATH_PROVENANCE_VERSION,
    backfill_path_field_provenance,
    rollback_path_field_provenance,
)
from ami.lifecycle.path_metrics import fetch_signals, freeze_and_record
from ami.lifecycle.path_schema import init_path_schema, rollback_path_schema, schema_manifest


def make_disposable_copy(source_path, disposable_path) -> None:
    Path(disposable_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, disposable_path)


def row_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]


def content_hash_path_observations(conn) -> str:
    rows = conn.execute(
        "SELECT observation_id, signal_id, horizon_name, horizon_end_ts, known_at_ts, as_of_ts, "
        "observation_status, volatility_status, reference_price, reference_price_ts, "
        "effective_path_start_ts, endpoint_return_bps, mfe_bps, mae_bps, time_to_mfe_ms, "
        "time_to_mae_ms, intrabar_order_status, realized_vol_at_anchor, "
        "endpoint_return_anchor_vol_units, mfe_anchor_vol_units, mae_anchor_vol_units, "
        "horizon_outcome_class, expected_candle_count, observed_candle_count, gap_count, "
        "candle_definition_version, path_definition_version, observation_mode, schema_version "
        "FROM ami_lifecycle_path_observations ORDER BY observation_id"
    ).fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode("utf-8")).hexdigest()


def run_disposable_path_rehearsal(source_canonical_path, disposable_path) -> dict:
    report: dict = {"source_canonical_path": str(source_canonical_path), "disposable_path": str(disposable_path)}

    # 1. baseline: real DB fingerprint + counts, read-only
    conn_ro = sqlite3.connect(f"file:{source_canonical_path}?mode=ro", uri=True)
    try:
        report["schema_fingerprint_before"] = schema_fingerprint(conn_ro)
        pre_signal_count = conn_ro.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_event_count = conn_ro.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
        pre_direction_provenance_count = conn_ro.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_field_provenance WHERE field_name='direction'"
        ).fetchone()[0]
    finally:
        conn_ro.close()

    # 2. disposable copy
    make_disposable_copy(source_canonical_path, disposable_path)
    conn = sqlite3.connect(disposable_path)

    # 3. additive schema (path_observations table + the SHARED, pre-existing
    # field_provenance table -- init_field_provenance_schema is idempotent,
    # a no-op if the real source already carries it, which it does post-M-0028)
    init_path_schema(conn)
    init_field_provenance_schema(conn)
    report["schema_manifest"] = schema_manifest(conn)
    report["table_present_after_schema"] = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ami_lifecycle_path_observations'"
    ).fetchone() is not None

    # 4. compute + freeze (run 1) + field-provenance backfill (run 1)
    r1 = freeze_and_record(conn)
    signal_ids = [s["signal_id"] for s in fetch_signals(conn)]
    prov1 = backfill_path_field_provenance(conn, signal_ids)
    report["run1_summary"] = {k: v for k, v in r1.items() if k != "rows"}
    report["provenance_run1"] = prov1
    count_1 = row_count(conn)
    hash_1 = content_hash_path_observations(conn)

    # 5. rerun (run 2) -- idempotency, both path observations AND field provenance
    r2 = freeze_and_record(conn)
    prov2 = backfill_path_field_provenance(conn, signal_ids)
    report["run2_summary"] = {k: v for k, v in r2.items() if k != "rows"}
    report["provenance_run2"] = prov2
    count_2 = row_count(conn)
    hash_2 = content_hash_path_observations(conn)
    report["row_count_equal_across_reruns"] = (count_1 == count_2)
    report["content_hash_equal_across_reruns"] = (hash_1 == hash_2)
    report["row_count_run1"] = count_1
    report["row_count_run2"] = count_2
    report["provenance_rows_equal_across_reruns"] = (
        prov1["provenance_rows_actual_total"] == prov2["provenance_rows_actual_total"]
    )

    # 6. old-reader compatibility (pre-rollback)
    post_signal_count = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
    post_event_count = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
    post_direction_provenance_count = conn.execute(
        "SELECT COUNT(*) FROM ami_lifecycle_field_provenance WHERE field_name='direction'"
    ).fetchone()[0]
    report["old_reader_signal_count_unchanged"] = (pre_signal_count == post_signal_count)
    report["old_reader_event_count_unchanged"] = (pre_event_count == post_event_count)
    report["old_reader_direction_provenance_unchanged"] = (
        pre_direction_provenance_count == post_direction_provenance_count
    )

    # 7. rollback rehearsal (path_observations table + only THIS module's path_observations.*
    # provenance rows -- the shared field_provenance table and its pre-existing
    # direction/etc rows are never touched)
    rollback_path_schema(conn)
    rollback_path_field_provenance(conn)
    tables_after_rollback = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    report["rollback_removed_new_table"] = "ami_lifecycle_path_observations" not in tables_after_rollback
    report["rollback_preserved_signal_table"] = (
        conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_count
    )
    report["rollback_preserved_event_table"] = (
        conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0] == pre_event_count
    )
    report["rollback_preserved_direction_provenance"] = (
        conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_field_provenance WHERE field_name='direction'"
        ).fetchone()[0] == pre_direction_provenance_count
    )
    report["rollback_removed_path_provenance_rows"] = conn.execute(
        "SELECT COUNT(*) FROM ami_lifecycle_field_provenance WHERE field_name LIKE 'path_observations.%'"
    ).fetchone()[0] == 0

    # 8. reapply
    init_path_schema(conn)
    r3 = freeze_and_record(conn)
    prov3 = backfill_path_field_provenance(conn, signal_ids)
    report["run3_after_reapply_summary"] = {k: v for k, v in r3.items() if k != "rows"}
    report["provenance_run3_after_reapply"] = prov3
    count_3 = row_count(conn)
    hash_3 = content_hash_path_observations(conn)
    report["reapply_row_count_matches_pre_rollback"] = (count_3 == count_1)
    report["reapply_content_hash_matches_pre_rollback"] = (hash_3 == hash_1)
    report["reapply_provenance_count_matches_pre_rollback"] = (
        prov3["provenance_rows_actual_total"] == prov1["provenance_rows_actual_total"]
    )

    # 9. old-reader compatibility (post-reapply, final state)
    final_signal_count = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
    final_event_count = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
    report["old_reader_signal_count_unchanged_final"] = (final_signal_count == pre_signal_count)
    report["old_reader_event_count_unchanged_final"] = (final_event_count == pre_event_count)

    # 10. status distributions (final) -- path-validity and volatility-validity reported
    # SEPARATELY (operator lock #1), plus independent_cycle_id/source_event_id
    # denominators (counting-discipline lock #8)
    report["final_observation_status_distribution"] = r3["status_counts"]
    report["final_volatility_status_distribution"] = r3["volatility_status_counts"]
    report["final_path_valid_n"] = r3["path_valid_n"]
    report["final_volatility_valid_n"] = r3["volatility_valid_n"]
    report["final_raw_signal_n"] = r3["raw_signal_n"]
    report["final_raw_observation_row_n"] = r3["raw_observation_row_n"]
    report["final_raw_observation_row_n_expected_max"] = r3["raw_observation_row_n_expected_max"]
    report["final_distinct_independent_cycle_n"] = r3["distinct_independent_cycle_n"]
    report["final_distinct_source_event_n"] = r3["distinct_source_event_n"]
    report["final_as_of_by_symbol"] = r3["as_of_by_symbol"]
    report["final_provenance_expected_total"] = prov3["provenance_rows_expected_total"]
    report["final_provenance_actual_total"] = prov3["provenance_rows_actual_total"]
    report["final_provenance_missing"] = prov3["provenance_rows_missing"]
    report["final_provenance_duplicate_groups"] = prov3["provenance_rows_duplicate_groups"]

    conn.close()
    return report
