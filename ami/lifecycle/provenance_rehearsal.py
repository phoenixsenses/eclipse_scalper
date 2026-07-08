"""PHASE 7A-P1: Disposable rehearsal harness for the field-level provenance
closure (ami/lifecycle/canonical_field_provenance.py).

DISPOSABLE_DB_ONLY / NO_CANONICAL_DB_WRITE: identical discipline to
ami/lifecycle/migration_rehearsal.py -- the real canonical.sqlite is opened
ONLY read-only (schema fingerprint / row counts) or copied via
shutil.copy2. This module is kept SEPARATE from migration_rehearsal.py
(rather than extending it) so the already-validated, already-APPLIED Phase
7A migration rehearsal code is never touched by this follow-up closure.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from ami.lifecycle.canonical_backfill import backfill_lifecycle, correct_unvalidated_terminal_close
from ami.lifecycle.canonical_field_provenance import (
    FIELD_PROVENANCE_SPECS,
    PROVENANCE_VERSION,
    backfill_field_provenance,
    init_field_provenance_schema,
    rollback_field_provenance_schema,
    validate_field_provenance_complete,
)
from ami.lifecycle.canonical_schema import (
    count_effective_closed_signals,
    effective_lifecycle_status,
    init_lifecycle_schema,
    migrate_setup_version_nullable,
    rebuild_current_state,
)
from ami.lifecycle.migration_rehearsal import schema_fingerprint

# Columns of ami_signal_lifecycle that FIELD_PROVENANCE_SPECS tracks --
# used to report the ACTUAL VALUE null/non-null distribution (distinct from
# provenance-ROW presence, which is 270/270 for every field by construction).
_LIFECYCLE_VALUE_COLUMNS = list(FIELD_PROVENANCE_SPECS.keys())


def lifecycle_value_null_matrix(conn) -> dict:
    """For each field tracked by FIELD_PROVENANCE_SPECS, counts how many
    ami_signal_lifecycle rows have a NULL vs non-NULL ACTUAL VALUE. This is
    independent of whether a provenance record exists (it always does, once
    backfilled) -- it answers "what is actually stored", not "is this field
    documented"."""
    out = {}
    for field in _LIFECYCLE_VALUE_COLUMNS:
        row = conn.execute(
            f"SELECT SUM(CASE WHEN {field} IS NULL THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN {field} IS NOT NULL THEN 1 ELSE 0 END) FROM ami_signal_lifecycle"
        ).fetchone()
        out[field] = {"null": row[0] or 0, "non_null": row[1] or 0}
    return out


def make_disposable_copy(source_path, disposable_path) -> None:
    Path(disposable_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, disposable_path)


def provenance_row_counts(conn) -> dict:
    """[PHASE 7B-CANON] Scoped to provenance_version=PROVENANCE_VERSION
    ('field-provenance-v1', the 16 ami_signal_lifecycle fields this module
    backfills) -- NOT a raw COUNT(*) over the whole shared table. Since
    BATCH-P7B-CANON, ami_lifecycle_field_provenance is also written by
    ami.lifecycle.path_field_provenance (270 signals x 23 'path_observations.*'
    fields, provenance_version='path-observations-field-provenance-v1') --
    an unscoped count would silently fold that unrelated writer's rows into
    this module's own row-count/idempotency checks. Scoping by
    provenance_version keeps the two writers' bookkeeping independent
    without needing a source-table column on the shared table."""
    n = conn.execute(
        "SELECT COUNT(*) FROM ami_lifecycle_field_provenance WHERE provenance_version=?",
        (PROVENANCE_VERSION,),
    ).fetchone()[0]
    return {"ami_lifecycle_field_provenance": n}


def provenance_content_hash(conn) -> str:
    """Scoped to provenance_version=PROVENANCE_VERSION -- see
    provenance_row_counts() for why an unscoped query would be wrong now
    that the table has a second writer (ami.lifecycle.path_field_provenance)."""
    rows = conn.execute(
        "SELECT provenance_id, signal_id, field_name, field_classification, is_proxy, "
        "derivation_method, source_reference, limitations, provenance_version, schema_version "
        "FROM ami_lifecycle_field_provenance WHERE provenance_version=? ORDER BY provenance_id",
        (PROVENANCE_VERSION,),
    ).fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode("utf-8")).hexdigest()


def lifecycle_row_counts(conn) -> dict:
    n_sig = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
    n_trn = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
    return {"ami_signal_lifecycle": n_sig, "ami_lifecycle_transitions": n_trn}


def lifecycle_status_counts(conn) -> dict:
    """Denormalized ami_signal_lifecycle.lifecycle_status distribution --
    TradeLifecycleState only actually implements OPEN/CLOSED in this backfill
    (no HEALTHY/ACCELERATING/etc path labels this phase); reported as an
    exhaustive dict so an unexpected third value would be visible, not hidden."""
    rows = conn.execute(
        "SELECT lifecycle_status, COUNT(*) FROM ami_signal_lifecycle GROUP BY lifecycle_status"
    ).fetchall()
    return dict(rows)


def transition_type_counts(conn) -> dict:
    """(reason_code, new_status) -> count, across the WHOLE append-only ledger
    (never filtered/deleted) -- the exact taxonomy of what has ever been
    asserted, including any now-superseded rows (superseded rows are kept,
    never removed, only countered by a later CORRECTION row)."""
    rows = conn.execute(
        "SELECT reason_code, new_status, COUNT(*) FROM ami_lifecycle_transitions "
        "GROUP BY reason_code, new_status ORDER BY reason_code, new_status"
    ).fetchall()
    return {f"{reason_code}->{new_status}": n for reason_code, new_status, n in rows}


def effective_ledger_row_counts(conn) -> dict:
    """[PHASE 7A-P1 semantic closure, round 3] Raw (immutable, append-only)
    vs effective (superseded + pure-reversal corrections excluded) ledger row
    counts -- the two-layer contract's exact size report."""
    raw_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
    effective_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_effective_transitions").fetchone()[0]
    return {"raw_ledger_rows": raw_n, "effective_ledger_rows": effective_n}


def effective_rebuild_consistency(conn) -> dict:
    """Compares effective_lifecycle_status() (reads the effective view) against
    rebuild_current_state() (reads the raw table's latest row) for EVERY
    signal -- both must agree today (every correction so far is a pure
    reversal); this is the regression guard against a future, non-reversal
    correction silently diverging the two ledgers."""
    signal_ids = [r[0] for r in conn.execute("SELECT signal_id FROM ami_signal_lifecycle").fetchall()]
    mismatches = []
    for sid in signal_ids:
        raw = rebuild_current_state(conn, sid)
        effective = effective_lifecycle_status(conn, sid)
        if raw is None or effective is None or raw["current_status"] != effective["current_status"]:
            mismatches.append(sid)
    return {"signals_checked": len(signal_ids), "mismatches_n": len(mismatches),
            "consistent": len(mismatches) == 0}


def naive_raw_ledger_duration_contamination_check(conn, signal_ids: list[str]) -> dict:
    """[PHASE 7A-P1 semantic closure, round 3] Demonstrates, on the REAL
    (post-correction) dataset, that a naive downstream query computing a
    "hold duration" directly off the RAW ledger would fabricate a nonzero
    interval for every one of the 266 corrected signals (using the
    now-superseded TERMINAL_CLOSE row's timestamp), while the SAME query
    against the EFFECTIVE view correctly finds no CLOSED row -- hence no
    interval -- to compute a duration from."""
    contaminated_n = 0
    for sid in signal_ids:
        raw_closed = conn.execute(
            "SELECT transition_ts FROM ami_lifecycle_transitions WHERE signal_id=? AND new_status='CLOSED' "
            "ORDER BY transition_ts LIMIT 1", (sid,),
        ).fetchone()
        effective_closed = conn.execute(
            "SELECT transition_ts FROM ami_lifecycle_effective_transitions WHERE signal_id=? AND new_status='CLOSED' "
            "ORDER BY transition_ts LIMIT 1", (sid,),
        ).fetchone()
        if raw_closed is not None and effective_closed is None:
            contaminated_n += 1  # a naive raw-ledger query would fabricate an interval here
    return {"signals_with_fake_raw_closed_interval": contaminated_n,
            "signals_with_genuine_effective_closed_interval": count_effective_closed_signals(conn)}


def current_state_rebuild_consistency(conn) -> dict:
    """For every signal, verifies rebuild_current_state() (pure ledger fold)
    agrees with the denormalized ami_signal_lifecycle.lifecycle_status column
    -- this is the exact LIFECYCLE_TERMINAL_SEMANTIC_BLOCKER check: a
    consistent-but-wrong pair (both CLOSED, off an unvalidated timestamp) is
    NOT caught by this check alone -- see the accompanying null-matrix/
    transition-semantics report for that. This check catches denormalized
    cache DIVERGING from ledger truth."""
    signal_ids = [r[0] for r in conn.execute("SELECT signal_id FROM ami_signal_lifecycle").fetchall()]
    mismatches = []
    for sid in signal_ids:
        rebuilt = rebuild_current_state(conn, sid)
        stored = conn.execute(
            "SELECT lifecycle_status FROM ami_signal_lifecycle WHERE signal_id=?", (sid,)
        ).fetchone()[0]
        if rebuilt is None or rebuilt["current_status"] != stored:
            mismatches.append(sid)
    return {"signals_checked": len(signal_ids), "mismatches_n": len(mismatches),
            "consistent": len(mismatches) == 0}


def run_disposable_provenance_rehearsal(source_canonical_path, disposable_path) -> dict:
    report: dict = {"source_canonical_path": str(source_canonical_path), "disposable_path": str(disposable_path)}

    conn_ro = sqlite3.connect(f"file:{source_canonical_path}?mode=ro", uri=True)
    try:
        report["schema_version_before"] = conn_ro.execute(
            "SELECT version FROM schema_versions WHERE component='canonical_warehouse'"
        ).fetchone()[0]
        report["schema_fingerprint_before"] = schema_fingerprint(conn_ro)
        lifecycle_before = lifecycle_row_counts(conn_ro)
    finally:
        conn_ro.close()

    make_disposable_copy(source_canonical_path, disposable_path)
    conn = sqlite3.connect(disposable_path)

    # identity snapshot BEFORE the semantic correction (proves signal_id/
    # source_event_id/independent_cycle_id and the transition ledger's
    # content are byte-identical after the correction below)
    identity_before = set(conn.execute(
        "SELECT signal_id, source_event_id, independent_cycle_id FROM ami_signal_lifecycle"
    ).fetchall())
    transitions_before = set(conn.execute(
        "SELECT transition_id, signal_id, previous_status, new_status, transition_ts FROM ami_lifecycle_transitions"
    ).fetchall())

    # [PHASE 7A-P1 semantic closure, round 2 -- pre-batch baseline] this is
    # the disposable copy's UNMODIFIED starting state (== the REAL, already
    # M-0023-migrated canonical.sqlite's actual current state): the
    # LIFECYCLE_TERMINAL_SEMANTIC_BLOCKER check. lifecycle_status/ledger are
    # CONSISTENT with each other here (both agree CLOSED for 266 signals) --
    # the contradiction is semantic (CLOSED derived from the same
    # event_end_ts_ms that terminal_ts already refuses to echo as
    # DETERMINISTIC_HISTORICAL_SAFE), not a cache/ledger divergence.
    report["lifecycle_status_counts_pre_batch"] = lifecycle_status_counts(conn)
    report["transition_type_counts_pre_batch"] = transition_type_counts(conn)
    report["current_state_rebuild_consistency_pre_batch"] = current_state_rebuild_consistency(conn)
    report["lifecycle_terminal_semantic_blocker_found"] = (
        report["lifecycle_status_counts_pre_batch"].get("CLOSED", 0) > 0
    )

    # [PHASE 7A-P1 semantic closure] BEFORE the field-provenance table exists
    # (no FK complications): relax setup_version to nullable (table rebuild,
    # proposed schema v8->v9), then re-run the (corrected) lifecycle backfill
    # so the ON CONFLICT UPDATE path nulls out the existing setup_version
    # ("setup-v1" placeholder) and terminal_ts (event_end_ts_ms echo) values
    # on the 270 already-present rows -- signal_id/source_event_id/
    # independent_cycle_id and all 536 ami_lifecycle_transitions rows are
    # NEVER touched by this (verified below).
    report["setup_version_schema_migrated"] = migrate_setup_version_nullable(conn)
    # [PHASE 7A-P1 semantic closure, round 3] ensures ami_lifecycle_effective_transitions
    # exists on this disposable copy -- the copied source (real canonical.sqlite) has the
    # RAW tables (from the already-applied M-0023 migration) but never had this view
    # (round 3 is new, still unapplied to the real DB); additive/idempotent, does not
    # touch ami_signal_lifecycle/ami_lifecycle_transitions row content in any way.
    init_lifecycle_schema(conn)
    corrected_lifecycle = backfill_lifecycle(conn, conn, provenance="phase-7a-p1-semantic-closure")
    report["semantic_correction_backfill"] = corrected_lifecycle

    identity_after = set(conn.execute(
        "SELECT signal_id, source_event_id, independent_cycle_id FROM ami_signal_lifecycle"
    ).fetchall())
    transitions_after = set(conn.execute(
        "SELECT transition_id, signal_id, previous_status, new_status, transition_ts FROM ami_lifecycle_transitions"
    ).fetchall())
    report["identity_unchanged_by_semantic_correction"] = (identity_before == identity_after)
    report["transitions_unchanged_by_semantic_correction"] = (transitions_before == transitions_after)

    report["actual_value_null_matrix"] = lifecycle_value_null_matrix(conn)
    row = conn.execute(
        "SELECT SUM(CASE WHEN setup_version IS NULL THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN terminal_ts IS NULL THEN 1 ELSE 0 END) FROM ami_signal_lifecycle"
    ).fetchone()
    report["setup_version_all_null"] = (row[0] == 270)
    report["terminal_ts_all_null"] = (row[1] == 270)

    # [PHASE 7A-P1 semantic closure, round 2] retroactively reverses any
    # PRE-EXISTING TERMINAL_CLOSE-derived CLOSED status (inherited from the
    # copied source's already-applied, now-superseded M-0023 backfill) via an
    # appended CORRECTION transition -- append-only, the original
    # TERMINAL_CLOSE rows are NEVER edited/deleted, only countered.
    identity_before_terminal_correction = set(conn.execute(
        "SELECT signal_id, source_event_id, independent_cycle_id FROM ami_signal_lifecycle"
    ).fetchall())
    transitions_before_terminal_correction = set(conn.execute(
        "SELECT transition_id, signal_id, previous_status, new_status, transition_ts FROM ami_lifecycle_transitions"
    ).fetchall())

    report["terminal_close_correction"] = correct_unvalidated_terminal_close(
        conn, provenance="phase-7a-p1-semantic-closure-round2")

    identity_after_terminal_correction = set(conn.execute(
        "SELECT signal_id, source_event_id, independent_cycle_id FROM ami_signal_lifecycle"
    ).fetchall())
    transitions_after_terminal_correction = set(conn.execute(
        "SELECT transition_id, signal_id, previous_status, new_status, transition_ts FROM ami_lifecycle_transitions"
    ).fetchall())
    report["identity_unchanged_by_terminal_correction"] = (
        identity_before_terminal_correction == identity_after_terminal_correction)
    # the ledger is EXPECTED to grow (new CORRECTION rows) -- report the exact
    # delta and prove every pre-existing row (incl. the original TERMINAL_CLOSE
    # rows) is still present unmodified (append-only, nothing removed/edited)
    report["transitions_all_pre_existing_rows_preserved"] = transitions_before_terminal_correction.issubset(
        transitions_after_terminal_correction)
    report["transitions_added_by_terminal_correction_n"] = (
        len(transitions_after_terminal_correction) - len(transitions_before_terminal_correction))

    report["lifecycle_status_counts_post_batch"] = lifecycle_status_counts(conn)
    report["transition_type_counts_post_batch"] = transition_type_counts(conn)
    report["current_state_rebuild_consistency_post_batch"] = current_state_rebuild_consistency(conn)
    report["lifecycle_terminal_semantic_blocker_resolved"] = (
        report["lifecycle_status_counts_post_batch"].get("CLOSED", 0) == 0
        and report["current_state_rebuild_consistency_post_batch"]["consistent"]
    )

    # [PHASE 7A-P1 semantic closure, round 3] effective-ledger contract report
    signal_ids_for_duration_check = [r[0] for r in conn.execute("SELECT signal_id FROM ami_signal_lifecycle").fetchall()]
    report["effective_ledger_row_counts"] = effective_ledger_row_counts(conn)
    report["effective_rebuild_consistency"] = effective_rebuild_consistency(conn)
    report["duration_contamination_check"] = naive_raw_ledger_duration_contamination_check(
        conn, signal_ids_for_duration_check)
    # every effective row must be genesis-only for the 266 corrected signals --
    # i.e. the effective ledger has exactly 1 row per signal (no fake CLOSED survives)
    report["effective_rows_per_signal_all_genesis_only"] = (
        report["effective_ledger_row_counts"]["effective_ledger_rows"] == len(signal_ids_for_duration_check)
    )

    # migration/rerun idempotency (round 3): re-running the schema init +
    # correction a SECOND time within this same batch must add zero rows
    raw_rows_before_rerun = lifecycle_row_counts(conn)["ami_lifecycle_transitions"]
    init_lifecycle_schema(conn)
    rerun_correction = correct_unvalidated_terminal_close(
        conn, provenance="phase-7a-p1-semantic-closure-round2")
    report["terminal_close_correction_rerun"] = rerun_correction
    report["terminal_close_correction_rerun_adds_zero_rows"] = (rerun_correction["signals_corrected"] == 0)
    report["raw_ledger_rows_unchanged_after_rerun"] = (
        lifecycle_row_counts(conn)["ami_lifecycle_transitions"] == raw_rows_before_rerun
    )

    # 1+2. schema-v8(->v9) disposable copy + additive provenance migration
    init_field_provenance_schema(conn)
    report["schema_fingerprint_after_migration"] = schema_fingerprint(conn)
    report["new_objects_present"] = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND "
            "name IN ('ami_lifecycle_field_provenance','ami_lifecycle_direction_view')"
        ).fetchall()
    } == {"ami_lifecycle_field_provenance", "ami_lifecycle_direction_view"}

    signal_ids = [r[0] for r in conn.execute("SELECT signal_id FROM ami_signal_lifecycle").fetchall()]

    # 3. exact provenance backfill (run 1)
    r1 = backfill_field_provenance(conn, signal_ids)
    report["backfill_run1"] = r1
    counts_1 = provenance_row_counts(conn)
    hash_1 = provenance_content_hash(conn)

    # 4. second backfill run -- 0 new duplicates expected
    r2 = backfill_field_provenance(conn, signal_ids)
    report["backfill_run2"] = r2
    counts_2 = provenance_row_counts(conn)
    hash_2 = provenance_content_hash(conn)

    # 5. row-count / content-hash equality
    report["row_count_equal_across_reruns"] = (counts_1 == counts_2)
    report["content_hash_equal_across_reruns"] = (hash_1 == hash_2)
    report["provenance_counts"] = counts_2
    report["expected_provenance_rows"] = len(signal_ids) * len(FIELD_PROVENANCE_SPECS)

    # validation: every signal has complete provenance for every spec'd field
    try:
        validate_field_provenance_complete(conn, signal_ids)
        report["field_provenance_complete"] = True
        report["missing_provenance_count"] = 0
    except Exception as e:
        report["field_provenance_complete"] = False
        report["field_provenance_error"] = str(e)

    # direction-specific checks (the field the operator flagged)
    direction_rows = conn.execute(
        "SELECT field_classification, is_proxy, COUNT(*) FROM ami_lifecycle_field_provenance "
        "WHERE field_name='direction' GROUP BY field_classification, is_proxy"
    ).fetchall()
    report["direction_provenance_distribution"] = direction_rows
    report["direction_all_historical_proxy"] = (
        direction_rows == [("HISTORICAL_PROXY", 1, len(signal_ids))]
    )

    # 9. canonical lifecycle/current-state equality: **signal count** (270)
    # unchanged; **transition count** intentionally GROWS by the terminal-close
    # correction (536 -> 536+corrected_n) -- reported precisely, not asserted
    # blanket-unchanged (that would misreport an intentional, load-bearing fix
    # as a regression).
    lifecycle_after = lifecycle_row_counts(conn)
    report["signal_count_unchanged"] = (
        lifecycle_before["ami_signal_lifecycle"] == lifecycle_after["ami_signal_lifecycle"])
    report["transition_count_before_batch"] = lifecycle_before["ami_lifecycle_transitions"]
    report["transition_count_after_batch"] = lifecycle_after["ami_lifecycle_transitions"]
    report["lifecycle_counts"] = lifecycle_after

    # 6. rollback rehearsal
    rollback_field_provenance_schema(conn)
    remaining = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND "
            "name IN ('ami_lifecycle_field_provenance','ami_lifecycle_direction_view')"
        ).fetchall()
    }
    report["rollback_removed_new_objects"] = (len(remaining) == 0)
    lifecycle_after_rollback = lifecycle_row_counts(conn)
    # rollback only drops the field-provenance table/view -- it must preserve
    # whatever the lifecycle tables looked like AT THIS POINT (post terminal-
    # close correction), not the original pre-batch snapshot (which predates
    # an intentional, expected change)
    report["rollback_preserved_lifecycle_tables"] = (lifecycle_after_rollback == lifecycle_after)

    # 7. migration rerun idempotency (reapply + backfill again)
    init_field_provenance_schema(conn)
    r3 = backfill_field_provenance(conn, signal_ids)
    counts_3 = provenance_row_counts(conn)
    report["reapply_counts_match_pre_rollback"] = (counts_3 == counts_2)

    # 8. old-reader compatibility: existing lifecycle tables/feature_gateway paths unaffected
    from ami.research.feature_gateway import fetch_events
    events_after = fetch_events(conn, "provenance-rehearsal-old-reader", symbol="ETHUSDT",
                                source_quality="REAL_LIQUIDATION")
    report["old_reader_fetch_events_count"] = len(events_after)

    conn.close()
    return report
