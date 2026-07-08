"""PHASE 7A.1: Disposable migration rehearsal harness.

DISPOSABLE_DB_ONLY / NO_LIVE_CANONICAL_DB_MIGRATION: `run_disposable_rehearsal`
opens the real canonical.sqlite ONLY with a `mode=ro` connection (to fingerprint
its existing schema and count ami_events, read-only) and via `shutil.copy2`
(a filesystem READ of the source, WRITE only to the new disposable path) --
it never opens the real path for writing. All schema/backfill/rollback
operations run exclusively against the disposable copy.

Implements the 14-step flow from the operator's Phase 7A.1 approval,
minus steps 13 (full pytest suite) and 14 (protected-diff gate), which are
external, non-portable checks run separately (not embedded in this module,
matching the Phase 7A-0 lesson about not gluing git/process concerns into
unit-testable code).
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from ami.lifecycle.canonical_backfill import backfill_lifecycle, correct_unvalidated_terminal_close
from ami.lifecycle.canonical_schema import (
    init_lifecycle_schema,
    migrate_setup_version_nullable,
    rebuild_current_state,
    rollback_lifecycle_schema,
)


def schema_fingerprint(conn) -> str:
    """Hash of every non-null `sql` in sqlite_master (tables+indexes),
    ordered deterministically -- a single scalar proof that "nothing else in
    the schema changed" across a step."""
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
    ).fetchall()
    text = "\n".join(f"{t}|{n}|{sql}" for t, n, sql in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_disposable_copy(source_path, disposable_path) -> None:
    Path(disposable_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, disposable_path)


def row_counts(conn) -> dict:
    n_sig = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
    n_trn = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
    return {"ami_signal_lifecycle": n_sig, "ami_lifecycle_transitions": n_trn}


def content_hash_lifecycle_tables(conn) -> str:
    sig_rows = conn.execute(
        "SELECT signal_id, setup_id, setup_version, source_event_id, independent_cycle_id, symbol, "
        "direction, timeframe, route_version, signal_birth_ts, first_known_ts, first_executable_ts, "
        "last_confirmation_ts, invalidation_ts, terminal_ts, lifecycle_status, lifecycle_reason_code, "
        "observation_mode, evidence_layer, is_proxy, executability_status, identity_version, "
        "schema_version, source_hash FROM ami_signal_lifecycle ORDER BY signal_id"
    ).fetchall()
    trn_rows = conn.execute(
        "SELECT transition_id, signal_id, previous_status, new_status, transition_ts, known_at_ts, "
        "reason_code, transition_version, observation_mode, correction_of, evidence_ref "
        "FROM ami_lifecycle_transitions ORDER BY transition_id"
    ).fetchall()
    text = json.dumps({"signals": sig_rows, "transitions": trn_rows}, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_disposable_rehearsal(source_canonical_path, disposable_path) -> dict:
    report: dict = {"source_canonical_path": str(source_canonical_path), "disposable_path": str(disposable_path)}

    # 1. exact current schema fingerprint (read-only against the REAL db)
    conn_ro = sqlite3.connect(f"file:{source_canonical_path}?mode=ro", uri=True)
    try:
        report["schema_fingerprint_before"] = schema_fingerprint(conn_ro)
        pre_event_count = conn_ro.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
    finally:
        conn_ro.close()

    # 2. disposable copy
    make_disposable_copy(source_canonical_path, disposable_path)
    conn = sqlite3.connect(disposable_path)

    # 3+4. additive migration + constraints/indexes (part of the schema itself)
    init_lifecycle_schema(conn)
    # [PHASE 7A-P1 semantic closure] the disposable copy inherits the REAL
    # canonical.sqlite's already-applied v8 table (setup_version TEXT NOT
    # NULL), so init_lifecycle_schema's CREATE TABLE IF NOT EXISTS is a
    # no-op here -- the nullable relaxation requires this explicit,
    # idempotent table-rebuild step (no-op on a fresh v9-shaped table).
    report["setup_version_schema_migrated"] = migrate_setup_version_nullable(conn)
    report["schema_fingerprint_after_migration"] = schema_fingerprint(conn)
    report["new_tables_present"] = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name IN ('ami_signal_lifecycle','ami_lifecycle_transitions')"
        ).fetchall()
    } == {"ami_signal_lifecycle", "ami_lifecycle_transitions"}

    # 5. deterministic historical-safe backfill (run 1) -- backfill_lifecycle's
    # OWN idempotency is checked in isolation here (runs 1+2), unmixed with the
    # terminal-close correction below (which is called exactly once, AFTER
    # both runs -- backfill_lifecycle's ON CONFLICT UPDATE unconditionally
    # resets lifecycle_reason_code to SIGNAL_BIRTH, which would otherwise
    # clobber the correction's CORRECTION marker on a 3rd rerun and falsely
    # look like a hash/idempotency regression).
    r1 = backfill_lifecycle(conn, conn)
    report["backfill_run1"] = r1
    counts_1 = row_counts(conn)
    hash_1 = content_hash_lifecycle_tables(conn)

    # 6. same backfill run again
    r2 = backfill_lifecycle(conn, conn)
    report["backfill_run2"] = r2
    counts_2 = row_counts(conn)
    hash_2 = content_hash_lifecycle_tables(conn)

    # 7. row-count comparison
    report["row_count_equal_across_reruns"] = (counts_1 == counts_2)

    # 8. content-hash comparison
    report["content_hash_equal_across_reruns"] = (hash_1 == hash_2)

    # [PHASE 7A-P1 semantic closure, round 2] the disposable copy inherits the
    # REAL canonical.sqlite's already-applied (M-0023) legacy TERMINAL_CLOSE
    # transitions (derived from unvalidated event_end_ts_ms) -- backfill_lifecycle's
    # ON CONFLICT UPDATE alone flips the denormalized lifecycle_status column to
    # OPEN but cannot retroactively touch the append-only ledger; this explicit,
    # idempotent correction step reverses it there too (see
    # ami/lifecycle/provenance_rehearsal.py for the fuller before/after report).
    # Run once, after backfill_lifecycle's own idempotency is already proven above.
    report["terminal_close_correction"] = correct_unvalidated_terminal_close(conn)
    report["counts"] = row_counts(conn)

    # 9. current-state rebuild (spot-check against the denormalized column)
    sample_ids = [r[0] for r in conn.execute("SELECT signal_id FROM ami_signal_lifecycle").fetchall()]
    mismatches = []
    for sid in sample_ids:
        rebuilt = rebuild_current_state(conn, sid)
        stored = conn.execute(
            "SELECT lifecycle_status FROM ami_signal_lifecycle WHERE signal_id=?", (sid,)
        ).fetchone()[0]
        if rebuilt is None or rebuilt["current_status"] != stored:
            mismatches.append(sid)
    report["current_state_rebuild_equality"] = (len(mismatches) == 0)
    report["current_state_rebuild_mismatches_n"] = len(mismatches)
    report["signals_checked_n"] = len(sample_ids)

    # old-reader compatibility (pre-rollback): existing ami_events read is unaffected
    post_migration_event_count = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
    report["old_reader_ami_events_count_unchanged"] = (pre_event_count == post_migration_event_count)

    # 10. rollback rehearsal
    rollback_lifecycle_schema(conn)
    tables_after_rollback = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    report["rollback_removed_new_tables"] = (
        "ami_signal_lifecycle" not in tables_after_rollback
        and "ami_lifecycle_transitions" not in tables_after_rollback
    )
    post_rollback_event_count = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
    report["rollback_preserved_existing_tables"] = (post_rollback_event_count == pre_event_count)

    # 11. migration reapplied
    init_lifecycle_schema(conn)
    report["setup_version_schema_migrated_after_reapply"] = migrate_setup_version_nullable(conn)
    r3 = backfill_lifecycle(conn, conn)
    report["backfill_run3_after_reapply"] = r3
    # rollback_lifecycle_schema DROPs both tables entirely -- reapply rebuilds
    # them FRESH from ami_events with the corrected backfill code, which never
    # writes a TERMINAL_CLOSE transition in the first place (no legacy mistake
    # exists to retroactively correct here, unlike the pre-rollback copy that
    # inherited the REAL DB's already-applied M-0023 legacy rows). Called for
    # symmetry/idempotency proof -- expected to find and correct 0 signals.
    report["terminal_close_correction_after_reapply"] = correct_unvalidated_terminal_close(conn)
    counts_3 = row_counts(conn)
    # signal count LEGITIMATELY does NOT match either, for a second, independent reason
    # (post BATCH-SHORT-NOISY-V1-CANON-BACKFILL): backfill_lifecycle()/derive_signals() is only
    # ONE of potentially several canonicalization sources that write into ami_signal_lifecycle --
    # it reconstructs exactly the ami_events.route_version-token-derivable population (270), never
    # a signal added via a separate identity path such as
    # ami.lifecycle.short_noisy_v1_rehearsal.backfill_short_noisy_v1 (54, BTC-confirmation-anchored,
    # not present in any event's route_version token list). A from-scratch DROP+rebuild via THIS
    # module alone was never meant to be (and is not) a full reconstruction of every
    # ami_signal_lifecycle row that has ever been canonicalized by ANY source -- asserting blanket
    # equality here would misreport that expected, structural asymmetry as a regression. (This is
    # the same category of asymmetry the transition-count comment above already documents for
    # TERMINAL_CLOSE/CORRECTION rows -- now also true for signal count itself.)
    pre_rollback_counts = report["counts"]
    report["reapply_signal_count"] = counts_3["ami_signal_lifecycle"]
    report["pre_rollback_signal_count"] = pre_rollback_counts["ami_signal_lifecycle"]
    report["reapply_signal_count_matches_pre_rollback"] = (
        counts_3["ami_signal_lifecycle"] == pre_rollback_counts["ami_signal_lifecycle"])
    report["reapply_transition_count"] = counts_3["ami_lifecycle_transitions"]
    report["pre_rollback_transition_count"] = pre_rollback_counts["ami_lifecycle_transitions"]

    # 12. old-reader compatibility (post-reapply, final state)
    final_event_count = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
    report["old_reader_ami_events_count_unchanged_final"] = (final_event_count == pre_event_count)

    conn.close()
    return report
