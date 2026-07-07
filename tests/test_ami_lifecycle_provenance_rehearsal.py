"""PHASE 7A-P1: tests for ami/lifecycle/provenance_rehearsal.py -- the full
disposable-copy field-provenance closure flow, run against a throwaway copy
of the real (schema-v8, already-migrated) canonical.sqlite.

DISPOSABLE_DB_ONLY / NO_CANONICAL_DB_WRITE: the real data/ami/canonical.sqlite
is opened ONLY mode=ro or copied via shutil.copy2. `test_disposable_db_only_
source_untouched` proves the source file's hash/mtime are unchanged after
the full rehearsal.

Run: pytest tests/test_ami_lifecycle_provenance_rehearsal.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import hashlib
import os
import sqlite3

from ami.lifecycle.provenance_rehearsal import run_disposable_provenance_rehearsal
from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH


def _file_hash(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_disposable_db_only_source_untouched(tmp_path):
    hash_before = _file_hash(REAL_CANONICAL_PATH)
    mtime_before = os.path.getmtime(REAL_CANONICAL_PATH)

    run_disposable_provenance_rehearsal(REAL_CANONICAL_PATH, tmp_path / "disposable.sqlite")

    assert _file_hash(REAL_CANONICAL_PATH) == hash_before
    assert os.path.getmtime(REAL_CANONICAL_PATH) == mtime_before


def test_full_provenance_rehearsal_real_data(tmp_path):
    report = run_disposable_provenance_rehearsal(REAL_CANONICAL_PATH, tmp_path / "disposable.sqlite")

    # [PHASE 7A-P CANONICAL PROVENANCE MIGRATION] this assertion was written
    # when the real canonical.sqlite was still at schema v8 (pre-migration),
    # so schema_version_before was always 8 and terminal_close_correction
    # always found NEW signals to fix. The real, approved v9 migration (with
    # the 266 append-only CORRECTION transitions) has since been APPLIED to
    # the real DB -- schema_version_before is now durably >=9, and
    # terminal_close_correction finds 0 NEW corrections needed (all 266
    # already applied). This branches on whichever state the real DB is
    # currently in, matching the precedent set by
    # test_ami_lifecycle_migration_rehearsal.py's
    # test_schema_fingerprint_changes_only_by_addition: this is proof the
    # migration is durably present, not a regression.
    # [PHASE 7B-CANON] v9->v10 (APPROVE PHASE 7B CANONICAL PATH METRICS
    # MIGRATION) only ADDED ami_lifecycle_path_observations -- it changed
    # nothing about the lifecycle-transition/provenance semantics this test
    # exercises, so v10 belongs in the same "already_migrated" branch as v9.
    # [BATCH-AMI-BIRTH-TRUNCATED-GEOMETRY-CANONICAL-MIGRATION] v10->v11 only
    # ADDED ami_birth_truncated_cascade_geometry/field_provenance/field_
    # quality_v2 -- same reasoning, v11 joins the same branch.
    # [BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1] v11->v12 only ADDED
    # ami_agg_trades_repaired/ami_cvd_repair_batch_ledger/ami_cvd_windowed_flow(
    # _proxy)/ami_cvd_bucket_exclusions/ami_cvd_window_quality_v1 -- same
    # reasoning, v12 joins the same branch.
    # [BATCH-CASCADE-ABSORPTION-IMPACT-CANONICAL-MIGRATION-V1, M-0035] v12->v13
    # only ADDED ami_absorption_impact_windowed_flow/window_quality_v1/
    # exclusions -- same reasoning, v13 joins the same branch.
    assert report["schema_version_before"] in (8, 9, 10, 11, 12, 13)
    assert report["new_objects_present"] is True
    already_migrated = report["schema_version_before"] in (9, 10, 11, 12, 13)

    # [POST BATCH-SHORT-NOISY-V1-CANON-BACKFILL] signal population is now 324 (270 original +
    # 54 SHORT_NOISY_BTC200K_CONFIRMED_V1) and raw transitions 856 (802 + 54 new SIGNAL_BIRTH) --
    # this rehearsal copies the real DB as-is (make_disposable_copy) and re-runs
    # backfill_lifecycle() (idempotent, touches only the original 270-signal route_version-token
    # population), so the 54 new signals pass through this rehearsal completely untouched and
    # OPEN throughout, exactly like the pre-existing 270.
    if already_migrated:
        assert report["lifecycle_terminal_semantic_blocker_found"] is False
        assert report["lifecycle_status_counts_pre_batch"] == {"OPEN": 324}
        assert report["terminal_close_correction"] == {
            "terminal_close_transitions_found": 266, "signals_corrected": 0, "already_open_or_corrected": 266,
        }
        assert report["transitions_added_by_terminal_correction_n"] == 0
        expected_transition_count_before_batch = 856
    else:
        # [PHASE 7A-P1 semantic closure, round 2] LIFECYCLE_TERMINAL_SEMANTIC_BLOCKER:
        # the disposable copy inherits the REAL DB's already-applied (M-0023) state,
        # where 266/270 signals were wrongly marked CLOSED off unvalidated
        # event_end_ts_ms -- found here, then resolved by an appended (never an
        # edit/delete) CORRECTION transition per signal.
        assert report["lifecycle_terminal_semantic_blocker_found"] is True
        assert report["lifecycle_status_counts_pre_batch"] == {"CLOSED": 266, "OPEN": 4}
        assert report["terminal_close_correction"] == {
            "terminal_close_transitions_found": 266, "signals_corrected": 266, "already_open_or_corrected": 0,
        }
        assert report["transitions_added_by_terminal_correction_n"] == 266
        expected_transition_count_before_batch = 536
    assert report["current_state_rebuild_consistency_pre_batch"]["consistent"] is True  # consistent (wrong pre-fix, right post-fix)

    assert report["identity_unchanged_by_terminal_correction"] is True
    assert report["transitions_all_pre_existing_rows_preserved"] is True  # append-only: nothing removed/edited

    assert report["lifecycle_terminal_semantic_blocker_resolved"] is True
    assert report["lifecycle_status_counts_post_batch"] == {"OPEN": 324}
    assert report["current_state_rebuild_consistency_post_batch"] == {
        "signals_checked": 324, "mismatches_n": 0, "consistent": True,
    }

    # signal count unchanged (324, post BATCH-SHORT-NOISY-V1-CANON-BACKFILL); transition count
    # intentionally GROWS to 856 whenever new corrections are applied (pre-migration source), or
    # stays at 856 when the source is already fully corrected (post-migration)
    assert report["lifecycle_counts"] == {"ami_signal_lifecycle": 324, "ami_lifecycle_transitions": 856}
    assert report["signal_count_unchanged"] is True
    assert report["transition_count_before_batch"] == expected_transition_count_before_batch
    assert report["transition_count_after_batch"] == 856

    # [PHASE 7A-P1 semantic closure, round 3] two-layer ledger contract:
    # raw (immutable, 856) vs effective (superseded + pure-reversal correction
    # pairs excluded, 324 -- exactly one genesis row per signal, since every
    # existing correction in this dataset is a pure reversal)
    assert report["effective_ledger_row_counts"] == {"raw_ledger_rows": 856, "effective_ledger_rows": 324}
    assert report["effective_rows_per_signal_all_genesis_only"] is True
    assert report["effective_rebuild_consistency"] == {
        "signals_checked": 324, "mismatches_n": 0, "consistent": True,
    }
    # duration-contamination proof: a naive raw-ledger query would fabricate a
    # CLOSED interval for all 266 corrected signals; the effective view has
    # none (0 genuine CLOSED intervals) -- this is the exact failure mode a
    # downstream researcher reading the raw ledger directly must be protected from
    assert report["duration_contamination_check"] == {
        "signals_with_fake_raw_closed_interval": 266,
        "signals_with_genuine_effective_closed_interval": 0,
    }
    # migration/rerun idempotency: re-running schema init + correction a
    # second time adds zero new rows (raw ledger stays at 802)
    assert report["terminal_close_correction_rerun"] == {
        "terminal_close_transitions_found": 266, "signals_corrected": 0, "already_open_or_corrected": 266,
    }
    assert report["terminal_close_correction_rerun_adds_zero_rows"] is True
    assert report["raw_ledger_rows_unchanged_after_rerun"] is True

    # exact 324-signal direction provenance backfill (16 fields/signal = 5184 rows)
    # [POST BATCH-SHORT-NOISY-V1-CANON-BACKFILL] this rehearsal's backfill_field_provenance()
    # call is scoped to ALL signal_ids (line "signal_ids = [...] FROM ami_signal_lifecycle"), so
    # on the disposable copy used here it also re-writes GENERIC canonical_field_provenance.
    # FIELD_PROVENANCE_SPECS text for the 54 new SHORT_NOISY_BTC200K_CONFIRMED_V1 signals'
    # signal_birth_ts/setup_id/route_version fields (field_classification stays
    # DETERMINISTIC_HISTORICAL_SAFE, unchanged -- but the derivation_method/source_reference text
    # reverts to the generic "ami_events.anchor_ts_ms"/"route_version_comma_split" description,
    # which is factually inaccurate for this cohort's actual derivation). This is a KNOWN,
    # disposable-copy-only side effect of this Phase 7A-P1 rehearsal script never having been
    # scoped to a specific setup_id population -- it does NOT touch the real canonical.sqlite
    # (this test only ever runs against tmp_path's disposable copy), so the real DB's correctly
    # overridden provenance (written by ami.lifecycle.short_noisy_v1_rehearsal.
    # backfill_short_noisy_v1_field_provenance) is unaffected. Flagged, not fixed, in this batch.
    assert report["backfill_run1"]["signals_covered"] == 324
    assert report["provenance_counts"]["ami_lifecycle_field_provenance"] == 324 * 16
    assert report["expected_provenance_rows"] == 324 * 16

    # second run: 0 duplicates
    assert report["row_count_equal_across_reruns"] is True
    assert report["content_hash_equal_across_reruns"] is True

    # direction: all 324 rows HISTORICAL_PROXY / is_proxy=1 (SHORT_NOISY_BTC200K_CONFIRMED_V1's
    # direction is also derived via classify_direction_from_setup_id, same as every other setup)
    assert report["direction_all_historical_proxy"] is True
    assert report["direction_provenance_distribution"] == [("HISTORICAL_PROXY", 1, 324)]

    # [PHASE 7A-P1 semantic closure, round 3] canonical query contract: every
    # signal is now lifecycle_status=OPEN, but this must NEVER be read as
    # "currently live/active" -- there is no terminal-evidence mechanism
    # (terminal_ts stays NOT_IMPLEMENTED for all 324). A downstream
    # terminal/hold-duration researcher must observe BOTH facts together
    # before treating any signal as resolved.
    conn_check = sqlite3.connect(tmp_path / "disposable.sqlite")
    try:
        terminal_ts_classifications = {
            r[0] for r in conn_check.execute(
                "SELECT DISTINCT field_classification FROM ami_lifecycle_field_provenance "
                "WHERE field_name='terminal_ts'"
            ).fetchall()
        }
    finally:
        conn_check.close()
    assert terminal_ts_classifications == {"NOT_IMPLEMENTED"}

    # missing-provenance validation passes (complete coverage)
    assert report["field_provenance_complete"] is True
    assert report["missing_provenance_count"] == 0

    # rollback + reapply
    assert report["rollback_removed_new_objects"] is True
    assert report["rollback_preserved_lifecycle_tables"] is True
    assert report["reapply_counts_match_pre_rollback"] is True

    # old-reader compatibility
    assert report["old_reader_fetch_events_count"] == 252


def test_direction_provenance_matches_actual_derivation_code(tmp_path):
    # cross-check: the provenance record's classification must match what
    # ami.lifecycle.canonical_schema.classify_direction_from_setup_id ACTUALLY
    # implements today -- not a hardcoded assumption independent of the code.
    import sqlite3

    from ami.lifecycle.canonical_schema import classify_direction_from_setup_id
    from ami.lifecycle.provenance_rehearsal import make_disposable_copy
    from ami.lifecycle.canonical_field_provenance import init_field_provenance_schema, backfill_field_provenance

    disposable = tmp_path / "disposable.sqlite"
    make_disposable_copy(REAL_CANONICAL_PATH, disposable)
    conn = sqlite3.connect(disposable)
    init_field_provenance_schema(conn)
    signal_ids = [r[0] for r in conn.execute("SELECT signal_id FROM ami_signal_lifecycle").fetchall()]
    backfill_field_provenance(conn, signal_ids)

    rows = conn.execute("SELECT signal_id, setup_id, direction FROM ami_signal_lifecycle").fetchall()
    mismatches = [r for r in rows if classify_direction_from_setup_id(r[1]) != r[2]]
    conn.close()
    assert mismatches == []  # every stored direction matches a fresh call to the real function
