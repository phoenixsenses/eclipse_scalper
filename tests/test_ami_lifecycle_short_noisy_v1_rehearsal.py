"""BATCH: SHORT_NOISY_BTC200K_CONFIRMED_V1 disposable canonicalization
rehearsal -- tests for ami/lifecycle/short_noisy_v1_rehearsal.py and
ami/lifecycle/short_noisy_v1_migration_rehearsal.py.

DISPOSABLE_DB_ONLY / NO_REAL_CANONICAL_WRITE: the real data/ami/canonical.sqlite
is opened ONLY mode=ro (fingerprint + counts) or copied via shutil.copy2 (a
read of the source). data/microstructure.db is opened ONLY mode=ro -- never
copied, never written, in this file or the module under test.

Run: pytest tests/test_ami_lifecycle_short_noisy_v1_rehearsal.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import hashlib
import inspect
import os
from pathlib import Path

from ami.lifecycle.short_noisy_v1_migration_rehearsal import run_disposable_rehearsal
from ami.lifecycle import short_noisy_v1_rehearsal as snv1
from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH  # captured before conftest redirection

MICROSTRUCTURE_DB_PATH = Path("D:/eclipse_scalper/data/microstructure.db")

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "reentry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
)
# outcome/selection terms that must never appear as executable identifiers (variable/function
# names, conditionals) -- excludes the module's own docstring prose explicitly DISCLAIMING these
# (e.g. "no conviction score, no hour filter, no funding filter"), which legitimately mentions the
# words to document what is NOT done, not to implement it
_FORBIDDEN_SELECTION_IDENTIFIERS = (
    "win_rate", "pnl", "mfe_bps", "mae_bps", "threshold_sweep", "conviction_score",
)


def _file_hash_chunked(path, max_bytes: int | None = None) -> str:
    """Chunked hash -- never loads the whole file into memory. `max_bytes` caps
    total bytes read (used for the multi-hundred-GB microstructure.db, where a
    full-file hash is impractical and unnecessary -- mtime is the change signal
    there; this is only a bounded sanity prefix-check)."""
    h = hashlib.sha256()
    read_total = 0
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
            read_total += len(chunk)
            if max_bytes is not None and read_total >= max_bytes:
                break
    return h.hexdigest()


def _file_hash(path) -> str:
    return _file_hash_chunked(path)


def test_no_graveyarded_management_or_selection_terms_in_module_source():
    src = (inspect.getsource(snv1)).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS if t in src]
    assert hits == [], f"forbidden management-rule terms found: {hits}"
    sel_hits = [t for t in _FORBIDDEN_SELECTION_IDENTIFIERS if t in src]
    assert sel_hits == [], f"forbidden outcome/selection identifiers found: {sel_hits}"


def test_no_action_permission_escalation_in_module_source():
    src = inspect.getsource(snv1)
    assert "authorize(" not in src
    assert ".promote(" not in src
    assert "OPEN_LONG" not in src
    assert "OPEN_SHORT" not in src
    assert "import ami.governance" not in src
    assert "from ami.governance" not in src


def test_no_order_router_or_execution_import():
    src = inspect.getsource(snv1)
    for forbidden in ("execution.", "risk.", "brain.", "order_router", "entry_loop", "position_manager"):
        assert forbidden not in src, f"forbidden import/reference: {forbidden}"


def test_setup_id_is_not_the_old_short_noisy_route():
    assert snv1.SETUP_ID != "SHORT_NOISY_BTC1M_D5_H180"
    assert snv1.classify_direction_from_setup_id(snv1.SETUP_ID) == "SHORT"


def test_disposable_db_and_microstructure_db_untouched(tmp_path):
    canon_hash_before = _file_hash(REAL_CANONICAL_PATH)
    canon_mtime_before = os.path.getmtime(REAL_CANONICAL_PATH)
    # [operator instruction, BATCH W8-LONG VOLATILITY-STATE DEFINITION AND COVERAGE AUDIT]:
    # microstructure.db's mtime is NOT a reliable untouched-assertion while its live collector is
    # running -- the collector appends new rows continuously (observed: mtime shifts by mere
    # seconds between consecutive checks in this same test session), which is expected, benign
    # collector activity completely unrelated to whether THIS module wrote to the file. The bounded
    # 64MB PREFIX content hash is the collector-aware invariant instead: new inserts extend the
    # file/allocate pages near its end, so the prefix is stable regardless of concurrent, unrelated
    # collector writes, while still catching any write this module itself might make.
    micro_prefix_hash_before = _file_hash_chunked(MICROSTRUCTURE_DB_PATH, max_bytes=64 * 1024 * 1024)

    run_disposable_rehearsal(REAL_CANONICAL_PATH, tmp_path / "disposable.sqlite", MICROSTRUCTURE_DB_PATH)

    assert _file_hash(REAL_CANONICAL_PATH) == canon_hash_before
    assert os.path.getmtime(REAL_CANONICAL_PATH) == canon_mtime_before
    assert _file_hash_chunked(MICROSTRUCTURE_DB_PATH, max_bytes=64 * 1024 * 1024) == micro_prefix_hash_before


def test_full_rehearsal_flow_real_data(tmp_path):
    report = run_disposable_rehearsal(
        REAL_CANONICAL_PATH, tmp_path / "disposable.sqlite", MICROSTRUCTURE_DB_PATH
    )

    # [POST BATCH-SHORT-NOISY-V1-CANON-BACKFILL] the real canonical.sqlite now permanently
    # carries this setup's 54 signals (applied via the separately-approved "APPROVE
    # SHORT_NOISY_BTC200K_CONFIRMED_V1 CONTROLLED CANONICAL DATA BACKFILL"). Either 0 (a
    # pristine/rolled-back copy, e.g. right after rollback_short_noisy_v1()) or exactly
    # candidate_n (the real, already-migrated state) is valid -- anything else means an
    # unexpected partial/duplicate write.
    assert report["pre_existing_setup_id_signal_n"] in (0, report["candidate_n"])

    # known-at / no-lookahead mandatory checks
    assert report["identity_deterministic_across_reruns"] is True
    assert report["all_conf_ts_after_noisy_ts_plus_5m"] is True
    assert report["all_noisy_ts_after_anchor_plus_1m"] is True
    assert report["all_signal_births_equal_conf_ts"] is True
    assert report["duplicate_conf_ts_did_not_merge_distinct_events"] is True

    # idempotency across every layer
    assert report["idempotent_signal_upsert_count"] is True
    assert report["idempotent_transitions_zero_new_on_rerun"] is True
    assert report["idempotent_content_hash"] is True
    assert report["idempotent_field_provenance_rows"] is True
    assert report["idempotent_path_field_provenance_rows"] is True

    # old-reader compatibility -- the pre-existing 270/1080/(4320+6210) population untouched
    assert report["old_reader_pre_existing_signal_n_unchanged"] is True
    assert report["old_reader_pre_existing_path_n_unchanged"] is True
    assert report["old_reader_pre_existing_provenance_n_unchanged"] is True
    assert report["old_reader_event_n_unchanged"] is True

    # rollback + reapply
    assert report["rollback_signal_count_matches"] is True
    assert report["rollback_preserved_pre_existing_signal_n"] is True
    assert report["rollback_preserved_pre_existing_path_n"] is True
    assert report["rollback_preserved_pre_existing_provenance_n"] is True
    assert report["rollback_preserved_event_n"] is True
    assert report["reapply_signal_ids_match"] is True
    assert report["reapply_content_hash_matches_pre_rollback"] is True

    # no DDL/schema change anywhere in this batch
    assert report["schema_fingerprint_unchanged"] is True


def test_counting_discipline_and_expected_population_size(tmp_path):
    report = run_disposable_rehearsal(
        REAL_CANONICAL_PATH, tmp_path / "disposable.sqlite", MICROSTRUCTURE_DB_PATH
    )
    # expected per the read-only reconciliation frozen 2026-07-04 (may legitimately grow with
    # forward accumulation of new liquidation/event data -- never asserted as an upper bound
    # beyond the population that actually exists in the real DB at rehearsal time)
    assert report["candidate_n"] == report["lifecycle_run1"]["candidate_n"]
    assert report["candidate_source_event_n"] == report["candidate_n"]  # 1 signal per source event
    assert report["new_path_observation_row_n"] == report["new_path_observation_row_n_expected_max"]
    assert report["new_path_observation_row_n_expected_max"] == report["candidate_n"] * 4

    overlap = report["overlap_matrix"]
    assert overlap["new_signal_n"] == report["candidate_n"]
    assert overlap["distinct_qualifying_cycle_n"] <= overlap["source_event_n"]
    assert (overlap["cycles_already_short_represented"] + overlap["cycles_newly_short_represented"]
            == overlap["distinct_qualifying_cycle_n"])


def test_field_provenance_overrides_present_and_classification_unchanged(tmp_path):
    report = run_disposable_rehearsal(
        REAL_CANONICAL_PATH, tmp_path / "disposable.sqlite", MICROSTRUCTURE_DB_PATH
    )
    fp1 = report["field_provenance_run1"]
    assert fp1["fields_per_signal"] == 16
    assert fp1["provenance_rows_written"] == report["candidate_n"] * 16

    path_fp1 = report["path_field_provenance_run1"]
    assert path_fp1["provenance_rows_written_this_call"] == report["candidate_n"] * 23

    for field_name, override in snv1.FIELD_SPEC_OVERRIDES.items():
        base = snv1.FIELD_PROVENANCE_SPECS[field_name]
        assert override["field_classification"] == base["field_classification"], (
            f"{field_name} override changed classification -- forbidden proxy<->safe transition"
        )


def test_reconciliation_decision_verdict_from_report(tmp_path):
    """Applies the operator's exact decision rule from the report -- this test
    does not itself decide the verdict, it only proves the report contains
    enough to compute it and that no identity/known-at/idempotency blocker
    was found (a necessary, not sufficient, condition for
    CANONICAL_MIGRATION_READY_LIMITED_DESCRIPTIVE_POPULATION)."""
    report = run_disposable_rehearsal(
        REAL_CANONICAL_PATH, tmp_path / "disposable.sqlite", MICROSTRUCTURE_DB_PATH
    )
    blockers = []
    if not report["identity_deterministic_across_reruns"]:
        blockers.append("IDENTITY_NOT_DETERMINISTIC")
    if not (report["all_conf_ts_after_noisy_ts_plus_5m"] and report["all_noisy_ts_after_anchor_plus_1m"]):
        blockers.append("LOOKAHEAD_VIOLATION")
    if not report["all_signal_births_equal_conf_ts"]:
        blockers.append("LOOKAHEAD_VIOLATION")
    if not report["duplicate_conf_ts_did_not_merge_distinct_events"]:
        blockers.append("IDENTITY_COLLISION")
    if not report["idempotent_content_hash"]:
        blockers.append("NOT_IDEMPOTENT")
    if not report["reapply_content_hash_matches_pre_rollback"]:
        blockers.append("REAPPLY_MISMATCH")
    if not report["schema_fingerprint_unchanged"]:
        blockers.append("UNEXPECTED_SCHEMA_CHANGE")
    assert blockers == []

    overlap = report["overlap_matrix"]
    newly_short_n = overlap["cycles_newly_short_represented"]
    if newly_short_n < 20:
        verdict = "RETIRE_AS_SHORT_N_SOLUTION"
    else:
        verdict = "CANONICAL_MIGRATION_READY_LIMITED_DESCRIPTIVE_POPULATION"
    report["_computed_verdict"] = verdict
    assert verdict == "CANONICAL_MIGRATION_READY_LIMITED_DESCRIPTIVE_POPULATION"
