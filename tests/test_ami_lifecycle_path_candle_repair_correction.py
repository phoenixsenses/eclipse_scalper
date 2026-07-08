"""BATCH: AMI HISTORICAL CANDLE REPAIR -- tests for
ami/lifecycle/path_candle_repair_correction.py (versioned path correction:
insert-only, targeted, never overwrites the original "path-v2" rows).

Run: pytest tests/test_ami_lifecycle_path_candle_repair_correction.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import ami.lifecycle.path_candle_repair_correction as corr

_FORBIDDEN_TERMS = ("stop_loss", "take_profit", "re_entry", "hold_optimization", "win_rate", "p_value")


def test_no_management_or_outcome_terms_in_module_source():
    src = inspect.getsource(corr).lower()
    hits = [t for t in _FORBIDDEN_TERMS if t in src]
    assert hits == [], f"forbidden terms found: {hits}"


def test_new_path_data_version_distinct_from_old():
    assert corr.PATH_DATA_VERSION_CANDLE_REPAIR_R1 != corr.OLD_PATH_DEFINITION_VERSION
    assert corr.PATH_DATA_VERSION_CANDLE_REPAIR_R1 == "path-v2-candle-repair-r1"


# ---- real-data integration: identify + apply + rollback against the real, already-repaired DB ----
# (disposable copy only -- these tests never write to the real canonical.sqlite; conftest.py's
# session-scoped isolation redirects ami.warehouse.schema.DEFAULT_PATH to a disposable copy for
# the whole test session)

def test_identify_affected_pairs_matches_known_batch_result():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        result = corr.identify_affected_pairs(conn)
        # the isolated test-session copy already carries the REAL, applied candle repair (since
        # conftest.py copies the real, already-repaired canonical.sqlite) -- re-identifying
        # against it must reproduce exactly the same classification the real batch found
        assert len(result["class_a"]) == 149
        assert len(result["class_b"]) == 21
        assert len(result["unaffected"]) == 1126
        assert set(result["class_a"]).isdisjoint(set(result["class_b"]))
        assert set(result["class_a"]).isdisjoint(set(result["unaffected"]))
    finally:
        conn.close()


def test_apply_targeted_path_correction_is_idempotent_and_never_touches_old_rows():
    import ami.warehouse.schema as schema_mod
    import hashlib
    import json

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        def old_rows_hash():
            rows = conn.execute(
                "SELECT observation_id, signal_id, horizon_name, observation_status, mfe_bps, mae_bps "
                "FROM ami_lifecycle_path_observations WHERE path_definition_version=? ORDER BY observation_id",
                (corr.OLD_PATH_DEFINITION_VERSION,),
            ).fetchall()
            return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()

        before_old_hash = old_rows_hash()
        pre_corrected_n = conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE path_definition_version=?",
            (corr.PATH_DATA_VERSION_CANDLE_REPAIR_R1,),
        ).fetchone()[0]

        result = corr.identify_affected_pairs(conn)
        affected = result["class_a"] + result["class_b"]
        r1 = corr.apply_targeted_path_correction(conn, affected)
        assert r1["rows_written"] == len(affected)

        after_old_hash = old_rows_hash()
        assert after_old_hash == before_old_hash  # original "path-v2" rows never touched

        post_corrected_n = conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE path_definition_version=?",
            (corr.PATH_DATA_VERSION_CANDLE_REPAIR_R1,),
        ).fetchone()[0]
        assert post_corrected_n == pre_corrected_n  # idempotent -- already applied in this isolated copy

        # rerun again explicitly for idempotency proof
        r2 = corr.apply_targeted_path_correction(conn, affected)
        post_corrected_n_2 = conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE path_definition_version=?",
            (corr.PATH_DATA_VERSION_CANDLE_REPAIR_R1,),
        ).fetchone()[0]
        assert post_corrected_n_2 == post_corrected_n
        assert old_rows_hash() == before_old_hash
    finally:
        conn.close()


def test_rollback_and_reapply_path_correction_disposable_copy(tmp_path):
    """Explicit disposable-copy rollback/reapply -- never runs against the
    real canonical.sqlite (conftest already isolates DEFAULT_PATH, but this
    test additionally makes its OWN throwaway copy so a destructive rollback
    of the CORRECTED rows here can never affect other tests in this session)."""
    import shutil
    import sqlite3
    import ami.warehouse.schema as schema_mod

    disposable = tmp_path / "disposable.sqlite"
    shutil.copy2(schema_mod.DEFAULT_PATH, disposable)
    conn = sqlite3.connect(disposable)
    try:
        pre_total = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_corrected_n = conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE path_definition_version=?",
            (corr.PATH_DATA_VERSION_CANDLE_REPAIR_R1,),
        ).fetchone()[0]

        rb = corr.rollback_path_correction(conn)
        assert rb["rows_deleted"] == pre_corrected_n
        post_rollback_total = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        assert post_rollback_total == pre_total - pre_corrected_n
        remaining_corrected = conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE path_definition_version=?",
            (corr.PATH_DATA_VERSION_CANDLE_REPAIR_R1,),
        ).fetchone()[0]
        assert remaining_corrected == 0

        # reapply
        result = corr.identify_affected_pairs(conn)
        affected = result["class_a"] + result["class_b"]
        r = corr.apply_targeted_path_correction(conn, affected)
        assert r["rows_written"] == pre_corrected_n
        post_reapply_total = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        assert post_reapply_total == pre_total
    finally:
        conn.close()


def test_effective_observation_prefers_corrected_row_over_old():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        result = corr.identify_affected_pairs(conn)
        assert result["class_a"], "expected at least one affected pair in the isolated test copy"
        sid, horizon = result["class_a"][0]
        eff = corr.effective_observation(conn, sid, horizon)
        assert eff is not None
        assert eff["path_definition_version"] == corr.PATH_DATA_VERSION_CANDLE_REPAIR_R1
        assert eff["observation_status"] == "OK"

        # an unaffected pair must fall back to the original "path-v2" row
        if result["unaffected"]:
            sid2, horizon2 = result["unaffected"][0]
            eff2 = corr.effective_observation(conn, sid2, horizon2)
            assert eff2["path_definition_version"] == corr.OLD_PATH_DEFINITION_VERSION
    finally:
        conn.close()
