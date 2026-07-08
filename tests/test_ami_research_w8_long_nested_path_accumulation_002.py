"""BATCH-P7B-1 (W8-LONG-NESTED-PATH-ACCUMULATION-002-CANDLE-REPAIR): tests for
ami/research/w8_long_nested_path_accumulation_002_candle_repair.py -- the
corrected-data (post candle-gap-repair) rerun, and for
ami.lifecycle.path_candle_repair_correction's effective-path-selection
machinery this rerun depends on.

Run: pytest tests/test_ami_research_w8_long_nested_path_accumulation_002.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import ami.lifecycle.path_candle_repair_correction as corr
import ami.research.w8_long_nested_path_accumulation as w8np_v1
import ami.research.w8_long_nested_path_accumulation_002_candle_repair as w8np_v2

_FORBIDDEN_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "hold_optimization",
    "win_rate", "threshold_sweep", "conviction_score",
)


def test_no_management_or_outcome_terms_in_module_source():
    src = inspect.getsource(w8np_v2).lower()
    hits = [t for t in _FORBIDDEN_TERMS if t in src]
    assert hits == [], f"forbidden terms found: {hits}"


def test_new_experiment_id_distinct_from_v001_and_never_overwrites_it():
    assert w8np_v2.EXPERIMENT_ID != w8np_v1.EXPERIMENT_ID
    assert w8np_v2.EXPERIMENT_ID == "E-W8-LONG-NESTED-PATH-ACCUMULATION-002-CANDLE-REPAIR"
    assert w8np_v2.CORRECTED_DATA_RERUN_OF == w8np_v1.EXPERIMENT_ID


def test_version_identifiers_frozen():
    assert w8np_v2.CANDLE_DATA_VERSION == "candle-binance-fapi-repair-v1"
    assert w8np_v2.PATH_DATA_VERSION == "path-v2-candle-repair-r1"


def test_reuses_v001_and_shared_machinery_verbatim():
    import ami.research.w8_short_expanded_baseline as w8se
    assert w8np_v2.compute_derived_fields is w8np_v1.compute_derived_fields
    assert w8np_v2.assert_nested_nonnegativity is w8np_v1.assert_nested_nonnegativity
    assert w8np_v2.compute_cell is w8np_v1.compute_cell
    assert w8np_v2.compute_secondary_descriptive is w8np_v1.compute_secondary_descriptive
    assert w8np_v2.ALL_DELTA_FIELDS is w8np_v1.ALL_DELTA_FIELDS
    assert w8np_v2.compute_global_cycle_split is w8se.compute_global_cycle_split
    assert w8np_v2.split_rows_by_cycle_keys is w8se.split_rows_by_cycle_keys
    assert w8np_v2.assert_zero_cycle_straddling is w8se.assert_zero_cycle_straddling


def test_only_fetch_common_cohort_is_overridden():
    # fetch_common_cohort must NOT be the same object as v001's -- everything else must be
    assert w8np_v2.fetch_common_cohort is not w8np_v1.fetch_common_cohort


# ---- Part 0: effective path selection safety (synthetic, full schema via init_schema) ----

def _synthetic_conn():
    import sqlite3
    import ami.warehouse.schema as schema_mod
    conn = sqlite3.connect(":memory:")
    schema_mod.init_schema(conn)
    return conn


def _insert_obs(conn, observation_id, signal_id, horizon_name, observation_status, path_definition_version, now=1):
    volatility_status = "OK" if observation_status == "OK" else "NOT_APPLICABLE"
    conn.execute(
        "INSERT INTO ami_lifecycle_path_observations (observation_id, signal_id, horizon_name, "
        "horizon_end_ts, known_at_ts, as_of_ts, observation_status, volatility_status, "
        "expected_candle_count, observed_candle_count, gap_count, path_definition_version, "
        "observation_mode, provenance, schema_version, created_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (observation_id, signal_id, horizon_name, now, now, now, observation_status, volatility_status,
         1, 1, 0, path_definition_version, "HISTORICAL_REPLAY", "test", 10, now),
    )


def test_fetch_effective_path_observations_prefers_corrected_row():
    conn = _synthetic_conn()
    _insert_obs(conn, "OBS-old", "SIG-1", "scalp_30m", "MISSING_INTERNAL_GAP", "path-v2")
    _insert_obs(conn, "OBS-new", "SIG-1", "scalp_30m", "OK", "path-v2-candle-repair-r1")
    _insert_obs(conn, "OBS-unaffected", "SIG-2", "scalp_30m", "OK", "path-v2")
    conn.commit()

    effective = corr.fetch_effective_path_observations(conn, "test-ctx")
    by_signal = {r["signal_id"]: r for r in effective}
    assert len(effective) == 2  # exactly one row per (signal, horizon), never 2 for SIG-1
    assert by_signal["SIG-1"]["path_definition_version"] == "path-v2-candle-repair-r1"
    assert by_signal["SIG-1"]["observation_status"] == "OK"
    assert by_signal["SIG-2"]["path_definition_version"] == "path-v2"
    conn.close()


def test_fetch_effective_path_observations_applies_filter_after_reduction():
    """The confirmed hazard: a pair where BOTH the original and corrected row
    already show observation_status='OK' must not be double-counted by an
    equals={'observation_status': 'OK'} filter."""
    conn = _synthetic_conn()
    for obs_id, version in (("OBS-a", "path-v2"), ("OBS-b", "path-v2-candle-repair-r1")):
        _insert_obs(conn, obs_id, "SIG-1", "scalp_30m", "OK", version)
    conn.commit()
    effective = corr.fetch_effective_path_observations(conn, "test-ctx", equals={"observation_status": "OK"})
    assert len(effective) == 1
    assert effective[0]["path_definition_version"] == "path-v2-candle-repair-r1"
    conn.close()


# ---- real-data integration (disposable copy via conftest isolation) ----

def test_effective_path_selection_audit_real_data():
    import ami.warehouse.schema as schema_mod
    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        audit = corr.effective_path_selection_audit(conn, "part0-audit-test")
        assert audit["duplicate_effective_pair_n"] == 0
        assert audit["effective_row_count"] == 1296
        assert audit["duplicate_physical_pair_n"] == 170
        assert audit["corrected_rows_supersede_n"] == 170
    finally:
        conn.close()


def test_real_data_smoke_freeze_and_record_and_idempotent():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        pre_signal_n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_path_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_provenance_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]
        pre_v001_hash = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8np_v1.EXPERIMENT_ID,),
        ).fetchall()

        r1 = w8np_v2.freeze_and_record(conn)
        assert len(r1["cell_order"]) == 6
        assert r1["population_report"]["signal_n"] > 0
        assert r1["population_report"]["cycle_straddling_violations"] == 0
        assert r1["nested_nonnegativity_check"]["violation_n"] == 0

        # v001 must remain byte-identical
        post_v001_hash = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8np_v1.EXPERIMENT_ID,),
        ).fetchall()
        assert post_v001_hash == pre_v001_hash

        # canonical lifecycle/path/provenance tables must be completely unaffected
        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0] == pre_provenance_n

        assert r1["family_verdict"] in (
            "LONG_NESTED_PATH_STABLE_CORRECTED_DATA", "LONG_NESTED_PATH_REGIME_DEPENDENT_CORRECTED_DATA",
            "MIXED_BY_INTERVAL_OR_METRIC_CORRECTED_DATA", "INSUFFICIENT_CORRECTED_COMMON_COHORT",
        )
        comp = r1["comparison_with_v001"]
        assert comp["comparison_conclusion"] in (
            "REPLICATED_ON_CORRECTED_EXPANDED_COHORT", "PARTIALLY_REPLICATED",
            "MATERIAL_RESULT_CHANGE", "INSUFFICIENT_CORRECTED_COHORT",
        )
        assert "not_independent_replications_note" in comp

        n_results_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8np_v2.EXPERIMENT_ID,)
        ).fetchone()[0]
        r2 = w8np_v2.freeze_and_record(conn)
        n_results_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8np_v2.EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_results_1 == n_results_2
        assert r2["family_verdict"] == r1["family_verdict"]
        assert r2["population_report"] == r1["population_report"]
    finally:
        conn.close()
