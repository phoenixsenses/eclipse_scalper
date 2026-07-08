"""BATCH-P7B-1 (W8-SHORT-EXPANDED-BASELINE-003-CANDLE-REPAIR): tests for
ami/research/w8_short_expanded_baseline_003_candle_repair.py -- the
corrected-data (post candle-gap-repair) rerun of
E-W8-HOLD-BASELINE-002-SHORT-EXPANDED / E-W8-VOL-NORMALIZED-BASELINE-002-
SHORT-EXPANDED.

Run: pytest tests/test_ami_research_w8_short_expanded_baseline_003.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import pytest

import ami.research.w8_short_expanded_baseline as w8se_v002
import ami.research.w8_short_expanded_baseline_003_candle_repair as w8se3
from ami.warehouse.experiment_ledger import ImmutableExperimentConflict

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "reentry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
)
# "pnl" is deliberately excluded -- the module's own NO_ECONOMIC_CLAIM disclaimer legitimately
# mentions "PnL" in prose (same convention as w8_short_expanded_baseline.py's own docstring)
_FORBIDDEN_OUTCOME_TERMS = ("win_rate", "alpha_claim", "economic_edge")


def test_no_management_or_outcome_terms_in_module_source():
    src = inspect.getsource(w8se3).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS + _FORBIDDEN_OUTCOME_TERMS if t in src]
    assert hits == [], f"forbidden terms found: {hits}"


def test_new_experiment_ids_distinct_from_002_and_never_overwrite_it():
    assert w8se3.RAW_BPS_EXPERIMENT_ID == "E-W8-HOLD-BASELINE-003-SHORT-EXPANDED-CANDLE-REPAIR"
    assert w8se3.VOL_NORMALIZED_EXPERIMENT_ID == "E-W8-VOL-NORMALIZED-BASELINE-003-SHORT-EXPANDED-CANDLE-REPAIR"
    assert w8se3.OLD_RAW_BPS_EXPERIMENT_ID == "E-W8-HOLD-BASELINE-002-SHORT-EXPANDED"
    assert w8se3.OLD_VOL_NORMALIZED_EXPERIMENT_ID == "E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED"
    assert w8se3.RAW_BPS_EXPERIMENT_ID != w8se3.OLD_RAW_BPS_EXPERIMENT_ID
    assert w8se3.VOL_NORMALIZED_EXPERIMENT_ID != w8se3.OLD_VOL_NORMALIZED_EXPERIMENT_ID
    assert w8se3._CORRECTED_DATA_RERUN_OF[w8se3.RAW_BPS_EXPERIMENT_ID] == w8se3.OLD_RAW_BPS_EXPERIMENT_ID
    assert w8se3._CORRECTED_DATA_RERUN_OF[w8se3.VOL_NORMALIZED_EXPERIMENT_ID] == w8se3.OLD_VOL_NORMALIZED_EXPERIMENT_ID


def test_version_identifiers_frozen():
    assert w8se3.CANDLE_DATA_VERSION == "candle-binance-fapi-repair-v1"
    assert w8se3.PATH_DATA_VERSION == "path-v2-candle-repair-r1"


def test_reuses_002_and_shared_machinery_verbatim():
    """Every function EXCEPT fetch_raw_bps_population/fetch_vol_normalized_population
    is imported from w8_short_expanded_baseline (or w8_hold_baseline via it),
    never reimplemented -- proven by `is` identity."""
    assert w8se3.compute_cell is w8se_v002.compute_cell
    assert w8se3.compute_global_cycle_split is w8se_v002.compute_global_cycle_split
    assert w8se3.split_rows_by_cycle_keys is w8se_v002.split_rows_by_cycle_keys
    assert w8se3.assert_zero_cycle_straddling is w8se_v002.assert_zero_cycle_straddling
    assert w8se3._cycle_key is w8se_v002._cycle_key
    assert w8se3._cell_rows is w8se_v002._cell_rows
    assert w8se3.compute_composition_diagnostic is w8se_v002.compute_composition_diagnostic
    assert w8se3.RAW_BPS_METRICS == w8se_v002.RAW_BPS_METRICS
    assert w8se3.VOL_NORMALIZED_METRICS == w8se_v002.VOL_NORMALIZED_METRICS
    assert w8se3.DIRECTION == w8se_v002.DIRECTION == "SHORT"
    assert w8se3.NEW_SETUP_ID == w8se_v002.NEW_SETUP_ID


def test_only_population_fetch_functions_are_overridden():
    """fetch_raw_bps_population/fetch_vol_normalized_population must NOT be
    `is`-identical to -002's versions (they are the two functions this module
    deliberately rewrites to use the effective-path selector)."""
    assert w8se3.fetch_raw_bps_population is not w8se_v002.fetch_raw_bps_population
    assert w8se3.fetch_vol_normalized_population is not w8se_v002.fetch_vol_normalized_population


# ---- synthetic tests for the mandatory effective-path integrity gate ----

def test_verify_effective_path_selection_integrity_passes_on_matching_audit(monkeypatch):
    monkeypatch.setattr(w8se3, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 1466, "duplicate_physical_pair_n": 170,
        "effective_row_count": 1296, "duplicate_effective_pair_n": 0,
    })
    result = w8se3.verify_effective_path_selection_integrity(conn=None)
    assert result["passed"] is True
    assert result["mismatches"] == {}


def test_verify_effective_path_selection_integrity_fails_closed_on_mismatch(monkeypatch):
    monkeypatch.setattr(w8se3, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 1466, "duplicate_physical_pair_n": 170,
        "effective_row_count": 1290,  # WRONG -- should be 1296
        "duplicate_effective_pair_n": 3,  # WRONG -- should be 0
    })
    result = w8se3.verify_effective_path_selection_integrity(conn=None)
    assert result["passed"] is False
    assert "effective_row_count" in result["mismatches"]
    assert "duplicate_effective_pair_n" in result["mismatches"]
    assert result["mismatches"]["effective_row_count"] == {"actual": 1290, "expected": 1296}


def test_compute_family_blocks_and_computes_zero_cells_on_integrity_failure(monkeypatch):
    monkeypatch.setattr(w8se3, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 999, "duplicate_physical_pair_n": 0,
        "effective_row_count": 0, "duplicate_effective_pair_n": 5,
    })
    family = w8se3.compute_family(conn=None)
    assert family["blocked"] is True
    assert family["family_verdict"] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
    assert "cells" not in family  # no population fetched, no cell computed


def test_freeze_and_record_blocked_branch_writes_verdict_and_leaves_002_untouched(tmp_path, monkeypatch):
    """Fresh, empty DB (no -002 rows at all) -- old_experiments_untouched is
    trivially True (both remain absent), which is itself a correct proof that
    the blocked branch never fabricates a -002 row."""
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(tmp_path / "blocked_test.sqlite")
    schema_mod.init_schema(conn)

    monkeypatch.setattr(w8se3, "effective_path_selection_audit", lambda c, ctx: {
        "physical_row_count_total": 1, "duplicate_physical_pair_n": 1,
        "effective_row_count": 1, "duplicate_effective_pair_n": 1,
    })

    result = w8se3.freeze_and_record(conn)
    assert result["family_verdict"] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
    assert result["old_experiments_untouched"] is True

    for experiment_id in (w8se3.RAW_BPS_EXPERIMENT_ID, w8se3.VOL_NORMALIZED_EXPERIMENT_ID):
        row = conn.execute(
            "SELECT scientific_verdict, supersedes_experiment_id FROM experiment_registry WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
        assert row[1] == w8se3._CORRECTED_DATA_RERUN_OF[experiment_id]
        assert conn.execute(
            "SELECT metric_value FROM experiment_results WHERE experiment_id=? AND metric_name='candle_data_version'",
            (experiment_id,),
        ).fetchone()[0] == w8se3.CANDLE_DATA_VERSION
    conn.close()


# ---- real-data integration (disposable copy via conftest isolation) ----

def test_correction_impact_audit_real_data_matches_reconciliation_expectations():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        audit = w8se3.compute_correction_impact_audit(conn)
        assert audit["mismatches"] == {}
        assert audit["actual"] == {
            "affected_physical_row_n": 45, "distinct_signal_n": 28, "distinct_event_n": 24,
            "distinct_cycle_n": 18, "class_b_n": 0,
        }
        assert audit["concentrated_in_swing_24h"] is True
    finally:
        conn.close()


def test_real_data_smoke_freeze_and_record_and_idempotent_and_002_untouched():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)

        pre_signal_n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_path_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_provenance_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]

        before_002_raw = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8se3.OLD_RAW_BPS_EXPERIMENT_ID,),
        ).fetchall()
        before_002_vol = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8se3.OLD_VOL_NORMALIZED_EXPERIMENT_ID,),
        ).fetchall()

        r1 = w8se3.freeze_and_record(conn)

        assert r1["old_experiments_untouched"] is True
        after_002_raw = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8se3.OLD_RAW_BPS_EXPERIMENT_ID,),
        ).fetchall()
        after_002_vol = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8se3.OLD_VOL_NORMALIZED_EXPERIMENT_ID,),
        ).fetchall()
        assert after_002_raw == before_002_raw
        assert after_002_vol == before_002_vol

        # mandatory effective-path integrity gate passed on real data
        assert r1["effective_path_integrity"]["passed"] is True
        assert r1["effective_path_integrity"]["audit"]["physical_row_count_total"] == 1466
        assert r1["effective_path_integrity"]["audit"]["duplicate_physical_pair_n"] == 170
        assert r1["effective_path_integrity"]["audit"]["effective_row_count"] == 1296
        assert r1["effective_path_integrity"]["audit"]["duplicate_effective_pair_n"] == 0

        assert len(r1["cell_order"]) == 16
        assert r1["raw_signal_n_population"] > 0

        # global SHORT cycle split -- verified, not forced, against the coverage-only expectation
        assert r1["global_split"]["total_cycle_n"] == 61
        assert r1["global_split"]["train_cycle_n"] == 42
        assert r1["global_split"]["test_cycle_n"] == 19
        assert r1["coverage_expectation_check"]["matches_expectation"] is True

        # zero cycle straddling in every horizon + every cell
        for horizon, rep in r1["per_horizon_split_report"].items():
            assert rep["cycle_straddling_violations"] == 0
        for key in r1["cell_order"]:
            assert r1["cells"][key]["cycle_straddling_violations"] == 0

        # PRE-OUTCOME SUFFICIENCY GATE: TEST partition has only 19 cycles (<MIN_BUCKET_N=20) globally,
        # and every horizon's own test_cycle_n is also <20 -- every cell must be INSUFFICIENT_SAMPLE,
        # verified rather than assumed
        for horizon, rep in r1["per_horizon_split_report"].items():
            assert rep["test_cycle_n"] < 20
            assert rep["sufficiency_verdict"] == "INSUFFICIENT_SAMPLE"
        for key in r1["cell_order"]:
            cell = r1["cells"][key]
            assert cell["sample_sufficiency"] == "INSUFFICIENT_SAMPLE"
            assert cell["bootstrap_ci95"] == (None, None)
            assert cell["bootstrap_n_valid_draws"] == 0
            assert cell["permutation_observed_diff"] is None
            assert cell["permutation_p_value"] is None
            assert cell["permutation_p_value_holm_adjusted"] is None  # Holm did NO work (n=0 valid p-values)
            assert cell["closure_classification"] == "INSUFFICIENT_SAMPLE"
            assert cell["descriptive_only_label"] == "DESCRIPTIVE_ONLY_NOT_INFERENTIAL"
            # descriptive-only fields still present (full-population, not converted into a claim)
            assert cell["full_median"] is not None or cell["raw_signal_n"] == 0
            assert "q50" in cell and "iqr" in cell

        assert r1["family_verdict"] == "EXPANDED_SHORT_INSUFFICIENT_AFTER_CYCLE_GROUPED_SPLIT_CORRECTED_DATA"

        # correction impact audit matches the reconciliation's SHORT-only expectations
        assert r1["correction_impact_audit"]["mismatches"] == {}
        assert r1["correction_impact_audit"]["concentrated_in_swing_24h"] is True

        # canonical lifecycle/path/provenance tables completely unaffected
        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0] == pre_provenance_n

        n_raw_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se3.RAW_BPS_EXPERIMENT_ID,)
        ).fetchone()[0]
        n_vol_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se3.VOL_NORMALIZED_EXPERIMENT_ID,)
        ).fetchone()[0]

        # idempotent rerun -- must resolve to NOOP_IDENTICAL under the new immutable ledger, not a
        # duplicate INSERT or an IMMUTABLE_EXPERIMENT_CONFLICT (content is byte-identical since
        # nothing about the corrected population changed between the two calls)
        r2 = w8se3.freeze_and_record(conn)
        assert r2["family_verdict"] == r1["family_verdict"]
        assert r2["old_experiments_untouched"] is True
        n_raw_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se3.RAW_BPS_EXPERIMENT_ID,)
        ).fetchone()[0]
        n_vol_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se3.VOL_NORMALIZED_EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_raw_1 == n_raw_2
        assert n_vol_1 == n_vol_2

        assert r1["comparison_with_v002"]["v003_family_verdict"] == r1["family_verdict"]
    finally:
        conn.close()


def test_comparison_with_v002_real_data_structure():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        family = w8se3.compute_family(conn)
        assert family["blocked"] is False
        comparison = w8se3.compare_with_v002(conn, family)
        assert comparison["v002_raw_bps_experiment_id"] == w8se3.OLD_RAW_BPS_EXPERIMENT_ID
        assert comparison["v002_vol_normalized_experiment_id"] == w8se3.OLD_VOL_NORMALIZED_EXPERIMENT_ID
        assert len(comparison["cell_verdict_changes"]) == 16
        assert isinstance(comparison["any_cell_changed_from_insufficient_sample"], bool)
        # both -002 and -003 remain fully insufficient (population growth wasn't enough to cross
        # MIN_BUCKET_N=20 on the TEST side) -- reported, never forced
        assert comparison["any_cell_changed_from_insufficient_sample"] is False
        assert "not_independent_replications_note" in comparison
        assert len(comparison["horizon_sufficiency_changes"]) == 4
    finally:
        conn.close()


def test_ledger_write_is_insert_then_noop_identical_real_data():
    """Explicit proof of the mandatory immutable-ledger contract for this
    experiment's own two ids: first execution against the real data (already
    completed by a prior batch run in this isolated copy, OR fresh) must
    never raise, and a byte-identical rerun must leave experiment_results
    row counts unchanged (NOOP_IDENTICAL, not a duplicate insert)."""
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)
        w8se3.freeze_and_record(conn)  # first call: INSERT or already-NOOP_IDENTICAL from a prior test
        n1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se3.RAW_BPS_EXPERIMENT_ID,)
        ).fetchone()[0]
        w8se3.freeze_and_record(conn)  # second call: must be a no-op, never raise, never duplicate
        n2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se3.RAW_BPS_EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n1 == n2
    finally:
        conn.close()
