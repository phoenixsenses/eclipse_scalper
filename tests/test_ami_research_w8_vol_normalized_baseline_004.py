"""BATCH-P7B-1 (W8-VOL-NORMALIZED-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED):
tests for ami/research/w8_vol_normalized_baseline_004_long_corrected_cycle_grouped.py
-- the corrected-data + cycle-grouped-split rerun of
E-W8-VOL-NORMALIZED-BASELINE-001's LONG-only portion, paired with the
already-completed E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED.

Run: pytest tests/test_ami_research_w8_vol_normalized_baseline_004.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import pytest

import ami.research.w8_hold_baseline_004_long_corrected_cycle_grouped as w8h4
import ami.research.w8_short_expanded_baseline as w8se
import ami.research.w8_vol_normalized_baseline as w8v_v001
import ami.research.w8_vol_normalized_baseline_004_long_corrected_cycle_grouped as w8v4

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "reentry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
)
_FORBIDDEN_OUTCOME_TERMS = ("win_rate", "alpha_claim", "economic_edge")
_FORBIDDEN_STRATIFICATION_TERMS = ("high_vol", "low_vol", "vol_bucket", "vol_threshold", "median_volatility")


def test_no_management_outcome_or_stratification_terms_in_module_source():
    src = inspect.getsource(w8v4).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS + _FORBIDDEN_OUTCOME_TERMS + _FORBIDDEN_STRATIFICATION_TERMS
            if t in src]
    assert hits == [], f"forbidden terms found: {hits}"


def test_no_volatility_state_classification_functions():
    """This is not volatility-state stratification -- no HIGH/LOW label, no
    median-threshold, no regime-fitting function anywhere in this module."""
    assert not hasattr(w8v4, "_vol_bucket")
    assert not hasattr(w8v4, "build_match_profile")
    assert not hasattr(w8v4, "classify_volatility_regime")


def test_only_vol_normalized_metrics_in_this_batch():
    assert w8v4.METRICS == ("mfe_anchor_vol_units", "mae_anchor_vol_units")
    assert "mfe_bps" not in w8v4.METRICS
    assert "mae_bps" not in w8v4.METRICS


def test_new_experiment_id_records_all_required_metadata():
    assert w8v4.EXPERIMENT_ID == "E-W8-VOL-NORMALIZED-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED"
    assert w8v4.HISTORICAL_REFERENCE_EXPERIMENT_ID == w8v_v001.EXPERIMENT_ID == "E-W8-VOL-NORMALIZED-BASELINE-001"
    assert (
        w8v4.PAIRED_RAW_BASELINE_EXPERIMENT_ID == w8h4.EXPERIMENT_ID
        == "E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED"
    )
    assert w8v4.CANDLE_DATA_VERSION == "candle-binance-fapi-repair-v1"
    assert w8v4.PATH_DATA_VERSION == "path-v2-candle-repair-r1"
    assert w8v4.METHODOLOGICAL_CHANGE == "SIGNAL_LEVEL_SPLIT_TO_INDEPENDENT_CYCLE_GROUPED_SPLIT"


def test_reuses_shared_split_and_cell_machinery_verbatim():
    assert w8v4.compute_global_cycle_split is w8se.compute_global_cycle_split
    assert w8v4.split_rows_by_cycle_keys is w8se.split_rows_by_cycle_keys
    assert w8v4.assert_zero_cycle_straddling is w8se.assert_zero_cycle_straddling
    assert w8v4._cycle_key is w8se._cycle_key
    assert w8v4.compute_cell is w8se.compute_cell
    # raw population fetch reused directly from the paired raw baseline module, never reimplemented
    assert w8v4.fetch_raw_population is w8h4.fetch_population


# ---- synthetic tests for the mandatory effective-path integrity gate ----

def test_verify_effective_path_selection_integrity_passes_on_matching_audit(monkeypatch):
    monkeypatch.setattr(w8v4, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 1466, "duplicate_physical_pair_n": 170,
        "effective_row_count": 1296, "duplicate_effective_pair_n": 0,
    })
    result = w8v4.verify_effective_path_selection_integrity(conn=None)
    assert result["passed"] is True


def test_verify_effective_path_selection_integrity_fails_closed_on_mismatch(monkeypatch):
    monkeypatch.setattr(w8v4, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 1466, "duplicate_physical_pair_n": 170,
        "effective_row_count": 1200, "duplicate_effective_pair_n": 2,
    })
    result = w8v4.verify_effective_path_selection_integrity(conn=None)
    assert result["passed"] is False
    assert "effective_row_count" in result["mismatches"]


def test_compute_family_blocks_on_integrity_failure(monkeypatch):
    monkeypatch.setattr(w8v4, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 0, "duplicate_physical_pair_n": 0,
        "effective_row_count": 0, "duplicate_effective_pair_n": 9,
    })
    family = w8v4.compute_family(conn=None)
    assert family["blocked"] is True
    assert family["family_verdict"] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
    assert "cells" not in family


def test_freeze_and_record_blocked_branch_writes_verdict(tmp_path, monkeypatch):
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(tmp_path / "blocked_vol_test.sqlite")
    schema_mod.init_schema(conn)

    monkeypatch.setattr(w8v4, "effective_path_selection_audit", lambda c, ctx: {
        "physical_row_count_total": 1, "duplicate_physical_pair_n": 1,
        "effective_row_count": 1, "duplicate_effective_pair_n": 1,
    })

    result = w8v4.freeze_and_record(conn)
    assert result["family_verdict"] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"

    row = conn.execute(
        "SELECT scientific_verdict, supersedes_experiment_id FROM experiment_registry WHERE experiment_id=?",
        (w8v4.EXPERIMENT_ID,),
    ).fetchone()
    assert row is not None
    assert row[0] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
    assert row[1] == w8v4.HISTORICAL_REFERENCE_EXPERIMENT_ID
    conn.close()


# ---- real-data integration (disposable copy via conftest isolation) ----

def test_real_data_smoke_split_reused_byte_exact_from_paired_raw_v004():
    """The mandatory split-reuse contract: this module's own freshly computed
    split must match E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED's
    ALREADY-STORED global_cycle_split exactly -- proven, not assumed."""
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        family = w8v4.compute_family(conn)
        assert family["blocked"] is False
        assert family["split_integrity"]["matches"] is True
        assert family["global_split"] == family["split_integrity"]["stored_paired_split"]
    finally:
        conn.close()


def test_real_data_smoke_freeze_and_record_and_idempotent_and_prior_experiments_untouched():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)

        pre_signal_n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_path_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_provenance_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]
        before_v001 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8v4.HISTORICAL_REFERENCE_EXPERIMENT_ID,),
        ).fetchall()
        before_raw_v004 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8v4.PAIRED_RAW_BASELINE_EXPERIMENT_ID,),
        ).fetchall()

        r1 = w8v4.freeze_and_record(conn)

        # both prior experiments (historical v001 AND paired raw v004) must remain byte-identical
        after_v001 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8v4.HISTORICAL_REFERENCE_EXPERIMENT_ID,),
        ).fetchall()
        after_raw_v004 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8v4.PAIRED_RAW_BASELINE_EXPERIMENT_ID,),
        ).fetchall()
        assert after_v001 == before_v001
        assert after_raw_v004 == before_raw_v004

        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0] == pre_provenance_n

        assert r1["effective_path_integrity"]["passed"] is True
        assert r1["split_integrity"]["matches"] is True

        assert len(r1["cell_order"]) == 8
        assert r1["raw_signal_n_population"] > 0

        # split reuse: MIN_BUCKET_N never reduced, split never recomputed from vol-filtered population
        assert r1["global_split"]["train_cycle_n"] >= 20
        assert r1["global_split"]["test_cycle_n"] >= 20

        for horizon, rep in r1["coverage_report"]["per_horizon"].items():
            assert rep["cycle_straddling_violations"] == 0
            assert rep["sufficiency_verdict"] in ("OK", "INSUFFICIENT_SAMPLE")
            assert "volatility_invalid_signal_n" in rep
            assert "monthly_distribution" in rep
            assert "setup_composition" in rep
            assert "paired_raw_v004_signal_n" in rep
            # volatility exclusions can only shrink the vol population relative to the raw one
            assert rep["vol_signal_n"] <= rep["raw_signal_n"]

        for key in r1["cell_order"]:
            cell = r1["cells"][key]
            assert cell["closure_classification"] in (
                "INSUFFICIENT_SAMPLE", "ANSWERED_SUPPORTED_STABLE_BASELINE", "ANSWERED_REGIME_DEPENDENT_BASELINE",
            )
            if cell["sample_sufficiency"] == "INSUFFICIENT_SAMPLE":
                assert cell["train_cycle_n"] < 20 or cell["test_cycle_n"] < 20
                assert cell["permutation_p_value"] is None
                assert cell["descriptive_only_label"] == "DESCRIPTIVE_ONLY_NOT_INFERENTIAL"
            else:
                assert cell["train_cycle_n"] >= 20 and cell["test_cycle_n"] >= 20

        assert r1["family_verdict"] in (
            "LONG_VOL_NORMALIZED_BASELINE_STABLE_CORRECTED_CYCLE_GROUPED",
            "LONG_VOL_NORMALIZED_BASELINE_REGIME_DEPENDENT_CORRECTED_CYCLE_GROUPED",
            "MIXED_BY_HORIZON_OR_METRIC_CORRECTED_CYCLE_GROUPED",
            "LONG_VOL_NORMALIZED_BASELINE_INSUFFICIENT",
        )

        cmp_raw = r1["comparison_with_raw_v004"]
        assert cmp_raw["paired_raw_baseline_experiment_id"] == w8v4.PAIRED_RAW_BASELINE_EXPERIMENT_ID
        assert cmp_raw["comparison_label"] in (
            "RAW_AND_VOL_NORMALIZED_LONG_BASELINES_CONSISTENT", "PARTIALLY_CONSISTENT_AFTER_NORMALIZATION",
            "MATERIAL_NORMALIZATION_EFFECT", "INSUFFICIENT_VOL_NORMALIZED_POPULATION",
        )
        assert len(cmp_raw["cell_comparisons"]) == 8

        cmp_v001 = r1["comparison_with_v001"]
        assert cmp_v001["v001_experiment_id"] == "E-W8-VOL-NORMALIZED-BASELINE-001"
        assert cmp_v001["comparison_label"] in (
            "QUALITATIVELY_CONSISTENT_AFTER_CORRECTION_AND_CYCLE_GROUPING", "PARTIALLY_CONSISTENT",
            "MATERIAL_BASELINE_CHANGE", "NOT_COMPARABLE_DUE_TO_METHOD_CHANGE",
        )
        assert "not_comparable_note" in cmp_v001["population_changes"]

        n_results_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8v4.EXPERIMENT_ID,)
        ).fetchone()[0]

        r2 = w8v4.freeze_and_record(conn)
        n_results_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8v4.EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_results_1 == n_results_2
        assert r2["family_verdict"] == r1["family_verdict"]

        after_v001_again = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8v4.HISTORICAL_REFERENCE_EXPERIMENT_ID,),
        ).fetchall()
        after_raw_v004_again = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8v4.PAIRED_RAW_BASELINE_EXPERIMENT_ID,),
        ).fetchall()
        assert after_v001_again == before_v001
        assert after_raw_v004_again == before_raw_v004
    finally:
        conn.close()
