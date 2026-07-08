"""BATCH-P7B-1 (W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED): tests for
ami/research/w8_hold_baseline_004_long_corrected_cycle_grouped.py -- the
corrected-data + cycle-grouped-split rerun of E-W8-HOLD-BASELINE-001's
LONG-only raw mfe_bps/mae_bps portion.

Run: pytest tests/test_ami_research_w8_hold_baseline_004.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import pytest

import ami.research.w8_hold_baseline as w8h_v001
import ami.research.w8_hold_baseline_004_long_corrected_cycle_grouped as w8h4
import ami.research.w8_short_expanded_baseline as w8se

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "reentry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
)
# "pnl" excluded -- the module's own NO_ECONOMIC_CLAIM disclaimer legitimately mentions "PnL" in prose
_FORBIDDEN_OUTCOME_TERMS = ("win_rate", "alpha_claim", "economic_edge")


def test_no_management_or_outcome_terms_in_module_source():
    src = inspect.getsource(w8h4).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS + _FORBIDDEN_OUTCOME_TERMS if t in src]
    assert hits == [], f"forbidden terms found: {hits}"


def test_no_volatility_normalized_metrics_in_this_batch():
    assert w8h4.METRICS == ("mfe_bps", "mae_bps")
    assert "mfe_anchor_vol_units" not in w8h4.METRICS
    assert "mae_anchor_vol_units" not in w8h4.METRICS


def test_new_experiment_id_distinct_from_001_and_records_methodology():
    assert w8h4.EXPERIMENT_ID == "E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED"
    assert w8h4.HISTORICAL_REFERENCE_EXPERIMENT_ID == w8h_v001.EXPERIMENT_ID == "E-W8-HOLD-BASELINE-001"
    assert w8h4.EXPERIMENT_ID != w8h4.HISTORICAL_REFERENCE_EXPERIMENT_ID
    assert w8h4.CANDLE_DATA_VERSION == "candle-binance-fapi-repair-v1"
    assert w8h4.PATH_DATA_VERSION == "path-v2-candle-repair-r1"
    assert w8h4.METHODOLOGICAL_CHANGE == "SIGNAL_LEVEL_SPLIT_TO_INDEPENDENT_CYCLE_GROUPED_SPLIT"


def test_reuses_shared_cycle_split_machinery_verbatim():
    """The cycle-grouped split machinery and compute_cell() are imported from
    w8_short_expanded_baseline UNCHANGED -- proven by `is` identity."""
    assert w8h4.compute_global_cycle_split is w8se.compute_global_cycle_split
    assert w8h4.split_rows_by_cycle_keys is w8se.split_rows_by_cycle_keys
    assert w8h4.assert_zero_cycle_straddling is w8se.assert_zero_cycle_straddling
    assert w8h4._cycle_key is w8se._cycle_key
    assert w8h4.compute_cell is w8se.compute_cell


# ---- synthetic tests for the mandatory effective-path integrity gate ----

def test_verify_effective_path_selection_integrity_passes_on_matching_audit(monkeypatch):
    monkeypatch.setattr(w8h4, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 1466, "duplicate_physical_pair_n": 170,
        "effective_row_count": 1296, "duplicate_effective_pair_n": 0,
    })
    result = w8h4.verify_effective_path_selection_integrity(conn=None)
    assert result["passed"] is True
    assert result["mismatches"] == {}


def test_verify_effective_path_selection_integrity_fails_closed_on_mismatch(monkeypatch):
    monkeypatch.setattr(w8h4, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 1466, "duplicate_physical_pair_n": 170,
        "effective_row_count": 1290, "duplicate_effective_pair_n": 4,
    })
    result = w8h4.verify_effective_path_selection_integrity(conn=None)
    assert result["passed"] is False
    assert "effective_row_count" in result["mismatches"]
    assert "duplicate_effective_pair_n" in result["mismatches"]


def test_compute_family_blocks_on_integrity_failure(monkeypatch):
    monkeypatch.setattr(w8h4, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 0, "duplicate_physical_pair_n": 0,
        "effective_row_count": 0, "duplicate_effective_pair_n": 9,
    })
    family = w8h4.compute_family(conn=None)
    assert family["blocked"] is True
    assert family["family_verdict"] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
    assert "cells" not in family


def test_freeze_and_record_blocked_branch_writes_verdict(tmp_path, monkeypatch):
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(tmp_path / "blocked_test.sqlite")
    schema_mod.init_schema(conn)

    monkeypatch.setattr(w8h4, "effective_path_selection_audit", lambda c, ctx: {
        "physical_row_count_total": 1, "duplicate_physical_pair_n": 1,
        "effective_row_count": 1, "duplicate_effective_pair_n": 1,
    })

    result = w8h4.freeze_and_record(conn)
    assert result["family_verdict"] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"

    row = conn.execute(
        "SELECT scientific_verdict, supersedes_experiment_id FROM experiment_registry WHERE experiment_id=?",
        (w8h4.EXPERIMENT_ID,),
    ).fetchone()
    assert row is not None
    assert row[0] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
    assert row[1] == w8h4.HISTORICAL_REFERENCE_EXPERIMENT_ID
    assert conn.execute(
        "SELECT metric_value FROM experiment_results WHERE experiment_id=? AND metric_name='candle_data_version'",
        (w8h4.EXPERIMENT_ID,),
    ).fetchone()[0] == w8h4.CANDLE_DATA_VERSION
    conn.close()


# ---- real-data integration (disposable copy via conftest isolation) ----

def test_real_data_smoke_freeze_and_record_and_idempotent_and_001_untouched():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)

        pre_signal_n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_path_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_provenance_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]
        before_001 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8h4.HISTORICAL_REFERENCE_EXPERIMENT_ID,),
        ).fetchall()

        r1 = w8h4.freeze_and_record(conn)

        # v001 must remain byte-identical
        after_001 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8h4.HISTORICAL_REFERENCE_EXPERIMENT_ID,),
        ).fetchall()
        assert after_001 == before_001

        # canonical tables completely unaffected
        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0] == pre_provenance_n

        # mandatory effective-path integrity gate passed on real data
        assert r1["effective_path_integrity"]["passed"] is True
        assert r1["effective_path_integrity"]["audit"]["physical_row_count_total"] == 1466
        assert r1["effective_path_integrity"]["audit"]["duplicate_physical_pair_n"] == 170
        assert r1["effective_path_integrity"]["audit"]["effective_row_count"] == 1296
        assert r1["effective_path_integrity"]["audit"]["duplicate_effective_pair_n"] == 0

        assert len(r1["cell_order"]) == 8
        assert r1["raw_signal_n_population"] > 0

        # global cycle split -- MIN_BUCKET_N never reduced, no signal-level reversion
        assert r1["global_split"]["train_cycle_n"] >= 20
        assert r1["global_split"]["test_cycle_n"] >= 20

        # pre-outcome coverage report: 4 horizons, zero straddling, sufficiency computed
        for horizon, rep in r1["coverage_report"]["per_horizon"].items():
            assert rep["cycle_straddling_violations"] == 0
            assert rep["sufficiency_verdict"] in ("OK", "INSUFFICIENT_SAMPLE")
            assert "monthly_distribution" in rep
            assert "setup_composition" in rep
        assert "signals_sharing_independent_cycle" in r1["coverage_report"]
        assert "source_events_carrying_multiple_long_signals_n" in r1["coverage_report"]

        # every cell's verdict is one of the allowed labels; MIN_BUCKET_N respected
        for key in r1["cell_order"]:
            cell = r1["cells"][key]
            assert cell["closure_classification"] in (
                "INSUFFICIENT_SAMPLE", "ANSWERED_SUPPORTED_STABLE_BASELINE", "ANSWERED_REGIME_DEPENDENT_BASELINE",
            )
            if cell["sample_sufficiency"] == "INSUFFICIENT_SAMPLE":
                assert cell["train_cycle_n"] < 20 or cell["test_cycle_n"] < 20
                assert cell["bootstrap_ci95"] == (None, None)
                assert cell["permutation_p_value"] is None
                assert cell["descriptive_only_label"] == "DESCRIPTIVE_ONLY_NOT_INFERENTIAL"
            else:
                assert cell["train_cycle_n"] >= 20 and cell["test_cycle_n"] >= 20

        assert r1["family_verdict"] in (
            "LONG_RAW_HOLD_BASELINE_STABLE_CORRECTED_CYCLE_GROUPED",
            "LONG_RAW_HOLD_BASELINE_REGIME_DEPENDENT_CORRECTED_CYCLE_GROUPED",
            "MIXED_BY_HORIZON_OR_METRIC_CORRECTED_CYCLE_GROUPED",
            "LONG_RAW_HOLD_BASELINE_INSUFFICIENT",
        )

        # comparison with v001: never claims independent replication, never forces agreement
        cmp = r1["comparison_with_v001"]
        assert cmp["v001_experiment_id"] == "E-W8-HOLD-BASELINE-001"
        assert cmp["comparison_label"] in (
            "QUALITATIVELY_CONSISTENT_AFTER_CORRECTION_AND_CYCLE_GROUPING", "PARTIALLY_CONSISTENT",
            "MATERIAL_BASELINE_CHANGE", "NOT_COMPARABLE_DUE_TO_METHOD_CHANGE",
        )
        assert "not_comparable_note" in cmp["population_changes"]
        assert len(cmp["population_changes"]["per_horizon_population_changes"]) == 4
        assert len(cmp["cell_changes"]) == 8

        n_results_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8h4.EXPERIMENT_ID,)
        ).fetchone()[0]

        r2 = w8h4.freeze_and_record(conn)
        n_results_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8h4.EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_results_1 == n_results_2
        assert r2["family_verdict"] == r1["family_verdict"]

        after_001_again = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8h4.HISTORICAL_REFERENCE_EXPERIMENT_ID,),
        ).fetchall()
        assert after_001_again == before_001
    finally:
        conn.close()


def test_no_matched_control_reconstruction_in_module_source():
    """This module must never reopen/fabricate the matched-control direction
    assignment question -- no negative-control machinery is imported or
    implemented (the module docstring's own NO_MATCHED_CONTROL_RECONSTRUCTION
    disclaimer legitimately mentions "matched-control" in prose, so this
    checks for actual functions/imports, not the documentation string)."""
    assert not hasattr(w8h4, "sample_matched_control_candidates")
    assert not hasattr(w8h4, "build_match_profile")
    assert not hasattr(w8h4, "compute_negative_control")
