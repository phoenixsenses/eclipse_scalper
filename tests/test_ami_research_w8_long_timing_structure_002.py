"""BATCH-P7B-1 (W8-LONG-TIMING-STRUCTURE-002-CANDLE-REPAIR-CYCLE-GROUPED):
tests for ami/research/w8_long_timing_structure_002_candle_repair_cycle_grouped.py
-- the corrected-data + cycle-grouped-split rerun of
E-W8-LONG-TIMING-STRUCTURE-001, reusing the paired
E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED's frozen cycle split.

Run: pytest tests/test_ami_research_w8_long_timing_structure_002.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import pytest

import ami.research.w8_hold_baseline_004_long_corrected_cycle_grouped as w8h4
import ami.research.w8_long_timing_structure as w8t_v001
import ami.research.w8_long_timing_structure_002_candle_repair_cycle_grouped as w8t2
import ami.research.w8_short_expanded_baseline as w8se

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "reentry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
    "entry_rule", "exit_rule", "hold_optimization",
)
# "alpha_claim"/"economic_claim" excluded -- the module's own NO_ECONOMIC_OR_ALPHA_CLAIM disclaimer
# legitimately contains that substring in prose (same convention as sibling modules' own tests)
_FORBIDDEN_OUTCOME_TERMS = ("win_rate", "economic_edge")


def test_no_management_or_outcome_terms_in_module_source():
    src = inspect.getsource(w8t2).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS + _FORBIDDEN_OUTCOME_TERMS if t in src]
    assert hits == [], f"forbidden terms found: {hits}"


def test_no_short_pooling():
    assert w8t2.DIRECTION == "LONG"


def test_new_experiment_id_records_all_required_metadata():
    assert w8t2.EXPERIMENT_ID == "E-W8-LONG-TIMING-STRUCTURE-002-CANDLE-REPAIR-CYCLE-GROUPED"
    assert w8t2.CORRECTED_DATA_RERUN_OF == w8t_v001.EXPERIMENT_ID == "E-W8-LONG-TIMING-STRUCTURE-001"
    assert (
        w8t2.PAIRED_CYCLE_SPLIT_EXPERIMENT_ID == w8h4.EXPERIMENT_ID
        == "E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED"
    )
    assert w8t2.CANDLE_DATA_VERSION == "candle-binance-fapi-repair-v1"
    assert w8t2.PATH_DATA_VERSION == "path-v2-candle-repair-r1"


def test_reuses_shared_machinery_verbatim():
    """Split machinery from w8_short_expanded_baseline, timing-family
    machinery (compute_cell/compute_horizon_descriptive/_rate/TIMING_METRICS)
    from v001, and the raw population fetch from the paired v004 baseline --
    all reused by `is` identity, none reimplemented."""
    assert w8t2.compute_global_cycle_split is w8se.compute_global_cycle_split
    assert w8t2.split_rows_by_cycle_keys is w8se.split_rows_by_cycle_keys
    assert w8t2.assert_zero_cycle_straddling is w8se.assert_zero_cycle_straddling
    assert w8t2._cycle_key is w8se._cycle_key
    assert w8t2.compute_cell is w8t_v001.compute_cell
    assert w8t2.compute_horizon_descriptive is w8t_v001.compute_horizon_descriptive
    assert w8t2._rate is w8t_v001._rate
    assert w8t2.TIMING_METRICS == w8t_v001.TIMING_METRICS == ("time_to_mfe_ms", "time_to_mae_ms")
    assert w8t2.fetch_raw_population is w8h4.fetch_population


# ---- synthetic tests for the mandatory effective-path integrity gate ----

def test_verify_effective_path_selection_integrity_fails_closed_on_mismatch(monkeypatch):
    monkeypatch.setattr(w8t2, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 1466, "duplicate_physical_pair_n": 170,
        "effective_row_count": 1290, "duplicate_effective_pair_n": 3,
    })
    result = w8t2.verify_effective_path_selection_integrity(conn=None)
    assert result["passed"] is False
    assert "effective_row_count" in result["mismatches"]


def test_compute_family_blocks_on_integrity_failure(monkeypatch):
    monkeypatch.setattr(w8t2, "effective_path_selection_audit", lambda conn, ctx: {
        "physical_row_count_total": 0, "duplicate_physical_pair_n": 0,
        "effective_row_count": 0, "duplicate_effective_pair_n": 9,
    })
    family = w8t2.compute_family(conn=None)
    assert family["blocked"] is True
    assert family["family_verdict"] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
    assert "cells" not in family


def test_freeze_and_record_blocked_branch_writes_verdict(tmp_path, monkeypatch):
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(tmp_path / "blocked_timing_test.sqlite")
    schema_mod.init_schema(conn)

    monkeypatch.setattr(w8t2, "effective_path_selection_audit", lambda c, ctx: {
        "physical_row_count_total": 1, "duplicate_physical_pair_n": 1,
        "effective_row_count": 1, "duplicate_effective_pair_n": 1,
    })

    result = w8t2.freeze_and_record(conn)
    assert result["family_verdict"] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"

    row = conn.execute(
        "SELECT scientific_verdict, supersedes_experiment_id FROM experiment_registry WHERE experiment_id=?",
        (w8t2.EXPERIMENT_ID,),
    ).fetchone()
    assert row is not None
    assert row[0] == "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"
    assert row[1] == w8t2.CORRECTED_DATA_RERUN_OF
    conn.close()


# ---- real-data integration (disposable copy via conftest isolation) ----

def test_correction_impact_audit_real_data_matches_reconciliation_expectations():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        audit = w8t2.compute_correction_impact_audit(conn)
        assert audit["mismatches"] == {}
        assert audit["actual"] == {
            "affected_physical_row_n": 104, "distinct_signal_n": 71, "distinct_event_n": 71, "distinct_cycle_n": 49,
        }
        assert audit["newly_eligible_signal_n"] == 71
        assert audit["newly_eligible_cycle_n"] == 49
    finally:
        conn.close()


def test_real_data_smoke_split_reused_byte_exact_from_paired_v004():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        family = w8t2.compute_family(conn)
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
            (w8t2.CORRECTED_DATA_RERUN_OF,),
        ).fetchall()
        before_raw_v004 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8t2.PAIRED_CYCLE_SPLIT_EXPERIMENT_ID,),
        ).fetchall()

        r1 = w8t2.freeze_and_record(conn)

        after_v001 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8t2.CORRECTED_DATA_RERUN_OF,),
        ).fetchall()
        after_raw_v004 = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8t2.PAIRED_CYCLE_SPLIT_EXPERIMENT_ID,),
        ).fetchall()
        assert after_v001 == before_v001
        assert after_raw_v004 == before_raw_v004

        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0] == pre_provenance_n

        assert r1["effective_path_integrity"]["passed"] is True
        assert r1["split_integrity"]["matches"] is True

        assert len(r1["cell_order"]) == 8
        assert r1["global_split"]["train_cycle_n"] >= 20
        assert r1["global_split"]["test_cycle_n"] >= 20

        for horizon, rep in r1["coverage_report"]["per_horizon"].items():
            assert rep["cycle_straddling_violations"] == 0
            assert rep["sufficiency_verdict"] in ("OK", "INSUFFICIENT_SAMPLE")
            assert "monthly_distribution" in rep
            assert "setup_composition" in rep
            assert rep["paired_raw_v004_signal_n"] == rep["raw_signal_n"]

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

        # secondary descriptive present per horizon, no p-values
        for horizon in ("scalp_30m", "scalp_1h", "swing_4h", "swing_24h"):
            desc = r1["descriptive_by_horizon"][horizon]
            assert "intrabar_order_status_rates" in desc
            assert "zero_at_reference_rate" in desc
            assert "timing_delta_ms_quantiles" in desc
            assert "permutation_p_value" not in desc

        assert r1["family_verdict"] in (
            "LONG_TIMING_STRUCTURE_STABLE_CORRECTED_DATA", "LONG_TIMING_STRUCTURE_REGIME_DEPENDENT_CORRECTED_DATA",
            "MIXED_LONG_TIMING_STRUCTURE_CORRECTED_DATA", "LONG_TIMING_STRUCTURE_INSUFFICIENT_CORRECTED_DATA",
        )

        cmp = r1["comparison_with_v001"]
        assert cmp["v001_experiment_id"] == "E-W8-LONG-TIMING-STRUCTURE-001"
        assert cmp["comparison_label"] in (
            "TIMING_STRUCTURE_CONSISTENT_ON_CORRECTED_EXPANDED_COHORT", "PARTIALLY_CONSISTENT",
            "MATERIAL_TIMING_STRUCTURE_CHANGE", "INSUFFICIENT_CORRECTED_TIMING_POPULATION",
        )
        assert len(cmp["cell_changes"]) == 8
        assert isinstance(cmp["mae_first_increases_with_horizon_survives"], bool)

        n_results_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8t2.EXPERIMENT_ID,)
        ).fetchone()[0]

        r2 = w8t2.freeze_and_record(conn)
        n_results_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8t2.EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_results_1 == n_results_2
        assert r2["family_verdict"] == r1["family_verdict"]

        after_v001_again = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8t2.CORRECTED_DATA_RERUN_OF,),
        ).fetchall()
        after_raw_v004_again = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8t2.PAIRED_CYCLE_SPLIT_EXPERIMENT_ID,),
        ).fetchall()
        assert after_v001_again == before_v001
        assert after_raw_v004_again == before_raw_v004
    finally:
        conn.close()
