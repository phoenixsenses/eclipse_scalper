"""BATCH-P7B-1 (W8-LONG-NESTED-PATH-ACCUMULATION-001): tests for
ami/research/w8_long_nested_path_accumulation.py -- nested MFE/|MAE|
accumulation across 30m/1h/4h/24h for a fixed, horizon-complete LONG cohort.

Run: pytest tests/test_ami_research_w8_long_nested_path_accumulation.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import ami.research.w8_long_nested_path_accumulation as w8np
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
    "recommended_hold",
)
_FORBIDDEN_SELECTION_IDENTIFIERS = ("win_rate", "threshold_sweep", "conviction_score")


def test_no_graveyarded_management_terms_in_module_source():
    src = inspect.getsource(w8np).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS if t in src]
    assert hits == [], f"forbidden management-rule terms found: {hits}"
    sel_hits = [t for t in _FORBIDDEN_SELECTION_IDENTIFIERS if t in src]
    assert sel_hits == [], f"forbidden outcome/selection identifiers found: {sel_hits}"


def test_no_action_permission_escalation_in_module_source():
    src = inspect.getsource(w8np)
    assert "authorize(" not in src
    assert ".promote(" not in src
    assert "OPEN_LONG" not in src
    assert "OPEN_SHORT" not in src


def test_no_order_router_or_execution_import():
    src = inspect.getsource(w8np)
    for forbidden in ("execution.", "risk.", "brain.", "order_router", "entry_loop", "position_manager"):
        assert forbidden not in src, f"forbidden import/reference: {forbidden}"


def test_direction_is_long_only():
    assert w8np.DIRECTION == "LONG"


def test_reuses_split_machinery_verbatim_from_short_expanded_module():
    import ami.research.w8_short_expanded_baseline as w8se
    assert w8np.compute_global_cycle_split is w8se.compute_global_cycle_split
    assert w8np.split_rows_by_cycle_keys is w8se.split_rows_by_cycle_keys
    assert w8np.assert_zero_cycle_straddling is w8se.assert_zero_cycle_straddling
    assert w8np._cycle_key is w8se._cycle_key


def test_primary_family_is_exactly_6_cells():
    assert len(w8np.ALL_DELTA_FIELDS) == 6
    assert len(w8np.MFE_DELTA_FIELDS) == 3
    assert len(w8np.MAE_DELTA_FIELDS) == 3


def test_frozen_field_names_match_operator_spec():
    assert w8np.MFE_DELTA_FIELDS == ("delta_mfe_30m_to_1h", "delta_mfe_1h_to_4h", "delta_mfe_4h_to_24h")
    assert w8np.MAE_DELTA_FIELDS == (
        "delta_abs_mae_30m_to_1h", "delta_abs_mae_1h_to_4h", "delta_abs_mae_4h_to_24h",
    )


# ---- derived fields + non-negativity (synthetic, no DB needed) ----

def test_compute_derived_fields_matches_frozen_formulas():
    row = {
        "mfe_bps_30m": 10.0, "mfe_bps_1h": 25.0, "mfe_bps_4h": 60.0, "mfe_bps_24h": 90.0,
        "mae_bps_30m": -5.0, "mae_bps_1h": -12.0, "mae_bps_4h": -12.0, "mae_bps_24h": -30.0,
    }
    out = w8np.compute_derived_fields(row)
    assert out["delta_mfe_30m_to_1h"] == 15.0
    assert out["delta_mfe_1h_to_4h"] == 35.0
    assert out["delta_mfe_4h_to_24h"] == 30.0
    assert out["delta_abs_mae_30m_to_1h"] == 7.0
    assert out["delta_abs_mae_1h_to_4h"] == 0.0
    assert out["delta_abs_mae_4h_to_24h"] == 18.0
    assert out["delta_diff_30m_to_1h"] == 8.0  # 15.0 - 7.0


def test_assert_nested_nonnegativity_detects_a_real_violation():
    rows = [w8np.compute_derived_fields({
        "signal_id": "S1", "mfe_bps_30m": 10.0, "mfe_bps_1h": 5.0, "mfe_bps_4h": 20.0, "mfe_bps_24h": 30.0,
        "mae_bps_30m": -5.0, "mae_bps_1h": -3.0, "mae_bps_4h": -8.0, "mae_bps_24h": -10.0,
    })]
    result = w8np.assert_nested_nonnegativity(rows)
    # mfe_bps_1h (5.0) < mfe_bps_30m (10.0) is a genuine violation of the nesting property
    assert result["violation_n"] >= 1
    assert any(v["field"] == "delta_mfe_30m_to_1h" for v in result["violations"])


def test_assert_nested_nonnegativity_clean_population_has_zero_violations():
    rows = [w8np.compute_derived_fields({
        "signal_id": "S1", "mfe_bps_30m": 10.0, "mfe_bps_1h": 10.0, "mfe_bps_4h": 20.0, "mfe_bps_24h": 30.0,
        "mae_bps_30m": -5.0, "mae_bps_1h": -5.0, "mae_bps_4h": -8.0, "mae_bps_24h": -8.0,
    })]
    result = w8np.assert_nested_nonnegativity(rows)
    assert result["violation_n"] == 0


# ---- capture-fraction zero-denominator handling ----

def test_capture_fraction_excludes_zero_denominator_never_divides_by_zero():
    rows = [
        {"mfe_bps_30m": 5.0, "mfe_bps_24h": 0.0},   # zero denominator -- must be excluded, not crash
        {"mfe_bps_30m": 10.0, "mfe_bps_24h": 20.0},  # ratio = 0.5
    ]
    result = w8np._capture_fraction_stats(rows, "mfe_bps_30m", "mfe_bps_24h")
    assert result["n_excluded_zero_denominator"] == 1
    assert result["n_included"] == 1
    assert result["median_ratio"] == 0.5


# ---- sufficiency (cycle-count based) ----

def _cohort_row(signal_id, cycle_id, birth_ts, mfe30, mfe1h, mfe4h, mfe24h, mae30, mae1h, mae4h, mae24h):
    row = {
        "signal_id": signal_id, "independent_cycle_id": cycle_id, "source_event_id": f"EVT-{signal_id}",
        "signal_birth_ts": birth_ts,
        "mfe_bps_30m": mfe30, "mfe_bps_1h": mfe1h, "mfe_bps_4h": mfe4h, "mfe_bps_24h": mfe24h,
        "mae_bps_30m": mae30, "mae_bps_1h": mae1h, "mae_bps_4h": mae4h, "mae_bps_24h": mae24h,
    }
    return w8np.compute_derived_fields(row)


def test_sufficiency_is_cycle_count_based():
    rows = []
    for i in range(25):
        cyc = "CYC-A" if i < 20 else "CYC-B"
        rows.append(_cohort_row(f"S{i}", cyc, i * 1000, 1, 2, 3, 4, -1, -2, -3, -4))
    split = w8np.compute_global_cycle_split(rows)
    cell = w8np.compute_cell(rows, "delta_mfe_30m_to_1h", split["train_cycle_keys"], split["test_cycle_keys"])
    assert cell["distinct_independent_cycle_n"] == 2
    assert cell["sample_sufficiency"] == "INSUFFICIENT_SAMPLE"


def test_sufficient_when_enough_independent_cycles():
    rows = [_cohort_row(f"S{i}", f"CYC-{i}", i * 1000, i % 5, i % 5 + 1, i % 5 + 2, i % 5 + 3,
                         -(i % 3), -(i % 3), -(i % 3 + 1), -(i % 3 + 1))
            for i in range(100)]
    split = w8np.compute_global_cycle_split(rows)
    assert split["train_cycle_n"] >= MIN_BUCKET_N
    assert split["test_cycle_n"] >= MIN_BUCKET_N
    cell = w8np.compute_cell(rows, "delta_mfe_30m_to_1h", split["train_cycle_keys"], split["test_cycle_keys"])
    assert cell["sample_sufficiency"] == "OK"
    assert cell["cycle_straddling_violations"] == 0


# ---- real-data smoke test (disposable copy only, via conftest isolation) ----

def test_real_data_smoke_freeze_and_record_and_idempotent():
    import ami.warehouse.schema as schema_mod
    from ami.lifecycle.path_schema import init_path_schema

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)
        init_path_schema(conn)

        pre_signal_n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_path_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_provenance_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]

        r1 = w8np.freeze_and_record(conn)
        assert len(r1["cell_order"]) == 6
        assert r1["population_report"]["signal_n"] > 0
        assert r1["population_report"]["independent_cycle_n"] <= r1["population_report"]["source_event_n"]
        assert r1["population_report"]["source_event_n"] <= r1["population_report"]["signal_n"]
        assert r1["population_report"]["cycle_straddling_violations"] == 0

        # the common-cohort must be a proper subset of the full LONG population
        assert r1["population_report"]["excluded_incomplete_horizon_n"] >= 0
        assert (r1["population_report"]["signal_n"] + r1["population_report"]["excluded_incomplete_horizon_n"]
                == r1["population_report"]["long_signals_with_any_ok_horizon_n"])

        # nested non-negativity must hold on the real population (or be explicitly reported, never hidden)
        assert r1["nested_nonnegativity_check"]["violation_n"] == 0

        for key in r1["cell_order"]:
            assert r1["cells"][key]["cycle_straddling_violations"] == 0
            assert r1["cells"][key]["closure_classification"] in (
                "INSUFFICIENT_SAMPLE", "ANSWERED_SUPPORTED_STABLE_BASELINE", "ANSWERED_REGIME_DEPENDENT_BASELINE",
            )

        assert r1["family_verdict"] in (
            "LONG_NESTED_PATH_STABLE", "LONG_NESTED_PATH_REGIME_DEPENDENT",
            "MIXED_BY_INTERVAL_OR_METRIC", "INSUFFICIENT_COMMON_COHORT",
        )

        # canonical lifecycle/path/provenance tables must be completely unaffected
        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0] == pre_provenance_n

        n_results_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8np.EXPERIMENT_ID,)
        ).fetchone()[0]
        r2 = w8np.freeze_and_record(conn)
        n_results_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8np.EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_results_1 == n_results_2
        assert r2["family_verdict"] == r1["family_verdict"]
        assert r2["population_report"] == r1["population_report"]
    finally:
        conn.close()
