"""BATCH-P7B-1: tests for ami/research/w8_vol_normalized_baseline.py
(W8-VOL-NORMALIZED-BASELINE -- Candidate E, volatility-normalized
continuation of W8-HOLD-BASELINE, NOT a management/exit/stop/re-entry wave).

Run: pytest tests/test_ami_research_w8_vol_normalized_baseline.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import pytest

import ami.research.w8_vol_normalized_baseline as w8v
from ami.research.w8_hold_baseline import DIRECTIONS, HORIZONS

PROV = "test"

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "reentry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
)


def test_no_graveyarded_management_rule_terms_in_module_source():
    src = inspect.getsource(w8v).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS if t in src]
    assert hits == [], f"forbidden management-rule terms found: {hits}"


def test_no_action_permission_escalation_in_module_source():
    src = inspect.getsource(w8v)
    assert "authorize(" not in src
    assert ".promote(" not in src
    assert "OPEN_LONG" not in src
    assert "OPEN_SHORT" not in src
    assert "import ami.governance" not in src
    assert "from ami.governance" not in src


def test_no_order_router_or_execution_import():
    src = inspect.getsource(w8v)
    for forbidden in ("execution.", "risk.", "brain.", "order_router", "entry_loop", "position_manager"):
        assert forbidden not in src, f"forbidden import/reference: {forbidden}"


# ---- reuse verification: same generic machinery as W8-HOLD-BASELINE, not reimplemented ----

def test_reuses_w8_hold_baseline_helpers_verbatim():
    import ami.research.w8_hold_baseline as w8
    assert w8v._cell_rows is w8._cell_rows
    assert w8v.compute_cell is w8.compute_cell
    assert w8v.classify_cell_verdict is w8.classify_cell_verdict


def test_family_is_exactly_16_cells():
    assert len(w8v.METRICS) * len(HORIZONS) * len(DIRECTIONS) == 16


def test_metrics_are_vol_normalized_not_raw_bps():
    assert w8v.METRICS == ("mfe_anchor_vol_units", "mae_anchor_vol_units")
    assert "mfe_bps" not in w8v.METRICS
    assert "mae_bps" not in w8v.METRICS


def test_raw_bps_metric_mapping_is_complete_and_correct():
    assert w8v._RAW_BPS_METRIC_OF == {
        "mfe_anchor_vol_units": "mfe_bps",
        "mae_anchor_vol_units": "mae_bps",
    }


# ---- population filter: observation_status=OK AND volatility_status=OK ----

def test_fetch_population_uses_feature_gateway_equals_filter():
    src = inspect.getsource(w8v.fetch_population)
    assert '"observation_status": "OK"' in src
    assert '"volatility_status": "OK"' in src
    assert "fetch_path_observations" in src
    assert "fetch_lifecycle_signals" in src
    # never a raw SQL SELECT on ami_lifecycle_path_observations/ami_signal_lifecycle
    assert "conn.execute(" not in src


# ---- comparison logic (synthetic, no DB needed) ----

def test_compare_flags_long_cell_regression_from_stable_to_regime_dependent():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE experiment_results (experiment_id TEXT, metric_name TEXT, metric_value TEXT)"
    )
    raw_cell = {
        "raw_signal_n": 216, "train_minus_test_median_diff": 0.5, "iqr": 10.0,
        "closure_classification": "ANSWERED_SUPPORTED_STABLE_BASELINE",
    }
    conn.execute(
        "INSERT INTO experiment_results VALUES (?,?,?)",
        (w8v.RAW_BPS_EXPERIMENT_ID, "cell_mfe_bps|scalp_30m|LONG", str(raw_cell)),
    )
    conn.commit()

    vol_cells = {
        "mfe_anchor_vol_units|scalp_30m|LONG": {
            "raw_signal_n": 216, "train_minus_test_median_diff": 5.0, "iqr": 2.0,
            "closure_classification": "ANSWERED_REGIME_DEPENDENT_BASELINE",
        }
    }
    result = w8v.compare_with_raw_bps_baseline(conn, vol_cells)
    conn.close()

    assert result["any_long_cell_stable_to_regime_dependent"] == ["mfe_anchor_vol_units|scalp_30m|LONG"]
    cmp = result["per_cell"]["mfe_anchor_vol_units|scalp_30m|LONG"]
    assert cmp["verdict_changed"] is True
    assert cmp["raw_bps_verdict"] == "ANSWERED_SUPPORTED_STABLE_BASELINE"
    assert cmp["vol_normalized_verdict"] == "ANSWERED_REGIME_DEPENDENT_BASELINE"


def test_compare_no_flag_when_both_stable():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE experiment_results (experiment_id TEXT, metric_name TEXT, metric_value TEXT)"
    )
    raw_cell = {
        "raw_signal_n": 216, "train_minus_test_median_diff": 0.5, "iqr": 10.0,
        "closure_classification": "ANSWERED_SUPPORTED_STABLE_BASELINE",
    }
    conn.execute(
        "INSERT INTO experiment_results VALUES (?,?,?)",
        (w8v.RAW_BPS_EXPERIMENT_ID, "cell_mfe_bps|scalp_30m|LONG", str(raw_cell)),
    )
    conn.commit()
    vol_cells = {
        "mfe_anchor_vol_units|scalp_30m|LONG": {
            "raw_signal_n": 216, "train_minus_test_median_diff": 0.1, "iqr": 3.0,
            "closure_classification": "ANSWERED_SUPPORTED_STABLE_BASELINE",
        }
    }
    result = w8v.compare_with_raw_bps_baseline(conn, vol_cells)
    conn.close()
    assert result["any_long_cell_stable_to_regime_dependent"] == []
    assert result["per_cell"]["mfe_anchor_vol_units|scalp_30m|LONG"]["verdict_changed"] is False


def test_compare_insufficient_sample_never_relabeled():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE experiment_results (experiment_id TEXT, metric_name TEXT, metric_value TEXT)"
    )
    raw_cell = {
        "raw_signal_n": 50, "train_minus_test_median_diff": None, "iqr": None,
        "closure_classification": "INSUFFICIENT_SAMPLE",
    }
    conn.execute(
        "INSERT INTO experiment_results VALUES (?,?,?)",
        (w8v.RAW_BPS_EXPERIMENT_ID, "cell_mfe_bps|scalp_30m|SHORT", str(raw_cell)),
    )
    conn.commit()
    vol_cells = {
        "mfe_anchor_vol_units|scalp_30m|SHORT": {
            "raw_signal_n": 50, "train_minus_test_median_diff": None, "iqr": None,
            "closure_classification": "INSUFFICIENT_SAMPLE",
        }
    }
    result = w8v.compare_with_raw_bps_baseline(conn, vol_cells)
    conn.close()
    cmp = result["per_cell"]["mfe_anchor_vol_units|scalp_30m|SHORT"]
    assert cmp["raw_bps_verdict"] == "INSUFFICIENT_SAMPLE"
    assert cmp["vol_normalized_verdict"] == "INSUFFICIENT_SAMPLE"
    assert cmp["verdict_changed"] is False
    # explicit contract: normalization is documented as NOT repairing sample size
    assert result["normalization_repairs_sample_size"] is False


def test_iqr_ratio_helper():
    assert w8v._iqr_ratio({"train_minus_test_median_diff": 2.0, "iqr": 4.0}) == 0.5
    assert w8v._iqr_ratio({"train_minus_test_median_diff": None, "iqr": 4.0}) is None
    assert w8v._iqr_ratio({"train_minus_test_median_diff": 2.0, "iqr": 0}) is None


# ---- real-data smoke test (writes only experiment_registry/experiment_results,
# via conftest.py's session-scoped isolation -- never the real file) ----

def test_compute_metrics_real_data_smoke():
    """Exercises the compute-only path (compute_metrics() +
    compare_with_raw_bps_baseline(), never freeze_and_record()'s write)
    against the real (isolated-copy) canonical data. compare_with_raw_bps_baseline
    reads E-W8-HOLD-BASELINE-001's ALREADY-STORED results back (read-only --
    no dependency on freshly re-running w8_hold_baseline's own
    freeze_and_record(), which is no longer safe to call against the current,
    grown population under its old experiment_id -- see
    test_ami_research_w8_hold_baseline.py::test_freeze_and_record_fails_closed_on_real_population_drift)."""
    import ami.warehouse.schema as schema_mod
    from ami.lifecycle.path_schema import init_path_schema

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)
        init_path_schema(conn)

        metrics = w8v.compute_metrics(conn)
        assert len(metrics["cell_order"]) == 16
        assert metrics["raw_signal_n_population"] > 0
        assert metrics["distinct_independent_cycle_n_population"] <= metrics["distinct_source_event_n_population"]
        assert metrics["distinct_source_event_n_population"] <= metrics["raw_signal_n_population"]

        for key in metrics["cell_order"]:
            assert metrics["cells"][key]["closure_classification"] in (
                "INSUFFICIENT_SAMPLE", "ANSWERED_SUPPORTED_STABLE_BASELINE", "ANSWERED_REGIME_DEPENDENT_BASELINE",
            )

        # [POST BATCH-SHORT-NOISY-V1-CANON-BACKFILL, flagged not interpreted -- NO_OUTCOME_ANALYSIS/
        # NO_TRAIN_TEST_CLAIM apply to that batch] the 54 newly-canonicalized SHORT_NOISY_BTC200K_
        # CONFIRMED_V1 signals are picked up automatically by this population query (direction=SHORT,
        # no setup_id filter). This test intentionally does NOT assert or interpret the resulting
        # classification (that would be an outcome/train-test claim) -- it only proves the machinery
        # still runs end-to-end. A separate, explicitly-approved re-run under a NEW experiment_id
        # would be required to responsibly interpret this population growth.
        for horizon in HORIZONS:
            for metric in w8v.METRICS:
                key = f"{metric}|{horizon}|SHORT"
                assert metrics["cells"][key]["sample_sufficiency"] in ("INSUFFICIENT_SAMPLE", "OK")
                assert metrics["cells"][key]["closure_classification"] in (
                    "INSUFFICIENT_SAMPLE", "ANSWERED_SUPPORTED_STABLE_BASELINE", "ANSWERED_REGIME_DEPENDENT_BASELINE",
                )

        cmp = w8v.compare_with_raw_bps_baseline(conn, metrics["cells"])
        assert len(cmp["per_cell"]) == 16
        assert isinstance(cmp["any_long_cell_stable_to_regime_dependent"], list)
        assert cmp["normalization_repairs_sample_size"] is False
    finally:
        conn.close()


def test_freeze_and_record_fails_closed_on_real_population_drift():
    """[BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY
    HARDENING, GOAL B] Same drift as E-W8-HOLD-BASELINE-001 (see
    test_ami_research_w8_hold_baseline.py's analogous test) --
    E-W8-VOL-NORMALIZED-BASELINE-001 was frozen before
    BATCH-SHORT-NOISY-V1-CANON-BACKFILL grew the SHORT population; calling
    freeze_and_record() today must fail closed rather than silently
    overwrite the frozen row."""
    import ami.warehouse.schema as schema_mod
    from ami.lifecycle.path_schema import init_path_schema
    from ami.warehouse.experiment_ledger import ImmutableExperimentConflict

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)
        init_path_schema(conn)

        pre_signal_n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_path_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_registry_row = conn.execute(
            "SELECT * FROM experiment_registry WHERE experiment_id=?", (w8v.EXPERIMENT_ID,)
        ).fetchone()
        assert pre_registry_row is not None

        with pytest.raises(ImmutableExperimentConflict, match="IMMUTABLE_EXPERIMENT_CONFLICT"):
            w8v.freeze_and_record(conn)

        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        post_registry_row = conn.execute(
            "SELECT * FROM experiment_registry WHERE experiment_id=?", (w8v.EXPERIMENT_ID,)
        ).fetchone()
        assert post_registry_row == pre_registry_row
    finally:
        conn.close()
