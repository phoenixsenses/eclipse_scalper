"""BATCH-P7B-1 (W8-LONG-TIMING-STRUCTURE-001): tests for
ami/research/w8_long_timing_structure.py -- WHEN (not whether) LONG's MFE/MAE
occur within the fixed horizon, cycle-grouped chronological stability tested.

Run: pytest tests/test_ami_research_w8_long_timing_structure.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import ami.research.w8_long_timing_structure as w8lt
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
)
_FORBIDDEN_SELECTION_IDENTIFIERS = ("win_rate", "threshold_sweep", "conviction_score")


def test_no_graveyarded_management_terms_in_module_source():
    src = inspect.getsource(w8lt).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS if t in src]
    assert hits == [], f"forbidden management-rule terms found: {hits}"
    sel_hits = [t for t in _FORBIDDEN_SELECTION_IDENTIFIERS if t in src]
    assert sel_hits == [], f"forbidden outcome/selection identifiers found: {sel_hits}"


def test_no_action_permission_escalation_in_module_source():
    src = inspect.getsource(w8lt)
    assert "authorize(" not in src
    assert ".promote(" not in src
    assert "OPEN_LONG" not in src
    assert "OPEN_SHORT" not in src
    assert "import ami.governance" not in src


def test_no_order_router_or_execution_import():
    src = inspect.getsource(w8lt)
    for forbidden in ("execution.", "risk.", "brain.", "order_router", "entry_loop", "position_manager"):
        assert forbidden not in src, f"forbidden import/reference: {forbidden}"


def test_direction_is_long_only_no_short_pooling():
    assert w8lt.DIRECTION == "LONG"
    src = inspect.getsource(w8lt.fetch_population)
    assert '"SHORT"' not in src


def test_reuses_split_machinery_verbatim_from_short_expanded_module():
    import ami.research.w8_short_expanded_baseline as w8se
    assert w8lt.compute_global_cycle_split is w8se.compute_global_cycle_split
    assert w8lt.split_rows_by_cycle_keys is w8se.split_rows_by_cycle_keys
    assert w8lt.assert_zero_cycle_straddling is w8se.assert_zero_cycle_straddling
    assert w8lt._cycle_key is w8se._cycle_key


def test_primary_family_is_exactly_8_cells():
    assert len(w8lt.TIMING_METRICS) == 2
    assert len(w8lt.TIMING_METRICS) * len(w8lt.HORIZONS) == 8


# ---- descriptive helpers (synthetic, no DB needed) ----

def _row(signal_id, cycle_id, birth_ts, horizon, mfe_ms, mae_ms, intrabar):
    return {
        "signal_id": signal_id, "independent_cycle_id": cycle_id, "source_event_id": f"EVT-{signal_id}",
        "signal_birth_ts": birth_ts, "horizon_name": horizon, "direction": "LONG",
        "time_to_mfe_ms": mfe_ms, "time_to_mae_ms": mae_ms, "intrabar_order_status": intrabar,
        "time_to_mfe_fraction_of_horizon": None, "time_to_mae_fraction_of_horizon": None,
        "timing_delta_ms": (mfe_ms - mae_ms) if mfe_ms is not None and mae_ms is not None else None,
    }


def test_zero_at_reference_detected_when_both_times_are_zero():
    rows = [
        _row("S1", "CYC-1", 0, "scalp_30m", 0, 0, "SAME_CANDLE_UNKNOWN"),
        _row("S2", "CYC-2", 1000, "scalp_30m", 500, 200, "MAE_FIRST"),
    ]
    desc = w8lt.compute_horizon_descriptive(rows, rows[:1], rows[1:])
    assert desc["zero_at_reference_n"] == 1
    assert desc["zero_at_reference_rate"] == 0.5


def test_intrabar_status_counts_and_rates_sum_correctly():
    rows = [
        _row("S1", "CYC-1", 0, "scalp_30m", 100, 200, "MFE_FIRST"),
        _row("S2", "CYC-2", 1000, "scalp_30m", 300, 100, "MAE_FIRST"),
        _row("S3", "CYC-3", 2000, "scalp_30m", 50, 50, "SAME_CANDLE_UNKNOWN"),
    ]
    desc = w8lt.compute_horizon_descriptive(rows, rows, [])
    assert desc["intrabar_order_status_counts"] == {"MFE_FIRST": 1, "MAE_FIRST": 1, "SAME_CANDLE_UNKNOWN": 1}
    assert abs(sum(desc["intrabar_order_status_rates"].values()) - 1.0) < 1e-5


def test_sufficiency_is_cycle_count_based_per_horizon():
    # 25 signals, only 2 cycles -- must be INSUFFICIENT_SAMPLE regardless of signal-level N
    rows = []
    for i in range(25):
        cyc = "CYC-A" if i < 20 else "CYC-B"
        rows.append(_row(f"S{i}", cyc, i * 1000, "scalp_30m", float(i), float(i) / 2, "MFE_FIRST"))
    split = w8lt.compute_global_cycle_split(rows)
    cell = w8lt.compute_cell(rows, "time_to_mfe_ms", split["train_cycle_keys"], split["test_cycle_keys"])
    assert cell["distinct_independent_cycle_n"] == 2
    assert cell["sample_sufficiency"] == "INSUFFICIENT_SAMPLE"


def test_sufficient_when_enough_independent_cycles():
    rows = [_row(f"S{i}", f"CYC-{i}", i * 1000, "scalp_30m", float(i % 11), float(i % 7), "MFE_FIRST")
            for i in range(100)]
    split = w8lt.compute_global_cycle_split(rows)
    assert split["train_cycle_n"] >= MIN_BUCKET_N
    assert split["test_cycle_n"] >= MIN_BUCKET_N
    cell = w8lt.compute_cell(rows, "time_to_mfe_ms", split["train_cycle_keys"], split["test_cycle_keys"])
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

        r1 = w8lt.freeze_and_record(conn)
        assert len(r1["cell_order"]) == 8
        assert r1["raw_signal_n_population"] > 0
        assert r1["distinct_independent_cycle_n_population"] <= r1["distinct_source_event_n_population"]
        assert r1["distinct_source_event_n_population"] <= r1["raw_signal_n_population"]

        for key in r1["cell_order"]:
            assert r1["cells"][key]["cycle_straddling_violations"] == 0
            assert r1["cells"][key]["closure_classification"] in (
                "INSUFFICIENT_SAMPLE", "ANSWERED_SUPPORTED_STABLE_BASELINE", "ANSWERED_REGIME_DEPENDENT_BASELINE",
            )

        assert r1["family_verdict"] in (
            "LONG_TIMING_STRUCTURE_STABLE", "LONG_TIMING_STRUCTURE_REGIME_DEPENDENT",
            "MIXED_BY_HORIZON", "INSUFFICIENT_SAMPLE",
        )

        # canonical lifecycle/path/provenance tables must be completely unaffected
        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0] == pre_provenance_n

        n_results_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8lt.EXPERIMENT_ID,)
        ).fetchone()[0]
        r2 = w8lt.freeze_and_record(conn)
        n_results_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8lt.EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_results_1 == n_results_2
        assert r2["family_verdict"] == r1["family_verdict"]
        assert r2["raw_signal_n_population"] == r1["raw_signal_n_population"]
    finally:
        conn.close()
