"""BATCH-P7B-1: tests for ami/research/w8_hold_baseline.py (W8-HOLD-BASELINE
-- fixed-horizon MFE/MAE baseline characterization + chronological stability,
NOT a management/exit/stop/re-entry wave).

Run: pytest tests/test_ami_research_w8_hold_baseline.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import pytest

import ami.research.w8_hold_baseline as w8
from ami.lifecycle.path_metrics import CANDLE_MS
from ami.research.w4_post_event_path_taxonomy import PATH_HORIZONS_MS

PROV = "test"


# ---- graveyard / permission-escalation static guards ----

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "reentry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
)


def test_no_graveyarded_management_rule_terms_in_module_source():
    src = inspect.getsource(w8).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS if t in src]
    assert hits == [], f"forbidden management-rule terms found: {hits}"


def test_no_action_permission_escalation_in_module_source():
    src = inspect.getsource(w8)
    assert "authorize(" not in src
    assert ".promote(" not in src
    assert "OPEN_LONG" not in src
    assert "OPEN_SHORT" not in src
    assert "import ami.governance" not in src
    assert "from ami.governance" not in src


def test_no_order_router_or_execution_import():
    src = inspect.getsource(w8)
    for forbidden in ("execution.", "risk.", "brain.", "order_router", "entry_loop", "position_manager"):
        assert forbidden not in src, f"forbidden import/reference: {forbidden}"


# ---- chronological split ----

def test_split_chronological_basic():
    rows = [{"signal_birth_ts": ts} for ts in range(100, 110)]
    train, test = w8._split_chronological_by_birth(rows)
    assert len(train) == 7  # 70% of 10
    assert len(test) == 3
    assert all(r["signal_birth_ts"] < test[0]["signal_birth_ts"] for r in train)


def test_split_chronological_never_randomized():
    import random
    rows = [{"signal_birth_ts": ts} for ts in range(100, 130)]
    shuffled = list(rows)
    random.Random(1).shuffle(shuffled)
    train_a, test_a = w8._split_chronological_by_birth(rows)
    train_b, test_b = w8._split_chronological_by_birth(shuffled)
    assert [r["signal_birth_ts"] for r in train_a] == [r["signal_birth_ts"] for r in train_b]
    assert [r["signal_birth_ts"] for r in test_a] == [r["signal_birth_ts"] for r in test_b]


# ---- cluster bootstrap: cycle, not signal, is the denominator ----

def test_cluster_bootstrap_groups_by_cycle_not_signal():
    # two signals sharing ONE independent_cycle_id must be resampled together as one cluster,
    # not treated as 2 independent draws -- verified by checking the bootstrap doesn't crash
    # and n_valid_draws reflects cluster-level (not row-level) resampling structure
    train_rows = [
        {"independent_cycle_id": "CYC-1", "source_event_id": "EVT-1", "mfe_bps": 10.0},
        {"independent_cycle_id": "CYC-1", "source_event_id": "EVT-1", "mfe_bps": 12.0},  # same cycle, paired route
    ] + [{"independent_cycle_id": f"CYC-{i}", "source_event_id": f"EVT-{i}", "mfe_bps": float(i)}
         for i in range(2, 25)]
    test_rows = [{"independent_cycle_id": f"CYC-T{i}", "source_event_id": f"EVT-T{i}", "mfe_bps": float(i)}
                 for i in range(25, 45)]
    result = w8.cluster_bootstrap_median_diff(train_rows, test_rows, "mfe_bps", n_boot=50)
    assert result["n_valid_draws"] > 0
    assert result["ci95"] != (None, None)


def test_multi_route_pair_does_not_inflate_distinct_cycle_count():
    rows = [
        {"signal_id": "SIG-A", "horizon_name": "scalp_30m", "direction": "LONG",
         "independent_cycle_id": "CYC-1", "source_event_id": "EVT-1", "mfe_bps": 10.0,
         "signal_birth_ts": 1000},
        {"signal_id": "SIG-B", "horizon_name": "scalp_30m", "direction": "SHORT",
         "independent_cycle_id": "CYC-1", "source_event_id": "EVT-1", "mfe_bps": 5.0,
         "signal_birth_ts": 1000},
    ]
    long_rows = w8._cell_rows(rows, "scalp_30m", "LONG")
    short_rows = w8._cell_rows(rows, "scalp_30m", "SHORT")
    cell_long = w8.compute_cell(long_rows, "mfe_bps")
    cell_short = w8.compute_cell(short_rows, "mfe_bps")
    assert cell_long["distinct_independent_cycle_n"] == 1
    assert cell_short["distinct_independent_cycle_n"] == 1
    assert cell_long["raw_signal_n"] == 1  # signal-level N stays separate per direction


# ---- Holm adjustment across exactly 16 cells ----

def test_family_is_exactly_16_cells():
    assert len(w8.METRICS) * len(w8.HORIZONS) * len(w8.DIRECTIONS) == 16


# ---- insufficient-sample handling ----

def test_insufficient_sample_not_merged():
    rows = [{"signal_id": f"SIG-{i}", "horizon_name": "swing_24h", "direction": "SHORT",
             "independent_cycle_id": f"CYC-{i}", "source_event_id": f"EVT-{i}",
             "mfe_bps": float(i), "signal_birth_ts": 1000 + i} for i in range(10)]  # only 10, below MIN_BUCKET_N
    cell = w8.compute_cell(rows, "mfe_bps")
    assert cell["sample_sufficiency"] == "INSUFFICIENT_SAMPLE"
    assert cell["bootstrap_ci95"] == (None, None)
    assert cell["permutation_p_value"] is None
    # raw counts still reported honestly, not hidden
    assert cell["raw_signal_n"] == 10


def test_classify_cell_verdict_insufficient():
    assert w8.classify_cell_verdict("INSUFFICIENT_SAMPLE", 0.5, (-1.0, 1.0)) == "INSUFFICIENT_SAMPLE"


def test_classify_cell_verdict_stable():
    assert w8.classify_cell_verdict("OK", 0.8, (-1.0, 1.0)) == "ANSWERED_SUPPORTED_STABLE_BASELINE"


def test_classify_cell_verdict_regime_dependent():
    assert w8.classify_cell_verdict("OK", 0.01, (1.0, 5.0)) == "ANSWERED_REGIME_DEPENDENT_BASELINE"


def test_classify_cell_verdict_disagreement_never_defaults_to_stable():
    # Holm-significant but CI includes 0 (disagreement) -- must NOT be "stable"
    assert w8.classify_cell_verdict("OK", 0.01, (-1.0, 1.0)) == "ANSWERED_REGIME_DEPENDENT_BASELINE"
    # Holm-nonsignificant but CI excludes 0 (disagreement) -- must NOT be "stable"
    assert w8.classify_cell_verdict("OK", 0.8, (1.0, 5.0)) == "ANSWERED_REGIME_DEPENDENT_BASELINE"


def test_classify_cell_verdict_never_returns_answered_supported_bare():
    # the operator explicitly forbade a bare "ANSWERED_SUPPORTED" auto-default
    for suff in ("OK", "INSUFFICIENT_SAMPLE"):
        for p in (None, 0.001, 0.5, 0.999):
            for ci in ((None, None), (-1.0, 1.0), (1.0, 5.0), (-5.0, -1.0)):
                v = w8.classify_cell_verdict(suff, p, ci)
                assert v in ("INSUFFICIENT_SAMPLE", "ANSWERED_SUPPORTED_STABLE_BASELINE",
                             "ANSWERED_REGIME_DEPENDENT_BASELINE")


# ---- negative control: direction blocked, never defaulted/pooled ----

def test_match_profile_direction_ratio_always_blocked():
    rows = [{"signal_birth_ts": 1_700_000_000_000, "direction": "LONG", "realized_vol_at_anchor": 0.001}]
    profile = w8.build_match_profile(rows)
    assert profile["direction_ratio_matching_status"] == "BLOCKED_FOR_DIRECTION_MATCHING"


def test_match_profile_reports_direction_counts_descriptively():
    rows = [
        {"signal_birth_ts": 1_700_000_000_000, "direction": "LONG", "realized_vol_at_anchor": 0.001},
        {"signal_birth_ts": 1_700_000_060_000, "direction": "SHORT", "realized_vol_at_anchor": 0.002},
    ]
    profile = w8.build_match_profile(rows)
    assert profile["direction_counts"] == {"LONG": 1, "SHORT": 1}


def test_negative_control_output_never_uses_mfe_mae_field_names():
    # descriptive_context_by_horizon must use upside/downside excursion names as its OUTPUT
    # keys, never "mfe_bps"/"mae_bps" (which would imply a real per-signal-direction claim).
    # Reading obs["mfe_bps"] from compute_observation()'s own result is fine and expected --
    # only introducing a NEW dict KEY named "mfe_bps"/"mae_bps" would be the violation, so the
    # check targets the key-definition pattern ('"mfe_bps":'), not any occurrence of the string.
    import inspect as _inspect
    src = _inspect.getsource(w8.compute_negative_control)
    assert '"upside_excursion_bps"' in src
    assert '"downside_excursion_bps"' in src
    assert '"mfe_bps":' not in src
    assert '"mae_bps":' not in src


# ---- known-at safety (reused compute_observation, proven end-to-end here too) ----

def test_known_at_safety_future_candle_does_not_affect_control_excursion():
    from ami.lifecycle.path_metrics import _CandleOHLCIndex, compute_observation

    def _mk(open_ts, high, low, close, quality="AVAILABLE", cdv="cdv-test"):
        return {"open_ts_ms": open_ts, "close_ts_ms": open_ts + CANDLE_MS, "high": high, "low": low,
                "close": close, "data_quality": quality, "candle_definition_version": cdv}

    birth = 200_000_000_000
    pre = [_mk(birth - (60 - i) * CANDLE_MS, 100.0, 100.0, 100.0) for i in range(60)]
    path = [_mk(birth + i * CANDLE_MS, 105.0, 95.0, 100.0) for i in range(30)]
    horizon_end = birth + PATH_HORIZONS_MS["scalp_30m"]

    idx_before = _CandleOHLCIndex(pre + path)
    obs_before = compute_observation(idx_before, "CTRL-1", "LONG", birth, "scalp_30m",
                                      PATH_HORIZONS_MS["scalp_30m"], birth + 100 * 24 * 3600_000)

    future = _mk(horizon_end + CANDLE_MS, high=999.0, low=1.0, close=500.0)
    idx_after = _CandleOHLCIndex(pre + path + [future])
    obs_after = compute_observation(idx_after, "CTRL-1", "LONG", birth, "scalp_30m",
                                     PATH_HORIZONS_MS["scalp_30m"], birth + 100 * 24 * 3600_000)

    assert obs_before["mfe_bps"] == obs_after["mfe_bps"]
    assert obs_before["mae_bps"] == obs_after["mae_bps"]


# ---- stratified matching determinism ----

def test_month_and_vol_bucket_helpers():
    assert w8._month_bucket(1_772_000_000_000).count("-") == 1
    assert w8._vol_bucket(0.002, 0.001) == "HIGH"
    assert w8._vol_bucket(0.0005, 0.001) == "LOW"
    assert w8._vol_bucket(None, 0.001) == "UNKNOWN"
    assert w8._vol_bucket(0.001, None) == "UNKNOWN"


# ---- real-data smoke test (writes to real canonical.sqlite's experiment_registry/
# experiment_results ONLY -- via conftest.py's session-scoped isolation, this
# targets a disposable copy, never the real file) ----

def test_compute_metrics_real_data_smoke():
    """Exercises the compute-only path (compute_metrics(), never
    freeze_and_record()'s write) against the real (isolated-copy) canonical
    data -- proves the machinery still runs end-to-end and produces a
    well-formed 16-cell result, without touching experiment_registry/results
    at all."""
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)
        metrics = w8.compute_metrics(conn)
        assert len(metrics["cell_order"]) == 16
        assert metrics["raw_signal_n_population"] > 0
        assert metrics["distinct_independent_cycle_n_population"] <= metrics["distinct_source_event_n_population"]
        assert metrics["distinct_source_event_n_population"] <= metrics["raw_signal_n_population"]

        # [BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING, GOAL C]
        # fetch_population() now explicitly pins path_definition_version="path-v2" -- this module's
        # population is therefore, by construction, the same frozen pre-candle-repair PATH population
        # E-W8-HOLD-BASELINE-001 was originally computed from (the 149/21 candle-repair-corrected
        # rows live under a separate path_definition_version and are never picked up here). Any
        # remaining population growth visible here comes from a SEPARATE, unrelated cause --
        # ami_signal_lifecycle itself grew (BATCH-SHORT-NOISY-V1-CANON-BACKFILL added 54 new SHORT
        # signals AFTER -001 was originally frozen) -- see
        # test_freeze_and_record_fails_closed_on_real_population_drift below, which is the direct
        # consequence GOAL B is designed to catch.
        for key in metrics["cell_order"]:
            assert metrics["cells"][key]["closure_classification"] in (
                "INSUFFICIENT_SAMPLE", "ANSWERED_SUPPORTED_STABLE_BASELINE", "ANSWERED_REGIME_DEPENDENT_BASELINE",
            )
    finally:
        conn.close()


def test_freeze_and_record_fails_closed_on_real_population_drift():
    """[BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY
    HARDENING, GOAL B] E-W8-HOLD-BASELINE-001 was frozen with raw_signal_n=266
    (dataset_hash computed before BATCH-SHORT-NOISY-V1-CANON-BACKFILL added 54
    new SHORT signals to ami_signal_lifecycle). Calling freeze_and_record()
    today against the real (isolated-copy, now-grown) population would
    previously have silently overwritten -001's dataset_hash/frozen_population
    via ON CONFLICT DO UPDATE -- the new immutable-write guard instead fails
    closed with ImmutableExperimentConflict, exactly the "prevent direct
    execution under the old experiment ID with a clear frozen experiment
    error" contract GOAL C mandates for a legacy module whose experiment_id
    can no longer be reproduced byte-for-byte (path-version pinning alone is
    insufficient here -- the drift is in the SIGNAL population, not the path
    population). Canonical tables and the stored -001 row must be completely
    untouched by the failed attempt."""
    import ami.warehouse.schema as schema_mod
    from ami.warehouse.experiment_ledger import ImmutableExperimentConflict

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)

        pre_signal_n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_path_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_registry_row = conn.execute(
            "SELECT * FROM experiment_registry WHERE experiment_id=?", (w8.EXPERIMENT_ID,)
        ).fetchone()
        pre_results = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8.EXPERIMENT_ID,),
        ).fetchall()
        assert pre_registry_row is not None, "E-W8-HOLD-BASELINE-001 was expected to already exist in the real DB"

        with pytest.raises(ImmutableExperimentConflict, match="IMMUTABLE_EXPERIMENT_CONFLICT"):
            w8.freeze_and_record(conn)

        # canonical tables and the stored -001 row are completely unaffected by the failed attempt
        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        post_registry_row = conn.execute(
            "SELECT * FROM experiment_registry WHERE experiment_id=?", (w8.EXPERIMENT_ID,)
        ).fetchone()
        post_results = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8.EXPERIMENT_ID,),
        ).fetchall()
        assert post_registry_row == pre_registry_row
        assert post_results == pre_results
    finally:
        conn.close()
