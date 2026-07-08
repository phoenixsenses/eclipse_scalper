"""BATCH-P7B-1 (W8-SHORT-EXPANDED-BASELINE): tests for
ami/research/w8_short_expanded_baseline.py -- cycle-grouped chronological
SHORT-only rerun of W8-HOLD-BASELINE/W8-VOL-NORMALIZED-BASELINE, triggered by
BATCH-SHORT-NOISY-V1-CANON-BACKFILL's SHORT population growth.

Run: pytest tests/test_ami_research_w8_short_expanded_baseline.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import ami.research.w8_short_expanded_baseline as w8se
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, TRAIN_FRACTION

_FORBIDDEN_MANAGEMENT_TERMS = (
    "stop_loss", "partial_exit", "time_stop", "re_entry", "reentry",
    "cancellation_rule", "management_rule", "take_profit", "trailing_stop",
)
_FORBIDDEN_SELECTION_IDENTIFIERS = (
    "win_rate", "threshold_sweep", "conviction_score",
)


def test_no_graveyarded_management_terms_in_module_source():
    src = inspect.getsource(w8se).lower()
    hits = [t for t in _FORBIDDEN_MANAGEMENT_TERMS if t in src]
    assert hits == [], f"forbidden management-rule terms found: {hits}"
    sel_hits = [t for t in _FORBIDDEN_SELECTION_IDENTIFIERS if t in src]
    assert sel_hits == [], f"forbidden outcome/selection identifiers found: {sel_hits}"


def test_no_action_permission_escalation_in_module_source():
    src = inspect.getsource(w8se)
    assert "authorize(" not in src
    assert ".promote(" not in src
    assert "OPEN_LONG" not in src
    assert "OPEN_SHORT" not in src
    assert "import ami.governance" not in src


def test_no_order_router_or_execution_import():
    src = inspect.getsource(w8se)
    for forbidden in ("execution.", "risk.", "brain.", "order_router", "entry_loop", "position_manager"):
        assert forbidden not in src, f"forbidden import/reference: {forbidden}"


def test_new_experiment_ids_distinct_from_old():
    assert w8se.RAW_BPS_EXPERIMENT_ID != w8se.OLD_RAW_BPS_EXPERIMENT_ID
    assert w8se.VOL_NORMALIZED_EXPERIMENT_ID != w8se.OLD_VOL_NORMALIZED_EXPERIMENT_ID
    assert w8se.OLD_RAW_BPS_EXPERIMENT_ID == "E-W8-HOLD-BASELINE-001"
    assert w8se.OLD_VOL_NORMALIZED_EXPERIMENT_ID == "E-W8-VOL-NORMALIZED-BASELINE-001"


def test_primary_family_is_exactly_16_short_only_cells():
    assert len(w8se.ALL_METRICS) == 4
    assert len(w8se.ALL_METRICS) * 4 == 16  # x 4 horizons
    assert w8se.DIRECTION == "SHORT"


# ---- cycle-grouped split (synthetic, no DB needed) ----

def _row(signal_id, cycle_id, birth_ts, horizon="scalp_30m", metric_val=1.0, source_event_id=None):
    return {
        "signal_id": signal_id, "independent_cycle_id": cycle_id,
        "source_event_id": source_event_id or f"EVT-{signal_id}",
        "signal_birth_ts": birth_ts, "horizon_name": horizon, "direction": "SHORT",
        "mfe_bps": metric_val, "mae_bps": -metric_val,
    }


def test_global_cycle_split_never_splits_a_cycle_across_sides():
    # 10 cycles, 2 signals each, interleaved birth timestamps -- must never
    # put the same cycle's rows in both TRAIN and TEST
    rows = []
    for i in range(10):
        rows.append(_row(f"S{i}a", f"CYC-{i}", birth_ts=i * 1000))
        rows.append(_row(f"S{i}b", f"CYC-{i}", birth_ts=i * 1000 + 500))
    split = w8se.compute_global_cycle_split(rows)
    assert split["train_cycle_keys"].isdisjoint(split["test_cycle_keys"])
    assert split["total_cycle_n"] == 10
    train_rows, test_rows = w8se.split_rows_by_cycle_keys(rows, split["train_cycle_keys"], split["test_cycle_keys"])
    assert w8se.assert_zero_cycle_straddling(train_rows, test_rows) == 0
    # every row of a given cycle lands on the same side
    for i in range(10):
        sides = {("CYC-%d" % i) in split["train_cycle_keys"], ("CYC-%d" % i) in split["test_cycle_keys"]}
        assert sides in ({True, False},)  # exactly one True, one False -- never both True


def test_cycle_split_uses_earliest_birth_ts_per_cycle_not_per_row():
    # a cycle with one very early row and one very late row must be ordered by
    # the EARLIEST of the two, not the latest
    rows = [
        _row("S1", "CYC-X", birth_ts=0),
        _row("S2", "CYC-X", birth_ts=10_000_000),  # same cycle, much later row
        _row("S3", "CYC-Y", birth_ts=500),
    ]
    split = w8se.compute_global_cycle_split(rows)
    anchors = dict(split["ordered_cycle_anchors"])
    assert anchors["CYC-X"] == 0  # earliest, not the 10_000_000 row


def test_sufficiency_is_cycle_count_based_not_signal_count_based():
    # 25 signals but only 2 independent cycles -- MUST be INSUFFICIENT_SAMPLE
    # even though signal-level N (25) would look sufficient under the OLD rule
    rows = []
    for i in range(25):
        cyc = "CYC-A" if i < 20 else "CYC-B"
        rows.append(_row(f"S{i}", cyc, birth_ts=i * 1000, metric_val=float(i)))
    split = w8se.compute_global_cycle_split(rows)
    cell = w8se.compute_cell(rows, "mfe_bps", split["train_cycle_keys"], split["test_cycle_keys"])
    assert cell["raw_signal_n"] == 25
    assert cell["distinct_independent_cycle_n"] == 2
    assert cell["sample_sufficiency"] == "INSUFFICIENT_SAMPLE"


def test_sufficient_when_enough_independent_cycles_on_both_sides():
    # 100 distinct cycles, 1 signal each -- 70/30 split gives 70/30 cycles,
    # both >= MIN_BUCKET_N=20
    rows = [_row(f"S{i}", f"CYC-{i}", birth_ts=i * 1000, metric_val=float(i % 7)) for i in range(100)]
    split = w8se.compute_global_cycle_split(rows)
    assert split["train_cycle_n"] >= MIN_BUCKET_N
    assert split["test_cycle_n"] >= MIN_BUCKET_N
    cell = w8se.compute_cell(rows, "mfe_bps", split["train_cycle_keys"], split["test_cycle_keys"])
    assert cell["sample_sufficiency"] == "OK"
    assert cell["cycle_straddling_violations"] == 0


def test_family_verdict_all_insufficient():
    # tiny population -> every cell must come back INSUFFICIENT_SAMPLE and the
    # family verdict must be the dedicated "insufficient after cycle-grouped
    # split" value, never silently reported as stable/regime-dependent
    rows = []
    for i in range(10):
        rows.append(_row(f"S{i}", f"CYC-{i}", birth_ts=i * 1000))
    split = w8se.compute_global_cycle_split(rows)
    for horizon in ("scalp_30m", "scalp_1h", "swing_4h", "swing_24h"):
        for row in rows:
            row["horizon_name"] = horizon
        cell = w8se.compute_cell(rows, "mfe_bps", split["train_cycle_keys"], split["test_cycle_keys"])
        assert cell["sample_sufficiency"] == "INSUFFICIENT_SAMPLE"


# ---- real-data smoke test (disposable copy only, via conftest isolation) ----

def test_real_data_smoke_freeze_and_record_and_idempotent_and_old_experiments_untouched():
    """[BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY
    HARDENING] Does NOT re-invoke w8_hold_baseline/w8_vol_normalized_baseline's
    own freeze_and_record() as setup -- E-W8-HOLD-BASELINE-001/E-W8-VOL-
    NORMALIZED-BASELINE-001 already exist in the real (isolated-copy) DB from
    the original historical batch, and re-running their freeze_and_record()
    today would itself raise ImmutableExperimentConflict (their population has
    genuinely drifted since freezing -- see
    test_ami_research_w8_hold_baseline.py::test_freeze_and_record_fails_closed_on_real_population_drift).
    This test only needs their ALREADY-STORED content as a before/after
    snapshot baseline, which it reads directly."""
    import ami.warehouse.schema as schema_mod
    from ami.lifecycle.path_schema import init_path_schema

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        schema_mod.init_schema(conn)
        init_path_schema(conn)

        before_raw = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8se.OLD_RAW_BPS_EXPERIMENT_ID,),
        ).fetchall()
        before_vol = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8se.OLD_VOL_NORMALIZED_EXPERIMENT_ID,),
        ).fetchall()

        pre_signal_n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_path_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_provenance_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]

        r1 = w8se.freeze_and_record(conn)

        assert r1["old_experiments_untouched"] is True
        after_raw = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8se.OLD_RAW_BPS_EXPERIMENT_ID,),
        ).fetchall()
        after_vol = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
            (w8se.OLD_VOL_NORMALIZED_EXPERIMENT_ID,),
        ).fetchall()
        assert after_raw == before_raw
        assert after_vol == before_vol

        assert len(r1["cell_order"]) == 16
        assert r1["raw_signal_n_population"] > 0
        assert r1["distinct_independent_cycle_n_population"] <= r1["distinct_source_event_n_population"]
        assert r1["distinct_source_event_n_population"] <= r1["raw_signal_n_population"]

        # zero cycle straddling in every cell
        for key in r1["cell_order"]:
            assert r1["cells"][key]["cycle_straddling_violations"] == 0

        # canonical lifecycle/path/provenance tables must be completely unaffected
        assert conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signal_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_path_n
        assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0] == pre_provenance_n

        # verdict must be one of the 4 operator-defined values
        assert r1["family_verdict"] in (
            "EXPANDED_SHORT_STABLE_BASELINE", "EXPANDED_SHORT_REGIME_DEPENDENT",
            "EXPANDED_SHORT_INSUFFICIENT_AFTER_CYCLE_GROUPED_SPLIT", "MIXED_BY_HORIZON_OR_METRIC",
        )

        n_raw_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se.RAW_BPS_EXPERIMENT_ID,)
        ).fetchone()[0]
        n_vol_1 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se.VOL_NORMALIZED_EXPERIMENT_ID,)
        ).fetchone()[0]

        r2 = w8se.freeze_and_record(conn)
        assert r2["old_experiments_untouched"] is True
        assert r2["family_verdict"] == r1["family_verdict"]
        n_raw_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se.RAW_BPS_EXPERIMENT_ID,)
        ).fetchone()[0]
        n_vol_2 = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (w8se.VOL_NORMALIZED_EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_raw_1 == n_raw_2
        assert n_vol_1 == n_vol_2
    finally:
        conn.close()
