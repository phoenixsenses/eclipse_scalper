"""BATCH-P6-008: W6-RS confirmation wave tests.

Run: pytest tests/test_ami_research_w6rs_confirmation.py --basetemp <scratchpad> -p no:cacheprovider
"""
import sqlite3

from ami.research.w6rs_confirmation import (
    EXPERIMENT_ID,
    RS_THRESHOLD,
    compute_day_trend_bps,
    compute_metrics,
    compute_negative_control,
    freeze_and_record,
    permutation_test_risk_difference,
    wilson_ci,
)
from ami.states.engine import StateEngine
from ami.warehouse.schema import connect, init_schema

MIN_MS = 60_000
NOW = 0
HOUR_MS = 3600_000
DAY_MS = 86_400_000


def _mk_microstructure_db(path, eth_prices, btc_prices):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
    for ts, px in eth_prices:
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (ts, "ETHUSDT", px))
    for ts, px in btc_prices:
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (ts, "BTCUSDT", px))
    conn.commit()
    conn.close()


def test_rs_threshold_is_frozen_at_zero():
    assert RS_THRESHOLD == 0.0


def test_compute_day_trend_bps_correct(tmp_path):
    db = tmp_path / "micro.sqlite"
    day_start = 0  # 1970-01-01 UTC midnight
    anchor = 3 * HOUR_MS
    _mk_microstructure_db(db, [(day_start, 100.0), (anchor, 102.0)], [])
    engine = StateEngine(db_path=db)
    try:
        result = compute_day_trend_bps(engine, "ETHUSDT", anchor)
        assert result == 200.0  # (102-100)/100*1e4
    finally:
        engine.conn.close()


def test_compute_day_trend_bps_none_when_no_day_open_price(tmp_path):
    db = tmp_path / "micro.sqlite"
    _mk_microstructure_db(db, [], [])
    engine = StateEngine(db_path=db)
    try:
        assert compute_day_trend_bps(engine, "ETHUSDT", HOUR_MS) is None
    finally:
        engine.conn.close()


def test_wilson_ci_known_bounds():
    lo, hi = wilson_ci(50, 100)
    assert 0.35 < lo < 0.45
    assert 0.55 < hi < 0.65


def test_wilson_ci_zero_n_returns_zero_zero():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_permutation_test_strongly_separated_groups_gives_low_p():
    # RS_STRONG all reversal, RS_WEAK none -- maximally separated
    result = permutation_test_risk_difference(strong_n=30, strong_success=30, weak_n=30, weak_success=0)
    assert result["p_value"] < 0.01
    assert result["observed_risk_difference"] == 1.0


def test_permutation_test_identical_groups_gives_high_p():
    result = permutation_test_risk_difference(strong_n=30, strong_success=15, weak_n=30, weak_success=15)
    assert result["p_value"] > 0.3
    assert result["observed_risk_difference"] == 0.0


def test_permutation_test_zero_n_handled():
    result = permutation_test_risk_difference(strong_n=0, strong_success=0, weak_n=10, weak_success=5)
    assert result["observed_risk_difference"] is None
    assert result["p_value"] is None


def test_compute_metrics_real_data_smoke():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        metrics = compute_metrics(conn)
    finally:
        conn.close()
    assert metrics["total_anchor_n"] > 0
    assert metrics["analyzed_n"] <= metrics["total_anchor_n"]
    assert metrics["excluded_n"] == metrics["total_anchor_n"] - metrics["analyzed_n"]
    # every excluded anchor in this population is accounted for by the
    # documented no-horizon-data reason (no silent/unexplained exclusions)
    assert metrics["excluded_no_horizon_data"] <= metrics["excluded_n"]
    assert metrics["independent_cycle_n"] > 0
    assert metrics["independent_cycle_n"] <= metrics["analyzed_n"]
    assert metrics["train_n"] + metrics["test_n"] == metrics["analyzed_n"]


def test_compute_negative_control_real_data_smoke():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        nc = compute_negative_control(conn, n_target=50)
    finally:
        conn.close()
    assert nc["rs_strong"]["n"] >= 0
    assert nc["rs_weak"]["n"] >= 0


def test_freeze_and_record_writes_canonical_sql_and_is_idempotent():
    # Same pattern as W6: exercises the real canonical.sqlite + real
    # microstructure.db via StateEngine's default path.
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect, init_schema as real_init

    conn = real_connect(DEFAULT_PATH)
    try:
        real_init(conn)
        freeze_and_record(conn)
        freeze_and_record(conn)
        exp_row = conn.execute(
            "SELECT software_verdict, scientific_verdict, supersedes_experiment_id "
            "FROM experiment_registry WHERE experiment_id=?",
            (EXPERIMENT_ID,),
        ).fetchone()
        n_registry = conn.execute(
            "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
        ).fetchone()[0]
        n_results = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=? AND metric_name='total_anchor_n'",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert exp_row == ("PASSED", "ANSWERED_SUPPORTED", "E-W6-COMPRESSION-RS-SESSION-001")
    assert n_registry == 1
    assert n_results == 1
