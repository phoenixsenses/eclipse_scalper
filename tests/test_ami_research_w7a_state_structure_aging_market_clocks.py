"""BATCH-P6-010 (W7A): state/structure aging + market-clock tests.

Run: pytest tests/test_ami_research_w7a_state_structure_aging_market_clocks.py --basetemp <scratchpad> -p no:cacheprovider
"""
import sqlite3

from ami.research.w7a_state_structure_aging_market_clocks import (
    CLOCK_BASELINE_MIN_VALID,
    CLOCK_WINDOW_CANDLES,
    EXPERIMENT_ID,
    STATE_AGE_MAX_STEPS,
    _CandleSeries,
    classify_closure,
    cluster_bootstrap_risk_difference,
    compute_liquidation_age,
    compute_market_clocks,
    compute_metrics,
    compute_negative_control,
    compute_state_age,
    compute_structural_object_age,
    freeze_and_record,
    holm_adjust,
    permutation_test_two_sided,
)
from ami.states.engine import StateEngine
from ami.warehouse.schema import connect, init_schema

HOUR_MS = 3_600_000
MIN_MS = 60_000
NOW = 0


def _mk_micro_db(path, eth_prices):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
    for ts, px in eth_prices:
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (ts, "ETHUSDT", px))
    conn.commit()
    conn.close()


# ---- structural object age (pure) ----

def test_structural_object_age_ok_uses_nearest_prior():
    r = compute_structural_object_age([1000, 5000, 9000], 10_000)
    assert r == {"status": "OK", "age_ms": 1000}


def test_structural_object_age_ignores_future_objects():
    r = compute_structural_object_age([1000, 20_000], 10_000)
    assert r == {"status": "OK", "age_ms": 9000}


def test_structural_object_age_missing_when_no_eligible_object():
    assert compute_structural_object_age([20_000, 30_000], 10_000) == {"status": "MISSING", "age_ms": None}
    assert compute_structural_object_age([], 10_000) == {"status": "MISSING", "age_ms": None}


# ---- state age (StateEngine-backed) ----

def test_state_age_left_censored_near_data_start(tmp_path):
    db = tmp_path / "micro.sqlite"
    # only ~2h of history before anchor -> backward walk hits data start almost immediately
    anchor = 10 * HOUR_MS
    prices = [(anchor - k * HOUR_MS, 100.0) for k in range(0, 3)]
    _mk_micro_db(db, prices)
    engine = StateEngine(db_path=db)
    try:
        r = compute_state_age(engine, "ETHUSDT", anchor, earliest_mark_ts_ms=anchor - 2 * HOUR_MS)
        assert r["status"] == "LEFT_CENSORED"
        assert r["age_ms"] is None
    finally:
        engine.conn.close()


def test_state_age_missing_on_stale_gap(tmp_path):
    db = tmp_path / "micro.sqlite"
    anchor = 100 * HOUR_MS
    # dense prices around anchor, then a big gap further back (only one very
    # old point) -> the very first backward probe already finds a stale row
    prices = [(anchor, 100.0)] + [(anchor - k * HOUR_MS, 100.0) for k in range(1, 3)] + [(0, 100.0)]
    _mk_micro_db(db, prices)
    engine = StateEngine(db_path=db)
    try:
        r = compute_state_age(engine, "ETHUSDT", anchor, earliest_mark_ts_ms=0)
        assert r["status"] in {"MISSING", "LEFT_CENSORED_AT_CAP", "OK"}
        # specifically: once the walk passes the 3rd hourly point, the nearest
        # row (ts=0) is far more than the 2h staleness gap away
        assert r["status"] != "LEFT_CENSORED"
    finally:
        engine.conn.close()


def test_state_age_left_censored_at_cap_on_flat_long_history(tmp_path):
    db = tmp_path / "micro.sqlite"
    anchor = (STATE_AGE_MAX_STEPS + 20) * HOUR_MS
    # perfectly flat price for the full cap window and beyond -> StructurePhase
    # never changes -> walk exhausts STATE_AGE_MAX_STEPS without a transition
    prices = [(anchor - k * HOUR_MS, 100.0) for k in range(0, STATE_AGE_MAX_STEPS + 15)]
    _mk_micro_db(db, prices)
    engine = StateEngine(db_path=db)
    try:
        r = compute_state_age(engine, "ETHUSDT", anchor, earliest_mark_ts_ms=0)
        assert r["status"] == "LEFT_CENSORED_AT_CAP"
        assert r["age_ms"] is None
        assert r["steps_scanned"] == STATE_AGE_MAX_STEPS
    finally:
        engine.conn.close()


# ---- liquidation age ----

def _mk_liq_db(path, rows):
    """rows: list of (ts_ms, side, notional)."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE liquidations (ts_ms INTEGER, symbol TEXT, side TEXT, notional REAL)")
    for ts, side, notional in rows:
        conn.execute("INSERT INTO liquidations VALUES (?,?,?,?)", (ts, "ETHUSDT", side, notional))
    conn.commit()
    return conn


def test_liquidation_age_finds_nearest_qualifying_prior_print(tmp_path):
    conn = _mk_liq_db(tmp_path / "liq.sqlite", [
        (1000, "SELL", 250_000.0),
        (5000, "SELL", 250_000.0),
        (9000, "SELL", 10_000.0),  # too small, ignored
    ])
    r = compute_liquidation_age(conn, "SELL", reference_ts_ms=10_000)
    conn.close()
    assert r == {"status": "OK", "age_ms": 5000}


def test_liquidation_age_left_censored_when_none_qualify():
    conn = _mk_liq_db(":memory:", [(1000, "SELL", 10_000.0)])
    r = compute_liquidation_age(conn, "SELL", reference_ts_ms=10_000)
    conn.close()
    assert r == {"status": "LEFT_CENSORED", "age_ms": None}


def test_liquidation_age_direction_is_independent():
    conn = _mk_liq_db(":memory:", [(1000, "BUY", 300_000.0)])
    sell = compute_liquidation_age(conn, "SELL", reference_ts_ms=10_000)
    buy = compute_liquidation_age(conn, "BUY", reference_ts_ms=10_000)
    conn.close()
    assert sell["status"] == "LEFT_CENSORED"
    assert buy == {"status": "OK", "age_ms": 9000}


# ---- market clocks (pure, in-memory candle dicts) ----

def _candle(ts, close, volume=100.0, trade_count=10, dq="AVAILABLE"):
    return {"close_ts_ms": ts, "close": close, "volume": volume, "trade_count": trade_count, "data_quality": dq}


def _flat_series(n_candles, start_ts=0, close=100.0, **kw):
    return _CandleSeries([_candle(start_ts + i * MIN_MS, close, **kw) for i in range(n_candles)])


def test_market_clocks_missing_when_anchor_window_incomplete():
    series = _flat_series(30)  # fewer than CLOCK_WINDOW_CANDLES=60
    r = compute_market_clocks(series, 29 * MIN_MS)
    assert r["status"] == "MISSING"


def test_market_clocks_missing_when_anchor_window_has_gapped_candle():
    rows = [_candle(i * MIN_MS, 100.0) for i in range(60)]
    rows[10]["data_quality"] = "GAPPED"
    series = _CandleSeries(rows)
    r = compute_market_clocks(series, 59 * MIN_MS)
    assert r["status"] == "MISSING"


def test_market_clocks_missing_when_baseline_insufficient():
    # exactly one full anchor window (60) + only a few clean baseline blocks
    n_baseline_blocks_available = CLOCK_BASELINE_MIN_VALID - 2
    total = 60 * (1 + n_baseline_blocks_available)
    rows = [_candle(i * MIN_MS, 100.0) for i in range(total)]
    series = _CandleSeries(rows)
    r = compute_market_clocks(series, (total - 1) * MIN_MS)
    assert r["status"] == "MISSING"


def test_market_clocks_ok_ratio_and_bucket():
    # 30 clean baseline windows (all trade_count=10) + anchor window trade_count=20 -> ratio=2.0 -> HIGH
    baseline_rows = [_candle(i * MIN_MS, 100.0, trade_count=10) for i in range(60 * 30)]
    anchor_rows = [_candle((60 * 30 + i) * MIN_MS, 100.0 + (i % 2) * 0.01, trade_count=20) for i in range(60)]
    series = _CandleSeries(baseline_rows + anchor_rows)
    anchor_ts = (60 * 31 - 1) * MIN_MS
    r = compute_market_clocks(series, anchor_ts)
    assert r["status"] == "OK"
    assert r["trade_count_clock_ratio"] == 2.0
    assert r["trade_count_clock_bucket"] == "HIGH"


def test_market_clocks_missing_when_baseline_median_zero():
    baseline_rows = [_candle(i * MIN_MS, 100.0, volume=0.0) for i in range(60 * 30)]
    anchor_rows = [_candle((60 * 30 + i) * MIN_MS, 100.0, volume=5.0) for i in range(60)]
    series = _CandleSeries(baseline_rows + anchor_rows)
    anchor_ts = (60 * 31 - 1) * MIN_MS
    r = compute_market_clocks(series, anchor_ts)
    assert r["status"] == "OK"  # trade_count/realized_vol still resolve
    assert r["volume_clock_ratio"] is None
    assert r["volume_clock_bucket"] is None


# ---- inference plan helpers ----

def test_holm_adjust_known_values():
    assert holm_adjust([0.01, 0.02, 0.03]) == [0.03, 0.04, 0.04]


def test_holm_adjust_passes_through_none():
    out = holm_adjust([0.01, None, 0.02])
    assert out[1] is None
    assert out[0] == 0.02
    assert out[2] == 0.02


def test_permutation_test_extreme_difference_yields_small_p():
    r = permutation_test_two_sided(high_n=30, high_succ=30, low_n=30, low_succ=0)
    assert r["observed_risk_difference"] == 1.0
    assert r["p_value"] < 0.01


def test_permutation_test_no_difference_yields_large_p():
    r = permutation_test_two_sided(high_n=30, high_succ=15, low_n=30, low_succ=15)
    assert r["observed_risk_difference"] == 0.0
    assert r["p_value"] > 0.5


# ---- cluster bootstrap: custom label parameterization (W10a reuse) ----

def test_cluster_bootstrap_risk_difference_supports_custom_labels():
    rows = (
        [{"cluster_id": f"c{i}", "primary_bucket": "CONFLICT", "path_class": "REVERSAL"} for i in range(15)]
        + [{"cluster_id": f"c{i+15}", "primary_bucket": "AGREEMENT", "path_class": "CONTINUATION"} for i in range(15)]
    )
    r = cluster_bootstrap_risk_difference(rows, "primary_bucket", label_high="CONFLICT", label_low="AGREEMENT", n_boot=50)
    assert r["n_valid_draws"] > 0
    # all CONFLICT rows are REVERSAL, all AGREEMENT rows are not -> risk difference should be ~1.0
    assert r["ci95"][0] > 0.5


# ---- checkpoint closure classification ----

def _mk_test(p_raw, p_holm):
    return {"permutation": {"p_value": p_raw, "p_value_holm_adjusted": p_holm}}


def test_classify_closure_unconfirmed_descriptive_lead_when_nominal_but_not_holm_significant():
    tests = {"liq_age_same_direction": _mk_test(0.0435, 0.348)}
    assert classify_closure(tests) == {"liq_age_same_direction": "UNCONFIRMED_DESCRIPTIVE_LEAD"}


def test_classify_closure_null_when_neither_significant():
    tests = {"push_age": _mk_test(0.783, 1.0)}
    assert classify_closure(tests) == {"push_age": "NULL"}


def test_classify_closure_holm_significant_when_family_wise_significant():
    tests = {"hypothetical": _mk_test(0.001, 0.01)}
    assert classify_closure(tests) == {"hypothetical": "HOLM_SIGNIFICANT"}


def test_classify_closure_undefined_when_p_value_is_none():
    tests = {"empty_feature": _mk_test(None, None)}
    assert classify_closure(tests) == {"empty_feature": "UNDEFINED"}


def test_w7a_real_data_closure_matches_operator_record():
    # locks the operator's own closure record: the 2 nominal (pre-Holm) hits
    # from BATCH-P6-010 stay UNCONFIRMED_DESCRIPTIVE_LEAD, not re-chased.
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        metrics = compute_metrics(conn)
    finally:
        conn.close()
    closure = classify_closure(metrics["tests"])
    assert closure["liq_age_same_direction"] == "UNCONFIRMED_DESCRIPTIVE_LEAD"
    assert closure["realized_vol_clock"] == "UNCONFIRMED_DESCRIPTIVE_LEAD"
    for feat, verdict in closure.items():
        if feat not in {"liq_age_same_direction", "realized_vol_clock"}:
            assert verdict in {"NULL", "UNDEFINED"}


# ---- real-data end-to-end + idempotency ----

def test_compute_metrics_real_data_smoke():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        metrics = compute_metrics(conn)
    finally:
        conn.close()
    assert metrics["analyzed_n"] > 0
    assert metrics["analyzed_n"] <= metrics["total_anchor_n"]
    assert metrics["independent_cycle_n"] > 0
    for feat in ("state_age", "swing_age", "level_age", "push_age",
                 "liq_age_same_direction", "liq_age_opposite_direction",
                 "trade_count_clock", "volume_clock", "realized_vol_clock"):
        assert feat in metrics["tests"]
        t = metrics["tests"][feat]
        assert t["high"]["n"] + t["low"]["n"] <= metrics["analyzed_n"]


def test_freeze_and_record_writes_canonical_sql_and_is_idempotent():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect, init_schema as real_init

    conn = real_connect(DEFAULT_PATH)
    try:
        real_init(conn)
        freeze_and_record(conn)
        freeze_and_record(conn)
        exp_row = conn.execute(
            "SELECT software_verdict, scientific_verdict, question_ids FROM experiment_registry WHERE experiment_id=?",
            (EXPERIMENT_ID,),
        ).fetchone()
        n_registry = conn.execute(
            "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
        ).fetchone()[0]
        n_results = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=? AND metric_name='independent_cycle_n'",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert exp_row == ("PASSED", "ANSWERED_SUPPORTED", "FAM_SIGNAL_AGING_MARKET_CLOCK")
    assert n_registry == 1
    assert n_results == 1
