"""BATCH-P4-001: candle morphology tests (Chart-Native Extension §6.1/§6.3).

Run: pytest tests/test_ami_chart_candle_morphology.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.chart.candle_morphology import compute_morphology, seed
from ami.warehouse.schema import connect, init_schema


def test_bullish_candle_close_near_high():
    m = compute_morphology({"open": 100.0, "high": 110.0, "low": 99.0, "close": 109.0})
    assert m["range_abs"] == 11.0
    assert round(m["close_location_value"], 2) == round((109 - 99) / 11, 2)
    assert m["close_quality_label"] == "CLOSE_NEAR_HIGH"
    assert m["directional_body"] > 0


def test_bearish_candle_close_near_low():
    m = compute_morphology({"open": 110.0, "high": 111.0, "low": 100.0, "close": 101.0})
    assert m["close_quality_label"] == "CLOSE_NEAR_LOW"
    assert m["directional_body"] < 0


def test_mid_range_close():
    m = compute_morphology({"open": 100.0, "high": 110.0, "low": 100.0, "close": 105.0})
    assert m["close_quality_label"] == "MID_RANGE_CLOSE"


def test_zero_range_returns_none_not_zero():
    # No fabrication: a flat candle (open==high==low==close) has mathematically
    # undefined ratios -- must be None, not silently 0.
    m = compute_morphology({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    assert m["range_abs"] == 0
    assert m["body_ratio"] is None
    assert m["close_location_value"] is None
    assert m["close_quality_label"] is None


def test_wick_ratios_sum_reasonably():
    m = compute_morphology({"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0})
    body = abs(102.0 - 100.0)
    rng = 105.0 - 95.0
    assert round(m["body_ratio"], 4) == round(body / rng, 4)


def test_seed_populates_and_is_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    now = 0
    conn.execute(
        "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, high, low, "
        "close, is_closed, partial_status, known_at_ts, data_quality, candle_definition_version, "
        "schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("CDL-TEST-1", "ETHUSDT", "1m", 0, 60_000, 100.0, 110.0, 99.0, 109.0, 1, "CLOSED", 60_000,
         "AVAILABLE", "test-v1", 4, "test", now, now),
    )
    conn.commit()
    n1 = seed(conn)
    n2 = seed(conn)
    row = conn.execute(
        "SELECT close_quality_label FROM ami_candle_morphology WHERE candle_id='CDL-TEST-1'"
    ).fetchone()
    conn.close()
    assert n1 == n2 == 1
    assert row == ("CLOSE_NEAR_HIGH",)
