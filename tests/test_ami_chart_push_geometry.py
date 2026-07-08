"""BATCH-P4-004: Push object tests (Chart-Native Extension §7.1).

Run: pytest tests/test_ami_chart_push_geometry.py --basetemp <scratchpad> -p no:cacheprovider
"""
import pytest

from ami.chart.push_geometry import build_pushes, seed
from ami.warehouse.schema import connect, init_schema

MIN_MS = 60_000


def _swing(sid, stype, ts, price, known_at):
    return {"swing_id": sid, "swing_type": stype, "pivot_ts": ts, "pivot_price": price, "known_at_ts": known_at}


def _candle(ts, close, volume=1.0):
    return {"open_ts_ms": ts, "close_ts_ms": ts + MIN_MS, "close": close, "volume": volume}


def test_alternating_swings_produce_up_and_down_pushes():
    swings = [
        _swing("S0", "LOW", 0, 100.0, MIN_MS),
        _swing("S1", "HIGH", 3 * MIN_MS, 110.0, 4 * MIN_MS),
        _swing("S2", "LOW", 6 * MIN_MS, 102.0, 7 * MIN_MS),
    ]
    candles = [_candle(i * MIN_MS, 100 + i) for i in range(8)]
    pushes = build_pushes(swings, candles, "ETHUSDT", "1m")
    assert len(pushes) == 2
    assert pushes[0]["direction"] == "UP"
    assert pushes[0]["displacement_bps"] > 0
    assert pushes[1]["direction"] == "DOWN"
    assert pushes[1]["displacement_bps"] < 0


def test_same_type_consecutive_swings_not_paired():
    swings = [
        _swing("S0", "LOW", 0, 100.0, MIN_MS),
        _swing("S1", "LOW", 3 * MIN_MS, 95.0, 4 * MIN_MS),  # same type -- not a push
        _swing("S2", "HIGH", 6 * MIN_MS, 110.0, 7 * MIN_MS),
    ]
    candles = [_candle(i * MIN_MS, 100 + i) for i in range(8)]
    pushes = build_pushes(swings, candles, "ETHUSDT", "1m")
    assert len(pushes) == 1
    assert pushes[0]["start_swing_id"] == "S1"
    assert pushes[0]["end_swing_id"] == "S2"


def test_pullback_after_bps_from_next_push_and_known_at_advances():
    swings = [
        _swing("S0", "LOW", 0, 100.0, MIN_MS),
        _swing("S1", "HIGH", 3 * MIN_MS, 110.0, 4 * MIN_MS),
        _swing("S2", "LOW", 6 * MIN_MS, 105.0, 7 * MIN_MS),
    ]
    candles = [_candle(i * MIN_MS, 100 + i) for i in range(8)]
    pushes = build_pushes(swings, candles, "ETHUSDT", "1m")
    first, second = pushes[0], pushes[1]
    assert first["pullback_after_bps"] == abs(second["displacement_bps"])
    # first push's own end-swing (S1) confirms at 4*MIN_MS, but pullback_after
    # depends on S2 (known at 7*MIN_MS) -- known_at_ts must advance to reflect that.
    assert first["known_at_ts"] == 7 * MIN_MS
    assert second["pullback_after_bps"] is None  # last push -- no following reversal yet


def test_push_id_deterministic():
    swings = [_swing("S0", "LOW", 0, 100.0, MIN_MS), _swing("S1", "HIGH", 3 * MIN_MS, 110.0, 4 * MIN_MS)]
    candles = [_candle(i * MIN_MS, 100 + i) for i in range(4)]
    p1 = build_pushes(swings, candles, "ETHUSDT", "1m")
    p2 = build_pushes(swings, candles, "ETHUSDT", "1m")
    assert p1[0]["push_id"] == p2[0]["push_id"]
    assert p1[0]["push_id"].startswith("PSH-")


def test_efficiency_ratio_never_exceeds_one():
    # triangle inequality: net displacement can never exceed total path length
    # when both are measured against the same fixed reference price.
    swings = [
        _swing("S0", "LOW", 0, 100.0, MIN_MS),
        _swing("S1", "HIGH", 5 * MIN_MS, 108.0, 6 * MIN_MS),
    ]
    # a choppy, non-monotonic path between start and end (up, down, up, up)
    candles = [_candle(0, 100.0), _candle(MIN_MS, 104.0), _candle(2 * MIN_MS, 101.0),
               _candle(3 * MIN_MS, 106.0), _candle(4 * MIN_MS, 103.0), _candle(5 * MIN_MS, 108.0)]
    pushes = build_pushes(swings, candles, "ETHUSDT", "1m")
    assert pushes[0]["efficiency_ratio"] <= 1.0 + 1e-9


def test_no_intervening_candles_gives_direct_path_efficiency_one():
    # With zero intervening candles there is no evidence of a detour -- the
    # path is exactly the direct swing-to-swing distance, so efficiency is 1.0
    # (not None: the boundary segments alone are enough to compute a path).
    swings = [_swing("S0", "LOW", 0, 100.0, MIN_MS), _swing("S1", "HIGH", MIN_MS, 110.0, 2 * MIN_MS)]
    pushes = build_pushes(swings, [], "ETHUSDT", "1m")
    assert pushes[0]["path_length_bps"] == pytest.approx(abs(pushes[0]["displacement_bps"]))
    assert pushes[0]["efficiency_ratio"] == pytest.approx(1.0)


def test_seed_is_idempotent_and_real_data_smoke(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    now = 0
    for i, (sid, stype, price) in enumerate([
        ("S0", "LOW", 100.0), ("S1", "HIGH", 110.0), ("S2", "LOW", 102.0), ("S3", "HIGH", 115.0),
    ]):
        ts = i * 3 * MIN_MS
        conn.execute(
            "INSERT INTO ami_swings (swing_id, symbol, timeframe, swing_type, pivot_ts, pivot_price, "
            "confirmation_ts, confirmation_method, known_at_ts, swing_definition_version, schema_version, "
            "provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, "ETHUSDT", "1m", stype, ts, price, ts + MIN_MS, "test", ts + MIN_MS, "test-v1",
             4, "test", now, now),
        )
    for i in range(12):
        ts = i * MIN_MS
        conn.execute(
            "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, high, "
            "low, close, volume, is_closed, partial_status, known_at_ts, data_quality, "
            "candle_definition_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"CDL-{i}", "ETHUSDT", "1m", ts, ts + MIN_MS, 100 + i, 101 + i, 99 + i, 100 + i, 1.0,
             1, "CLOSED", ts + MIN_MS, "AVAILABLE", "test-v1", 4, "test", now, now),
        )
    conn.commit()
    n1 = seed(conn, symbol="ETHUSDT", timeframe="1m")
    n2 = seed(conn, symbol="ETHUSDT", timeframe="1m")
    count = conn.execute("SELECT COUNT(*) FROM ami_pushes").fetchone()[0]
    conn.close()
    assert n1 == n2 == count == 3
