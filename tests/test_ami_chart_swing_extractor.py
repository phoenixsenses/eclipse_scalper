"""BATCH-P4-002: confirmed swing extraction tests (Chart-Native Extension §4.2).

Run: pytest tests/test_ami_chart_swing_extractor.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.chart.swing_extractor import extract_swings, seed
from ami.warehouse.schema import connect, init_schema

MIN_MS = 60_000


def _mk_candles(highs, lows, start_ts=0):
    return [
        {"open_ts_ms": start_ts + i * MIN_MS, "close_ts_ms": start_ts + (i + 1) * MIN_MS,
         "high": h, "low": l}
        for i, (h, l) in enumerate(zip(highs, lows))
    ]


def test_confirmed_swing_high_detected_with_correct_known_at():
    # index 3 is a clear single peak, confirmed 3 bars later (index 6)
    highs = [100, 101, 102, 110, 103, 102, 101, 100]
    lows = [99, 100, 101, 105, 100, 99, 98, 97]
    candles = _mk_candles(highs, lows)
    swings = extract_swings(candles, "ETHUSDT", "1m", confirmation_bars=3)
    highs_found = [s for s in swings if s["swing_type"] == "HIGH"]
    assert len(highs_found) == 1
    s = highs_found[0]
    assert s["pivot_price"] == 110
    assert s["pivot_ts"] == 3 * MIN_MS
    assert s["known_at_ts"] == candles[6]["close_ts_ms"]
    assert s["known_at_ts"] > s["pivot_ts"]  # never knowable at the pivot itself


def test_confirmed_swing_low_detected():
    highs = [110, 109, 108, 95, 107, 108, 109, 110]
    lows = [100, 99, 98, 90, 97, 98, 99, 100]
    candles = _mk_candles(highs, lows)
    swings = extract_swings(candles, "ETHUSDT", "1m", confirmation_bars=3)
    lows_found = [s for s in swings if s["swing_type"] == "LOW"]
    assert len(lows_found) == 1
    assert lows_found[0]["pivot_price"] == 90


def test_pivot_too_close_to_end_is_never_emitted():
    # a peak at the very last index has zero bars after it -- cannot be confirmed
    highs = [100, 101, 102, 110]
    lows = [99, 100, 101, 105]
    candles = _mk_candles(highs, lows)
    swings = extract_swings(candles, "ETHUSDT", "1m", confirmation_bars=3)
    assert swings == []


def test_flat_top_tie_is_not_fabricated_into_a_swing():
    highs = [100, 101, 110, 110, 101, 100, 99, 98]
    lows = [99, 100, 105, 105, 100, 99, 98, 97]
    candles = _mk_candles(highs, lows)
    swings = extract_swings(candles, "ETHUSDT", "1m", confirmation_bars=3)
    # neither of the tied bars (index 2 or 3) should be reported as THE pivot
    assert all(s["pivot_price"] != 110 for s in swings)


def test_swing_id_deterministic():
    highs = [100, 101, 102, 110, 103, 102, 101, 100]
    lows = [99, 100, 101, 105, 100, 99, 98, 97]
    candles = _mk_candles(highs, lows)
    s1 = extract_swings(candles, "ETHUSDT", "1m", confirmation_bars=3)
    s2 = extract_swings(candles, "ETHUSDT", "1m", confirmation_bars=3)
    assert s1[0]["swing_id"] == s2[0]["swing_id"]
    assert s1[0]["swing_id"].startswith("SWG-")


def test_seed_against_real_candles_and_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    now = 0
    highs = [100, 101, 102, 110, 103, 102, 101, 100, 99, 98]
    lows = [99, 100, 101, 105, 100, 99, 98, 97, 96, 95]
    for i, (h, l) in enumerate(zip(highs, lows)):
        conn.execute(
            "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, high, "
            "low, close, is_closed, partial_status, known_at_ts, data_quality, candle_definition_version, "
            "schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"CDL-{i}", "ETHUSDT", "1m", i * MIN_MS, (i + 1) * MIN_MS, h, h, l, h, 1, "CLOSED",
             (i + 1) * MIN_MS, "AVAILABLE", "test-v1", 4, "test", now, now),
        )
    conn.commit()
    n1 = seed(conn, symbol="ETHUSDT", timeframe="1m")
    n2 = seed(conn, symbol="ETHUSDT", timeframe="1m")
    count = conn.execute("SELECT COUNT(*) FROM ami_swings").fetchone()[0]
    conn.close()
    assert n1 == n2 == count
    assert n1 > 0
