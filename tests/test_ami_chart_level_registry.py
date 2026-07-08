"""BATCH-P4-003: Level object registry tests (Chart-Native Extension §4.3).

Run: pytest tests/test_ami_chart_level_registry.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.chart.level_registry import (
    _touch_stats,
    compute_previous_day_levels,
    compute_session_levels,
    compute_swing_levels,
    seed,
)
from ami.warehouse.schema import connect, init_schema

HOUR_MS = 3600_000
MIN_MS = 60_000


def _candle(open_ts_ms, h, l, c=None, close_ts_ms=None):
    return {"open_ts_ms": open_ts_ms, "close_ts_ms": close_ts_ms or open_ts_ms + MIN_MS,
            "high": h, "low": l, "close": c if c is not None else h}


def test_session_levels_exclude_still_open_last_session():
    # ASIA session 00:00-07:00 UTC (day 0), then a candle in EUROPE (07:xx) --
    # ASIA is fully elapsed and should be reported; EUROPE (only 1 candle so
    # far) must NOT be reported as it may still be accumulating.
    candles = [
        _candle(0, 105.0, 95.0),               # 00:00 ASIA
        _candle(3 * HOUR_MS, 110.0, 90.0),     # 03:00 ASIA
        _candle(7 * HOUR_MS, 100.0, 99.0),     # 07:00 EUROPE (still open)
    ]
    levels = compute_session_levels(candles, "ETHUSDT", "1m")
    session_highs = [lv for lv in levels if lv["level_type"] == "SESSION_HIGH"]
    assert len(session_highs) == 1
    assert session_highs[0]["price"] == 110.0
    # F-B1: known_at_ts is the session's true boundary end (07:00), not merely
    # the last observed candle's close (which is earlier here, at 03:01).
    assert session_highs[0]["known_at_ts"] == 7 * HOUR_MS


def test_session_level_known_at_uses_last_candle_when_it_closes_after_boundary():
    # if data extends past the nominal boundary (e.g. a late-closing bar),
    # known_at_ts must reflect that later timestamp, not regress to the
    # boundary alone.
    candles = [
        _candle(0, 105.0, 95.0, close_ts_ms=7 * HOUR_MS + 5 * MIN_MS),  # closes after boundary
        _candle(7 * HOUR_MS + 10 * MIN_MS, 100.0, 99.0),  # next session, still open
    ]
    levels = compute_session_levels(candles, "ETHUSDT", "1m")
    session_highs = [lv for lv in levels if lv["level_type"] == "SESSION_HIGH"]
    assert session_highs[0]["known_at_ts"] == 7 * HOUR_MS + 5 * MIN_MS


def test_truncated_first_session_is_skipped_not_fabricated():
    # F-B4: the lookback window starts at 03:00 (mid-ASIA-session), so ASIA's
    # true high/low (00:00-07:00) cannot be known -- must not be emitted.
    candles = [
        _candle(3 * HOUR_MS, 110.0, 90.0),   # ASIA, but window starts mid-session
        _candle(7 * HOUR_MS, 105.0, 95.0),   # EUROPE, fully elapsed
        _candle(13 * HOUR_MS, 108.0, 98.0),  # US, still open
    ]
    levels = compute_session_levels(candles, "ETHUSDT", "1m")
    origins = {lv["origin_ts"] for lv in levels}
    assert 0 not in origins  # ASIA's origin (00:00) never appears -- truncated period skipped
    assert 7 * HOUR_MS in origins  # EUROPE is complete and reported


def test_previous_day_levels_exclude_still_open_last_day():
    DAY = 86_400_000
    candles = [_candle(0, 100.0, 90.0), _candle(DAY - MIN_MS, 105.0, 95.0), _candle(DAY, 102.0, 98.0)]
    levels = compute_previous_day_levels(candles, "ETHUSDT", "1m")
    day_highs = [lv for lv in levels if lv["level_type"] == "PREVIOUS_DAY_HIGH"]
    assert len(day_highs) == 1
    assert day_highs[0]["price"] == 105.0
    # F-B1: known_at_ts is the true day boundary (00:00 next day), not merely
    # the last observed candle's close (DAY - MIN_MS + MIN_MS = DAY, which
    # happens to coincide here, but the invariant is the max(), not a coincidence).
    assert day_highs[0]["known_at_ts"] == DAY


def test_truncated_first_day_is_skipped_not_fabricated():
    DAY = 86_400_000
    candles = [
        _candle(DAY // 2, 110.0, 90.0),  # day 0, but window starts mid-day -- truncated
        _candle(DAY, 105.0, 95.0),        # day 1, fully elapsed
        _candle(2 * DAY, 108.0, 98.0),    # day 2, still open
    ]
    levels = compute_previous_day_levels(candles, "ETHUSDT", "1m")
    origins = {lv["origin_ts"] for lv in levels}
    assert 0 not in origins       # day 0's origin never appears -- truncated
    assert DAY in origins         # day 1 is complete and reported


def test_swing_levels_pass_through():
    swings = [{"swing_type": "HIGH", "pivot_ts": 100, "pivot_price": 50.0, "known_at_ts": 200},
              {"swing_type": "LOW", "pivot_ts": 300, "pivot_price": 10.0, "known_at_ts": 400}]
    levels = compute_swing_levels(swings)
    types = {lv["level_type"] for lv in levels}
    assert types == {"SWING_HIGH", "SWING_LOW"}


def test_touch_stats_high_type_acceptance_and_rejection():
    candles = [
        _candle(0, 105.0, 95.0, c=98.0),    # touch, close below price=100 -> rejection
        _candle(MIN_MS, 106.0, 99.0, c=102.0),  # touch, close above -> acceptance
        _candle(2 * MIN_MS, 90.0, 85.0, c=88.0),  # no touch
    ]
    stats = _touch_stats("SESSION_HIGH", 100.0, known_at_ts=0, candles=candles)
    assert stats["touch_count"] == 2
    assert stats["rejection_count"] == 1
    assert stats["acceptance_count"] == 1
    assert stats["last_touch_ts"] == candles[1]["close_ts_ms"]


def test_touch_stats_ignores_candles_before_known_at():
    candles = [_candle(0, 105.0, 95.0, c=102.0)]  # would touch/accept, but before known_at_ts
    stats = _touch_stats("SESSION_HIGH", 100.0, known_at_ts=MIN_MS, candles=candles)
    assert stats["touch_count"] == 0


def _insert_hourly_candles(conn, n_days: int, definition_version: str = "test-v1"):
    """Hourly candles (timeframe='1h') covering n_days full UTC days -- dense
    enough that every session/day boundary hour is exactly represented (no
    accidental F-B4 truncation from a sparse synthetic fixture)."""
    now = 0
    DAY = 86_400_000
    i = 0
    for day in range(n_days):
        for hour in range(24):
            ts = day * DAY + hour * HOUR_MS
            conn.execute(
                "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, "
                "high, low, close, is_closed, partial_status, known_at_ts, data_quality, "
                "candle_definition_version, schema_version, provenance, created_ms, updated_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"CDL-{i}", "ETHUSDT", "1h", ts, ts + HOUR_MS, 100.0 + i, 105.0 + i, 95.0 + i, 100.0 + i,
                 1, "CLOSED", ts + HOUR_MS, "AVAILABLE", definition_version, 5, "test", now, now),
            )
            i += 1
    conn.commit()


def test_seed_real_candles_and_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    _insert_hourly_candles(conn, n_days=2)
    n1 = seed(conn, symbol="ETHUSDT", timeframe="1h")
    n2 = seed(conn, symbol="ETHUSDT", timeframe="1h")
    count = conn.execute("SELECT COUNT(*) FROM ami_levels").fetchone()[0]
    conn.close()
    assert n1 == n2 == count
    assert n1 > 0


def test_seed_marks_touch_stats_not_point_in_time_safe(tmp_path):
    # F-B2: touch/rejection/acceptance are a single build-time cumulative
    # aggregate -- touch_stats_point_in_time must stay 0 until a real
    # point-in-time recomputation engine exists.
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    _insert_hourly_candles(conn, n_days=2)
    seed(conn, symbol="ETHUSDT", timeframe="1h")
    flags = {r[0] for r in conn.execute("SELECT DISTINCT touch_stats_point_in_time FROM ami_levels")}
    conn.close()
    assert flags == {0}


def test_seed_cleans_up_superseded_level_definition_version(tmp_path):
    # F-B4/F-B1: rows from the old buggy "level-v1" definition are a software
    # defect, not a research verdict -- seed() must remove them (controlled,
    # reproducible cleanup), not leave them alongside the corrected version.
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    now = 0
    conn.execute(
        "INSERT INTO ami_levels (level_id, symbol, level_type, price, origin_ts, known_at_ts, timeframe, "
        "source_type, level_definition_version, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("LVL-OLD-BUGGY", "ETHUSDT", "SESSION_HIGH", 999.0, 0, 1000, "1h", "SESSION",
         "level-v1", 4, "test", now, now),
    )
    conn.commit()
    _insert_hourly_candles(conn, n_days=2)
    seed(conn, symbol="ETHUSDT", timeframe="1h")
    remaining_old = conn.execute(
        "SELECT COUNT(*) FROM ami_levels WHERE level_definition_version='level-v1'"
    ).fetchone()[0]
    conn.close()
    assert remaining_old == 0
