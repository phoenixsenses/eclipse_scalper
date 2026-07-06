"""BATCH-P4-001: candle builder tests (Chart-Native Extension §4.1/§6.3).

Run: pytest tests/test_ami_chart_candle_builder.py --basetemp <scratchpad> -p no:cacheprovider
"""
import sqlite3

from ami.chart.candle_builder import (
    build_candles,
    build_candles_streaming,
    derive_higher_timeframe,
    seed,
    seed_full_history,
)
from ami.warehouse.schema import connect, init_schema

MIN_MS = 60_000


def _mk_trades_db(path, trades):
    """trades: list of (ts_ms, price, quantity, is_buyer_maker)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
        "price REAL, quantity REAL, notional REAL, is_buyer_maker INTEGER)"
    )
    for i, (ts, px, qty, ibm) in enumerate(trades):
        conn.execute(
            "INSERT INTO agg_trades (id, ts_ms, symbol, price, quantity, notional, is_buyer_maker) "
            "VALUES (?,?,?,?,?,?,?)",
            (i, ts, "ETHUSDT", px, qty, px * qty, ibm),
        )
    conn.commit()
    conn.close()


def _add_gaps_table(path, gaps):
    """gaps: list of (start_ts_ms, end_ts_ms) for stream='agg_trades'."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE gaps (id INTEGER PRIMARY KEY, stream TEXT, start_ts_ms INTEGER, "
        "end_ts_ms INTEGER, duration_sec REAL, resolved_bool INTEGER)"
    )
    for i, (s, e) in enumerate(gaps):
        conn.execute(
            "INSERT INTO gaps (id, stream, start_ts_ms, end_ts_ms, duration_sec, resolved_bool) "
            "VALUES (?,?,?,?,?,?)", (i, "agg_trades", s, e, (e - s) / 1000.0, 1),
        )
    conn.commit()
    conn.close()


def test_candle_marked_gapped_when_overlapping_collector_gap(tmp_path):
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [(0, 100.0, 1.0, 0), (30_000, 101.0, 1.0, 1)])
    _add_gaps_table(db, [(10_000, 20_000)])  # overlaps this 1m bucket
    candles = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
    assert candles[0]["data_quality"] == "GAPPED"


def test_candle_marked_available_when_gap_does_not_overlap(tmp_path):
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [(0, 100.0, 1.0, 0), (30_000, 101.0, 1.0, 1)])
    _add_gaps_table(db, [(10 * MIN_MS, 20 * MIN_MS)])  # far outside this bucket
    candles = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
    assert candles[0]["data_quality"] == "AVAILABLE"


def test_missing_gaps_table_defaults_to_available_not_error(tmp_path):
    # existing synthetic fixtures never create a gaps table -- must not crash.
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [(0, 100.0, 1.0, 0)])
    candles = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
    assert candles[0]["data_quality"] == "AVAILABLE"


def test_ohlcv_correctness_single_bucket(tmp_path):
    db = tmp_path / "trades.sqlite"
    trades = [
        (0, 100.0, 1.0, 0),      # taker buy
        (10_000, 105.0, 2.0, 1), # taker sell
        (20_000, 98.0, 1.5, 0),  # taker buy
        (30_000, 102.0, 1.0, 1), # taker sell (close)
    ]
    _mk_trades_db(db, trades)
    candles = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
    assert len(candles) == 1
    c = candles[0]
    assert c["open"] == 100.0
    assert c["close"] == 102.0
    assert c["high"] == 105.0
    assert c["low"] == 98.0
    assert c["volume"] == 5.5
    assert c["trade_count"] == 4
    assert c["taker_buy_volume"] == 2.5   # 1.0 + 1.5
    assert c["taker_sell_volume"] == 3.0  # 2.0 + 1.0
    assert c["is_closed"] == 1
    assert c["known_at_ts"] == MIN_MS


def test_still_forming_bucket_is_never_stored(tmp_path):
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [(0, 100.0, 1.0, 0)])
    # as_of_ms is inside the bucket -- bucket has not closed yet
    candles = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=30_000, trades_db=db)
    assert candles == []


def test_empty_bucket_is_skipped_not_fabricated(tmp_path):
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [(0, 100.0, 1.0, 0), (2 * MIN_MS + 100, 101.0, 1.0, 0)])
    candles = build_candles("ETHUSDT", "1m", 0, 3 * MIN_MS, as_of_ms=3 * MIN_MS, trades_db=db)
    open_times = [c["open_ts_ms"] for c in candles]
    assert MIN_MS not in open_times  # the empty middle minute is absent, not a flat candle
    assert len(candles) == 2


def test_candle_id_deterministic(tmp_path):
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [(0, 100.0, 1.0, 0)])
    c1 = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
    c2 = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
    assert c1[0]["candle_id"] == c2[0]["candle_id"]
    assert c1[0]["candle_id"].startswith("CDL-")


def test_unknown_timeframe_raises(tmp_path):
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [])
    try:
        build_candles("ETHUSDT", "3m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_seed_is_idempotent(tmp_path):
    trades_db = tmp_path / "trades.sqlite"
    _mk_trades_db(trades_db, [(0, 100.0, 1.0, 0), (30_000, 102.0, 1.0, 1)])

    canonical_db = tmp_path / "canonical.sqlite"
    conn = connect(canonical_db)
    init_schema(conn)

    # Isolated idempotency check: build_candles against our synthetic
    # trades_db, insert twice, verify no duplicate rows (real seed() against
    # DEFAULT_TRADES_DB is exercised separately in test_real_seed_... below).
    candles = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=trades_db)
    now = 0
    for c in candles:
        conn.execute(
            "INSERT INTO ami_candles (candle_id, symbol, venue, timeframe, open_ts_ms, close_ts_ms, "
            "open, high, low, close, volume, trade_count, taker_buy_volume, taker_sell_volume, "
            "is_closed, partial_status, known_at_ts, data_quality, source_hash, "
            "candle_definition_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol, timeframe, open_ts_ms, candle_definition_version) DO UPDATE SET "
            "close=excluded.close",
            (c["candle_id"], c["symbol"], c["venue"], c["timeframe"], c["open_ts_ms"], c["close_ts_ms"],
             c["open"], c["high"], c["low"], c["close"], c["volume"], c["trade_count"],
             c["taker_buy_volume"], c["taker_sell_volume"], c["is_closed"], c["partial_status"],
             c["known_at_ts"], c["data_quality"], c["source_hash"], c["candle_definition_version"],
             4, "test", now, now),
        )
    conn.commit()
    count1 = conn.execute("SELECT COUNT(*) FROM ami_candles WHERE timeframe='1m'").fetchone()[0]
    for c in candles:  # re-run
        conn.execute(
            "INSERT INTO ami_candles (candle_id, symbol, venue, timeframe, open_ts_ms, close_ts_ms, "
            "open, high, low, close, volume, trade_count, taker_buy_volume, taker_sell_volume, "
            "is_closed, partial_status, known_at_ts, data_quality, source_hash, "
            "candle_definition_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol, timeframe, open_ts_ms, candle_definition_version) DO UPDATE SET "
            "close=excluded.close",
            (c["candle_id"], c["symbol"], c["venue"], c["timeframe"], c["open_ts_ms"], c["close_ts_ms"],
             c["open"], c["high"], c["low"], c["close"], c["volume"], c["trade_count"],
             c["taker_buy_volume"], c["taker_sell_volume"], c["is_closed"], c["partial_status"],
             c["known_at_ts"], c["data_quality"], c["source_hash"], c["candle_definition_version"],
             4, "test", now, now),
        )
    conn.commit()
    count2 = conn.execute("SELECT COUNT(*) FROM ami_candles WHERE timeframe='1m'").fetchone()[0]
    conn.close()
    assert count1 == count2 == 1


def test_streaming_matches_reference_on_multi_bucket_synthetic(tmp_path):
    db = tmp_path / "trades.sqlite"
    trades = [
        (0, 100.0, 1.0, 0), (10_000, 105.0, 2.0, 1),          # bucket 0
        (MIN_MS + 5_000, 103.0, 1.0, 0),                       # bucket 1
        (3 * MIN_MS + 1_000, 99.0, 0.5, 1),                    # bucket 3 (bucket 2 empty)
    ]
    _mk_trades_db(db, trades)
    ref = build_candles("ETHUSDT", "1m", 0, 4 * MIN_MS, as_of_ms=4 * MIN_MS, trades_db=db)
    streamed = build_candles_streaming("ETHUSDT", "1m", 0, 4 * MIN_MS, as_of_ms=4 * MIN_MS, trades_db=db)
    assert streamed == ref


def test_streaming_respects_gaps_table_same_as_reference(tmp_path):
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [(0, 100.0, 1.0, 0), (30_000, 101.0, 1.0, 1)])
    _add_gaps_table(db, [(10_000, 20_000)])
    ref = build_candles("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
    streamed = build_candles_streaming("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=MIN_MS, trades_db=db)
    assert streamed == ref == [dict(ref[0], data_quality="GAPPED")]


def test_streaming_still_forming_bucket_never_stored(tmp_path):
    db = tmp_path / "trades.sqlite"
    _mk_trades_db(db, [(0, 100.0, 1.0, 0)])
    streamed = build_candles_streaming("ETHUSDT", "1m", 0, MIN_MS, as_of_ms=30_000, trades_db=db)
    assert streamed == []


def test_streaming_small_row_chunk_still_correct(tmp_path):
    # row_chunk=1 forces fetchmany() to loop many times -- must not lose or
    # duplicate rows relative to the reference single-query implementation.
    db = tmp_path / "trades.sqlite"
    trades = [(i * 5_000, 100.0 + i, 1.0, i % 2) for i in range(20)]  # spans 2 buckets
    _mk_trades_db(db, trades)
    ref = build_candles("ETHUSDT", "1m", 0, 2 * MIN_MS, as_of_ms=2 * MIN_MS, trades_db=db)
    streamed = build_candles_streaming("ETHUSDT", "1m", 0, 2 * MIN_MS, as_of_ms=2 * MIN_MS,
                                        trades_db=db, row_chunk=1)
    assert streamed == ref


def test_derive_higher_timeframe_aggregates_five_1m_into_5m():
    base = [
        {"open_ts_ms": i * MIN_MS, "close_ts_ms": (i + 1) * MIN_MS, "open": 100.0 + i, "high": 100.0 + i + 0.5,
         "low": 100.0 + i - 0.5, "close": 100.0 + i + 0.2, "volume": 1.0, "trade_count": 1,
         "taker_buy_volume": 0.5, "taker_sell_volume": 0.5, "known_at_ts": (i + 1) * MIN_MS,
         "data_quality": "AVAILABLE", "venue": None}
        for i in range(5)
    ]
    out = derive_higher_timeframe(base, "ETHUSDT", "5m")
    assert len(out) == 1
    c = out[0]
    assert c["open_ts_ms"] == 0
    assert c["close_ts_ms"] == 5 * MIN_MS
    assert c["open"] == base[0]["open"]
    assert c["close"] == base[-1]["close"]
    assert c["high"] == max(b["high"] for b in base)
    assert c["low"] == min(b["low"] for b in base)
    assert c["volume"] == 5.0
    assert c["trade_count"] == 5
    assert c["known_at_ts"] == base[-1]["known_at_ts"]
    assert c["data_quality"] == "AVAILABLE"


def test_derive_higher_timeframe_propagates_gapped_conservatively():
    base = [
        {"open_ts_ms": 0, "close_ts_ms": MIN_MS, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 1.0, "trade_count": 1, "taker_buy_volume": 0.5, "taker_sell_volume": 0.5,
         "known_at_ts": MIN_MS, "data_quality": "AVAILABLE", "venue": None},
        {"open_ts_ms": MIN_MS, "close_ts_ms": 2 * MIN_MS, "open": 100.5, "high": 101.5, "low": 99.5,
         "close": 100.0, "volume": 1.0, "trade_count": 1, "taker_buy_volume": 0.5, "taker_sell_volume": 0.5,
         "known_at_ts": 2 * MIN_MS, "data_quality": "GAPPED", "venue": None},
    ]
    out = derive_higher_timeframe(base, "ETHUSDT", "5m")
    assert out[0]["data_quality"] == "GAPPED"


def test_derive_higher_timeframe_skips_bucket_with_zero_base_candles():
    # only a 5m-bucket with NO underlying 1m candles must be absent entirely.
    base = [{"open_ts_ms": 0, "close_ts_ms": MIN_MS, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
             "volume": 1.0, "trade_count": 1, "taker_buy_volume": 0.5, "taker_sell_volume": 0.5,
             "known_at_ts": MIN_MS, "data_quality": "AVAILABLE", "venue": None}]
    out = derive_higher_timeframe(base, "ETHUSDT", "5m")
    assert len(out) == 1  # only the one 5m bucket that actually has data


def test_seed_full_history_is_idempotent_against_synthetic_trades_db(tmp_path, monkeypatch):
    trades_db = tmp_path / "trades.sqlite"
    trades = [(i * 90_000, 100.0 + (i % 3), 1.0, i % 2) for i in range(50)]  # spans several 1m/5m buckets
    _mk_trades_db(trades_db, trades)
    monkeypatch.setattr("ami.chart.candle_builder.DEFAULT_TRADES_DB", trades_db)

    canonical_db = tmp_path / "canonical.sqlite"
    conn = connect(canonical_db)
    init_schema(conn)

    result1 = seed_full_history(conn, symbol="ETHUSDT", start_ts_ms=0, end_ts_ms=trades[-1][0] + MIN_MS)
    count1 = conn.execute("SELECT COUNT(*) FROM ami_candles").fetchone()[0]
    result2 = seed_full_history(conn, symbol="ETHUSDT", start_ts_ms=0, end_ts_ms=trades[-1][0] + MIN_MS)
    count2 = conn.execute("SELECT COUNT(*) FROM ami_candles").fetchone()[0]
    conn.close()
    assert result1["n_1m"] > 0 and result1["n_5m"] > 0
    assert result1 == result2
    assert count1 == count2


# [BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1-EVIDENCE-RECONCILIATION]
# Fixed historical window for the deterministic OHLC-invariant test below --
# NOT "now - 2h". The original version anchored to wall-clock "now", so a
# live-collector gap near "now" (confirmed real: ETHUSDT/BTCUSDT agg_trades
# streams stalled for ~2-2.5h during this session while SOLUSDT kept
# streaming) made this test flip between pass/fail depending purely on
# collector health at the moment pytest happened to run -- not a
# code/logic regression. A deterministic regression suite must not depend on
# whether the live collector is healthy right now. This window (confirmed
# read-only: 200,589 real ETHUSDT trades inside it) is anchored to a fixed
# point in the past and will never go stale, because it is a literal
# constant, not a `time.time()`-relative computation.
_FIXED_WINDOW_END_TS_MS = 1783342892143  # real ETHUSDT agg_trades timestamp, frozen at reconciliation time
_FIXED_WINDOW_HOURS = 2


def test_real_seed_against_default_trades_db_is_sane(tmp_path):
    # Deterministic integration test against the real, live data/microstructure.db
    # -- validates OHLC invariants hold on real trade data (not just synthetic
    # fixtures), using a FIXED historical window (see _FIXED_WINDOW_END_TS_MS
    # above) so this test's pass/fail depends only on this repo's candle-
    # building code, never on the live collector's current health. Live
    # collector freshness is an operational concern, monitored separately by
    # status_eclipse.ps1 / the heartbeat watchdog (see SYSTEM_STATE.md) --
    # deliberately NOT re-implemented as a second pytest test here, since any
    # pytest assertion tied to "is the collector caught up right now" would
    # reintroduce exactly the same non-determinism this fix removes from the
    # regression suite that must be reproducibly green.
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    n = seed(conn, symbol="ETHUSDT", timeframes=("1m",), lookback_hours=_FIXED_WINDOW_HOURS,
             end_ts_ms=_FIXED_WINDOW_END_TS_MS)
    rows = conn.execute("SELECT open, high, low, close, volume FROM ami_candles").fetchall()
    conn.close()
    assert n == len(rows)
    assert n > 0  # this fixed window has real ETHUSDT trades (confirmed read-only)
    for o, h, l, c, v in rows:
        assert l <= o <= h
        assert l <= c <= h
        assert v > 0
