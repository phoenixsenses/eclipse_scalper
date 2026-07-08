"""BATCH-P6-007 (W6): compression + relative strength + session tests.

Run: pytest tests/test_ami_research_w6_compression_rs_session.py --basetemp <scratchpad> -p no:cacheprovider
"""
import sqlite3

from ami.research.w6_compression_rs_session import (
    EXPERIMENT_ID,
    classify_compression,
    classify_relative_strength,
    classify_session,
    compute_metrics,
    freeze_and_record,
)
from ami.states.engine import StateEngine
from ami.warehouse.schema import connect, init_schema

MIN_MS = 60_000
NOW = 0
HOUR_MS = 3600_000


def _mk_microstructure_db(path, eth_prices, btc_prices):
    """eth_prices/btc_prices: list of (ts_ms, price)."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
    for ts, px in eth_prices:
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (ts, "ETHUSDT", px))
    for ts, px in btc_prices:
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (ts, "BTCUSDT", px))
    conn.commit()
    conn.close()


def test_classify_session_boundaries():
    # 2026-01-01T00:30 UTC -> ASIA; 07:30 -> EUROPE; 13:30 -> US; 21:30 -> OFF
    import datetime as dt

    def ts(hour):
        return int(dt.datetime(2026, 1, 1, hour, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)

    assert classify_session(ts(0)) == "ASIA"
    assert classify_session(ts(7)) == "EUROPE"
    assert classify_session(ts(13)) == "US"
    assert classify_session(ts(21)) == "OFF"


def test_classify_relative_strength_sign(tmp_path):
    db = tmp_path / "micro.sqlite"
    # ETH flat, BTC down over the lookback -> ETH relatively STRONG
    eth = [(-HOUR_MS, 100.0), (0, 100.0)]
    btc = [(-HOUR_MS, 100.0), (0, 95.0)]
    _mk_microstructure_db(db, eth, btc)
    engine = StateEngine(db_path=db)
    try:
        assert classify_relative_strength(engine, 0) == "RS_ETH_STRONG"
    finally:
        engine.conn.close()


def test_classify_relative_strength_weak(tmp_path):
    db = tmp_path / "micro.sqlite"
    eth = [(-HOUR_MS, 100.0), (0, 95.0)]
    btc = [(-HOUR_MS, 100.0), (0, 100.0)]
    _mk_microstructure_db(db, eth, btc)
    engine = StateEngine(db_path=db)
    try:
        assert classify_relative_strength(engine, 0) == "RS_ETH_WEAK"
    finally:
        engine.conn.close()


def test_classify_relative_strength_none_when_no_price_data(tmp_path):
    db = tmp_path / "micro.sqlite"
    _mk_microstructure_db(db, [], [])
    engine = StateEngine(db_path=db)
    try:
        assert classify_relative_strength(engine, 0) is None
    finally:
        engine.conn.close()


def test_classify_compression_binary_label(tmp_path):
    db = tmp_path / "micro.sqlite"
    # flat, low-vol prices for both prior and current windows (not compression by
    # this data), just verify the function returns one of the 2 allowed labels
    prices = [(i * 300_000, 100.0 + (i % 3) * 0.01) for i in range(-24, 1)]
    _mk_microstructure_db(db, prices, prices)
    engine = StateEngine(db_path=db)
    try:
        label = classify_compression(engine, "ETHUSDT", 0)
        assert label in {"COMPRESSION", "NOT_COMPRESSION"}
    finally:
        engine.conn.close()


def _db(tmp_path):
    conn = connect(tmp_path / "canonical.sqlite")
    init_schema(conn)
    return conn


def _insert_event(conn, eid, anchor_ts, symbol="ETHUSDT"):
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
        "event_definition_version, censor_status, event_count, schema_version, provenance, created_ms, "
        "updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (eid, "FAM_A", symbol, anchor_ts, "REAL_LIQUIDATION", "test-v1", "COMPLETED", 1, 7, "test", NOW, NOW),
    )


def _insert_candle(conn, cid, ts, close, symbol="ETHUSDT"):
    conn.execute(
        "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, high, low, "
        "close, is_closed, partial_status, known_at_ts, data_quality, candle_definition_version, "
        "schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, symbol, "1m", ts, ts + MIN_MS, close, close, close, close, 1, "CLOSED", ts + MIN_MS,
         "AVAILABLE", "test-v1", 7, "test", NOW, NOW),
    )


def test_compute_metrics_real_data_smoke():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        metrics = compute_metrics(conn)
    finally:
        conn.close()
    assert metrics["anchor_n"] > 0
    # every anchor got a session label (session is always computable, no missing data)
    assert all(a["session"] in {"ASIA", "EUROPE", "US", "OFF"} for a in metrics["per_anchor"])


def test_freeze_and_record_writes_canonical_sql_and_is_idempotent():
    # exercises against the real canonical.sqlite + real microstructure.db,
    # since StateEngine's default db_path is not easily swapped inside a
    # from-scratch synthetic canonical.sqlite fixture without also faking a
    # full microstructure.db -- reuses the same pattern as W1/W4 real-data checks.
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect, init_schema as real_init

    conn = real_connect(DEFAULT_PATH)
    try:
        real_init(conn)
        freeze_and_record(conn)
        freeze_and_record(conn)
        exp_row = conn.execute(
            "SELECT software_verdict, scientific_verdict FROM experiment_registry WHERE experiment_id=?",
            (EXPERIMENT_ID,),
        ).fetchone()
        n_registry = conn.execute(
            "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
        ).fetchone()[0]
        n_results = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=? AND metric_name='anchor_n'",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert exp_row == ("PASSED", "ANSWERED_SUPPORTED")
    assert n_registry == 1
    assert n_results == 1
