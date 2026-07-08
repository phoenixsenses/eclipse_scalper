"""BATCH: AMI HISTORICAL CANDLE GAP REMEDIATION -- tests for
ami/chart/candle_gap_repair_rehearsal.py (source audit + disposable
rehearsal ONLY, no real canonical write).

Run: pytest tests/test_ami_chart_candle_gap_repair_rehearsal.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect

import ami.chart.candle_gap_repair_rehearsal as repair

_FORBIDDEN_TERMS = ("forward_fill",)


def test_no_interpolation_or_synthesis_terms_in_module_source():
    src = inspect.getsource(repair).lower()
    hits = [t for t in _FORBIDDEN_TERMS if t in src]
    assert hits == [], f"forbidden interpolation/synthesis terms found: {hits}"


def test_candle_definition_version_is_distinct_from_original():
    assert repair.CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR != "candle-agg_trades-v1"


def _valid_kline(open_ts_ms, o=100.0, h=101.0, l=99.0, c=100.5, v=10.0, n_trades=5, taker_buy=6.0):
    return [open_ts_ms, o, h, l, c, v, open_ts_ms + 59999, v * o, n_trades, taker_buy, taker_buy * o, "0"]


def test_validate_kline_row_accepts_well_formed_row():
    ok, reason = repair.validate_kline_row(_valid_kline(60_000))
    assert ok is True
    assert reason is None


def test_validate_kline_row_rejects_non_1m_aligned():
    ok, reason = repair.validate_kline_row(_valid_kline(60_500))
    assert ok is False
    assert reason == "NOT_1M_ALIGNED"


def test_validate_kline_row_rejects_invalid_ohlc():
    bad = _valid_kline(60_000, o=100.0, h=90.0, l=99.0, c=100.5)  # high < open
    ok, reason = repair.validate_kline_row(bad)
    assert ok is False
    assert reason == "INVALID_OHLC_RELATIONSHIP"


def test_validate_kline_row_rejects_close_time_mismatch():
    bad = _valid_kline(60_000)
    bad[6] = 60_000 + 100  # wrong close_time
    ok, reason = repair.validate_kline_row(bad)
    assert ok is False
    assert reason == "CLOSE_TIME_MISMATCH"


def test_build_candidate_rows_rejects_duplicate_timestamp_in_batch():
    klines = {"gap1": [_valid_kline(60_000), _valid_kline(60_000)]}
    built = repair.build_candidate_rows(klines, retrieval_ts_ms=1000)
    assert len(built["accepted"]) == 1
    assert len(built["rejected"]) == 1
    assert built["rejected"][0]["reason"] == "DUPLICATE_TIMESTAMP_IN_BATCH"


def test_build_candidate_rows_accepts_clean_batch():
    klines = {"gap1": [_valid_kline(60_000), _valid_kline(120_000)]}
    built = repair.build_candidate_rows(klines, retrieval_ts_ms=1000)
    assert len(built["accepted"]) == 2
    assert built["rejected"] == []
    for row in built["accepted"]:
        assert row["candle_definition_version"] == repair.CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR
        assert row["data_quality"] == "AVAILABLE"
        assert row["is_closed"] == 1


# ---- disposable rehearsal (real canonical.sqlite copy, no real write) ----

def test_disposable_rehearsal_source_untouched(tmp_path):
    """Proves run_disposable_rehearsal() never writes to the real canonical.sqlite,
    regardless of whatever real repair state that file already carries."""
    import hashlib
    from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH

    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    hash_before = file_hash(REAL_CANONICAL_PATH)
    aligned_base = (1_000_000_000_000 // 60_000) * 60_000
    klines = {"synthetic_gap": [_valid_kline(aligned_base + i * 60_000) for i in range(5)]}
    repair.run_disposable_rehearsal(REAL_CANONICAL_PATH, tmp_path / "disposable.sqlite", klines, retrieval_ts_ms=1000)
    assert file_hash(REAL_CANONICAL_PATH) == hash_before


def _minimal_ami_candles_schema_db(path) -> None:
    """A minimal, deliberately-pristine (no pre-existing repair rows) ami_candles
    table -- avoids entangling this test's rollback/reapply assertions with
    whatever real repair state data/ami/canonical.sqlite happens to carry (that
    file legitimately carries real candle-repair rows post BATCH-CANDLE-GAP-
    REMEDIATION, which would otherwise make a whole-table manifest-equality
    check meaningless -- see the module-level `apply_repair_rows`/
    `rollback_repair` docstrings for why rollback is intentionally NOT
    scoped to a single call's own rows)."""
    import sqlite3
    conn = sqlite3.connect(path)
    conn.executescript("""
CREATE TABLE ami_candles (
    candle_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, venue TEXT, timeframe TEXT NOT NULL,
    open_ts_ms INTEGER NOT NULL, close_ts_ms INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
    low REAL NOT NULL, close REAL NOT NULL, volume REAL, trade_count INTEGER, taker_buy_volume REAL,
    taker_sell_volume REAL, is_closed INTEGER NOT NULL DEFAULT 1, partial_status TEXT NOT NULL DEFAULT 'CLOSED',
    known_at_ts INTEGER NOT NULL, data_quality TEXT NOT NULL, source_hash TEXT,
    candle_definition_version TEXT NOT NULL, schema_version INTEGER NOT NULL, provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL,
    UNIQUE (symbol, timeframe, open_ts_ms, candle_definition_version)
);
""")
    conn.commit()
    conn.close()


def test_disposable_rehearsal_rollback_and_reapply_on_pristine_source(tmp_path):
    """Full rollback/reapply manifest-equality check, against a deliberately
    pristine (no pre-existing repair rows) source -- the scenario
    run_disposable_rehearsal() was originally designed to prove."""
    source_path = tmp_path / "pristine_source.sqlite"
    _minimal_ami_candles_schema_db(source_path)

    aligned_base = (1_000_000_000_000 // 60_000) * 60_000
    klines = {"synthetic_gap": [_valid_kline(aligned_base + i * 60_000) for i in range(5)]}
    report = repair.run_disposable_rehearsal(source_path, tmp_path / "disposable.sqlite", klines, retrieval_ts_ms=1000)

    assert report["accepted_rows"] == 5
    assert report["rejected_rows_by_reason"] == {}
    assert report["conflict_check"]["conflict_n"] == 0
    assert report["run1_rows_written"] == 5
    assert report["deterministic_rerun_manifest_matches"] is True
    assert report["rerun_did_not_duplicate_rows"] is True
    assert report["rollback_result"]["rows_deleted"] == 5
    assert report["rollback_restores_pre_repair_manifest"] is True
    assert report["reapply_matches_post_repair_manifest"] is True


# ---- 5m re-derivation with source-version traceability ----

def _insert_1m_row(conn, open_ts_ms, version, symbol="ETHUSDT", o=100.0, h=101.0, l=99.0, c=100.5, v=10.0):
    import hashlib
    candle_id = "CDL-" + hashlib.sha256(f"{symbol}|1m|{open_ts_ms}|{version}".encode()).hexdigest()[:24]
    conn.execute(
        "INSERT INTO ami_candles (candle_id, symbol, venue, timeframe, open_ts_ms, close_ts_ms, open, high, "
        "low, close, volume, trade_count, taker_buy_volume, taker_sell_volume, is_closed, partial_status, "
        "known_at_ts, data_quality, source_hash, candle_definition_version, schema_version, provenance, "
        "created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (candle_id, symbol, "BINANCE", "1m", open_ts_ms, open_ts_ms + 60_000, o, h, l, c, v, 5, v / 2, v / 2,
         1, "CLOSED", open_ts_ms + 60_000, "AVAILABLE", "hash", version, 5, "test", 1, 1),
    )


def test_rederive_5m_blends_version_when_1m_children_have_mixed_source(tmp_path):
    import sqlite3
    path = tmp_path / "mixed_5m.sqlite"
    _minimal_ami_candles_schema_db(path)
    conn = sqlite3.connect(path)
    bucket_open = 0  # aligned 5m bucket
    # 3 children from the original source, 2 from the repair source -- one 5m bucket, mixed origin
    for i in range(3):
        _insert_1m_row(conn, bucket_open + i * 60_000, "candle-agg_trades-v1")
    for i in range(3, 5):
        _insert_1m_row(conn, bucket_open + i * 60_000, repair.CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR)
    conn.commit()

    result = repair.rederive_5m_with_source_traceability(conn)
    assert result["total_5m_buckets"] == 1
    assert result["version_corrected_n"] == 1  # blended -- differs from derive_higher_timeframe()'s hardcoded default

    row = conn.execute(
        "SELECT candle_definition_version FROM ami_candles WHERE symbol='ETHUSDT' AND timeframe='5m'"
    ).fetchone()
    versions_in_label = set(row[0].split(","))
    assert versions_in_label == {"candle-agg_trades-v1", repair.CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR}
    conn.close()


def test_rederive_5m_keeps_pure_original_version_when_no_repair_children():
    import sqlite3
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        _minimal_ami_candles_schema_db(path)
        conn = sqlite3.connect(path)
        for i in range(5):
            _insert_1m_row(conn, i * 60_000, "candle-agg_trades-v1")
        conn.commit()
        result = repair.rederive_5m_with_source_traceability(conn)
        assert result["version_corrected_n"] == 0  # pure-original bucket needs no correction
        row = conn.execute(
            "SELECT candle_definition_version FROM ami_candles WHERE symbol='ETHUSDT' AND timeframe='5m'"
        ).fetchone()
        assert row[0] == "candle-agg_trades-v1"
        conn.close()
    finally:
        os.remove(path)


def test_rollback_repair_removes_blended_5m_rows_too():
    import sqlite3, tempfile, os
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        _minimal_ami_candles_schema_db(path)
        conn = sqlite3.connect(path)
        for i in range(3):
            _insert_1m_row(conn, i * 60_000, "candle-agg_trades-v1")
        for i in range(3, 5):
            _insert_1m_row(conn, i * 60_000, repair.CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR)
        conn.commit()
        repair.rederive_5m_with_source_traceability(conn)
        pre_5m_n = conn.execute("SELECT COUNT(*) FROM ami_candles WHERE timeframe='5m'").fetchone()[0]
        assert pre_5m_n == 1

        rb = repair.rollback_repair(conn)
        post_5m_n = conn.execute("SELECT COUNT(*) FROM ami_candles WHERE timeframe='5m'").fetchone()[0]
        # the blended 5m bucket references the repair version via LIKE match -- must be removed
        assert post_5m_n == 0
        remaining_1m = conn.execute(
            "SELECT candle_definition_version FROM ami_candles WHERE timeframe='1m' ORDER BY open_ts_ms"
        ).fetchall()
        assert all(v == ("candle-agg_trades-v1",) for v in remaining_1m)  # only original 3 remain
        conn.close()
    finally:
        os.remove(path)
