"""Acceptance tests (Phases 20-21): reproduces the accepted mark_prices
disposable rehearsal (commit `6fbe0571`) using the bounded-implementation
pipeline against the REAL, live `microstructure.db` (read-only), plus
minimal deterministic fixtures for `agg_trades` and `book_ticker`
covering the required scenario matrix.

All real-database access is bounded, indexed, and read-only. Disposable
outputs are written only under `.runtime_temp/` and cleaned up (or kept
tiny) by each test via `tmp_path`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3

import pytest

from ami.storage import archive as A
from ami.storage import partition as PT
from ami.storage import source_access as SRC
from ami.storage import verifier as V
from ami.storage.registry import get_table_spec

# Frozen accepted result from commit 6fbe0571 (the disposable dry-run).
ACCEPTED_ROW_COUNT = 260657
ACCEPTED_WATERMARK = 13265132
ACCEPTED_SCIENTIFIC_HASH = "228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999"
ACCEPTED_PARQUET_SHA256 = "6f91914400dcbe84b662c9260a24f9e5eb7f56b2d9db34adfde55a53af8e900f"


# ---------------------------------------------------------------------------
# mark_prices: live reproduction
# ---------------------------------------------------------------------------

def test_mark_prices_live_reproduction_matches_accepted_dry_run(tmp_path):
    """Read-only against the real microstructure.db. Reproduces the
    exact accepted partition; any difference is reported, never forced."""
    conn, log = SRC.open_read_only()
    try:
        plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5)
        assert plan.estimated_row_count == ACCEPTED_ROW_COUNT, (
            f"row population drifted: {plan.estimated_row_count} != {ACCEPTED_ROW_COUNT} "
            "(explained, not forced: repository/database state may have changed since acceptance)")
        assert plan.partition.source_watermark_value == ACCEPTED_WATERMARK
        assert plan.archive_rehearsal_eligible is True
        assert plan.purge_eligible is False

        output_root = str(tmp_path / "mark_prices_acceptance")
        result = A.export_partition(conn, "mark_prices", plan.partition, output_root,
                                    allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()

    assert result["row_count"] == ACCEPTED_ROW_COUNT
    assert result["scientific_content_hash"] == ACCEPTED_SCIENTIFIC_HASH
    assert result["parquet_sha256"] == ACCEPTED_PARQUET_SHA256, (
        "Parquet file bytes differ from the accepted dry-run -- if this is due to an "
        "intentional writer/library configuration change, the difference must be explained "
        "at the schema/scientific-content/row-accounting level, not silently forced to match. "
        "Scientific-content hash equality (asserted above) is the actual acceptance bar.")


def test_mark_prices_zero_source_mutation_during_acceptance(tmp_path):
    conn, log = SRC.open_read_only()
    try:
        plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5)
        output_root = str(tmp_path / "mark_prices_mutation_check")
        A.export_partition(conn, "mark_prices", plan.partition, output_root,
                           allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    finally:
        conn.close()
    assert log == []  # zero write attempts issued, zero denied


def test_mark_prices_zero_production_archive_created(tmp_path):
    conn, log = SRC.open_read_only()
    try:
        plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5)
        output_root = str(tmp_path / "mark_prices_no_prod")
        result = A.export_partition(conn, "mark_prices", plan.partition, output_root,
                                    allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    finally:
        conn.close()
    import os
    assert output_root in os.path.abspath(result["final_path"])
    assert "data/ami/backups" not in result["final_path"].replace("\\", "/")


# ---------------------------------------------------------------------------
# agg_trades: minimal deterministic fixture, full scenario matrix
# ---------------------------------------------------------------------------

def _agg_trades_fixture():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE agg_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
        price REAL NOT NULL, quantity REAL NOT NULL, notional REAL NOT NULL, is_buyer_maker INTEGER NOT NULL)""")
    conn.execute("CREATE INDEX idx_trade_ts ON agg_trades(ts_ms)")
    conn.execute("CREATE INDEX idx_trade_symbol_ts ON agg_trades(symbol, ts_ms)")
    conn.execute("CREATE TABLE gaps (id INTEGER PRIMARY KEY, stream TEXT, start_ts_ms INTEGER, "
                 "end_ts_ms INTEGER, resolved_bool INTEGER)")
    may_start, may_end = 1777593600000, 1780272000000
    rows = [
        (may_start, "ETHUSDT", 3000.0, 1.0, 3000.0, 0),           # exactly at start
        (may_end - 1, "ETHUSDT", 3005.0, 1.5, 4507.5, 1),         # exactly before end
        (may_start + 60000, "ETHUSDT", 3001.0, 2.0, 6002.0, 0),   # ordinary
        (may_start + 90000, "BTCUSDT", 60000.0, 0.1, 6000.0, 1),  # different symbol
    ]
    conn.executemany("INSERT INTO agg_trades (ts_ms,symbol,price,quantity,notional,is_buyer_maker) "
                     "VALUES (?,?,?,?,?,?)", rows)
    # row exactly AT end (excluded) and a row above what will become the watermark
    conn.execute("INSERT INTO agg_trades (ts_ms,symbol,price,quantity,notional,is_buyer_maker) "
                 "VALUES (?,?,?,?,?,?)", (may_end, "ETHUSDT", 3010.0, 1.0, 3010.0, 0))
    conn.commit()
    return conn


def test_agg_trades_fixture_row_exactly_at_start_included():
    conn = _agg_trades_fixture()
    spec = get_table_spec("agg_trades")
    partition = PT.build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=999, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    rows = A.fetch_partition_rows(conn, spec, partition)
    ts_values = [r[1] for r in rows]
    assert partition.partition_start_ms in ts_values
    conn.close()


def test_agg_trades_fixture_row_exactly_at_end_excluded():
    conn = _agg_trades_fixture()
    spec = get_table_spec("agg_trades")
    partition = PT.build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=999, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    rows = A.fetch_partition_rows(conn, spec, partition)
    ts_values = [r[1] for r in rows]
    assert partition.partition_end_ms not in ts_values  # half-open: end is excluded
    conn.close()


def test_agg_trades_fixture_wrong_symbol_excluded():
    conn = _agg_trades_fixture()
    spec = get_table_spec("agg_trades")
    partition = PT.build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=999, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    rows = A.fetch_partition_rows(conn, spec, partition)
    symbols = {r[2] for r in rows}
    assert symbols == {"ETHUSDT"}
    conn.close()


def test_agg_trades_fixture_rows_above_watermark_excluded():
    conn = _agg_trades_fixture()
    spec = get_table_spec("agg_trades")
    # freeze watermark at the 3rd inserted row's id (id=3)
    partition = PT.build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=3, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    rows = A.fetch_partition_rows(conn, spec, partition)
    ids = [r[0] for r in rows]
    assert all(i <= 3 for i in ids)
    conn.close()


def test_agg_trades_fixture_active_horizon_blocker():
    with pytest.raises(PT.PartitionValidationError, match="active retention horizon"):
        PT.build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=6,
                                    source_watermark_value=1, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))


def test_agg_trades_fixture_current_month_blocker():
    with pytest.raises(PT.PartitionValidationError, match="current UTC month"):
        PT.build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=7,
                                    source_watermark_value=1, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))


def test_agg_trades_fixture_export_and_verify(tmp_path):
    conn = _agg_trades_fixture()
    partition = PT.build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=3, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    output_root = str(tmp_path / "agg_trades_out")
    result = A.export_partition(conn, "agg_trades", partition, output_root,
                                allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    assert result["row_count"] == 3  # ids 1,2,3 are ETHUSDT within window at-or-below watermark
    conn.close()


def test_agg_trades_boolean_column_int64_not_narrowed(tmp_path):
    """is_buyer_maker is stored as SQLite INTEGER (0/1) -- must not be
    silently coerced to a boolean/float type in the archive."""
    conn = _agg_trades_fixture()
    partition = PT.build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=3, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    output_root = str(tmp_path / "agg_trades_bool")
    result = A.export_partition(conn, "agg_trades", partition, output_root,
                                allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    import pyarrow.parquet as pq
    schema = pq.read_schema(result["final_path"])
    assert str(schema.field("is_buyer_maker").type) == "int64"
    conn.close()


# ---------------------------------------------------------------------------
# book_ticker: minimal deterministic fixture, nullable column + duplicate scenario
# ---------------------------------------------------------------------------

def _book_ticker_fixture():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE book_ticker (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
        bid_price REAL NOT NULL, bid_qty REAL NOT NULL, ask_price REAL NOT NULL, ask_qty REAL NOT NULL,
        mid_price REAL NOT NULL, spread_pct REAL NOT NULL, book_imbalance REAL NOT NULL, bid_depth_usd REAL)""")
    conn.execute("CREATE INDEX idx_bt_symbol_ts ON book_ticker(symbol, ts_ms)")
    conn.execute("CREATE INDEX idx_bt_ts ON book_ticker(ts_ms)")
    may_start = 1777593600000
    rows = [
        (may_start, "ETHUSDT", 2999.5, 1.0, 3000.5, 1.0, 3000.0, 0.033, 0.0, 50000.0),
        (may_start + 1000, "ETHUSDT", 2999.6, 1.1, 3000.4, 0.9, 3000.0, 0.026, 0.05, None),  # nullable None
    ]
    conn.executemany("INSERT INTO book_ticker (ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,"
                     "spread_pct,book_imbalance,bid_depth_usd) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return conn


def test_book_ticker_fixture_nullable_bid_depth_preserved(tmp_path):
    conn = _book_ticker_fixture()
    partition = PT.build_partition_identity(table="book_ticker", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=2, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    output_root = str(tmp_path / "book_ticker_out")
    result = A.export_partition(conn, "book_ticker", partition, output_root,
                                allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    import pyarrow.parquet as pq
    table = pq.read_table(result["final_path"])
    bid_depth = table.to_pydict()["bid_depth_usd"]
    assert None in bid_depth
    assert 50000.0 in bid_depth
    conn.close()


def test_book_ticker_fixture_no_duplicate_ids(tmp_path):
    conn = _book_ticker_fixture()
    partition = PT.build_partition_identity(table="book_ticker", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=2, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    spec = get_table_spec("book_ticker")
    rows = A.fetch_partition_rows(conn, spec, partition)
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids))
    conn.close()


def test_book_ticker_fixture_scientific_content_hash_stable(tmp_path):
    conn = _book_ticker_fixture()
    partition = PT.build_partition_identity(table="book_ticker", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=2, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    spec = get_table_spec("book_ticker")
    rows_a = A.fetch_partition_rows(conn, spec, partition)
    rows_b = A.fetch_partition_rows(conn, spec, partition)
    assert A.canonical_row_hash(rows_a) == A.canonical_row_hash(rows_b)
    conn.close()


def test_book_ticker_source_schema_mismatch_detected():
    """A source table missing an expected registry column should not be
    silently exported -- fetch_partition_rows references
    spec.preserved_columns directly, so a genuinely missing column raises
    a plain sqlite3.OperationalError (no silent column-skip)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE book_ticker (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT)")  # missing columns
    conn.execute("CREATE INDEX idx_bt_symbol_ts ON book_ticker(symbol, ts_ms)")
    conn.execute("CREATE INDEX idx_bt_ts ON book_ticker(ts_ms)")
    partition = PT.build_partition_identity(table="book_ticker", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                                            source_watermark_value=1, now=dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc))
    spec = get_table_spec("book_ticker")
    with pytest.raises(sqlite3.OperationalError):
        A.fetch_partition_rows(conn, spec, partition)
    conn.close()
