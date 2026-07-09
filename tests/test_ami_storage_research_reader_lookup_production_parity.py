"""Production smoke/parity for `ami.storage.research_reader.
lookup_latest_at_or_before` (BATCH-STORAGE-ROTATION-RETENTION-POINT-
LOOKUP-HELPER-FOR-ORDER-BY-TS-DESC-LIMIT-1-V1). Read-only bounded
lookups against the real 3 production archive partitions and the real
source database (`data/microstructure.db`, opened strictly `mode=ro`),
compared row-for-row against a direct `ORDER BY ts_ms DESC, id DESC
LIMIT 1` SQL oracle. Skips (does not fail) if the real archive root /
source database are not present in this checkout.
"""
from __future__ import annotations

import os

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC

REAL_ROOT = "D:/eclipse_scalper/data/archives/raw_v1"
REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(REAL_ROOT, PR.ROOT_INDEX_NAME)) and os.path.exists(REAL_SOURCE_DB)),
    reason="real production archive/source database not present in this checkout",
)

TABLE_COLUMNS = {
    "mark_prices": ("id", "ts_ms", "symbol", "mark_price", "funding_rate", "next_funding_time_ms"),
    "agg_trades": ("id", "ts_ms", "symbol", "price", "quantity", "notional", "is_buyer_maker"),
    "book_ticker": ("id", "ts_ms", "symbol", "bid_price", "bid_qty", "ask_price", "ask_qty",
                    "mid_price", "spread_pct", "book_imbalance", "bid_depth_usd"),
}


def _oracle(table: str, symbol: str, ts_ms: int) -> tuple | None:
    cols = TABLE_COLUMNS[table]
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        cols_sql = ",".join(cols)
        row = conn.execute(
            f"SELECT {cols_sql} FROM {table} WHERE symbol=? AND ts_ms<=? "
            f"ORDER BY ts_ms DESC, id DESC LIMIT 1", (symbol, ts_ms)).fetchone()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    return tuple(row) if row else None


def _lookup(table, symbol, ts_ms):
    return RR.lookup_latest_at_or_before(REAL_ROOT, table=table, symbol=symbol, ts_ms=ts_ms,
                                          columns=TABLE_COLUMNS[table], source_db_path=REAL_SOURCE_DB)


# --- mark_prices: archive-only lookup, within the archived May-2026 partition ---

def test_mark_prices_archive_only_lookup_parity():
    ts_ms = 1778500000000
    oracle = _oracle("mark_prices", "ETHUSDT", ts_ms)
    result = _lookup("mark_prices", "ETHUSDT", ts_ms)
    assert oracle is not None
    assert result.found
    assert result.row == oracle
    assert result.provenance["source_type"] == "ARCHIVE_ONLY"
    assert result.provenance["result_source"] == "ARCHIVE"
    assert result.provenance["result_ts_ms"] == oracle[1]


# --- agg_trades: archive-only lookup, within the archived Feb-2026 partition ---

def test_agg_trades_archive_only_lookup_parity():
    ts_ms = 1771200000000
    oracle = _oracle("agg_trades", "ETHUSDT", ts_ms)
    result = _lookup("agg_trades", "ETHUSDT", ts_ms)
    assert oracle is not None
    assert result.found
    assert result.row == oracle
    assert result.provenance["source_type"] == "ARCHIVE_ONLY"
    assert result.provenance["result_source"] == "ARCHIVE"


# --- book_ticker: multi-shard archive lookup, near the real shard-0/shard-1 boundary ---

def test_book_ticker_shard_boundary_lookup_parity():
    for ts_ms in (1776606541718, 1776606541740, 1776606541760):
        oracle = _oracle("book_ticker", "SOLUSDT", ts_ms)
        result = _lookup("book_ticker", "SOLUSDT", ts_ms)
        assert oracle is not None
        assert result.row == oracle, f"mismatch at ts_ms={ts_ms}"
        assert result.provenance["source_type"] == "ARCHIVE_ONLY"


def test_book_ticker_mid_partition_lookup_parity():
    ts_ms = 1776700000000
    oracle = _oracle("book_ticker", "SOLUSDT", ts_ms)
    result = _lookup("book_ticker", "SOLUSDT", ts_ms)
    assert oracle is not None
    assert result.row == oracle
    assert result.provenance["source_type"] == "ARCHIVE_ONLY"


# --- SQLite-only: recent/live range, no archive for this symbol (book_ticker has no ETHUSDT archive) ---

def test_book_ticker_sqlite_only_lookup_parity():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    oracle = _oracle("book_ticker", "ETHUSDT", max_ts)
    result = _lookup("book_ticker", "ETHUSDT", max_ts)
    assert oracle is not None
    assert result.row == oracle
    assert result.provenance["source_type"] == "SQLITE_ONLY"
    assert result.provenance["result_source"] == "SQLITE"


# --- Hybrid boundary: mark_prices query just after the archived-May/live-June boundary ---

def test_mark_prices_hybrid_boundary_lookup_parity():
    ts_ms = 1780272060000
    oracle = _oracle("mark_prices", "ETHUSDT", ts_ms)
    result = _lookup("mark_prices", "ETHUSDT", ts_ms)
    assert oracle is not None
    assert result.row == oracle
    # this ts is naturally SQLite's turf (after the archive ends) and SQLite
    # has live data there already, so no cross-source fallback is needed --
    # SQLITE_ONLY is the correct, minimal-consultation outcome, not HYBRID.
    assert result.provenance["source_type"] == "SQLITE_ONLY"
    assert result.provenance["result_source"] == "SQLITE"


# --- Read-only invariants ---

def test_lookup_leaves_catalog_and_manifest_unchanged():
    idx_path = os.path.join(REAL_ROOT, PR.ROOT_INDEX_NAME)
    manifest_path = os.path.join(
        REAL_ROOT, "table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
        "symbol=ETHUSDT", "year=2026", "month=05", "version=v1", PR.MANIFEST_NAME)
    idx_before, manifest_before = os.path.getmtime(idx_path), os.path.getmtime(manifest_path)
    _lookup("mark_prices", "ETHUSDT", 1778500000000)
    idx_after, manifest_after = os.path.getmtime(idx_path), os.path.getmtime(manifest_path)
    assert idx_before == idx_after
    assert manifest_before == manifest_after
