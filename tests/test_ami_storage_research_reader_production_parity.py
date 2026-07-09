"""Production smoke/parity tests for `ami.storage.research_reader` (Task
#27 of BATCH-STORAGE-ROTATION-RETENTION-RESEARCH-READER-INTEGRATION-V1).

Reads small, bounded, real time windows from the ACTUAL production
archive (`data/archives/raw_v1`, 3 published partitions) and the ACTUAL
source database (`data/microstructure.db`, opened strictly `mode=ro`),
via the reader's public two-phase API, and compares against a direct
read-only SQLite/parquet reference query for the same window. This is a
parity smoke test, not the full scientific reverify (that stays behind
`ami.storage.reverify_guard.run_guarded_reverify`, invoked separately
and already exercised in `test_ami_storage_reverify_hardening.py`).

Windows below were discovered and confirmed non-empty via read-only
diagnostic queries against the real database before being hardcoded
here (see BATCH-STORAGE-ROTATION-RETENTION-RESEARCH-READER-INTEGRATION-V1
session notes). Skips (does not fail) if the real archive root or
source database is not present, e.g. in an AMI-only checkout without
the full data estate.
"""
from __future__ import annotations

import os
import sqlite3

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


def _direct_sqlite_rows(table: str, symbol: str, start_ms: int, end_ms: int, columns: tuple[str, ...]) -> list[tuple]:
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        cols_sql = ",".join(columns)
        cur = conn.execute(
            f"SELECT {cols_sql} FROM {table} WHERE symbol=? AND ts_ms>=? AND ts_ms<? "
            f"ORDER BY ts_ms ASC, id ASC",
            (symbol, start_ms, end_ms))
        rows = cur.fetchall()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    return rows


def _direct_parquet_rows(final_dir: str, manifest: dict, symbol: str, start_ms: int, end_ms: int,
                          columns: tuple[str, ...]) -> list[tuple]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    all_shard_paths = RR._manifest_shard_paths(final_dir, manifest)
    _, shard_paths = RR._select_overlapping_shards(manifest, tuple(all_shard_paths), start_ms, end_ms)
    out: list[tuple] = []
    fetch_cols = columns if "ts_ms" in columns else columns + ("ts_ms",)
    for path in shard_paths:
        table_data = pq.ParquetFile(path).read(columns=list(fetch_cols))
        mask = pc.and_(pc.greater_equal(table_data.column("ts_ms"), start_ms),
                       pc.less(table_data.column("ts_ms"), end_ms))
        filtered = table_data.filter(mask)
        out.extend(zip(*[filtered.column(c).to_pylist() for c in columns]))
    out.sort(key=lambda r: (r[columns.index("ts_ms")], r[columns.index("id")]))
    return out


def _read_via_reader(table: str, symbol: str, start_ms: int, end_ms: int,
                      columns: tuple[str, ...]) -> tuple[list[tuple], dict, str]:
    plan = RR.plan_read(REAL_ROOT, table=table, symbol=symbol, start_ms=start_ms, end_ms=end_ms)
    result = RR.execute_read(plan, columns=columns, source_db_path=REAL_SOURCE_DB)
    rows: list[tuple] = []
    for batch in result.iter_batches():
        rows.extend(batch)
    return rows, result.provenance, plan.mode


# --- mark_prices: HYBRID window straddling the archived-May/live-June boundary ---

MARK_PRICES_HYBRID_START_MS = 1780264800000  # 2026-05-31T22:00:00Z
MARK_PRICES_HYBRID_END_MS = 1780279200000  # 2026-06-01T02:00:00Z


def test_mark_prices_hybrid_window_parity():
    columns = ("id", "ts_ms", "symbol", "mark_price")
    rows, provenance, mode = _read_via_reader("mark_prices", "ETHUSDT",
                                               MARK_PRICES_HYBRID_START_MS, MARK_PRICES_HYBRID_END_MS, columns)
    assert mode == "HYBRID"
    reference = _direct_sqlite_rows("mark_prices", "ETHUSDT",
                                     MARK_PRICES_HYBRID_START_MS, MARK_PRICES_HYBRID_END_MS, columns)
    assert len(rows) == 787
    assert rows == reference
    assert provenance["row_count"] == 787
    assert provenance["source_type"] == "HYBRID"
    assert len(provenance["archive_segments"]) == 1
    assert len(provenance["sqlite_ranges"]) == 1


# --- book_ticker: multi-shard ARCHIVE_ONLY window crossing the real shard-0/shard-1 boundary ---

BOOK_TICKER_MULTISHARD_START_MS = 1776606536718
BOOK_TICKER_MULTISHARD_END_MS = 1776606546760


def test_book_ticker_multishard_window_parity():
    columns = ("id", "ts_ms", "symbol", "bid_price", "ask_price")
    rows, provenance, mode = _read_via_reader("book_ticker", "SOLUSDT",
                                               BOOK_TICKER_MULTISHARD_START_MS, BOOK_TICKER_MULTISHARD_END_MS, columns)
    assert mode == "ARCHIVE_ONLY"
    reference = _direct_sqlite_rows("book_ticker", "SOLUSDT",
                                     BOOK_TICKER_MULTISHARD_START_MS, BOOK_TICKER_MULTISHARD_END_MS, columns)
    assert len(rows) == 2442
    assert rows == reference
    assert provenance["row_count"] == 2442
    seg = provenance["archive_segments"][0]
    assert seg["shard_count"] == 12  # full manifest shard set is always opened; boundary crossed within it


# --- book_ticker: HYBRID window straddling the archived-April/live-May boundary ---

BOOK_TICKER_HYBRID_START_MS = 1777593599000
BOOK_TICKER_HYBRID_END_MS = 1777593601000


def test_book_ticker_hybrid_window_parity():
    columns = ("id", "ts_ms", "symbol", "mid_price")
    rows, provenance, mode = _read_via_reader("book_ticker", "SOLUSDT",
                                               BOOK_TICKER_HYBRID_START_MS, BOOK_TICKER_HYBRID_END_MS, columns)
    assert mode == "HYBRID"
    reference = _direct_sqlite_rows("book_ticker", "SOLUSDT",
                                     BOOK_TICKER_HYBRID_START_MS, BOOK_TICKER_HYBRID_END_MS, columns)
    assert len(rows) == 367
    assert rows == reference
    assert provenance["row_count"] == 367


# --- agg_trades: ARCHIVE_ONLY window shortly after real data collection began
# (declared partition starts 2026-02-01T00:00:00Z but real ETHUSDT agg_trades
# rows only begin 2026-02-15T14:26:27.967Z -- confirmed via read-only MIN(ts_ms)
# diagnostic query against the real database before hardcoding this window) ---

AGG_TRADES_ARCHIVE_START_MS = 1771165588000
AGG_TRADES_ARCHIVE_END_MS = 1771165598000


def test_agg_trades_archive_only_window_parity():
    columns = ("id", "ts_ms", "symbol", "price", "quantity")
    rows, provenance, mode = _read_via_reader("agg_trades", "ETHUSDT",
                                               AGG_TRADES_ARCHIVE_START_MS, AGG_TRADES_ARCHIVE_END_MS, columns)
    assert mode == "ARCHIVE_ONLY"
    reference = _direct_sqlite_rows("agg_trades", "ETHUSDT",
                                     AGG_TRADES_ARCHIVE_START_MS, AGG_TRADES_ARCHIVE_END_MS, columns)
    assert len(rows) == 249
    assert rows == reference
    assert provenance["row_count"] == 249


# --- agg_trades: HYBRID window straddling the archived-Feb/live-March boundary ---

AGG_TRADES_HYBRID_START_MS = 1772323195000
AGG_TRADES_HYBRID_END_MS = 1772323205000


def test_agg_trades_hybrid_window_parity():
    columns = ("id", "ts_ms", "symbol", "price", "quantity")
    rows, provenance, mode = _read_via_reader("agg_trades", "ETHUSDT",
                                               AGG_TRADES_HYBRID_START_MS, AGG_TRADES_HYBRID_END_MS, columns)
    assert mode == "HYBRID"
    reference = _direct_sqlite_rows("agg_trades", "ETHUSDT",
                                     AGG_TRADES_HYBRID_START_MS, AGG_TRADES_HYBRID_END_MS, columns)
    assert len(rows) == 241
    assert rows == reference
    assert provenance["row_count"] == 241


# --- Direct parquet reference cross-check for the multi-shard window
# (guards against a bug where reader + direct-sqlite reference agree by
# coincidence, e.g. both silently reading only from SQLite) ---

def test_book_ticker_multishard_window_matches_direct_parquet_reference():
    entries = RR._root_catalog_entries(REAL_ROOT)
    entry = next(e for e in entries if e["source_table"] == "book_ticker" and e["symbol"] == "SOLUSDT")
    final_dir = os.path.join(REAL_ROOT, entry["archive_relative_path"])
    manifest, _, _ = RR._verify_archive_trust(REAL_ROOT, entry)
    columns = ("id", "ts_ms", "symbol", "bid_price", "ask_price")
    parquet_reference = _direct_parquet_rows(final_dir, manifest, "SOLUSDT",
                                              BOOK_TICKER_MULTISHARD_START_MS, BOOK_TICKER_MULTISHARD_END_MS, columns)
    rows, _, mode = _read_via_reader("book_ticker", "SOLUSDT",
                                      BOOK_TICKER_MULTISHARD_START_MS, BOOK_TICKER_MULTISHARD_END_MS, columns)
    assert mode == "ARCHIVE_ONLY"
    assert rows == parquet_reference


# --- Read-only invariants: catalog/manifest/source untouched by any of the above reads ---

def test_reads_leave_catalog_and_manifests_unchanged():
    idx_path = os.path.join(REAL_ROOT, PR.ROOT_INDEX_NAME)
    manifest_path = os.path.join(
        REAL_ROOT, "table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
        "symbol=ETHUSDT", "year=2026", "month=05", "version=v1", PR.MANIFEST_NAME)
    assert os.path.exists(manifest_path)
    idx_before = os.path.getmtime(idx_path)
    manifest_before = os.path.getmtime(manifest_path)
    result = RR.execute_read(
        RR.plan_read(REAL_ROOT, table="mark_prices", symbol="ETHUSDT",
                     start_ms=MARK_PRICES_HYBRID_START_MS, end_ms=MARK_PRICES_HYBRID_END_MS),
        source_db_path=REAL_SOURCE_DB,
    )
    consumed = list(result.iter_rows())
    assert len(consumed) == 787
    idx_after = os.path.getmtime(idx_path)
    manifest_after = os.path.getmtime(manifest_path)
    assert idx_before == idx_after
    assert manifest_before == manifest_after
