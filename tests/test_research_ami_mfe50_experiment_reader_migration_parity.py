"""Pilot migration parity proof (Task #28 of BATCH-STORAGE-ROTATION-
RETENTION-RESEARCH-READER-INTEGRATION-V1).

`tools/research_ami_mfe50_experiment.py` is the bounded pilot consumer:
its two ad-hoc direct-SQL bounded-range aggregates in `feats_at_hit`
(agg_trades window notional sum, book_ticker window bid_qty average)
were migrated to route through `ami.storage.research_reader` instead.
This test proves the migrated helpers produce IDENTICAL results to the
original direct-SQL queries they replaced, across:
  * a recent live window (SQLITE_ONLY -- the actual access pattern real
    usage of this script always hits, since anchors are near "now")
  * a window fully inside the archived agg_trades/ETHUSDT/2026-02
    partition (ARCHIVE_ONLY -- proves the migration is also correct for
    data this consumer would touch if it were ever run against older
    anchors)
  * a window straddling the archive/live boundary (HYBRID)

book_ticker has no ETHUSDT archive partition (only SOLUSDT is
archived), so its case only exercises SQLITE_ONLY -- documented here
rather than silently only testing one table's hybrid path.

Skips (does not fail) if the real source database is not present.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage import source_access as SRC
from tools.research_ami_mfe50_experiment import window_agg_trades_notional, window_avg_book_ticker_bid_qty

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_SOURCE_DB), reason="real source database not present in this checkout")


def _old_agg_trades_notional(symbol: str, start_ms: int, end_ms: int) -> tuple[float, float]:
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        row = conn.execute(
            "SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END), SUM(notional) "
            "FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?", (symbol, start_ms, end_ms)).fetchone()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    return float(row[0] or 0), float(row[1] or 0)


def _old_avg_book_ticker_bid_qty(symbol: str, start_ms: int, end_ms: int) -> float | None:
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        row = conn.execute(
            "SELECT AVG(bid_qty) FROM book_ticker WHERE symbol=? AND ts_ms>=? AND ts_ms<?",
            (symbol, start_ms, end_ms)).fetchone()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    return float(row[0]) if row and row[0] is not None else None


def _root():
    root, _source = PR.resolve_production_root()
    return root


# --- agg_trades: recent live window (real access pattern) ---

def test_agg_trades_notional_parity_recent_live_window():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM agg_trades WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    start_ms, end_ms = max_ts - 600_000, max_ts
    old_buy, old_tot = _old_agg_trades_notional("ETHUSDT", start_ms, end_ms)
    new_buy, new_tot = window_agg_trades_notional(_root(), "ETHUSDT", start_ms, end_ms)
    assert new_tot > 0  # sanity: window actually has data
    assert new_buy == pytest.approx(old_buy, rel=1e-9)
    assert new_tot == pytest.approx(old_tot, rel=1e-9)


# --- agg_trades: fully inside the archived Feb 2026 partition ---

AGG_TRADES_ARCHIVE_START_MS = 1771165588000
AGG_TRADES_ARCHIVE_END_MS = 1771165598000


def test_agg_trades_notional_parity_archive_only_window():
    old_buy, old_tot = _old_agg_trades_notional(
        "ETHUSDT", AGG_TRADES_ARCHIVE_START_MS, AGG_TRADES_ARCHIVE_END_MS)
    new_buy, new_tot = window_agg_trades_notional(
        _root(), "ETHUSDT", AGG_TRADES_ARCHIVE_START_MS, AGG_TRADES_ARCHIVE_END_MS)
    assert new_tot > 0
    assert new_buy == pytest.approx(old_buy, rel=1e-9)
    assert new_tot == pytest.approx(old_tot, rel=1e-9)


# --- agg_trades: straddles the archive/live boundary ---

AGG_TRADES_HYBRID_START_MS = 1772323195000
AGG_TRADES_HYBRID_END_MS = 1772323205000


def test_agg_trades_notional_parity_hybrid_window():
    old_buy, old_tot = _old_agg_trades_notional(
        "ETHUSDT", AGG_TRADES_HYBRID_START_MS, AGG_TRADES_HYBRID_END_MS)
    new_buy, new_tot = window_agg_trades_notional(
        _root(), "ETHUSDT", AGG_TRADES_HYBRID_START_MS, AGG_TRADES_HYBRID_END_MS)
    assert new_tot > 0
    assert new_buy == pytest.approx(old_buy, rel=1e-9)
    assert new_tot == pytest.approx(old_tot, rel=1e-9)


# --- book_ticker: recent live window (ETHUSDT has no archive partition -- SQLITE_ONLY only) ---

def test_book_ticker_avg_bid_qty_parity_recent_live_window():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    start_ms, end_ms = max_ts - 600_000, max_ts
    old_avg = _old_avg_book_ticker_bid_qty("ETHUSDT", start_ms, end_ms)
    new_avg = window_avg_book_ticker_bid_qty(_root(), "ETHUSDT", start_ms, end_ms)
    assert old_avg is not None
    assert new_avg == pytest.approx(old_avg, rel=1e-9)
