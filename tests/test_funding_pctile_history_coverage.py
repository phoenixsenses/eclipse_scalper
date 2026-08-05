"""`funding_pctile_14d` must be invalid when the estate cannot back 14 days.

Section 265 measured "how much history does a standing role need?" by grepping each
role's own *_LOOKBACK_MS constants. This window lived two imports away, inside an
expression, and was missed -- the answer it produced (7 days) was wrong by a factor
of two, and the live DB holds less than 14 days today. Deleting the frozen segment
would have turned a 14-day percentile into a 12-day one, still labelled 14d, in two
standing roles.

SQL will not complain about that, so the coverage check has to.
"""

from __future__ import annotations

import sqlite3

import pytest

from tools import liq_indicator_library as LIB


DAY = 86_400_000
NOW = 1_785_000_000_000


def _db(path, earliest_ms):
    """mark_prices with funding, reaching back to `earliest_ms` at 6h cadence."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, "
        "symbol TEXT NOT NULL, mark_price REAL NOT NULL, funding_rate REAL, "
        "next_funding_time_ms INTEGER)"
    )
    conn.execute("CREATE INDEX idx_mark_ts ON mark_prices(ts_ms)")
    conn.execute("CREATE INDEX idx_mark_symbol_ts ON mark_prices(symbol, ts_ms)")
    rows, ts, i = [], earliest_ms, 0
    while ts <= NOW:
        rows.append((ts, "ETHUSDT", 3000.0 + i, 0.0001 * (i % 7)))
        ts += 6 * 3_600_000
        i += 1
    rows.append((NOW, "ETHUSDT", 3100.0, 0.0004))  # fresh funding at as_of
    conn.executemany(
        "INSERT INTO mark_prices(ts_ms, symbol, mark_price, funding_rate) VALUES (?,?,?,?)", rows)
    # the other tables compute_indicators touches; empty, but with the real column
    # names so the vector degrades gracefully instead of erroring
    conn.execute("CREATE TABLE vol_state (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
                 "rv_5m REAL, vol_decile INTEGER, high_vol_alert INTEGER)")
    conn.execute("CREATE TABLE open_interest (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
                 "open_interest_usd REAL)")
    conn.execute("CREATE TABLE spot_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
                 "spot_price REAL)")
    conn.execute("CREATE TABLE book_ticker (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
                 "spread_pct REAL, book_imbalance REAL, bid_depth_usd REAL)")
    conn.execute("CREATE TABLE liquidations (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
                 "side TEXT, notional REAL)")
    conn.execute("CREATE TABLE agg_trades (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
                 "price REAL, quantity REAL, notional REAL, is_buyer_maker INTEGER)")
    conn.commit()
    return conn


def _pctile(conn):
    ind = LIB.compute_indicators(conn, NOW)
    return ind.values.get("funding_pctile_14d"), ind.fresh.get("funding_pctile_14d")


def test_deep_enough_history_produces_a_valid_percentile(tmp_path):
    conn = _db(tmp_path / "deep.db", NOW - 30 * DAY)
    try:
        value, fresh = _pctile(conn)
        assert value is not None
        assert fresh is True
    finally:
        conn.close()


def test_shallow_history_marks_the_percentile_invalid(tmp_path):
    """The exact Phase-4 shape: 12 days of data behind a 14-day window."""
    conn = _db(tmp_path / "shallow.db", NOW - 12 * DAY)
    try:
        value, fresh = _pctile(conn)
        assert value is None
        assert fresh is False
    finally:
        conn.close()


def test_the_boundary_is_inclusive(tmp_path):
    conn = _db(tmp_path / "exact.db", NOW - LIB.FUNDING_PCTILE_WINDOW_MS)
    try:
        _, fresh = _pctile(conn)
        assert fresh is True
    finally:
        conn.close()


def test_one_hour_short_is_already_invalid(tmp_path):
    """No silent grace band -- 13d23h is not 14d."""
    conn = _db(tmp_path / "near.db", NOW - LIB.FUNDING_PCTILE_WINDOW_MS + 3_600_000)
    try:
        _, fresh = _pctile(conn)
        assert fresh is False
    finally:
        conn.close()


def test_a_shallow_estate_does_not_break_the_other_indicators(tmp_path):
    """Degrading one indicator must not take the vector down with it."""
    conn = _db(tmp_path / "shallow2.db", NOW - 12 * DAY)
    try:
        ind = LIB.compute_indicators(conn, NOW)
        assert ind.values.get("funding_rate") is not None
        assert ind.fresh.get("funding_rate") is True
        assert ind.values.get("mark_price") is not None
    finally:
        conn.close()


def test_the_window_is_a_named_constant(tmp_path):
    """It was invisible to a repo-wide audit because it was inlined in an expression."""
    assert LIB.FUNDING_PCTILE_WINDOW_MS == 14 * DAY
    src = (LIB.__file__)
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    assert "14 * 86_400_000" not in body.split("FUNDING_PCTILE_WINDOW_MS =")[-1].split("\n", 1)[-1]
