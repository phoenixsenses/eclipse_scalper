"""The canonical dashboard's funding probe: same answer, without the union blow-up.

`SELECT funding_rate ... ORDER BY ts_ms DESC LIMIT 1` is the shape the union_reader
docstring measures at 7560x: SQLite merges an ordered compound only when the ORDER BY
term is a RESULT column, and this one selects funding_rate alone, so the whole
compound is materialised and sorted to return a single row. The section 258/259 sweep
fixed the shadow runner's copy of the probe and missed the dashboard's.

The answer must not change, so these tests compare the new implementation against the
literal old query across the cutoff.
"""

from __future__ import annotations

import sqlite3

import pytest

from ami.storage import union_reader as UR
from tools import s34_cascade_navigation_dashboard as D


_SCHEMA = (
    "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, "
    "symbol TEXT NOT NULL, mark_price REAL NOT NULL, funding_rate REAL, "
    "next_funding_time_ms INTEGER)"
)
HOUR = 3_600_000


def _mk(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute(_SCHEMA)
    conn.execute("CREATE INDEX idx_mark_symbol_ts ON mark_prices(symbol, ts_ms)")
    conn.executemany(
        "INSERT INTO mark_prices(ts_ms, symbol, mark_price, funding_rate) VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _rotated(tmp_path, live_rows, frozen_rows, cutoff):
    live, frozen = tmp_path / "live.db", tmp_path / "frozen0.db"
    _mk(live, live_rows)
    _mk(frozen, frozen_rows)
    return UR.RotationState(
        live_db_path=str(live), cutoff_ms=cutoff,
        frozen_segments=(UR.FrozenSegment(path=str(frozen), start_ms=None, end_ms=cutoff),))


def _old_query(conn, symbol, ts_ms):
    """The shape being replaced, kept as the oracle for 'same answer'."""
    row = conn.execute(
        "SELECT funding_rate FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


BASE = 1_800_000_000_000


@pytest.fixture()
def conn(tmp_path):
    state = _rotated(
        tmp_path,
        live_rows=[(BASE, "ETHUSDT", 3000.0, 0.0004),
                   (BASE + 2 * HOUR, "ETHUSDT", 3010.0, None),
                   (BASE + 8 * HOUR, "ETHUSDT", 3020.0, 0.0005)],
        frozen_rows=[(BASE - 16 * HOUR, "ETHUSDT", 2900.0, 0.0001),
                     (BASE - 8 * HOUR, "ETHUSDT", 2950.0, 0.0002)],
        cutoff=BASE,
    )
    c = UR.open_union_ro(state)
    yield c
    c.close()


@pytest.mark.parametrize("offset_h", [0, 1, 3, 9, -1, -9, -17])
def test_matches_the_query_it_replaces_on_both_sides_of_the_cutoff(conn, offset_h):
    ts = BASE + offset_h * HOUR
    assert D.funding_rate_at(conn, "ETHUSDT", ts) == _old_query(conn, "ETHUSDT", ts)


def test_skips_rows_where_funding_is_null(conn):
    """At BASE+2h the newest row has NULL funding; the answer is the 0.0004 before it."""
    assert D.funding_rate_at(conn, "ETHUSDT", BASE + 2 * HOUR) == pytest.approx(0.0004)


def test_reads_across_the_cutoff_into_the_frozen_segment(conn):
    assert D.funding_rate_at(conn, "ETHUSDT", BASE - HOUR) == pytest.approx(0.0002)


def test_never_issues_the_unqualified_compound_form(conn):
    """The defect is the query SHAPE, so assert on the SQL the function emits."""
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    try:
        D.funding_rate_at(conn, "ETHUSDT", BASE + HOUR)
    finally:
        conn.set_trace_callback(None)

    reads = [s for s in seen if "FUNDING_RATE" in s.upper() and "SELECT" in s.upper()]
    assert reads, "no funding query was issued"
    for sql in reads:
        assert "main.mark_prices" in sql or "frozen0.mark_prices" in sql, (
            f"unqualified read of the union view: {sql!r}")
        assert "TS_MS" in sql.upper().split("FROM")[0], (
            "ts_ms must be a RESULT column or the compound cannot merge")


def test_a_dead_source_returns_none_rather_than_a_months_old_value(conn):
    """Funding prints ~8h; nothing within a day means the feed is gone, not flat."""
    assert D.funding_rate_at(conn, "ETHUSDT", BASE + 40 * HOUR) is None
    # ...while the unbounded form it replaced would have happily returned one
    assert _old_query(conn, "ETHUSDT", BASE + 40 * HOUR) == pytest.approx(0.0005)


def test_an_unknown_symbol_is_none_not_an_error(conn):
    assert D.funding_rate_at(conn, "NOPEUSDT", BASE) is None
