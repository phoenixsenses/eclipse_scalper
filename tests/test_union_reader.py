"""Tests for ami/storage/union_reader.py -- the Phase-1a union read-only
connection factory. Uses only small synthetic SQLite files; never opens the
real microstructure.db.

Run (per repo guardrails):
    pytest tests/test_union_reader.py --basetemp=<scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from ami.storage import union_reader as UR

_MARK_SCHEMA = (
    "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, "
    "symbol TEXT NOT NULL, mark_price REAL NOT NULL, funding_rate REAL, next_funding_time_ms INTEGER)"
)


def _make_db(path, rows):
    """rows: list of (ts_ms, symbol, mark_price)."""
    conn = sqlite3.connect(str(path))
    conn.execute(_MARK_SCHEMA)
    conn.execute("CREATE INDEX idx_mark_symbol_ts ON mark_prices(symbol, ts_ms)")
    conn.executemany("INSERT INTO mark_prices(ts_ms, symbol, mark_price) VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()


def _state(tmp_path, live_rows, frozen_rows_list, cutoff):
    live = tmp_path / "live.db"
    _make_db(live, live_rows)
    segments = []
    for i, fr in enumerate(frozen_rows_list):
        fp = tmp_path / f"frozen{i}.db"
        _make_db(fp, fr)
        segments.append(UR.FrozenSegment(path=str(fp), start_ms=None, end_ms=cutoff))
    return UR.RotationState(live_db_path=str(live), cutoff_ms=cutoff, frozen_segments=tuple(segments))


# ---------------------------------------------------------------------------
# rotation_state loading
# ---------------------------------------------------------------------------

def test_load_absent_file_is_pre_rotation():
    st = UR.load_rotation_state("C:/nonexistent/rotation_state.json")
    assert st.is_pre_rotation
    assert st.cutoff_ms is None
    assert st.frozen_segments == ()


def test_load_valid_state(tmp_path):
    p = tmp_path / "rotation_state.json"
    p.write_text(json.dumps({
        "live_db_path": "X/live.db", "cutoff_ms": 1000,
        "frozen_segments": [{"path": "X/frozen0.db", "start_ms": 0, "end_ms": 1000}],
    }), encoding="utf-8")
    st = UR.load_rotation_state(p)
    assert st.live_db_path == "X/live.db"
    assert st.cutoff_ms == 1000
    assert len(st.frozen_segments) == 1
    assert st.frozen_segments[0].path == "X/frozen0.db"
    assert not st.is_pre_rotation


def test_load_malformed_fails_closed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(UR.RotationStateError):
        UR.load_rotation_state(p)


def test_load_bad_cutoff_type_fails_closed(tmp_path):
    p = tmp_path / "bad2.json"
    p.write_text(json.dumps({"cutoff_ms": "soon", "frozen_segments": []}), encoding="utf-8")
    with pytest.raises(UR.RotationStateError):
        UR.load_rotation_state(p)


# ---------------------------------------------------------------------------
# pass-through (pre-rotation): byte-identical to a plain read-only open
# ---------------------------------------------------------------------------

def test_pass_through_reads_live_only(tmp_path):
    st = _state(tmp_path, live_rows=[(100, "ETHUSDT", 10.0), (200, "ETHUSDT", 20.0)],
                frozen_rows_list=[], cutoff=None)
    assert st.is_pre_rotation
    conn = UR.open_union_ro(st)
    try:
        assert conn.execute("SELECT MAX(ts_ms) FROM mark_prices").fetchone()[0] == 200
        assert conn.execute("SELECT COUNT(*) FROM mark_prices").fetchone()[0] == 2
    finally:
        conn.close()


def test_pass_through_is_read_only(tmp_path):
    st = _state(tmp_path, live_rows=[(100, "ETHUSDT", 10.0)], frozen_rows_list=[], cutoff=None)
    conn = UR.open_union_ro(st)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO mark_prices(ts_ms, symbol, mark_price) VALUES (300,'ETHUSDT',30.0)")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# union across live + one frozen file
# ---------------------------------------------------------------------------

def test_union_spans_both_files_aggregates(tmp_path):
    # frozen holds ts < 1000 (cutoff), live holds ts >= 1000 -- disjoint.
    st = _state(tmp_path,
                live_rows=[(1000, "ETHUSDT", 30.0), (1200, "ETHUSDT", 40.0)],
                frozen_rows_list=[[(100, "ETHUSDT", 10.0), (500, "ETHUSDT", 20.0)]],
                cutoff=1000)
    conn = UR.open_union_ro(st)
    try:
        # COUNT spans both, no double count
        assert conn.execute("SELECT COUNT(*) FROM mark_prices").fetchone()[0] == 4
        # MAX = live tail; MIN = frozen head
        assert conn.execute("SELECT MAX(ts_ms) FROM mark_prices").fetchone()[0] == 1200
        assert conn.execute("SELECT MIN(ts_ms) FROM mark_prices").fetchone()[0] == 100
    finally:
        conn.close()


def test_union_as_of_lookup_routes_by_time(tmp_path):
    st = _state(tmp_path,
                live_rows=[(1000, "ETHUSDT", 30.0), (1200, "ETHUSDT", 40.0)],
                frozen_rows_list=[[(100, "ETHUSDT", 10.0), (500, "ETHUSDT", 20.0)]],
                cutoff=1000)
    conn = UR.open_union_ro(st)
    try:
        def as_of(ts):
            r = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? "
                             "ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
            return r[0] if r else None
        assert as_of(600) == 20.0     # falls in the frozen era
        assert as_of(1000) == 30.0    # exactly at cutoff -> live
        assert as_of(1500) == 40.0    # live tail
        assert as_of(50) is None      # before any data
    finally:
        conn.close()


def test_union_range_sum_spanning_cutoff(tmp_path):
    st = _state(tmp_path,
                live_rows=[(1000, "ETHUSDT", 30.0)],
                frozen_rows_list=[[(500, "ETHUSDT", 20.0)]],
                cutoff=1000)
    conn = UR.open_union_ro(st)
    try:
        total = conn.execute("SELECT SUM(mark_price) FROM mark_prices WHERE ts_ms BETWEEN 400 AND 1100").fetchone()[0]
        assert total == 50.0  # 20 (frozen) + 30 (live), counted once each
    finally:
        conn.close()


def test_union_is_read_only(tmp_path):
    st = _state(tmp_path, live_rows=[(1000, "ETHUSDT", 30.0)],
                frozen_rows_list=[[(500, "ETHUSDT", 20.0)]], cutoff=1000)
    conn = UR.open_union_ro(st)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO main.mark_prices(ts_ms, symbol, mark_price) VALUES (2000,'ETHUSDT',99.0)")
    finally:
        conn.close()


def test_union_multiple_frozen_segments(tmp_path):
    st = _state(tmp_path,
                live_rows=[(2000, "ETHUSDT", 30.0)],
                frozen_rows_list=[[(100, "ETHUSDT", 10.0)], [(1000, "ETHUSDT", 20.0)]],
                cutoff=2000)
    conn = UR.open_union_ro(st)
    try:
        assert conn.execute("SELECT COUNT(*) FROM mark_prices").fetchone()[0] == 3
        assert conn.execute("SELECT MIN(ts_ms) FROM mark_prices").fetchone()[0] == 100
        assert conn.execute("SELECT MAX(ts_ms) FROM mark_prices").fetchone()[0] == 2000
    finally:
        conn.close()


def test_missing_frozen_file_fails_closed(tmp_path):
    live = tmp_path / "live.db"
    _make_db(live, [(1000, "ETHUSDT", 30.0)])
    st = UR.RotationState(live_db_path=str(live), cutoff_ms=1000,
                          frozen_segments=(UR.FrozenSegment(path=str(tmp_path / "gone.db"),
                                                            start_ms=None, end_ms=1000),))
    with pytest.raises(UR.RotationStateError):
        UR.open_union_ro(st)


def test_missing_live_file_fails_closed(tmp_path):
    st = UR.RotationState(live_db_path=str(tmp_path / "nolive.db"), cutoff_ms=None, frozen_segments=())
    with pytest.raises(UR.RotationStateError):
        UR.open_union_ro(st)


# ---------------------------------------------------------------------------
# correction-round coverage: id blanking (finding #1), schema drift (finding #2),
# query_only as an independent lock (gap 4d), malformed frozen-segment entries
# ---------------------------------------------------------------------------

def test_union_blanks_id_to_null(tmp_path):
    # frozen and live both number id from 1 -> a raw union would collide them.
    # The view exposes id as NULL so id-keyed queries fail loud, while ts_ms-keyed
    # reads still span both files correctly.
    st = _state(tmp_path,
                live_rows=[(1000, "ETHUSDT", 30.0), (1200, "ETHUSDT", 40.0)],
                frozen_rows_list=[[(100, "ETHUSDT", 10.0), (500, "ETHUSDT", 20.0)]],
                cutoff=1000)
    conn = UR.open_union_ro(st)
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM mark_prices").fetchall()]
        assert ids == [None, None, None, None]          # every id blanked
        # an id filter therefore matches nothing (fail loud, not wrong-row)
        assert conn.execute("SELECT COUNT(*) FROM mark_prices WHERE id IN (1,2)").fetchone()[0] == 0
        # time-keyed reads unaffected
        assert conn.execute("SELECT COUNT(*) FROM mark_prices").fetchone()[0] == 4
        assert conn.execute("SELECT mark_price FROM mark_prices WHERE ts_ms=500").fetchone()[0] == 20.0
    finally:
        conn.close()


def test_passthrough_keeps_real_id(tmp_path):
    # single-file pass-through has no collision hazard -> id stays real.
    st = _state(tmp_path, live_rows=[(100, "ETHUSDT", 10.0), (200, "ETHUSDT", 20.0)],
                frozen_rows_list=[], cutoff=None)
    conn = UR.open_union_ro(st)
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM mark_prices ORDER BY id").fetchall()]
        assert ids == [1, 2]
    finally:
        conn.close()


def test_schema_drift_fails_closed(tmp_path):
    # frozen table missing a column present in live -> refuse at open, no half-view.
    live = tmp_path / "live.db"
    _make_db(live, [(1000, "ETHUSDT", 30.0)])
    frozen = tmp_path / "frozen0.db"
    c = sqlite3.connect(str(frozen))
    # missing next_funding_time_ms (present in the live schema)
    c.execute("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, "
              "symbol TEXT, mark_price REAL, funding_rate REAL)")
    c.execute("INSERT INTO mark_prices(ts_ms, symbol, mark_price) VALUES (500,'ETHUSDT',20.0)")
    c.commit(); c.close()
    st = UR.RotationState(live_db_path=str(live), cutoff_ms=1000,
                          frozen_segments=(UR.FrozenSegment(path=str(frozen), start_ms=None, end_ms=1000),))
    with pytest.raises(UR.RotationStateError):
        UR.open_union_ro(st)


def test_query_only_blocks_temp_ddl(tmp_path):
    # query_only=ON is an independent lock beyond mode=ro: it blocks even in-memory
    # temp DDL, which mode=ro alone would allow. Guards a regression that removes it.
    st = _state(tmp_path, live_rows=[(100, "ETHUSDT", 10.0)], frozen_rows_list=[], cutoff=None)
    conn = UR.open_union_ro(st)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TEMP VIEW zz AS SELECT 1")
    finally:
        conn.close()


def test_load_malformed_segment_entry_fails_closed(tmp_path):
    p = tmp_path / "seg.json"
    p.write_text(json.dumps({"frozen_segments": ["not-a-dict"]}), encoding="utf-8")
    with pytest.raises(UR.RotationStateError):
        UR.load_rotation_state(p)
    p.write_text(json.dumps({"frozen_segments": [{"start_ms": 0}]}), encoding="utf-8")  # missing 'path'
    with pytest.raises(UR.RotationStateError):
        UR.load_rotation_state(p)


# ---------------------------------------------------------------------------
# current_live_db_path / open_live_ro (freshness / latest probes -> LIVE only)
# ---------------------------------------------------------------------------

def test_current_live_db_path_pre_rotation():
    assert UR.current_live_db_path(UR.default_pre_rotation_state()) == str(UR.DEFAULT_LIVE_PATH)


def test_current_live_db_path_post_rotation(tmp_path):
    live = tmp_path / "live.db"
    _make_db(live, [(1, "ETHUSDT", 1.0)])
    st = UR.RotationState(live_db_path=str(live), cutoff_ms=1000,
                          frozen_segments=(UR.FrozenSegment(path=str(tmp_path / "f.db"),
                                                            start_ms=None, end_ms=1000),))
    assert UR.current_live_db_path(st) == str(live)  # the fresh file, NOT the frozen one


def test_open_live_ro_is_live_only_and_read_only(tmp_path):
    live = tmp_path / "live.db"
    _make_db(live, [(100, "ETHUSDT", 10.0), (200, "ETHUSDT", 20.0)])
    # a frozen segment is DECLARED but open_live_ro must ignore it (no union)
    st = UR.RotationState(live_db_path=str(live), cutoff_ms=1000,
                          frozen_segments=(UR.FrozenSegment(path=str(tmp_path / "frozen0.db"),
                                                            start_ms=None, end_ms=1000),))
    conn = UR.open_live_ro(st)
    try:
        assert conn.execute("SELECT COUNT(*) FROM mark_prices").fetchone()[0] == 2  # live only, not unioned
        # rowid is a real base-table rowid here (a union VIEW would not expose it)
        assert conn.execute("SELECT ts_ms FROM mark_prices ORDER BY rowid DESC LIMIT 1").fetchone()[0] == 200
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (300,'ETHUSDT',30.0)")
    finally:
        conn.close()


def test_open_live_ro_missing_live_fails_closed(tmp_path):
    st = UR.RotationState(live_db_path=str(tmp_path / "nolive.db"), cutoff_ms=None, frozen_segments=())
    with pytest.raises(UR.RotationStateError):
        UR.open_live_ro(st)


def test_max_clock_touches_only_live_not_frozen(tmp_path):
    # Regression lock for the Phase-1b blocker: latest-clock MAX(ts_ms) probes must run on the
    # LIVE file only. Over the union view MAX touches BOTH files (loses the index min/max fast-path
    # -> a 2-file scan of the biggest table post-rotation); over open_live_ro it touches only live.
    live = tmp_path / "live.db"
    _make_db(live, [(1000, "ETHUSDT", 30.0)])
    frozen = tmp_path / "frozen0.db"
    _make_db(frozen, [(100, "ETHUSDT", 10.0)])
    st = UR.RotationState(live_db_path=str(live), cutoff_ms=1000,
                          frozen_segments=(UR.FrozenSegment(path=str(frozen), start_ms=None, end_ms=1000),))
    q = "SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'"
    uconn = UR.open_union_ro(st)
    lconn = UR.open_live_ro(st)
    try:
        uplan = " ".join(str(r) for r in uconn.execute("EXPLAIN QUERY PLAN " + q).fetchall()).lower()
        lplan = " ".join(str(r) for r in lconn.execute("EXPLAIN QUERY PLAN " + q).fetchall()).lower()
        assert "frozen0" in uplan          # union clock scans the frozen file too (slow path)
        assert "frozen0" not in lplan       # live clock stays single-file (fast path)
        assert uconn.execute(q).fetchone()[0] == 1000  # both still compute the correct global max
        assert lconn.execute(q).fetchone()[0] == 1000
    finally:
        uconn.close()
        lconn.close()


# ---------------------------------------------------------------------------
# as_of_row -- the as-of lookup that must not go through the union view
# ---------------------------------------------------------------------------

_AS_OF_SQL = ("SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
              "ORDER BY ts_ms DESC LIMIT 1")


def _two_file_state(tmp_path):
    """frozen holds ts < 1000, live holds ts >= 1000 (disjoint, as a real rotation)."""
    return _state(tmp_path,
                  live_rows=[(1000, "ETHUSDT", 100.0), (1500, "ETHUSDT", 150.0),
                             (1200, "BTCUSDT", 12.0)],
                  frozen_rows_list=[[(100, "ETHUSDT", 10.0), (500, "ETHUSDT", 50.0),
                                     (300, "BTCUSDT", 3.0)]],
                  cutoff=1000)


def test_union_schemas_lists_main_then_frozen(tmp_path):
    st = _two_file_state(tmp_path)
    conn = UR.open_union_ro(st)
    try:
        assert UR.union_schemas(conn) == ["main", "frozen0"]
    finally:
        conn.close()
    conn = UR.open_union_ro(UR.RotationState(live_db_path=st.live_db_path, cutoff_ms=None,
                                             frozen_segments=()))
    try:
        assert UR.union_schemas(conn) == ["main"]
    finally:
        conn.close()


def test_as_of_row_answer_is_identical_to_the_union_view(tmp_path):
    """Equivalence is the whole point: the fast path must never change an answer.
    Probes straddle both files, the cutoff, exact hits and the empty region."""
    st = _two_file_state(tmp_path)
    conn = UR.open_union_ro(st)
    try:
        for probe in (0, 99, 100, 101, 499, 500, 501, 999, 1000, 1001, 1499, 1500, 1501, 9999):
            slow = conn.execute(_AS_OF_SQL, ("ETHUSDT", probe)).fetchone()
            fast = UR.as_of_row(conn, "mark_prices", "mark_price",
                                where="symbol=? AND ts_ms<=?", params=("ETHUSDT", probe))
            assert fast == slow, f"probe={probe}: {fast!r} != {slow!r}"
    finally:
        conn.close()


def test_as_of_row_crosses_files_and_respects_symbol(tmp_path):
    st = _two_file_state(tmp_path)
    conn = UR.open_union_ro(st)
    try:
        # before the cutoff -> answer lives in the FROZEN file
        assert UR.as_of_row(conn, "mark_prices", "mark_price",
                            where="symbol=? AND ts_ms<=?", params=("ETHUSDT", 900)) == (500, 50.0)
        # after the cutoff -> answer lives in the LIVE file, frozen must not win
        assert UR.as_of_row(conn, "mark_prices", "mark_price",
                            where="symbol=? AND ts_ms<=?", params=("ETHUSDT", 2000)) == (1500, 150.0)
        # a different symbol resolves independently in each file
        assert UR.as_of_row(conn, "mark_prices", "mark_price",
                            where="symbol=? AND ts_ms<=?", params=("BTCUSDT", 900)) == (300, 3.0)
    finally:
        conn.close()


def test_as_of_row_returns_none_when_nothing_matches(tmp_path):
    st = _two_file_state(tmp_path)
    conn = UR.open_union_ro(st)
    try:
        assert UR.as_of_row(conn, "mark_prices", "mark_price",
                            where="symbol=? AND ts_ms<=?", params=("ETHUSDT", 99)) is None
        assert UR.as_of_row(conn, "mark_prices", "mark_price",
                            where="symbol=? AND ts_ms<=?", params=("SOLUSDT", 9999)) is None
    finally:
        conn.close()


def test_as_of_row_honours_an_extra_predicate(tmp_path):
    """The funding-rate call sites filter `funding_rate IS NOT NULL`; the newest row
    overall must NOT be returned when it fails the caller's predicate."""
    st = _two_file_state(tmp_path)
    w = sqlite3.connect(st.live_db_path)
    w.execute("UPDATE mark_prices SET funding_rate=0.01 WHERE ts_ms=1000")
    w.commit()
    w.close()
    conn = UR.open_union_ro(st)
    try:
        got = UR.as_of_row(conn, "mark_prices", "funding_rate",
                           where="symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL",
                           params=("ETHUSDT", 9999))
        assert got == (1000, 0.01)  # not the newer ts=1500 row, whose funding_rate is NULL
    finally:
        conn.close()


def test_as_of_row_pre_rotation_is_a_plain_seek(tmp_path):
    live = tmp_path / "live.db"
    _make_db(live, [(100, "ETHUSDT", 10.0), (200, "ETHUSDT", 20.0)])
    st = UR.RotationState(live_db_path=str(live), cutoff_ms=None, frozen_segments=())
    conn = UR.open_union_ro(st)
    try:
        assert UR.as_of_row(conn, "mark_prices", "mark_price",
                            where="symbol=? AND ts_ms<=?", params=("ETHUSDT", 150)) == (100, 10.0)
    finally:
        conn.close()


def test_as_of_row_validates_interpolated_identifiers(tmp_path):
    st = _two_file_state(tmp_path)
    conn = UR.open_union_ro(st)
    try:
        with pytest.raises(ValueError):
            UR.as_of_row(conn, "sqlite_master", "name")           # not in UNION_TABLES
        with pytest.raises(ValueError):
            UR.as_of_row(conn, "mark_prices", "mark_price", ts_col="ts_ms; DROP TABLE x")
    finally:
        conn.close()


def test_as_of_row_plan_has_no_sort_while_the_view_does(tmp_path):
    """Regression lock for the 7560x blow-up measured 2026-07-26.

    Pins the exact trigger: an as-of lookup through the union view only merges when the
    ORDER BY column is a RESULT COLUMN of the compound. Drop `ts_ms` from the select list
    -- as every production call site did -- and the same query falls back to materialising
    and sorting the whole compound. The per-schema form has no compound at all, so it is an
    index seek on every schema regardless of what the planner decides."""
    st = _two_file_state(tmp_path)
    conn = UR.open_union_ro(st)
    try:
        def plan(sql):
            return " ".join(str(r) for r in
                            conn.execute("EXPLAIN QUERY PLAN " + sql,
                                         ("ETHUSDT", 9999)).fetchall()).upper()

        # ts_ms NOT selected -> the degenerate plan that cost 7560x in production
        slow = plan("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
                    "ORDER BY ts_ms DESC LIMIT 1")
        assert "TEMP B-TREE" in slow and "COMPOUND" in slow, slow
        # ts_ms selected -> the planner merges instead (correct, but heuristic-dependent)
        assert "TEMP B-TREE" not in plan(_AS_OF_SQL)

        for schema in UR.union_schemas(conn):
            plan = " ".join(str(r) for r in conn.execute(
                f"EXPLAIN QUERY PLAN SELECT ts_ms, mark_price FROM {schema}.mark_prices "
                f"WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                ("ETHUSDT", 9999)).fetchall()).upper()
            assert "TEMP B-TREE" not in plan, f"{schema}: {plan}"
            assert "USING INDEX" in plan, f"{schema}: {plan}"
    finally:
        conn.close()


def test_as_of_row_leaves_the_connection_read_only(tmp_path):
    st = _two_file_state(tmp_path)
    conn = UR.open_union_ro(st)
    try:
        UR.as_of_row(conn, "mark_prices", "mark_price",
                     where="symbol=? AND ts_ms<=?", params=("ETHUSDT", 9999))
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO main.mark_prices(ts_ms,symbol,mark_price) VALUES (9,'X',1.0)")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# history floor: the guard that replaces the frozen segment once it is deleted
# ---------------------------------------------------------------------------


def test_history_floor_spans_frozen_segments(tmp_path):
    state = _state(tmp_path, [(9000, "ETHUSDT", 31.0)], [[(1000, "ETHUSDT", 30.0)]], 5000)
    conn = UR.open_union_ro(state)
    try:
        assert UR.history_floor_ms(conn, "mark_prices") == 1000
    finally:
        conn.close()


def test_history_floor_pre_rotation_is_the_live_minimum(tmp_path):
    state = _state(tmp_path, [(9000, "ETHUSDT", 31.0), (9500, "ETHUSDT", 32.0)], [], None)
    conn = UR.open_union_ro(state)
    try:
        assert UR.history_floor_ms(conn, "mark_prices") == 9000
    finally:
        conn.close()


def test_history_floor_rises_when_the_frozen_segment_goes_away(tmp_path):
    """The Phase-4 delete in miniature: same live file, frozen dropped."""
    with_frozen = _state(tmp_path, [(9000, "ETHUSDT", 31.0)], [[(1000, "ETHUSDT", 30.0)]], 5000)
    conn = UR.open_union_ro(with_frozen)
    try:
        deep = UR.history_floor_ms(conn, "mark_prices")
    finally:
        conn.close()

    without = UR.RotationState(live_db_path=with_frozen.live_db_path,
                               cutoff_ms=None, frozen_segments=())
    conn = UR.open_union_ro(without)
    try:
        shallow = UR.history_floor_ms(conn, "mark_prices")
    finally:
        conn.close()
    assert deep == 1000 and shallow == 9000


def test_history_floor_is_none_for_an_empty_table(tmp_path):
    state = _state(tmp_path, [], [], None)
    conn = UR.open_union_ro(state)
    try:
        assert UR.history_floor_ms(conn, "mark_prices") is None
    finally:
        conn.close()


def test_assert_history_covers_passes_when_history_is_deep_enough(tmp_path):
    state = _state(tmp_path, [(9000, "ETHUSDT", 31.0)], [[(1000, "ETHUSDT", 30.0)]], 5000)
    conn = UR.open_union_ro(state)
    try:
        assert UR.assert_history_covers(conn, "mark_prices", 2000) == 1000
        assert UR.assert_history_covers(conn, "mark_prices", 1000) == 1000  # exactly at the floor
    finally:
        conn.close()


def test_assert_history_covers_refuses_a_query_that_would_be_truncated(tmp_path):
    """Without this the answer is computed on partial history and looks normal."""
    state = _state(tmp_path, [(9000, "ETHUSDT", 31.0)], [], None)
    conn = UR.open_union_ro(state)
    try:
        with pytest.raises(UR.InsufficientHistoryError, match="days short"):
            UR.assert_history_covers(conn, "mark_prices", 9000 - 7 * 86_400_000,
                                     what="liq_flip 7d lookback")
    finally:
        conn.close()


def test_assert_history_covers_refuses_an_empty_table(tmp_path):
    state = _state(tmp_path, [], [], None)
    conn = UR.open_union_ro(state)
    try:
        with pytest.raises(UR.InsufficientHistoryError, match="no rows"):
            UR.assert_history_covers(conn, "mark_prices", 1000)
    finally:
        conn.close()


def test_history_floor_ignores_the_union_view_and_seeks_each_schema(tmp_path):
    """Going through the temp view would materialise the compound; assert the plan."""
    state = _state(tmp_path, [(9000, "ETHUSDT", 31.0)], [[(1000, "ETHUSDT", 30.0)]], 5000)
    conn = UR.open_union_ro(state)
    try:
        plan = " ".join(r[3] for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT MIN(ts_ms) FROM main.mark_prices"))
        assert "SCAN" not in plan.upper() or "INDEX" in plan.upper()
        assert UR.history_floor_ms(conn, "mark_prices") == 1000
    finally:
        conn.close()


def test_history_floor_validates_the_table_identifier(tmp_path):
    state = _state(tmp_path, [(9000, "ETHUSDT", 31.0)], [], None)
    conn = UR.open_union_ro(state)
    try:
        with pytest.raises(ValueError, match="unsafe table identifier"):
            UR.history_floor_ms(conn, "mark_prices; DROP TABLE mark_prices")
    finally:
        conn.close()


def test_history_floor_leaves_the_connection_read_only(tmp_path):
    state = _state(tmp_path, [(9000, "ETHUSDT", 31.0)], [[(1000, "ETHUSDT", 30.0)]], 5000)
    conn = UR.open_union_ro(state)
    try:
        UR.history_floor_ms(conn, "mark_prices")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM main.mark_prices")
    finally:
        conn.close()
