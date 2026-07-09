"""Range-read consumer migration V2 parity #2 (BATCH-STORAGE-ROTATION-
RETENTION-RANGE-READ-CONSUMER-MIGRATION-V2).

`tools/research_s34_sell_reversal_filter.py`'s two allowlisted bounded
range aggregates (previously inline in `day_so_far`) are extracted into
named oracles and migrated to reader-backed counterparts via `plan_read`/
`execute_read`:
  - `_day_high_low`   (mark_prices: `MAX(mark_price), MIN(mark_price)`)
                      -> `_day_high_low_v2`
  - `_day_agg_count`  (agg_trades:  `COUNT(*)`) -> `_day_agg_count_v2`
Both oracles are kept unchanged as parity references. Symbol is hardcoded
'ETHUSDT' in both (as in the original SQL) -> real ARCHIVE_ONLY production
smokes for both tables (mark_prices/ETHUSDT/2026-05, agg_trades/ETHUSDT/
2026-02), plus SQLITE_ONLY and HYBRID.

DELIBERATELY OUT OF SCOPE (asserted below to remain on direct SQL):
  - `mark_at`: an ASOF-style point lookup (ORDER BY ts_ms ASC/DESC LIMIT 1)
    -- belongs to the ASOF track, not this range-read gate.
  - the two `liquidations` SUM(notional) queries -- `liquidations` is an
    out-of-allowlist table with no archive partition / reader support.

Range-boundary note: the oracles use an INCLUSIVE upper bound
(`ts_ms<=end_ms`); the reader uses half-open `[start, end)`, so the `_v2`
helpers pass `end_ms+1`. Proven exact via a boundary test.
"""
from __future__ import annotations

import json
import os

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
import tools.research_s34_sell_reversal_filter as mod
from tools.research_s34_sell_reversal_filter import (
    _day_high_low, _day_high_low_v2, _day_agg_count, _day_agg_count_v2,
)

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)

pytestmark = pytest.mark.filterwarnings("ignore")

AGG_ARCHIVE_START_MS = 1771165588000
AGG_PARTITION_END_MS = 1772323200000
MARK_ARCHIVE_START_MS = 1777593600000
MARK_PARTITION_END_MS = 1780272000000


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old_high_low(start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _day_high_low(conn, start_ms, end_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_agg_count(start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _day_agg_count(conn, start_ms, end_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _agg_max_ts():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute("SELECT MAX(ts_ms) FROM agg_trades WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _mark_max_ts():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- MIN/MAX aggregate parity ---

def test_high_low_parity_archive_only_real():
    _needs_real_db()
    start_ms, end_ms = MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old_hi, old_lo = _old_high_low(start_ms, end_ms)
    new_hi, new_lo = _day_high_low_v2(_root(), start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert old_hi is not None
    assert new_hi == pytest.approx(old_hi, rel=1e-12)
    assert new_lo == pytest.approx(old_lo, rel=1e-12)


def test_high_low_parity_hybrid_real_boundary():
    _needs_real_db()
    start_ms, end_ms = MARK_PARTITION_END_MS - 150_000, MARK_PARTITION_END_MS + 150_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "HYBRID"
    old_hi, old_lo = _old_high_low(start_ms, end_ms)
    new_hi, new_lo = _day_high_low_v2(_root(), start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_hi == pytest.approx(old_hi, rel=1e-12)
    assert new_lo == pytest.approx(old_lo, rel=1e-12)


def test_high_low_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _mark_max_ts()
    start_ms, end_ms = max_ts - 300_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_hi, old_lo = _old_high_low(start_ms, end_ms)
    new_hi, new_lo = _day_high_low_v2(_root(), start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_hi == pytest.approx(old_hi, rel=1e-12)
    assert new_lo == pytest.approx(old_lo, rel=1e-12)


def test_high_low_empty_range_returns_none_none():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000
    assert _old_high_low(lo, hi) == (None, None)
    assert _day_high_low_v2(_root(), lo, hi, source_db_path=REAL_SOURCE_DB) == (None, None)


# --- COUNT aggregate parity ---

def test_agg_count_parity_archive_only_real():
    _needs_real_db()
    start_ms, end_ms = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old_n = _old_agg_count(start_ms, end_ms)
    new_n = _day_agg_count_v2(_root(), start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert old_n > 0
    assert new_n == old_n


def test_agg_count_parity_hybrid_real_boundary():
    _needs_real_db()
    start_ms, end_ms = AGG_PARTITION_END_MS - 150_000, AGG_PARTITION_END_MS + 150_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "HYBRID"
    old_n = _old_agg_count(start_ms, end_ms)
    new_n = _day_agg_count_v2(_root(), start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_n == old_n


def test_agg_count_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _agg_max_ts()
    start_ms, end_ms = max_ts - 60_000, max_ts
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_n = _old_agg_count(start_ms, end_ms)
    new_n = _day_agg_count_v2(_root(), start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_n == old_n


def test_agg_count_empty_range_returns_zero():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000
    assert _old_agg_count(lo, hi) == 0
    assert _day_agg_count_v2(_root(), lo, hi, source_db_path=REAL_SOURCE_DB) == 0


# --- inclusive-end boundary parity (both aggregates) ---

def test_inclusive_end_boundary_parity_real():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts = conn.execute(
            "SELECT ts_ms FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    n_at = _day_agg_count_v2(_root(), AGG_ARCHIVE_START_MS, int(last_ts), source_db_path=REAL_SOURCE_DB)
    n_before = _day_agg_count_v2(_root(), AGG_ARCHIVE_START_MS, int(last_ts) - 1, source_db_path=REAL_SOURCE_DB)
    old_at = _old_agg_count(AGG_ARCHIVE_START_MS, int(last_ts))
    old_before = _old_agg_count(AGG_ARCHIVE_START_MS, int(last_ts) - 1)
    assert n_at == old_at
    assert n_before == old_before
    assert n_at >= n_before + 1  # boundary row(s) at last_ts included


def test_repeated_run_determinism():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000
    vals = [_day_high_low_v2(_root(), s, e, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_provenance_reports_archive_source():
    _needs_real_db()
    start_ms, end_ms = MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["table"] == "mark_prices"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["mark_price"]
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert prov["row_count"] > 0


def test_no_full_reverify_referenced():
    import inspect
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    _day_high_low_v2(root, MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000, source_db_path=REAL_SOURCE_DB)
    _day_agg_count_v2(root, AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


# --- out-of-scope queries remain on direct SQL ---

def test_mark_at_asof_lookup_left_on_direct_sql():
    import inspect
    src = inspect.getsource(mod.mark_at)
    assert "mark_prices" in src and "limit 1" in src
    assert "RR.lookup_latest_at_or_before(" not in src
    assert "RR.execute_read(" not in src


def test_liquidations_sum_left_on_direct_sql_out_of_allowlist():
    import inspect
    src = inspect.getsource(mod.day_so_far)
    # both liquidations SUM(notional) queries stay direct on `con`
    assert src.count("from liquidations") == 2
    assert "RR.execute_read(" not in src  # day_so_far calls _v2 helpers, never execute_read directly
    # and the migrated aggregates DO go through the reader helpers
    assert "_day_high_low_v2(" in src and "_day_agg_count_v2(" in src


def test_day_so_far_migrated_aggregates_not_inline():
    import inspect
    src = inspect.getsource(mod.day_so_far)
    assert "max(mark_price)" not in src   # moved into the _day_high_low oracle
    assert "count(*)" not in src          # moved into the _day_agg_count oracle


# --- synthetic NULL / empty edge for MIN/MAX + symbol-mismatch safety ---

def _catalog(root, entries):
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}, f)


MP_DDL = ("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
          "mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)")


def _sqlite_mp(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute(MP_DDL)
    c.execute("CREATE INDEX ix ON mark_prices(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


MAY = 1777593600000


def test_high_low_symbol_mismatch_safety_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mp(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0, None),
        (2, MAY + 1000, "ETHUSDT", 3050.0, 0.0, None),
        (3, MAY + 2000, "ETHUSDT", 2990.0, 0.0, None),
        (4, MAY + 500, "SOLUSDT", 99999.0, 0.0, None),   # must not leak into ETHUSDT hi/lo
    ])
    hi, lo = _day_high_low_v2(root, MAY, MAY + 2000, source_db_path=db)
    assert hi == 3050.0 and lo == 2990.0  # SOLUSDT's 99999 excluded
    # empty window -> (None, None), matching SQL MAX/MIN over no rows
    assert _day_high_low_v2(root, MAY + 10_000, MAY + 20_000, source_db_path=db) == (None, None)
