"""Range-read consumer migration parity -- `tools/s34_regime_filter_shadow_
eval.py`'s `mark_rows` (bounded `mark_prices` range read) migration to
`mark_rows_v2` (reader-backed, via `ami.storage.research_reader.plan_read`/
`execute_read`).

This is the SECOND leg of this file's migration. The FIRST leg (the
`book_ticker` ASOF point-lookup, `book_at` -> `book_at_v2`) was already
migrated and proven in `tests/test_s34_regime_filter_shadow_eval_lookup_
migration_parity.py` (Batch 4 / commit b40441f2) and is NOT re-tested
here -- `book_at`/`book_at_v2` are untouched by this migration.

Boundary mapping: `mark_rows`'s oracle is INCLUSIVE on BOTH bounds
(`ts_ms>=start_ms AND ts_ms<=end_ms`); `research_reader.plan_read`/
`execute_read` use a half-open `[start_ms, end_ms)` window. Only the UPPER
bound is shifted (`end_ms+1`) to reproduce the oracle's inclusive upper
bound exactly for integer ts_ms -- the LOWER bound is passed through
UNCHANGED. This is the identical mapping used for the structurally
identical `_mark_prices_range`/`_mark_prices_range_v2` oracle pair in
`tools/micro_edge_smoke.py` (Range-Read V4), NOT the different
lower-bound-shift mapping that applies to a different, EXCLUSIVE-lower
oracle shape used elsewhere in this migration series.

CRITICAL COUPLING under test: `simulate_counterfactual()` derives
`exit_ts_ms` from whatever `mark_rows`/`mark_rows_v2` returns, and feeds
that `exit_ts_ms` into `book_at_v2` (the ASOF quote lookup) to compute the
executable fill. A boundary bug in the range read would silently corrupt
`exit_ts_ms` and, downstream, the ASOF result -- so parity is proven not
only at the `mark_rows_v2` level (isolated function calls) but also
end-to-end at the `simulate_counterfactual()` level, comparing the FULL
result dict (and the raw ASOF `quote` fed to `book_at_v2`) between a
legacy-`mark_rows`-backed run and a `mark_rows_v2`-backed run of the exact
same function, over the exact same fixture.
"""
from __future__ import annotations

import json
import hashlib
import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
from ami.storage.registry import get_table_spec
import tools.s34_regime_filter_shadow_eval as mod
from tools.s34_regime_filter_shadow_eval import mark_rows, mark_rows_v2, simulate_counterfactual

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
MARK_SPEC = get_table_spec("mark_prices")
BOOK_SPEC = get_table_spec("book_ticker")

pytestmark = pytest.mark.filterwarnings("ignore")

# Real mark_prices ETHUSDT archive partition boundaries (2026-05), shared
# table-global constants -- same values used for the mark_prices archive
# in tests/test_micro_edge_smoke_reader_migration_parity.py.
MARK_ARCHIVE_START_MS = 1777593600000
MARK_PARTITION_END_MS = 1780272000000

MAY = 1777593600000
APR = 1775001600000


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old_mark(symbol, s, e, db_path=REAL_SOURCE_DB):
    conn, log = SRC.open_read_only(db_path)
    try:
        return mark_rows(conn, symbol, s, e)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _eth_max_ts(table="mark_prices", db_path=REAL_SOURCE_DB):
    conn, log = SRC.open_read_only(db_path)
    try:
        return conn.execute(f"SELECT MAX(ts_ms) FROM {table} WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

MP_DDL = ("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
          "mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)")
BT_DDL = ("CREATE TABLE book_ticker (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, bid_price REAL, "
          "bid_qty REAL, ask_price REAL, ask_qty REAL, mid_price REAL, spread_pct REAL, book_imbalance REAL, "
          "bid_depth_usd REAL)")


def _sqlite_mark(path, rows):
    c = sqlite3.connect(path)
    c.execute(MP_DDL)
    c.execute("CREATE INDEX ix ON mark_prices(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


def _ensure_book_ticker(path, rows=()):
    c = sqlite3.connect(path)
    c.execute(BT_DDL)
    c.execute("CREATE INDEX ixb ON book_ticker(symbol, ts_ms)")
    if rows:
        c.executemany("INSERT INTO book_ticker VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


def _mp_rows(start_id, start_ms, n, symbol="ETHUSDT", step_ms=1000, price0=3000.0):
    return [(start_id + i, start_ms + i * step_ms, symbol, price0 + i, None, None) for i in range(n)]


def _bt_rows(start_id, start_ms, n, symbol="ETHUSDT", step_ms=1000, bid0=3000.0):
    return [(start_id + i, start_ms + i * step_ms, symbol, bid0 + i, 1.0, bid0 + i + 1.0, 1.0,
             bid0 + i + 0.5, 0.005, 0.1, 1000.0) for i in range(n)]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _catalog(root, entries):
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}, f)


def _build_mark_archive(root, *, symbol, partition_start_ms, partition_end_ms, rows, corrupt_manifest=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema

    rel = os.path.join("table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                        f"symbol={symbol}", "year=2026", "month=04", "version=v1")
    fd = os.path.join(root, rel)
    os.makedirs(fd, exist_ok=True)
    pp = os.path.join(fd, "part-00000.parquet")
    schema = build_pyarrow_schema(MARK_SPEC)
    arrays = [pa.array([r[i] for r in rows], type=schema.field(c).type)
              for i, c in enumerate(MARK_SPEC.preserved_columns)]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), pp, compression="zstd")
    manifest = {"source_table": "mark_prices", "symbol": symbol, "venue": MARK_SPEC.venue,
                "market_segment": MARK_SPEC.market_segment, "row_count": len(rows), "shards": None,
                "ordered_scientific_content_hash": "0" * 64 if corrupt_manifest else "irrelevant",
                "partition_id": f"t-{symbol}", "parquet_path": os.path.join(rel, "part-00000.parquet"),
                "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
                "source_watermark_field": "id", "source_watermark_value": rows[-1][0],
                "parquet_sha256": _sha256_file(pp)}
    mp = os.path.join(fd, "manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(fd, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("ok\n")
    entry = {"archive_identity": f"t-{symbol}", "archive_relative_path": rel, "source_table": "mark_prices",
             "symbol": symbol, "venue": MARK_SPEC.venue, "market_segment": MARK_SPEC.market_segment,
             "partition_start_ms": partition_start_ms, "partition_end_ms": partition_end_ms, "row_count": len(rows),
             "manifest_sha256": "f" * 64 if corrupt_manifest else _sha256_file(mp),
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": manifest["ordered_scientific_content_hash"]}
    with open(os.path.join(fd, "catalog_entry.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f)
    return entry


# ---------------------------------------------------------------------------
# 1-7: boundary / window-shape parity (synthetic, SQLITE_ONLY)
# ---------------------------------------------------------------------------

def test_lower_bound_exact_match_included(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, _mp_rows(1, MAY, 5))
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY, MAY + 4000)
    new = mark_rows_v2(root, "ETHUSDT", MAY, MAY + 4000, source_db_path=db)
    assert new == old
    assert new[0] == (MAY, 3000.0)  # exact lower bound row present


def test_lower_bound_off_by_one_ms(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, _mp_rows(1, MAY, 5))  # ts: MAY, MAY+1000, ... MAY+4000
    # window starting 1ms BELOW the first row's ts: that first row (at MAY)
    # is one ms ABOVE this new start, so it must still be INCLUDED.
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY - 1, MAY + 4000)
    new = mark_rows_v2(root, "ETHUSDT", MAY - 1, MAY + 4000, source_db_path=db)
    assert new == old
    assert new[0] == (MAY, 3000.0)
    # window starting exactly 1ms ABOVE the first row's ts: that row must
    # now be EXCLUDED.
    old2 = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY + 1, MAY + 4000)
    new2 = mark_rows_v2(root, "ETHUSDT", MAY + 1, MAY + 4000, source_db_path=db)
    assert new2 == old2
    assert all(ts != MAY for ts, _ in new2)
    assert new2[0] == (MAY + 1000, 3001.0)


def test_upper_bound_exact_match_included(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, _mp_rows(1, MAY, 5))  # last row ts = MAY+4000
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY, MAY + 4000)
    new = mark_rows_v2(root, "ETHUSDT", MAY, MAY + 4000, source_db_path=db)
    assert new == old
    assert new[-1] == (MAY + 4000, 3004.0)  # exact upper bound row present (inclusive-upper)


def test_upper_bound_off_by_one_ms(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, _mp_rows(1, MAY, 5))  # last row ts = MAY+4000
    # window ending 1ms ABOVE the last row's ts: last row still included.
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY, MAY + 4001)
    new = mark_rows_v2(root, "ETHUSDT", MAY, MAY + 4001, source_db_path=db)
    assert new == old
    assert new[-1] == (MAY + 4000, 3004.0)
    # window ending exactly 1ms BELOW the last row's ts: last row excluded.
    old2 = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY, MAY + 3999)
    new2 = mark_rows_v2(root, "ETHUSDT", MAY, MAY + 3999, source_db_path=db)
    assert new2 == old2
    assert all(ts != MAY + 4000 for ts, _ in new2)
    assert new2[-1] == (MAY + 3000, 3003.0)


def test_zero_width_window_single_instant(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, _mp_rows(1, MAY, 5))
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY + 2000, MAY + 2000)
    new = mark_rows_v2(root, "ETHUSDT", MAY + 2000, MAY + 2000, source_db_path=db)
    assert new == old == [(MAY + 2000, 3002.0)]


def test_one_millisecond_window(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, _mp_rows(1, MAY, 5))
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY + 2000, MAY + 2001)
    new = mark_rows_v2(root, "ETHUSDT", MAY + 2000, MAY + 2001, source_db_path=db)
    # the +1ms tick (MAY+2001) has no row in this fixture (rows step by
    # 1000ms); only the MAY+2000 row falls inside this 1ms-wide window.
    assert new == old
    assert new == [(MAY + 2000, 3002.0)]


def test_empty_window_no_rows_either_side(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, _mp_rows(1, MAY, 5))
    lo, hi = MAY - 10_000_000, MAY - 9_000_000  # nowhere near any row
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", lo, hi)
    new = mark_rows_v2(root, "ETHUSDT", lo, hi, source_db_path=db)
    assert old == [] and new == []


# ---------------------------------------------------------------------------
# 8: duplicate timestamps -- both same-price and different-price, don't
# assume the real DB's "always same price for a duplicate ts" property.
# ---------------------------------------------------------------------------

def test_duplicate_timestamps_same_price(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, [
        (1, MAY, "ETHUSDT", 3000.0, None, None),
        (2, MAY, "ETHUSDT", 3000.0, None, None),
        (3, MAY + 1000, "ETHUSDT", 3010.0, None, None),
    ])
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY, MAY + 1000)
    new = mark_rows_v2(root, "ETHUSDT", MAY, MAY + 1000, source_db_path=db)
    assert new == old
    assert len([r for r in new if r[0] == MAY]) == 2


def test_duplicate_timestamps_different_price(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, [
        (1, MAY, "ETHUSDT", 3000.0, None, None),
        (2, MAY, "ETHUSDT", 3005.5, None, None),  # same ts_ms, DIFFERENT price -- do not assume this can't happen
        (3, MAY + 1000, "ETHUSDT", 3010.0, None, None),
    ])
    old = mark_rows(sqlite3.connect(db), "ETHUSDT", MAY, MAY + 1000)
    new = mark_rows_v2(root, "ETHUSDT", MAY, MAY + 1000, source_db_path=db)
    assert new == old  # row-for-row, in-order equality -- including tie order
    dup_prices = [price for ts, price in new if ts == MAY]
    assert sorted(dup_prices) == [3000.0, 3005.5]


# ---------------------------------------------------------------------------
# 9-11: SQLITE_ONLY / ARCHIVE_ONLY / HYBRID plan-mode coverage
# ---------------------------------------------------------------------------

def test_sqlite_only_plan_mode_is_correct(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, _mp_rows(1, MAY, 5))
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=MAY, end_ms=MAY + 4000 + 1)
    assert plan.mode == "SQLITE_ONLY"


def test_archive_only_boundary_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_mark_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                                 rows=_mp_rows(1, APR, 10))
    _catalog(root, [entry])
    legacy_db = str(tmp_path / "legacy.sqlite")
    _sqlite_mark(legacy_db, _mp_rows(1, APR, 10))
    old = mark_rows(sqlite3.connect(legacy_db), "ETHUSDT", APR, APR + 9000)
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=APR + 9000 + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    new = mark_rows_v2(root, "ETHUSDT", APR, APR + 9000)
    assert new == old
    assert len(new) == 10


def test_archive_only_real_ethusdt_2026_05():
    _needs_real_db()
    root = _root()
    if not os.path.exists(os.path.join(root, PR.ROOT_INDEX_NAME)):
        pytest.skip("production archive root/catalog not present in this checkout")
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    if plan.mode != "ARCHIVE_ONLY":
        pytest.skip(f"real archive partition boundaries drifted (mode={plan.mode}), not this gate's concern")
    old = _old_mark("ETHUSDT", s, e)
    new = mark_rows_v2(root, "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert len(new) == len(old) and len(old) > 0
    assert new == old


def test_hybrid_boundary_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    archive_rows = _mp_rows(1, MAY - 5000, 5)  # ts: MAY-5000 .. MAY-1000
    live_rows = _mp_rows(6, MAY, 5)  # ts: MAY .. MAY+4000
    entry = _build_mark_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                                 rows=archive_rows)
    _catalog(root, [entry])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, live_rows)

    legacy_db = str(tmp_path / "legacy.sqlite")
    _sqlite_mark(legacy_db, archive_rows + live_rows)

    s, e = MAY - 5000, MAY + 3000
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "HYBRID"
    old = mark_rows(sqlite3.connect(legacy_db), "ETHUSDT", s, e)
    new = mark_rows_v2(root, "ETHUSDT", s, e, source_db_path=db)
    assert new == old
    # 5 archive rows (MAY-5000..MAY-1000) + 4 live rows (MAY..MAY+3000,
    # MAY+4000 excluded by the e=MAY+3000 inclusive-upper bound) = 9.
    assert len(new) == 9


def test_hybrid_boundary_real_ethusdt():
    _needs_real_db()
    root = _root()
    if not os.path.exists(os.path.join(root, PR.ROOT_INDEX_NAME)):
        pytest.skip("production archive root/catalog not present in this checkout")
    s, e = MARK_PARTITION_END_MS - 120_000, MARK_PARTITION_END_MS + 60_000
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    if plan.mode != "HYBRID":
        pytest.skip(f"real archive partition boundaries drifted (mode={plan.mode}), not this gate's concern")
    old = _old_mark("ETHUSDT", s, e)
    new = mark_rows_v2(root, "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


# ---------------------------------------------------------------------------
# 12: legacy oracle run independently of v2 (no test may call mark_rows_v2
# twice and compare it to itself)
# ---------------------------------------------------------------------------

def test_legacy_oracle_never_calls_v2():
    import inspect
    src = inspect.getsource(mod.mark_rows)
    assert "mark_rows_v2(" not in src  # docstring mentions the name; the CALL must not exist
    assert "RR." not in src
    assert "plan_read" not in src and "execute_read" not in src


def test_v2_never_calls_legacy_oracle():
    import inspect
    src = inspect.getsource(mod.mark_rows_v2)
    assert "con.execute" not in src
    assert "mark_rows(" not in src  # must not shell out to the legacy oracle


# ---------------------------------------------------------------------------
# 13: real-DB direct parity smoke (both oracle and v2 run once each,
# independently, against the real database)
# ---------------------------------------------------------------------------

def test_real_db_direct_parity_recent_window():
    _needs_real_db()
    max_ts = _eth_max_ts()
    assert max_ts is not None
    s, e = max_ts - 60_000, max_ts
    old = _old_mark("ETHUSDT", s, e)
    new = mark_rows_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_real_db_repeated_run_determinism():
    _needs_real_db()
    max_ts = _eth_max_ts()
    root = _root()
    vals = [mark_rows_v2(root, "ETHUSDT", max_ts - 10_000, max_ts, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_real_db_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    max_ts = _eth_max_ts()
    mark_rows_v2(root, "ETHUSDT", max_ts - 10_000, max_ts, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


# ---------------------------------------------------------------------------
# Call-site / structural checks
# ---------------------------------------------------------------------------

def test_simulate_counterfactual_call_site_uses_v2():
    import inspect
    src = inspect.getsource(mod.simulate_counterfactual)
    assert "mark_rows_v2(root, symbol, entry_ts_ms, end_ms" in src
    assert "mark_rows(con" not in src  # legacy call site removed


# ---------------------------------------------------------------------------
# 14 (MOST CRITICAL): end-to-end simulate_counterfactual() parity --
# proves the mark_rows -> mark_rows_v2 swap does not change exit_ts_ms,
# exit_mark, exit_reason, net_bps, mfe_bps, mae_bps, fill_source, or the
# ASOF `quote` fed into book_at_v2.
# ---------------------------------------------------------------------------

def _base_trade(**overrides):
    trade = {
        "trade_id": "T-PARITY-1",
        "symbol": "ETHUSDT",
        "direction": "LONG",
        "entry_ts_ms": MAY,
        "entry_price": 3000.0,
        "signal_ts_ms": MAY,
        "rule": {
            "name": "TEST_RULE",
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "tp_bps": 1000.0,
            "sl_bps": 1000.0,
            "be_trigger_bps": 500.0,
            "max_horizon_sec": 10.0,
            "taker_fee_bps": 4.0,
        },
        "regime": {"checks": {"a": True}},
        "signal": {"liq_total_notional": 12345.0, "liq_count": 3},
    }
    trade.update(overrides)
    return trade


def _spy_book_at_v2(monkeypatch, sink):
    real = mod.book_at_v2

    def _spy(root_, symbol_, ts_ms_, staleness_, source_db_path=None):
        q = real(root_, symbol_, ts_ms_, staleness_, source_db_path=source_db_path)
        sink.append(q)
        return q

    monkeypatch.setattr(mod, "book_at_v2", _spy)


def test_end_to_end_simulate_counterfactual_parity_sqlite_only(tmp_path, monkeypatch):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    # mark_prices: window [MAY, MAY+10000], gentle wiggle, never trips the
    # 1000bps TP/SL bands -> exit_reason ends as TIME, exit at last row.
    mark_data = [(i + 1, MAY + i * 1000, "ETHUSDT", 3000.0 + (i % 3), None, None) for i in range(11)]
    _sqlite_mark(db, mark_data)
    # book_ticker: a quote available at/near the expected exit_ts_ms.
    _ensure_book_ticker(db, _bt_rows(1, MAY, 11))

    trade = _base_trade()
    now_ms = MAY + 3_600_000  # far beyond the 10s horizon -> complete_horizon

    quotes_legacy, quotes_new = [], []
    real_mark_rows_v2 = mod.mark_rows_v2

    def legacy_mark_rows_v2(root_, symbol_, start_ms_, end_ms_, source_db_path=None):
        conn = sqlite3.connect(db)
        try:
            return mod.mark_rows(conn, symbol_, start_ms_, end_ms_)
        finally:
            conn.close()

    conn = sqlite3.connect(db)
    try:
        _spy_book_at_v2(monkeypatch, quotes_legacy)
        monkeypatch.setattr(mod, "mark_rows_v2", legacy_mark_rows_v2)
        legacy_result = simulate_counterfactual(conn, trade, now_ms, 3600.0, root=root, source_db_path=db)

        monkeypatch.setattr(mod, "mark_rows_v2", real_mark_rows_v2)
        _spy_book_at_v2(monkeypatch, quotes_new)
        new_result = simulate_counterfactual(conn, trade, now_ms, 3600.0, root=root, source_db_path=db)
    finally:
        conn.close()

    assert legacy_result is not None and new_result is not None
    assert legacy_result == new_result  # full-dict parity
    for key in ("exit_ts_ms", "exit_mark", "exit_reason", "net_bps", "mfe_bps", "mae_bps", "exit_fill_source"):
        assert legacy_result[key] == new_result[key], key
    assert legacy_result["exit_reason"] == "TIME"
    assert quotes_legacy and quotes_new
    assert quotes_legacy[-1] == quotes_new[-1]  # identical ASOF quote fed downstream


def test_end_to_end_simulate_counterfactual_parity_hybrid_boundary(tmp_path, monkeypatch):
    root = str(tmp_path / "r")
    os.makedirs(root)
    archive_rows = _mp_rows(1, MAY - 5000, 5)  # ts: MAY-5000..MAY-1000
    live_rows = _mp_rows(6, MAY, 6)  # ts: MAY..MAY+5000
    entry = _build_mark_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                                 rows=archive_rows)
    _catalog(root, [entry])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, live_rows)
    _ensure_book_ticker(db, _bt_rows(1, MAY - 5000, 11))

    legacy_db = str(tmp_path / "legacy.sqlite")
    _sqlite_mark(legacy_db, archive_rows + live_rows)

    trade = _base_trade(entry_ts_ms=MAY - 5000, signal_ts_ms=MAY - 5000,
                        rule={"name": "TEST_RULE_HYBRID", "symbol": "ETHUSDT", "direction": "LONG",
                              "tp_bps": 1000.0, "sl_bps": 1000.0, "be_trigger_bps": 500.0,
                              "max_horizon_sec": 10.0, "taker_fee_bps": 4.0})
    now_ms = MAY + 3_600_000

    real_mark_rows_v2 = mod.mark_rows_v2
    quotes_legacy, quotes_new = [], []

    def legacy_mark_rows_v2(root_, symbol_, start_ms_, end_ms_, source_db_path=None):
        conn = sqlite3.connect(legacy_db)
        try:
            return mod.mark_rows(conn, symbol_, start_ms_, end_ms_)
        finally:
            conn.close()

    conn = sqlite3.connect(db)
    try:
        _spy_book_at_v2(monkeypatch, quotes_legacy)
        monkeypatch.setattr(mod, "mark_rows_v2", legacy_mark_rows_v2)
        legacy_result = simulate_counterfactual(conn, trade, now_ms, 3600.0, root=root, source_db_path=db)

        monkeypatch.setattr(mod, "mark_rows_v2", real_mark_rows_v2)
        _spy_book_at_v2(monkeypatch, quotes_new)
        new_result = simulate_counterfactual(conn, trade, now_ms, 3600.0, root=root, source_db_path=db)
    finally:
        conn.close()

    # Sanity: the real (v2) path actually resolved a HYBRID plan for this
    # window -- otherwise this test wouldn't be exercising the coupling it
    # claims to.
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT",
                        start_ms=MAY - 5000, end_ms=MAY + 4000 + 1)
    assert plan.mode == "HYBRID"

    assert legacy_result is not None and new_result is not None
    assert legacy_result == new_result
    for key in ("exit_ts_ms", "exit_mark", "exit_reason", "net_bps", "mfe_bps", "mae_bps", "exit_fill_source"):
        assert legacy_result[key] == new_result[key], key
    assert quotes_legacy and quotes_new
    assert quotes_legacy[-1] == quotes_new[-1]


def test_end_to_end_missing_asof_quote_parity_unaffected_by_mark_migration(tmp_path, monkeypatch):
    # book_ticker table exists but is EMPTY -> book_at_v2 returns None ->
    # simulate_counterfactual must fall back to MARK_FALLBACK identically
    # regardless of which mark_rows variant produced exit_ts_ms/exit_mark.
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    mark_data = [(i + 1, MAY + i * 1000, "ETHUSDT", 3000.0 + (i % 3), None, None) for i in range(11)]
    _sqlite_mark(db, mark_data)
    _ensure_book_ticker(db, ())  # empty -- no quote possible

    trade = _base_trade()
    now_ms = MAY + 3_600_000
    real_mark_rows_v2 = mod.mark_rows_v2

    def legacy_mark_rows_v2(root_, symbol_, start_ms_, end_ms_, source_db_path=None):
        conn = sqlite3.connect(db)
        try:
            return mod.mark_rows(conn, symbol_, start_ms_, end_ms_)
        finally:
            conn.close()

    conn = sqlite3.connect(db)
    try:
        monkeypatch.setattr(mod, "mark_rows_v2", legacy_mark_rows_v2)
        legacy_result = simulate_counterfactual(conn, trade, now_ms, 3600.0, root=root, source_db_path=db)

        monkeypatch.setattr(mod, "mark_rows_v2", real_mark_rows_v2)
        new_result = simulate_counterfactual(conn, trade, now_ms, 3600.0, root=root, source_db_path=db)
    finally:
        conn.close()

    assert legacy_result is not None and new_result is not None
    assert legacy_result == new_result
    assert legacy_result["exit_fill_source"] == "MARK_FALLBACK"
    assert legacy_result["exit_fill"] == legacy_result["exit_mark"]
