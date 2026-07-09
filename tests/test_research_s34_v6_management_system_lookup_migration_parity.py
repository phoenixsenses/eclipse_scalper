"""ASOF Batch 6 (final) consumer migration parity #3 (BATCH-STORAGE-
ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V9).

`tools/research_s34_v6_management_system.py`'s `book_at_or_before` (an
`ORDER BY ts_ms DESC LIMIT 1` as-of point lookup returning
`{ts_ms,bid,ask,mid}`) is migrated to `book_at_or_before_v2` (reader-
backed). `book_at_or_before` is kept unchanged as the parity oracle.

Call-site audit: symbol is HARDCODED directly in the oracle's SQL
(`WHERE symbol='ETHUSDT'`) -- there is no symbol parameter in this
function at all, so it can never vary. book_ticker's only archived
partition is SOLUSDT/2026-04 -- ETHUSDT has NO archive coverage, so real
production use of this file resolves SQLITE_ONLY only. Archive-only and
hybrid coverage are proven with synthetic fixtures.

This file has two OTHER queries left on direct SQL, deliberately out of
scope for this gate:
  - `first_book_cross`: forward-scanning bounded range read on book_ticker
    with a price predicate (ORDER BY ts_ms ASC LIMIT 1), the opposite
    direction/semantics from lookup_latest_at_or_before.
  - `through_quote_notional`: aggregate SUM over a bounded range read on
    agg_trades, not a point lookup at all.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
from ami.storage.registry import get_table_spec
import tools.research_s34_v6_management_system as mod
from tools.research_s34_v6_management_system import book_at_or_before, book_at_or_before_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("book_ticker")

pytestmark = pytest.mark.filterwarnings("ignore")


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old(ts):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return book_at_or_before(conn, ts)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _eth_max_ts():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- real production smoke: hardcoded ETHUSDT resolves SQLITE_ONLY ---

def test_v2_parity_ethusdt_sqlite_only_exact_hit():
    _needs_real_db()
    max_ts = _eth_max_ts()
    assert max_ts is not None
    plan = RR.plan_read(_root(), table="book_ticker", symbol="ETHUSDT", start_ms=max_ts - 1000, end_ms=max_ts + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old(max_ts)
    new = book_at_or_before_v2(_root(), max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None and new is not None
    assert new == old


def test_v2_parity_ethusdt_between_timestamps():
    _needs_real_db()
    q = int(_eth_max_ts()) - 1234
    old = _old(q)
    new = book_at_or_before_v2(_root(), q, source_db_path=REAL_SOURCE_DB)
    assert old is not None and new is not None
    assert new == old
    assert int(new["ts_ms"]) <= q


def test_v2_repeated_run_determinism():
    _needs_real_db()
    max_ts = _eth_max_ts()
    root = _root()
    vals = [book_at_or_before_v2(root, max_ts, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    book_at_or_before_v2(root, _eth_max_ts(), source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_no_full_reverify_referenced():
    import inspect
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_first_book_cross_left_on_direct_sql_out_of_scope():
    import inspect
    src = inspect.getsource(mod.first_book_cross)
    assert "book_ticker" in src and "ORDER BY ts_ms LIMIT 1" in src
    assert "RR.lookup_latest_at_or_before(" not in src


def test_through_quote_notional_left_on_direct_sql_out_of_scope():
    import inspect
    src = inspect.getsource(mod.through_quote_notional)
    assert "agg_trades" in src and "SUM(notional)" in src
    assert "RR.lookup_latest_at_or_before(" not in src


def test_simulate_tick_queue_event_uses_v2():
    import inspect
    src = inspect.getsource(mod.simulate_tick_queue_event)
    assert "book_at_or_before_v2(root, end_ts" in src


# ---------------------------------------------------------------------------
# Synthetic fixtures for archive-only / hybrid / trust coverage.
# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema
    schema = build_pyarrow_schema(SPEC)
    arrays = [pa.array([r[i] for r in rows], type=schema.field(c).type)
              for i, c in enumerate(SPEC.preserved_columns)]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path, compression="zstd")


def _rows(start_id, start_ms, n, symbol="ETHUSDT", step_ms=1000, bid0=3000.0):
    return [(start_id + i, start_ms + i * step_ms, symbol, bid0 + i, 1.0, bid0 + i + 1.0, 1.0,
             bid0 + i + 0.5, 0.005, 0.1, 1000.0) for i in range(n)]


def _build_archive(root, *, symbol, partition_start_ms, partition_end_ms, rows, corrupt_manifest=False):
    rel = os.path.join("table=book_ticker", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                       f"symbol={symbol}", "year=2026", "month=04", "version=v1")
    fd = os.path.join(root, rel)
    os.makedirs(fd, exist_ok=True)
    pp = os.path.join(fd, "part-00000.parquet")
    _write_parquet(pp, rows)
    manifest = {"source_table": "book_ticker", "symbol": symbol, "venue": SPEC.venue,
                "market_segment": SPEC.market_segment, "row_count": len(rows), "shards": None,
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
    entry = {"archive_identity": f"t-{symbol}", "archive_relative_path": rel, "source_table": "book_ticker",
             "symbol": symbol, "venue": SPEC.venue, "market_segment": SPEC.market_segment,
             "partition_start_ms": partition_start_ms, "partition_end_ms": partition_end_ms, "row_count": len(rows),
             "manifest_sha256": "f" * 64 if corrupt_manifest else _sha256_file(mp),
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": manifest["ordered_scientific_content_hash"]}
    with open(os.path.join(fd, "catalog_entry.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f)
    return entry


def _catalog(root, entries):
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}, f)


BT_DDL = ("CREATE TABLE book_ticker (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, bid_price REAL, "
          "bid_qty REAL, ask_price REAL, ask_qty REAL, mid_price REAL, spread_pct REAL, book_imbalance REAL, "
          "bid_depth_usd REAL)")


def _sqlite(path, rows):
    c = sqlite3.connect(path)
    c.execute(BT_DDL)
    c.execute("CREATE INDEX ix ON book_ticker(symbol, ts_ms)")
    c.executemany("INSERT INTO book_ticker VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


APR = 1775001600000
MAY = 1777593600000


def test_archive_only_lookup_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY, rows=_rows(1, APR, 10))
    _catalog(root, [entry])
    val = book_at_or_before_v2(root, APR + 5000)
    assert val is not None and val["ts_ms"] == APR + 5000
    assert val["bid"] == 3005.0 and val["mid"] == 3005.5
    lk = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="ETHUSDT", ts_ms=APR + 5000,
                                       columns=mod._BOOK_COLS)
    assert lk.provenance["source_type"] == "ARCHIVE_ONLY"


def test_hybrid_fallback_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                           rows=_rows(1, MAY - 3000, 3, step_ms=1000))
    _catalog(root, [entry])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [])
    q = MAY + 500
    lk = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="ETHUSDT", ts_ms=q,
                                       columns=mod._BOOK_COLS, source_db_path=db)
    assert lk.found and lk.provenance["source_type"] == "HYBRID" and lk.provenance["result_source"] == "ARCHIVE"
    val = book_at_or_before_v2(root, q, source_db_path=db)
    assert val is not None and val["ts_ms"] == MAY - 1000


def test_exact_hit_and_between_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 3, step_ms=1000))
    exact = book_at_or_before_v2(root, MAY + 2000, source_db_path=db)
    assert exact["ts_ms"] == MAY + 2000
    between = book_at_or_before_v2(root, MAY + 2000 + 300, source_db_path=db)
    assert between["ts_ms"] == MAY + 2000


def test_empty_no_prior_row_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    assert book_at_or_before_v2(root, MAY - 3600_000, source_db_path=db) is None


def test_tie_break_parity_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    tied = MAY + 1000
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 1.0, 3001.0, 1.0, 3000.5, 0.005, 0.1, 1000.0),
                 (2, tied, "ETHUSDT", 3010.0, 1.0, 3011.0, 1.0, 3010.5, 0.005, 0.1, 1000.0),
                 (3, tied, "ETHUSDT", 3020.0, 1.0, 3021.0, 1.0, 3020.5, 0.005, 0.1, 1000.0)])
    c = sqlite3.connect(db)
    oracle_id = c.execute("SELECT id FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? "
                          "ORDER BY ts_ms DESC, id DESC LIMIT 1", (tied,)).fetchone()[0]
    c.close()
    result = book_at_or_before_v2(root, tied, source_db_path=db)
    assert oracle_id == 3
    assert result["bid"] == 3020.0


def test_symbol_mismatch_safety_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 1.0, 3001.0, 1.0, 3000.5, 0.005, 0.1, 1000.0),
                 (2, MAY, "SOLUSDT", 100.0, 1.0, 100.5, 1.0, 100.25, 0.005, 0.1, 1000.0)])
    eth = book_at_or_before_v2(root, MAY, source_db_path=db)
    assert eth["bid"] == 3000.0
    sol = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="SOLUSDT", ts_ms=MAY,
                                        columns=("ts_ms", "bid_price"), source_db_path=db)
    assert sol.found and sol.row[1] == 100.0


def test_provenance_correctness_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    lk = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="ETHUSDT", ts_ms=MAY + 2000,
                                       columns=mod._BOOK_COLS, source_db_path=db)
    assert lk.provenance["source_type"] == "SQLITE_ONLY"
    assert lk.provenance["result_source"] == "SQLITE"
    assert lk.provenance["symbol"] == "ETHUSDT"
    assert lk.provenance["ordering"] == "(ts_ms DESC, id DESC)"


def test_trust_failure_fails_closed_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                           rows=_rows(1, APR, 10), corrupt_manifest=True)
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        book_at_or_before_v2(root, APR + 5000)
