"""ASOF Batch 2 consumer migration parity #3 (BATCH-STORAGE-ROTATION-
RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V5).

`tools/research_s34_day_context_scan.py`'s `book_ticker_at` (an
`ORDER BY ts_ms DESC LIMIT 1` staleness-checked lookup with NO `symbol`
parameter at all -- hardcoded directly to the module-level
`SYMBOL="ETHUSDT"` constant in the SQL bind) is migrated to
`book_ticker_at_v2` (reader-backed, same hardcoded-symbol design
preserved exactly -- no signature change beyond adding `root`/
`source_db_path`). `book_ticker_at` is kept unchanged as the parity
oracle.

Call-site audit: `SYMBOL` is a module constant, never overridden
anywhere in this file (confirmed by direct source inspection) -- this
file ALWAYS queries book_ticker for ETHUSDT. book_ticker's only
archived partition is SOLUSDT/2026-04, so real production use of this
file resolves SQLITE_ONLY only. Archive-only and hybrid coverage are
proven with synthetic fixtures, not real production smoke (same
pattern as counter_regime_realfill.py and the btc_microtrend_eth_
quality pilot). Skips if the real source database is not present.
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
from tools.research_s34_day_context_scan import SYMBOL, book_ticker_at, book_ticker_at_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("book_ticker")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_SOURCE_DB), reason="real source database not present in this checkout")


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _old(ts):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return book_ticker_at(conn, ts)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def test_symbol_is_hardcoded_ethusdt():
    assert SYMBOL == "ETHUSDT"


# --- real production smoke: confirms SQLITE_ONLY (no ETHUSDT archive) ---

def test_book_ticker_at_v2_parity_ethusdt_sqlite_only_confirms_no_archive():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    plan = RR.plan_read(_root(), table="book_ticker", symbol="ETHUSDT", start_ms=max_ts - 1000, end_ms=max_ts + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old(max_ts)
    new = book_ticker_at_v2(_root(), max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == old


def test_book_ticker_at_v2_repeated_run_determinism():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    root = _root()
    vals = [book_ticker_at_v2(root, max_ts, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_book_ticker_at_v2_source_mutation_zero():
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    book_ticker_at_v2(root, max_ts, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_no_full_reverify_referenced():
    import inspect
    import tools.research_s34_day_context_scan as mod
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


# ---------------------------------------------------------------------------
# Synthetic fixtures
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


def _rows(start_id, start_ms, n, symbol="ETHUSDT", step_ms=1000):
    return [(start_id + i, start_ms + i * step_ms, symbol, 3000.0, 1.0, 3001.0, 1.0, 3000.5, 0.005, 0.1, 1000.0)
            for i in range(n)]


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
    val = book_ticker_at_v2(root, APR + 5000)
    assert val is not None and val["ts_ms"] == APR + 5000
    lk = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="ETHUSDT", ts_ms=APR + 5000,
                                       columns=("ts_ms", "bid_price", "ask_price", "mid_price"))
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
                                       columns=("ts_ms", "bid_price", "ask_price", "mid_price"), source_db_path=db)
    assert lk.found
    assert lk.provenance["source_type"] == "HYBRID"
    assert lk.provenance["result_source"] == "ARCHIVE"
    val = book_ticker_at_v2(root, q, source_db_path=db)
    assert val is not None and val["ts_ms"] == MAY - 1000


def test_exact_hit_and_between_and_staleness_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 3, step_ms=1000))
    exact = book_ticker_at_v2(root, MAY + 2000, source_db_path=db)
    assert exact["ts_ms"] == MAY + 2000
    between = book_ticker_at_v2(root, MAY + 2000 + 300, source_db_path=db)
    assert between["ts_ms"] == MAY + 2000
    stale = book_ticker_at_v2(root, MAY + 2000 + 5001, source_db_path=db)
    assert stale is None


def test_empty_no_prior_row_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    assert book_ticker_at_v2(root, MAY - 3600_000, source_db_path=db) is None


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
    result = book_ticker_at_v2(root, tied, source_db_path=db)
    assert oracle_id == 3
    assert result["bid"] == 3020.0


def test_symbol_isolation_synthetic(tmp_path):
    """Even though book_ticker_at_v2 hardcodes SYMBOL internally, confirm
    it never picks up a different symbol's row sharing the same ts_ms."""
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "SOLUSDT", 100.0, 1.0, 100.5, 1.0, 100.25, 0.005, 0.1, 1000.0),
                 (2, MAY, "ETHUSDT", 3000.0, 1.0, 3001.0, 1.0, 3000.5, 0.005, 0.1, 1000.0)])
    result = book_ticker_at_v2(root, MAY, source_db_path=db)
    assert result["bid"] == 3000.0


def test_provenance_correctness_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    lk = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="ETHUSDT", ts_ms=MAY + 2000,
                                       columns=("ts_ms", "bid_price", "ask_price", "mid_price"), source_db_path=db)
    assert lk.provenance["source_type"] == "SQLITE_ONLY"
    assert lk.provenance["result_source"] == "SQLITE"
    assert lk.provenance["ordering"] == "(ts_ms DESC, id DESC)"


def test_trust_failure_fails_closed_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                            rows=_rows(1, APR, 10), corrupt_manifest=True)
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        book_ticker_at_v2(root, APR + 5000)
