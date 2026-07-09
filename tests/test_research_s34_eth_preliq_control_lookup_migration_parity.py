"""ASOF Batch 3 consumer migration parity #1 (BATCH-STORAGE-ROTATION-
RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V6).

`tools/research_s34_eth_preliq_control.py`'s `_book_at_or_before` (an
`ORDER BY ts_ms DESC LIMIT 1` as-of point lookup) is migrated to
`_book_at_or_before_v2` (reader-backed). `_book_at_or_before` is kept
unchanged as the parity oracle.

Call-site audit: the lookup's symbol is HARDCODED to the string literal
`'ETHUSDT'` in both the oracle SQL and `_book_at_or_before_v2` (this file
never varies symbol -- confirmed by direct source inspection). book_ticker's
only archived partition is SOLUSDT/2026-04 -- ETHUSDT has NO archive
coverage, so real production use of this file resolves SQLITE_ONLY only.
Archive-only and hybrid coverage are therefore proven with synthetic
fixtures, NOT real production smoke. Skips real-smoke tests if the source
database is not present.

The file's other book_ticker query, `_book_series` (a bounded BETWEEN range
read), is deliberately left on direct SQL -- out of scope for the as-of
point-lookup track -- and is not exercised here.
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
import tools.research_s34_eth_preliq_control as mod
from tools.research_s34_eth_preliq_control import _book_at_or_before, _book_at_or_before_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("book_ticker")

pytestmark = pytest.mark.filterwarnings("ignore")


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old_row(ts):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        conn.row_factory = sqlite3.Row
        return _book_at_or_before(conn, ts)
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


# --- real production smoke: this file's hardcoded ETHUSDT resolves SQLITE_ONLY,
# and the reader-backed path is byte-for-byte identical to the direct oracle ---

def test_v2_parity_ethusdt_sqlite_only_exact_hit():
    _needs_real_db()
    max_ts = _eth_max_ts()
    assert max_ts is not None
    plan = RR.plan_read(_root(), table="book_ticker", symbol="ETHUSDT", start_ms=max_ts - 1000, end_ms=max_ts + 1)
    assert plan.mode == "SQLITE_ONLY"  # confirms the claim: no ETHUSDT book_ticker archive exists
    old = _old_row(max_ts)
    new = _book_at_or_before_v2(_root(), max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None and new is not None
    assert dict(old) == new  # selected-column + value + tie-break parity


def test_v2_parity_ethusdt_between_timestamps():
    _needs_real_db()
    max_ts = _eth_max_ts()
    q = int(max_ts) - 1234  # generally lands between rows -> earlier row via <=
    old = _old_row(q)
    new = _book_at_or_before_v2(_root(), q, source_db_path=REAL_SOURCE_DB)
    assert old is not None and new is not None
    assert dict(old) == new
    assert int(new["ts_ms"]) <= q


def test_v2_repeated_run_determinism():
    _needs_real_db()
    max_ts = _eth_max_ts()
    root = _root()
    vals = [_book_at_or_before_v2(root, max_ts, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    _book_at_or_before_v2(root, _eth_max_ts(), source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_no_full_reverify_referenced():
    import inspect
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_book_series_left_on_direct_sql_out_of_scope():
    # The range read is intentionally NOT migrated in this gate.
    import inspect
    src = inspect.getsource(mod._book_series)
    assert "book_ticker" in src and "ORDER BY ts_ms ASC" in src
    assert "lookup_latest_at_or_before" not in src


# ---------------------------------------------------------------------------
# Synthetic fixtures: since this file's ETHUSDT calls never touch a real
# archive, archive-only / hybrid / trust coverage is proven here. The migrated
# helper itself is symbol-generic; only THIS file's call site fixes ETHUSDT.
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
    val = _book_at_or_before_v2(root, APR + 5000)
    assert val is not None and val["ts_ms"] == APR + 5000
    assert val["bid_price"] == 3005.0  # 6th row (index 5)
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
    _sqlite(db, [])  # live empty -> query after archive end falls back to archive
    q = MAY + 500
    lk = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="ETHUSDT", ts_ms=q,
                                       columns=mod._BOOK_COLS, source_db_path=db)
    assert lk.found
    assert lk.provenance["source_type"] == "HYBRID"
    assert lk.provenance["result_source"] == "ARCHIVE"
    val = _book_at_or_before_v2(root, q, source_db_path=db)
    assert val is not None and val["ts_ms"] == MAY - 1000


def test_exact_hit_and_between_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 3, step_ms=1000))
    exact = _book_at_or_before_v2(root, MAY + 2000, source_db_path=db)
    assert exact["ts_ms"] == MAY + 2000
    between = _book_at_or_before_v2(root, MAY + 2000 + 300, source_db_path=db)
    assert between["ts_ms"] == MAY + 2000


def test_empty_no_prior_row_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    assert _book_at_or_before_v2(root, MAY - 3600_000, source_db_path=db) is None


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
    result = _book_at_or_before_v2(root, tied, source_db_path=db)
    assert oracle_id == 3
    assert result["bid_price"] == 3020.0  # helper's (ts DESC, id DESC) tie-break matches the oracle


def test_symbol_mismatch_safety_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 1.0, 3001.0, 1.0, 3000.5, 0.005, 0.1, 1000.0),
                 (2, MAY, "SOLUSDT", 100.0, 1.0, 100.5, 1.0, 100.25, 0.005, 0.1, 1000.0)])
    # This file's helper is hardcoded ETHUSDT -> it must return the ETH row.
    eth = _book_at_or_before_v2(root, MAY, source_db_path=db)
    assert eth["bid_price"] == 3000.0
    # And the underlying helper never leaks ETH's row into a SOLUSDT lookup.
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
        _book_at_or_before_v2(root, APR + 5000)
