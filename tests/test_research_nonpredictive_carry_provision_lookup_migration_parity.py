"""ASOF Batch 6 (final) consumer migration parity #1 (BATCH-STORAGE-
ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V9).

`tools/research_nonpredictive_carry_provision.py`'s `book_at_or_before`
(an `ORDER BY ts_ms DESC LIMIT 1` as-of point lookup returning
`{ts_ms,bid,ask,mid}`) is migrated to `book_at_or_before_v2` (reader-
backed). `book_at_or_before` is kept unchanged as the parity oracle.

Call-site audit: `symbol` is a GENUINE runtime parameter, sourced in
`provision_one_event` from `row["symbol"]`. Direct inspection of the real
absorption pool JSON
(`reports/research/s34/S34_ABSORPTION_SYNC_2X2_POOL.json`) shows the
vdepth_band-filtered rows that actually reach this call site span all
three symbols: ETHUSDT (128), BTCUSDT (77), SOLUSDT (90). book_ticker's
only archived partition is SOLUSDT/2026-04 -- and 11 of those real
SOLUSDT rows have `entry_ts_ms` genuinely inside that archived window, so
a REAL archive-backed production smoke is possible here (unlike most
other ASOF files in this project, which only ever touch ETHUSDT/BTCUSDT
and can only prove SQLITE_ONLY for real data). ETHUSDT/BTCUSDT real
invocations still resolve SQLITE_ONLY.

`first_touch` (a forward-scanning bounded range read with a price
predicate, ORDER BY ts_ms ASC LIMIT 1) is left on direct SQL -- out of
scope for this as-of point-lookup gate -- and is not exercised here.
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
import tools.research_nonpredictive_carry_provision as mod
from tools.research_nonpredictive_carry_provision import book_at_or_before, book_at_or_before_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("book_ticker")

pytestmark = pytest.mark.filterwarnings("ignore")

POOL_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "reports", "research", "s34", "S34_ABSORPTION_SYNC_2X2_POOL.json")
APR = 1775001600000
MAY = 1777593600000


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old(symbol, ts):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return book_at_or_before(conn, symbol, ts)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _real_sol_archive_ts():
    if not os.path.exists(POOL_JSON_PATH):
        return None
    payload = json.loads(open(POOL_JSON_PATH, encoding="utf-8").read())
    rows = payload.get("rows") or []
    sol = [r for r in rows if str(r.get("vdepth_band")) in {"v28_40", "v40_60", "v60_plus"}
           and str(r.get("symbol")) == "SOLUSDT"]
    in_window = [r for r in sol if APR <= int(r.get("entry_ts_ms") or 0) < MAY]
    return int(in_window[0]["entry_ts_ms"]) if in_window else None


def test_real_call_site_symbols_span_three_including_solusdt():
    # Confirms the call-site claim structurally: vdepth_band-filtered pool
    # rows span ETHUSDT/BTCUSDT/SOLUSDT, not just one symbol.
    if not os.path.exists(POOL_JSON_PATH):
        pytest.skip("real absorption pool JSON not present in this checkout")
    payload = json.loads(open(POOL_JSON_PATH, encoding="utf-8").read())
    rows = payload.get("rows") or []
    scoped = [r for r in rows if str(r.get("vdepth_band")) in {"v28_40", "v40_60", "v60_plus"}]
    syms = {str(r.get("symbol")) for r in scoped}
    assert {"ETHUSDT", "BTCUSDT", "SOLUSDT"} <= syms


# --- real production smoke: SOLUSDT resolves ARCHIVE-backed (real coverage
# exists); ETHUSDT/BTCUSDT resolve SQLITE_ONLY ---

def test_v2_parity_solusdt_real_archive_backed():
    _needs_real_db()
    ts = _real_sol_archive_ts()
    if ts is None:
        pytest.skip("no real SOLUSDT pool row falls inside the archived 2026-04 window in this checkout")
    plan = RR.plan_read(_root(), table="book_ticker", symbol="SOLUSDT", start_ms=ts - 1000, end_ms=ts + 1)
    assert plan.mode in ("ARCHIVE_ONLY", "HYBRID")
    old = _old("SOLUSDT", ts)
    new = book_at_or_before_v2(_root(), "SOLUSDT", ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None and new is not None
    assert new == old


def test_v2_parity_ethusdt_sqlite_only_exact_hit():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    assert max_ts is not None
    plan = RR.plan_read(_root(), table="book_ticker", symbol="ETHUSDT", start_ms=max_ts - 1000, end_ms=max_ts + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old("ETHUSDT", max_ts)
    new = book_at_or_before_v2(_root(), "ETHUSDT", max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None and new is not None
    assert new == old


def test_v2_repeated_run_determinism():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    root = _root()
    vals = [book_at_or_before_v2(root, "ETHUSDT", max_ts, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    book_at_or_before_v2(root, "ETHUSDT", max_ts, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_no_full_reverify_referenced():
    import inspect
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_first_touch_left_on_direct_sql_out_of_scope():
    import inspect
    src = inspect.getsource(mod.first_touch)
    assert "book_ticker" in src and "ORDER BY ts_ms" in src
    assert "RR.lookup_latest_at_or_before(" not in src


# ---------------------------------------------------------------------------
# Synthetic fixtures for exact-hit / between / empty / tie-break / hybrid /
# trust-failure coverage, symbol-generic.
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


def test_archive_only_lookup_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="SOLUSDT", partition_start_ms=APR, partition_end_ms=MAY, rows=_rows(1, APR, 10, symbol="SOLUSDT"))
    _catalog(root, [entry])
    val = book_at_or_before_v2(root, "SOLUSDT", APR + 5000)
    assert val is not None and val["ts_ms"] == APR + 5000
    assert val["bid"] == 3005.0 and val["mid"] == 3005.5
    lk = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="SOLUSDT", ts_ms=APR + 5000,
                                       columns=mod._BOOK_COLS)
    assert lk.provenance["source_type"] == "ARCHIVE_ONLY"


def test_hybrid_fallback_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="SOLUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                           rows=_rows(1, MAY - 3000, 3, symbol="SOLUSDT", step_ms=1000))
    _catalog(root, [entry])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [])
    q = MAY + 500
    lk = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="SOLUSDT", ts_ms=q,
                                       columns=mod._BOOK_COLS, source_db_path=db)
    assert lk.found and lk.provenance["source_type"] == "HYBRID" and lk.provenance["result_source"] == "ARCHIVE"
    val = book_at_or_before_v2(root, "SOLUSDT", q, source_db_path=db)
    assert val is not None and val["ts_ms"] == MAY - 1000


def test_exact_hit_and_between_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 3, step_ms=1000))
    exact = book_at_or_before_v2(root, "ETHUSDT", MAY + 2000, source_db_path=db)
    assert exact["ts_ms"] == MAY + 2000
    between = book_at_or_before_v2(root, "ETHUSDT", MAY + 2000 + 300, source_db_path=db)
    assert between["ts_ms"] == MAY + 2000


def test_empty_no_prior_row_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    assert book_at_or_before_v2(root, "ETHUSDT", MAY - 3600_000, source_db_path=db) is None


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
    result = book_at_or_before_v2(root, "ETHUSDT", tied, source_db_path=db)
    assert oracle_id == 3
    assert result["bid"] == 3020.0


def test_symbol_mismatch_safety_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 1.0, 3001.0, 1.0, 3000.5, 0.005, 0.1, 1000.0),
                 (2, MAY, "SOLUSDT", 100.0, 1.0, 100.5, 1.0, 100.25, 0.005, 0.1, 1000.0),
                 (3, MAY, "BTCUSDT", 60000.0, 1.0, 60001.0, 1.0, 60000.5, 0.005, 0.1, 1000.0)])
    eth = book_at_or_before_v2(root, "ETHUSDT", MAY, source_db_path=db)
    sol = book_at_or_before_v2(root, "SOLUSDT", MAY, source_db_path=db)
    btc = book_at_or_before_v2(root, "BTCUSDT", MAY, source_db_path=db)
    assert eth["bid"] == 3000.0
    assert sol["bid"] == 100.0
    assert btc["bid"] == 60000.0


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
    entry = _build_archive(root, symbol="SOLUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                           rows=_rows(1, APR, 10, symbol="SOLUSDT"), corrupt_manifest=True)
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        book_at_or_before_v2(root, "SOLUSDT", APR + 5000)
