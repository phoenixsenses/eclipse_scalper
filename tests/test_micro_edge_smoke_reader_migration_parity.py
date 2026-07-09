"""Range-read consumer migration V4 parity #2 (BATCH-STORAGE-ROTATION-
RETENTION-RANGE-READ-CONSUMER-MIGRATION-V4).

`tools/micro_edge_smoke.py`'s `_load_symbol_trades_marks_and_liqs` bundled
three range reads. The two allowlisted ones are extracted into named
oracles and migrated to reader-backed counterparts via plan_read/
execute_read:
  - agg_trades range -> `_agg_trades_range` (oracle) / `_agg_trades_range_v2`
  - mark_prices range -> `_mark_prices_range` (oracle) / `_mark_prices_range_v2`
Both oracles are kept unchanged as parity references.

`symbol` is a genuine runtime parameter (`--symbols` CLI arg, default
BTCUSDT,ETHUSDT). Both allowlisted tables have real archive partitions in
DIFFERENT months (agg_trades/ETHUSDT/2026-02, mark_prices/ETHUSDT/
2026-05), so real ARCHIVE_ONLY smoke is proven per-table, plus per-table
HYBRID and a SQLITE_ONLY recent-live window.

Finding (pre-existing, not introduced by this migration): this file's
`conn` was opened WITHOUT `mode=ro` (`sqlite3.connect(db,
check_same_thread=False)`), unlike almost every other migrated file.
Migrating the two allowlisted reads onto the reader (whose SQLite fallback
always goes through `SRC.open_read_only`) closes that gap for THOSE two
reads; the still-direct `liquidations` read keeps using the original,
non-mode=ro `conn` (out-of-allowlist, out of scope for this gate).

DELIBERATELY OUT OF SCOPE (asserted below to remain on direct SQL):
  - the `liquidations` range read inside `_load_symbol_trades_marks_and_liqs`
    -- out-of-allowlist table, no archive partition / reader support.

Range-boundary note: the oracles use an INCLUSIVE upper bound
(`ts_ms<=?`); the reader uses half-open `[start, end)`, so both `_v2`
helpers pass `end_ms+1`. Proven exact via boundary tests.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
from ami.storage.registry import get_table_spec
import tools.micro_edge_smoke as mod
from tools.micro_edge_smoke import (
    _agg_trades_range, _agg_trades_range_v2, _mark_prices_range, _mark_prices_range_v2,
)

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
AGG_SPEC = get_table_spec("agg_trades")
MARK_SPEC = get_table_spec("mark_prices")

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


def _old_agg(symbol, s, e):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _agg_trades_range(conn, symbol, s, e)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_mark(symbol, s, e):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _mark_prices_range(conn, symbol, s, e)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _digest(seq):
    h = hashlib.sha256()
    for item in seq:
        h.update(repr(item).encode())
    return h.hexdigest()


def _eth_max_ts(table):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute(f"SELECT MAX(ts_ms) FROM {table} WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- agg_trades: real ARCHIVE_ONLY / SQLITE_ONLY / HYBRID ---

def test_agg_trades_parity_archive_only_real():
    _needs_real_db()
    s, e = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 120_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old_agg("ETHUSDT", s, e)
    new = _agg_trades_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert len(new) == len(old) and len(old) > 0
    assert new == old
    assert _digest(new) == _digest(old)


def test_agg_trades_parity_hybrid_real_boundary():
    _needs_real_db()
    s, e = AGG_PARTITION_END_MS - 120_000, AGG_PARTITION_END_MS + 60_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "HYBRID"
    old = _old_agg("ETHUSDT", s, e)
    new = _agg_trades_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_agg_trades_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _eth_max_ts("agg_trades")
    s, e = max_ts - 60_000, max_ts
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old_agg("ETHUSDT", s, e)
    new = _agg_trades_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


# --- mark_prices: real ARCHIVE_ONLY / SQLITE_ONLY / HYBRID ---

def test_mark_prices_parity_archive_only_real():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old_mark("ETHUSDT", s, e)
    new = _mark_prices_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert len(new) == len(old) and len(old) > 0
    assert new == old


def test_mark_prices_parity_hybrid_real_boundary():
    _needs_real_db()
    s, e = MARK_PARTITION_END_MS - 120_000, MARK_PARTITION_END_MS + 60_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "HYBRID"
    old = _old_mark("ETHUSDT", s, e)
    new = _mark_prices_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_mark_prices_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _eth_max_ts("mark_prices")
    s, e = max_ts - 900_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old_mark("ETHUSDT", s, e)
    new = _mark_prices_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_provenance_reports_archive_source_agg():
    _needs_real_db()
    s, e = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 120_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "price", "quantity", "is_buyer_maker"), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["table"] == "agg_trades"
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert len(prov["archive_segments"]) == 1
    assert prov["archive_segments"][0]["manifest_sha256"]
    assert prov["row_count"] > 0


def test_inclusive_end_boundary_parity_real_mark():
    _needs_real_db()
    s = MARK_ARCHIVE_START_MS
    rows = _old_mark("ETHUSDT", s, s + 300_000)
    assert rows
    last_ts = rows[-1][0]
    at = _mark_prices_range_v2(_root(), "ETHUSDT", s, int(last_ts), source_db_path=REAL_SOURCE_DB)
    before = _mark_prices_range_v2(_root(), "ETHUSDT", s, int(last_ts) - 1, source_db_path=REAL_SOURCE_DB)
    assert len(at) == len(before) + 1
    assert at == _old_mark("ETHUSDT", s, int(last_ts))


def test_empty_range_parity():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000
    assert _old_agg("ETHUSDT", lo, hi) == []
    assert _agg_trades_range_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB) == []
    assert _old_mark("ETHUSDT", lo, hi) == []
    assert _mark_prices_range_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB) == []


def test_repeated_run_determinism():
    _needs_real_db()
    s, e = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000
    vals = [_agg_trades_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_no_full_reverify_referenced():
    import inspect
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    _agg_trades_range_v2(root, "ETHUSDT", AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    _mark_prices_range_v2(root, "ETHUSDT", MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_liquidations_left_on_direct_sql_out_of_allowlist():
    import inspect
    src = inspect.getsource(mod._load_symbol_trades_marks_and_liqs)
    assert "liquidations" in src
    assert src.count("RR.execute_read(") == 0  # this function itself doesn't call execute_read directly
    assert "_agg_trades_range_v2(" in src and "_mark_prices_range_v2(" in src


def test_analyze_symbol_threads_root_and_source_db_path():
    import inspect
    src = inspect.getsource(mod.analyze_symbol)
    assert "root=root" in src and "source_db_path=source_db_path" in src


# ---------------------------------------------------------------------------
# Synthetic fixtures: symbol-mismatch safety + trust-failure fail-closed.
# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _catalog(root, entries):
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}, f)


AT_DDL = ("CREATE TABLE agg_trades (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, price REAL, "
          "quantity REAL, notional REAL, is_buyer_maker INTEGER)")
MP_DDL = ("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
          "mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)")


def _sqlite_agg(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute(AT_DDL)
    c.execute("CREATE INDEX ix ON agg_trades(symbol, ts_ms)")
    c.executemany("INSERT INTO agg_trades VALUES (?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


def _sqlite_mark(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute(MP_DDL)
    c.execute("CREATE INDEX ix ON mark_prices(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


MAY = 1777593600000


def test_symbol_mismatch_safety_synthetic_agg(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_agg(db, [
        (1, MAY, "ETHUSDT", 3000.0, 1.0, 3000.0, 0),
        (2, MAY, "SOLUSDT", 100.0, 1.0, 100.0, 0),
    ])
    eth = _agg_trades_range_v2(root, "ETHUSDT", MAY, MAY, source_db_path=db)
    sol = _agg_trades_range_v2(root, "SOLUSDT", MAY, MAY, source_db_path=db)
    assert eth == [(MAY, 3000.0, 1.0, 0)]
    assert sol == [(MAY, 100.0, 1.0, 0)]


def test_symbol_mismatch_safety_synthetic_mark(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0, None),
        (2, MAY, "SOLUSDT", 100.0, 0.0, None),
    ])
    eth = _mark_prices_range_v2(root, "ETHUSDT", MAY, MAY, source_db_path=db)
    sol = _mark_prices_range_v2(root, "SOLUSDT", MAY, MAY, source_db_path=db)
    assert eth == [(MAY, 3000.0)]
    assert sol == [(MAY, 100.0)]


def test_trust_failure_fails_closed_synthetic_agg(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema

    root = str(tmp_path / "r")
    rel = os.path.join("table=agg_trades", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                       "symbol=ETHUSDT", "year=2026", "month=02", "version=v1")
    final_dir = os.path.join(root, rel)
    os.makedirs(final_dir, exist_ok=True)
    schema = build_pyarrow_schema(AGG_SPEC)
    rows = [(1, MAY, "ETHUSDT", 3000.0, 1.0, 3000.0, 0)]
    arrays = [pa.array([r[i] for r in rows], type=schema.field(c).type)
              for i, c in enumerate(AGG_SPEC.preserved_columns)]
    pp = os.path.join(final_dir, "part-00000.parquet")
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), pp, compression="zstd")
    manifest = {"source_table": "agg_trades", "symbol": "ETHUSDT", "venue": AGG_SPEC.venue,
                "market_segment": AGG_SPEC.market_segment, "row_count": 1, "shards": None,
                "ordered_scientific_content_hash": "irrelevant", "partition_id": "corrupt-test",
                "parquet_path": os.path.join(rel, "part-00000.parquet"),
                "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
                "source_watermark_field": "id", "source_watermark_value": 1,
                "parquet_sha256": _sha256_file(pp)}
    with open(os.path.join(final_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("ok\n")
    entry = {"archive_identity": "corrupt-test", "archive_relative_path": rel, "source_table": "agg_trades",
             "symbol": "ETHUSDT", "venue": AGG_SPEC.venue, "market_segment": AGG_SPEC.market_segment,
             "partition_start_ms": MAY, "partition_end_ms": MAY + 3600_000, "row_count": 1,
             "manifest_sha256": "0" * 64,
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": "irrelevant"}
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        _agg_trades_range_v2(root, "ETHUSDT", MAY, MAY + 1_000)
