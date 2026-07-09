"""Range-read consumer migration V4 parity #3 (BATCH-STORAGE-ROTATION-
RETENTION-RANGE-READ-CONSUMER-MIGRATION-V4).

`tools/research_s34_symbol_compare.py` has THREE allowlisted range-read
patterns, all exact precedents from prior gates, each extracted into a
named oracle and migrated to a reader-backed `_v2` (kept unchanged as
parity references):
  - `_day_high_low` / `_day_high_low_v2`   (mark_prices MAX/MIN, from
    `day_so_far` -- same shape as V2's sell_reversal_filter.py)
  - `_day_agg_count` / `_day_agg_count_v2` (agg_trades COUNT, from
    `day_so_far` -- same shape as V2's sell_reversal_filter.py)
  - `_horizon_marks` / `_horizon_marks_v2` (mark_prices window, from
    `simulate_route` -- same shape as V3's sell/buy_reversal files)

`symbol` is a GENUINE runtime parameter for all three (SYMBOLS =
BTCUSDT/ETHUSDT/SOLUSDT, iterated in main()). Both allowlisted tables have
real archive partitions in DIFFERENT months (agg_trades/ETHUSDT/2026-02,
mark_prices/ETHUSDT/2026-05), so real ARCHIVE_ONLY/HYBRID/SQLITE_ONLY
smokes are proven per-table by direct invocation.

DELIBERATELY OUT OF SCOPE (asserted below to remain on direct SQL):
  - `mark_at`/`ret_bps`: ASOF-style point lookups -- belong to the ASOF
    track, not this range-read gate.
  - `load_clusters`'s `liquidations` GROUP BY aggregate and `day_so_far`'s
    `liquidations` SUM -- out-of-allowlist table.

Range-boundary note: all three oracles use an INCLUSIVE upper bound
(`ts_ms<=?`); the reader uses half-open `[start, end)`, so each `_v2`
passes `end_ms+1`. Proven exact via boundary tests.
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
import tools.research_s34_symbol_compare as mod
from tools.research_s34_symbol_compare import (
    _day_high_low, _day_high_low_v2, _day_agg_count, _day_agg_count_v2,
    _horizon_marks, _horizon_marks_v2,
)

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
MARK_SPEC = get_table_spec("mark_prices")
AGG_SPEC = get_table_spec("agg_trades")

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


def _old_high_low(symbol, s, e):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _day_high_low(conn, symbol, s, e)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_agg_count(symbol, s, e):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _day_agg_count(conn, symbol, s, e)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_horizon_marks(symbol, s, e):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _horizon_marks(conn, symbol, s, e)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _eth_max_ts(table):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute(f"SELECT MAX(ts_ms) FROM {table} WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- _day_high_low: real ARCHIVE_ONLY / SQLITE_ONLY / HYBRID (mark_prices) ---

def test_high_low_parity_archive_only_real():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old_high_low("ETHUSDT", s, e)
    new = _day_high_low_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert old[0] is not None
    assert new[0] == pytest.approx(old[0], rel=1e-12)
    assert new[1] == pytest.approx(old[1], rel=1e-12)


def test_high_low_parity_hybrid_real_boundary():
    _needs_real_db()
    s, e = MARK_PARTITION_END_MS - 120_000, MARK_PARTITION_END_MS + 60_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "HYBRID"
    old = _old_high_low("ETHUSDT", s, e)
    new = _day_high_low_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new[0] == pytest.approx(old[0], rel=1e-12)
    assert new[1] == pytest.approx(old[1], rel=1e-12)


def test_high_low_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _eth_max_ts("mark_prices")
    s, e = max_ts - 900_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old_high_low("ETHUSDT", s, e)
    new = _day_high_low_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new[0] == pytest.approx(old[0], rel=1e-12)
    assert new[1] == pytest.approx(old[1], rel=1e-12)


def test_high_low_empty_range_returns_none_none():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000
    assert _old_high_low("ETHUSDT", lo, hi) == (None, None)
    assert _day_high_low_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB) == (None, None)


# --- _day_agg_count: real ARCHIVE_ONLY / SQLITE_ONLY / HYBRID (agg_trades) ---

def test_agg_count_parity_archive_only_real():
    _needs_real_db()
    s, e = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 120_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old_n = _old_agg_count("ETHUSDT", s, e)
    new_n = _day_agg_count_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert old_n > 0
    assert new_n == old_n


def test_agg_count_parity_hybrid_real_boundary():
    _needs_real_db()
    s, e = AGG_PARTITION_END_MS - 120_000, AGG_PARTITION_END_MS + 60_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "HYBRID"
    old_n = _old_agg_count("ETHUSDT", s, e)
    new_n = _day_agg_count_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new_n == old_n


def test_agg_count_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _eth_max_ts("agg_trades")
    s, e = max_ts - 60_000, max_ts
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_n = _old_agg_count("ETHUSDT", s, e)
    new_n = _day_agg_count_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new_n == old_n


def test_agg_count_empty_range_returns_zero():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000
    assert _old_agg_count("ETHUSDT", lo, hi) == 0
    assert _day_agg_count_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB) == 0


# --- _horizon_marks: real ARCHIVE_ONLY / SQLITE_ONLY / HYBRID (mark_prices) ---

def test_horizon_marks_parity_archive_only_real():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old_horizon_marks("ETHUSDT", s, e)
    new = _horizon_marks_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert len(new) == len(old) and len(old) > 0
    assert new == old


def test_horizon_marks_parity_hybrid_real_boundary():
    _needs_real_db()
    s, e = MARK_PARTITION_END_MS - 120_000, MARK_PARTITION_END_MS + 60_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "HYBRID"
    old = _old_horizon_marks("ETHUSDT", s, e)
    new = _horizon_marks_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_horizon_marks_inclusive_end_boundary_parity_real():
    _needs_real_db()
    s = MARK_ARCHIVE_START_MS
    rows = _old_horizon_marks("ETHUSDT", s, s + 300_000)
    assert rows
    last_ts = rows[-1][0]
    at = _horizon_marks_v2(_root(), "ETHUSDT", s, int(last_ts), source_db_path=REAL_SOURCE_DB)
    before = _horizon_marks_v2(_root(), "ETHUSDT", s, int(last_ts) - 1, source_db_path=REAL_SOURCE_DB)
    assert len(at) == len(before) + 1
    assert at == _old_horizon_marks("ETHUSDT", s, int(last_ts))


def test_horizon_marks_empty_range_parity():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000
    assert _old_horizon_marks("ETHUSDT", lo, hi) == []
    assert _horizon_marks_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB) == []


# --- provenance / determinism / mutation / reverify (representative) ---

def test_provenance_reports_archive_source():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert prov["row_count"] > 0


def test_repeated_run_determinism():
    _needs_real_db()
    s, e = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000
    vals = [_day_agg_count_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
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
    _day_high_low_v2(root, "ETHUSDT", MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    _day_agg_count_v2(root, "ETHUSDT", AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    _horizon_marks_v2(root, "ETHUSDT", MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


# --- out-of-scope queries remain on direct SQL ---

def test_mark_at_asof_lookup_left_on_direct_sql():
    import inspect
    src = inspect.getsource(mod.mark_at)
    assert "mark_prices" in src and "limit 1" in src
    assert "RR.execute_read(" not in src


def test_liquidations_queries_left_on_direct_sql_out_of_allowlist():
    import inspect
    src_clusters = inspect.getsource(mod.load_clusters)
    src_day = inspect.getsource(mod.day_so_far)
    assert "liquidations" in src_clusters
    assert "liquidations" in src_day
    assert "RR.execute_read(" not in src_clusters


def test_day_so_far_and_simulate_route_use_v2_not_inline_queries():
    import inspect
    src_day = inspect.getsource(mod.day_so_far)
    src_route = inspect.getsource(mod.simulate_route)
    assert "_day_high_low_v2(root," in src_day and "_day_agg_count_v2(root," in src_day
    assert "max(mark_price)" not in src_day.lower()
    assert "count(*)" not in src_day.lower()
    assert "_horizon_marks_v2(root," in src_route
    assert "select ts_ms, mark_price" not in src_route.lower()


# ---------------------------------------------------------------------------
# Synthetic fixtures: symbol-mismatch safety across all 3 migrated
# functions + trust-failure fail-closed.
# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _catalog(root, entries):
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}, f)


MP_DDL = ("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
          "mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)")
AT_DDL = ("CREATE TABLE agg_trades (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, price REAL, "
          "quantity REAL, notional REAL, is_buyer_maker INTEGER)")


def _sqlite_mark(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute(MP_DDL)
    c.execute("CREATE INDEX ix ON mark_prices(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


def _sqlite_agg(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute(AT_DDL)
    c.execute("CREATE INDEX ix ON agg_trades(symbol, ts_ms)")
    c.executemany("INSERT INTO agg_trades VALUES (?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


MAY = 1777593600000


def test_high_low_and_horizon_marks_symbol_mismatch_safety_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_mark(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0, None),
        (2, MAY + 1000, "ETHUSDT", 3010.0, 0.0, None),
        (3, MAY + 500, "SOLUSDT", 99999.0, 0.0, None),
    ])
    eth_hl = _day_high_low_v2(root, "ETHUSDT", MAY, MAY + 1000, source_db_path=db)
    assert eth_hl == (3010.0, 3000.0)  # SOLUSDT's 99999 excluded
    sol_hl = _day_high_low_v2(root, "SOLUSDT", MAY, MAY + 1000, source_db_path=db)
    assert sol_hl == (99999.0, 99999.0)
    eth_marks = _horizon_marks_v2(root, "ETHUSDT", MAY, MAY + 1000, source_db_path=db)
    assert eth_marks == [(MAY, 3000.0), (MAY + 1000, 3010.0)]


def test_agg_count_symbol_mismatch_safety_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite_agg(db, [
        (1, MAY, "ETHUSDT", 3000.0, 1.0, 3000.0, 0),
        (2, MAY, "SOLUSDT", 100.0, 1.0, 100.0, 0),
        (3, MAY, "SOLUSDT", 100.0, 1.0, 100.0, 1),
    ])
    assert _day_agg_count_v2(root, "ETHUSDT", MAY, MAY, source_db_path=db) == 1
    assert _day_agg_count_v2(root, "SOLUSDT", MAY, MAY, source_db_path=db) == 2


def test_trust_failure_fails_closed_synthetic_mark(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema

    root = str(tmp_path / "r")
    rel = os.path.join("table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                       "symbol=ETHUSDT", "year=2026", "month=05", "version=v1")
    final_dir = os.path.join(root, rel)
    os.makedirs(final_dir, exist_ok=True)
    schema = build_pyarrow_schema(MARK_SPEC)
    rows = [(1, MAY, "ETHUSDT", 3000.0, 0.0, None)]
    arrays = [pa.array([r[i] for r in rows], type=schema.field(c).type)
              for i, c in enumerate(MARK_SPEC.preserved_columns)]
    pp = os.path.join(final_dir, "part-00000.parquet")
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), pp, compression="zstd")
    manifest = {"source_table": "mark_prices", "symbol": "ETHUSDT", "venue": MARK_SPEC.venue,
                "market_segment": MARK_SPEC.market_segment, "row_count": 1, "shards": None,
                "ordered_scientific_content_hash": "irrelevant", "partition_id": "corrupt-test",
                "parquet_path": os.path.join(rel, "part-00000.parquet"),
                "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
                "source_watermark_field": "id", "source_watermark_value": 1,
                "parquet_sha256": _sha256_file(pp)}
    with open(os.path.join(final_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("ok\n")
    entry = {"archive_identity": "corrupt-test", "archive_relative_path": rel, "source_table": "mark_prices",
             "symbol": "ETHUSDT", "venue": MARK_SPEC.venue, "market_segment": MARK_SPEC.market_segment,
             "partition_start_ms": MAY, "partition_end_ms": MAY + 3600_000, "row_count": 1,
             "manifest_sha256": "0" * 64,
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": "irrelevant"}
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        _day_high_low_v2(root, "ETHUSDT", MAY, MAY + 1_000)
    with pytest.raises(RR.ArchiveTrustError):
        _horizon_marks_v2(root, "ETHUSDT", MAY, MAY + 1_000)
