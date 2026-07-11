"""Range-read consumer migration (BATCH-STORAGE-ROTATION-RETENTION-
RANGE-READ-CONSUMER-MIGRATION-V9), target 2 of 3.

`tools/research_s34_trailing_oos_realfill.py`'s bounded range read inside
`path_marks()` (`mark_prices`, `ts_ms>=? AND ts_ms<=? ORDER BY ts_ms` --
INCLUSIVE lower / INCLUSIVE upper) was preserved unchanged as the parity
oracle (`path_marks`) and migrated to `path_marks_v2()` (reader-backed via
`plan_read`/`execute_read`). `path_marks()` is no longer called from
`simulate_current_be30`/`simulate_trail_half_mfe` (the live path).

Call-site audit: `symbol` is a genuine function parameter, but both live
call sites always pass the module constant `SYMBOL="ETHUSDT"` -- the helper
stays symbol-generic regardless. The two simulate functions were threaded
with `root`/`source_db_path` (both already resolved in `main`).

Range-boundary note: the oracle uses `ts_ms>=?` (INCLUSIVE lower) and
`ts_ms<=?` (INCLUSIVE upper). `plan_read`/`execute_read` use the reader's
half-open `[start_ms, end_ms)` convention. Since `ts_ms` is an integer
column, `ts_ms<=hi` is exactly equivalent to `ts_ms<hi+1` -- so
`path_marks_v2` passes `start_ms=lo, end_ms=hi+1`, proven bit-for-bit
below.

Out-of-scope, untouched (this gate is range-read only):
  * `mark_at()` -- dual-direction ASOF (`ORDER BY ts_ms {asc|desc} LIMIT
    1`), left on direct SQL.
  * `book_ticker_at_v2()` -- already migrated in a prior ASOF gate.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
from ami.storage.registry import get_table_spec
import tools.research_s34_trailing_oos_realfill as mod
from tools.research_s34_trailing_oos_realfill import path_marks, path_marks_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")
SYMBOL = mod.SYMBOL
HORIZON_MS = mod.MAX_HORIZON_SEC * 1000

pytestmark = pytest.mark.filterwarnings("ignore")

MAY_ARCHIVE_START_MS = 1777593600000
MAY_ARCHIVE_END_MS = 1780272000000
MAY = 1777593600000
MP_DDL = ("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
          "mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)")


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old(symbol, entry_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return path_marks(conn, symbol, entry_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _max_ts(symbol):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol=?", (symbol,)).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=False).encode()).hexdigest()


def _sqlite(path, rows):
    c = sqlite3.connect(path)
    c.execute(MP_DDL)
    c.execute("CREATE INDEX ix ON mark_prices(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


def _catalog(root, entries):
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}, f)


def _old_synth(db, symbol, entry_ms):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return path_marks(conn, symbol, entry_ms)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Real production smoke (skip-guarded): ARCHIVE_ONLY / HYBRID / SQLITE_ONLY
# ---------------------------------------------------------------------------

def test_v2_parity_archive_only_real_window_and_plan_mode():
    _needs_real_db()
    entry = MAY_ARCHIVE_START_MS
    plan = RR.plan_read(_root(), table="mark_prices", symbol=SYMBOL, start_ms=entry, end_ms=entry + HORIZON_MS + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old_rows = _old(SYMBOL, entry)
    new_rows = path_marks_v2(_root(), SYMBOL, entry, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


def test_v2_parity_hybrid_real_boundary_and_plan_mode():
    _needs_real_db()
    entry = MAY_ARCHIVE_END_MS - HORIZON_MS // 2
    plan = RR.plan_read(_root(), table="mark_prices", symbol=SYMBOL, start_ms=entry, end_ms=entry + HORIZON_MS + 1)
    assert plan.mode == "HYBRID"
    old_rows = _old(SYMBOL, entry)
    new_rows = path_marks_v2(_root(), SYMBOL, entry, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


def test_v2_parity_sqlite_only_recent_live_window():
    _needs_real_db()
    max_ts = _max_ts(SYMBOL)
    assert max_ts is not None
    entry = int(max_ts) - HORIZON_MS
    plan = RR.plan_read(_root(), table="mark_prices", symbol=SYMBOL, start_ms=entry, end_ms=entry + HORIZON_MS + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_rows = _old(SYMBOL, entry)
    new_rows = path_marks_v2(_root(), SYMBOL, entry, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


def test_v2_parity_sqlite_only_real_no_archive_symbol_sol():
    _needs_real_db()
    max_ts = _max_ts("SOLUSDT")
    assert max_ts is not None
    entry = int(max_ts) - HORIZON_MS
    plan = RR.plan_read(_root(), table="mark_prices", symbol="SOLUSDT", start_ms=entry, end_ms=entry + HORIZON_MS + 1)
    assert plan.mode == "SQLITE_ONLY"
    assert plan.archive_segments == ()
    old_rows = _old("SOLUSDT", entry)
    new_rows = path_marks_v2(_root(), "SOLUSDT", entry, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


def test_source_mutation_zero_real():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    path_marks_v2(root, SYMBOL, MAY_ARCHIVE_START_MS, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


# ---------------------------------------------------------------------------
# Synthetic SQLITE_ONLY parity, boundaries, symbol filter, empty
# ---------------------------------------------------------------------------

def test_full_window_parity_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY - 1, "ETHUSDT", 2999.0, 0.0, None),
        (2, MAY, "ETHUSDT", 3000.0, 0.0, None),
        (3, MAY + 1000, "ETHUSDT", 3001.0, 0.0, None),
        (4, MAY + HORIZON_MS, "ETHUSDT", 3050.0, 0.0, None),
        (5, MAY + HORIZON_MS + 1, "ETHUSDT", 3060.0, 0.0, None),
        (6, MAY + 500, "SOLUSDT", 100.0, 0.0, None),
    ])
    old_rows = _old_synth(db, "ETHUSDT", MAY)
    new_rows = path_marks_v2(root, "ETHUSDT", MAY, source_db_path=db)
    assert new_rows == old_rows
    assert new_rows == [(MAY, 3000.0), (MAY + 1000, 3001.0), (MAY + HORIZON_MS, 3050.0)]


def test_symbol_filter_parity_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0, None),
        (2, MAY, "SOLUSDT", 100.0, 0.0, None),
        (3, MAY + 500, "ETHUSDT", 3001.0, 0.0, None),
        (4, MAY + 500, "SOLUSDT", 101.0, 0.0, None),
    ])
    eth = path_marks_v2(root, "ETHUSDT", MAY, source_db_path=db)
    sol = path_marks_v2(root, "SOLUSDT", MAY, source_db_path=db)
    assert eth == _old_synth(db, "ETHUSDT", MAY) == [(MAY, 3000.0), (MAY + 500, 3001.0)]
    assert sol == _old_synth(db, "SOLUSDT", MAY) == [(MAY, 100.0), (MAY + 500, 101.0)]


def test_boundaries_inclusive_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0, None),               # == lo INCLUDED
        (2, MAY + HORIZON_MS, "ETHUSDT", 3050.0, 0.0, None),  # == hi INCLUDED
    ])
    rows = path_marks_v2(root, "ETHUSDT", MAY, source_db_path=db)
    assert rows[0] == (MAY, 3000.0)
    assert rows[-1] == (MAY + HORIZON_MS, 3050.0)
    assert rows == _old_synth(db, "ETHUSDT", MAY)


def test_empty_range_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 0.0, None)])
    assert path_marks_v2(root, "ETHUSDT", MAY + HORIZON_MS + 10_000, source_db_path=db) == []


def test_determinism_and_batch_size_invariance_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    rows = [(i, MAY + i * 1000, "ETHUSDT", 3000.0 + i, 0.0, None) for i in range(1, 200)]
    _sqlite(db, rows)
    d1 = _digest(path_marks_v2(root, "ETHUSDT", MAY, source_db_path=db))
    d2 = _digest(path_marks_v2(root, "ETHUSDT", MAY, source_db_path=db))
    assert d1 == d2

    def _rows(batch_size):
        plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=MAY, end_ms=MAY + HORIZON_MS + 1)
        result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=db, batch_size=batch_size)
        return [(int(ts), float(px)) for ts, px in result.iter_rows()]

    assert _rows(1_000_000) == _rows(7) == _old_synth(db, "ETHUSDT", MAY)


def test_trust_failure_fails_closed_synthetic(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema

    root = str(tmp_path / "r")
    rel = os.path.join("table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                        "symbol=ETHUSDT", "year=2026", "month=05", "version=v1")
    final_dir = os.path.join(root, rel)
    os.makedirs(final_dir, exist_ok=True)
    schema = build_pyarrow_schema(SPEC)
    rows = [(1, MAY, "ETHUSDT", 3000.0, 0.0001, None)]
    arrays = [pa.array([r[i] for r in rows], type=schema.field(c).type)
              for i, c in enumerate(SPEC.preserved_columns)]
    pp = os.path.join(final_dir, "part-00000.parquet")
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), pp, compression="zstd")
    manifest = {"source_table": "mark_prices", "symbol": "ETHUSDT", "venue": SPEC.venue,
                "market_segment": SPEC.market_segment, "row_count": 1, "shards": None,
                "ordered_scientific_content_hash": "irrelevant", "partition_id": "corrupt-test",
                "parquet_path": os.path.join(rel, "part-00000.parquet"),
                "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
                "source_watermark_field": "id", "source_watermark_value": 1,
                "parquet_sha256": hashlib.sha256(open(pp, "rb").read()).hexdigest()}
    with open(os.path.join(final_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("ok\n")
    entry = {"archive_identity": "corrupt-test", "archive_relative_path": rel, "source_table": "mark_prices",
             "symbol": "ETHUSDT", "venue": SPEC.venue, "market_segment": SPEC.market_segment,
             "partition_start_ms": MAY, "partition_end_ms": MAY + 3600_000, "row_count": 1,
             "manifest_sha256": "0" * 64,
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": "irrelevant"}
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        path_marks_v2(root, "ETHUSDT", MAY)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def test_oracle_no_longer_called_from_live_path():
    for fn in (mod.simulate_current_be30, mod.simulate_trail_half_mfe):
        assert "path_marks(" not in inspect.getsource(fn)
        assert "path_marks_v2(" in inspect.getsource(fn)


def test_mark_at_asof_left_untouched():
    src = inspect.getsource(mod)
    assert "def mark_at" in src
    assert "lookup_latest_at_or_before" in src  # book_ticker ASOF already migrated, unchanged


def test_no_full_reverify_referenced():
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")
