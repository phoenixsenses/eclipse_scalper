"""Range-read consumer migration (BATCH-STORAGE-ROTATION-RETENTION-
RANGE-READ-CONSUMER-MIGRATION-V8), target 2 of 3.

`tools/research_s34_session_analysis.py`'s inline range read inside
`simulate()` (`mark_prices`, `ts_ms>? AND ts_ms<=? ORDER BY ts_ms` --
same exclusive-lower/inclusive-upper shape as
`research_s34_hold_sweep.py`, but projecting ONLY `mark_price`, no
`ts_ms`) was extracted into `_price_path()` (the parity oracle,
unchanged) and migrated to `_price_path_v2()` (reader-backed via
`plan_read`/`execute_read`). `_price_path()` is no longer called from
`simulate()`'s live path.

Call-site audit: `symbol` is a genuine runtime parameter (ETHUSDT/
SOLUSDT, sourced from each `main()` call site) -- real ARCHIVE_ONLY/
HYBRID/SQLITE_ONLY production smoke for ETHUSDT, real SQLITE_ONLY
(no-archive) smoke for SOLUSDT.

Range-boundary note: identical to `research_s34_hold_sweep.py` --
`start_ms=gt_ms+1, end_ms=le_ms+1` reproduces the oracle's
`ts_ms>gt_ms AND ts_ms<=le_ms` bit-for-bit.

Column-projection note: the oracle's `SELECT mark_price` (single
column, no `ts_ms`) is reproduced via `columns=("mark_price",)`,
so `_price_path_v2` returns single-element `(mark_price,)` tuples
matching the oracle's `fetchall()` shape exactly -- the call site's
`for (mp,) in rows:` unpacking is unchanged.

Out-of-scope, untouched (this gate is range-read only):
  * `mark_at()` -- backward ASOF, left on direct SQL.
  * `build_cascades()` -- `liquidations` (out-of-allowlist), untouched.
  * `INTEL_DB` (`data/s34_intelligence.db`) -- defined but never
    queried anywhere in this file; confirmed unused, left untouched.
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
import tools.research_s34_session_analysis as mod
from tools.research_s34_session_analysis import _price_path, _price_path_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.filterwarnings("ignore")

MAY_ARCHIVE_START_MS = 1777593600000
MAY_ARCHIVE_END_MS = 1780272000000


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old(symbol, gt_ms, le_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _price_path(conn, symbol, gt_ms, le_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_row_count(symbol, gt_ms, le_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM mark_prices WHERE symbol=? AND ts_ms>? AND ts_ms<=?",
            (symbol, int(gt_ms), int(le_ms)),
        ).fetchone()[0]
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


# ---------------------------------------------------------------------------
# Real production smoke: ARCHIVE_ONLY / SQLITE_ONLY (both symbols) / HYBRID
# ---------------------------------------------------------------------------

def test_v2_parity_archive_only_real_window_and_plan_mode():
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=gt_ms + 1, end_ms=le_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old_rows = _old("ETHUSDT", gt_ms, le_ms)
    new_rows = _price_path_v2(_root(), "ETHUSDT", gt_ms, le_ms, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0
    assert all(len(r) == 1 for r in new_rows)  # single-column (mark_price,) shape preserved


def test_v2_parity_sqlite_only_recent_live_window_eth():
    _needs_real_db()
    max_ts = _max_ts("ETHUSDT")
    gt_ms, le_ms = max_ts - 300_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=gt_ms + 1, end_ms=le_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_rows = _old("ETHUSDT", gt_ms, le_ms)
    new_rows = _price_path_v2(_root(), "ETHUSDT", gt_ms, le_ms, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


def test_v2_parity_sqlite_only_real_no_archive_symbol_sol():
    _needs_real_db()
    max_ts = _max_ts("SOLUSDT")
    gt_ms, le_ms = max_ts - 300_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="SOLUSDT", start_ms=gt_ms + 1, end_ms=le_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    assert plan.archive_segments == ()
    old_rows = _old("SOLUSDT", gt_ms, le_ms)
    new_rows = _price_path_v2(_root(), "SOLUSDT", gt_ms, le_ms, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


HYBRID_GT_MS = MAY_ARCHIVE_END_MS - 150_000
HYBRID_LE_MS = MAY_ARCHIVE_END_MS + 150_000


def test_v2_parity_hybrid_real_boundary_and_plan_mode():
    _needs_real_db()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=HYBRID_GT_MS + 1, end_ms=HYBRID_LE_MS + 1)
    assert plan.mode == "HYBRID"
    old_rows = _old("ETHUSDT", HYBRID_GT_MS, HYBRID_LE_MS)
    new_rows = _price_path_v2(_root(), "ETHUSDT", HYBRID_GT_MS, HYBRID_LE_MS, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


def test_provenance_reports_hybrid_source_and_columns():
    _needs_real_db()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=HYBRID_GT_MS + 1, end_ms=HYBRID_LE_MS + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "HYBRID"
    assert prov["columns"] == ["mark_price"]
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert len(prov["archive_segments"]) == 1
    assert len(prov["sqlite_ranges"]) == 1
    assert prov["row_count"] > 0


# ---------------------------------------------------------------------------
# Row count / ordering-behind-the-scenes / digest / batch-size parity
# ---------------------------------------------------------------------------

def test_row_count_parity_real_archive_window():
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    old_n = _old_row_count("ETHUSDT", gt_ms, le_ms)
    new_rows = _price_path_v2(_root(), "ETHUSDT", gt_ms, le_ms, source_db_path=REAL_SOURCE_DB)
    assert len(new_rows) == old_n


def test_digest_parity_real_data():
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    old_rows = _old("ETHUSDT", gt_ms, le_ms)
    new_rows = _price_path_v2(_root(), "ETHUSDT", gt_ms, le_ms, source_db_path=REAL_SOURCE_DB)
    assert _digest(old_rows) == _digest(new_rows)


def test_repeated_run_determinism():
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    d1 = _digest(_price_path_v2(_root(), "ETHUSDT", gt_ms, le_ms, source_db_path=REAL_SOURCE_DB))
    d2 = _digest(_price_path_v2(_root(), "ETHUSDT", gt_ms, le_ms, source_db_path=REAL_SOURCE_DB))
    assert d1 == d2


def test_output_invariant_to_batch_size_real_data():
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=gt_ms + 1, end_ms=le_ms + 1)

    def _rows(batch_size):
        result = RR.execute_read(plan, columns=("mark_price",), source_db_path=REAL_SOURCE_DB, batch_size=batch_size)
        return list(result.iter_rows())

    assert _rows(1_000_000) == _rows(11)


# ---------------------------------------------------------------------------
# Exclusive-lower / inclusive-upper boundary parity
# ---------------------------------------------------------------------------

def test_exclusive_lower_boundary_parity_real_data():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        first_ts = conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
            (MAY_ARCHIVE_START_MS,),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    le_ms = int(first_ts) + 300_000
    old_rows = _old("ETHUSDT", int(first_ts), le_ms)
    new_rows = _price_path_v2(_root(), "ETHUSDT", int(first_ts), le_ms, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    old_n_incl = _old_row_count("ETHUSDT", int(first_ts) - 1, le_ms)  # includes the boundary row
    assert old_n_incl == len(old_rows) + 1  # confirms first_ts row genuinely excluded


def test_inclusive_upper_boundary_parity_real_data():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts = conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old_n_at = _old_row_count("ETHUSDT", MAY_ARCHIVE_START_MS, int(last_ts))
    old_n_before = _old_row_count("ETHUSDT", MAY_ARCHIVE_START_MS, int(last_ts) - 1)
    assert old_n_at == old_n_before + 1
    rows_at = _price_path_v2(_root(), "ETHUSDT", MAY_ARCHIVE_START_MS, int(last_ts), source_db_path=REAL_SOURCE_DB)
    rows_before = _price_path_v2(_root(), "ETHUSDT", MAY_ARCHIVE_START_MS, int(last_ts) - 1, source_db_path=REAL_SOURCE_DB)
    assert len(rows_at) == old_n_at
    assert len(rows_before) == old_n_before


# ---------------------------------------------------------------------------
# Empty / bounded window
# ---------------------------------------------------------------------------

def test_empty_range_returns_empty_list():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_300_000
    old_rows = _old("ETHUSDT", lo, hi)
    new_rows = _price_path_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB)
    assert old_rows == []
    assert new_rows == []


def test_bounded_window_row_count_sane_real_data():
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 60_000
    rows = _price_path_v2(_root(), "ETHUSDT", gt_ms, le_ms, source_db_path=REAL_SOURCE_DB)
    assert 0 < len(rows) < 100_000
    for (px,) in rows:
        assert px > 0.0


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def test_no_full_reverify_referenced():
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    _price_path_v2(root, "ETHUSDT", MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_price_path_v1_no_longer_called_from_live_path():
    for fn in (mod.simulate, mod.analyze, mod.main):
        assert "_price_path(" not in inspect.getsource(fn)
    assert "_price_path_v2(" in inspect.getsource(mod.simulate)


def test_mark_at_asof_and_liquidations_left_untouched():
    src = inspect.getsource(mod)
    assert "ORDER BY ts_ms DESC LIMIT 1" in src
    assert "FROM liquidations" in src
    assert "lookup_latest_at_or_before" not in src


def test_intel_db_unused_and_untouched():
    """`INTEL_DB` is defined but never queried anywhere in this file --
    confirmed by static search (the identifier occurs exactly once: its
    own definition), not migrated because there is nothing to migrate."""
    assert mod.INTEL_DB.name == "s34_intelligence.db"
    src = inspect.getsource(mod)
    assert src.count("INTEL_DB") == 1


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

MP_DDL = ("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
          "mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)")


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


MAY = 1777593600000


def test_exclusive_lower_boundary_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0001, None),
        (2, MAY + 1000, "ETHUSDT", 3001.0, 0.0002, None),
    ])
    rows = _price_path_v2(root, "ETHUSDT", MAY, MAY + 1000, source_db_path=db)
    assert rows == [(3001.0,)]


def test_empty_range_synthetic_returns_empty(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 0.0001, None)])
    assert _price_path_v2(root, "ETHUSDT", MAY + 10_000, MAY + 20_000, source_db_path=db) == []


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
    mp = os.path.join(final_dir, "manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
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
        _price_path_v2(root, "ETHUSDT", MAY, MAY + 1000)
