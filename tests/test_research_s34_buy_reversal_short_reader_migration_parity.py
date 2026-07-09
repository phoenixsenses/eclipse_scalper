"""Range-read consumer migration V3 parity #2 (BATCH-STORAGE-ROTATION-
RETENTION-RANGE-READ-CONSUMER-MIGRATION-V3).

`tools/research_s34_buy_reversal_short.py`'s `sim_short` had an inline
mark_prices horizon-window range read; it is extracted into a named oracle
`_horizon_marks` and migrated to `_horizon_marks_v2` (reader-backed via
plan_read/execute_read). `_horizon_marks` is kept unchanged as the parity
reference. This file is structurally identical to
`research_s34_sell_reversal_quality.py` (SHORT direction / BUY-liq events
instead of LONG / SELL-liq), so this suite mirrors that one exactly.

Symbol is hardcoded 'ETHUSDT'. mark_prices has a real archive partition
(ETHUSDT/2026-05), so real ARCHIVE_ONLY, SQLITE_ONLY, and HYBRID-boundary
production smokes are all proven directly against the real DB.

DELIBERATELY OUT OF SCOPE (asserted below to remain on direct SQL):
  - `mark_at`: an ASOF-style point lookup -- belongs to the ASOF track.
  - `events`: a `liquidations` range read -- out-of-allowlist table.

Range-boundary note: the oracle uses an INCLUSIVE upper bound
(`ts_ms<=?`); the reader uses half-open `[start, end)`, so `_v2` passes
`end_ms+1`. Proven exact via a boundary test.
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
import tools.research_s34_buy_reversal_short as mod
from tools.research_s34_buy_reversal_short import _horizon_marks, _horizon_marks_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
MARK_SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.filterwarnings("ignore")

MARK_ARCHIVE_START_MS = 1777593600000
MARK_PARTITION_END_MS = 1780272000000


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old(start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _horizon_marks(conn, start_ms, end_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _digest(seq):
    h = hashlib.sha256()
    for item in seq:
        h.update(repr(item).encode())
    return h.hexdigest()


def _eth_max_ts():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- real production smoke: ARCHIVE_ONLY / SQLITE_ONLY / HYBRID ---

def test_parity_archive_only_real_window():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old(s, e)
    new = _horizon_marks_v2(_root(), s, e, source_db_path=REAL_SOURCE_DB)
    assert len(new) == len(old) and len(old) > 0
    assert new == old
    assert _digest(new) == _digest(old)


def test_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _eth_max_ts()
    s, e = max_ts - 900_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old(s, e)
    new = _horizon_marks_v2(_root(), s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_parity_hybrid_real_boundary():
    _needs_real_db()
    s, e = MARK_PARTITION_END_MS - 120_000, MARK_PARTITION_END_MS + 60_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "HYBRID"
    old = _old(s, e)
    new = _horizon_marks_v2(_root(), s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_provenance_reports_archive_source_and_columns():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["table"] == "mark_prices"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["ts_ms", "mark_price"]
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert len(prov["archive_segments"]) == 1
    assert prov["archive_segments"][0]["manifest_sha256"]
    assert prov["row_count"] > 0


def test_inclusive_end_boundary_parity_real():
    _needs_real_db()
    s = MARK_ARCHIVE_START_MS
    rows = _old(s, s + 300_000)
    assert rows
    last_ts = rows[-1][0]
    at = _horizon_marks_v2(_root(), s, int(last_ts), source_db_path=REAL_SOURCE_DB)
    before = _horizon_marks_v2(_root(), s, int(last_ts) - 1, source_db_path=REAL_SOURCE_DB)
    assert at[-1][0] == last_ts
    assert len(at) == len(before) + 1
    assert at == _old(s, int(last_ts))
    assert before == _old(s, int(last_ts) - 1)


def test_empty_range_parity():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000
    old = _old(lo, hi)
    new = _horizon_marks_v2(_root(), lo, hi, source_db_path=REAL_SOURCE_DB)
    assert old == [] and new == []


def test_repeated_run_determinism():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 60_000
    vals = [_horizon_marks_v2(_root(), s, e, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
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
    _horizon_marks_v2(root, MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_mark_at_asof_lookup_left_on_direct_sql():
    import inspect
    src = inspect.getsource(mod.mark_at)
    assert "mark_prices" in src and "limit 1" in src
    assert "RR.execute_read(" not in src


def test_events_liquidations_left_on_direct_sql_out_of_allowlist():
    import inspect
    src = inspect.getsource(mod.events)
    assert "liquidations" in src
    assert "RR.execute_read(" not in src


def test_sim_short_uses_v2_not_inline_query():
    import inspect
    src = inspect.getsource(mod.sim_short)
    assert "_horizon_marks_v2(root," in src
    assert "select ts_ms, mark_price" not in src.lower()


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


MP_DDL = ("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
          "mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)")


def _sqlite(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute(MP_DDL)
    c.execute("CREATE INDEX ix ON mark_prices(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


MAY = 1777593600000


def test_symbol_mismatch_safety_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0, None),
        (2, MAY + 1000, "ETHUSDT", 3001.0, 0.0, None),
        (3, MAY + 500, "SOLUSDT", 99999.0, 0.0, None),
    ])
    rows = _horizon_marks_v2(root, MAY, MAY + 1000, source_db_path=db)
    assert rows == [(MAY, 3000.0), (MAY + 1000, 3001.0)]


def test_tie_break_ordering_parity_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0, None),
        (2, MAY, "ETHUSDT", 3000.5, 0.0, None),
    ])
    rows = _horizon_marks_v2(root, MAY, MAY, source_db_path=db)
    assert rows == [(MAY, 3000.0), (MAY, 3000.5)]


def test_trust_failure_fails_closed_synthetic(tmp_path):
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
        _horizon_marks_v2(root, MAY, MAY + 1_000)
