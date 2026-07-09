"""Range-read consumer migration V5 parity #3 (BATCH-STORAGE-ROTATION-
RETENTION-RANGE-READ-CONSUMER-MIGRATION-V5).

`tools/research_s34_sell_liq_bounce.py`'s `_mfe_bps` had an inline
mark_prices range read; it is extracted into a named oracle
`_mfe_marks_range` and migrated to `_mfe_marks_range_v2` (reader-backed
via plan_read/execute_read). `_mfe_marks_range` is kept unchanged as the
parity reference.

`symbol` is a genuine runtime parameter (`trade["symbol"]`), though this
file's real call site always resolves to "ETHUSDT" (the hardcoded
`S34Rule(symbol="ETHUSDT", ...)` in `_run_config`). mark_prices has a real
archive partition for ETHUSDT/2026-05, so real ARCHIVE_ONLY/SQLITE_ONLY/
HYBRID smokes are proven directly against the real DB.

DELIBERATELY OUT OF SCOPE (asserted below to remain on direct SQL):
  - the two `SELECT MIN(ts_ms), MAX(ts_ms) ... WHERE symbol='ETHUSDT'`
    queries (one on `liquidations` in `_run_config`, one on `mark_prices`
    in `main`) -- both are UNBOUNDED full-symbol-history scans (no
    `ts_ms>=?/<=?` window), a genuinely different query shape from the
    bounded range reads `plan_read`/`execute_read` targets; the
    liquidations one is additionally out-of-allowlist.
  - `runner._bucket_events`/`_paper_trade_from_signal`/`_evaluate_trade`
    (imported from `s34_shadow_paper_runner`): external module calls that
    own their own direct-SQL internals in that module -- untouched here.

Range-boundary note: the oracle uses an INCLUSIVE upper bound
(`ts_ms<=?`); the reader uses half-open `[start, end)`, so `_v2` passes
`end_ms+1`. Proven exact via a boundary test. Ordering does not affect
the oracle's result (only max() over the value set matters).
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
import tools.research_s34_sell_liq_bounce as mod
from tools.research_s34_sell_liq_bounce import _mfe_marks_range, _mfe_marks_range_v2, _mfe_bps

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


def _old(symbol, start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _mfe_marks_range(conn, symbol, start_ms, end_ms)
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
    old = _old("ETHUSDT", s, e)
    new = _mfe_marks_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert len(new) == len(old) and len(old) > 0
    assert new == old
    assert _digest(new) == _digest(old)


def test_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _eth_max_ts()
    s, e = max_ts - 900_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old("ETHUSDT", s, e)
    new = _mfe_marks_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_parity_hybrid_real_boundary():
    _needs_real_db()
    s, e = MARK_PARTITION_END_MS - 120_000, MARK_PARTITION_END_MS + 60_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    assert plan.mode == "HYBRID"
    old = _old("ETHUSDT", s, e)
    new = _mfe_marks_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_full_mfe_bps_parity_real():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    rows = _old("ETHUSDT", s, e)
    entry_ref = float(rows[0][0])
    prices = [float(r[0]) for r in rows]
    old_mfe = max(p - entry_ref for p in prices) / entry_ref * 10_000
    trade = {"symbol": "ETHUSDT", "entry_ts_ms": s, "exit_ts_ms": e, "entry_reference_price": entry_ref}
    new_mfe = _mfe_bps(None, trade, root=_root(), source_db_path=REAL_SOURCE_DB)
    assert new_mfe == pytest.approx(old_mfe, rel=1e-12)


def test_provenance_reports_archive_source_and_columns():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS + 3_600_000, MARK_ARCHIVE_START_MS + 3_600_000 + 900_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=s, end_ms=e + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["table"] == "mark_prices"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["mark_price"]
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert len(prov["archive_segments"]) == 1
    assert prov["archive_segments"][0]["manifest_sha256"]
    assert prov["row_count"] > 0


def test_inclusive_end_boundary_parity_real():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts = conn.execute(
            "SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?",
            (MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000)).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    at = _mfe_marks_range_v2(_root(), "ETHUSDT", MARK_ARCHIVE_START_MS, int(last_ts), source_db_path=REAL_SOURCE_DB)
    before = _mfe_marks_range_v2(_root(), "ETHUSDT", MARK_ARCHIVE_START_MS, int(last_ts) - 1, source_db_path=REAL_SOURCE_DB)
    assert len(at) == len(before) + 1
    assert at == _old("ETHUSDT", MARK_ARCHIVE_START_MS, int(last_ts))


def test_empty_range_parity():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000
    old = _old("ETHUSDT", lo, hi)
    new = _mfe_marks_range_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB)
    assert old == [] and new == []


def test_repeated_run_determinism():
    _needs_real_db()
    s, e = MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 60_000
    vals = [_mfe_marks_range_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
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
    _mfe_marks_range_v2(root, "ETHUSDT", MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_unbounded_minmax_scans_left_on_direct_sql_out_of_scope():
    import inspect
    src_run_config = inspect.getsource(mod._run_config)
    src_main = inspect.getsource(mod.main)
    assert "MIN(ts_ms), MAX(ts_ms)" in src_run_config and "liquidations" in src_run_config
    assert "MIN(ts_ms), MAX(ts_ms)" in src_main and "mark_prices" in src_main
    assert "RR.execute_read(" not in src_run_config
    assert "RR.execute_read(" not in src_main


def test_mfe_bps_uses_v2_not_inline_query():
    import inspect
    src = inspect.getsource(mod._mfe_bps)
    assert "_mfe_marks_range_v2(root," in src
    assert "select mark_price" not in src.lower()


def test_external_shadow_paper_runner_calls_untouched():
    import inspect
    src = inspect.getsource(mod._run_config)
    assert "runner._bucket_events(" in src
    assert "runner._paper_trade_from_signal(" in src
    assert "runner._evaluate_trade(" in src


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
    eth = _mfe_marks_range_v2(root, "ETHUSDT", MAY, MAY + 1000, source_db_path=db)
    sol = _mfe_marks_range_v2(root, "SOLUSDT", MAY, MAY + 1000, source_db_path=db)
    assert eth == [(3000.0,), (3001.0,)]
    assert sol == [(99999.0,)]


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
        _mfe_marks_range_v2(root, "ETHUSDT", MAY, MAY + 1_000)
