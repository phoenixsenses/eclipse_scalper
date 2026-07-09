"""Range-read consumer migration pilot #1 (BATCH-STORAGE-ROTATION-
RETENTION-RANGE-READ-CONSUMER-MIGRATION-V1).

`tools/funding_rate_analysis.py`'s `_avg_funding_rate` (a bounded range
read + `AVG(funding_rate)` aggregate over `mark_prices`) is migrated to
`_avg_funding_rate_v2` (reader-backed via `plan_read`/`execute_read`).
`_avg_funding_rate` is kept unchanged as the parity oracle.

Call-site audit: `symbol` is a GENUINE runtime parameter, sourced
per-trade from `trades.symbol` in `data/paper_trades.db` (falling back
to `--symbol` only when that column is absent). The real `paper_trades.db`
in this checkout has a `symbol` column but zero rows satisfying
`entry_time>0 AND exit_time>entry_time`, so no real call-site invocation
happens today; nonetheless this suite proves a REAL production smoke
directly against the real archived `mark_prices/ETHUSDT/2026-05`
partition and the real source DB (not merely synthetic), per the
mark_prices special rule, plus real SQLITE_ONLY and HYBRID-boundary
smokes.

Range-boundary note: the oracle's SQL uses an INCLUSIVE upper bound
(`ts_ms<=end_ms`); `plan_read`/`execute_read` use the reader's half-open
`[start_ms, end_ms)` convention, so `_avg_funding_rate_v2` passes
`end_ms+1`. A dedicated test proves a row exactly AT `end_ms` is included
and a row at `end_ms+1` is excluded, matching the oracle bit-for-bit.

This file has no ASOF (`ORDER BY ts_ms DESC LIMIT 1`) pattern at all
(confirmed by inventory scan: `desc_limit1_count: 0`) -- nothing is
out-of-scope here.
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
import tools.funding_rate_analysis as mod
from tools.funding_rate_analysis import _avg_funding_rate, _avg_funding_rate_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.filterwarnings("ignore")

APR_ARCHIVE_START_MS = 1777593600000  # real mark_prices/ETHUSDT/2026-05 partition start
APR_ARCHIVE_END_MS = 1780272000000    # real mark_prices/ETHUSDT/2026-05 partition end


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old(symbol, start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _avg_funding_rate(conn, symbol, start_ms, end_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_row_count(symbol, start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
            (symbol, int(start_ms), int(end_ms)),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _eth_max_ts():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- real production smoke: ARCHIVE_ONLY inside mark_prices/ETHUSDT/2026-05 ---

def test_v2_parity_archive_only_real_window_and_plan_mode():
    _needs_real_db()
    start_ms, end_ms = APR_ARCHIVE_START_MS, APR_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old_avg = _old(  "ETHUSDT", start_ms, end_ms)
    new_avg = _avg_funding_rate_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_avg == pytest.approx(old_avg, rel=1e-9)
    old_n = _old_row_count("ETHUSDT", start_ms, end_ms)
    assert old_n > 0  # sanity: the window actually has real rows


# --- real production smoke: recent live SQLITE_ONLY window ---

def test_v2_parity_sqlite_only_recent_live_window():
    _needs_real_db()
    max_ts = _eth_max_ts()
    assert max_ts is not None
    start_ms, end_ms = max_ts - 300_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_avg = _old("ETHUSDT", start_ms, end_ms)
    new_avg = _avg_funding_rate_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_avg == pytest.approx(old_avg, rel=1e-9)


# --- real production smoke: HYBRID window straddling the archive/live boundary ---

HYBRID_START_MS = APR_ARCHIVE_END_MS - 150_000
HYBRID_END_MS = APR_ARCHIVE_END_MS + 150_000


def test_v2_parity_hybrid_real_boundary_and_plan_mode():
    _needs_real_db()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=HYBRID_START_MS, end_ms=HYBRID_END_MS + 1)
    assert plan.mode == "HYBRID"
    old_avg = _old("ETHUSDT", HYBRID_START_MS, HYBRID_END_MS)
    new_avg = _avg_funding_rate_v2(_root(), "ETHUSDT", HYBRID_START_MS, HYBRID_END_MS, source_db_path=REAL_SOURCE_DB)
    assert new_avg == pytest.approx(old_avg, rel=1e-9)


def test_provenance_reports_hybrid_source_and_columns():
    _needs_real_db()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=HYBRID_START_MS, end_ms=HYBRID_END_MS + 1)
    result = RR.execute_read(plan, columns=("funding_rate",), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "HYBRID"
    assert prov["table"] == "mark_prices"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["funding_rate"]
    assert prov["requested_start_ms"] == HYBRID_START_MS
    assert prov["requested_end_ms"] == HYBRID_END_MS + 1
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert len(prov["archive_segments"]) == 1
    assert prov["archive_segments"][0]["archive_identity"]
    assert prov["archive_segments"][0]["manifest_sha256"]
    assert len(prov["sqlite_ranges"]) == 1
    assert prov["row_count"] > 0


# --- row count parity (direct oracle COUNT(*) vs reader iter_rows count) ---

def test_row_count_parity_real_archive_window():
    _needs_real_db()
    start_ms, end_ms = APR_ARCHIVE_START_MS, APR_ARCHIVE_START_MS + 300_000
    old_n = _old_row_count("ETHUSDT", start_ms, end_ms)
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    result = RR.execute_read(plan, columns=("funding_rate",), source_db_path=REAL_SOURCE_DB)
    new_n = sum(1 for _ in result.iter_rows())
    assert new_n == old_n


# --- inclusive-boundary parity: a row exactly AT end_ms must be included,
# a row at end_ms+1 must be excluded (matches oracle's ts_ms<=end_ms) ---

def test_inclusive_end_boundary_parity_real_data():
    _needs_real_db()
    # Use the window's own last real row's ts_ms as the exact end_ms
    # boundary, so we know a row genuinely sits ON the boundary.
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts = conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (APR_ARCHIVE_START_MS, APR_ARCHIVE_START_MS + 300_000),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old_n_at_boundary = _old_row_count("ETHUSDT", APR_ARCHIVE_START_MS, int(last_ts))
    old_n_before_boundary = _old_row_count("ETHUSDT", APR_ARCHIVE_START_MS, int(last_ts) - 1)
    assert old_n_at_boundary == old_n_before_boundary + 1  # the boundary row itself is included

    plan_at = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=APR_ARCHIVE_START_MS, end_ms=int(last_ts) + 1)
    n_at = sum(1 for _ in RR.execute_read(plan_at, columns=("funding_rate",), source_db_path=REAL_SOURCE_DB).iter_rows())
    plan_before = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=APR_ARCHIVE_START_MS, end_ms=int(last_ts))
    n_before = sum(1 for _ in RR.execute_read(plan_before, columns=("funding_rate",), source_db_path=REAL_SOURCE_DB).iter_rows())
    assert n_at == old_n_at_boundary
    assert n_before == old_n_before_boundary


# --- empty range ---

def test_empty_range_returns_zero():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_300_000  # 2020-09, long before any collection
    old_avg = _old("ETHUSDT", lo, hi)
    new_avg = _avg_funding_rate_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB)
    assert old_avg == 0.0
    assert new_avg == 0.0


# --- deterministic repeated run ---

def test_repeated_run_determinism():
    _needs_real_db()
    start_ms, end_ms = APR_ARCHIVE_START_MS, APR_ARCHIVE_START_MS + 300_000
    vals = [_avg_funding_rate_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


# --- batch-size invariance (same AVG regardless of streaming batch size) ---

def test_avg_invariant_to_batch_size():
    _needs_real_db()
    start_ms, end_ms = APR_ARCHIVE_START_MS, APR_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)

    def _reduce(batch_size):
        result = RR.execute_read(plan, columns=("funding_rate",), source_db_path=REAL_SOURCE_DB, batch_size=batch_size)
        total = 0.0
        count = 0
        for (fr,) in result.iter_rows():
            if fr is not None:
                total += float(fr)
                count += 1
        return (total / count) if count else 0.0

    big_batch = _reduce(1_000_000)
    tiny_batch = _reduce(7)
    assert big_batch == pytest.approx(tiny_batch, rel=1e-9)


def test_no_full_reverify_referenced():
    import inspect
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    _avg_funding_rate_v2(root, "ETHUSDT", APR_ARCHIVE_START_MS, APR_ARCHIVE_START_MS + 300_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


# ---------------------------------------------------------------------------
# Synthetic fixtures: NULL-skip semantics, symbol-mismatch safety, and
# trust-failure fail-closed.
# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


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


def test_null_funding_rate_skipped_like_sql_avg_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    # Two real values (0.0001, 0.0003) and one NULL -- SQL AVG skips the
    # NULL, giving mean([0.0001, 0.0003]) == 0.0002, NOT mean of all 3 rows.
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0001, None),
        (2, MAY + 1000, "ETHUSDT", 3001.0, None, None),
        (3, MAY + 2000, "ETHUSDT", 3002.0, 0.0003, None),
    ])
    conn = sqlite3.connect(db)
    sql_avg = conn.execute(
        "SELECT AVG(funding_rate) FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?",
        (MAY, MAY + 2000),
    ).fetchone()[0]
    conn.close()
    assert sql_avg == pytest.approx(0.0002)
    new_avg = _avg_funding_rate_v2(root, "ETHUSDT", MAY, MAY + 2000, source_db_path=db)
    assert new_avg == pytest.approx(sql_avg, rel=1e-9)


def test_symbol_filter_parity_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0001, None),
        (2, MAY, "SOLUSDT", 100.0, 0.0009, None),
    ])
    eth_avg = _avg_funding_rate_v2(root, "ETHUSDT", MAY, MAY, source_db_path=db)
    sol_avg = _avg_funding_rate_v2(root, "SOLUSDT", MAY, MAY, source_db_path=db)
    assert eth_avg == pytest.approx(0.0001)
    assert sol_avg == pytest.approx(0.0009)


def test_empty_range_synthetic_returns_zero(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 0.0001, None)])
    assert _avg_funding_rate_v2(root, "ETHUSDT", MAY + 10_000, MAY + 20_000, source_db_path=db) == 0.0


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
                "parquet_sha256": _sha256_file(pp)}
    mp = os.path.join(final_dir, "manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("ok\n")
    entry = {"archive_identity": "corrupt-test", "archive_relative_path": rel, "source_table": "mark_prices",
             "symbol": "ETHUSDT", "venue": SPEC.venue, "market_segment": SPEC.market_segment,
             "partition_start_ms": MAY, "partition_end_ms": MAY + 3600_000, "row_count": 1,
             "manifest_sha256": "0" * 64,  # deliberately WRONG -- must fail the trust check
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": "irrelevant"}
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        _avg_funding_rate_v2(root, "ETHUSDT", MAY, MAY + 1000)


def test_mconn_dead_connection_dropped():
    import inspect
    src = inspect.getsource(mod.main)
    assert "sqlite3.connect(str(mdb)" not in src
    assert "resolve_production_root" in src
