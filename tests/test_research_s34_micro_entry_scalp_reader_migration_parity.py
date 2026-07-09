"""Range-read consumer migration pilot (BATCH-STORAGE-ROTATION-RETENTION-
RANGE-READ-CONSUMER-MIGRATION-V7).

`tools/research_s34_micro_entry_scalp.py`'s `_mark_series` (a bounded
range read over `mark_prices`, `ts_ms>=? AND ts_ms<=? ORDER BY ts_ms
ASC`) is migrated to `_mark_series_v2` (reader-backed via
`plan_read`/`execute_read`). `_mark_series` is kept unchanged as the
parity oracle; it is no longer called from the live path (`_choose_entry`,
`_simulate_variant`, `_path_stats`, `main`), which now call
`_mark_series_v2` exclusively.

Call-site audit: `symbol` is a genuine runtime parameter -- one of
SOLUSDT/ETHUSDT/BTCUSDT depending on which `BASE_BUCKETS` entry is being
simulated. Only ETHUSDT has a real `mark_prices` archive partition
(`ETHUSDT/2026-05`); SOLUSDT and BTCUSDT calls are real, production
SQLITE_ONLY reads (not synthetic) -- there is no archive partition for
either symbol in `mark_prices` to straddle, so their coverage here is
genuinely real data, just never archive-backed.

Range-boundary note: the oracle's SQL uses an INCLUSIVE upper bound
(`ts_ms<=end_ms`); `plan_read`/`execute_read` use the reader's half-open
`[start_ms, end_ms)` convention, so `_mark_series_v2` passes `end_ms+1`.
A dedicated test proves a row exactly AT `end_ms` is included and a row
at `end_ms+1` is excluded, matching the oracle bit-for-bit.

Out-of-scope, untouched (this gate is range-read only): `_bucket_events`,
`_fill_quote`, `_close_trade`, `_mark_at`, `_paper_trade_from_signal`
(all imported unchanged from `tools.s34_shadow_paper_runner`, a
DO_NOT_TOUCH-by-name module) and the `liquidations` table read inside
`main()`/`_bucket_events` (out-of-allowlist table, never migrated in any
range-read gate). This file has no ASOF (`ORDER BY ts_ms DESC LIMIT 1`)
pattern of its own on an allowlisted table.
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
import tools.research_s34_micro_entry_scalp as mod
from tools.research_s34_micro_entry_scalp import _mark_series, _mark_series_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.filterwarnings("ignore")

MAY_ARCHIVE_START_MS = 1777593600000  # real mark_prices/ETHUSDT/2026-05 partition start
MAY_ARCHIVE_END_MS = 1780272000000    # real mark_prices/ETHUSDT/2026-05 partition end


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old(symbol, start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _mark_series(conn, symbol, start_ms, end_ms)
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
# Real production smoke: ARCHIVE_ONLY inside mark_prices/ETHUSDT/2026-05
# ---------------------------------------------------------------------------

def test_v2_parity_archive_only_real_window_and_plan_mode():
    _needs_real_db()
    start_ms, end_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old_rows = _old("ETHUSDT", start_ms, end_ms)
    new_rows = _mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0  # sanity: the window actually has real rows


# ---------------------------------------------------------------------------
# Real production smoke: recent live SQLITE_ONLY window (ETHUSDT)
# ---------------------------------------------------------------------------

def test_v2_parity_sqlite_only_recent_live_window_eth():
    _needs_real_db()
    max_ts = _max_ts("ETHUSDT")
    assert max_ts is not None
    start_ms, end_ms = max_ts - 300_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_rows = _old("ETHUSDT", start_ms, end_ms)
    new_rows = _mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


# ---------------------------------------------------------------------------
# Real production smoke: symbol-mismatch-safe SQLITE_ONLY for SOLUSDT and
# BTCUSDT -- neither has a mark_prices archive partition at all, so any
# real window for these symbols is genuinely SQLITE_ONLY (not synthetic).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol", ["SOLUSDT", "BTCUSDT"])
def test_v2_parity_sqlite_only_real_no_archive_symbol(symbol):
    _needs_real_db()
    max_ts = _max_ts(symbol)
    assert max_ts is not None
    start_ms, end_ms = max_ts - 300_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol=symbol, start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    assert plan.archive_segments == ()
    old_rows = _old(symbol, start_ms, end_ms)
    new_rows = _mark_series_v2(_root(), symbol, start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


# ---------------------------------------------------------------------------
# Real production smoke: HYBRID window straddling the archive/live boundary
# ---------------------------------------------------------------------------

HYBRID_START_MS = MAY_ARCHIVE_END_MS - 150_000
HYBRID_END_MS = MAY_ARCHIVE_END_MS + 150_000


def test_v2_parity_hybrid_real_boundary_and_plan_mode():
    _needs_real_db()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=HYBRID_START_MS, end_ms=HYBRID_END_MS + 1)
    assert plan.mode == "HYBRID"
    old_rows = _old("ETHUSDT", HYBRID_START_MS, HYBRID_END_MS)
    new_rows = _mark_series_v2(_root(), "ETHUSDT", HYBRID_START_MS, HYBRID_END_MS, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


def test_provenance_reports_hybrid_source_and_columns():
    _needs_real_db()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=HYBRID_START_MS, end_ms=HYBRID_END_MS + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "HYBRID"
    assert prov["table"] == "mark_prices"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["ts_ms", "mark_price"]
    assert prov["requested_start_ms"] == HYBRID_START_MS
    assert prov["requested_end_ms"] == HYBRID_END_MS + 1
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert len(prov["archive_segments"]) == 1
    assert prov["archive_segments"][0]["archive_identity"]
    assert prov["archive_segments"][0]["manifest_sha256"]
    assert len(prov["sqlite_ranges"]) == 1
    assert prov["row_count"] > 0


# ---------------------------------------------------------------------------
# Row count / column / ordering / first-last-timestamp / digest parity
# ---------------------------------------------------------------------------

def test_row_count_parity_real_archive_window():
    _needs_real_db()
    start_ms, end_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    old_n = _old_row_count("ETHUSDT", start_ms, end_ms)
    new_rows = _mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert len(new_rows) == old_n


def test_ordering_parity_strictly_ascending_real_data():
    _needs_real_db()
    start_ms, end_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    rows = _mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    ts_seq = [r[0] for r in rows]
    assert ts_seq == sorted(ts_seq)


def test_first_last_timestamp_parity_real_data():
    _needs_real_db()
    start_ms, end_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    old_rows = _old("ETHUSDT", start_ms, end_ms)
    new_rows = _mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert old_rows[0] == new_rows[0]
    assert old_rows[-1] == new_rows[-1]


def test_deterministic_digest_parity_real_data():
    _needs_real_db()
    start_ms, end_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    old_rows = _old("ETHUSDT", start_ms, end_ms)
    new_rows = _mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert _digest(old_rows) == _digest(new_rows)


def test_repeated_run_determinism():
    _needs_real_db()
    start_ms, end_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    d1 = _digest(_mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB))
    d2 = _digest(_mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB))
    d3 = _digest(_mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB))
    assert d1 == d2 == d3


def test_output_invariant_to_batch_size_real_data():
    _needs_real_db()
    start_ms, end_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)

    def _rows(batch_size):
        result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=REAL_SOURCE_DB, batch_size=batch_size)
        return [(int(ts), float(px)) for ts, px in result.iter_rows()]

    big_batch = _rows(1_000_000)
    tiny_batch = _rows(7)
    assert big_batch == tiny_batch


# ---------------------------------------------------------------------------
# Inclusive-boundary parity: a row exactly AT end_ms must be included, a
# row at end_ms+1 must be excluded (matches oracle's ts_ms<=end_ms)
# ---------------------------------------------------------------------------

def test_inclusive_end_boundary_parity_real_data():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts = conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old_n_at_boundary = _old_row_count("ETHUSDT", MAY_ARCHIVE_START_MS, int(last_ts))
    old_n_before_boundary = _old_row_count("ETHUSDT", MAY_ARCHIVE_START_MS, int(last_ts) - 1)
    assert old_n_at_boundary == old_n_before_boundary + 1  # the boundary row itself is included

    rows_at = _mark_series_v2(_root(), "ETHUSDT", MAY_ARCHIVE_START_MS, int(last_ts), source_db_path=REAL_SOURCE_DB)
    rows_before = _mark_series_v2(_root(), "ETHUSDT", MAY_ARCHIVE_START_MS, int(last_ts) - 1, source_db_path=REAL_SOURCE_DB)
    assert len(rows_at) == old_n_at_boundary
    assert len(rows_before) == old_n_before_boundary
    assert rows_at[-1][0] == int(last_ts)


# ---------------------------------------------------------------------------
# Empty range / bounded window
# ---------------------------------------------------------------------------

def test_empty_range_returns_empty_list():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_300_000  # 2020-09, long before any collection
    old_rows = _old("ETHUSDT", lo, hi)
    new_rows = _mark_series_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB)
    assert old_rows == []
    assert new_rows == []


def test_bounded_window_row_count_sane_real_data():
    _needs_real_db()
    start_ms, end_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 60_000
    rows = _mark_series_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert 0 < len(rows) < 100_000  # a 60s window is never anywhere near this large
    for ts, px in rows:
        assert start_ms <= ts <= end_ms
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
    _mark_series_v2(root, "ETHUSDT", MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_mark_series_v1_no_longer_called_from_live_path():
    for fn in (mod._choose_entry, mod._simulate_variant, mod._path_stats, mod.main):
        assert "_mark_series(" not in inspect.getsource(fn)
    for fn in (mod._choose_entry, mod._simulate_variant, mod._path_stats):
        assert "_mark_series_v2(" in inspect.getsource(fn)


def test_liquidations_and_shadow_runner_imports_untouched():
    """Out-of-allowlist `liquidations` reads (an unbounded `MAX(ts_ms)`
    aggregate in `main()`, plus whatever `_bucket_events` does
    internally) and the protected `s34_shadow_paper_runner` imports are
    explicitly out of scope for this range-read gate -- confirm they
    remain present/unchanged, not migrated."""
    src = inspect.getsource(mod)
    assert "FROM liquidations" in src  # main()'s MAX(ts_ms) probe -- untouched, out-of-allowlist
    assert "import tools.s34_shadow_paper_runner as runner" in src
    assert "_bucket_events" in src and "_fill_quote" in src and "_close_trade" in src


# ---------------------------------------------------------------------------
# Synthetic fixtures: symbol-mismatch safety and trust-failure fail-closed
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


def test_symbol_filter_parity_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 3000.0, 0.0001, None),
        (2, MAY, "SOLUSDT", 100.0, 0.0009, None),
        (3, MAY + 1000, "ETHUSDT", 3001.0, 0.0002, None),
    ])
    eth_rows = _mark_series_v2(root, "ETHUSDT", MAY, MAY + 1000, source_db_path=db)
    sol_rows = _mark_series_v2(root, "SOLUSDT", MAY, MAY + 1000, source_db_path=db)
    assert eth_rows == [(MAY, 3000.0), (MAY + 1000, 3001.0)]
    assert sol_rows == [(MAY, 100.0)]


def test_empty_range_synthetic_returns_empty(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 0.0001, None)])
    assert _mark_series_v2(root, "ETHUSDT", MAY + 10_000, MAY + 20_000, source_db_path=db) == []


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
             "manifest_sha256": "0" * 64,  # deliberately WRONG -- must fail the trust check
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": "irrelevant"}
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        _mark_series_v2(root, "ETHUSDT", MAY, MAY + 1000)
