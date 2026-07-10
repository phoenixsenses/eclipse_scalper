"""Range-read consumer migration (BATCH-STORAGE-ROTATION-RETENTION-
RANGE-READ-CONSUMER-MIGRATION-V8), target 3 of 3.

`tools/research_s34_post_tp_continuation.py`'s inline "first TP
crossing" query inside `main()` (`mark_prices`, `ts_ms > ? AND
ts_ms <= ? AND mark_price >= ? ORDER BY ts_ms ASC LIMIT 1` -- a bounded
range PLUS a value filter PLUS take-first-row semantics) was extracted
into `_find_tp_crossing()` (the parity oracle, unchanged) and migrated
to `_find_tp_crossing_v2()` (reader-backed via `plan_read`/
`execute_read`, using its `filters=` kwarg for `mark_price>=?` and a
short-circuit `for ... return` for `LIMIT 1`). `_find_tp_crossing()` is
no longer called from `main()`'s live path.

Call-site audit: symbol is hardcoded to `'ETHUSDT'` via a SQL literal
in the oracle (not a parameter) -- the file's only call site never
uses another symbol, so `_find_tp_crossing_v2` also hardcodes it,
documented as such rather than pretending it is a genuine runtime
parameter. Real ARCHIVE_ONLY/HYBRID/SQLITE_ONLY production smoke is
available (ETHUSDT has a real `mark_prices` archive partition).

Range-boundary note: identical exclusive-lower/inclusive-upper mapping
as `research_s34_hold_sweep.py`/`research_s34_session_analysis.py` --
`start_ms=gt_ms+1, end_ms=le_ms+1`.

LIMIT-1-with-extra-filter note: `execute_read` streams rows in
canonical `(ts_ms ASC, id ASC)` order with `mark_price>=threshold`
applied per-row via `filters=`; taking the FIRST row from that ordered,
filtered stream is exactly `ORDER BY ts_ms ASC LIMIT 1` over the same
WHERE predicate. Proven both for "a match exists" and "no match at
all" (oracle returns `None`, `_v2` returns `None`).

Out-of-scope, untouched (this gate is range-read only):
  * `mark_price_at()` -- backward ASOF, left on direct SQL.
  * `funding_rates` table queries (2 call sites) -- a distinct table
    from `mark_prices`, not in the research-reader allowlist at all
    (confirmed to exist in the real DB, just never migrated).
  * `s34_trades` / `INTEL_DB` (`data/s34_intelligence.db`) queries (2
    call sites) -- a different SQLite database entirely.
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
import tools.research_s34_post_tp_continuation as mod
from tools.research_s34_post_tp_continuation import _find_tp_crossing, _find_tp_crossing_v2

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


def _old(gt_ms, le_ms, threshold):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return _find_tp_crossing(conn, gt_ms, le_ms, threshold)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# ---------------------------------------------------------------------------
# Real production smoke: ARCHIVE_ONLY, a threshold that genuinely matches
# somewhere in the middle of the window (not trivially first/last row)
# ---------------------------------------------------------------------------

def test_v2_parity_archive_only_real_match_and_plan_mode():
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=gt_ms + 1, end_ms=le_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    threshold = 2260.0  # real data in this window ranges ~2256.57-2262.86
    old_row = _old(gt_ms, le_ms, threshold)
    new_row = _find_tp_crossing_v2(_root(), gt_ms, le_ms, threshold, source_db_path=REAL_SOURCE_DB)
    assert old_row is not None
    assert new_row == old_row
    assert new_row[1] >= threshold


def test_v2_parity_archive_only_no_match_real_data():
    """Threshold above every real price in the window -- both oracle
    and reader-backed path must agree on `None` (no crossing found)."""
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    threshold = 999_999.0  # far above any real ETHUSDT price
    old_row = _old(gt_ms, le_ms, threshold)
    new_row = _find_tp_crossing_v2(_root(), gt_ms, le_ms, threshold, source_db_path=REAL_SOURCE_DB)
    assert old_row is None
    assert new_row is None


# ---------------------------------------------------------------------------
# Real production smoke: SQLITE_ONLY recent live window
# ---------------------------------------------------------------------------

def test_v2_parity_sqlite_only_recent_live_window():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
        recent_rows = conn.execute(
            "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms ASC",
            (max_ts - 300_000, max_ts),
        ).fetchall()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    assert recent_rows
    threshold = sorted(p for (p,) in recent_rows)[len(recent_rows) // 2]  # median price -- guarantees a real match
    gt_ms, le_ms = max_ts - 300_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=gt_ms + 1, end_ms=le_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_row = _old(gt_ms, le_ms, threshold)
    new_row = _find_tp_crossing_v2(_root(), gt_ms, le_ms, threshold, source_db_path=REAL_SOURCE_DB)
    assert old_row is not None
    assert new_row == old_row


# ---------------------------------------------------------------------------
# Real production smoke: HYBRID window straddling the archive/live boundary
# ---------------------------------------------------------------------------

HYBRID_GT_MS = MAY_ARCHIVE_END_MS - 150_000
HYBRID_LE_MS = MAY_ARCHIVE_END_MS + 150_000


def test_v2_parity_hybrid_real_boundary_and_plan_mode():
    _needs_real_db()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=HYBRID_GT_MS + 1, end_ms=HYBRID_LE_MS + 1)
    assert plan.mode == "HYBRID"
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        rows = conn.execute(
            "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms ASC",
            (HYBRID_GT_MS, HYBRID_LE_MS),
        ).fetchall()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    assert rows
    threshold = sorted(p for (p,) in rows)[len(rows) // 2]
    old_row = _old(HYBRID_GT_MS, HYBRID_LE_MS, threshold)
    new_row = _find_tp_crossing_v2(_root(), HYBRID_GT_MS, HYBRID_LE_MS, threshold, source_db_path=REAL_SOURCE_DB)
    assert old_row is not None
    assert new_row == old_row


def test_provenance_reports_hybrid_source_and_filters():
    _needs_real_db()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=HYBRID_GT_MS + 1, end_ms=HYBRID_LE_MS + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"),
                              filters=(("mark_price", ">=", 0.0),), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "HYBRID"
    assert prov["filters"] == [("mark_price", ">=", 0.0)]
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"


# ---------------------------------------------------------------------------
# LIMIT-1 first-match semantics: prove the returned row is genuinely the
# FIRST (by ts_ms ASC) qualifying row, not merely A qualifying row.
# ---------------------------------------------------------------------------

def test_first_match_is_genuinely_first_by_ts_ms_real_data():
    _needs_real_db()
    gt_ms, le_ms = MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000
    threshold = 2260.0
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        all_qualifying = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? AND mark_price>=? ORDER BY ts_ms ASC",
            (gt_ms, le_ms, threshold),
        ).fetchall()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    assert len(all_qualifying) > 1  # need at least 2 qualifying rows to prove "first", not "only"
    new_row = _find_tp_crossing_v2(_root(), gt_ms, le_ms, threshold, source_db_path=REAL_SOURCE_DB)
    assert tuple(new_row) == tuple(all_qualifying[0])
    assert new_row != tuple(all_qualifying[-1])  # sanity: window has real variation, not a single repeated row


# ---------------------------------------------------------------------------
# Exclusive-lower / inclusive-upper boundary parity
# ---------------------------------------------------------------------------

def test_exclusive_lower_boundary_parity_real_data():
    """A qualifying row exactly AT gt_ms must be EXCLUDED."""
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        first_ts, first_px = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
            (MAY_ARCHIVE_START_MS,),
        ).fetchone()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    le_ms = int(first_ts) + 300_000
    threshold = float(first_px)  # the boundary row itself would qualify if included
    old_row = _old(int(first_ts), le_ms, threshold)
    new_row = _find_tp_crossing_v2(_root(), int(first_ts), le_ms, threshold, source_db_path=REAL_SOURCE_DB)
    assert new_row == old_row
    if old_row is not None:
        assert old_row[0] > first_ts  # never the boundary row itself


def test_inclusive_upper_boundary_parity_real_data():
    """A qualifying row exactly AT le_ms must be INCLUDED."""
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts, last_px = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000),
        ).fetchone()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    # threshold set so ONLY the last row qualifies -- proves inclusive upper bound
    threshold = float(last_px)
    gt_ms = int(last_ts) - 1  # window of exactly the last row
    old_row = _old(gt_ms, int(last_ts), threshold)
    new_row = _find_tp_crossing_v2(_root(), gt_ms, int(last_ts), threshold, source_db_path=REAL_SOURCE_DB)
    assert old_row == (int(last_ts), float(last_px))
    assert new_row == old_row


# ---------------------------------------------------------------------------
# Empty range / no rows at all
# ---------------------------------------------------------------------------

def test_empty_range_returns_none():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_300_000
    old_row = _old(lo, hi, 0.0)
    new_row = _find_tp_crossing_v2(_root(), lo, hi, 0.0, source_db_path=REAL_SOURCE_DB)
    assert old_row is None
    assert new_row is None


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
    _find_tp_crossing_v2(root, MAY_ARCHIVE_START_MS, MAY_ARCHIVE_START_MS + 300_000, 2260.0, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_find_tp_crossing_v1_no_longer_called_from_live_path():
    src = inspect.getsource(mod.main)
    assert "_find_tp_crossing(" not in src
    assert "_find_tp_crossing_v2(" in src


def test_out_of_scope_patterns_left_untouched():
    """`mark_price_at()` (backward ASOF), `funding_rates` (out-of-
    allowlist table), and `s34_trades`/`INTEL_DB` (different DB) are
    explicitly out of scope for this range-read gate -- confirmed still
    on direct SQL, unmigrated."""
    src = inspect.getsource(mod)
    assert "ORDER BY ts_ms DESC LIMIT 1" in src  # mark_price_at(), untouched
    assert "FROM funding_rates" in src  # untouched, out-of-allowlist table
    assert src.count("FROM funding_rates") == 2  # both call sites present, unmigrated
    assert "FROM s34_trades" in src  # untouched, different DB
    assert "lookup_latest_at_or_before" not in src


def test_funding_rates_table_exists_in_real_db_confirming_not_dead_code():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='funding_rates'").fetchone()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    assert row is not None  # confirms the untouched query targets a real, existing table


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


def test_first_match_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [
        (1, MAY, "ETHUSDT", 100.0, 0.0001, None),      # excluded: at gt_ms boundary
        (2, MAY + 1000, "ETHUSDT", 99.0, 0.0001, None),  # below threshold
        (3, MAY + 2000, "ETHUSDT", 105.0, 0.0001, None), # first qualifying row
        (4, MAY + 3000, "ETHUSDT", 110.0, 0.0001, None), # also qualifies but not first
    ])
    row = _find_tp_crossing_v2(root, MAY, MAY + 3000, 100.0, source_db_path=db)
    assert row == (MAY + 2000, 105.0)


def test_no_match_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY + 1000, "ETHUSDT", 50.0, 0.0001, None)])
    assert _find_tp_crossing_v2(root, MAY, MAY + 3000, 100.0, source_db_path=db) is None


def test_empty_range_synthetic_returns_none(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [(1, MAY, "ETHUSDT", 100.0, 0.0001, None)])
    assert _find_tp_crossing_v2(root, MAY + 10_000, MAY + 20_000, 0.0, source_db_path=db) is None


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
    rows = [(1, MAY, "ETHUSDT", 100.0, 0.0001, None)]
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
        _find_tp_crossing_v2(root, MAY, MAY + 1000, 0.0)
