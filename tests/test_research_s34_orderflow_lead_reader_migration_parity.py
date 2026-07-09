"""Range-read consumer migration V2 parity #1 (BATCH-STORAGE-ROTATION-
RETENTION-RANGE-READ-CONSUMER-MIGRATION-V2).

`tools/research_s34_orderflow_lead.py`'s two bounded range reads are
migrated to reader-backed counterparts via `plan_read`/`execute_read`:
  - `load_ofi_bins`  (agg_trades: `GROUP BY ts_ms/bin_ms` + signed SUM)
                     -> `load_ofi_bins_v2`
  - `load_marks_range` (mark_prices: `ORDER BY ts_ms` window) ->
                     `load_marks_range_v2`
Both oracles are kept unchanged as parity references.

Call-site audit: `symbol` is a GENUINE runtime parameter (SYMBOLS =
ETHUSDT/SOLUSDT/BTCUSDT, `--symbols` arg). agg_trades has a real archive
partition for ETHUSDT/2026-02 and mark_prices for ETHUSDT/2026-05, so this
suite proves REAL ARCHIVE_ONLY and HYBRID production smokes for ETHUSDT (in
addition to SQLITE_ONLY recent-window smokes), per the allowlisted-table
rules.

Range-boundary note: the oracles use an INCLUSIVE upper bound
(`ts_ms<=end_ms`); the reader uses half-open `[start, end)`, so the `_v2`
helpers pass `end_ms+1`. A dedicated test proves a row exactly AT the
boundary is included.
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
import tools.research_s34_orderflow_lead as mod
from tools.research_s34_orderflow_lead import (
    load_ofi_bins, load_ofi_bins_v2, load_marks_range, load_marks_range_v2,
)

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)

pytestmark = pytest.mark.filterwarnings("ignore")

# real agg_trades/ETHUSDT/2026-02 archive: data begins ~1771165587967
AGG_ARCHIVE_START_MS = 1771165588000
AGG_PARTITION_END_MS = 1772323200000       # 2026-02 -> 2026-03 boundary
MARK_ARCHIVE_START_MS = 1777593600000      # mark_prices/ETHUSDT/2026-05 partition start


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old_ofi(symbol, start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return load_ofi_bins(conn, symbol, start_ms, end_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_marks(symbol, start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        m = load_marks_range(conn, symbol, start_ms, end_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    return list(zip(m.ts, m.px))


def _agg_max_ts(symbol):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute("SELECT MAX(ts_ms) FROM agg_trades WHERE symbol=?", (symbol,)).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _digest(seq):
    h = hashlib.sha256()
    for item in seq:
        h.update(repr(item).encode())
    return h.hexdigest()


# --- OFI bins: real ARCHIVE_ONLY parity (agg_trades/ETHUSDT/2026-02) ---

def test_ofi_bins_parity_archive_only_real():
    _needs_real_db()
    start_ms, end_ms = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old_ofi("ETHUSDT", start_ms, end_ms)
    new = load_ofi_bins_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert len(new) == len(old) and len(old) > 0
    assert [b for b, _ in new] == [b for b, _ in old]            # bin timestamps + ordering parity
    for (bo, vo), (bn, vn) in zip(old, new):
        assert bn == bo
        assert vn == pytest.approx(vo, rel=1e-9)                  # signed-SUM aggregate parity
    assert _digest((b, round(v, 6)) for b, v in new) == _digest((b, round(v, 6)) for b, v in old)


def test_ofi_bins_parity_hybrid_real_boundary():
    _needs_real_db()
    start_ms, end_ms = AGG_PARTITION_END_MS - 150_000, AGG_PARTITION_END_MS + 150_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "HYBRID"
    old = _old_ofi("ETHUSDT", start_ms, end_ms)
    new = load_ofi_bins_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert [b for b, _ in new] == [b for b, _ in old]
    for (bo, vo), (bn, vn) in zip(old, new):
        assert vn == pytest.approx(vo, rel=1e-9)


def test_ofi_bins_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _agg_max_ts("ETHUSDT")
    start_ms, end_ms = max_ts - 60_000, max_ts
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old_ofi("ETHUSDT", start_ms, end_ms)
    new = load_ofi_bins_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert [b for b, _ in new] == [b for b, _ in old]
    for (bo, vo), (bn, vn) in zip(old, new):
        assert vn == pytest.approx(vo, rel=1e-9)


def test_ofi_bins_batch_size_invariance():
    _needs_real_db()
    start_ms, end_ms = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000
    bin_ms = mod.BIN_SEC * 1000

    def _reduce(batch_size):
        plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
        result = RR.execute_read(plan, columns=("ts_ms", "notional", "is_buyer_maker"),
                                 source_db_path=REAL_SOURCE_DB, batch_size=batch_size)
        bins: dict[int, float] = {}
        for ts_ms, notional, is_buyer_maker in result.iter_rows():
            b = int(ts_ms) // bin_ms
            bins[b] = bins.get(b, 0.0) + (float(notional) if is_buyer_maker == 0 else -float(notional))
        return [((b + 1) * bin_ms, bins[b]) for b in sorted(bins)]

    assert _reduce(1_000_000) == _reduce(11)


# --- marks range: real ARCHIVE_ONLY parity (mark_prices/ETHUSDT/2026-05) ---

def test_marks_range_parity_archive_only_real():
    _needs_real_db()
    start_ms, end_ms = MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old_marks("ETHUSDT", start_ms, end_ms)
    new = load_marks_range_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    new_rows = list(zip(new.ts, new.px))
    assert len(new_rows) == len(old) and len(old) > 0             # row-count parity
    assert new.ts == [t for t, _ in old]                          # first/last ts + ordering parity
    assert _digest(new_rows) == _digest(old)                      # deterministic digest parity


def test_marks_range_parity_sqlite_only_recent_live():
    _needs_real_db()
    max_ts = _agg_max_ts("ETHUSDT")  # marks exist well past agg; use agg max as a safe live anchor
    start_ms, end_ms = max_ts - 300_000, max_ts
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old_marks("ETHUSDT", start_ms, end_ms)
    new = load_marks_range_v2(_root(), "ETHUSDT", start_ms, end_ms, source_db_path=REAL_SOURCE_DB)
    assert list(zip(new.ts, new.px)) == old


def test_inclusive_end_boundary_parity_real():
    _needs_real_db()
    # Use the window's own last real mark_prices ts as the exact end_ms.
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts = conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old_at = _old_marks("ETHUSDT", MARK_ARCHIVE_START_MS, int(last_ts))
    new_at = load_marks_range_v2(_root(), "ETHUSDT", MARK_ARCHIVE_START_MS, int(last_ts), source_db_path=REAL_SOURCE_DB)
    assert list(zip(new_at.ts, new_at.px)) == old_at
    assert new_at.ts[-1] == int(last_ts)  # the boundary row itself is included


def test_empty_range_parity():
    _needs_real_db()
    lo, hi = 1_600_000_000_000, 1_600_000_060_000  # 2020-09, before any collection
    assert _old_ofi("ETHUSDT", lo, hi) == []
    assert load_ofi_bins_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB) == []
    old_marks = _old_marks("ETHUSDT", lo, hi)
    new_marks = load_marks_range_v2(_root(), "ETHUSDT", lo, hi, source_db_path=REAL_SOURCE_DB)
    assert old_marks == []
    assert list(zip(new_marks.ts, new_marks.px)) == []


def test_repeated_run_determinism():
    _needs_real_db()
    s, e = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000
    vals = [load_ofi_bins_v2(_root(), "ETHUSDT", s, e, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_provenance_reports_archive_source_and_columns():
    _needs_real_db()
    start_ms, end_ms = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=start_ms, end_ms=end_ms + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "notional", "is_buyer_maker"), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["table"] == "agg_trades"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["ts_ms", "notional", "is_buyer_maker"]
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert len(prov["archive_segments"]) == 1
    assert prov["archive_segments"][0]["manifest_sha256"]
    assert prov["row_count"] > 0


def test_no_full_reverify_referenced():
    import inspect
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    load_ofi_bins_v2(root, "ETHUSDT", AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 60_000, source_db_path=REAL_SOURCE_DB)
    load_marks_range_v2(root, "ETHUSDT", MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


# --- symbol-mismatch safety + SQLITE_ONLY for a non-archived symbol (synthetic) ---

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


def _sqlite_agg(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute(AT_DDL)
    c.execute("CREATE INDEX ix ON agg_trades(symbol, ts_ms)")
    c.executemany("INSERT INTO agg_trades VALUES (?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


def test_ofi_bins_symbol_mismatch_safety_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    bin_ms = mod.BIN_SEC * 1000
    base = 1_777_000_000_000
    _sqlite_agg(db, [
        (1, base + 100, "ETHUSDT", 3000.0, 1.0, 3000.0, 0),   # ETH taker-buy +3000
        (2, base + 200, "ETHUSDT", 3000.0, 1.0, 1000.0, 1),   # ETH taker-sell -1000
        (3, base + 100, "SOLUSDT", 100.0, 1.0, 9999.0, 0),    # SOL, must not leak into ETH
    ])
    eth = load_ofi_bins_v2(root, "ETHUSDT", base, base + bin_ms, source_db_path=db)
    assert eth == [((base // bin_ms + 1) * bin_ms, 2000.0)]     # 3000 - 1000, SOL excluded
    sol = load_ofi_bins_v2(root, "SOLUSDT", base, base + bin_ms, source_db_path=db)
    assert sol == [((base // bin_ms + 1) * bin_ms, 9999.0)]
