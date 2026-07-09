"""Range-read consumer migration parity: source_quality_reconciliation
(BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-CONSUMER-MIGRATION-SOURCE-
QUALITY-V1).

`tools/research_s34_source_quality_reconciliation.py`'s `window_health`
(two bounded allowlisted range reads: a mark_prices window + an agg_trades
COUNT) is migrated to `window_health_v2` (reader-backed via plan_read/
execute_read). `window_health` is kept unchanged as the direct-SQL oracle
(it reads the module-level `conn_m`); tests drive it by assigning a real
`mode=ro` connection to `mod.conn_m`.

Symbol is hardcoded 'ETHUSDT' in both reads. Both allowlisted tables have
real archive partitions (mark_prices/ETHUSDT/2026-05, agg_trades/ETHUSDT/
2026-02) -- since those are DIFFERENT months, a single window is
archive-backed for at most one table at a time, so real ARCHIVE_ONLY smoke
is proven per-table: mark ARCHIVE in a 2026-05 window, agg ARCHIVE in a
2026-02 window. HYBRID and SQLITE_ONLY are covered per-table too.

Range-boundary note: the oracle uses INCLUSIVE upper bounds (`ts_ms<=?`);
the reader uses half-open `[start, end)`, so window_health_v2 passes
`end_ms+1` on each read. Proven exact by an inclusive-boundary test.

Import-safety (from the prep gate) must remain intact; a test re-checks it.
The out-of-allowlist `gaps`/`liquidations` reads and the whole main() flow
are untouched and NOT exercised here (no full 220-signal reconciliation,
no CSV/JSON artifact written to any real path).
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
import tools.research_s34_source_quality_reconciliation as mod
from tools.research_s34_source_quality_reconciliation import window_health, window_health_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
MARK_SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.filterwarnings("ignore")

# real archive partition windows
MARK_ARCHIVE_START_MS = 1777593600000        # mark_prices/ETHUSDT/2026-05 partition start
MARK_PARTITION_END_MS = 1780272000000
AGG_ARCHIVE_START_MS = 1771165588000         # agg_trades/ETHUSDT/2026-02 real data start
AGG_PARTITION_END_MS = 1772323200000


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _oracle(ws, birth):
    """Drive the direct-SQL oracle by injecting a real mode=ro conn as the
    module-level `conn_m` it reads (restored afterward)."""
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    saved = getattr(mod, "conn_m", None)
    mod.conn_m = conn
    try:
        return window_health(ws, birth)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
        mod.conn_m = saved


def _mark_rows(ws, birth):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms",
            (ws - 15_000, birth + 15_000)).fetchall()
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _digest(seq):
    h = hashlib.sha256()
    for item in seq:
        h.update(repr(item).encode())
    return h.hexdigest()


# --- full window_health dict parity: real ARCHIVE-backed windows (per table) ---

def test_window_health_parity_mark_archive_window():
    _needs_real_db()
    ws, birth = MARK_ARCHIVE_START_MS + 60_000, MARK_ARCHIVE_START_MS + 60_000 + 120_000
    mplan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ws - 15_000, end_ms=birth + 15_000 + 1)
    assert mplan.mode == "ARCHIVE_ONLY"  # mark read is genuinely archive-served
    old = _oracle(ws, birth)
    new = window_health_v2(_root(), ws, birth, source_db_path=REAL_SOURCE_DB)
    assert old["mark_rows_n"] > 0
    assert new == old


def test_window_health_parity_agg_archive_window():
    _needs_real_db()
    ws, birth = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 120_000
    aplan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=ws, end_ms=birth + 1)
    assert aplan.mode == "ARCHIVE_ONLY"  # agg COUNT is genuinely archive-served
    old = _oracle(ws, birth)
    new = window_health_v2(_root(), ws, birth, source_db_path=REAL_SOURCE_DB)
    assert old["agg_rows_n"] > 0
    assert new == old


def test_window_health_parity_sqlite_only_recent_live():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    ws, birth = max_ts - 120_000, max_ts
    mplan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ws - 15_000, end_ms=birth + 15_000 + 1)
    assert mplan.mode == "SQLITE_ONLY"
    old = _oracle(ws, birth)
    new = window_health_v2(_root(), ws, birth, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_window_health_parity_hybrid_mark_boundary():
    _needs_real_db()
    ws, birth = MARK_PARTITION_END_MS - 120_000, MARK_PARTITION_END_MS + 60_000
    mplan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ws - 15_000, end_ms=birth + 15_000 + 1)
    assert mplan.mode == "HYBRID"  # mark read straddles the 2026-05 partition boundary
    old = _oracle(ws, birth)
    new = window_health_v2(_root(), ws, birth, source_db_path=REAL_SOURCE_DB)
    assert new == old


def test_window_health_parity_hybrid_agg_boundary():
    _needs_real_db()
    ws, birth = AGG_PARTITION_END_MS - 120_000, AGG_PARTITION_END_MS + 60_000
    aplan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=ws, end_ms=birth + 1)
    assert aplan.mode == "HYBRID"  # agg COUNT straddles the 2026-02 partition boundary
    old = _oracle(ws, birth)
    new = window_health_v2(_root(), ws, birth, source_db_path=REAL_SOURCE_DB)
    assert new == old


# --- component-level parity: mark row count / ordering / digest, agg COUNT ---

def test_mark_read_rowcount_ordering_digest_parity():
    _needs_real_db()
    ws, birth = MARK_ARCHIVE_START_MS + 60_000, MARK_ARCHIVE_START_MS + 60_000 + 120_000
    old_rows = [r[0] for r in _mark_rows(ws, birth)]
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ws - 15_000, end_ms=birth + 15_000 + 1)
    result = RR.execute_read(plan, columns=("ts_ms",), source_db_path=REAL_SOURCE_DB)
    new_rows = [t for (t,) in result.iter_rows()]
    assert len(new_rows) == len(old_rows) and len(old_rows) > 0     # row-count parity
    assert new_rows == old_rows                                     # ordering + first/last ts parity
    assert _digest(new_rows) == _digest(old_rows)                   # deterministic digest parity


def test_agg_count_parity_matches_direct_count():
    _needs_real_db()
    ws, birth = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 120_000
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        direct_n = conn.execute(
            "SELECT COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms >= ? AND ts_ms <= ?",
            (ws, birth)).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=ws, end_ms=birth + 1)
    reader_n = sum(1 for _ in RR.execute_read(plan, columns=("ts_ms",), source_db_path=REAL_SOURCE_DB).iter_rows())
    assert reader_n == direct_n and direct_n > 0


def test_inclusive_end_boundary_parity_real():
    _needs_real_db()
    # a mark_prices row exactly AT birth must be counted; at birth+1 excluded.
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts = conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (MARK_ARCHIVE_START_MS, MARK_ARCHIVE_START_MS + 300_000)).fetchone()[0]
        direct_at = conn.execute(
            "SELECT COUNT(*) FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?",
            (MARK_ARCHIVE_START_MS, int(last_ts))).fetchone()[0]
        direct_before = conn.execute(
            "SELECT COUNT(*) FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?",
            (MARK_ARCHIVE_START_MS, int(last_ts) - 1)).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    plan_at = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=MARK_ARCHIVE_START_MS, end_ms=int(last_ts) + 1)
    n_at = sum(1 for _ in RR.execute_read(plan_at, columns=("ts_ms",), source_db_path=REAL_SOURCE_DB).iter_rows())
    plan_before = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=MARK_ARCHIVE_START_MS, end_ms=int(last_ts))
    n_before = sum(1 for _ in RR.execute_read(plan_before, columns=("ts_ms",), source_db_path=REAL_SOURCE_DB).iter_rows())
    assert n_at == direct_at
    assert n_before == direct_before
    assert n_at >= n_before + 1  # the boundary row itself is included


def test_empty_window_parity():
    _needs_real_db()
    ws, birth = 1_600_000_000_000, 1_600_000_060_000  # 2020-09, before any collection
    old = _oracle(ws, birth)
    new = window_health_v2(_root(), ws, birth, source_db_path=REAL_SOURCE_DB)
    assert old["mark_rows_n"] == 0 and old["agg_rows_n"] == 0 and old["mark_max_gap_ms"] == (birth - ws)
    assert new == old


def test_repeated_run_determinism():
    _needs_real_db()
    ws, birth = AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 120_000
    vals = [window_health_v2(_root(), ws, birth, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_provenance_reports_archive_source():
    _needs_real_db()
    ws, birth = MARK_ARCHIVE_START_MS + 60_000, MARK_ARCHIVE_START_MS + 60_000 + 120_000
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ws - 15_000, end_ms=birth + 15_000 + 1)
    result = RR.execute_read(plan, columns=("ts_ms",), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["table"] == "mark_prices"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["ts_ms"]
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
    window_health_v2(root, AGG_ARCHIVE_START_MS, AGG_ARCHIVE_START_MS + 120_000, source_db_path=REAL_SOURCE_DB)
    window_health_v2(root, MARK_ARCHIVE_START_MS + 60_000, MARK_ARCHIVE_START_MS + 60_000 + 120_000, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_main_still_uses_v2_not_oracle():
    import inspect
    src = inspect.getsource(mod.main)
    assert "window_health_v2(root, ws, birth" in src
    assert "= window_health(ws, birth)" not in src  # oracle no longer wired into main


def test_import_safety_still_intact():
    # The prep gate's import-safety must survive this migration: importing
    # (again, fresh) must not have opened DBs or written files. We assert the
    # structural invariants that guarantee it.
    import ast
    with open(mod.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if isinstance(node, ast.Assign):
            continue
        if isinstance(node, ast.If):  # __main__ guard
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # docstring
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            assert isinstance(fn, ast.Attribute) and fn.attr == "insert"  # sys.path bootstrap only
            continue
        raise AssertionError(f"unexpected top-level executable statement: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Synthetic SQLITE_ONLY window + trust-failure fail-closed.
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

MAY = 1777593600000


def _sqlite(path, mark_rows, agg_rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute(MP_DDL)
    c.execute("CREATE INDEX ixm ON mark_prices(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?,?)", mark_rows)
    c.execute(AT_DDL)
    c.execute("CREATE INDEX ixa ON agg_trades(symbol, ts_ms)")
    c.executemany("INSERT INTO agg_trades VALUES (?,?,?,?,?,?,?)", agg_rows)
    c.commit()
    c.close()


def test_window_health_v2_sqlite_only_synthetic_and_symbol_filter(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    ws, birth = MAY, MAY + 5_000
    _sqlite(
        db,
        mark_rows=[
            (1, ws, "ETHUSDT", 3000.0, 0.0, None),
            (2, ws + 1_000, "ETHUSDT", 3001.0, 0.0, None),
            (3, birth, "ETHUSDT", 3002.0, 0.0, None),
            (4, ws + 500, "SOLUSDT", 99999.0, 0.0, None),   # other symbol must not leak
        ],
        agg_rows=[
            (1, ws + 100, "ETHUSDT", 3000.0, 1.0, 3000.0, 0),
            (2, ws + 200, "ETHUSDT", 3000.0, 1.0, 3000.0, 1),
            (3, ws + 300, "SOLUSDT", 99999.0, 1.0, 99999.0, 0),  # other symbol must not count
        ],
    )
    res = window_health_v2(root, ws, birth, source_db_path=db)
    assert res["mark_rows_n"] == 3          # 3 ETHUSDT rows in [ws, birth]; SOLUSDT excluded
    assert res["agg_rows_n"] == 2           # 2 ETHUSDT agg rows in [ws, birth]; SOLUSDT excluded
    assert res["mark_lead_ok"] is True and res["mark_trail_ok"] is True


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
             "manifest_sha256": "0" * 64,  # deliberately WRONG -> trust check must fail closed
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": "irrelevant"}
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        window_health_v2(root, MAY, MAY + 1_000)
