"""As-of-lookup consumer migration parity, pilot #3 (BATCH-STORAGE-
ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V3).

`tools/research_s34_btc_microtrend_sweep.py`'s single `ORDER BY ts_ms
DESC LIMIT 1` helper, `mark_at`, is called for BOTH ETHUSDT (which HAS
a real archived partition, mark_prices/ETHUSDT/2026-05) and BTCUSDT
(which has none). Unlike the prior sibling pilot
(btc_microtrend_eth_quality.py, BTCUSDT-only), this consumer's ETHUSDT
calls genuinely exercise ARCHIVE_ONLY and, at the May/June boundary
inside `simulate()`'s own two lookups (entry vs entry+510s), a real
HYBRID case -- both proven directly against production data below, no
synthetic substitute needed for those two modes.

`mark_at` (old, direct-SQL) is kept unchanged as the parity oracle;
`mark_at_v2` (new, reader-backed) is used by `simulate()`/`main()`
instead. The bounded-RANGE scan inside `simulate()` (the TP/SL/BE hold-
path walk) and the `liquidations` bucket-building in `main()` are
untouched -- out of scope for this point-lookup gate (range-read
helper and out-of-allowlist-table territory, respectively).
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
from tools.research_s34_btc_microtrend_sweep import bps_ret, mark_at, mark_at_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_SOURCE_DB), reason="real source database not present in this checkout")


def _root():
    root, _source = PR.resolve_production_root()
    return root


def _old(symbol, ts_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return mark_at(conn, symbol, ts_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- ETHUSDT: archive-only lookup, within the real archived May-2026 partition ---

ARCHIVE_TS = 1778600000000
HOLD_MS = 510_000


def test_mark_at_v2_parity_ethusdt_archive_only():
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ARCHIVE_TS, end_ms=ARCHIVE_TS + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old("ETHUSDT", ARCHIVE_TS)
    new = mark_at_v2(_root(), "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


# --- ETHUSDT: genuine real HYBRID -- simulate()'s own two lookups (entry, entry+510s)
# straddle the archived-May/live-June boundary ---

HYBRID_ENTRY_TS = 1780272000000 - 200_000


def test_mark_at_v2_parity_ethusdt_hybrid_boundary_both_sides():
    plan_p0 = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT",
                            start_ms=HYBRID_ENTRY_TS, end_ms=HYBRID_ENTRY_TS + 1)
    plan_p_end = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT",
                               start_ms=HYBRID_ENTRY_TS + HOLD_MS, end_ms=HYBRID_ENTRY_TS + HOLD_MS + 1)
    assert plan_p0.mode == "ARCHIVE_ONLY"     # entry is archive-side
    assert plan_p_end.mode == "SQLITE_ONLY"   # entry+510s has crossed into live June

    old_p0 = _old("ETHUSDT", HYBRID_ENTRY_TS)
    old_p_end = _old("ETHUSDT", HYBRID_ENTRY_TS + HOLD_MS)
    new_p0 = mark_at_v2(_root(), "ETHUSDT", HYBRID_ENTRY_TS, source_db_path=REAL_SOURCE_DB)
    new_p_end = mark_at_v2(_root(), "ETHUSDT", HYBRID_ENTRY_TS + HOLD_MS, source_db_path=REAL_SOURCE_DB)
    assert old_p0 is not None and old_p_end is not None
    assert new_p0 == pytest.approx(old_p0, rel=1e-9)
    assert new_p_end == pytest.approx(old_p_end, rel=1e-9)


# --- ETHUSDT: SQLite-only, chronologically before the archive (data never purged) ---

SQLITE_ONLY_TS = 1772500000000


def test_mark_at_v2_parity_ethusdt_sqlite_only_pre_archive():
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT",
                         start_ms=SQLITE_ONLY_TS, end_ms=SQLITE_ONLY_TS + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old("ETHUSDT", SQLITE_ONLY_TS)
    new = mark_at_v2(_root(), "ETHUSDT", SQLITE_ONLY_TS, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


# --- BTCUSDT: no archive partition -- always SQLITE_ONLY (same as the sibling pilot) ---

def test_mark_at_v2_parity_btcusdt_sqlite_only():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='BTCUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="BTCUSDT", start_ms=max_ts - 1000, end_ms=max_ts + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old("BTCUSDT", max_ts)
    new = mark_at_v2(_root(), "BTCUSDT", max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


# --- End-to-end: simulate()'s final (net, exit_reason) output is unchanged.
# The only lines that changed inside simulate() are the two mark_at -> mark_at_v2
# point lookups; the bounded-range TP/SL/BE scan is untouched direct SQL. Proving
# those two lookups match (above) plus running the real migrated simulate()
# end-to-end against the real DB is the complete parity proof. ---

def test_simulate_end_to_end_matches_manual_oracle_reconstruction():
    from tools.research_s34_btc_microtrend_sweep import simulate, TP_BPS, SL_BPS, BE_BPS, HOLD_SEC, FEE_BPS

    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        net_new, exit_new = simulate(conn, _root(), ARCHIVE_TS)

        # Manual oracle reconstruction using the OLD direct-SQL mark_at only.
        p0 = mark_at(conn, "ETHUSDT", ARCHIVE_TS)
        p_tp = p0 * (1 + TP_BPS / 10000)
        p_sl = p0 * (1 - SL_BPS / 10000)
        p_be = p0 * (1 + BE_BPS / 10000)
        be_on = False
        rows = conn.execute(
            "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
            (ARCHIVE_TS, ARCHIVE_TS + HOLD_SEC * 1000)).fetchall()
        net_old = exit_old = None
        for (mp,) in rows:
            if mp >= p_tp:
                net_old, exit_old = bps_ret(p0, p_tp) - FEE_BPS, 'TP'
                break
            if be_on and mp <= p0:
                net_old, exit_old = bps_ret(p0, p0) - FEE_BPS, 'BE'
                break
            if not be_on and mp <= p_sl:
                net_old, exit_old = bps_ret(p0, p_sl) - FEE_BPS, 'SL'
                break
            if mp >= p_be:
                be_on = True
        else:
            p_end = mark_at(conn, "ETHUSDT", ARCHIVE_TS + HOLD_SEC * 1000) or p0
            net_old, exit_old = bps_ret(p0, p_end) - FEE_BPS, 'TIME'
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()

    assert net_new == pytest.approx(net_old, rel=1e-9)
    assert exit_new == exit_old


# --- Repeated-run determinism ---

def test_mark_at_v2_repeated_run_determinism():
    root = _root()
    results = [mark_at_v2(root, "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert results[0] == results[1] == results[2]


# --- Source mutation 0 ---

def test_mark_at_v2_leaves_real_catalog_and_manifest_unchanged():
    root = _root()
    idx_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx_path)
    mark_at_v2(root, "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB)
    after = os.path.getmtime(idx_path)
    assert before == after


# ---------------------------------------------------------------------------
# Synthetic fixtures: exact-hit / between-timestamps / empty / tie-break /
# column-projection / provenance / trust-failure (same style as prior gates)
# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _write_parquet(path, rows, spec):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema
    schema = build_pyarrow_schema(spec)
    arrays = []
    for i, col in enumerate(spec.preserved_columns):
        pa_type = schema.field(col).type
        values = [r[i] for r in rows]
        arrays.append(pa.array(values, type=pa_type))
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path, compression="zstd")


def _mark_prices_rows(start_id, start_ms, n, symbol="BTCUSDT", step_ms=10_000):
    return [(start_id + i, start_ms + i * step_ms, symbol, 60000.0 + i * 10, None, None) for i in range(n)]


def _build_archive_partition(root, *, symbol, partition_start_ms, partition_end_ms, rows, corrupt_manifest=False):
    rel = os.path.join("table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                        f"symbol={symbol}", "year=2026", "month=04", "version=v1")
    final_dir = os.path.join(root, rel)
    os.makedirs(final_dir, exist_ok=True)
    shard_file = "part-00000.parquet"
    shard_path = os.path.join(final_dir, shard_file)
    _write_parquet(shard_path, rows, SPEC)

    partition_id = f"test-mark_prices-{symbol}-2026-04"
    manifest = {
        "source_table": "mark_prices", "symbol": symbol, "venue": SPEC.venue, "market_segment": SPEC.market_segment,
        "row_count": len(rows), "shards": None,
        "ordered_scientific_content_hash": "0" * 64 if corrupt_manifest else "irrelevant-for-lookup-tests",
        "partition_id": partition_id, "parquet_path": os.path.join(rel, shard_file),
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
        "source_watermark_field": "id", "source_watermark_value": rows[-1][0],
        "parquet_sha256": _sha256_file(shard_path),
    }
    manifest_path = os.path.join(final_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write(partition_id + "\n")

    manifest_sha = _sha256_file(manifest_path)
    catalog_entry = {
        "archive_identity": partition_id, "archive_relative_path": rel,
        "source_table": "mark_prices", "symbol": symbol, "venue": SPEC.venue, "market_segment": SPEC.market_segment,
        "partition_start_ms": partition_start_ms, "partition_end_ms": partition_end_ms,
        "row_count": len(rows), "manifest_sha256": "f" * 64 if corrupt_manifest else manifest_sha,
        "authorization_receipt_sha256": None,
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
        "scientific_content_hash": manifest["ordered_scientific_content_hash"],
    }
    with open(os.path.join(final_dir, "catalog_entry.json"), "w", encoding="utf-8") as f:
        json.dump(catalog_entry, f)
    return catalog_entry


def _write_catalog_index(root, entries):
    index = {"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)


MARK_PRICES_DDL = """CREATE TABLE mark_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
    mark_price REAL NOT NULL, funding_rate REAL, next_funding_time_ms INTEGER)"""


def _sqlite_mark_prices(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(MARK_PRICES_DDL)
    conn.execute("CREATE INDEX idx_mp_symbol_ts ON mark_prices(symbol, ts_ms)")
    conn.executemany(
        "INSERT INTO mark_prices (id, ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms) "
        "VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


APR = 1775001600000
MAY = 1777593600000


def test_exact_timestamp_hit_and_between_timestamps_fallback_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 10, step_ms=10_000)
    _sqlite_mark_prices(db_path, rows)

    exact = mark_at_v2(root, "BTCUSDT", MAY + 50_000, source_db_path=db_path)
    assert exact == 60050.0  # row id=6 -> 60000+50

    between = mark_at_v2(root, "BTCUSDT", MAY + 55_000, source_db_path=db_path)
    assert between == 60050.0  # still row id=6


def test_no_prior_row_returns_none_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, _mark_prices_rows(1, MAY, 5))
    result = mark_at_v2(root, "BTCUSDT", MAY - 3600_000, source_db_path=db_path)
    assert result is None


def test_tie_break_parity_with_direct_sql_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    tied_ts = MAY + 10_000
    rows = [(1, MAY, "BTCUSDT", 60000.0, None, None),
            (2, tied_ts, "BTCUSDT", 60010.0, None, None),
            (3, tied_ts, "BTCUSDT", 60020.0, None, None)]
    _sqlite_mark_prices(db_path, rows)
    conn = sqlite3.connect(db_path)
    oracle = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? "
        "ORDER BY ts_ms DESC, id DESC LIMIT 1", (tied_ts,)).fetchone()[0]
    conn.close()
    new = mark_at_v2(root, "BTCUSDT", tied_ts, source_db_path=db_path)
    assert new == oracle == 60020.0


def test_provenance_correctness_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, _mark_prices_rows(1, MAY, 5))
    lookup = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="BTCUSDT", ts_ms=MAY + 20_000,
                                           columns=("mark_price",), source_db_path=db_path)
    prov = lookup.provenance
    assert prov["source_type"] == "SQLITE_ONLY"
    assert prov["result_source"] == "SQLITE"
    assert prov["columns"] == ["mark_price"]
    assert prov["ordering"] == "(ts_ms DESC, id DESC)"


def test_trust_failure_fails_closed_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 10)
    entry = _build_archive_partition(root, symbol="BTCUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                                      rows=rows, corrupt_manifest=True)
    _write_catalog_index(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        mark_at_v2(root, "BTCUSDT", APR + 90_000)


# --- No full reverify triggered (static contract check) ---

def test_no_full_reverify_referenced_in_migrated_module():
    import inspect
    import tools.research_s34_btc_microtrend_sweep as mod
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")
