"""ASOF Batch 1 consumer migration parity #2 (BATCH-STORAGE-ROTATION-
RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V4).

`tools/s34_mechanism_taxonomy.py`'s `mbps` helper (two `mark_prices`
`ORDER BY ts_ms DESC LIMIT 1` lookups per call -- price at-or-before
`ts-lb` and at-or-before `ts`, returning the bps return between them)
is migrated to `mbps_v2` (reader-backed). `mbps` is kept unchanged as
the parity oracle.

The script also writes gate columns into its own `mechanism_store.sqlite`
feature store -- that write path is entirely untouched by this
migration (only the two `mark_prices` READ lookups against the
read-only `microstructure.db` connection are migrated). No `main()` is
ever run by these tests.

ETHUSDT resolves through the real mark_prices/ETHUSDT/2026-05 archive;
the two-endpoint call naturally straddles the archive/live boundary
(real HYBRID-shaped coverage: one endpoint ARCHIVE_ONLY, the other
SQLITE_ONLY, both parity-checked). Skips if the real DB is absent.
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
from tools.s34_mechanism_taxonomy import mbps, mbps_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_SOURCE_DB), reason="real source database not present in this checkout")


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _old(sym, ts, lb):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return mbps(conn, sym, ts, lb)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


ARCHIVE_TS = 1778600000000
LB_4H = 4 * 3600_000


# --- real archive-only parity (both endpoints inside mark_prices/2026-05) ---

def test_mbps_v2_parity_archive_only_both_endpoints():
    for ts in (ARCHIVE_TS, ARCHIVE_TS - LB_4H):
        plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ts, end_ms=ts + 1)
        assert plan.mode == "ARCHIVE_ONLY"
    old = _old("ETHUSDT", ARCHIVE_TS, LB_4H)
    new = mbps_v2(_root(), "ETHUSDT", ARCHIVE_TS, LB_4H, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


# --- real straddle: ts-lb archive-side, ts live-side (both endpoints proven) ---

BND = 1780272000000
STRADDLE_TS = BND + 3600_000  # entry 1h after boundary (live)
STRADDLE_LB = 4 * 3600_000    # ts-lb = boundary - 3h (archive)


def test_mbps_v2_parity_straddle_boundary():
    plan_a = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT",
                          start_ms=STRADDLE_TS - STRADDLE_LB, end_ms=STRADDLE_TS - STRADDLE_LB + 1)
    plan_b = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT",
                          start_ms=STRADDLE_TS, end_ms=STRADDLE_TS + 1)
    assert plan_a.mode == "ARCHIVE_ONLY"
    assert plan_b.mode == "SQLITE_ONLY"
    old = _old("ETHUSDT", STRADDLE_TS, STRADDLE_LB)
    new = mbps_v2(_root(), "ETHUSDT", STRADDLE_TS, STRADDLE_LB, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


# --- real SQLite-only parity (BTCUSDT: no mark_prices archive at all) ---

def test_mbps_v2_parity_btcusdt_sqlite_only():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='BTCUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    plan = RR.plan_read(_root(), table="mark_prices", symbol="BTCUSDT", start_ms=max_ts - 1000, end_ms=max_ts + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old("BTCUSDT", max_ts, LB_4H)
    new = mbps_v2(_root(), "BTCUSDT", max_ts, LB_4H, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


def test_mbps_v2_repeated_run_determinism():
    root = _root()
    vals = [mbps_v2(root, "ETHUSDT", ARCHIVE_TS, LB_4H, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_mbps_v2_provenance_archive_identity():
    result = RR.lookup_latest_at_or_before(_root(), table="mark_prices", symbol="ETHUSDT", ts_ms=ARCHIVE_TS,
                                           columns=("mark_price",), source_db_path=REAL_SOURCE_DB)
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["columns"] == ["mark_price"]
    assert prov["archive_identity"]
    assert prov["ordering"] == "(ts_ms DESC, id DESC)"


def test_mbps_v2_source_mutation_zero():
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    mbps_v2(root, "ETHUSDT", ARCHIVE_TS, LB_4H, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_no_full_reverify_referenced():
    import inspect
    import tools.s34_mechanism_taxonomy as mod
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema
    schema = build_pyarrow_schema(SPEC)
    arrays = [pa.array([r[i] for r in rows], type=schema.field(c).type)
              for i, c in enumerate(SPEC.preserved_columns)]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path, compression="zstd")


def _rows(start_id, start_ms, n, symbol="ETHUSDT", step_ms=60_000):
    return [(start_id + i, start_ms + i * step_ms, symbol, 3000.0 + i, None, None) for i in range(n)]


def _build_archive(root, *, symbol, partition_start_ms, partition_end_ms, rows, corrupt_manifest=False):
    rel = os.path.join("table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                        f"symbol={symbol}", "year=2026", "month=04", "version=v1")
    fd = os.path.join(root, rel)
    os.makedirs(fd, exist_ok=True)
    pp = os.path.join(fd, "part-00000.parquet")
    _write_parquet(pp, rows)
    manifest = {"source_table": "mark_prices", "symbol": symbol, "venue": SPEC.venue,
                "market_segment": SPEC.market_segment, "row_count": len(rows), "shards": None,
                "ordered_scientific_content_hash": "0" * 64 if corrupt_manifest else "irrelevant",
                "partition_id": f"t-{symbol}", "parquet_path": os.path.join(rel, "part-00000.parquet"),
                "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
                "source_watermark_field": "id", "source_watermark_value": rows[-1][0],
                "parquet_sha256": _sha256_file(pp)}
    mp = os.path.join(fd, "manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(fd, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("ok\n")
    entry = {"archive_identity": f"t-{symbol}", "archive_relative_path": rel, "source_table": "mark_prices",
             "symbol": symbol, "venue": SPEC.venue, "market_segment": SPEC.market_segment,
             "partition_start_ms": partition_start_ms, "partition_end_ms": partition_end_ms, "row_count": len(rows),
             "manifest_sha256": "f" * 64 if corrupt_manifest else _sha256_file(mp),
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": manifest["ordered_scientific_content_hash"]}
    with open(os.path.join(fd, "catalog_entry.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f)
    return entry


def _catalog(root, entries):
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}, f)


DDL = ("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, mark_price REAL, "
       "funding_rate REAL, next_funding_time_ms INTEGER)")


def _sqlite(path, rows):
    c = sqlite3.connect(path)
    c.execute(DDL)
    c.execute("CREATE INDEX ix ON mark_prices(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


APR = 1775001600000
MAY = 1777593600000


def test_exact_hit_and_between_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 10, step_ms=60_000))
    # mbps between two exact rows: ts=MAY+5min (row6=3005), ts-lb=MAY+2min (row3=3002)
    val = mbps_v2(root, "ETHUSDT", MAY + 5 * 60_000, 3 * 60_000, source_db_path=db)
    expected = (3005.0 - 3002.0) / 3002.0 * 1e4
    assert val == pytest.approx(expected)


def test_empty_endpoint_returns_none_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    # ts-lb before any row -> a endpoint missing -> None
    assert mbps_v2(root, "ETHUSDT", MAY + 60_000, 3600_000, source_db_path=db) is None


def test_tie_break_parity_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    tied = MAY + 60_000
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, None, None),
                 (2, tied, "ETHUSDT", 3001.0, None, None),
                 (3, tied, "ETHUSDT", 3002.0, None, None)])
    c = sqlite3.connect(db)
    b = c.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? "
                  "ORDER BY ts_ms DESC, id DESC LIMIT 1", (tied,)).fetchone()[0]
    a = c.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? "
                  "ORDER BY ts_ms DESC, id DESC LIMIT 1", (MAY,)).fetchone()[0]
    c.close()
    expected = (b - a) / a * 1e4
    assert mbps_v2(root, "ETHUSDT", tied, 60_000, source_db_path=db) == pytest.approx(expected)


def test_trust_failure_fails_closed_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                            rows=_rows(1, APR, 10), corrupt_manifest=True)
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        mbps_v2(root, "ETHUSDT", APR + 5 * 60_000, 60_000)
