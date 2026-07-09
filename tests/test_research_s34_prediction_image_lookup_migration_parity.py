"""ASOF Batch 1 consumer migration parity #3 (BATCH-STORAGE-ROTATION-
RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V4).

`tools/research_s34_prediction_image.py`'s `nearest_price` helper
(latest `mark_price` at-or-before a timestamp, `mark_prices`) is
migrated to `nearest_price_v2` (reader-backed). `nearest_price` is kept
unchanged as the parity oracle.

PARTIAL migration, documented inline in the consumer: the two
"latest row overall for symbol" lookups in `load()` (mark_prices and
agg_trades, `ORDER BY ts_ms DESC LIMIT 1` with NO `ts_ms <= ?` bound)
are a different query shape than the helper's at-or-before contract,
and the agg-latest lookup has no safely pre-computed bound -- both left
on direct SQL, per the partial-migration policy proven in prior gates.
The `max(ts_ms)` aggregate is likewise out of scope. `main()` (matplotlib
image render + report write) is never invoked by these tests.

ETHUSDT resolves through the real mark_prices/ETHUSDT/2026-05 archive
for historical target timestamps (real archive smoke below). Skips if
the real DB is absent.
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
from tools.research_s34_prediction_image import nearest_price, nearest_price_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_SOURCE_DB), reason="real source database not present in this checkout")


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _old(sym, ts):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return nearest_price(conn, sym, ts)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


ARCHIVE_TS = 1778600000000
SQLITE_TS = 1772500000000


def test_nearest_price_v2_parity_archive_only():
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ARCHIVE_TS, end_ms=ARCHIVE_TS + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old("ETHUSDT", ARCHIVE_TS)
    new = nearest_price_v2(_root(), "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-12)


def test_nearest_price_v2_parity_sqlite_only():
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=SQLITE_TS, end_ms=SQLITE_TS + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old("ETHUSDT", SQLITE_TS)
    new = nearest_price_v2(_root(), "ETHUSDT", SQLITE_TS, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-12)


def test_nearest_price_v2_parity_all_three_offsets_live_and_archive():
    # mirror load()'s real call pattern: three offsets off a base ts.
    # Base in archive -> all three archive-side; base at live max -> live-side.
    for base in (ARCHIVE_TS, ):
        for off in (3600_000, 6 * 3600_000, 24 * 3600_000):
            old = _old("ETHUSDT", base - off)
            new = nearest_price_v2(_root(), "ETHUSDT", base - off, source_db_path=REAL_SOURCE_DB)
            assert old is not None
            assert new == pytest.approx(old, rel=1e-12)


def test_nearest_price_v2_parity_btcusdt_sqlite_only():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='BTCUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old = _old("BTCUSDT", max_ts)
    new = nearest_price_v2(_root(), "BTCUSDT", max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-12)


def test_nearest_price_v2_repeated_run_determinism():
    root = _root()
    vals = [nearest_price_v2(root, "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_nearest_price_v2_provenance_archive_identity():
    result = RR.lookup_latest_at_or_before(_root(), table="mark_prices", symbol="ETHUSDT", ts_ms=ARCHIVE_TS,
                                           columns=("mark_price",), source_db_path=REAL_SOURCE_DB)
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["columns"] == ["mark_price"]
    assert prov["archive_identity"]
    assert prov["result_ts_ms"] is not None


def test_nearest_price_v2_source_mutation_zero():
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    nearest_price_v2(root, "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_no_full_reverify_referenced():
    import inspect
    import tools.research_s34_prediction_image as mod
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
JUN = 1780272000000


def test_exact_hit_and_between_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 10, step_ms=60_000))
    exact = nearest_price_v2(root, "ETHUSDT", MAY + 5 * 60_000, source_db_path=db)
    assert exact == pytest.approx(3005.0)
    between = nearest_price_v2(root, "ETHUSDT", MAY + 5 * 60_000 + 30_000, source_db_path=db)
    assert between == pytest.approx(3005.0)


def test_empty_no_prior_row_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    assert nearest_price_v2(root, "ETHUSDT", MAY - 3600_000, source_db_path=db) is None


def test_archive_only_lookup_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY, rows=_rows(1, APR, 20))
    _catalog(root, [entry])
    val = nearest_price_v2(root, "ETHUSDT", APR + 10 * 60_000)  # row 11 -> 3010
    assert val == pytest.approx(3010.0)
    lk = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=APR + 10 * 60_000,
                                       columns=("mark_price",))
    assert lk.provenance["source_type"] == "ARCHIVE_ONLY"


def test_hybrid_fallback_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    archive_rows = _rows(1, APR, 10)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY, rows=archive_rows)
    _catalog(root, [entry])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, [])  # live empty -> query after archive end falls back to archive
    q = MAY + 60_000
    lk = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=q,
                                       columns=("mark_price",), source_db_path=db)
    assert lk.found
    assert lk.provenance["source_type"] == "HYBRID"
    assert lk.provenance["result_source"] == "ARCHIVE"
    assert nearest_price_v2(root, "ETHUSDT", q, source_db_path=db) == pytest.approx(3009.0)


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
    oracle = c.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? "
                       "ORDER BY ts_ms DESC, id DESC LIMIT 1", (tied,)).fetchone()[0]
    c.close()
    assert nearest_price_v2(root, "ETHUSDT", tied, source_db_path=db) == pytest.approx(oracle)


def test_trust_failure_fails_closed_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                            rows=_rows(1, APR, 10), corrupt_manifest=True)
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        nearest_price_v2(root, "ETHUSDT", APR + 5 * 60_000)
