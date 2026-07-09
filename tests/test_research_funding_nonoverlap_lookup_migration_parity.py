"""ASOF Batch 1 consumer migration parity #1 (BATCH-STORAGE-ROTATION-
RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V4).

`tools/research_funding_nonoverlap.py`'s single `ORDER BY ts_ms DESC
LIMIT 1` helper, `funding_at` (latest non-null `funding_rate` at-or-
before a timestamp, `mark_prices`), is migrated to `funding_at_v2`
(reader-backed, `funding_rate IS NOT NULL` expressed as a helper
filter). `funding_at` is kept unchanged as the parity oracle.

Its `load_mark_index` bulk read (range-shaped, imported from another
module) is untouched -- out of scope for this point-lookup gate.

ETHUSDT resolves through the real mark_prices/ETHUSDT/2026-05 archive
for historical timestamps (real archive smoke below). Skips if the
real source database is not present.
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
from tools.research_funding_nonoverlap import funding_at, funding_at_v2

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
        return funding_at(conn, sym, ts)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- real archive-only parity (ETHUSDT inside mark_prices/2026-05) ---

ARCHIVE_TS = 1778600000000


def test_funding_at_v2_parity_archive_only():
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=ARCHIVE_TS, end_ms=ARCHIVE_TS + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old = _old("ETHUSDT", ARCHIVE_TS)
    new = funding_at_v2(_root(), "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-12)


# --- real SQLite-only parity (ETHUSDT pre-archive) ---

SQLITE_TS = 1772500000000


def test_funding_at_v2_parity_sqlite_only():
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=SQLITE_TS, end_ms=SQLITE_TS + 1)
    assert plan.mode == "SQLITE_ONLY"
    old = _old("ETHUSDT", SQLITE_TS)
    new = funding_at_v2(_root(), "ETHUSDT", SQLITE_TS, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-12)


def test_funding_at_v2_parity_live_max():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old = _old("ETHUSDT", max_ts)
    new = funding_at_v2(_root(), "ETHUSDT", max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-12)


def test_funding_at_v2_provenance_reports_filter_and_archive_identity():
    result = RR.lookup_latest_at_or_before(_root(), table="mark_prices", symbol="ETHUSDT", ts_ms=ARCHIVE_TS,
                                           columns=("funding_rate",), filters=(("funding_rate", "!=", None),),
                                           source_db_path=REAL_SOURCE_DB)
    prov = result.provenance
    assert result.found
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["filters"] == [("funding_rate", "!=", None)]
    assert prov["columns"] == ["funding_rate"]
    assert prov["archive_identity"]
    assert prov["ordering"] == "(ts_ms DESC, id DESC)"


def test_funding_at_v2_repeated_run_determinism():
    root = _root()
    vals = [funding_at_v2(root, "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_funding_at_v2_source_mutation_zero():
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    funding_at_v2(root, "ETHUSDT", ARCHIVE_TS, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_no_full_reverify_referenced():
    import inspect
    import tools.research_funding_nonoverlap as mod
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


# ---------------------------------------------------------------------------
# Synthetic fixtures (exact-hit / between / empty / tie-break / filter /
# hybrid-fallback / trust-failure) -- self-contained, same style as prior gates.
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


def _rows(start_id, start_ms, n, symbol="ETHUSDT", step_ms=8 * 3600_000, funding=None):
    return [(start_id + i, start_ms + i * step_ms, symbol, 3000.0 + i,
             funding[i] if funding else 0.0001 * (i + 1), None) for i in range(n)]


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
    _sqlite(db, _rows(1, MAY, 5, funding=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005]))
    exact = funding_at_v2(root, "ETHUSDT", MAY + 2 * 8 * 3600_000, source_db_path=db)
    assert exact == pytest.approx(0.0003)
    between = funding_at_v2(root, "ETHUSDT", MAY + 2 * 8 * 3600_000 + 60_000, source_db_path=db)
    assert between == pytest.approx(0.0003)


def test_empty_no_prior_row_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, MAY, 5))
    assert funding_at_v2(root, "ETHUSDT", MAY - 3600_000, source_db_path=db) is None


def test_filter_skips_null_funding_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    # latest two rows have NULL funding -> must walk back to row 3
    _sqlite(db, _rows(1, MAY, 5, funding=[0.0001, 0.0002, 0.0003, None, None]))
    val = funding_at_v2(root, "ETHUSDT", MAY + 4 * 8 * 3600_000, source_db_path=db)
    assert val == pytest.approx(0.0003)


def test_tie_break_parity_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _catalog(root, [])
    db = str(tmp_path / "s.sqlite")
    tied = MAY + 8 * 3600_000
    _sqlite(db, [(1, MAY, "ETHUSDT", 3000.0, 0.0001, None),
                 (2, tied, "ETHUSDT", 3001.0, 0.0002, None),
                 (3, tied, "ETHUSDT", 3002.0, 0.0003, None)])
    c = sqlite3.connect(db)
    oracle = c.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND "
                       "funding_rate IS NOT NULL ORDER BY ts_ms DESC, id DESC LIMIT 1", (tied,)).fetchone()[0]
    c.close()
    assert funding_at_v2(root, "ETHUSDT", tied, source_db_path=db) == pytest.approx(oracle)


def test_hybrid_fallback_synthetic(tmp_path):
    """Archive partition's own data starts later than its declared range;
    a query inside the declared-but-empty early sub-range falls back to
    an earlier SQLite row -> HYBRID."""
    root = str(tmp_path / "r")
    os.makedirs(root)
    archive_rows = _rows(11, APR + 20 * 8 * 3600_000, 5)  # archive data starts well after APR
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY, rows=archive_rows)
    _catalog(root, [entry])
    db = str(tmp_path / "s.sqlite")
    _sqlite(db, _rows(1, APR - 8 * 3600_000, 3))  # earlier live rows
    q = APR + 5 * 8 * 3600_000  # inside archive's declared range, before its real first row
    lookup = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=q,
                                           columns=("funding_rate",), filters=(("funding_rate", "!=", None),),
                                           source_db_path=db)
    assert lookup.found
    assert lookup.provenance["source_type"] == "HYBRID"
    assert lookup.provenance["result_source"] == "SQLITE"


def test_trust_failure_fails_closed_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    entry = _build_archive(root, symbol="ETHUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                            rows=_rows(1, APR, 5), corrupt_manifest=True)
    _catalog(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        funding_at_v2(root, "ETHUSDT", APR + 8 * 3600_000)
