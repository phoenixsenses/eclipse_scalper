"""As-of-lookup consumer migration parity (BATCH-STORAGE-ROTATION-
RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V1).

`tools/research_s34_consensus_composite.py` is the pilot consumer for
`ami.storage.research_reader.lookup_latest_at_or_before`. Three of its
direct-SQL `ORDER BY ts_ms DESC LIMIT 1` helpers were migrated to
reader-backed counterparts (old ones kept as parity oracles, unchanged,
no longer called by `build()`):
  * `mbps` (mark_prices, 2 lookups/call)     -> `mbps_v2`
  * `bimb` (book_ticker, staleness-checked)  -> `bimb_v2`
  * `nf`   (mark_prices, IS NOT NULL filter) -> `nf_v2`

Two helpers stay on direct SQL, explicitly out of scope for this gate:
  * `rv5`  -- queries `vol_state`, not in the research-reader's 3-table
    allowlist.
  * `ofir` -- a bounded-RANGE aggregate (SUM/COUNT over a window), not
    an `ORDER BY ts_ms DESC LIMIT 1` point lookup; belongs to the
    range-read helper exercised by the two prior consumer-migration
    gates, not this gate's point-lookup primitive.

Windows below were confirmed non-empty via read-only diagnostic
queries against the real database before being hardcoded. Skips (does
not fail) if the real source database is not present.
"""
from __future__ import annotations

import os

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
from tools.research_s34_consensus_composite import bimb, bimb_v2, mbps, mbps_v2, nf, nf_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_SOURCE_DB), reason="real source database not present in this checkout")


def _root():
    root, _source = PR.resolve_production_root()
    return root


def _old_mbps(symbol, ts, lb):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return mbps(conn, symbol, ts, lb)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_bimb(ts):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return bimb(conn, ts)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_nf(ts):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return nf(conn, ts)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# --- mbps: archive-only window, fully inside mark_prices/ETHUSDT/2026-05 ---

MBPS_ARCHIVE_TS = 1778600000000
MBPS_ARCHIVE_LB = 3600_000


def test_mbps_v2_parity_archive_only_window():
    old = _old_mbps("ETHUSDT", MBPS_ARCHIVE_TS, MBPS_ARCHIVE_LB)
    new = mbps_v2(_root(), "ETHUSDT", MBPS_ARCHIVE_TS, MBPS_ARCHIVE_LB, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


# --- mbps: SQLite-only, recent live window ---

def test_mbps_v2_parity_sqlite_only_window():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old = _old_mbps("ETHUSDT", max_ts, 3600_000)
    new = mbps_v2(_root(), "ETHUSDT", max_ts, 3600_000, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


# --- mbps: hybrid window -- ts-lb falls in the archive, ts falls in live SQLite ---

MBPS_HYBRID_TS = 1780272000000 + 3600_000
MBPS_HYBRID_LB = 2 * 3600_000


def test_mbps_v2_parity_hybrid_window():
    plan_a = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT",
                           start_ms=MBPS_HYBRID_TS - MBPS_HYBRID_LB, end_ms=MBPS_HYBRID_TS - MBPS_HYBRID_LB + 1)
    plan_b = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT",
                           start_ms=MBPS_HYBRID_TS, end_ms=MBPS_HYBRID_TS + 1)
    assert plan_a.mode == "ARCHIVE_ONLY"  # confirms ts-lb really is archive-side
    assert plan_b.mode == "SQLITE_ONLY"   # confirms ts really is live-side
    old = _old_mbps("ETHUSDT", MBPS_HYBRID_TS, MBPS_HYBRID_LB)
    new = mbps_v2(_root(), "ETHUSDT", MBPS_HYBRID_TS, MBPS_HYBRID_LB, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)


# --- bimb: book_ticker/ETHUSDT has no archive partition (only SOLUSDT is
# archived) -- always SQLITE_ONLY in real usage; documented, not silently
# assumed hybrid-capable. ---

def test_bimb_v2_parity_sqlite_only():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old = _old_bimb(max_ts)
    new = bimb_v2(_root(), max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == pytest.approx(old, rel=1e-9)
    plan = RR.plan_read(_root(), table="book_ticker", symbol="ETHUSDT", start_ms=max_ts - 1000, end_ms=max_ts + 1)
    assert plan.mode == "SQLITE_ONLY"


# --- nf: mark_prices IS NOT NULL filter, both archive-side and live-side ---

def test_nf_v2_parity_archive_side():
    old = _old_nf(MBPS_ARCHIVE_TS)
    new = nf_v2(_root(), MBPS_ARCHIVE_TS, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == old


def test_nf_v2_parity_live_side():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old = _old_nf(max_ts)
    new = nf_v2(_root(), max_ts, source_db_path=REAL_SOURCE_DB)
    assert old is not None
    assert new == old


# --- Provenance correctness on the migrated path ---

def test_mbps_v2_provenance_reports_archive_identity():
    result = RR.lookup_latest_at_or_before(_root(), table="mark_prices", symbol="ETHUSDT",
                                            ts_ms=MBPS_ARCHIVE_TS, columns=("mark_price",),
                                            source_db_path=REAL_SOURCE_DB)
    prov = result.provenance
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["result_source"] == "ARCHIVE"
    assert prov["query_ts_ms"] == MBPS_ARCHIVE_TS
    assert prov["inclusive"] is True
    assert prov["columns"] == ["mark_price"]
    assert prov["archive_identity"]
    assert prov["manifest_sha256"]
    assert prov["result_ts_ms"] is not None
    assert prov["ordering"] == "(ts_ms DESC, id DESC)"


def test_nf_v2_provenance_reports_filter():
    result = RR.lookup_latest_at_or_before(_root(), table="mark_prices", symbol="ETHUSDT",
                                            ts_ms=MBPS_ARCHIVE_TS, columns=("next_funding_time_ms",),
                                            filters=(("next_funding_time_ms", "!=", None),),
                                            source_db_path=REAL_SOURCE_DB)
    assert result.found
    assert result.provenance["filters"] == [("next_funding_time_ms", "!=", None)]


# --- Repeated-run determinism ---

def test_mbps_v2_repeated_run_determinism():
    root = _root()
    results = [mbps_v2(root, "ETHUSDT", MBPS_ARCHIVE_TS, MBPS_ARCHIVE_LB, source_db_path=REAL_SOURCE_DB)
               for _ in range(3)]
    assert results[0] == results[1] == results[2]


# --- Trust failure fail-closed (synthetic corrupt archive, same shape as prior gates) ---

def _build_corrupt_mark_prices_root(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = str(tmp_path / "corrupt_root")
    rel_parts = ("table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                 "symbol=ETHUSDT", "year=2026", "month=05", "version=v1")
    final_dir = os.path.join(root, *rel_parts)
    os.makedirs(final_dir, exist_ok=True)

    schema = pa.schema([pa.field("id", pa.int64()), pa.field("ts_ms", pa.int64()), pa.field("symbol", pa.string()),
                        pa.field("mark_price", pa.float64()), pa.field("funding_rate", pa.float64()),
                        pa.field("next_funding_time_ms", pa.int64())])
    arrays = [pa.array([1], type=pa.int64()), pa.array([MBPS_ARCHIVE_TS], type=pa.int64()),
              pa.array(["ETHUSDT"], type=pa.string()), pa.array([2000.0], type=pa.float64()),
              pa.array([0.0001], type=pa.float64()), pa.array([MBPS_ARCHIVE_TS + 1000], type=pa.int64())]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema),
                   os.path.join(final_dir, "part-00000.parquet"), compression="zstd")

    manifest = {"source_table": "mark_prices", "symbol": "ETHUSDT", "venue": "BINANCE_USDM_PERP",
                "market_segment": "PERPETUAL_FUTURES", "row_count": 1, "shards": None,
                "parquet_path": os.path.join(*rel_parts, "part-00000.parquet"),
                "ordered_scientific_content_hash": "irrelevant-for-trust-check", "partition_id": "corrupt-test",
                "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
                "source_watermark_field": "id", "source_watermark_value": 1}
    manifest_path = os.path.join(final_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("corrupt-test\n")

    catalog_entry = {
        "archive_identity": "corrupt-test", "archive_relative_path": os.path.join(*rel_parts),
        "source_table": "mark_prices", "symbol": "ETHUSDT", "venue": "BINANCE_USDM_PERP",
        "market_segment": "PERPETUAL_FUTURES",
        "partition_start_ms": 1777593600000, "partition_end_ms": 1780272000000,
        "row_count": 1, "manifest_sha256": "0" * 64,  # deliberately WRONG
        "authorization_receipt_sha256": None,
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
        "scientific_content_hash": "irrelevant-for-trust-check",
    }
    index = {"catalog_contract_version": "v1", "entry_count": 1, "entries": [catalog_entry]}
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return root


def test_mbps_v2_fails_closed_on_corrupt_archive_trust(tmp_path):
    corrupt_root = _build_corrupt_mark_prices_root(tmp_path)
    with pytest.raises(RR.ArchiveTrustError):
        mbps_v2(corrupt_root, "ETHUSDT", MBPS_ARCHIVE_TS, MBPS_ARCHIVE_LB)


# --- Source mutation 0 ---

def test_mbps_v2_leaves_real_catalog_and_manifest_unchanged():
    root = _root()
    idx_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    idx_before = os.path.getmtime(idx_path)
    mbps_v2(root, "ETHUSDT", MBPS_ARCHIVE_TS, MBPS_ARCHIVE_LB, source_db_path=REAL_SOURCE_DB)
    idx_after = os.path.getmtime(idx_path)
    assert idx_before == idx_after
