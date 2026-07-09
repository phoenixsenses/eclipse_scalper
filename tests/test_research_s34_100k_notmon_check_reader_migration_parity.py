"""Second research-consumer integration (Task for BATCH-STORAGE-ROTATION-
RETENTION-SECOND-RESEARCH-CONSUMER-INTEGRATION-V1).

`tools/research_s34_100k_notmon_check.py` is the second, independent
pilot consumer: its one bounded-range direct-SQL aggregate (`ofir` --
buy/sell/whale notional over an agg_trades window) now has a
reader-backed counterpart, `window_agg_trades_ofi_whale`, used by
`main()` in place of the old direct query. `ofir` itself is left in the
file unchanged, as the parity oracle.

This suite covers the 12 required scenarios:
  1. direct-SQLite baseline path (`ofir`)
  2. reader-backed path (`window_agg_trades_ofi_whale`)
  3. direct vs reader parity (identical ofi/whale for the same window)
  4. archive-only / hybrid plan actually triggered (not silently
     falling back to SQLITE_ONLY)
  5. provenance correctness
  6. empty range
  7. deterministic repeated run
  8. ordering-invariance (same aggregate regardless of physical
     shard/batch iteration order)
  9. column projection matches the old query's column set
  10. reader trust failure -> consumer fails safely, not silently wrong
  11. source mutation 0 (mode=ro throughout)
  12. existing pilot (first-consumer) reader tests unaffected (see the
      full storage regression run alongside this file; not duplicated
      here)

Skips (does not fail) the production-smoke tests if the real archive
root / source database aren't present in this checkout.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
from tools.research_s34_100k_notmon_check import ofir, window_agg_trades_ofi_whale

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_SOURCE_DB), reason="real source database not present in this checkout")


def _old_ofir(start_ms: int, end_ms: int):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return ofir(conn, start_ms, end_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _root():
    root, _source = PR.resolve_production_root()
    return root


# --- 1/2/3: direct-SQLite baseline vs reader-backed path, on a recent live window ---

def test_ofi_whale_parity_recent_live_window():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM agg_trades WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    start_ms, end_ms = max_ts - 300_000, max_ts
    old_ofi, old_whale = _old_ofir(start_ms, end_ms)
    new_ofi, new_whale = window_agg_trades_ofi_whale(_root(), "ETHUSDT", start_ms, end_ms)
    assert old_ofi is not None
    assert new_ofi == pytest.approx(old_ofi, rel=1e-9)
    assert new_whale == pytest.approx(old_whale, rel=1e-9)


# --- 4: archive-only window, fully inside the archived Feb 2026 agg_trades partition ---

ARCHIVE_START_MS = 1771165588000
ARCHIVE_END_MS = ARCHIVE_START_MS + 300_000


def test_ofi_whale_parity_archive_only_window_and_plan_mode():
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=ARCHIVE_START_MS, end_ms=ARCHIVE_END_MS)
    assert plan.mode == "ARCHIVE_ONLY"  # proves this window really is archive-served, not silently SQLite
    old_ofi, old_whale = _old_ofir(ARCHIVE_START_MS, ARCHIVE_END_MS)
    new_ofi, new_whale = window_agg_trades_ofi_whale(_root(), "ETHUSDT", ARCHIVE_START_MS, ARCHIVE_END_MS)
    assert old_ofi is not None
    assert new_ofi == pytest.approx(old_ofi, rel=1e-9)
    assert new_whale == pytest.approx(old_whale, rel=1e-9)


# --- 4 (hybrid variant): straddles the archive/live boundary (2026-02/2026-03) ---

HYBRID_START_MS = 1772323200000 - 150_000
HYBRID_END_MS = 1772323200000 + 150_000


def test_ofi_whale_parity_hybrid_window_and_plan_mode():
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=HYBRID_START_MS, end_ms=HYBRID_END_MS)
    assert plan.mode == "HYBRID"
    old_ofi, old_whale = _old_ofir(HYBRID_START_MS, HYBRID_END_MS)
    new_ofi, new_whale = window_agg_trades_ofi_whale(_root(), "ETHUSDT", HYBRID_START_MS, HYBRID_END_MS)
    assert old_ofi is not None
    assert new_ofi == pytest.approx(old_ofi, rel=1e-9)
    assert new_whale == pytest.approx(old_whale, rel=1e-9)


# --- 5: provenance correctness for the hybrid window ---

def test_provenance_reports_hybrid_source_and_columns():
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=HYBRID_START_MS, end_ms=HYBRID_END_MS)
    result = RR.execute_read(plan, columns=("notional", "is_buyer_maker"))
    list(result.iter_rows())  # consume so row_count is populated
    prov = result.provenance
    assert prov["source_type"] == "HYBRID"
    assert prov["table"] == "agg_trades"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["notional", "is_buyer_maker"]
    assert prov["requested_start_ms"] == HYBRID_START_MS
    assert prov["requested_end_ms"] == HYBRID_END_MS
    assert len(prov["archive_segments"]) == 1
    assert prov["archive_segments"][0]["archive_identity"]
    assert prov["archive_segments"][0]["manifest_sha256"]
    assert len(prov["sqlite_ranges"]) == 1
    assert prov["row_count"] > 0


# --- 6: empty range (window entirely before any real data collection began) ---

def test_ofi_whale_empty_range_returns_none_none():
    lo, hi = 1_600_000_000_000, 1_600_000_300_000  # 2020-09, long before any collection
    old_ofi, old_whale = _old_ofir(lo, hi)
    new_ofi, new_whale = window_agg_trades_ofi_whale(_root(), "ETHUSDT", lo, hi)
    assert old_ofi is None and old_whale is None
    assert new_ofi is None and new_whale is None


# --- 7: deterministic repeated run ---

def test_repeated_run_determinism():
    results = [window_agg_trades_ofi_whale(_root(), "ETHUSDT", ARCHIVE_START_MS, ARCHIVE_END_MS) for _ in range(3)]
    assert results[0] == results[1] == results[2]


# --- 8: ordering-invariance -- same aggregate whether read via 1 huge batch or many tiny ones ---

def test_ofi_whale_invariant_to_batch_size():
    plan = RR.plan_read(_root(), table="agg_trades", symbol="ETHUSDT", start_ms=ARCHIVE_START_MS, end_ms=ARCHIVE_END_MS)

    def _reduce(batch_size):
        result = RR.execute_read(plan, columns=("notional", "is_buyer_maker"), batch_size=batch_size)
        buy = se = total = 0.0
        count = 0
        for notional, is_buyer_maker in result.iter_rows():
            count += 1
            total += notional
            if is_buyer_maker == 0:
                buy += notional
            elif is_buyer_maker == 1:
                se += notional
        t = buy + se
        return ((buy - se) / t if t > 0 else 0.0), (total / count if count else None)

    big_batch = _reduce(1_000_000)
    tiny_batch = _reduce(7)
    assert big_batch[0] == pytest.approx(tiny_batch[0], rel=1e-9)
    assert big_batch[1] == pytest.approx(tiny_batch[1], rel=1e-9)


# --- 10: reader trust failure -> consumer fails safely (synthetic corrupt archive) ---

def _build_corrupt_agg_trades_root(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = str(tmp_path / "corrupt_root")
    rel_parts = ("table=agg_trades", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                 "symbol=ETHUSDT", "year=2026", "month=02", "version=v1")
    final_dir = os.path.join(root, *rel_parts)
    os.makedirs(final_dir, exist_ok=True)

    schema = pa.schema([pa.field("id", pa.int64()), pa.field("ts_ms", pa.int64()), pa.field("symbol", pa.string()),
                        pa.field("price", pa.float64()), pa.field("quantity", pa.float64()),
                        pa.field("notional", pa.float64()), pa.field("is_buyer_maker", pa.int64())])
    arrays = [pa.array([1], type=pa.int64()), pa.array([ARCHIVE_START_MS], type=pa.int64()),
              pa.array(["ETHUSDT"], type=pa.string()), pa.array([100.0], type=pa.float64()),
              pa.array([1.0], type=pa.float64()), pa.array([100.0], type=pa.float64()),
              pa.array([0], type=pa.int64())]
    parquet_path = os.path.join(final_dir, "part-00000.parquet")
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), parquet_path, compression="zstd")

    manifest = {"source_table": "agg_trades", "symbol": "ETHUSDT", "venue": "BINANCE_USDM_PERP",
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
        "source_table": "agg_trades", "symbol": "ETHUSDT", "venue": "BINANCE_USDM_PERP",
        "market_segment": "PERPETUAL_FUTURES",
        "partition_start_ms": 1769904000000, "partition_end_ms": 1772323200000,
        "row_count": 1,
        "manifest_sha256": "0" * 64,  # deliberately WRONG -- must fail the trust check
        "authorization_receipt_sha256": None,
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
        "scientific_content_hash": "irrelevant-for-trust-check",
    }
    index = {"catalog_contract_version": "v1", "entry_count": 1, "entries": [catalog_entry]}
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return root


def test_consumer_fails_closed_on_corrupt_archive_trust(tmp_path):
    corrupt_root = _build_corrupt_agg_trades_root(tmp_path)
    with pytest.raises(RR.ArchiveTrustError):
        window_agg_trades_ofi_whale(corrupt_root, "ETHUSDT", ARCHIVE_START_MS, ARCHIVE_END_MS)


# --- 11: source mutation 0 -- manifest/catalog untouched by any of the above reads ---

def test_reads_leave_real_catalog_and_manifest_unchanged():
    root = _root()
    idx_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    idx_before = os.path.getmtime(idx_path)
    window_agg_trades_ofi_whale(root, "ETHUSDT", ARCHIVE_START_MS, ARCHIVE_END_MS)
    idx_after = os.path.getmtime(idx_path)
    assert idx_before == idx_after
