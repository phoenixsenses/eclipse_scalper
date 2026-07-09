"""Tests for `ami.storage.research_reader.lookup_latest_at_or_before` --
the `ORDER BY ts_ms DESC LIMIT 1` point-lookup primitive added in
BATCH-STORAGE-ROTATION-RETENTION-POINT-LOOKUP-HELPER-FOR-ORDER-BY-TS-
DESC-LIMIT-1-V1. Small synthetic fixtures only (mirrors the fixture
style of test_ami_storage_research_reader.py); real production smoke
lives in test_ami_storage_research_reader_lookup_production_parity.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest

from ami.storage import research_reader as RR
from ami.storage.registry import get_table_spec

SPEC = get_table_spec("mark_prices")
BT_SPEC = get_table_spec("book_ticker")


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


def _mark_prices_rows(start_id, start_ms, n, symbol="ETHUSDT", step_ms=60_000, funding=None):
    return [(start_id + i, start_ms + i * step_ms, symbol, 3000.0 + i,
             funding[i] if funding else None, None) for i in range(n)]


def _book_ticker_rows(start_id, start_ms, n, symbol="SOLUSDT", step_ms=1000):
    return [(start_id + i, start_ms + i * step_ms, symbol, 100.0, 1.0, 100.5, 1.0, 100.25, 0.005, 0.1, 1000.0)
            for i in range(n)]


def _build_archive_partition(root, *, table, spec, symbol, utc_year, utc_month,
                              partition_start_ms, partition_end_ms, rows, shard_sizes=None,
                              no_receipt=False, corrupt_manifest=False, missing_shard=False,
                              corrupt_receipt=False):
    rel = os.path.join(f"table={table}", f"venue={spec.venue}", f"market_segment={spec.market_segment}",
                        f"symbol={symbol}", f"year={utc_year:04d}", f"month={utc_month:02d}", "version=v1")
    final_dir = os.path.join(root, rel)
    os.makedirs(final_dir, exist_ok=True)

    if shard_sizes is None:
        shard_sizes = [len(rows)]
    shards = []
    idx = 0
    shard_index = 0
    for size in shard_sizes:
        chunk = rows[idx:idx + size]
        idx += size
        if not chunk:
            continue
        shard_file = f"part-{shard_index:05d}.parquet"
        shard_path = os.path.join(final_dir, shard_file)
        _write_parquet(shard_path, chunk, spec)
        shards.append({
            "shard_index": shard_index, "shard_file": shard_file, "row_count": len(chunk),
            "min_id": chunk[0][0], "max_id": chunk[-1][0],
            "min_ts": chunk[0][1], "max_ts": chunk[-1][1],
            "byte_size": os.path.getsize(shard_path), "sha256": _sha256_file(shard_path),
        })
        shard_index += 1

    if missing_shard:
        os.remove(os.path.join(final_dir, shards[0]["shard_file"]))

    partition_id = f"test-{table}-{symbol}-{utc_year}-{utc_month:02d}"
    watermark = rows[-1][0] if rows else 0
    manifest = {
        "source_table": table, "symbol": symbol, "venue": spec.venue, "market_segment": spec.market_segment,
        "row_count": len(rows), "shards": shards,
        "ordered_scientific_content_hash": "0" * 64 if corrupt_manifest else "irrelevant-for-reader-tests",
        "partition_id": partition_id, "parquet_path": os.path.join(rel, shards[0]["shard_file"]) if shards else "",
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
        "source_watermark_field": "id", "source_watermark_value": watermark,
    }
    manifest_path = os.path.join(final_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    receipt_sha = None
    if not no_receipt:
        receipt = {"action": "CREATE_PRODUCTION_ARCHIVE_ONLY", "purge_authorization": "PROHIBITED",
                   "scheduler_authorization": "PROHIBITED", "vacuum_authorization": "PROHIBITED"}
        receipt_body_hash = hashlib.sha256(
            json.dumps(receipt, indent=2, sort_keys=True, default=str).encode()).hexdigest()
        receipt["receipt_sha256"] = receipt_body_hash
        receipt_path = os.path.join(final_dir, "authorization_receipt.json")
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f)
        if corrupt_receipt:
            with open(receipt_path, "r", encoding="utf-8") as f:
                tampered = json.load(f)
            tampered["purge_authorization"] = "TAMPERED"
            with open(receipt_path, "w", encoding="utf-8") as f:
                json.dump(tampered, f)
        receipt_sha = _sha256_file(receipt_path)

    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write(partition_id + "\n")

    manifest_sha = _sha256_file(manifest_path)
    recorded_manifest_sha = "f" * 64 if corrupt_manifest else manifest_sha

    catalog_entry = {
        "archive_identity": partition_id, "archive_relative_path": rel,
        "source_table": table, "symbol": symbol, "venue": spec.venue, "market_segment": spec.market_segment,
        "partition_start_ms": partition_start_ms, "partition_end_ms": partition_end_ms,
        "row_count": len(rows), "manifest_sha256": recorded_manifest_sha,
        "authorization_receipt_sha256": receipt_sha,
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
        "scientific_content_hash": manifest["ordered_scientific_content_hash"],
    }
    catalog_entry_path = os.path.join(final_dir, "catalog_entry.json")
    with open(catalog_entry_path, "w", encoding="utf-8") as f:
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


APR = 1775001600000   # 2026-04-01T00:00:00Z
MAY = 1777593600000   # 2026-05-01T00:00:00Z
JUN = 1780272000000   # 2026-06-01T00:00:00Z


# ---------------------------------------------------------------------------
# 1. SQLite-only lookup
# ---------------------------------------------------------------------------

def test_sqlite_only_lookup(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 20)  # ids 1..20, ts MAY..MAY+19min
    _sqlite_mark_prices(db_path, rows)

    query_ts = MAY + 10 * 60_000  # exactly row id=11's ts
    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=query_ts,
                                            source_db_path=db_path)
    assert result.found
    assert result.row[0] == 11
    assert result.provenance["source_type"] == "SQLITE_ONLY"
    assert result.provenance["result_source"] == "SQLITE"


# ---------------------------------------------------------------------------
# 2. Archive-only single-file partition lookup
# ---------------------------------------------------------------------------

def test_archive_only_single_file_lookup(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 20)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows)
    _write_catalog_index(root, [entry])

    query_ts = APR + 10 * 60_000
    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=query_ts)
    assert result.found
    assert result.row[0] == 11
    assert result.provenance["source_type"] == "ARCHIVE_ONLY"
    assert result.provenance["result_source"] == "ARCHIVE"
    assert result.provenance["archive_identity"] == entry["archive_identity"]
    assert result.provenance["manifest_sha256"] == entry["manifest_sha256"]


# ---------------------------------------------------------------------------
# 3 / 6. Archive-only multi-shard lookup, incl. shard-boundary lookup
# ---------------------------------------------------------------------------

def test_archive_only_multi_shard_lookup_and_shard_boundary(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _book_ticker_rows(1, APR, 25)  # ids 1..25, 1s apart
    entry = _build_archive_partition(root, table="book_ticker", spec=BT_SPEC, symbol="SOLUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, shard_sizes=[10, 10, 5])
    _write_catalog_index(root, [entry])

    # deep into the 3rd shard
    result = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="SOLUSDT", ts_ms=APR + 22_000)
    assert result.found and result.row[0] == 23
    assert result.provenance["source_type"] == "ARCHIVE_ONLY"

    # exactly at the shard-0/shard-1 boundary (row 10 is the last row of shard 0)
    result_b = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="SOLUSDT", ts_ms=APR + 9_000)
    assert result_b.found and result_b.row[0] == 10

    # 1ms before shard 1's first row -- must still resolve to shard 0's last row
    result_c = RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="SOLUSDT", ts_ms=APR + 9_999)
    assert result_c.found and result_c.row[0] == 10


# ---------------------------------------------------------------------------
# 4 / 5. Hybrid lookup (both sides), 16. gap provenance
# ---------------------------------------------------------------------------

def test_hybrid_lookup_ts_on_archive_side_falls_back_to_sqlite(tmp_path):
    """ts_ms falls chronologically inside the archived partition's
    declared range, but the archive itself has no row that early (the
    real 'data starts mid-month' pattern) -- must fall back to SQLite
    if SQLite happens to have an earlier row (synthetic proxy for that
    real-world gap), and report HYBRID since both sides were consulted."""
    root = str(tmp_path / "root")
    os.makedirs(root)
    archive_rows = _mark_prices_rows(11, APR + 20 * 60_000, 10)  # archive data starts at APR+20min, not APR
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=archive_rows)
    _write_catalog_index(root, [entry])
    db_path = str(tmp_path / "source.sqlite")
    sqlite_rows = _mark_prices_rows(1, APR - 10 * 60_000, 5)  # earlier "still-live" rows before the archive's own data
    _sqlite_mark_prices(db_path, sqlite_rows)

    query_ts = APR + 5 * 60_000  # inside archive's declared range, but before its real first row
    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=query_ts,
                                            source_db_path=db_path)
    assert result.found
    assert result.row[0] == 5  # last of the sqlite rows (id 1..5)
    assert result.provenance["source_type"] == "HYBRID"
    assert result.provenance["result_source"] == "SQLITE"


def test_hybrid_lookup_ts_on_sqlite_side_falls_back_to_archive(tmp_path):
    """ts_ms falls in SQLite's 'natural' turf (after the archive ends),
    but SQLite has no row that early yet -- must fall back to archive
    and report HYBRID."""
    root = str(tmp_path / "root")
    os.makedirs(root)
    archive_rows = _mark_prices_rows(1, APR, 10)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=archive_rows)
    _write_catalog_index(root, [entry])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, [])  # empty -- no live rows have arrived yet

    query_ts = MAY + 60_000  # just after the archive ends, but sqlite is empty
    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=query_ts,
                                            source_db_path=db_path)
    assert result.found
    assert result.row[0] == 10  # falls back to archive's last row
    assert result.provenance["source_type"] == "HYBRID"
    assert result.provenance["result_source"] == "ARCHIVE"


# ---------------------------------------------------------------------------
# 7. Exact timestamp hit / 8. between-timestamps fallback
# ---------------------------------------------------------------------------

def test_exact_timestamp_hit_and_between_timestamps_fallback(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 10, step_ms=60_000)  # ts at MAY, MAY+1min, ..., MAY+9min
    _sqlite_mark_prices(db_path, rows)

    exact = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT",
                                           ts_ms=MAY + 5 * 60_000, source_db_path=db_path)
    assert exact.found and exact.row[0] == 6 and exact.row_ts_ms == MAY + 5 * 60_000

    between = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT",
                                             ts_ms=MAY + 5 * 60_000 + 30_000, source_db_path=db_path)
    assert between.found and between.row[0] == 6  # still row 6, next row (7) is at +6min


# ---------------------------------------------------------------------------
# 9. No prior row / empty result
# ---------------------------------------------------------------------------

def test_no_prior_row_returns_empty_result_not_exception(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 10)
    _sqlite_mark_prices(db_path, rows)

    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT",
                                            ts_ms=MAY - 3600_000, source_db_path=db_path)
    assert result.found is False
    assert result.row is None
    assert result.row_ts_ms is None
    assert result.provenance["found"] is False


# ---------------------------------------------------------------------------
# 10 / 11. Inclusive <= vs exclusive < policy
# ---------------------------------------------------------------------------

def test_inclusive_vs_exclusive_policy(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 5, step_ms=60_000)  # ids 1..5
    _sqlite_mark_prices(db_path, rows)

    exact_ts = MAY + 2 * 60_000  # row id=3's exact ts
    inclusive = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=exact_ts,
                                               inclusive=True, source_db_path=db_path)
    exclusive = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=exact_ts,
                                               inclusive=False, source_db_path=db_path)
    assert inclusive.found and inclusive.row[0] == 3
    assert exclusive.found and exclusive.row[0] == 2
    assert inclusive.provenance["inclusive"] is True
    assert exclusive.provenance["inclusive"] is False


# ---------------------------------------------------------------------------
# 12. Selected-column projection
# ---------------------------------------------------------------------------

def test_column_projection(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 5)
    _sqlite_mark_prices(db_path, rows)

    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT",
                                            ts_ms=MAY + 3 * 60_000, columns=("mark_price",),
                                            source_db_path=db_path)
    assert result.found
    assert result.columns == ("mark_price",)
    assert len(result.row) == 1
    assert result.row[0] == 3003.0  # row id=4 (0-indexed i=3) -> 3000.0 + 3


# ---------------------------------------------------------------------------
# 13. Additional filter handling
# ---------------------------------------------------------------------------

def test_additional_filter_skips_non_matching_latest_row(tmp_path):
    """Mirrors the real `nextfund`-shaped pattern: latest row satisfying
    ts<=query AND funding_rate IS NOT NULL -- the single latest row by
    ts alone has funding_rate=None, so the search must keep walking
    backward past it."""
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    # funding present on rows 1,2 only; None afterwards
    rows = _mark_prices_rows(1, MAY, 5, funding=[0.0001, 0.0002, None, None, None])
    _sqlite_mark_prices(db_path, rows)

    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT",
                                            ts_ms=MAY + 4 * 60_000,
                                            filters=(("funding_rate", "!=", None),),
                                            source_db_path=db_path)
    assert result.found
    assert result.row[0] == 2  # last row with non-null funding_rate


# ---------------------------------------------------------------------------
# 14. Deterministic repeated run
# ---------------------------------------------------------------------------

def test_repeated_run_determinism(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _book_ticker_rows(1, APR, 25)
    entry = _build_archive_partition(root, table="book_ticker", spec=BT_SPEC, symbol="SOLUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, shard_sizes=[10, 10, 5])
    _write_catalog_index(root, [entry])
    results = [RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="SOLUSDT",
                                              ts_ms=APR + 22_000) for _ in range(3)]
    assert results[0].row == results[1].row == results[2].row


# ---------------------------------------------------------------------------
# 15. Ordering / tie-break parity with direct SQL (same ts_ms, differing id)
# ---------------------------------------------------------------------------

def test_tie_break_matches_direct_sql_ordering(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    tied_ts = MAY + 60_000
    rows = [(1, MAY, "ETHUSDT", 3000.0, None, None),
            (2, tied_ts, "ETHUSDT", 3001.0, None, None),
            (3, tied_ts, "ETHUSDT", 3002.0, None, None)]  # ids 2 and 3 share the same ts_ms
    _sqlite_mark_prices(db_path, rows)

    conn = sqlite3.connect(db_path)
    oracle = conn.execute(
        "SELECT id, ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms FROM mark_prices "
        "WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC, id DESC LIMIT 1", (tied_ts,)).fetchone()
    conn.close()

    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=tied_ts,
                                            source_db_path=db_path)
    assert result.found
    assert result.row == oracle
    assert result.row[0] == 3  # higher id wins the tie, matching direct-SQL DESC,DESC ordering


# ---------------------------------------------------------------------------
# 17. Overlap detection
# ---------------------------------------------------------------------------

def test_overlap_detection_raises(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows_a = _mark_prices_rows(1, APR, 10)
    entry_a = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                        utc_year=2026, utc_month=4, partition_start_ms=APR,
                                        partition_end_ms=MAY + 1000, rows=rows_a)
    rows_b = _mark_prices_rows(11, MAY, 10)
    entry_b = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                        utc_year=2026, utc_month=5, partition_start_ms=MAY - 1000,
                                        partition_end_ms=JUN, rows=rows_b)
    _write_catalog_index(root, [entry_a, entry_b])

    with pytest.raises(RR.OverlapDetectedError):
        RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=MAY + 500)


# ---------------------------------------------------------------------------
# 18 / 19 / 20. Missing shard / corrupt manifest / invalid receipt -> fail closed
# ---------------------------------------------------------------------------

def test_missing_shard_fails_closed(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _book_ticker_rows(1, APR, 20)
    entry = _build_archive_partition(root, table="book_ticker", spec=BT_SPEC, symbol="SOLUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, shard_sizes=[10, 10],
                                      missing_shard=True)
    _write_catalog_index(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        RR.lookup_latest_at_or_before(root, table="book_ticker", symbol="SOLUSDT", ts_ms=APR + 5000)


def test_corrupt_manifest_fails_closed(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 10)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, corrupt_manifest=True)
    _write_catalog_index(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=APR + 5000)


def test_invalid_receipt_fails_closed(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 10)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, corrupt_receipt=True)
    _write_catalog_index(root, [entry])
    with pytest.raises(RR.ArchiveTrustError, match="receipt self-hash mismatch"):
        RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=APR + 5000)


# ---------------------------------------------------------------------------
# 21. No full reverify triggered (static contract check)
# ---------------------------------------------------------------------------

def test_lookup_never_references_full_reverify():
    """Static contract check: `research_reader` must not import the
    guarded-reverify modules at all -- the full 114M-row scientific
    reverify stays reachable only via `ami.storage.reverify_guard`,
    invoked directly and separately by an operator, never transitively
    through a normal lookup/read call."""
    assert "reverify_guard" not in RR.__dict__
    assert not hasattr(RR, "run_guarded_reverify")
    assert not hasattr(RR, "verify_partition")


# ---------------------------------------------------------------------------
# max_lookback_ms bound
# ---------------------------------------------------------------------------

def test_max_lookback_bounds_the_search(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 5, step_ms=60_000)  # ids 1..5 at MAY..MAY+4min
    _sqlite_mark_prices(db_path, rows)

    query_ts = MAY + 20 * 60_000  # far past the last row
    unbounded = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=query_ts,
                                               source_db_path=db_path)
    bounded = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="ETHUSDT", ts_ms=query_ts,
                                             max_lookback_ms=5 * 60_000, source_db_path=db_path)
    assert unbounded.found  # row id=5 is within reach unbounded
    assert bounded.found is False  # row id=5 (at MAY+4min) is 16min before query -- outside the 5min lookback
    assert bounded.provenance["lower_bound_ms"] == query_ts - 5 * 60_000
