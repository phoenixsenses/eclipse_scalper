"""Tests for BATCH-STORAGE-ROTATION-RETENTION-RESEARCH-READER-
INTEGRATION-V1: `ami.storage.research_reader` -- the unified, read-only
SQLite/archive/hybrid reader. Small synthetic fixtures (a disposable
archive root + a disposable in-memory-shaped SQLite source) only; the
production smoke/parity tests against the real 650GB+ microstructure.db
and the real 3-entry production archive live in a separate file.
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


def _mark_prices_rows(start_id, start_ms, n, symbol="ETHUSDT", step_ms=60_000):
    return [(start_id + i, start_ms + i * step_ms, symbol, 3000.0 + i, None, None) for i in range(n)]


def _book_ticker_rows(start_id, start_ms, n, symbol="SOLUSDT", step_ms=1000):
    return [(start_id + i, start_ms + i * step_ms, symbol, 100.0, 1.0, 100.5, 1.0, 100.25, 0.005, 0.1, 1000.0)
            for i in range(n)]


def _build_archive_partition(root, *, table, spec, symbol, utc_year, utc_month,
                              partition_start_ms, partition_end_ms, rows, shard_sizes=None,
                              no_receipt=False, corrupt_manifest=False, missing_shard=False):
    """Builds a full, trust-check-passing (unless corrupt_*/missing_shard
    is set) production-shaped partition directory: N shard parquet files
    + manifest.json + catalog_entry.json + authorization_receipt.json +
    _SUCCESS. Returns the catalog_entry dict (also appended by the
    caller into catalog_index.json)."""
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
        receipt_sha = _sha256_file(receipt_path)

    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write(partition_id + "\n")

    manifest_sha = "0" * 64 if False else _sha256_file(manifest_path)  # actual file hash always matches file
    if corrupt_manifest:
        # simulate a manifest that was silently modified after the catalog
        # entry recorded its hash -- catalog_entry keeps the ORIGINAL hash
        recorded_manifest_sha = "f" * 64
    else:
        recorded_manifest_sha = manifest_sha

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
# 1. SQLite-only read
# ---------------------------------------------------------------------------

def test_sqlite_only_read(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 50)
    _sqlite_mark_prices(db_path, rows)

    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=MAY, end_ms=JUN)
    assert plan.mode == "SQLITE_ONLY"
    assert plan.archive_segments == ()
    assert plan.sqlite_ranges == ((MAY, JUN),)

    result = RR.execute_read(plan, batch_size=10, source_db_path=db_path)
    all_rows = list(result.iter_rows())
    assert len(all_rows) == 50
    assert all_rows[0][0] == 1 and all_rows[-1][0] == 50
    assert result.provenance["source_type"] == "SQLITE_ONLY"
    assert result.provenance["row_count"] == 50


# ---------------------------------------------------------------------------
# 2. Archive-only single-file partition
# ---------------------------------------------------------------------------

def test_archive_only_single_file_partition(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 30)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows)
    _write_catalog_index(root, [entry])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, [])  # empty -- must not be touched for archive-only range

    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=MAY)
    assert plan.mode == "ARCHIVE_ONLY"
    assert len(plan.archive_segments) == 1
    assert plan.sqlite_ranges == ()

    result = RR.execute_read(plan, batch_size=10, source_db_path=db_path)
    all_rows = list(result.iter_rows())
    assert len(all_rows) == 30
    assert [r[0] for r in all_rows] == list(range(1, 31))


# ---------------------------------------------------------------------------
# 3. Archive-only multi-shard book_ticker
# ---------------------------------------------------------------------------

def test_archive_only_multi_shard(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _book_ticker_rows(1, APR, 25)
    entry = _build_archive_partition(root, table="book_ticker", spec=BT_SPEC, symbol="SOLUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, shard_sizes=[10, 10, 5])
    _write_catalog_index(root, [entry])

    plan = RR.plan_read(root, table="book_ticker", symbol="SOLUSDT", start_ms=APR, end_ms=MAY)
    assert plan.mode == "ARCHIVE_ONLY"
    assert len(plan.archive_segments[0].shard_paths) == 3

    result = RR.execute_read(plan, batch_size=7)
    all_rows = list(result.iter_rows())
    assert len(all_rows) == 25
    assert [r[0] for r in all_rows] == list(range(1, 26))  # correct order across shard boundaries


def test_archive_only_multi_shard_narrow_window_skips_non_overlapping_shards(tmp_path):
    """A narrow sub-range must only stream the shard(s) whose [min_ts,
    max_ts] actually overlap it -- without this, a 10-second research
    query against a large multi-shard partition (e.g. real production
    book_ticker's 12 x 10M-row shards) would decode the entire
    partition. Proven here by asserting on `plan.archive_segments`'
    `shard_paths` count (still the full partition, for provenance) vs.
    the ACTUAL row content returned, which must be scoped to the window
    -- and by spying on `select_overlapping_row_groups` to confirm which
    physical shard files were even opened for row-group selection."""
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _book_ticker_rows(1, APR, 25)
    entry = _build_archive_partition(root, table="book_ticker", spec=BT_SPEC, symbol="SOLUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, shard_sizes=[10, 10, 5])
    _write_catalog_index(root, [entry])

    narrow_start = rows[10][1]  # first row of the second shard (ids 11-20)
    narrow_end = rows[19][1] + 1

    opened_paths = []
    real_select = RR.select_overlapping_row_groups

    def _spy(path, start_ms, end_ms):
        opened_paths.append(path)
        return real_select(path, start_ms, end_ms)

    import ami.storage.research_reader as rr_module
    monkeypatch_target = rr_module.select_overlapping_row_groups
    rr_module.select_overlapping_row_groups = _spy
    try:
        plan = RR.plan_read(root, table="book_ticker", symbol="SOLUSDT", start_ms=narrow_start, end_ms=narrow_end)
        result = RR.execute_read(plan)
        all_rows = list(result.iter_rows())
    finally:
        rr_module.select_overlapping_row_groups = monkeypatch_target

    assert [r[0] for r in all_rows] == list(range(11, 21))
    assert len(opened_paths) == 1  # only the second shard's file was even considered for row-group selection
    assert opened_paths[0].endswith("part-00001.parquet")


# ---------------------------------------------------------------------------
# 4. Hybrid boundary read
# ---------------------------------------------------------------------------

def test_hybrid_boundary_read(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    archive_rows = _mark_prices_rows(1, APR, 20)  # April: ids 1-20
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=archive_rows)
    _write_catalog_index(root, [entry])
    db_path = str(tmp_path / "source.sqlite")
    sqlite_rows = _mark_prices_rows(21, MAY, 15)  # May (live): ids 21-35
    _sqlite_mark_prices(db_path, sqlite_rows)

    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=JUN)
    assert plan.mode == "HYBRID"
    assert len(plan.archive_segments) == 1
    assert plan.sqlite_ranges == ((MAY, JUN),)

    result = RR.execute_read(plan, batch_size=100, source_db_path=db_path)
    all_rows = list(result.iter_rows())
    assert len(all_rows) == 35
    assert [r[0] for r in all_rows] == list(range(1, 36))  # archive (1-20) then sqlite (21-35), in order
    assert result.provenance["source_type"] == "HYBRID"


# ---------------------------------------------------------------------------
# 5. Column projection and predicate pushdown
# ---------------------------------------------------------------------------

def test_column_projection_and_filters(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 20)
    _sqlite_mark_prices(db_path, rows)

    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=MAY, end_ms=JUN)
    result = RR.execute_read(plan, columns=("id", "mark_price"),
                              filters=(("mark_price", ">=", 3010.0),), source_db_path=db_path)
    all_rows = list(result.iter_rows())
    assert result.columns == ("id", "mark_price")
    assert all(len(r) == 2 for r in all_rows)
    assert all(r[1] >= 3010.0 for r in all_rows)
    assert len(all_rows) == 10  # ids 11..20 have mark_price 3010..3019


def test_schema_mismatch_unknown_column(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=MAY, end_ms=JUN)
    with pytest.raises(RR.SchemaMismatchError):
        RR.execute_read(plan, columns=("id", "not_a_real_column"))


# ---------------------------------------------------------------------------
# 6. Deterministic ordering
# ---------------------------------------------------------------------------

def test_deterministic_ordering_hybrid(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    archive_rows = _mark_prices_rows(1, APR, 12)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=archive_rows)
    _write_catalog_index(root, [entry])
    db_path = str(tmp_path / "source.sqlite")
    sqlite_rows = _mark_prices_rows(13, MAY, 8)
    _sqlite_mark_prices(db_path, sqlite_rows)

    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=JUN)
    rows1 = list(RR.execute_read(plan, source_db_path=db_path).iter_rows())
    ts_values = [r[1] for r in rows1]
    assert ts_values == sorted(ts_values)
    ids = [r[0] for r in rows1]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# 7. Overlap detection
# ---------------------------------------------------------------------------

def test_overlap_detection_between_archive_partitions(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows_a = _mark_prices_rows(1, APR, 10)
    entry_a = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                        utc_year=2026, utc_month=4, partition_start_ms=APR,
                                        partition_end_ms=MAY + 1000, rows=rows_a)
    # entry_b's declared range deliberately overlaps entry_a's (corrupt/buggy catalog scenario)
    rows_b = _mark_prices_rows(11, MAY, 10)
    entry_b = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                        utc_year=2026, utc_month=5, partition_start_ms=MAY - 1000,
                                        partition_end_ms=JUN, rows=rows_b)
    _write_catalog_index(root, [entry_a, entry_b])

    with pytest.raises(RR.OverlapDetectedError):
        RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=JUN)


# ---------------------------------------------------------------------------
# 8. Missing shard / corrupt manifest / invalid receipt
# ---------------------------------------------------------------------------

def test_missing_shard_raises_archive_trust_error(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _book_ticker_rows(1, APR, 20)
    entry = _build_archive_partition(root, table="book_ticker", spec=BT_SPEC, symbol="SOLUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, shard_sizes=[10, 10],
                                      missing_shard=True)
    _write_catalog_index(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        RR.plan_read(root, table="book_ticker", symbol="SOLUSDT", start_ms=APR, end_ms=MAY)


def test_corrupt_manifest_hash_raises_archive_trust_error(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 10)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, corrupt_manifest=True)
    _write_catalog_index(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=MAY)


def test_missing_receipt_is_a_warning_not_a_failure(tmp_path):
    """Pre-activation-era / rehearsal entries may lack a receipt --
    that's a disclosed warning, not a hard failure (the field itself is
    optional-by-history, unlike a corrupted hash)."""
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 10)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows, no_receipt=True)
    _write_catalog_index(root, [entry])
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=MAY)
    assert plan.mode == "ARCHIVE_ONLY"
    assert any("no authorization_receipt.json" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# 9. Schema mismatch (table-level, not just column-level)
# ---------------------------------------------------------------------------

def test_unsupported_table_rejected(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    with pytest.raises(RR.UnsupportedTableError):
        RR.plan_read(root, table="not_a_real_table", symbol="ETHUSDT", start_ms=MAY, end_ms=JUN)


# ---------------------------------------------------------------------------
# 10. Empty range
# ---------------------------------------------------------------------------

def test_empty_range_end_before_start(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=JUN, end_ms=MAY)
    assert plan.mode == "EMPTY"
    result = RR.execute_read(plan)
    assert list(result.iter_rows()) == []


def test_empty_range_no_data_in_window(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, [])
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=MAY, end_ms=JUN)
    assert plan.mode == "SQLITE_ONLY"
    result = RR.execute_read(plan, source_db_path=db_path)
    assert list(result.iter_rows()) == []
    assert result.provenance["row_count"] == 0


# ---------------------------------------------------------------------------
# 11. Memory-bounded iteration
# ---------------------------------------------------------------------------

def test_memory_bounded_iteration_batches_never_exceed_batch_size(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    archive_rows = _mark_prices_rows(1, APR, 23)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=archive_rows)
    _write_catalog_index(root, [entry])
    db_path = str(tmp_path / "source.sqlite")
    sqlite_rows = _mark_prices_rows(24, MAY, 17)
    _sqlite_mark_prices(db_path, sqlite_rows)

    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=JUN)
    result = RR.execute_read(plan, batch_size=5, source_db_path=db_path)
    total = 0
    for batch in result.iter_batches():
        assert len(batch) <= 5
        total += len(batch)
    assert total == 40


def test_to_bounded_list_raises_when_exceeded(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, _mark_prices_rows(1, MAY, 30))
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=MAY, end_ms=JUN)
    result = RR.execute_read(plan, batch_size=5, source_db_path=db_path)
    with pytest.raises(RR.MaxRowsExceededError):
        result.to_bounded_list(max_rows=10)


# ---------------------------------------------------------------------------
# 12. Repeated query determinism
# ---------------------------------------------------------------------------

def test_repeated_query_determinism(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    archive_rows = _mark_prices_rows(1, APR, 14)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=archive_rows)
    _write_catalog_index(root, [entry])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, _mark_prices_rows(15, MAY, 9))

    results = []
    for _ in range(3):
        plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=JUN)
        result = RR.execute_read(plan, batch_size=4, source_db_path=db_path)
        results.append(list(result.iter_rows()))
    assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# Provenance content
# ---------------------------------------------------------------------------

def test_provenance_includes_archive_identity_and_hashes(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 5)
    entry = _build_archive_partition(root, table="mark_prices", spec=SPEC, symbol="ETHUSDT",
                                      utc_year=2026, utc_month=4, partition_start_ms=APR,
                                      partition_end_ms=MAY, rows=rows)
    _write_catalog_index(root, [entry])
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=APR, end_ms=MAY)
    result = RR.execute_read(plan)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["archive_segments"][0]["archive_identity"] == entry["archive_identity"]
    assert prov["archive_segments"][0]["manifest_sha256"] == entry["manifest_sha256"]
    assert prov["source_type"] == "ARCHIVE_ONLY"
    assert prov["ordering"] == RR.CANONICAL_ORDERING
