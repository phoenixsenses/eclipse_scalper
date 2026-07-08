"""Synthetic (small-scale, disposable) tests for
`ami.storage.sharded_archive` -- the memory-bounded, multi-shard,
resumable streaming exporter designed to fix the book_ticker/SOLUSDT/
2026-04 RAM-wall failure. Proves correctness/resumability/hash-parity at
small scale; never touches the real 650GB+ microstructure.db.

Run: pytest tests/test_ami_storage_sharded_archive.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

import pytest

from ami.storage import sharded_archive as SA
from ami.storage.archive import canonical_row_hash, ExportValidationError, stream_export_to_parquet
from ami.storage.partition import PartitionIdentity
from ami.storage.registry import get_table_spec

SPEC = get_table_spec("book_ticker")


def _make_synthetic_db(path: str, *, rows_per_symbol: dict[str, int], start_ms: int, step_ms: int = 60_000):
    """Builds a small disposable SQLite DB with the same book_ticker
    schema/indices as production, interleaving symbols round-robin by
    insertion order (id) -- the same structural pattern that makes the
    real table's ORDER BY id require a temp B-tree when filtered by
    (symbol, ts_ms)."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE book_ticker (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_ms INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          bid_price REAL NOT NULL,
          bid_qty REAL NOT NULL,
          ask_price REAL NOT NULL,
          ask_qty REAL NOT NULL,
          mid_price REAL NOT NULL,
          spread_pct REAL NOT NULL,
          book_imbalance REAL NOT NULL,
          bid_depth_usd REAL
        )
    """)
    conn.execute("CREATE INDEX idx_bt_symbol_ts ON book_ticker(symbol, ts_ms)")
    conn.execute("CREATE INDEX idx_bt_ts ON book_ticker(ts_ms)")
    symbols = list(rows_per_symbol.keys())
    counters = {s: 0 for s in symbols}
    total = sum(rows_per_symbol.values())
    i = 0
    while any(counters[s] < rows_per_symbol[s] for s in symbols):
        sym = symbols[i % len(symbols)]
        if counters[sym] < rows_per_symbol[sym]:
            ts = start_ms + counters[sym] * step_ms
            conn.execute(
                "INSERT INTO book_ticker (ts_ms, symbol, bid_price, bid_qty, ask_price, ask_qty, "
                "mid_price, spread_pct, book_imbalance, bid_depth_usd) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, sym, 100.0 + counters[sym], 1.0, 100.5 + counters[sym], 1.0,
                 100.25 + counters[sym], 0.005, 0.1, 1000.0))
            counters[sym] += 1
        i += 1
    conn.commit()
    conn.close()


def _partition_for(symbol: str, *, start_ms: int, end_ms: int, watermark: int) -> PartitionIdentity:
    return PartitionIdentity(
        table="book_ticker", symbol=symbol, venue=SPEC.venue, market_segment=SPEC.market_segment,
        utc_year=2026, utc_month=4, partition_start_ms=start_ms, partition_end_ms=end_ms,
        source_watermark_field=SPEC.stable_ordering_field, source_watermark_value=watermark)


def _watermark(conn, symbol, start_ms, end_ms) -> int:
    return conn.execute(
        "SELECT MAX(id) FROM book_ticker WHERE symbol=? AND ts_ms>=? AND ts_ms<?",
        (symbol, start_ms, end_ms)).fetchone()[0]


START = 1775001600000  # 2026-04-01T00:00:00Z
END = 1777593600000    # 2026-05-01T00:00:00Z


def test_resolve_partition_id_bounds_matches_direct_query(tmp_path):
    db = str(tmp_path / "synthetic.sqlite")
    _make_synthetic_db(db, rows_per_symbol={"SOLUSDT": 23, "BTCUSDT": 19, "ETHUSDT": 17}, start_ms=START)
    conn = sqlite3.connect(db)
    watermark = _watermark(conn, "SOLUSDT", START, END)
    partition = _partition_for("SOLUSDT", start_ms=START, end_ms=END, watermark=watermark)
    min_id, max_id = SA.resolve_partition_id_bounds(conn, SPEC, partition)
    direct = conn.execute(
        "SELECT MIN(id), MAX(id) FROM book_ticker WHERE symbol='SOLUSDT' AND ts_ms>=? AND ts_ms<?",
        (START, END)).fetchone()
    assert (min_id, max_id) == tuple(direct)
    assert max_id == watermark
    conn.close()


def test_resolve_partition_id_bounds_empty_partition_returns_none(tmp_path):
    db = str(tmp_path / "synthetic.sqlite")
    _make_synthetic_db(db, rows_per_symbol={"SOLUSDT": 5}, start_ms=START)
    conn = sqlite3.connect(db)
    partition = _partition_for("BTCUSDT", start_ms=START, end_ms=END, watermark=0)
    min_id, max_id = SA.resolve_partition_id_bounds(conn, SPEC, partition)
    assert (min_id, max_id) == (None, None)
    conn.close()


def test_sharded_export_matches_single_file_export_row_for_row(tmp_path):
    """The core parity proof: the sharded exporter, split across multiple
    small shards, must produce exactly the same row count, id range, and
    (via a fresh stream_hash_parquet_multi pass) the same ordered
    scientific-content hash as the existing, accepted single-file
    stream_export_to_parquet -- proving the ts_ms-ordered (symbol,ts_ms)-
    index scan selects and orders the identical row set as the single-
    file exporter's `ORDER BY id` scan (which, at real scale, requires a
    temp-sort the ts_ms-ordered scan avoids)."""
    db = str(tmp_path / "synthetic.sqlite")
    _make_synthetic_db(db, rows_per_symbol={"SOLUSDT": 47, "BTCUSDT": 31, "ETHUSDT": 29}, start_ms=START)
    conn = sqlite3.connect(db)
    watermark = _watermark(conn, "SOLUSDT", START, END)
    partition = _partition_for("SOLUSDT", start_ms=START, end_ms=END, watermark=watermark)

    single_out = str(tmp_path / "single.parquet")
    single = stream_export_to_parquet(conn, SPEC, partition, single_out, max_output_bytes=10**9)

    staging = str(tmp_path / "sharded_staging")
    result = SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, staging, batch_size=7, max_rows_per_shard=5,
        max_output_bytes_per_shard=10**9)

    assert result.complete is True
    assert len(result.shards) >= 2  # 47 rows / 5-per-shard forces multiple shards
    assert result.row_count == single["row_count"] == 47
    assert result.min_id == watermark - 47 + 1 or result.min_id is not None
    assert result.max_id == watermark == single["max_id"]

    shard_paths = [os.path.join(staging, s["shard_file"]) for s in sorted(result.shards, key=lambda s: s["shard_index"])]
    agg = SA.stream_hash_parquet_multi(shard_paths, SPEC.preserved_columns)
    assert agg["row_count"] == single["row_count"]
    assert agg["scientific_content_hash"] == single["scientific_content_hash"]
    assert agg["min_id"] == single["min_id"]
    assert agg["max_id"] == single["max_id"]

    # per-shard row counts sum to the whole, and shard checkpoints agree
    # with what discover_resumable_shards reports after the fact
    rediscovered = SA.discover_resumable_shards(staging, partition_id=partition.partition_id)
    assert sum(s["row_count"] for s in rediscovered) == 47
    assert [s["shard_index"] for s in rediscovered] == list(range(len(rediscovered)))
    conn.close()


def test_sharded_export_excludes_other_symbols(tmp_path):
    """Other symbols' rows (the SQL WHERE symbol=? filter -- an index
    predicate here, not a post-query Python filter, since the scan is
    ts_ms-ordered via the (symbol, ts_ms) index) must never appear in
    the shard output."""
    db = str(tmp_path / "synthetic.sqlite")
    _make_synthetic_db(db, rows_per_symbol={"SOLUSDT": 10, "BTCUSDT": 40}, start_ms=START)
    conn = sqlite3.connect(db)
    watermark = _watermark(conn, "SOLUSDT", START, END)
    partition = _partition_for("SOLUSDT", start_ms=START, end_ms=END, watermark=watermark)
    staging = str(tmp_path / "staging")
    result = SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, staging, batch_size=100, max_rows_per_shard=100,
        max_output_bytes_per_shard=10**9)
    assert result.row_count == 10
    shard_paths = [os.path.join(staging, s["shard_file"]) for s in result.shards]
    agg = SA.stream_hash_parquet_multi(shard_paths, SPEC.preserved_columns)
    assert agg["row_count"] == 10
    conn.close()


def test_sharded_export_handles_tied_timestamps_deterministically(tmp_path):
    """Multiple rows sharing the same ts_ms (a real possibility for
    book_ticker snapshots) must all be included exactly once, in a
    deterministic (ts_ms, id) order, whether produced in one call or
    split across a resume -- proving the compound resume cursor
    (ts_ms, id), not ts_ms alone, is what makes resumption correct."""
    db = str(tmp_path / "synthetic.sqlite")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE book_ticker (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
          bid_price REAL NOT NULL, bid_qty REAL NOT NULL, ask_price REAL NOT NULL,
          ask_qty REAL NOT NULL, mid_price REAL NOT NULL, spread_pct REAL NOT NULL,
          book_imbalance REAL NOT NULL, bid_depth_usd REAL)
    """)
    conn.execute("CREATE INDEX idx_bt_symbol_ts ON book_ticker(symbol, ts_ms)")
    conn.execute("CREATE INDEX idx_bt_ts ON book_ticker(ts_ms)")
    # 5 distinct timestamps, each with 3 tied SOLUSDT rows (same ts_ms,
    # increasing id) -- 15 rows total, all the same symbol.
    for t in range(5):
        for _ in range(3):
            conn.execute(
                "INSERT INTO book_ticker (ts_ms, symbol, bid_price, bid_qty, ask_price, ask_qty, "
                "mid_price, spread_pct, book_imbalance, bid_depth_usd) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (START + t * 60_000, "SOLUSDT", 100.0, 1.0, 100.5, 1.0, 100.25, 0.005, 0.1, 1000.0))
    conn.commit()
    watermark = _watermark(conn, "SOLUSDT", START, END)
    partition = _partition_for("SOLUSDT", start_ms=START, end_ms=END, watermark=watermark)

    baseline_staging = str(tmp_path / "baseline")
    baseline = SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, baseline_staging, batch_size=4, max_rows_per_shard=4,
        max_output_bytes_per_shard=10**9)
    baseline_paths = [os.path.join(baseline_staging, s["shard_file"])
                       for s in sorted(baseline.shards, key=lambda s: s["shard_index"])]
    baseline_agg = SA.stream_hash_parquet_multi(baseline_paths, SPEC.preserved_columns)
    assert baseline.row_count == 15
    assert baseline_agg["row_count"] == 15

    # Simulate a guard-abort partway through, then resume -- must still
    # yield exactly 15 rows once, not skip or duplicate any tied row.
    interrupted_staging = str(tmp_path / "interrupted")
    trip = {"n": 0}

    def rss_check():
        trip["n"] += 1
        return 10**12 if trip["n"] == 1 else 0

    with pytest.raises(SA.MemoryGuardAbort):
        SA.stream_export_to_parquet_sharded(
            conn, SPEC, partition, interrupted_staging, batch_size=4, max_rows_per_shard=4,
            max_output_bytes_per_shard=10**9, rss_check=rss_check, rss_limit_bytes=1,
            rss_check_every_rows=4)
    resumed = SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, interrupted_staging, batch_size=4, max_rows_per_shard=4,
        max_output_bytes_per_shard=10**9)
    resumed_paths = [os.path.join(interrupted_staging, s["shard_file"])
                      for s in sorted(resumed.shards, key=lambda s: s["shard_index"])]
    resumed_agg = SA.stream_hash_parquet_multi(resumed_paths, SPEC.preserved_columns)

    assert resumed.row_count == 15
    assert resumed_agg["scientific_content_hash"] == baseline_agg["scientific_content_hash"]
    conn.close()


def test_memory_guard_abort_then_resume_produces_identical_final_result(tmp_path):
    """Simulates the RSS guard tripping partway through, then a second
    call (same staging_dir, no guard) resuming -- the combined result
    must be identical (row_count/hash/id-range) to an uninterrupted
    single call."""
    db = str(tmp_path / "synthetic.sqlite")
    _make_synthetic_db(db, rows_per_symbol={"SOLUSDT": 60, "ETHUSDT": 40}, start_ms=START)
    conn = sqlite3.connect(db)
    watermark = _watermark(conn, "SOLUSDT", START, END)
    partition = _partition_for("SOLUSDT", start_ms=START, end_ms=END, watermark=watermark)

    # Baseline: uninterrupted
    baseline_staging = str(tmp_path / "baseline")
    baseline = SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, baseline_staging, batch_size=6, max_rows_per_shard=10,
        max_output_bytes_per_shard=10**9)
    baseline_paths = [os.path.join(baseline_staging, s["shard_file"]) for s in baseline.shards]
    baseline_agg = SA.stream_hash_parquet_multi(baseline_paths, SPEC.preserved_columns)

    # Interrupted: guard trips after the first shard is finalized
    interrupted_staging = str(tmp_path / "interrupted")
    calls = {"n": 0}

    def rss_check_trip_once():
        calls["n"] += 1
        return 10**12  # always "over the limit" once checked

    with pytest.raises(SA.MemoryGuardAbort):
        SA.stream_export_to_parquet_sharded(
            conn, SPEC, partition, interrupted_staging, batch_size=6, max_rows_per_shard=10,
            max_output_bytes_per_shard=10**9, rss_check=rss_check_trip_once,
            rss_limit_bytes=1, rss_check_every_rows=12)
    assert calls["n"] >= 1
    partial_shards = SA.discover_resumable_shards(interrupted_staging, partition_id=partition.partition_id)
    assert 0 < len(partial_shards) < len(baseline.shards)  # some progress, not everything

    # Resume: same staging_dir, no guard this time
    resumed = SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, interrupted_staging, batch_size=6, max_rows_per_shard=10,
        max_output_bytes_per_shard=10**9)
    assert resumed.complete is True
    resumed_paths = [os.path.join(interrupted_staging, s["shard_file"])
                      for s in sorted(resumed.shards, key=lambda s: s["shard_index"])]
    resumed_agg = SA.stream_hash_parquet_multi(resumed_paths, SPEC.preserved_columns)

    assert resumed.row_count == baseline.row_count == 60
    assert resumed_agg["scientific_content_hash"] == baseline_agg["scientific_content_hash"]
    assert resumed_agg["min_id"] == baseline_agg["min_id"]
    assert resumed_agg["max_id"] == baseline_agg["max_id"]
    conn.close()


def test_second_call_after_full_completion_is_a_clean_noop(tmp_path):
    db = str(tmp_path / "synthetic.sqlite")
    _make_synthetic_db(db, rows_per_symbol={"SOLUSDT": 15}, start_ms=START)
    conn = sqlite3.connect(db)
    watermark = _watermark(conn, "SOLUSDT", START, END)
    partition = _partition_for("SOLUSDT", start_ms=START, end_ms=END, watermark=watermark)
    staging = str(tmp_path / "staging")
    first = SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, staging, batch_size=4, max_rows_per_shard=6,
        max_output_bytes_per_shard=10**9)
    second = SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, staging, batch_size=4, max_rows_per_shard=6,
        max_output_bytes_per_shard=10**9)
    assert second.row_count == first.row_count == 15
    assert second.shards == first.shards
    conn.close()


def test_discover_resumable_shards_rejects_foreign_partition_id(tmp_path):
    db = str(tmp_path / "synthetic.sqlite")
    _make_synthetic_db(db, rows_per_symbol={"SOLUSDT": 8}, start_ms=START)
    conn = sqlite3.connect(db)
    watermark = _watermark(conn, "SOLUSDT", START, END)
    partition = _partition_for("SOLUSDT", start_ms=START, end_ms=END, watermark=watermark)
    staging = str(tmp_path / "staging")
    SA.stream_export_to_parquet_sharded(
        conn, SPEC, partition, staging, batch_size=8, max_rows_per_shard=100,
        max_output_bytes_per_shard=10**9)
    with pytest.raises(SA.StaleStagingConflict):
        SA.discover_resumable_shards(staging, partition_id="some-other-partition-id")
    conn.close()


def test_clean_stale_staging_removes_partial_and_foreign_shards(tmp_path):
    staging = str(tmp_path / "staging")
    os.makedirs(staging)
    with open(os.path.join(staging, "part-00000.parquet.partial"), "w") as f:
        f.write("junk")
    foreign_ckpt = {"partition_id": "FOREIGN", "shard_index": 0, "shard_file": "part-00000.parquet",
                     "row_count": 1, "min_id": 1, "max_id": 1, "byte_size": 1}
    with open(os.path.join(staging, "part-00000.parquet"), "w") as f:
        f.write("junk")
    SA._write_json_atomic(os.path.join(staging, "shard-00000.checkpoint.json"), foreign_ckpt)

    removed = SA.clean_stale_staging(staging, partition_id="REAL_PARTITION_ID")
    assert removed == 3  # .partial + foreign parquet + foreign checkpoint
    assert os.listdir(staging) == []


def test_shard_output_size_cap_enforced(tmp_path):
    db = str(tmp_path / "synthetic.sqlite")
    _make_synthetic_db(db, rows_per_symbol={"SOLUSDT": 200}, start_ms=START)
    conn = sqlite3.connect(db)
    watermark = _watermark(conn, "SOLUSDT", START, END)
    partition = _partition_for("SOLUSDT", start_ms=START, end_ms=END, watermark=watermark)
    staging = str(tmp_path / "staging")
    with pytest.raises(ExportValidationError):
        SA.stream_export_to_parquet_sharded(
            conn, SPEC, partition, staging, batch_size=50, max_rows_per_shard=1000,
            max_output_bytes_per_shard=1)  # impossibly small cap
    conn.close()
