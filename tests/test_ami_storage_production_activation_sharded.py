"""End-to-end tests for `ami.storage.production_activation.
publish_authorized_production_partition_sharded` -- the multi-shard,
memory-bounded, resumable production publisher built to recover from the
book_ticker/SOLUSDT/2026-04 RAM-wall failure (see `ami.storage.
sharded_archive` module docstring). Uses a small synthetic in-memory
book_ticker-shaped database; never touches the real 650GB+
microstructure.db or the real production archive root.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage import production_activation as PA
from ami.storage import sharded_archive as SHA
from ami.storage.partition import build_partition_identity
from ami.storage.registry import get_table_spec

NOW = dt.datetime(2026, 7, 8, 18, 0, 0, tzinfo=dt.timezone.utc)
SPEC = get_table_spec("book_ticker")
START = 1775001600000  # 2026-04-01T00:00:00Z
END = 1777593600000    # 2026-05-01T00:00:00Z


def _synthetic_conn(rows_per_symbol: dict[str, int]):
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE book_ticker (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
          bid_price REAL NOT NULL, bid_qty REAL NOT NULL, ask_price REAL NOT NULL,
          ask_qty REAL NOT NULL, mid_price REAL NOT NULL, spread_pct REAL NOT NULL,
          book_imbalance REAL NOT NULL, bid_depth_usd REAL)
    """)
    conn.execute("CREATE INDEX idx_bt_symbol_ts ON book_ticker(symbol, ts_ms)")
    conn.execute("CREATE INDEX idx_bt_ts ON book_ticker(ts_ms)")
    symbols = list(rows_per_symbol.keys())
    counters = {s: 0 for s in symbols}
    i = 0
    while any(counters[s] < rows_per_symbol[s] for s in symbols):
        sym = symbols[i % len(symbols)]
        if counters[sym] < rows_per_symbol[sym]:
            ts = START + counters[sym] * 60_000
            conn.execute(
                "INSERT INTO book_ticker (ts_ms, symbol, bid_price, bid_qty, ask_price, ask_qty, "
                "mid_price, spread_pct, book_imbalance, bid_depth_usd) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, sym, 100.0 + counters[sym], 1.0, 100.5, 1.0, 100.25, 0.005, 0.1, 1000.0))
            counters[sym] += 1
        i += 1
    conn.commit()
    return conn


def _watermark(conn, symbol):
    return conn.execute(
        "SELECT MAX(id) FROM book_ticker WHERE symbol=? AND ts_ms>=? AND ts_ms<?",
        (symbol, START, END)).fetchone()[0]


def _archive_schema_hash():
    import hashlib
    from ami.storage.archive import build_pyarrow_schema
    return hashlib.sha256(str(build_pyarrow_schema(SPEC)).encode()).hexdigest()


def _partition_and_receipt(conn, root, symbol="SOLUSDT"):
    watermark = _watermark(conn, symbol)
    partition = build_partition_identity(
        table="book_ticker", symbol=symbol, utc_year=2026, utc_month=4,
        source_watermark_value=watermark, now=NOW)
    receipt = PA.build_authorization_receipt(
        partition=partition, spec=SPEC, archive_version="v1", root=root,
        source_schema_hash="ssh", archive_schema_hash=_archive_schema_hash(), max_source_rows=10_000,
        max_source_bytes=1 << 30, max_output_bytes=1 << 30, approver="gate", justification="test")
    return partition, receipt


def test_sharded_publish_creates_multi_shard_partition(tmp_path):
    conn = _synthetic_conn({"SOLUSDT": 23, "BTCUSDT": 19, "ETHUSDT": 17})
    root = str(tmp_path / "raw_v1")
    partition, receipt = _partition_and_receipt(conn, root)

    result = PA.publish_authorized_production_partition_sharded(
        conn, root=root, partition=partition, spec=SPEC, archive_version="v1",
        receipt=receipt, job_identity="TEST-JOB-1", source_schema_hash="ssh",
        export_cutoff=NOW.isoformat(), max_rows_per_shard=5, max_output_bytes_per_shard=10**9,
        batch_size=5)

    assert result.status == "PUBLISHED"
    assert result.row_count == 23
    assert result.reverification_mismatch_count == 0
    assert os.path.isdir(result.final_partition_dir)

    with open(os.path.join(result.final_partition_dir, PR.MANIFEST_NAME)) as f:
        manifest = json.load(f)
    assert manifest["row_count"] == 23
    assert len(manifest["shards"]) >= 5  # 23 rows / 5-per-shard
    for s in manifest["shards"]:
        assert os.path.exists(os.path.join(result.final_partition_dir, s["shard_file"]))

    # staging is gone (renamed away), no lock left behind
    staging_dir = PR.staging_partition_dir(root, partition, "TEST-JOB-1", "v1")
    assert not os.path.exists(staging_dir)


def test_sharded_publish_root_index_has_expected_entry_count(tmp_path):
    conn = _synthetic_conn({"SOLUSDT": 12})
    root = str(tmp_path / "raw_v1")
    partition, receipt = _partition_and_receipt(conn, root)
    PA.publish_authorized_production_partition_sharded(
        conn, root=root, partition=partition, spec=SPEC, archive_version="v1",
        receipt=receipt, job_identity="TEST-JOB-1", source_schema_hash="ssh",
        export_cutoff=NOW.isoformat(), max_rows_per_shard=4, max_output_bytes_per_shard=10**9)
    with open(os.path.join(root, PR.ROOT_INDEX_NAME)) as f:
        index = json.load(f)
    assert index["entry_count"] == 1


def test_sharded_publish_rejects_if_final_already_exists(tmp_path):
    conn = _synthetic_conn({"SOLUSDT": 6})
    root = str(tmp_path / "raw_v1")
    partition, receipt = _partition_and_receipt(conn, root)
    PA.publish_authorized_production_partition_sharded(
        conn, root=root, partition=partition, spec=SPEC, archive_version="v1",
        receipt=receipt, job_identity="TEST-JOB-1", source_schema_hash="ssh",
        export_cutoff=NOW.isoformat(), max_rows_per_shard=100, max_output_bytes_per_shard=10**9)
    with pytest.raises(PR.ProductionPublicationConflict):
        PA.publish_authorized_production_partition_sharded(
            conn, root=root, partition=partition, spec=SPEC, archive_version="v1",
            receipt=receipt, job_identity="TEST-JOB-2", source_schema_hash="ssh",
            export_cutoff=NOW.isoformat(), max_rows_per_shard=100, max_output_bytes_per_shard=10**9)


def test_memory_guard_abort_during_publish_then_resume_with_same_job_identity(tmp_path):
    """The realistic recovery scenario: the guard trips mid-export
    (nothing published, no lock acquired, no partial catalog state), and
    a second call with the SAME job_identity resumes and completes."""
    conn = _synthetic_conn({"SOLUSDT": 60, "ETHUSDT": 40})
    root = str(tmp_path / "raw_v1")
    partition, receipt = _partition_and_receipt(conn, root)

    with pytest.raises(SHA.MemoryGuardAbort):
        PA.publish_authorized_production_partition_sharded(
            conn, root=root, partition=partition, spec=SPEC, archive_version="v1",
            receipt=receipt, job_identity="TEST-JOB-RESUME", source_schema_hash="ssh",
            export_cutoff=NOW.isoformat(), max_rows_per_shard=10, max_output_bytes_per_shard=10**9,
            batch_size=6, rss_check=lambda: 10**12, rss_limit_bytes=1, rss_check_every_rows=12)

    final_dir = PR.final_partition_dir(root, partition, "v1")
    assert not os.path.exists(final_dir)  # nothing published
    staging_dir = PR.staging_partition_dir(root, partition, "TEST-JOB-RESUME", "v1")
    assert os.path.isdir(staging_dir)  # left resumable

    result = PA.publish_authorized_production_partition_sharded(
        conn, root=root, partition=partition, spec=SPEC, archive_version="v1",
        receipt=receipt, job_identity="TEST-JOB-RESUME", source_schema_hash="ssh",
        export_cutoff=NOW.isoformat(), max_rows_per_shard=10, max_output_bytes_per_shard=10**9,
        batch_size=6)

    assert result.status == "PUBLISHED"
    assert result.row_count == 60
    assert result.reverification_mismatch_count == 0


def test_sharded_publish_rejects_mismatched_receipt(tmp_path):
    conn = _synthetic_conn({"SOLUSDT": 5})
    root = str(tmp_path / "raw_v1")
    partition, receipt = _partition_and_receipt(conn, root)
    tampered = {**receipt, "symbol": "BTCUSDT"}
    with pytest.raises(PA.AuthorizationReceiptRejected):
        PA.publish_authorized_production_partition_sharded(
            conn, root=root, partition=partition, spec=SPEC, archive_version="v1",
            receipt=tampered, job_identity="TEST-JOB-1", source_schema_hash="ssh",
            export_cutoff=NOW.isoformat(), max_rows_per_shard=100, max_output_bytes_per_shard=10**9)
