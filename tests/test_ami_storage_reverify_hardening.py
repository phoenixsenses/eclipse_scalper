"""Tests for BATCH-STORAGE-PRODUCTION-ARCHIVE-REVERIFY-MEMORY-HARDENING-V1:
`ami.storage.reverify_worker` (in-process verification logic) and
`ami.storage.reverify_guard` (fresh-subprocess runner with an external
RSS guard). Small synthetic multi-shard fixtures only -- never touches
the real 650GB+ microstructure.db or the real production archive root.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from ami.storage import reverify_guard as RG
from ami.storage import reverify_worker as RW
from ami.storage import sharded_archive as SA
from ami.storage.registry import get_table_spec

SPEC = get_table_spec("mark_prices")


def _write_shard(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        pa.field("id", pa.int64()), pa.field("ts_ms", pa.int64()), pa.field("symbol", pa.string()),
        pa.field("mark_price", pa.float64()), pa.field("funding_rate", pa.float64(), nullable=True),
        pa.field("next_funding_time_ms", pa.int64(), nullable=True),
    ])
    arrays = [pa.array([r[i] for r in rows], type=schema.field(i).type) for i in range(6)]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path, compression="zstd")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _build_fixture_partition(tmp_path, *, n_shards=2, rows_per_shard=3, corrupt_manifest_hash=False):
    """Builds a small, valid (or deliberately corrupted) published-
    partition-shaped directory: N shard Parquet files + a manifest.json
    with a real shard inventory (sha256 per shard) -- mirroring exactly
    what ami.storage.sharded_archive + production_activation publish."""
    final_dir = str(tmp_path / "final")
    os.makedirs(final_dir)
    shards = []
    all_rows = []
    rid = 1
    for shard_idx in range(n_shards):
        rows = []
        for _ in range(rows_per_shard):
            rows.append((rid, 1777593600000 + rid, "ETHUSDT", 3000.0 + rid, None, None))
            all_rows.append(rows[-1])
            rid += 1
        shard_file = f"part-{shard_idx:05d}.parquet"
        shard_path = os.path.join(final_dir, shard_file)
        _write_shard(shard_path, rows)
        shards.append({
            "shard_index": shard_idx, "shard_file": shard_file, "row_count": len(rows),
            "min_id": rows[0][0], "max_id": rows[-1][0],
            "byte_size": os.path.getsize(shard_path), "sha256": _sha256_file(shard_path),
        })

    shard_paths = [os.path.join(final_dir, s["shard_file"]) for s in shards]
    agg = SA.stream_hash_parquet_multi(shard_paths, SPEC.preserved_columns)
    manifest = {
        "source_table": "mark_prices", "row_count": agg["row_count"],
        "ordered_scientific_content_hash": (
            "0" * 64 if corrupt_manifest_hash else agg["scientific_content_hash"]),
        "shards": shards, "partition_id": "test-partition-id",
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
    }
    with open(os.path.join(final_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    receipt = {"action": "CREATE_PRODUCTION_ARCHIVE_ONLY", "purge_authorization": "PROHIBITED",
               "scheduler_authorization": "PROHIBITED", "vacuum_authorization": "PROHIBITED"}
    receipt_path = os.path.join(final_dir, "authorization_receipt.json")
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f)
    catalog_entry = {"authorization_receipt_sha256": _sha256_file(receipt_path),
                      "purge_authorization": "PROHIBITED", "source_retention_status": "SOURCE_PRESENT"}
    with open(os.path.join(final_dir, "catalog_entry.json"), "w", encoding="utf-8") as f:
        json.dump(catalog_entry, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("test-partition-id\n")

    return final_dir, manifest


# ---------------------------------------------------------------------------
# verify_partition (in-process logic, no subprocess) -- fixture + mismatch
# ---------------------------------------------------------------------------

def test_verify_partition_clean_fixture_zero_mismatches(tmp_path):
    final_dir, manifest = _build_fixture_partition(tmp_path)
    result = RW.verify_partition(final_dir, restore_temp_root=str(tmp_path / ".pytest_temp"))
    assert result["mismatch_count"] == 0
    assert result["shard_mismatches"] == []
    assert result["aggregate_matches_manifest"] is True
    assert result["restore_check_ok"] is True
    assert result["aggregate_row_count"] == manifest["row_count"]


def test_verify_partition_detects_manifest_hash_mismatch(tmp_path):
    final_dir, manifest = _build_fixture_partition(tmp_path, corrupt_manifest_hash=True)
    result = RW.verify_partition(final_dir, restore_temp_root=str(tmp_path / ".pytest_temp"))
    assert result["mismatch_count"] >= 1
    assert result["aggregate_matches_manifest"] is False
    # not a crash -- a clean, structured finding
    assert "aggregate_hash" in result


def test_verify_partition_detects_corrupted_shard_file(tmp_path):
    final_dir, manifest = _build_fixture_partition(tmp_path)
    # corrupt one shard's bytes on disk after the manifest recorded its hash
    shard_path = os.path.join(final_dir, manifest["shards"][0]["shard_file"])
    with open(shard_path, "r+b") as f:
        f.seek(10)
        f.write(b"\xff\xff\xff\xff")
    result = RW.verify_partition(final_dir, do_restore_check=False)
    assert result["mismatch_count"] >= 1
    assert len(result["shard_mismatches"]) == 1
    assert result["shard_mismatches"][0]["shard_index"] == 0


def test_verify_partition_raises_on_missing_manifest(tmp_path):
    final_dir = str(tmp_path / "nonexistent")
    os.makedirs(final_dir)
    with pytest.raises(FileNotFoundError):
        RW.verify_partition(final_dir)


def test_verify_partition_restore_check_picks_smallest_shard(tmp_path):
    final_dir = str(tmp_path / "final")
    os.makedirs(final_dir)
    shards = []
    rid = 1
    row_sets = [3, 1, 2]  # shard 1 is smallest
    all_paths = []
    for shard_idx, n in enumerate(row_sets):
        rows = []
        for _ in range(n):
            rows.append((rid, 1777593600000 + rid, "ETHUSDT", 3000.0 + rid, None, None))
            rid += 1
        shard_file = f"part-{shard_idx:05d}.parquet"
        shard_path = os.path.join(final_dir, shard_file)
        _write_shard(shard_path, rows)
        all_paths.append(shard_path)
        shards.append({"shard_index": shard_idx, "shard_file": shard_file, "row_count": n,
                        "min_id": rows[0][0], "max_id": rows[-1][0],
                        "byte_size": os.path.getsize(shard_path), "sha256": _sha256_file(shard_path)})
    agg = SA.stream_hash_parquet_multi(all_paths, SPEC.preserved_columns)
    manifest = {"source_table": "mark_prices", "row_count": agg["row_count"],
                "ordered_scientific_content_hash": agg["scientific_content_hash"], "shards": shards}
    with open(os.path.join(final_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    result = RW.verify_partition(final_dir, restore_temp_root=str(tmp_path / ".pytest_temp"))
    assert result["restore_check_shard_index"] == 1  # the 1-row shard
    assert result["restore_check_ok"] is True


# ---------------------------------------------------------------------------
# run_guarded_reverify (fresh subprocess + external RSS guard)
# ---------------------------------------------------------------------------

def test_guarded_reverify_successful_run(tmp_path):
    final_dir, manifest = _build_fixture_partition(tmp_path)
    result = RG.run_guarded_reverify(
        final_dir, rss_limit_bytes=2 * 1024 ** 3, poll_interval_seconds=0.2,
        restore_temp_root=str(tmp_path / ".pytest_temp"))
    assert result["mismatch_count"] == 0
    assert result["aggregate_matches_manifest"] is True
    assert "peak_rss_bytes" in result and result["peak_rss_bytes"] > 0
    assert result["rss_limit_bytes"] == 2 * 1024 ** 3


def test_guarded_reverify_detects_mismatch_without_raising(tmp_path):
    final_dir, manifest = _build_fixture_partition(tmp_path, corrupt_manifest_hash=True)
    result = RG.run_guarded_reverify(
        final_dir, rss_limit_bytes=2 * 1024 ** 3, poll_interval_seconds=0.2,
        restore_temp_root=str(tmp_path / ".pytest_temp"))
    assert result["mismatch_count"] >= 1
    assert result["aggregate_matches_manifest"] is False


def test_guarded_reverify_trips_on_absurdly_low_rss_limit(tmp_path):
    """Simulates a guard-exceeded scenario: any real Python subprocess
    (interpreter startup + pyarrow import) uses far more than 1KB
    resident memory, so this guard is guaranteed to trip on the very
    first poll -- proving the external-RSS-kill path works without
    needing a special memory-bloating test hook in production code."""
    final_dir, manifest = _build_fixture_partition(tmp_path)
    with pytest.raises(RG.ReverifyMemoryGuardAbort):
        RG.run_guarded_reverify(
            final_dir, rss_limit_bytes=1024, poll_interval_seconds=0.05,
            restore_temp_root=str(tmp_path / ".pytest_temp"))


def test_guarded_reverify_raises_on_worker_crash(tmp_path):
    """final_dir has no manifest.json at all -- the worker crashes with
    an unhandled FileNotFoundError, exits non-zero, and the guard must
    surface this as ReverifyWorkerCrashed, not silently succeed."""
    final_dir = str(tmp_path / "broken")
    os.makedirs(final_dir)
    with pytest.raises(RG.ReverifyWorkerCrashed):
        RG.run_guarded_reverify(
            final_dir, rss_limit_bytes=2 * 1024 ** 3, poll_interval_seconds=0.2,
            restore_temp_root=str(tmp_path / ".pytest_temp"))


def test_guarded_reverify_cleans_up_result_file(tmp_path):
    final_dir, manifest = _build_fixture_partition(tmp_path)
    RG.run_guarded_reverify(
        final_dir, rss_limit_bytes=2 * 1024 ** 3, poll_interval_seconds=0.2,
        restore_temp_root=str(tmp_path / ".pytest_temp"))
    # no leftover reverify_result_*.json in the restore_temp_root
    leftovers = [f for f in os.listdir(str(tmp_path / ".pytest_temp")) if f.startswith("reverify_result_")]
    assert leftovers == []


def test_guarded_reverify_does_not_touch_shard_or_manifest_files(tmp_path):
    final_dir, manifest = _build_fixture_partition(tmp_path)
    shard_path = os.path.join(final_dir, manifest["shards"][0]["shard_file"])
    manifest_path = os.path.join(final_dir, "manifest.json")
    before_shard_mtime = os.path.getmtime(shard_path)
    before_manifest_mtime = os.path.getmtime(manifest_path)
    before_shard_hash = _sha256_file(shard_path)

    RG.run_guarded_reverify(
        final_dir, rss_limit_bytes=2 * 1024 ** 3, poll_interval_seconds=0.2,
        restore_temp_root=str(tmp_path / ".pytest_temp"))

    assert os.path.getmtime(shard_path) == before_shard_mtime
    assert os.path.getmtime(manifest_path) == before_manifest_mtime
    assert _sha256_file(shard_path) == before_shard_hash
