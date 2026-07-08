"""Focused tests: ami.storage.reader.stream_read_partition_shards and
ami.storage.restorer.stream_restore_slice_multi -- the multi-shard,
streaming read/restore counterparts to the single-file reader/restorer,
needed because the sharded exporter (ami.storage.sharded_archive) never
produces a single Parquet file for a partition the size of
book_ticker/SOLUSDT/2026-04.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from ami.storage import reader as RD
from ami.storage import restorer as RS
from ami.storage.registry import get_table_spec

SPEC = get_table_spec("mark_prices")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_arrays(
        [pa.array([r[0] for r in rows], type=pa.int64()), pa.array([r[1] for r in rows], type=pa.string())],
        schema=pa.schema([pa.field("id", pa.int64()), pa.field("symbol", pa.string())]))
    pq.write_table(table, path, compression="zstd")


def _canonical_hash(rows):
    parts = ["\x1f".join(repr(v) for v in row) for row in rows]
    return hashlib.sha256("\x1e".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# stream_read_partition_shards
# ---------------------------------------------------------------------------

def test_stream_read_shards_requires_manifest(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_parquet(p, [(1, "ETHUSDT")])
    with pytest.raises(RD.ManifestRequiredError):
        list(RD.stream_read_partition_shards(parquet_paths=[p], manifest={}, requested_symbol="ETHUSDT"))


def test_stream_read_shards_rejects_symbol_mismatch(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_parquet(p, [(1, "ETHUSDT")])
    manifest = {"symbol": "BTCUSDT", "shards": [{"shard_file": "a.parquet", "sha256": _sha(p)}]}
    with pytest.raises(RD.PartitionMismatchError):
        list(RD.stream_read_partition_shards(parquet_paths=[p], manifest=manifest, requested_symbol="ETHUSDT"))


def test_stream_read_shards_rejects_shard_count_mismatch(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_parquet(p, [(1, "ETHUSDT")])
    manifest = {"symbol": "ETHUSDT", "shards": [{"shard_file": "a.parquet", "sha256": _sha(p)},
                                                  {"shard_file": "b.parquet", "sha256": "x"}]}
    with pytest.raises(RD.PartitionMismatchError):
        list(RD.stream_read_partition_shards(parquet_paths=[p], manifest=manifest, requested_symbol="ETHUSDT"))


def test_stream_read_shards_rejects_checksum_mismatch(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_parquet(p, [(1, "ETHUSDT")])
    manifest = {"symbol": "ETHUSDT", "shards": [{"shard_file": "a.parquet", "sha256": "WRONG"}]}
    with pytest.raises(RD.ArchiveCorruptionError):
        list(RD.stream_read_partition_shards(parquet_paths=[p], manifest=manifest, requested_symbol="ETHUSDT"))


def test_stream_read_shards_yields_rows_in_shard_order(tmp_path):
    p0 = str(tmp_path / "part-00000.parquet")
    p1 = str(tmp_path / "part-00001.parquet")
    _write_parquet(p0, [(1, "ETHUSDT"), (2, "ETHUSDT")])
    _write_parquet(p1, [(3, "ETHUSDT"), (4, "ETHUSDT")])
    manifest = {"symbol": "ETHUSDT", "shards": [
        {"shard_file": "part-00000.parquet", "sha256": _sha(p0)},
        {"shard_file": "part-00001.parquet", "sha256": _sha(p1)},
    ]}
    rows = list(RD.stream_read_partition_shards(parquet_paths=[p0, p1], manifest=manifest,
                                                  requested_symbol="ETHUSDT"))
    assert rows == [(1, "ETHUSDT"), (2, "ETHUSDT"), (3, "ETHUSDT"), (4, "ETHUSDT")]


def test_stream_read_shards_column_projection(tmp_path):
    p0 = str(tmp_path / "part-00000.parquet")
    _write_parquet(p0, [(1, "ETHUSDT")])
    manifest = {"symbol": "ETHUSDT", "shards": [{"shard_file": "part-00000.parquet", "sha256": _sha(p0)}]}
    rows = list(RD.stream_read_partition_shards(parquet_paths=[p0], manifest=manifest,
                                                  requested_symbol="ETHUSDT", columns=("id",)))
    assert rows == [(1,)]


def test_stream_read_shards_fails_closed_before_yielding_on_bad_second_shard(tmp_path):
    """Manifest-first discipline: if any shard's checksum is wrong, the
    generator must raise on iteration start (or at latest before any
    downstream row is consumed for real work) -- proven here by fully
    draining a list() and catching the raise, since checksum validation
    happens before the generator body ever runs (this function is not a
    generator itself; it returns one)."""
    p0 = str(tmp_path / "part-00000.parquet")
    p1 = str(tmp_path / "part-00001.parquet")
    _write_parquet(p0, [(1, "ETHUSDT")])
    _write_parquet(p1, [(2, "ETHUSDT")])
    manifest = {"symbol": "ETHUSDT", "shards": [
        {"shard_file": "part-00000.parquet", "sha256": _sha(p0)},
        {"shard_file": "part-00001.parquet", "sha256": "WRONG"},
    ]}
    with pytest.raises(RD.ArchiveCorruptionError):
        RD.stream_read_partition_shards(parquet_paths=[p0, p1], manifest=manifest, requested_symbol="ETHUSDT")


# ---------------------------------------------------------------------------
# stream_restore_slice_multi
# ---------------------------------------------------------------------------

def test_restore_multi_single_shard_matches_stream_restore_slice(tmp_path):
    rows = [(1, 1777593600001, "ETHUSDT", 3000.0, None, None),
            (2, 1777593600002, "ETHUSDT", 3001.0, 0.0001, 1777600000000)]
    h = RS._canonical_row_hash(rows)
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("ts_ms", pa.int64()),
                          pa.field("symbol", pa.string()), pa.field("mark_price", pa.float64()),
                          pa.field("funding_rate", pa.float64(), nullable=True),
                          pa.field("next_funding_time_ms", pa.int64(), nullable=True)])
    arrays = [pa.array([r[i] for r in rows], type=schema.field(i).type) for i in range(6)]
    p0 = str(tmp_path / "part-00000.parquet")
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), p0, compression="zstd")

    manifest = {"ordered_scientific_content_hash": h}
    dest = str(tmp_path / ".runtime_temp" / "restored.sqlite")
    result = RS.stream_restore_slice_multi(
        destination_path=dest, spec=SPEC, parquet_paths=[p0], manifest=manifest, expected_scientific_hash=h)
    assert result.row_count == 2
    assert result.scientific_content_hash == h
    assert os.path.exists(dest)


def test_restore_multi_across_two_shards_matches_single_file_equivalent(tmp_path):
    rows = [(1, 1777593600001, "ETHUSDT", 3000.0, None, None),
            (2, 1777593600002, "ETHUSDT", 3001.0, 0.0001, 1777600000000),
            (3, 1777593600003, "ETHUSDT", 3002.0, None, None)]
    h = RS._canonical_row_hash(rows)
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("ts_ms", pa.int64()),
                          pa.field("symbol", pa.string()), pa.field("mark_price", pa.float64()),
                          pa.field("funding_rate", pa.float64(), nullable=True),
                          pa.field("next_funding_time_ms", pa.int64(), nullable=True)])

    def _write(path, subset):
        arrays = [pa.array([r[i] for r in subset], type=schema.field(i).type) for i in range(6)]
        pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path, compression="zstd")

    p0 = str(tmp_path / "part-00000.parquet")
    p1 = str(tmp_path / "part-00001.parquet")
    _write(p0, rows[:2])
    _write(p1, rows[2:])

    manifest = {"ordered_scientific_content_hash": h}
    dest = str(tmp_path / ".runtime_temp" / "restored.sqlite")
    result = RS.stream_restore_slice_multi(
        destination_path=dest, spec=SPEC, parquet_paths=[p0, p1], manifest=manifest,
        expected_scientific_hash=h)
    assert result.row_count == 3
    assert result.scientific_content_hash == h


def test_restore_multi_rejects_manifest_mismatch(tmp_path):
    dest = str(tmp_path / ".runtime_temp" / "restored.sqlite")
    manifest = {"ordered_scientific_content_hash": "OTHER"}
    with pytest.raises(RS.RestoreManifestMismatchError):
        RS.stream_restore_slice_multi(
            destination_path=dest, spec=SPEC, parquet_paths=[], manifest=manifest,
            expected_scientific_hash="EXPECTED")


def test_restore_multi_rejects_destination_outside_approved_roots(tmp_path):
    dest = str(tmp_path / "not_approved" / "restored.sqlite")
    manifest = {"ordered_scientific_content_hash": "H"}
    with pytest.raises(RS.RestoreDestinationRejected):
        RS.stream_restore_slice_multi(
            destination_path=dest, spec=SPEC, parquet_paths=[], manifest=manifest, expected_scientific_hash="H")
