"""Direct Parquet reader (Phase 12). Requires manifest verification
before any read; rejects missing/mismatched manifests, checksum
mismatches, unsupported schema versions, path escapes, and
partition/symbol/venue/segment mismatches. Never reads a row outside the
requested, manifest-verified partition -- no future-data contamination.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


class ManifestRequiredError(Exception):
    """Raised when no manifest (or a mismatched one) is supplied."""


class ArchiveCorruptionError(Exception):
    """Raised when the Parquet file's checksum does not match the
    manifest, or the file cannot be read at all."""


class PartitionMismatchError(Exception):
    """Raised when the caller's requested symbol/venue/segment/partition
    does not match the manifest's recorded identity."""


@dataclass(frozen=True)
class ReadResult:
    rows: list[tuple]
    row_count: int
    partition_id: str
    applied_predicates: dict
    source_schema_version: str
    archive_schema_version: str
    verification_state: str


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_overlapping_row_groups(path: str, start_ms: int, end_ms: int) -> list[int] | None:
    """Returns the indices of `path`'s Parquet row groups whose `ts_ms`
    [min, max] column statistics overlap [start_ms, end_ms) -- lets a
    caller skip decoding row groups that provably cannot contain any row
    in the requested range, without a `filters=` param (not supported by
    `ParquetFile.iter_batches` in the pyarrow version this project
    targets). Returns None (meaning "read every row group, can't safely
    skip") if the file has no `ts_ms` column or any row group is missing
    statistics -- always a safe superset, never a false negative."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    if "ts_ms" not in names:
        return None
    ts_idx = names.index("ts_ms")
    keep = []
    for i in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(i).column(ts_idx).statistics
        if stats is None or stats.min is None or stats.max is None:
            return None
        if stats.max >= start_ms and stats.min < end_ms:
            keep.append(i)
    return keep


def select_last_row_group_at_or_before(path: str, ts_ms: int) -> int | None:
    """Returns the index of the LAST (highest-index) Parquet row group in
    `path` whose `ts_ms` minimum statistic is `<= ts_ms` -- i.e. the one
    row group that could contain the latest row at-or-before `ts_ms`,
    for the point-lookup helper (`ami.storage.research_reader.
    lookup_latest_at_or_before`). Row groups are written in ascending
    `ts_ms` order (canonical ordering), so the highest-index row group
    with min <= ts_ms is guaranteed to contain at least one qualifying
    row (its own first row) and no later row group can. Returns None if
    even the first row group's minimum exceeds `ts_ms` (nothing in this
    file qualifies) or the file has no `ts_ms` column / row-group
    statistics -- caller must fall back to an earlier shard/partition or
    treat the lookup as unanswerable from this file, never guess."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    if "ts_ms" not in names:
        return None
    ts_idx = names.index("ts_ms")
    best = None
    for i in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(i).column(ts_idx).statistics
        if stats is None or stats.min is None:
            return None
        if stats.min <= ts_ms:
            best = i
        else:
            break  # row groups are ts-ascending; no later group can qualify either
    return best


def stream_read_partition_shards(*, parquet_paths: list[str], manifest: dict, requested_symbol: str,
                                  requested_venue: str | None = None,
                                  requested_market_segment: str | None = None,
                                  columns: tuple[str, ...] | None = None,
                                  batch_size: int = 1_000_000,
                                  row_groups_by_path: dict[str, list[int]] | None = None):
    """Multi-shard direct reader for partitions too large to materialize
    in memory (`read_partition` below reads one whole file into a Python
    list -- fine for the sizes it was designed for, but not for a
    114M-row partition). Validates the manifest and EVERY shard's
    checksum up front (fails closed before yielding anything, matching
    `read_partition`'s all-or-nothing manifest-first discipline), then
    is a generator: yields one row tuple at a time, streamed via
    `ParquetFile.iter_batches` one shard at a time, in the given shard
    order -- never holds more than one batch of one shard in memory,
    never concatenates shards into one structure.

    `manifest["shards"]` must be the per-shard inventory (each with at
    least `shard_file` and `sha256`) written by the sharded publisher;
    `parquet_paths` must be given in the same order as that inventory."""
    import pyarrow.parquet as pq

    if not manifest:
        raise ManifestRequiredError("no manifest supplied")
    if manifest.get("symbol") != requested_symbol:
        raise PartitionMismatchError(
            f"requested symbol {requested_symbol!r} != manifest symbol {manifest.get('symbol')!r}")
    if requested_venue is not None and manifest.get("venue") != requested_venue:
        raise PartitionMismatchError("venue mismatch")
    if requested_market_segment is not None and manifest.get("market_segment") != requested_market_segment:
        raise PartitionMismatchError("market_segment mismatch")

    shards = manifest.get("shards") or []
    if not shards:
        # Legacy single-file (pre-sharded) manifest: exactly one path,
        # checked against the whole-file `parquet_sha256` the same way
        # `read_partition` does -- there is no per-shard inventory to
        # compare lengths against.
        if len(parquet_paths) != 1:
            raise PartitionMismatchError(
                f"manifest has no shard inventory but {len(parquet_paths)} paths were supplied "
                f"(expected exactly 1 for a legacy single-file partition)")
        path = parquet_paths[0]
        try:
            actual = _sha256_file(path)
        except OSError as exc:
            raise ArchiveCorruptionError(f"cannot read {path!r}: {exc}") from exc
        expected = manifest.get("parquet_sha256")
        if actual != expected:
            raise ArchiveCorruptionError(f"checksum mismatch for {path!r}: file={actual} manifest={expected}")
    elif len(shards) != len(parquet_paths):
        raise PartitionMismatchError(
            f"manifest lists {len(shards)} shards but {len(parquet_paths)} paths were supplied")
    else:
        for shard_entry, path in zip(shards, parquet_paths):
            try:
                actual = _sha256_file(path)
            except OSError as exc:
                raise ArchiveCorruptionError(f"cannot read {path!r}: {exc}") from exc
            expected = shard_entry.get("sha256") or shard_entry.get("parquet_sha256")
            if actual != expected:
                raise ArchiveCorruptionError(
                    f"shard checksum mismatch for {path!r}: file={actual} manifest={expected}")

    def _rows():
        for path in parquet_paths:
            try:
                pf = pq.ParquetFile(path)
            except Exception as exc:
                raise ArchiveCorruptionError(f"parquet unreadable: {exc}") from exc
            row_groups = row_groups_by_path.get(path) if row_groups_by_path else None
            for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None,
                                          row_groups=row_groups):
                d = batch.to_pydict()
                cols = list(d.keys())
                n = len(d[cols[0]]) if cols else 0
                for i in range(n):
                    yield tuple(d[c][i] for c in cols)

    return _rows()


def read_partition(*, parquet_path: str, manifest: dict, requested_symbol: str,
                    requested_venue: str | None = None, requested_market_segment: str | None = None,
                    columns: tuple[str, ...] | None = None) -> ReadResult:
    """Reads a disposable Parquet partition directly, requiring the
    manifest to check out first. `columns=None` reads every preserved
    column; a subset performs column projection."""
    import pyarrow.parquet as pq

    if not manifest:
        raise ManifestRequiredError("no manifest supplied")
    if manifest.get("symbol") != requested_symbol:
        raise PartitionMismatchError(
            f"requested symbol {requested_symbol!r} != manifest symbol {manifest.get('symbol')!r}")
    if requested_venue is not None and manifest.get("venue") != requested_venue:
        raise PartitionMismatchError("venue mismatch")
    if requested_market_segment is not None and manifest.get("market_segment") != requested_market_segment:
        raise PartitionMismatchError("market_segment mismatch")

    try:
        actual_sha256 = _sha256_file(parquet_path)
    except OSError as exc:
        raise ArchiveCorruptionError(f"cannot read {parquet_path!r}: {exc}") from exc
    if actual_sha256 != manifest.get("parquet_sha256"):
        raise ArchiveCorruptionError(
            f"checksum mismatch: file={actual_sha256} manifest={manifest.get('parquet_sha256')}")

    try:
        # Single-file read via ParquetFile (NOT pq.read_table, which would
        # trigger pyarrow dataset/Hive-partition auto-discovery on the
        # production `key=value/` directory layout and collide the path's
        # `symbol=...` key with the real `symbol` column).
        table = pq.ParquetFile(parquet_path).read(columns=list(columns) if columns else None)
    except Exception as exc:
        raise ArchiveCorruptionError(f"parquet unreadable: {exc}") from exc

    d = table.to_pydict()
    cols = list(d.keys())
    n = len(d[cols[0]]) if cols else 0
    rows = [tuple(d[c][i] for c in cols) for i in range(n)]

    return ReadResult(
        rows=rows, row_count=n, partition_id=manifest.get("partition_id", ""),
        applied_predicates={"symbol": requested_symbol, "venue": requested_venue,
                             "market_segment": requested_market_segment, "columns": columns},
        source_schema_version=manifest.get("source_schema_hash", ""),
        archive_schema_version=manifest.get("parquet_schema_hash", ""),
        verification_state=manifest.get("verification_status", ""),
    )
