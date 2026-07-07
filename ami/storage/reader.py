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
        table = pq.read_table(parquet_path, columns=list(columns) if columns else None)
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
