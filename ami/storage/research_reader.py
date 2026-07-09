"""Unified, read-only research reader (BATCH-STORAGE-ROTATION-RETENTION-
RESEARCH-READER-INTEGRATION-V1).

Gives research code one contract -- table/venue/market_segment/symbol/
UTC [start_ms, end_ms)/columns/filters -- that transparently resolves to
whichever physical source(s) actually hold that data: the live
production SQLite (hot/recent), the production archive's Parquet
partitions (cold/closed months, single-file or sharded), or both.
Research code never has to know which.

Two-phase API, mirroring every other module in this package:

    plan = plan_read(root, table=..., symbol=..., start_ms=..., end_ms=...)
    result = execute_read(plan, columns=..., filters=..., batch_size=...)
    for batch in result.iter_batches():
        ...
    provenance = result.provenance  # attach to research lineage, don't backfill old records with it

Absolute guarantees:
  * Never writes to the source SQLite (opens via
    `ami.storage.source_access.open_read_only`, the same authorizer-
    enforced read-only connection every other reader in this package
    uses).
  * Never mutates a manifest, catalog entry, authorization receipt, or
    the root catalog index -- `plan_read`/`execute_read` only ever read
    files that already exist.
  * Never re-runs the full scientific reverify (that stays exclusively
    behind `ami.storage.reverify_guard.run_guarded_reverify`, an
    explicit, separate operator action) -- trust here is a handful of
    cheap file-existence and self-hash checks (`_SUCCESS`, manifest
    self-hash, receipt self-hash, shard existence), not a full re-read
    of the Parquet content.
  * Never materializes a full partition or a full query result in
    memory -- both the SQLite side and the archive side stream in a
    single, deterministic `batch_size`.
  * Archive-covered time is never also read from SQLite (no silent
    double-count): any [start_ms, end_ms) sub-range covered by a
    verified archive partition is served ONLY from that partition; the
    remainder (if any) is served from SQLite. This is the one and only
    overlap-resolution policy this module implements; anything that
    doesn't fit it (e.g. two archive partitions overlapping each other,
    which should never happen by construction) raises rather than
    guessing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ami.storage import production as PR
from ami.storage import source_access as SRC
from ami.storage.reader import ArchiveCorruptionError, select_overlapping_row_groups, stream_read_partition_shards
from ami.storage.registry import get_table_spec, SourceTableSpec

ALLOWED_TABLES = ("agg_trades", "book_ticker", "mark_prices")
CANONICAL_ORDERING = "(ts_ms ASC, id ASC)"
DEFAULT_BATCH_SIZE = 1_000_000


class UnsupportedTableError(Exception):
    """Raised for any table outside the allowlist -- this module is an
    allowlist, not a discovery mechanism, matching every other reader in
    this package."""


class OverlapDetectedError(Exception):
    """Raised if two archive partitions matching the same table/symbol/
    venue/segment overlap in time -- should never happen given the
    publisher's one-partition-per-closed-month contract; fail rather
    than guess a resolution."""


class ArchiveTrustError(Exception):
    """Raised when a matched archive partition fails a cheap trust check
    (missing _SUCCESS, manifest self-hash mismatch, receipt self-hash
    mismatch, or a missing shard/parquet file). Distinct from a full
    scientific-content mismatch, which only `run_guarded_reverify`
    checks -- this is a much cheaper, much more limited "is this archive
    at least structurally intact and unmodified" check performed on
    every plan, not just on explicit demand."""


class SchemaMismatchError(Exception):
    """Raised when the requested columns are not a subset of the
    table's registered `preserved_columns`."""


class MaxRowsExceededError(Exception):
    """Raised by the bounded pandas-compatibility helper when a result
    would exceed the caller's explicit `max_rows` cap -- never silently
    truncates or silently loads an unbounded frame."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class ArchiveSegment:
    """One archive partition (or the covered slice of it) contributing
    to a read plan."""
    catalog_entry: dict
    manifest: dict
    shard_paths: tuple[str, ...]
    segment_start_ms: int
    segment_end_ms: int

    @property
    def archive_identity(self) -> str:
        return self.catalog_entry["archive_identity"]

    @property
    def is_full_partition(self) -> bool:
        return (self.segment_start_ms == self.catalog_entry["partition_start_ms"]
                and self.segment_end_ms == self.catalog_entry["partition_end_ms"])


@dataclass(frozen=True)
class ReadPlan:
    table: str
    venue: str
    market_segment: str
    symbol: str
    start_ms: int
    end_ms: int
    mode: str  # "EMPTY" | "SQLITE_ONLY" | "ARCHIVE_ONLY" | "HYBRID"
    archive_segments: tuple[ArchiveSegment, ...]
    sqlite_ranges: tuple[tuple[int, int], ...]
    ordering: str
    estimated_row_count: int | None
    overlap_status: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _root_catalog_entries(root: str) -> list[dict]:
    idx_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    if not os.path.exists(idx_path):
        return []
    idx = _read_json(idx_path)
    return idx.get("entries") or []


def _manifest_shard_paths(final_dir: str, manifest: dict) -> list[str]:
    shards = manifest.get("shards")
    if shards:
        ordered = sorted(shards, key=lambda s: s["shard_index"])
        return [os.path.join(final_dir, s["shard_file"]) for s in ordered]
    # Single-file (legacy, pre-sharded) manifest: mark_prices/agg_trades.
    parquet_name = os.path.basename(manifest["parquet_path"])
    return [os.path.join(final_dir, parquet_name)]


def _verify_archive_trust(root: str, entry: dict) -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
    """Cheap structural trust check -- NOT a full scientific reverify.
    Returns (manifest, shard_paths, warnings). Raises ArchiveTrustError
    fail-closed on anything that doesn't check out."""
    final_dir = os.path.join(root, entry["archive_relative_path"])
    warnings: list[str] = []

    success_path = os.path.join(final_dir, PR.SUCCESS_NAME)
    if not os.path.exists(success_path):
        raise ArchiveTrustError(f"_SUCCESS missing for archive {entry['archive_identity']!r} at {final_dir!r}")

    manifest_path = os.path.join(final_dir, PR.MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise ArchiveTrustError(f"manifest.json missing for archive {entry['archive_identity']!r}")
    actual_manifest_hash = _sha256_file(manifest_path)
    expected_manifest_hash = entry.get("manifest_sha256")
    if actual_manifest_hash != expected_manifest_hash:
        raise ArchiveTrustError(
            f"manifest self-hash mismatch for archive {entry['archive_identity']!r}: "
            f"file={actual_manifest_hash} catalog_entry={expected_manifest_hash}")
    manifest = _read_json(manifest_path)

    receipt_path = os.path.join(final_dir, "authorization_receipt.json")
    if os.path.exists(receipt_path):
        receipt = _read_json(receipt_path)
        body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        expected_self_hash = hashlib.sha256(
            json.dumps(body, indent=2, sort_keys=True, default=str).encode()).hexdigest()
        if receipt.get("receipt_sha256") != expected_self_hash:
            raise ArchiveTrustError(f"receipt self-hash mismatch for archive {entry['archive_identity']!r}")
        if entry.get("authorization_receipt_sha256") != _sha256_file(receipt_path):
            raise ArchiveTrustError(
                f"catalog_entry authorization_receipt_sha256 mismatch for archive {entry['archive_identity']!r}")
    else:
        warnings.append(f"archive {entry['archive_identity']!r} has no authorization_receipt.json "
                         f"(pre-activation-era or rehearsal entry)")

    shard_paths = _manifest_shard_paths(final_dir, manifest)
    for p in shard_paths:
        if not os.path.exists(p):
            raise ArchiveTrustError(f"archive {entry['archive_identity']!r} is missing file {p!r}")

    if entry.get("production_status") != "PRODUCTION_VERIFIED":
        raise ArchiveTrustError(
            f"archive {entry['archive_identity']!r} production_status is "
            f"{entry.get('production_status')!r}, not PRODUCTION_VERIFIED")
    if entry.get("purge_authorization") != "PROHIBITED":
        raise ArchiveTrustError(f"archive {entry['archive_identity']!r} purge_authorization is not PROHIBITED")

    return manifest, tuple(shard_paths), tuple(warnings)


def _merge_and_check_overlaps(entries: list[dict]) -> None:
    ordered = sorted(entries, key=lambda e: e["partition_start_ms"])
    for a, b in zip(ordered, ordered[1:]):
        if a["partition_end_ms"] > b["partition_start_ms"]:
            raise OverlapDetectedError(
                f"archive partitions {a['archive_identity']!r} and {b['archive_identity']!r} "
                f"overlap in time -- this should never happen given the one-partition-per-closed-"
                f"month contract; refusing to guess a resolution")


def _compute_gaps(start_ms: int, end_ms: int, covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """[start_ms, end_ms) minus the union of `covered` ranges (already
    checked non-overlapping and sorted by _merge_and_check_overlaps)."""
    gaps = []
    cursor = start_ms
    for c_start, c_end in sorted(covered):
        seg_start = max(c_start, start_ms)
        seg_end = min(c_end, end_ms)
        if seg_start >= seg_end:
            continue
        if cursor < seg_start:
            gaps.append((cursor, seg_start))
        cursor = max(cursor, seg_end)
    if cursor < end_ms:
        gaps.append((cursor, end_ms))
    return gaps


def plan_read(root: str, *, table: str, symbol: str, start_ms: int, end_ms: int,
              venue: str | None = None, market_segment: str | None = None) -> ReadPlan:
    """Read-only query planner. Reads only `<root>/catalog_index.json`
    and, for each matching archive entry, its own manifest/receipt/
    _SUCCESS/shard files -- never the source SQLite, never any Parquet
    row data. Safe to call as often as needed; never mutates anything."""
    if table not in ALLOWED_TABLES:
        raise UnsupportedTableError(f"{table!r} is not in the research-reader allowlist {ALLOWED_TABLES}")
    spec = get_table_spec(table)
    venue = venue or spec.venue
    market_segment = market_segment or spec.market_segment
    ordering = CANONICAL_ORDERING

    if end_ms <= start_ms:
        return ReadPlan(table=table, venue=venue, market_segment=market_segment, symbol=symbol,
                         start_ms=start_ms, end_ms=end_ms, mode="EMPTY", archive_segments=(),
                         sqlite_ranges=(), ordering=ordering, estimated_row_count=0,
                         overlap_status="NO_OVERLAP", warnings=("end_ms <= start_ms",))

    entries = _root_catalog_entries(root)
    matching = [e for e in entries if e["source_table"] == table and e["symbol"] == symbol
                and e["venue"] == venue and e["market_segment"] == market_segment
                and e["partition_start_ms"] < end_ms and e["partition_end_ms"] > start_ms]
    _merge_and_check_overlaps(matching)
    matching.sort(key=lambda e: e["partition_start_ms"])

    segments = []
    warnings: list[str] = []
    covered_ranges = []
    estimated_row_count = 0
    for e in matching:
        manifest, shard_paths, trust_warnings = _verify_archive_trust(root, e)
        warnings.extend(trust_warnings)
        seg_start = max(start_ms, e["partition_start_ms"])
        seg_end = min(end_ms, e["partition_end_ms"])
        segments.append(ArchiveSegment(catalog_entry=e, manifest=manifest, shard_paths=shard_paths,
                                        segment_start_ms=seg_start, segment_end_ms=seg_end))
        covered_ranges.append((e["partition_start_ms"], e["partition_end_ms"]))
        if seg_start == e["partition_start_ms"] and seg_end == e["partition_end_ms"]:
            estimated_row_count += e["row_count"]
        else:
            warnings.append(
                f"archive {e['archive_identity']!r} partially overlaps the requested range -- "
                f"its row_count is a whole-partition figure, not scoped to the overlap; "
                f"estimated_row_count omits it, actual count is only known after streaming")

    gaps = _compute_gaps(start_ms, end_ms, covered_ranges)
    sqlite_ranges = tuple(gaps)

    if segments and sqlite_ranges:
        mode = "HYBRID"
    elif segments:
        mode = "ARCHIVE_ONLY"
    elif sqlite_ranges:
        mode = "SQLITE_ONLY"
    else:
        mode = "EMPTY"

    return ReadPlan(table=table, venue=venue, market_segment=market_segment, symbol=symbol,
                     start_ms=start_ms, end_ms=end_ms, mode=mode,
                     archive_segments=tuple(segments), sqlite_ranges=sqlite_ranges,
                     ordering=ordering, estimated_row_count=estimated_row_count or None,
                     overlap_status="NO_OVERLAP", warnings=tuple(warnings))


def _validate_columns(spec: SourceTableSpec, columns: tuple[str, ...] | None) -> tuple[str, ...]:
    if columns is None:
        return tuple(spec.preserved_columns)
    unknown = [c for c in columns if c not in spec.preserved_columns]
    if unknown:
        raise SchemaMismatchError(f"columns {unknown} are not in {spec.table}'s preserved_columns "
                                   f"{spec.preserved_columns}")
    return tuple(columns)


def _apply_filters(row: tuple, columns: tuple[str, ...], filters: tuple[tuple[str, str, object], ...]) -> bool:
    idx = {c: i for i, c in enumerate(columns)}
    for col, op, value in filters:
        v = row[idx[col]]
        if op == "==" and not (v == value):
            return False
        if op == "!=" and not (v != value):
            return False
        if op == ">=" and not (v >= value):
            return False
        if op == "<=" and not (v <= value):
            return False
        if op == ">" and not (v > value):
            return False
        if op == "<" and not (v < value):
            return False
    return True


@dataclass
class ReadResult:
    plan: ReadPlan
    columns: tuple[str, ...]
    filters: tuple[tuple[str, str, object], ...]
    batch_size: int
    _row_iter_factory: object  # callable() -> Iterator[tuple[tuple, ...]] of row-batches
    _actual_row_count: list = field(default_factory=lambda: [0])

    def iter_batches(self):
        """Yields tuples of rows (each a plain Python tuple in
        `self.columns` order), each batch at most `self.batch_size` rows.
        Streams; never materializes the full result."""
        for batch in self._row_iter_factory():
            self._actual_row_count[0] += len(batch)
            yield batch

    def iter_rows(self):
        for batch in self.iter_batches():
            yield from batch

    def to_bounded_list(self, *, max_rows: int) -> list[tuple]:
        """Bounded materialization helper -- raises MaxRowsExceededError
        rather than silently loading an unbounded result. For pandas
        compatibility: `pandas.DataFrame(result.to_bounded_list(max_rows=...),
        columns=result.columns)`."""
        out: list[tuple] = []
        for batch in self.iter_batches():
            out.extend(batch)
            if len(out) > max_rows:
                raise MaxRowsExceededError(f"result exceeded max_rows={max_rows} "
                                            f"(at least {len(out)} rows and counting)")
        return out

    @property
    def provenance(self) -> dict:
        """Machine-readable provenance for this specific read. Attach to
        a research result's OWN lineage record; never used to retroactively
        modify a previously-recorded result."""
        archive_prov = []
        for seg in self.plan.archive_segments:
            archive_prov.append({
                "archive_identity": seg.archive_identity,
                "archive_relative_path": seg.catalog_entry["archive_relative_path"],
                "manifest_sha256": seg.catalog_entry.get("manifest_sha256"),
                "scientific_content_hash": seg.catalog_entry.get("scientific_content_hash"),
                "shard_count": len(seg.shard_paths),
                "segment_start_ms": seg.segment_start_ms, "segment_end_ms": seg.segment_end_ms,
                "is_full_partition": seg.is_full_partition,
                "source_watermark_field": seg.catalog_entry.get("source_watermark_field"),
                "source_watermark_value": seg.catalog_entry.get("source_watermark_value"),
                "trust_check": "structural (( _SUCCESS, manifest self-hash, receipt self-hash, "
                               "shard existence )) -- NOT a full scientific reverify",
            })
        return {
            "source_type": self.plan.mode,
            "table": self.plan.table, "venue": self.plan.venue,
            "market_segment": self.plan.market_segment, "symbol": self.plan.symbol,
            "requested_start_ms": self.plan.start_ms, "requested_end_ms": self.plan.end_ms,
            "ordering": self.plan.ordering,
            "columns": list(self.columns), "filters": list(self.filters),
            "batch_size": self.batch_size,
            "archive_segments": archive_prov,
            "sqlite_ranges": [list(r) for r in self.plan.sqlite_ranges],
            "row_count": self._actual_row_count[0],
            "overlap_status": self.plan.overlap_status,
            "plan_warnings": list(self.plan.warnings),
        }


def _select_overlapping_shards(manifest: dict, shard_paths: tuple[str, ...],
                                start_ms: int, end_ms: int) -> tuple[dict, list[str]]:
    """Trims `manifest`/`shard_paths` down to only the shards whose
    [min_ts, max_ts] overlaps [start_ms, end_ms) -- without this, a
    narrow time-bounded research query against a large multi-shard
    partition (e.g. book_ticker's 12 x 10M-row shards) would stream and
    checksum-validate every shard just to serve a handful of rows.
    Legacy single-file manifests (no `shards` list) pass through
    unchanged -- there is only one file, nothing to select between."""
    shards = manifest.get("shards")
    if not shards:
        return manifest, list(shard_paths)
    ordered = sorted(shards, key=lambda s: s["shard_index"])
    keep = [(s, p) for s, p in zip(ordered, shard_paths) if s["max_ts"] >= start_ms and s["min_ts"] < end_ms]
    if not keep:
        return manifest, []
    kept_shards, kept_paths = zip(*keep)
    trimmed_manifest = dict(manifest)
    trimmed_manifest["shards"] = list(kept_shards)
    return trimmed_manifest, list(kept_paths)


def _stream_archive_segment(seg: ArchiveSegment, fetch_columns: tuple[str, ...],
                             output_columns: tuple[str, ...], ts_idx: int, keep_idx: list[int],
                             filters: tuple[tuple[str, str, object], ...], batch_size: int):
    manifest, shard_paths = _select_overlapping_shards(seg.manifest, seg.shard_paths,
                                                         seg.segment_start_ms, seg.segment_end_ms)
    row_groups_by_path = {
        p: select_overlapping_row_groups(p, seg.segment_start_ms, seg.segment_end_ms) for p in shard_paths
    }
    row_iter = stream_read_partition_shards(
        parquet_paths=shard_paths, manifest=manifest, requested_symbol=seg.catalog_entry["symbol"],
        requested_venue=seg.catalog_entry["venue"], requested_market_segment=seg.catalog_entry["market_segment"],
        columns=fetch_columns, batch_size=batch_size, row_groups_by_path=row_groups_by_path)
    batch: list[tuple] = []
    for row in row_iter:
        if not (seg.segment_start_ms <= row[ts_idx] < seg.segment_end_ms):
            continue  # outside the requested overlap slice of this partition
        if filters and not _apply_filters(row, fetch_columns, filters):
            continue
        batch.append(tuple(row[i] for i in keep_idx))
        if len(batch) >= batch_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def _stream_sqlite_range(conn: sqlite3.Connection, spec: SourceTableSpec, symbol: str,
                          start_ms: int, end_ms: int, fetch_columns: tuple[str, ...],
                          keep_idx: list[int], filters: tuple[tuple[str, str, object], ...],
                          batch_size: int):
    cols_sql = ",".join(fetch_columns)
    # ORDER BY ts_ms ASC, id ASC against the (symbol, ts_ms) index needs no
    # sort buffer (proven via EXPLAIN QUERY PLAN during the book_ticker
    # recovery gate) -- the same pattern ami.storage.sharded_archive uses.
    cur = conn.execute(
        f"SELECT {cols_sql} FROM {spec.table} WHERE {spec.symbol_field}=? AND "
        f"{spec.partition_ts_field}>=? AND {spec.partition_ts_field}<? "
        f"ORDER BY {spec.partition_ts_field} ASC, {spec.stable_ordering_field} ASC",
        (symbol, start_ms, end_ms))
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        if filters:
            rows = [r for r in rows if _apply_filters(r, fetch_columns, filters)]
        if rows:
            yield tuple(tuple(r[i] for i in keep_idx) for r in rows)


def execute_read(plan: ReadPlan, *, columns: tuple[str, ...] | None = None,
                  filters: tuple[tuple[str, str, object], ...] = (),
                  batch_size: int = DEFAULT_BATCH_SIZE,
                  source_db_path: str | Path | None = None) -> ReadResult:
    """Executes a `ReadPlan`. Returns a `ReadResult` whose `.iter_batches()`
    streams rows in canonical `(ts_ms ASC, id ASC)` order: archive
    segments first (in chronological month order -- always the older,
    closed-month data), then the SQLite range(s) last (always the more
    recent, live tail) -- correct global ordering without an interleaved
    merge-sort, since archived months are, by construction, always
    strictly older than anything still live in SQLite.

    `filters`: tuple of `(column, op, value)` with op in
    `{"==","!=" ,">=","<=",">","<"}`, applied per-batch after column
    projection -- streaming and memory-bounded, but a row-level check
    (not a Parquet row-group-statistics skip), so it does not reduce I/O
    the way true storage-level pushdown would; documented here rather
    than overclaimed."""
    spec = get_table_spec(plan.table)
    resolved_columns = _validate_columns(spec, columns)
    if filters:
        unknown = [f[0] for f in filters if f[0] not in resolved_columns]
        if unknown:
            raise SchemaMismatchError(f"filter columns {unknown} are not in the projected columns "
                                       f"{resolved_columns}")

    # ts_ms is always fetched internally (needed to trim a partially-
    # overlapping archive segment to the requested range) even if the
    # caller didn't request it in the output -- trimmed back off below.
    fetch_columns = resolved_columns if "ts_ms" in resolved_columns else resolved_columns + ("ts_ms",)
    ts_idx = fetch_columns.index("ts_ms")
    keep_idx = [fetch_columns.index(c) for c in resolved_columns]

    def factory():
        for seg in plan.archive_segments:
            yield from _stream_archive_segment(seg, fetch_columns, resolved_columns, ts_idx, keep_idx,
                                                filters, batch_size)
        if plan.sqlite_ranges:
            conn, log = SRC.open_read_only(source_db_path or SRC.DEFAULT_SOURCE_PATH)
            try:
                for start_ms, end_ms in plan.sqlite_ranges:
                    yield from _stream_sqlite_range(conn, spec, plan.symbol, start_ms, end_ms,
                                                     fetch_columns, keep_idx, filters, batch_size)
            finally:
                SRC.assert_read_only_session_clean(log)
                conn.close()

    return ReadResult(plan=plan, columns=resolved_columns, filters=tuple(filters),
                       batch_size=batch_size, _row_iter_factory=factory)
