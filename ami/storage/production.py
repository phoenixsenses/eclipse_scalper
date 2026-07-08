"""Production-activation rehearsal (BATCH-STORAGE-ROTATION-RETENTION-
PRODUCTION-ACTIVATION-REHEARSAL-V1). Publishes EXACTLY ONE verified
production archive partition (mark_prices/ETHUSDT/2026-05/v1) under the
approved production root, via same-volume staging + atomic directory
rename, with a persistent filesystem-backed catalog (immutable
partition-local `catalog_entry.json` + rebuildable root
`catalog_index.json`).

This module NEVER deletes/updates/inserts a source row, never exposes a
purge/VACUUM/scheduler capability, and never enables general production
activation. It publishes one partition when the frozen rehearsal
authorization matches, and refuses everything else fail-closed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, asdict

from ami.storage.archive import (build_manifest, build_pyarrow_schema, canonical_row_hash,
                                   fetch_partition_rows)
from ami.storage.partition import PartitionIdentity, build_partition_identity
from ami.storage.policy import REHEARSAL_AUTHORIZATION
from ami.storage.registry import SourceTableSpec, get_table_spec

ARCHIVE_ROOT_FALLBACK = os.path.normpath("D:/eclipse_scalper/data/archives/raw_v1")
CATALOG_CONTRACT_VERSION = "v1"

# Paths a production root may never resolve into.
_FORBIDDEN_ROOT_SUBSTRINGS = (
    "appdata/local/temp", "/data/ami/backups", "/runtime/chrome_user_data_copy",
    "windows/temp", "/temp/",
)
_REQUIRED_ROOT_PREFIX = os.path.normpath("D:/eclipse_scalper/data/archives").replace("\\", "/").lower()
_REPO_ROOT = os.path.normpath("D:/eclipse_scalper").replace("\\", "/").lower()

# Production output file names.
PARQUET_NAME = "part-00000.parquet"
MANIFEST_NAME = "manifest.json"
CATALOG_ENTRY_NAME = "catalog_entry.json"
SUCCESS_NAME = "_SUCCESS"
ROOT_INDEX_NAME = "catalog_index.json"


class ProductionRootRejected(Exception):
    """Fail-closed: raised for any production root that traverses,
    escapes via a reparse point, or resolves into a forbidden location
    (OS temp, backups, source dirs, repo root, browser profiles)."""


class RehearsalAuthorizationDenied(Exception):
    """Raised when the requested partition is not the single frozen
    rehearsal partition (mark_prices/ETHUSDT/2026-05/v1)."""


class ProductionPublicationConflict(Exception):
    """Raised when the final partition path already exists and cannot be
    proven identical-and-accepted -- never overwritten."""


# ---------------------------------------------------------------------------
# Root resolution and validation (Phase 2)
# ---------------------------------------------------------------------------

def resolve_production_root(frozen_root: str | None = None) -> tuple[str, str]:
    """Returns (root, source_of_root). If `frozen_root` is supplied (from
    an accepted artifact), it is validated and used; otherwise the
    operator-approved fallback is used. Either way the root is validated
    fail-closed."""
    if frozen_root:
        root, source = os.path.normpath(frozen_root), "frozen_accepted_artifact"
    else:
        root, source = ARCHIVE_ROOT_FALLBACK, "operator_approved_fallback"
    validate_production_root(root)
    return root, source


def validate_production_root(root: str) -> None:
    norm = os.path.normpath(os.path.abspath(root)).replace("\\", "/")
    lower = norm.lower()
    if ".." in root.replace("\\", "/").split("/"):
        raise ProductionRootRejected(f"path traversal in {root!r}")
    if not lower.startswith(_REQUIRED_ROOT_PREFIX + "/") and lower != _REQUIRED_ROOT_PREFIX:
        raise ProductionRootRejected(
            f"{root!r} must reside within D:/eclipse_scalper/data/archives/")
    if lower == _REPO_ROOT:
        raise ProductionRootRejected("repository root is not a valid production root")
    for bad in _FORBIDDEN_ROOT_SUBSTRINGS:
        if bad in lower:
            raise ProductionRootRejected(f"{root!r} resolves into a forbidden location ({bad})")
    # Reject reparse-point escape anywhere in the existing ancestor chain.
    ancestor = norm
    while True:
        if os.path.exists(ancestor) and _is_reparse_point(ancestor):
            raise ProductionRootRejected(f"reparse point in path chain at {ancestor!r}")
        parent = os.path.dirname(ancestor)
        if parent == ancestor:
            break
        ancestor = parent


def _is_reparse_point(path: str) -> bool:
    try:
        return bool(os.lstat(path).st_file_attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except (OSError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Deterministic path layout (Phase 2)
# ---------------------------------------------------------------------------

def partition_relative_dir(partition: PartitionIdentity, archive_version: str = "v1") -> str:
    return os.path.join(
        f"table={partition.table}", f"venue={partition.venue}",
        f"market_segment={partition.market_segment}", f"symbol={partition.symbol}",
        f"year={partition.utc_year:04d}", f"month={partition.utc_month:02d}",
        f"version={archive_version}")


def final_partition_dir(root: str, partition: PartitionIdentity, archive_version: str = "v1") -> str:
    return os.path.normpath(os.path.join(root, partition_relative_dir(partition, archive_version)))


def staging_partition_dir(root: str, partition: PartitionIdentity, job_identity: str,
                           archive_version: str = "v1") -> str:
    staging_root = os.path.normpath(root + ".staging")
    tag = f"{partition.table}_{partition.symbol}_{partition.utc_year:04d}-{partition.utc_month:02d}_" \
          f"wm{partition.source_watermark_value}_{job_identity}"
    return os.path.normpath(os.path.join(staging_root, tag))


# ---------------------------------------------------------------------------
# Authorization (Phase 3)
# ---------------------------------------------------------------------------

def assert_rehearsal_authorized(*, table: str, symbol: str, venue: str, market_segment: str,
                                 utc_year: int, utc_month: int, archive_version: str) -> None:
    if not REHEARSAL_AUTHORIZATION.is_authorized(
            table=table, symbol=symbol, venue=venue, market_segment=market_segment,
            utc_year=utc_year, utc_month=utc_month, archive_version=archive_version):
        raise RehearsalAuthorizationDenied(
            f"partition ({table}/{symbol}/{venue}/{market_segment}/"
            f"{utc_year}-{utc_month:02d}/{archive_version}) is not the single frozen rehearsal "
            "partition (mark_prices/ETHUSDT/BINANCE_USDM_PERP/PERPETUAL_FUTURES/2026-05/v1)")


# ---------------------------------------------------------------------------
# Catalog entry + root index (Phases 4, 10)
# ---------------------------------------------------------------------------

def build_catalog_entry(*, partition: PartitionIdentity, spec: SourceTableSpec, archive_version: str,
                         row_count: int, archive_relative_path: str, manifest_relative_path: str,
                         parquet_relative_path: str, manifest_sha256: str, parquet_sha256: str,
                         scientific_hash: str, archive_schema_hash: str, source_schema_hash: str,
                         publication_timestamp: str) -> dict:
    return {
        "catalog_contract_version": CATALOG_CONTRACT_VERSION,
        "archive_identity": partition.partition_id,
        "archive_version": archive_version,
        "source_table": partition.table, "symbol": partition.symbol,
        "venue": partition.venue, "market_segment": partition.market_segment,
        "partition_start_ms": partition.partition_start_ms, "partition_end_ms": partition.partition_end_ms,
        "source_watermark_field": partition.source_watermark_field,
        "source_watermark_value": partition.source_watermark_value,
        "archive_relative_path": archive_relative_path,
        "manifest_relative_path": manifest_relative_path,
        "parquet_relative_path": parquet_relative_path,
        "manifest_sha256": manifest_sha256, "parquet_sha256": parquet_sha256,
        "scientific_content_hash": scientific_hash,
        "archive_schema_hash": archive_schema_hash, "source_schema_hash": source_schema_hash,
        "row_count": row_count,
        "production_status": "PRODUCTION_VERIFIED",
        "verification_status": "VERIFIED",
        "source_retention_status": "SOURCE_PRESENT",
        "purge_authorization": "PROHIBITED",
        "research_dependency_status": "BLOCKED",
        "scheduler_status": "DISABLED",
        "publication_timestamp": publication_timestamp,
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_json_atomic(path: str, obj: dict) -> str:
    """Write to `.partial`, then atomically replace. Returns the sha256
    of the exact bytes written."""
    data = json.dumps(obj, indent=2, sort_keys=True, default=str).encode()
    partial = path + ".partial"
    with open(partial, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)
    return _sha256_bytes(data)


def scan_partition_catalog_entries(root: str) -> list[dict]:
    """Walks the production root for every partition-local
    `catalog_entry.json`. Read-only; the root index is derived from these
    immutable entries, never the other way around."""
    entries = []
    if not os.path.isdir(root):
        return entries
    for dirpath, _, filenames in os.walk(root):
        if CATALOG_ENTRY_NAME in filenames:
            with open(os.path.join(dirpath, CATALOG_ENTRY_NAME)) as f:
                entries.append(json.load(f))
    return entries


def build_root_catalog_index(root: str) -> dict:
    """Deterministic, rebuildable index derived purely from
    partition-local entries. Rejects duplicate archive identities and any
    entry not PRODUCTION_VERIFIED / PROHIBITED."""
    entries = scan_partition_catalog_entries(root)
    by_id: dict[str, dict] = {}
    for e in entries:
        aid = e["archive_identity"]
        if e.get("purge_authorization") != "PROHIBITED":
            raise ProductionPublicationConflict(f"catalog entry {aid} has non-PROHIBITED purge_authorization")
        if e.get("verification_status") != "VERIFIED":
            raise ProductionPublicationConflict(f"catalog entry {aid} is not VERIFIED")
        if aid in by_id and by_id[aid]["scientific_content_hash"] != e["scientific_content_hash"]:
            raise ProductionPublicationConflict(f"duplicate archive identity {aid} with differing content")
        by_id[aid] = e
    ordered = sorted(by_id.values(), key=lambda e: (e["source_table"], e["symbol"],
                                                     e["partition_start_ms"], e["archive_version"]))
    index_body = {
        "catalog_contract_version": CATALOG_CONTRACT_VERSION,
        "entry_count": len(ordered),
        "entries": ordered,
    }
    index_body["index_self_hash"] = _sha256_bytes(
        json.dumps(index_body, indent=2, sort_keys=True, default=str).encode())
    return index_body


def publish_root_catalog_index(root: str) -> tuple[str, str]:
    """Builds and atomically publishes the root index. Returns
    (index_path, index_sha256). A failure here never corrupts a
    partition-local archive."""
    index = build_root_catalog_index(root)
    index_path = os.path.join(root, ROOT_INDEX_NAME)
    sha = _write_json_atomic(index_path, index)
    return index_path, sha


# ---------------------------------------------------------------------------
# Publication (Phases 6-9)
# ---------------------------------------------------------------------------

@dataclass
class PublicationResult:
    status: str
    final_partition_dir: str
    parquet_path: str
    manifest_path: str
    catalog_entry_path: str
    success_path: str
    row_count: int
    scientific_hash: str
    parquet_sha256: str
    manifest_sha256: str
    catalog_entry_sha256: str
    reverification_mismatch_count: int


def publish_production_partition(conn, *, root: str, partition: PartitionIdentity, spec: SourceTableSpec,
                                  archive_version: str, job_identity: str,
                                  source_schema_hash: str, export_cutoff: str,
                                  max_output_bytes: int = 1024 ** 3) -> PublicationResult:
    """Full staged publication of one authorized partition. Builds
    everything in a same-volume staging directory, verifies, then
    atomically renames the completed staging directory to the final path
    (which must not pre-exist). Fail-closed at every step."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    final_dir = final_partition_dir(root, partition, archive_version)
    if os.path.exists(final_dir):
        raise ProductionPublicationConflict(
            f"final partition path already exists: {final_dir!r} (never overwritten)")

    staging_dir = staging_partition_dir(root, partition, job_identity, archive_version)
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)  # gate-owned staging identity, safe to reset
    os.makedirs(staging_dir, exist_ok=True)

    # ---- export rows -> partial parquet ----
    rows = fetch_partition_rows(conn, spec, partition)
    scientific_hash = canonical_row_hash(rows)
    schema = build_pyarrow_schema(spec)
    cols = list(zip(*rows)) if rows else [[] for _ in spec.preserved_columns]
    arrays = []
    for i, col in enumerate(spec.preserved_columns):
        pa_type = schema.field(col).type
        values = list(cols[i]) if rows else []
        if pa_type == pa.int64():
            values = [int(v) if v is not None else None for v in values]
        arrays.append(pa.array(values, type=pa_type))
    table_obj = pa.Table.from_arrays(arrays, schema=schema)

    parquet_partial = os.path.join(staging_dir, PARQUET_NAME + ".partial")
    pq.write_table(table_obj, parquet_partial, compression="zstd", use_dictionary=False, write_statistics=True)

    if os.path.getsize(parquet_partial) > max_output_bytes:
        raise ProductionPublicationConflict("parquet exceeds production output cap")

    # validate before rename
    ids = [r[spec.preserved_columns.index(spec.stable_ordering_field)] for r in rows]
    symbols = {r[spec.preserved_columns.index(spec.symbol_field)] for r in rows}
    ts_idx = spec.preserved_columns.index(spec.partition_ts_field)
    assert len(ids) - len(set(ids)) == 0, "duplicate ids"
    assert symbols <= {partition.symbol}, "unexpected symbol"
    assert all(partition.partition_start_ms <= r[ts_idx] < partition.partition_end_ms for r in rows)
    assert all(i <= partition.source_watermark_value for i in ids)
    reread = pq.ParquetFile(parquet_partial).read().to_pydict()
    reread_rows = [tuple(reread[c][i] for c in spec.preserved_columns) for i in range(len(reread["id"]))]
    assert canonical_row_hash(reread_rows) == scientific_hash, "parquet scientific parity failed"

    parquet_final = os.path.join(staging_dir, PARQUET_NAME)
    os.replace(parquet_partial, parquet_final)
    parquet_sha = _sha256_file(parquet_final)
    parquet_schema_hash = hashlib.sha256(str(schema).encode()).hexdigest()

    # ---- manifest ----
    manifest = build_manifest(
        spec=spec, partition=partition, row_count=len(rows), scientific_hash=scientific_hash,
        parquet_path=os.path.join(partition_relative_dir(partition, archive_version), PARQUET_NAME),
        parquet_size=os.path.getsize(parquet_final), parquet_sha256=parquet_sha,
        source_schema_hash=source_schema_hash, parquet_schema_hash=parquet_schema_hash,
        unresolved_gap_count=0, export_cutoff=export_cutoff,
        publication_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        verification_status="VERIFIED", dry_run_identity=job_identity,
        production_status="PRODUCTION_VERIFIED")
    manifest["partition_id"] = partition.partition_id
    manifest_path = os.path.join(staging_dir, MANIFEST_NAME)
    manifest_sha = _write_json_atomic(manifest_path, manifest)

    # ---- catalog entry ----
    catalog_entry = build_catalog_entry(
        partition=partition, spec=spec, archive_version=archive_version, row_count=len(rows),
        archive_relative_path=partition_relative_dir(partition, archive_version),
        manifest_relative_path=os.path.join(partition_relative_dir(partition, archive_version), MANIFEST_NAME),
        parquet_relative_path=os.path.join(partition_relative_dir(partition, archive_version), PARQUET_NAME),
        manifest_sha256=manifest_sha, parquet_sha256=parquet_sha, scientific_hash=scientific_hash,
        archive_schema_hash=parquet_schema_hash, source_schema_hash=source_schema_hash,
        publication_timestamp=dt.datetime.now(dt.timezone.utc).isoformat())
    catalog_entry_path = os.path.join(staging_dir, CATALOG_ENTRY_NAME)
    catalog_entry_sha = _write_json_atomic(catalog_entry_path, catalog_entry)

    # ---- _SUCCESS ----
    success_partial = os.path.join(staging_dir, SUCCESS_NAME + ".partial")
    with open(success_partial, "w") as f:
        f.write(f"{partition.partition_id}\n")
    os.replace(success_partial, os.path.join(staging_dir, SUCCESS_NAME))

    # ---- validate complete staging dir, then atomic directory publication ----
    for required in (PARQUET_NAME, MANIFEST_NAME, CATALOG_ENTRY_NAME, SUCCESS_NAME):
        if not os.path.exists(os.path.join(staging_dir, required)):
            raise ProductionPublicationConflict(f"staging missing {required}")
    if any(fn.endswith(".partial") for fn in os.listdir(staging_dir)):
        raise ProductionPublicationConflict("staging still has a .partial file")
    if os.path.exists(final_dir):
        raise ProductionPublicationConflict(f"final path appeared during staging: {final_dir!r}")

    os.makedirs(os.path.dirname(final_dir), exist_ok=True)
    os.rename(staging_dir, final_dir)  # atomic same-volume directory rename

    # ---- post-publication re-verification (Phase 9) ----
    mismatches = reverify_published_partition(final_dir, partition)

    return PublicationResult(
        status="PUBLISHED", final_partition_dir=final_dir,
        parquet_path=os.path.join(final_dir, PARQUET_NAME),
        manifest_path=os.path.join(final_dir, MANIFEST_NAME),
        catalog_entry_path=os.path.join(final_dir, CATALOG_ENTRY_NAME),
        success_path=os.path.join(final_dir, SUCCESS_NAME),
        row_count=len(rows), scientific_hash=scientific_hash, parquet_sha256=parquet_sha,
        manifest_sha256=manifest_sha, catalog_entry_sha256=catalog_entry_sha,
        reverification_mismatch_count=mismatches)


def reverify_published_partition(final_dir: str, partition: PartitionIdentity) -> int:
    """Independently re-verifies a published partition from its final
    path. Returns the mismatch count (0 = fully verified)."""
    import pyarrow.parquet as pq
    mismatches = 0
    with open(os.path.join(final_dir, MANIFEST_NAME)) as f:
        manifest = json.load(f)
    with open(os.path.join(final_dir, CATALOG_ENTRY_NAME)) as f:
        entry = json.load(f)
    parquet_path = os.path.join(final_dir, PARQUET_NAME)

    if not os.path.exists(os.path.join(final_dir, SUCCESS_NAME)):
        mismatches += 1
    if _sha256_file(parquet_path) != manifest.get("parquet_sha256"):
        mismatches += 1
    if manifest.get("parquet_sha256") != entry.get("parquet_sha256"):
        mismatches += 1
    if manifest.get("ordered_scientific_content_hash") != entry.get("scientific_content_hash"):
        mismatches += 1
    if manifest.get("partition_id") != partition.partition_id:
        mismatches += 1
    if manifest.get("production_status") != "PRODUCTION_VERIFIED":
        mismatches += 1
    if manifest.get("purge_authorization") != "PROHIBITED":
        mismatches += 1
    if entry.get("purge_authorization") != "PROHIBITED":
        mismatches += 1
    if entry.get("source_retention_status") != "SOURCE_PRESENT":
        mismatches += 1
    # re-read rows and re-hash
    reread = pq.ParquetFile(parquet_path).read().to_pydict()
    cols = list(reread.keys())
    n = len(reread[cols[0]]) if cols else 0
    rows = [tuple(reread[c][i] for c in cols) for i in range(n)]
    if canonical_row_hash(rows) != manifest.get("ordered_scientific_content_hash"):
        mismatches += 1
    return mismatches


# ---------------------------------------------------------------------------
# Idempotency (Phase 13)
# ---------------------------------------------------------------------------

def check_idempotent_rerun(final_dir: str, *, current_scientific_hash: str,
                            current_watermark: int) -> str:
    """Given an already-published partition, determines the correct
    rerun disposition without republishing anything."""
    if not os.path.isdir(final_dir):
        return "NO_EXISTING_ARCHIVE"
    with open(os.path.join(final_dir, MANIFEST_NAME)) as f:
        manifest = json.load(f)
    existing_hash = manifest.get("ordered_scientific_content_hash")
    existing_wm = manifest.get("source_watermark_value")
    if existing_hash == current_scientific_hash and existing_wm == current_watermark:
        return "NOOP_IDENTICAL_PRODUCTION_ARCHIVE"
    if existing_wm != current_watermark or existing_hash != current_scientific_hash:
        return "SOURCE_CHANGED_NEW_VERSION_REQUIRED"
    return "CONFLICT_FAIL_CLOSED"


# ---------------------------------------------------------------------------
# Orchestration (used by both the CLI and tests) -- checks idempotency
# FIRST, before ever attempting an export, so a second run never crashes
# into ProductionPublicationConflict; it returns the correct disposition
# instead.
# ---------------------------------------------------------------------------

def run_production_activation_rehearsal(conn, *, root: str, root_source: str) -> dict:
    """The complete rehearsal orchestration: authorize -> capture fresh
    watermark -> check idempotency -> publish only if no existing
    identical archive -> publish/rebuild the root index. `conn` must
    already be an open, read-only, authorizer-guarded connection (see
    `ami.storage.source_access.open_read_only`)."""
    auth = REHEARSAL_AUTHORIZATION
    assert_rehearsal_authorized(
        table=auth.table, symbol=auth.symbol, venue=auth.venue, market_segment=auth.market_segment,
        utc_year=auth.utc_year, utc_month=auth.utc_month, archive_version=auth.archive_version)
    spec = get_table_spec(auth.table)

    watermark = conn.execute(
        f"SELECT MAX(id) FROM {spec.table} WHERE symbol=? AND ts_ms>=1777593600000 AND ts_ms<1780272000000",
        (auth.symbol,)).fetchone()[0]
    partition = build_partition_identity(
        table=auth.table, symbol=auth.symbol, utc_year=auth.utc_year, utc_month=auth.utc_month,
        source_watermark_value=watermark)

    rows = fetch_partition_rows(conn, spec, partition)
    current_hash = canonical_row_hash(rows)
    final_dir = final_partition_dir(root, partition, auth.archive_version)
    disposition = check_idempotent_rerun(final_dir, current_scientific_hash=current_hash,
                                          current_watermark=watermark)

    if disposition == "NOOP_IDENTICAL_PRODUCTION_ARCHIVE":
        mismatches = reverify_published_partition(final_dir, partition)
        return {"status": disposition, "final_partition_dir": final_dir,
                "reverification_mismatch_count": mismatches, "row_count": len(rows),
                "scientific_hash": current_hash, "root": root, "root_source": root_source}

    if disposition == "SOURCE_CHANGED_NEW_VERSION_REQUIRED":
        with open(os.path.join(final_dir, MANIFEST_NAME)) as f:
            existing = json.load(f)
        return {"status": disposition, "final_partition_dir": final_dir,
                "row_count_delta": len(rows) - existing.get("row_count", 0),
                "existing_row_count": existing.get("row_count"), "current_row_count": len(rows),
                "existing_watermark": existing.get("source_watermark_value"),
                "current_watermark": watermark, "root": root, "root_source": root_source}

    if disposition == "CONFLICT_FAIL_CLOSED":
        raise ProductionPublicationConflict(
            f"existing archive at {final_dir!r} conflicts and cannot be safely reconciled")

    # NO_EXISTING_ARCHIVE -- first publication
    source_schema_sql = conn.execute(
        f"SELECT sql FROM sqlite_master WHERE name='{spec.table}'").fetchone()[0]
    source_schema_hash = hashlib.sha256(source_schema_sql.encode()).hexdigest()
    job_identity = "PROD-REHEARSAL-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    result = publish_production_partition(
        conn, root=root, partition=partition, spec=spec, archive_version=auth.archive_version,
        job_identity=job_identity, source_schema_hash=source_schema_hash,
        export_cutoff=dt.datetime.now(dt.timezone.utc).isoformat())
    index_path, index_sha = publish_root_catalog_index(root)
    return {
        "status": result.status, "root": root, "root_source": root_source,
        "final_partition_dir": result.final_partition_dir, "row_count": result.row_count,
        "scientific_hash": result.scientific_hash, "parquet_sha256": result.parquet_sha256,
        "manifest_sha256": result.manifest_sha256, "catalog_entry_sha256": result.catalog_entry_sha256,
        "reverification_mismatch_count": result.reverification_mismatch_count,
        "root_index_path": index_path, "root_index_sha256": index_sha,
    }
