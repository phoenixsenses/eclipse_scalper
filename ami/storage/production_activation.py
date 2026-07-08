"""Manual-only production archive activation (BATCH-STORAGE-ROTATION-
RETENTION-PRODUCTION-ARCHIVE-ACTIVATION-V1). Adds, on top of the accepted
production rehearsal (`ami.storage.production`, commit 7b46b326):

  * an immutable, exact-partition-bound authorization receipt (Phase 3-4)
  * a single-writer filesystem catalog lock (Phase 5)
  * concurrency-safe, lock-protected root-index rebuild (Phase 6)
  * deterministic candidate selection (Phase 8)
  * six-file authorized production publication (receipt-required, Phase 12)

Nothing here deletes/updates/inserts a source row, exposes purge/VACUUM/
scheduler, or enables general unrestricted activation. It publishes one
authorized partition per validated receipt and refuses everything else
fail-closed.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import time
from dataclasses import dataclass, asdict

from ami.storage import production as PR
from ami.storage import reverify_guard as RG
from ami.storage import sharded_archive as SHA
from ami.storage.archive import (build_manifest, build_pyarrow_schema, stream_export_to_parquet,
                                   stream_hash_parquet)
from ami.storage.partition import PartitionIdentity, build_partition_identity, PartitionValidationError
from ami.storage.registry import SourceTableSpec, get_table_spec

AUTHORIZATION_CONTRACT_VERSION = "v1"
LOCK_CONTRACT_VERSION = "v1"
AUTHORIZATION_RECEIPT_NAME = "authorization_receipt.json"
CATALOG_LOCK_NAME = "raw_v1.catalog.lock"
CANDIDATE_SYMBOL_UNIVERSE = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

# Per-partition / combined resource limits (Phase 10).
MIN_FREE_BYTES_BEFORE_START = 500 * 1024 ** 3
MAX_SOURCE_BYTES_PER_PARTITION = 100 * 1024 ** 3
MAX_PARQUET_BYTES_PER_PARTITION = 30 * 1024 ** 3
MAX_STAGING_PLUS_VERIFICATION_BYTES = 40 * 1024 ** 3


class AuthorizationReceiptRejected(Exception):
    """Fail-closed: a receipt that does not exactly match the live plan
    (table/symbol/venue/segment/month/plan-hash/watermark/schema/root/
    action), is expired, is self-hash-invalid, or is missing."""


class CatalogLockConflict(Exception):
    """Raised when the catalog lock is already held by a live owner, or a
    non-owner attempts release."""


class CatalogLockRepairRequired(Exception):
    """Raised when a lock exists but cannot be proven owned by this job --
    classified for a separate repair action, never auto-deleted."""


class ResourceLimitExceeded(Exception):
    """Fail-closed resource guard -- no hidden force override."""


# ---------------------------------------------------------------------------
# Plan hash (Phase 3/9)
# ---------------------------------------------------------------------------

def plan_hash(*, partition: PartitionIdentity, archive_version: str, root: str,
              source_schema_hash: str, archive_schema_hash: str) -> str:
    payload = "|".join(str(x) for x in (
        partition.table, partition.symbol, partition.venue, partition.market_segment,
        partition.utc_year, partition.utc_month, partition.partition_start_ms,
        partition.partition_end_ms, partition.source_watermark_field,
        partition.source_watermark_value, archive_version,
        os.path.normpath(os.path.abspath(root)).replace("\\", "/").lower(),
        source_schema_hash, archive_schema_hash))
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Authorization receipt (Phase 3-4)
# ---------------------------------------------------------------------------

def _receipt_self_hash(body: dict) -> str:
    without = {k: v for k, v in body.items() if k != "receipt_sha256"}
    return hashlib.sha256(json.dumps(without, indent=2, sort_keys=True, default=str).encode()).hexdigest()


def build_authorization_receipt(*, partition: PartitionIdentity, spec: SourceTableSpec,
                                 archive_version: str, root: str, source_schema_hash: str,
                                 archive_schema_hash: str, max_source_rows: int,
                                 max_source_bytes: int, max_output_bytes: int,
                                 approver: str, justification: str,
                                 expiry_ms: int | None = None) -> dict:
    now_ms = int(time.time() * 1000)
    ph = plan_hash(partition=partition, archive_version=archive_version, root=root,
                   source_schema_hash=source_schema_hash, archive_schema_hash=archive_schema_hash)
    body = {
        "authorization_contract_version": AUTHORIZATION_CONTRACT_VERSION,
        "action": "CREATE_PRODUCTION_ARCHIVE_ONLY",
        "source_database_identity": spec.source_database,
        "source_table": partition.table, "symbol": partition.symbol,
        "venue": partition.venue, "market_segment": partition.market_segment,
        "utc_partition_start": partition.partition_start_ms,
        "utc_partition_end": partition.partition_end_ms,
        "archive_version": archive_version,
        "production_archive_root": os.path.normpath(os.path.abspath(root)).replace("\\", "/"),
        "exact_partition_plan_hash": ph,
        "source_watermark_field": partition.source_watermark_field,
        "source_watermark_value": partition.source_watermark_value,
        "source_schema_hash": source_schema_hash, "archive_schema_hash": archive_schema_hash,
        "max_authorized_source_rows": max_source_rows,
        "max_authorized_source_bytes": max_source_bytes,
        "max_authorized_output_bytes": max_output_bytes,
        "source_retention_requirement": "SOURCE_MUST_REMAIN_PRESENT",
        "purge_authorization": "PROHIBITED",
        "scheduler_authorization": "PROHIBITED",
        "vacuum_authorization": "PROHIBITED",
        "issued_ms": now_ms, "expiry_ms": expiry_ms,
        "authorization_identity": f"AUTH-PROD-{partition.partition_id}-{now_ms}",
        "canonical_serialization_version": "v1",
        "approver": approver, "justification": justification,
    }
    body["receipt_sha256"] = _receipt_self_hash(body)
    return body


def verify_authorization_receipt(receipt: dict, *, partition: PartitionIdentity, archive_version: str,
                                  root: str, source_schema_hash: str, archive_schema_hash: str) -> None:
    """Fail-closed: raises AuthorizationReceiptRejected unless the receipt
    exactly matches the live plan on every binding field, is self-hash-
    valid, is the right action, and has not expired."""
    if not receipt:
        raise AuthorizationReceiptRejected("no authorization receipt supplied")
    if receipt.get("receipt_sha256") != _receipt_self_hash(receipt):
        raise AuthorizationReceiptRejected("receipt self-hash mismatch (tampered)")
    if receipt.get("action") != "CREATE_PRODUCTION_ARCHIVE_ONLY":
        raise AuthorizationReceiptRejected(f"wrong action {receipt.get('action')!r}")
    for field in ("purge_authorization", "scheduler_authorization", "vacuum_authorization"):
        if receipt.get(field) != "PROHIBITED":
            raise AuthorizationReceiptRejected(f"{field} is not PROHIBITED")
    if receipt.get("source_retention_requirement") != "SOURCE_MUST_REMAIN_PRESENT":
        raise AuthorizationReceiptRejected("source_retention_requirement not enforced")
    expected_ph = plan_hash(partition=partition, archive_version=archive_version, root=root,
                            source_schema_hash=source_schema_hash, archive_schema_hash=archive_schema_hash)
    if receipt.get("exact_partition_plan_hash") != expected_ph:
        raise AuthorizationReceiptRejected("plan hash mismatch (partition/watermark/schema/root drift)")
    if receipt.get("source_table") != partition.table or receipt.get("symbol") != partition.symbol \
            or receipt.get("venue") != partition.venue \
            or receipt.get("market_segment") != partition.market_segment:
        raise AuthorizationReceiptRejected("table/symbol/venue/segment mismatch")
    if receipt.get("source_watermark_value") != partition.source_watermark_value:
        raise AuthorizationReceiptRejected("watermark mismatch")
    expiry = receipt.get("expiry_ms")
    if expiry is not None and int(time.time() * 1000) > expiry:
        raise AuthorizationReceiptRejected("receipt expired")


# ---------------------------------------------------------------------------
# Catalog lock (Phase 5)
# ---------------------------------------------------------------------------

def _lock_path(root: str) -> str:
    return os.path.join(os.path.dirname(os.path.normpath(root)), CATALOG_LOCK_NAME)


def acquire_catalog_lock(root: str, *, job_identity: str, plan_hash_value: str | None = None,
                          intended_partition: str | None = None, timeout_sec: float = 30.0,
                          poll_sec: float = 0.5) -> dict:
    """Atomic exclusive-create lock. Returns lock metadata including a
    secret `owner_token` required for release. Raises CatalogLockConflict
    on timeout (another live owner holds it)."""
    lock_path = _lock_path(root)
    owner_token = hashlib.sha256(os.urandom(32)).hexdigest()
    meta = {
        "lock_contract_version": LOCK_CONTRACT_VERSION, "job_identity": job_identity,
        "pid": os.getpid(), "host": socket.gethostname(),
        "started_ms": int(time.time() * 1000), "plan_hash": plan_hash_value,
        "intended_partition": intended_partition,
        "owner_token_sha256": hashlib.sha256(owner_token.encode()).hexdigest(),
    }
    deadline = time.time() + timeout_sec
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                raise CatalogLockConflict(
                    f"catalog lock held by another owner at {lock_path!r} (timed out after {timeout_sec}s)")
            time.sleep(poll_sec)
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return {**meta, "owner_token": owner_token, "lock_path": lock_path}


def release_catalog_lock(lock_info: dict) -> None:
    """Owner-only release: verifies the on-disk lock's owner-token hash
    matches this holder's token before removing it. A non-owner (wrong
    token) is refused."""
    lock_path = lock_info["lock_path"]
    if not os.path.exists(lock_path):
        return
    with open(lock_path) as f:
        on_disk = json.load(f)
    if on_disk.get("owner_token_sha256") != hashlib.sha256(lock_info["owner_token"].encode()).hexdigest():
        raise CatalogLockConflict("refusing to release a lock owned by a different job")
    os.remove(lock_path)


def catalog_lock_status(root: str) -> dict:
    lock_path = _lock_path(root)
    if not os.path.exists(lock_path):
        return {"present": False, "state": "ABSENT"}
    try:
        with open(lock_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"present": True, "state": "CATALOG_LOCK_REPAIR_REQUIRED", "reason": "unreadable lock metadata"}
    age_ms = int(time.time() * 1000) - meta.get("started_ms", 0)
    return {"present": True, "state": "HELD", "job_identity": meta.get("job_identity"),
            "pid": meta.get("pid"), "host": meta.get("host"), "age_ms": age_ms}


# ---------------------------------------------------------------------------
# Candidate selection (Phase 8)
# ---------------------------------------------------------------------------

def _month_bounds_ms(year: int, month: int) -> tuple[int, int]:
    start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
    days = calendar.monthrange(year, month)[1]
    end = start + dt.timedelta(days=days)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def select_candidate_partition(conn, *, table: str, now: dt.datetime | None = None,
                                active_retention_days: int = 30) -> dict:
    """Deterministic, outcome-blind. Earliest eligible closed UTC month
    (>= the table's earliest recorded row, fully closed, outside the
    active horizon); within it, the symbol with the smallest positive
    bounded row count (lexicographic tie-break). Records every candidate
    considered and every rejection reason."""
    spec = get_table_spec(table)
    now = now or dt.datetime.now(dt.timezone.utc)
    earliest_ts = conn.execute(
        f"SELECT MIN({spec.partition_ts_field}) FROM {table}").fetchone()[0]
    if earliest_ts is None:
        raise PartitionValidationError(f"{table} has no rows")
    earliest_dt = dt.datetime.fromtimestamp(earliest_ts / 1000, dt.timezone.utc)

    considered_months = []
    chosen_year = chosen_month = None
    y, m = earliest_dt.year, earliest_dt.month
    for _ in range(24):  # bounded scan forward
        try:
            build_partition_identity(table=table, symbol="_", utc_year=y, utc_month=m,
                                     source_watermark_value=0, now=now,
                                     active_retention_days=active_retention_days)
            considered_months.append({"year": y, "month": m, "eligible": True})
            chosen_year, chosen_month = y, m
            break
        except PartitionValidationError as exc:
            considered_months.append({"year": y, "month": m, "eligible": False, "reason": str(exc)})
        m += 1
        if m > 12:
            m = 1
            y += 1
    if chosen_year is None:
        raise PartitionValidationError(f"no eligible closed month found for {table}")

    start_ms, end_ms = _month_bounds_ms(chosen_year, chosen_month)
    symbol_counts = []
    for sym in CANDIDATE_SYMBOL_UNIVERSE:
        cnt = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {spec.symbol_field}=? AND "
            f"{spec.partition_ts_field}>=? AND {spec.partition_ts_field}<?",
            (sym, start_ms, end_ms)).fetchone()[0]
        symbol_counts.append({"symbol": sym, "count": cnt})
    positive = [c for c in symbol_counts if c["count"] > 0]
    if not positive:
        raise PartitionValidationError(f"no symbol with positive rows in {table} {chosen_year}-{chosen_month:02d}")
    positive.sort(key=lambda c: (c["count"], c["symbol"]))
    chosen = positive[0]

    return {
        "table": table, "chosen_year": chosen_year, "chosen_month": chosen_month,
        "chosen_symbol": chosen["symbol"], "chosen_row_count": chosen["count"],
        "considered_months": considered_months, "symbol_counts": symbol_counts,
        "partition_start_ms": start_ms, "partition_end_ms": end_ms,
    }


# ---------------------------------------------------------------------------
# Authorized production publication (Phase 12-14) -- six-file, streaming
# ---------------------------------------------------------------------------

def build_activation_catalog_entry(*, base_entry: dict, receipt_relative_path: str,
                                    receipt_sha256: str, authorization_identity: str) -> dict:
    """Extends the rehearsal-era catalog entry with activation-era
    authorization fields."""
    return {**base_entry,
            "authorization_receipt_path": receipt_relative_path,
            "authorization_receipt_sha256": receipt_sha256,
            "authorization_identity": authorization_identity,
            "action": "CREATE_PRODUCTION_ARCHIVE_ONLY",
            "activation_era": True}


@dataclass
class ActivationPublicationResult:
    status: str
    final_partition_dir: str
    row_count: int
    scientific_hash: str
    parquet_sha256: str
    manifest_sha256: str
    catalog_entry_sha256: str
    receipt_sha256: str
    reverification_mismatch_count: int


def publish_authorized_production_partition(conn, *, root: str, partition: PartitionIdentity,
                                             spec: SourceTableSpec, archive_version: str,
                                             receipt: dict, job_identity: str,
                                             source_schema_hash: str, export_cutoff: str,
                                             max_output_bytes: int = MAX_PARQUET_BYTES_PER_PARTITION
                                             ) -> ActivationPublicationResult:
    """Six-file authorized publication. The receipt is verified against
    the live plan BEFORE any production write; the final directory rename
    and root-index update happen under the catalog lock."""
    schema = build_pyarrow_schema(spec)
    archive_schema_hash = hashlib.sha256(str(schema).encode()).hexdigest()
    verify_authorization_receipt(receipt, partition=partition, archive_version=archive_version,
                                 root=root, source_schema_hash=source_schema_hash,
                                 archive_schema_hash=archive_schema_hash)

    final_dir = PR.final_partition_dir(root, partition, archive_version)
    if os.path.exists(final_dir):
        raise PR.ProductionPublicationConflict(f"final partition path already exists: {final_dir!r}")

    staging_dir = PR.staging_partition_dir(root, partition, job_identity, archive_version)
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # ---- streaming export (staging, before lock) ----
    parquet_partial = os.path.join(staging_dir, PR.PARQUET_NAME + ".partial")
    export = stream_export_to_parquet(conn, spec, partition, parquet_partial, max_output_bytes=max_output_bytes)
    row_count, scientific_hash = export["row_count"], export["scientific_content_hash"]
    reread = stream_hash_parquet(parquet_partial, spec.preserved_columns)
    if reread["scientific_content_hash"] != scientific_hash or reread["row_count"] != row_count:
        raise PR.ProductionPublicationConflict("parquet scientific parity failed")
    parquet_final = os.path.join(staging_dir, PR.PARQUET_NAME)
    os.replace(parquet_partial, parquet_final)
    parquet_sha = PR._sha256_file(parquet_final)

    rel = PR.partition_relative_dir(partition, archive_version)
    manifest = build_manifest(
        spec=spec, partition=partition, row_count=row_count, scientific_hash=scientific_hash,
        parquet_path=os.path.join(rel, PR.PARQUET_NAME), parquet_size=os.path.getsize(parquet_final),
        parquet_sha256=parquet_sha, source_schema_hash=source_schema_hash,
        parquet_schema_hash=archive_schema_hash, unresolved_gap_count=0, export_cutoff=export_cutoff,
        publication_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        verification_status="VERIFIED", dry_run_identity=job_identity,
        production_status="PRODUCTION_VERIFIED")
    manifest["partition_id"] = partition.partition_id
    manifest_path = os.path.join(staging_dir, PR.MANIFEST_NAME)
    manifest_sha = PR._write_json_atomic(manifest_path, manifest)

    # ---- authorization receipt ----
    receipt_path = os.path.join(staging_dir, AUTHORIZATION_RECEIPT_NAME)
    receipt_sha = PR._write_json_atomic(receipt_path, receipt)

    base_entry = PR.build_catalog_entry(
        partition=partition, spec=spec, archive_version=archive_version, row_count=row_count,
        archive_relative_path=rel, manifest_relative_path=os.path.join(rel, PR.MANIFEST_NAME),
        parquet_relative_path=os.path.join(rel, PR.PARQUET_NAME), manifest_sha256=manifest_sha,
        parquet_sha256=parquet_sha, scientific_hash=scientific_hash, archive_schema_hash=archive_schema_hash,
        source_schema_hash=source_schema_hash,
        publication_timestamp=dt.datetime.now(dt.timezone.utc).isoformat())
    catalog_entry = build_activation_catalog_entry(
        base_entry=base_entry, receipt_relative_path=os.path.join(rel, AUTHORIZATION_RECEIPT_NAME),
        receipt_sha256=receipt_sha, authorization_identity=receipt["authorization_identity"])
    catalog_entry_path = os.path.join(staging_dir, PR.CATALOG_ENTRY_NAME)
    catalog_entry_sha = PR._write_json_atomic(catalog_entry_path, catalog_entry)

    success_partial = os.path.join(staging_dir, PR.SUCCESS_NAME + ".partial")
    with open(success_partial, "w") as f:
        f.write(f"{partition.partition_id}\n")
    os.replace(success_partial, os.path.join(staging_dir, PR.SUCCESS_NAME))

    for required in (PR.PARQUET_NAME, PR.MANIFEST_NAME, AUTHORIZATION_RECEIPT_NAME,
                     PR.CATALOG_ENTRY_NAME, PR.SUCCESS_NAME):
        if not os.path.exists(os.path.join(staging_dir, required)):
            raise PR.ProductionPublicationConflict(f"staging missing {required}")
    if any(fn.endswith(".partial") for fn in os.listdir(staging_dir)):
        raise PR.ProductionPublicationConflict("staging still has a .partial file")

    # ---- catalog lock: final conflict recheck + publication + index under lock ----
    lock = acquire_catalog_lock(root, job_identity=job_identity,
                                plan_hash_value=receipt["exact_partition_plan_hash"],
                                intended_partition=partition.partition_id)
    try:
        if os.path.exists(final_dir):
            raise PR.ProductionPublicationConflict(f"final path appeared before publication: {final_dir!r}")
        os.makedirs(os.path.dirname(final_dir), exist_ok=True)
        os.rename(staging_dir, final_dir)
        mismatches = reverify_authorized_partition(final_dir, partition)
        if mismatches != 0:
            raise PR.ProductionPublicationConflict(f"post-publication re-verification found {mismatches} mismatches")
        rebuild_root_index_locked(root)
    finally:
        release_catalog_lock(lock)

    return ActivationPublicationResult(
        status="PUBLISHED", final_partition_dir=final_dir, row_count=row_count,
        scientific_hash=scientific_hash, parquet_sha256=parquet_sha, manifest_sha256=manifest_sha,
        catalog_entry_sha256=catalog_entry_sha, receipt_sha256=receipt_sha,
        reverification_mismatch_count=mismatches)


def reverify_authorized_partition(final_dir: str, partition: PartitionIdentity) -> int:
    """Rehearsal-era re-verification PLUS authorization-receipt checks.
    Uses streaming hash for the row re-read (RAM-bounded)."""
    mismatches = PR.reverify_published_partition(final_dir, partition, streaming=True)
    receipt_path = os.path.join(final_dir, AUTHORIZATION_RECEIPT_NAME)
    if not os.path.exists(receipt_path):
        return mismatches + 1
    with open(receipt_path) as f:
        receipt = json.load(f)
    with open(os.path.join(final_dir, PR.CATALOG_ENTRY_NAME)) as f:
        entry = json.load(f)
    if receipt.get("receipt_sha256") != _receipt_self_hash(receipt):
        mismatches += 1
    if entry.get("authorization_receipt_sha256") != PR._sha256_file(receipt_path):
        mismatches += 1
    if receipt.get("action") != "CREATE_PRODUCTION_ARCHIVE_ONLY":
        mismatches += 1
    for field in ("purge_authorization", "scheduler_authorization", "vacuum_authorization"):
        if receipt.get(field) != "PROHIBITED":
            mismatches += 1
    return mismatches


def rebuild_root_index_locked(root: str) -> tuple[str, str]:
    """Root-index rebuild that assumes the catalog lock is already held by
    the caller (used inside `publish_authorized_production_partition`)."""
    return PR.publish_root_catalog_index(root)


def rebuild_root_index_under_lock(root: str, *, job_identity: str) -> tuple[str, str]:
    """Standalone lock-protected rebuild (the `production-catalog-rebuild`
    CLI path). Acquires the lock, rebuilds, releases."""
    lock = acquire_catalog_lock(root, job_identity=job_identity, intended_partition="INDEX_REBUILD")
    try:
        return PR.publish_root_catalog_index(root)
    finally:
        release_catalog_lock(lock)


# ---------------------------------------------------------------------------
# Preflight + resource guards (Phases 9-10)
# ---------------------------------------------------------------------------

def preflight_partition(conn, *, root: str, table: str, now: dt.datetime | None = None) -> dict:
    """Read-only. Deterministic candidate selection + full plan
    resolution (partition identity, watermark, row count, schema hashes,
    gap/repair status, resource estimates) WITHOUT publishing. Raises on
    any structural problem; returns a plan dict otherwise."""
    spec = get_table_spec(table)
    sel = select_candidate_partition(conn, table=table, now=now)
    start_ms, end_ms = sel["partition_start_ms"], sel["partition_end_ms"]
    watermark = conn.execute(
        f"SELECT MAX({spec.stable_ordering_field}) FROM {table} WHERE {spec.symbol_field}=? AND "
        f"{spec.partition_ts_field}>=? AND {spec.partition_ts_field}<?",
        (sel["chosen_symbol"], start_ms, end_ms)).fetchone()[0]
    partition = build_partition_identity(
        table=table, symbol=sel["chosen_symbol"], utc_year=sel["chosen_year"],
        utc_month=sel["chosen_month"], source_watermark_value=watermark, now=now)
    schema = build_pyarrow_schema(spec)
    archive_schema_hash = hashlib.sha256(str(schema).encode()).hexdigest()
    source_schema_sql = conn.execute(f"SELECT sql FROM sqlite_master WHERE name='{table}'").fetchone()[0]
    source_schema_hash = hashlib.sha256(source_schema_sql.encode()).hexdigest()

    row_count = sel["chosen_row_count"]
    est_source_bytes = row_count * 64
    est_parquet_bytes = row_count * 80  # conservative upper bound

    gap_count = 0
    if spec.gap_ledger_stream:
        gap_count = conn.execute(
            "SELECT COUNT(*) FROM gaps WHERE stream=? AND start_ts_ms<? AND "
            "(end_ts_ms IS NULL OR end_ts_ms>=?) AND resolved_bool=0",
            (spec.gap_ledger_stream, end_ms, start_ms)).fetchone()[0]

    final_dir = PR.final_partition_dir(root, partition, "v1")
    ph = plan_hash(partition=partition, archive_version="v1", root=root,
                   source_schema_hash=source_schema_hash, archive_schema_hash=archive_schema_hash)
    return {
        "table": table, "partition": partition, "spec": spec,
        "selection": sel, "watermark": watermark, "row_count": row_count,
        "source_schema_hash": source_schema_hash, "archive_schema_hash": archive_schema_hash,
        "estimated_source_bytes": est_source_bytes, "estimated_parquet_bytes": est_parquet_bytes,
        "unresolved_gap_count": gap_count, "repair_status": spec.repair_layer or "NOT_APPLICABLE",
        "final_dir": final_dir, "final_path_exists": os.path.exists(final_dir),
        "plan_hash": ph,
    }


def check_resource_limits(*, free_bytes: int, plans: list[dict]) -> dict:
    """Fail-closed combined resource guard (Phase 10). No hidden override."""
    if free_bytes < MIN_FREE_BYTES_BEFORE_START:
        raise ResourceLimitExceeded(
            f"free space {free_bytes} < required {MIN_FREE_BYTES_BEFORE_START}")
    total_perm = 0
    for p in plans:
        if p["estimated_source_bytes"] > MAX_SOURCE_BYTES_PER_PARTITION:
            raise ResourceLimitExceeded(f"{p['table']} source estimate exceeds per-partition cap")
        if p["estimated_parquet_bytes"] > MAX_PARQUET_BYTES_PER_PARTITION:
            raise ResourceLimitExceeded(f"{p['table']} parquet estimate exceeds per-partition cap")
        total_perm += p["estimated_parquet_bytes"]
    projected_free = free_bytes - total_perm
    if projected_free < 400 * 1024 ** 3:
        raise ResourceLimitExceeded(f"projected free {projected_free} < 400GB floor")
    return {"total_estimated_permanent_bytes": total_perm, "projected_free_bytes": projected_free}


def issue_gate_authorized_receipt(plan: dict, *, root: str, approver: str, justification: str) -> dict:
    """Represents the gate's own authorization -- NOT a self-approving CLI
    path. Builds an exact-partition-bound receipt for one preflighted
    plan. The narrowly-scoped governance driver (this batch) is the only
    caller; future receipts require separate governance authorization."""
    return build_authorization_receipt(
        partition=plan["partition"], spec=plan["spec"], archive_version="v1", root=root,
        source_schema_hash=plan["source_schema_hash"], archive_schema_hash=plan["archive_schema_hash"],
        max_source_rows=plan["row_count"], max_source_bytes=MAX_SOURCE_BYTES_PER_PARTITION,
        max_output_bytes=MAX_PARQUET_BYTES_PER_PARTITION, approver=approver, justification=justification)


# ---------------------------------------------------------------------------
# Narrowly-scoped governance driver (Phase 3/9/12) -- the ONLY path that
# both issues a receipt and publishes. Not exposed as an arbitrary CLI
# command. Preflights, resource-checks, issues one receipt, publishes one
# partition. Idempotent (returns NOOP for an already-published identical
# partition rather than conflicting).
# ---------------------------------------------------------------------------

def gate_publish_partition(conn, *, root: str, table: str, approver: str, justification: str,
                            now: dt.datetime | None = None, free_bytes: int | None = None) -> dict:
    import shutil as _sh
    plan = preflight_partition(conn, root=root, table=table, now=now)
    partition = plan["partition"]

    # idempotency: if an identical archive already exists, NOOP (no republish)
    if plan["final_path_exists"]:
        final_dir = plan["final_dir"]
        with open(os.path.join(final_dir, PR.MANIFEST_NAME)) as f:
            existing = json.load(f)
        if existing.get("source_watermark_value") == partition.source_watermark_value \
                and existing.get("row_count") == plan["row_count"]:
            mismatches = reverify_authorized_partition(final_dir, partition)
            return {"status": "NOOP_IDENTICAL_PRODUCTION_ARCHIVE", "table": table,
                    "final_partition_dir": final_dir, "reverification_mismatch_count": mismatches,
                    "partition_id": partition.partition_id}
        return {"status": "SOURCE_CHANGED_NEW_VERSION_REQUIRED", "table": table,
                "final_partition_dir": final_dir,
                "existing_watermark": existing.get("source_watermark_value"),
                "current_watermark": partition.source_watermark_value}

    if free_bytes is None:
        free_bytes = _sh.disk_usage(os.path.splitdrive(os.path.abspath(root))[0] + os.sep).free
    check_resource_limits(free_bytes=free_bytes, plans=[plan])

    receipt = issue_gate_authorized_receipt(plan, root=root, approver=approver, justification=justification)
    job_identity = "PROD-ACTIVATION-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    result = publish_authorized_production_partition(
        conn, root=root, partition=partition, spec=plan["spec"], archive_version="v1",
        receipt=receipt, job_identity=job_identity, source_schema_hash=plan["source_schema_hash"],
        export_cutoff=dt.datetime.now(dt.timezone.utc).isoformat())
    return {
        "status": result.status, "table": table, "final_partition_dir": result.final_partition_dir,
        "symbol": partition.symbol, "utc_year": partition.utc_year, "utc_month": partition.utc_month,
        "row_count": result.row_count, "scientific_hash": result.scientific_hash,
        "parquet_sha256": result.parquet_sha256, "manifest_sha256": result.manifest_sha256,
        "catalog_entry_sha256": result.catalog_entry_sha256, "receipt_sha256": result.receipt_sha256,
        "reverification_mismatch_count": result.reverification_mismatch_count,
        "authorization_identity": receipt["authorization_identity"],
        "selection": {k: plan["selection"][k] for k in ("chosen_symbol", "chosen_year", "chosen_month",
                                                          "chosen_row_count", "symbol_counts", "considered_months")},
    }


# ---------------------------------------------------------------------------
# Sharded publication (BATCH-STORAGE-PRODUCTION-ARCHIVE-BOOK-TICKER-
# RECOVERY-V1) -- for partitions too large for the single-file design
# above to finalize within the RAM guardrail (book_ticker/SOLUSDT/
# 2026-04, 114,404,095 rows, hit a ~3.0GB resident-memory wall; see
# `ami.storage.sharded_archive` module docstring for the full root-cause
# and design rationale). Reuses every authorization/locking/atomicity
# mechanism above unchanged -- the ONLY structural difference is calling
# the sharded, resumable, RSS-guarded exporter instead of the single-file
# one, and writing a `shards` inventory into the manifest/catalog entry.
# Staging is resume-safe: unlike `publish_authorized_production_partition`
# (which always wipes staging at the start), this only removes `.partial`
# leftovers and any OTHER partition's stale shards via
# `SHA.clean_stale_staging` -- a MemoryGuardAbort mid-export leaves
# already-finalized shards in place for a subsequent call (same
# `job_identity`, hence same deterministic staging_dir) to resume from.
# ---------------------------------------------------------------------------


def publish_authorized_production_partition_sharded(
        conn, *, root: str, partition: PartitionIdentity, spec: SourceTableSpec,
        archive_version: str, receipt: dict, job_identity: str, source_schema_hash: str,
        export_cutoff: str, max_rows_per_shard: int = 10_000_000,
        max_output_bytes_per_shard: int = 2 * 1024 ** 3, rss_check=None,
        rss_limit_bytes: int | None = None, batch_size: int = 1_000_000,
        rss_check_every_rows: int = 2_000_000,
        reverify_rss_limit_bytes: int = 2 * 1024 ** 3,
        reverify_poll_interval_seconds: float = 1.0) -> ActivationPublicationResult:
    """Multi-shard counterpart to `publish_authorized_production_partition`.
    Same receipt-first authorization and same staging -> atomic-directory-
    rename -> catalog-lock -> reverify -> index-rebuild flow; the export
    step may raise `SHA.MemoryGuardAbort` (propagates to the caller --
    nothing is published, the catalog lock is never acquired, staging is
    left resumable). Retrying with the identical `job_identity` reuses the
    same deterministic staging directory and resumes from the next
    unwritten id rather than re-exporting already-finalized shards.

    The post-publish reverify step (BATCH-STORAGE-PRODUCTION-ARCHIVE-
    REVERIFY-MEMORY-HARDENING-V1) runs in a fresh, externally RSS-guarded
    subprocess (`ami.storage.reverify_guard.run_guarded_reverify`)
    instead of in-process -- this same process already ran one full
    stream_hash_parquet_multi pass a few lines above (the pre-publish
    reread parity check), and a second in-process pass was measured to
    push resident memory to ~4GB on the real book_ticker/SOLUSDT/2026-04
    publish (allocator fragmentation across the two passes, not a true
    live-memory requirement -- see `ami.storage.reverify_worker`'s module
    docstring). A subprocess reclaims 100% of its memory from the OS on
    exit regardless of that fragmentation. If the guard trips or the
    worker crashes, this raises (via `reverify_guard`) and nothing about
    the just-published partition or the catalog lock's index rebuild is
    left in an ambiguous state -- the lock is released either way
    (`finally`), but the index rebuild simply doesn't happen, matching
    the prior in-process behavior's own failure mode."""
    schema = build_pyarrow_schema(spec)
    archive_schema_hash = hashlib.sha256(str(schema).encode()).hexdigest()
    verify_authorization_receipt(receipt, partition=partition, archive_version=archive_version,
                                 root=root, source_schema_hash=source_schema_hash,
                                 archive_schema_hash=archive_schema_hash)

    final_dir = PR.final_partition_dir(root, partition, archive_version)
    if os.path.exists(final_dir):
        raise PR.ProductionPublicationConflict(f"final partition path already exists: {final_dir!r}")

    staging_dir = PR.staging_partition_dir(root, partition, job_identity, archive_version)
    os.makedirs(staging_dir, exist_ok=True)
    SHA.clean_stale_staging(staging_dir, partition_id=partition.partition_id)

    # ---- streaming, resumable, multi-shard export (staging, before lock) ----
    export = SHA.stream_export_to_parquet_sharded(
        conn, spec, partition, staging_dir, batch_size=batch_size, max_rows_per_shard=max_rows_per_shard,
        max_output_bytes_per_shard=max_output_bytes_per_shard,
        rss_check=rss_check, rss_limit_bytes=rss_limit_bytes, rss_check_every_rows=rss_check_every_rows)
    row_count = export.row_count

    ordered_shards = sorted(export.shards, key=lambda s: s["shard_index"])
    shard_paths = [os.path.join(staging_dir, s["shard_file"]) for s in ordered_shards]
    reread = SHA.stream_hash_parquet_multi(shard_paths, spec.preserved_columns)
    if reread["row_count"] != row_count:
        raise PR.ProductionPublicationConflict("sharded parquet row-count parity failed")
    scientific_hash = reread["scientific_content_hash"]

    shard_inventory = []
    total_bytes = 0
    for s, path in zip(ordered_shards, shard_paths):
        total_bytes += s["byte_size"]
        shard_inventory.append({**s, "sha256": PR._sha256_file(path)})

    rel = PR.partition_relative_dir(partition, archive_version)
    manifest = build_manifest(
        spec=spec, partition=partition, row_count=row_count, scientific_hash=scientific_hash,
        parquet_path=os.path.join(rel, shard_inventory[0]["shard_file"]), parquet_size=total_bytes,
        parquet_sha256=shard_inventory[0]["sha256"], source_schema_hash=source_schema_hash,
        parquet_schema_hash=archive_schema_hash, unresolved_gap_count=0, export_cutoff=export_cutoff,
        publication_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        verification_status="VERIFIED", dry_run_identity=job_identity,
        production_status="PRODUCTION_VERIFIED", shards=shard_inventory)
    manifest["partition_id"] = partition.partition_id
    manifest_path = os.path.join(staging_dir, PR.MANIFEST_NAME)
    manifest_sha = PR._write_json_atomic(manifest_path, manifest)

    receipt_path = os.path.join(staging_dir, AUTHORIZATION_RECEIPT_NAME)
    receipt_sha = PR._write_json_atomic(receipt_path, receipt)

    base_entry = PR.build_catalog_entry(
        partition=partition, spec=spec, archive_version=archive_version, row_count=row_count,
        archive_relative_path=rel, manifest_relative_path=os.path.join(rel, PR.MANIFEST_NAME),
        parquet_relative_path=os.path.join(rel, shard_inventory[0]["shard_file"]),
        manifest_sha256=manifest_sha, parquet_sha256=shard_inventory[0]["sha256"],
        scientific_hash=scientific_hash, archive_schema_hash=archive_schema_hash,
        source_schema_hash=source_schema_hash,
        publication_timestamp=dt.datetime.now(dt.timezone.utc).isoformat())
    catalog_entry = build_activation_catalog_entry(
        base_entry=base_entry, receipt_relative_path=os.path.join(rel, AUTHORIZATION_RECEIPT_NAME),
        receipt_sha256=receipt_sha, authorization_identity=receipt["authorization_identity"])
    catalog_entry["shard_count"] = len(shard_inventory)
    catalog_entry_path = os.path.join(staging_dir, PR.CATALOG_ENTRY_NAME)
    catalog_entry_sha = PR._write_json_atomic(catalog_entry_path, catalog_entry)

    success_partial = os.path.join(staging_dir, PR.SUCCESS_NAME + ".partial")
    with open(success_partial, "w") as f:
        f.write(f"{partition.partition_id}\n")
    os.replace(success_partial, os.path.join(staging_dir, PR.SUCCESS_NAME))

    # checkpoint sidecars are internal export bookkeeping, not part of the
    # published contract -- remove before validating/publishing staging
    for fn in list(os.listdir(staging_dir)):
        if fn.startswith(SHA.SHARD_CHECKPOINT_PREFIX) and fn.endswith(SHA.SHARD_CHECKPOINT_SUFFIX):
            os.remove(os.path.join(staging_dir, fn))

    required = [s["shard_file"] for s in shard_inventory] + [
        PR.MANIFEST_NAME, AUTHORIZATION_RECEIPT_NAME, PR.CATALOG_ENTRY_NAME, PR.SUCCESS_NAME]
    for req in required:
        if not os.path.exists(os.path.join(staging_dir, req)):
            raise PR.ProductionPublicationConflict(f"staging missing {req}")
    if any(fn.endswith(".partial") for fn in os.listdir(staging_dir)):
        raise PR.ProductionPublicationConflict("staging still has a .partial file")

    lock = acquire_catalog_lock(root, job_identity=job_identity,
                                plan_hash_value=receipt["exact_partition_plan_hash"],
                                intended_partition=partition.partition_id)
    try:
        if os.path.exists(final_dir):
            raise PR.ProductionPublicationConflict(f"final path appeared before publication: {final_dir!r}")
        os.makedirs(os.path.dirname(final_dir), exist_ok=True)
        os.rename(staging_dir, final_dir)
        reverify_result = RG.run_guarded_reverify(
            final_dir, rss_limit_bytes=reverify_rss_limit_bytes,
            poll_interval_seconds=reverify_poll_interval_seconds, batch_size=batch_size)
        mismatches = reverify_result["mismatch_count"]
        if mismatches != 0:
            raise PR.ProductionPublicationConflict(f"post-publication re-verification found {mismatches} mismatches")
        rebuild_root_index_locked(root)
    finally:
        release_catalog_lock(lock)

    return ActivationPublicationResult(
        status="PUBLISHED", final_partition_dir=final_dir, row_count=row_count,
        scientific_hash=scientific_hash, parquet_sha256=shard_inventory[0]["sha256"],
        manifest_sha256=manifest_sha, catalog_entry_sha256=catalog_entry_sha,
        receipt_sha256=receipt_sha, reverification_mismatch_count=mismatches)
