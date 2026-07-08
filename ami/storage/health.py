"""Storage archive health/readiness reporting (Phase 19). Pure
aggregation over already-collected inputs -- opens nothing itself. Not
wired to any live automation in this batch (report only).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StorageHealthReport:
    effective_policy_version: str
    tooling_versions: dict[str, str]
    source_registry_tables: tuple[str, ...]
    active_job_count: int
    abandoned_partial_count: int
    verified_disposable_count: int
    failed_count: int
    unverified_count: int
    archive_lag_days: float | None
    source_gap_blockers: tuple[str, ...]
    repair_blockers: tuple[str, ...]
    research_dependency_blockers: tuple[str, ...]
    source_database_size_bytes: int
    wal_size_bytes: int
    drive_free_bytes: int
    estimated_days_to_warning: float | None
    production_activation: str = "DISABLED"
    scheduler: str = "DISABLED"
    purge: str = "DISABLED"
    vacuum: str = "DISABLED"
    # Production archive fields (BATCH-...-PRODUCTION-ACTIVATION-REHEARSAL-V1)
    production_archive_root: str | None = None
    verified_production_partitions: int = 0
    failed_production_partitions: int = 0
    staging_directory_count: int = 0
    abandoned_staging_count: int = 0
    root_catalog_index_status: str = "NOT_BUILT"
    root_catalog_entry_count: int = 0
    total_archive_bytes: int = 0
    latest_publication_timestamp: str | None = None
    production_archive_rehearsal_status: str = "NOT_ATTEMPTED"


def build_health_report(*, policy_version: str, tooling_versions: dict[str, str],
                         source_registry_tables: tuple[str, ...], jobs: list,
                         source_gap_blockers: tuple[str, ...], repair_blockers: tuple[str, ...],
                         research_dependency_blockers: tuple[str, ...],
                         source_database_size_bytes: int, wal_size_bytes: int,
                         drive_free_bytes: int, growth_rate_bytes_per_day: float | None = None,
                         warning_threshold_free_bytes: int | None = None,
                         production_archive_root: str | None = None) -> StorageHealthReport:
    from ami.storage.job_state import VERIFIED_DISPOSABLE, FAILED, ABANDONED_PARTIAL

    verified = sum(1 for j in jobs if j.state == VERIFIED_DISPOSABLE)
    failed = sum(1 for j in jobs if j.state == FAILED)
    abandoned = sum(1 for j in jobs if j.state == ABANDONED_PARTIAL)
    unverified = len(jobs) - verified - failed - abandoned

    days_to_warning = None
    if growth_rate_bytes_per_day and warning_threshold_free_bytes is not None and growth_rate_bytes_per_day > 0:
        headroom = drive_free_bytes - warning_threshold_free_bytes
        days_to_warning = max(headroom, 0) / growth_rate_bytes_per_day

    production_fields = _scan_production_root(production_archive_root) if production_archive_root else {}

    return StorageHealthReport(
        effective_policy_version=policy_version, tooling_versions=tooling_versions,
        source_registry_tables=source_registry_tables, active_job_count=len(jobs),
        abandoned_partial_count=abandoned, verified_disposable_count=verified,
        failed_count=failed, unverified_count=unverified, archive_lag_days=None,
        source_gap_blockers=source_gap_blockers, repair_blockers=repair_blockers,
        research_dependency_blockers=research_dependency_blockers,
        source_database_size_bytes=source_database_size_bytes, wal_size_bytes=wal_size_bytes,
        drive_free_bytes=drive_free_bytes, estimated_days_to_warning=days_to_warning,
        **production_fields,
    )


def _scan_production_root(root: str) -> dict:
    """Read-only aggregation over an existing production root -- never
    creates, deletes, or mutates anything. Returns a partial kwargs dict
    matching StorageHealthReport's production_* fields."""
    import os
    from ami.storage import production as PR

    if not os.path.isdir(root):
        return {"production_archive_root": root, "root_catalog_index_status": "ROOT_NOT_FOUND"}

    entries = PR.scan_partition_catalog_entries(root)
    verified = sum(1 for e in entries if e.get("verification_status") == "VERIFIED")
    total_bytes = 0
    latest_ts = None
    for e in entries:
        parquet_path = os.path.join(root, e.get("parquet_relative_path", ""))
        if os.path.exists(parquet_path):
            total_bytes += os.path.getsize(parquet_path)
        ts = e.get("publication_timestamp")
        if ts and (latest_ts is None or ts > latest_ts):
            latest_ts = ts

    staging = _scan_staging(root)
    index_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    index_status = "PRESENT" if os.path.exists(index_path) else "NOT_BUILT"

    return {
        "production_archive_root": root,
        "verified_production_partitions": verified,
        "failed_production_partitions": len(entries) - verified,
        "staging_directory_count": staging["active_or_incomplete"],
        "abandoned_staging_count": staging["abandoned"],
        "root_catalog_index_status": index_status,
        "root_catalog_entry_count": len(entries),
        "total_archive_bytes": total_bytes,
        "latest_publication_timestamp": latest_ts,
        "production_archive_rehearsal_status": "COMPLETE" if verified > 0 else "NOT_ATTEMPTED",
    }


def _scan_staging(root: str) -> dict:
    """Real staging inspection (Phase 7): does NOT follow reparse points,
    does NOT delete anything. A staging dir with a `_SUCCESS` marker is
    an in-progress/complete-but-unpublished job; one without is
    abandoned/incomplete."""
    import os
    from ami.storage import production as PR
    staging_root = root.rstrip("/\\") + ".staging"
    active, abandoned, unrecognized, total_bytes = 0, 0, 0, 0
    oldest_ms = None
    if os.path.isdir(staging_root):
        for name in os.listdir(staging_root):
            d = os.path.join(staging_root, name)
            if not os.path.isdir(d):
                unrecognized += 1
                continue
            files = os.listdir(d)
            for fn in files:
                fp = os.path.join(d, fn)
                if os.path.isfile(fp):
                    total_bytes += os.path.getsize(fp)
            if PR.SUCCESS_NAME in files:
                active += 1
            else:
                abandoned += 1
            mt = int(os.path.getmtime(d) * 1000)
            oldest_ms = mt if oldest_ms is None else min(oldest_ms, mt)
    return {"active_or_incomplete": active, "abandoned": abandoned,
            "unrecognized": unrecognized, "total_bytes": total_bytes, "oldest_ms": oldest_ms}


def scan_production_archive_health(root: str) -> dict:
    """Standalone Phase 7 production-archive health report (used by the
    `production-health` CLI). Combines partition/index/staging/lock state
    into one deterministic read-only snapshot with an explicit health
    state. Follows no reparse points, deletes nothing."""
    import os
    from ami.storage import production as PR
    from ami.storage import production_activation as PA
    from ami.storage.policy import production_activation_states

    entries = PR.scan_partition_catalog_entries(root) if os.path.isdir(root) else []
    verified = sum(1 for e in entries if e.get("verification_status") == "VERIFIED")
    staging = _scan_staging(root)
    lock = PA.catalog_lock_status(root)
    index_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    index_present = os.path.exists(index_path)
    index_valid = False
    if index_present:
        try:
            import json
            with open(index_path) as f:
                idx = json.load(f)
            index_valid = idx.get("index_self_hash") is not None
        except (OSError, ValueError):
            index_valid = False

    if lock.get("state") == "CATALOG_LOCK_REPAIR_REQUIRED":
        state = "PRODUCTION_ARCHIVE_CATALOG_LOCK_REPAIR_REQUIRED"
    elif lock.get("present"):
        state = "PRODUCTION_ARCHIVE_CATALOG_LOCKED"
    elif index_present and not index_valid:
        state = "PRODUCTION_ARCHIVE_INDEX_REPAIR_REQUIRED"
    elif staging["abandoned"] > 0:
        state = "PRODUCTION_ARCHIVE_ABANDONED_STAGING"
    elif staging["active_or_incomplete"] > 0:
        state = "PRODUCTION_ARCHIVE_STAGING_ACTIVE"
    elif verified < len(entries):
        state = "PRODUCTION_ARCHIVE_VERIFICATION_FAILED"
    else:
        state = "PRODUCTION_ARCHIVE_HEALTHY"

    states = production_activation_states()
    return {
        "production_archive_root": root, "health_state": state,
        "verified_production_partitions": verified, "invalid_partition_count": len(entries) - verified,
        "root_catalog_entry_count": len(entries),
        "root_catalog_index_present": index_present, "root_catalog_index_valid": index_valid,
        "active_staging_directories": staging["active_or_incomplete"],
        "abandoned_staging_directories": staging["abandoned"],
        "unrecognized_staging_entries": staging["unrecognized"],
        "total_staging_bytes": staging["total_bytes"], "oldest_staging_ms": staging["oldest_ms"],
        "catalog_lock": lock,
        "manual_production_archive_creation": states["manual_production_archive_creation"],
        "general_unrestricted_activation": states["general_unrestricted_activation"],
        "scheduler": states["scheduler"], "purge": states["purge"], "vacuum": states["vacuum"],
        "source_retention": states["source_retention"],
    }
