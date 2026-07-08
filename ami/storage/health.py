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
    import json
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

    staging_root = root.rstrip("/\\") + ".staging"
    staging_dirs = []
    if os.path.isdir(staging_root):
        staging_dirs = [d for d in os.listdir(staging_root) if os.path.isdir(os.path.join(staging_root, d))]

    index_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    index_status = "PRESENT" if os.path.exists(index_path) else "NOT_BUILT"

    return {
        "production_archive_root": root,
        "verified_production_partitions": verified,
        "failed_production_partitions": len(entries) - verified,
        "staging_directory_count": 0,
        "abandoned_staging_count": len(staging_dirs),
        "root_catalog_index_status": index_status,
        "root_catalog_entry_count": len(entries),
        "total_archive_bytes": total_bytes,
        "latest_publication_timestamp": latest_ts,
        "production_archive_rehearsal_status": "COMPLETE" if verified > 0 else "NOT_ATTEMPTED",
    }
