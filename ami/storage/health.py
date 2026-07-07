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


def build_health_report(*, policy_version: str, tooling_versions: dict[str, str],
                         source_registry_tables: tuple[str, ...], jobs: list,
                         source_gap_blockers: tuple[str, ...], repair_blockers: tuple[str, ...],
                         research_dependency_blockers: tuple[str, ...],
                         source_database_size_bytes: int, wal_size_bytes: int,
                         drive_free_bytes: int, growth_rate_bytes_per_day: float | None = None,
                         warning_threshold_free_bytes: int | None = None) -> StorageHealthReport:
    from ami.storage.job_state import VERIFIED_DISPOSABLE, FAILED, ABANDONED_PARTIAL

    verified = sum(1 for j in jobs if j.state == VERIFIED_DISPOSABLE)
    failed = sum(1 for j in jobs if j.state == FAILED)
    abandoned = sum(1 for j in jobs if j.state == ABANDONED_PARTIAL)
    unverified = len(jobs) - verified - failed - abandoned

    days_to_warning = None
    if growth_rate_bytes_per_day and warning_threshold_free_bytes is not None and growth_rate_bytes_per_day > 0:
        headroom = drive_free_bytes - warning_threshold_free_bytes
        days_to_warning = max(headroom, 0) / growth_rate_bytes_per_day

    return StorageHealthReport(
        effective_policy_version=policy_version, tooling_versions=tooling_versions,
        source_registry_tables=source_registry_tables, active_job_count=len(jobs),
        abandoned_partial_count=abandoned, verified_disposable_count=verified,
        failed_count=failed, unverified_count=unverified, archive_lag_days=None,
        source_gap_blockers=source_gap_blockers, repair_blockers=repair_blockers,
        research_dependency_blockers=research_dependency_blockers,
        source_database_size_bytes=source_database_size_bytes, wal_size_bytes=wal_size_bytes,
        drive_free_bytes=drive_free_bytes, estimated_days_to_warning=days_to_warning,
    )
