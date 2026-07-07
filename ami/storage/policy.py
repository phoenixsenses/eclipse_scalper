"""Storage archive policy model (BATCH-STORAGE-ROTATION-RETENTION-BOUNDED-
IMPLEMENTATION-V1). One typed, fail-closed-validated, immutable-once-
constructed policy object. Every field mirrors an operator ruling already
accepted in `reports/governance/STORAGE_ROTATION_RETENTION_READINESS_AND_
CONTRACT_V1.md` and its dry-run successor (commit `6fbe0571`) -- nothing
here invents a new policy decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

POLICY_VERSION = "v1"
SUPPORTED_ARCHIVE_FORMATS = ("PARQUET",)
SUPPORTED_COMPRESSIONS = ("ZSTD",)
MIN_ACTIVE_RETENTION_DAYS = 30
SUPPORTED_PARTITION_CADENCES = ("CLOSED_CALENDAR_MONTH",)


class PolicyValidationError(Exception):
    """Fail-closed: raised for any policy configuration this batch refuses
    to accept, including every attempt to enable deletion, VACUUM,
    production activation, or scheduling -- there is no override."""


@dataclass
class StoragePolicy:
    policy_version: str = POLICY_VERSION
    active_raw_retention_days: int = 30
    archive_format: str = "PARQUET"
    compression: str = "ZSTD"
    partition_timezone: str = "UTC"
    partition_cadence: str = "CLOSED_CALENDAR_MONTH"
    partial_month_purge_allowed: bool = False
    automatic_purge_enabled: bool = False
    automatic_vacuum_enabled: bool = False
    production_activation_enabled: bool = False
    scheduler_activation_enabled: bool = False
    max_disposable_source_bytes_estimate: int = 2 * 1024 ** 3
    max_disposable_output_bytes: int = 4 * 1024 ** 3
    require_manifest_before_publication: bool = True
    require_verification_before_publication: bool = True
    require_research_dependency_guard: bool = True
    require_source_gap_guard: bool = True
    require_repair_guard: bool = True
    require_continuity_guard: bool = True
    require_purge_authorization: bool = True
    require_physical_reclamation_gate: bool = True

    def __post_init__(self) -> None:
        validate_policy(self)


def validate_policy(policy: StoragePolicy) -> None:
    """Fail-closed. Every branch raises `PolicyValidationError` -- there
    is no code path that silently coerces an unsafe value into a safe
    one; an unsafe policy is always rejected outright."""
    if policy.policy_version != POLICY_VERSION:
        raise PolicyValidationError(f"unknown policy_version {policy.policy_version!r}")
    if policy.archive_format not in SUPPORTED_ARCHIVE_FORMATS:
        raise PolicyValidationError(f"unsupported archive_format {policy.archive_format!r}")
    if policy.compression not in SUPPORTED_COMPRESSIONS:
        raise PolicyValidationError(f"unsupported compression {policy.compression!r}")
    if policy.partition_cadence not in SUPPORTED_PARTITION_CADENCES:
        raise PolicyValidationError(f"unsupported partition_cadence {policy.partition_cadence!r}")
    if policy.active_raw_retention_days < MIN_ACTIVE_RETENTION_DAYS:
        raise PolicyValidationError(
            f"active_raw_retention_days={policy.active_raw_retention_days} "
            f"< minimum {MIN_ACTIVE_RETENTION_DAYS}")
    if policy.partition_timezone != "UTC":
        raise PolicyValidationError(f"partition_timezone must be UTC, got {policy.partition_timezone!r}")
    if policy.partial_month_purge_allowed:
        raise PolicyValidationError("partial_month_purge_allowed may never be True")
    if policy.automatic_purge_enabled:
        raise PolicyValidationError("automatic_purge_enabled may never be True (no override exists)")
    if policy.automatic_vacuum_enabled:
        raise PolicyValidationError("automatic_vacuum_enabled may never be True (no override exists)")
    if policy.production_activation_enabled:
        raise PolicyValidationError("production_activation_enabled may never be True in this batch")
    if policy.scheduler_activation_enabled:
        raise PolicyValidationError("scheduler_activation_enabled may never be True in this batch")
    if not policy.require_manifest_before_publication:
        raise PolicyValidationError("require_manifest_before_publication may never be False")
    if not policy.require_verification_before_publication:
        raise PolicyValidationError("require_verification_before_publication may never be False")
    if not policy.require_purge_authorization:
        raise PolicyValidationError("require_purge_authorization may never be False")
    if policy.max_disposable_source_bytes_estimate <= 0:
        raise PolicyValidationError("max_disposable_source_bytes_estimate must be positive")
    if policy.max_disposable_output_bytes <= 0:
        raise PolicyValidationError("max_disposable_output_bytes must be positive")


DEFAULT_POLICY = StoragePolicy()
