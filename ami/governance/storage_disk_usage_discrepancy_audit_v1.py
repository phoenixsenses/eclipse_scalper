"""BATCH-STORAGE-DISK-USAGE-DISCREPANCY-AUDIT-V1.

Read-only filesystem/volume-accounting reconciliation. Every function
here is pure -- it takes already-collected byte counts (gathered by this
batch's own read-only PowerShell/`du` inspection, recorded verbatim in
the governance artifacts) and computes a deterministic reconciliation or
classification. This module never opens a file, never deletes/moves/
renames anything, and never imports `os`/`shutil`/`sqlite3` (proven by a
structural AST test, not just claimed).
"""
from __future__ import annotations

GB_DECIMAL = 1_000_000_000
GIB_BINARY = 1024 ** 3

EXPLAINED_THRESHOLD_PCT = 2.0  # Phase 13: remaining unexplained must be <= 2% of used space

CLEANUP_CLASSES = frozenset({
    "VERIFIED_DISPOSABLE_CANDIDATE",
    "POSSIBLE_CLEANUP_REQUIRES_OPERATOR_REVIEW",
    "KEEP_RESEARCH_CRITICAL",
    "KEEP_OPERATIONAL_CONTINUITY",
    "KEEP_ACCEPTED_BACKUP_OR_EVIDENCE",
    "SYSTEM_MANAGED_DO_NOT_TOUCH",
    "EXTERNAL_OR_UNRELATED_REVIEW_SEPARATELY",
})

CHROME_COPY_CLASSES = frozenset({
    "ACTIVE_REQUIRED",
    "REPRODUCIBLE_BUT_ACTIVE",
    "REPRODUCIBLE_INACTIVE_CLEANUP_CANDIDATE",
    "UNKNOWN_REQUIRES_OPERATOR_REVIEW",
    "SYSTEM_OR_EXTERNAL_DO_NOT_TOUCH",
})

PYTEST_SCRATCH_CLASSES = frozenset({
    "VERIFIED_DISPOSABLE_CANDIDATE",
    "RETAINED_ACCEPTED_EVIDENCE",
    "ACTIVE_TEMP",
    "UNKNOWN",
})

VERDICTS = frozenset({
    "STORAGE_DISK_USAGE_DISCREPANCY_EXPLAINED",
    "STORAGE_DISK_USAGE_DISCREPANCY_PARTIALLY_EXPLAINED",
    "STORAGE_DISK_USAGE_DISCREPANCY_BLOCKED_BY_PERMISSIONS",
    "STORAGE_DISK_USAGE_DISCREPANCY_BLOCKED_BY_MEASUREMENT_INCONSISTENCY",
})


def bytes_to_units(n: int) -> dict:
    """Deterministic dual-unit conversion -- every byte figure in this
    batch's artifacts is derived through this single function, so a
    GB/GiB mislabeling (the root cause this batch identified in the
    prior readiness batch's own report) cannot recur silently."""
    return {"bytes": n, "gb_decimal": n / GB_DECIMAL, "gib_binary": n / GIB_BINARY}


def reconcile_used_space(used_bytes: int, measured_items: dict[str, int]) -> dict:
    """Phase 13. `measured_items` maps a label to its measured byte
    count (top-level D: entries, including eclipse_scalper itself).
    Returns the remaining-unexplained byte count and percentage --
    never forces the numbers to reconcile, just reports the gap."""
    total_measured = sum(measured_items.values())
    remaining = used_bytes - total_measured
    pct_remaining = (remaining / used_bytes * 100.0) if used_bytes else float("nan")
    return {
        "used_bytes": used_bytes,
        "total_measured_bytes": total_measured,
        "remaining_unexplained_bytes": remaining,
        "remaining_unexplained_pct": pct_remaining,
        "explained_pct": 100.0 - pct_remaining,
        "meets_explained_threshold": pct_remaining <= EXPLAINED_THRESHOLD_PCT,
        "items": dict(measured_items),
    }


def reconcile_unit_labeling(old_label_value: float, old_label_unit: str, current_bytes: int) -> dict:
    """Explains an apparent cross-snapshot discrepancy by testing both
    GB-decimal and GiB-binary interpretations of a prior report's
    labeled figure against the current precise byte count. Does not
    assert which interpretation is correct -- shows both, transparently,
    per the gate's own 'do not force the numbers to reconcile' rule."""
    if old_label_unit not in ("GB", "GiB"):
        raise ValueError(f"old_label_unit must be 'GB' or 'GiB', got {old_label_unit!r}")
    as_decimal_gb_bytes = old_label_value * GB_DECIMAL
    as_gib_bytes = old_label_value * GIB_BINARY
    current_gib = current_bytes / GIB_BINARY
    current_gb = current_bytes / GB_DECIMAL
    return {
        "old_label_value": old_label_value, "old_label_unit": old_label_unit,
        "if_old_was_decimal_gb": {
            "old_bytes": as_decimal_gb_bytes,
            "delta_vs_current_bytes": as_decimal_gb_bytes - current_bytes,
            "delta_vs_current_gb": (as_decimal_gb_bytes - current_bytes) / GB_DECIMAL,
        },
        "if_old_was_gib_mislabeled_as_gb": {
            "old_bytes": as_gib_bytes,
            "delta_vs_current_bytes": as_gib_bytes - current_bytes,
            "delta_vs_current_gb": (as_gib_bytes - current_bytes) / GB_DECIMAL,
        },
        "current_gb_decimal": current_gb, "current_gib_binary": current_gib,
    }


def classify_chrome_copy(*, is_reparse_point: bool, referenced_by_active_process: bool,
                          referenced_by_repo_code: bool, days_since_last_write: float) -> str:
    """Deterministic classification from already-gathered evidence.
    Never authorizes deletion -- returns a classification label only."""
    if referenced_by_active_process:
        return "ACTIVE_REQUIRED"
    if is_reparse_point:
        return "UNKNOWN_REQUIRES_OPERATOR_REVIEW"
    if referenced_by_repo_code:
        return "REPRODUCIBLE_BUT_ACTIVE"
    if days_since_last_write >= 7:
        return "REPRODUCIBLE_INACTIVE_CLEANUP_CANDIDATE"
    return "UNKNOWN_REQUIRES_OPERATOR_REVIEW"


def classify_pytest_scratch(*, in_authorized_temp_dir: bool, referenced_by_active_process: bool,
                             is_only_copy_of_accepted_evidence: bool) -> str:
    if is_only_copy_of_accepted_evidence:
        return "RETAINED_ACCEPTED_EVIDENCE"
    if referenced_by_active_process:
        return "ACTIVE_TEMP"
    if in_authorized_temp_dir:
        return "ACTIVE_TEMP"
    return "VERIFIED_DISPOSABLE_CANDIDATE"


def determine_verdict(*, explained_pct: float, permission_blocked_material: bool,
                       measurement_inconsistent: bool) -> str:
    """Phase 16. Fail-closed precedence: a measurement inconsistency or
    a material permission block is reported before an EXPLAINED verdict
    is ever considered, regardless of how good the percentage looks."""
    if measurement_inconsistent:
        return "STORAGE_DISK_USAGE_DISCREPANCY_BLOCKED_BY_MEASUREMENT_INCONSISTENCY"
    if permission_blocked_material:
        return "STORAGE_DISK_USAGE_DISCREPANCY_BLOCKED_BY_PERMISSIONS"
    if explained_pct >= (100.0 - EXPLAINED_THRESHOLD_PCT):
        return "STORAGE_DISK_USAGE_DISCREPANCY_EXPLAINED"
    return "STORAGE_DISK_USAGE_DISCREPANCY_PARTIALLY_EXPLAINED"


def next_gate(verified_disposable_bytes: int) -> str:
    """Phase 17. >=1GB of verified-disposable candidates recommends the
    bounded-cleanup-authorization gate; otherwise the original Parquet
    dry-run gate remains the recommendation."""
    if verified_disposable_bytes >= GB_DECIMAL:
        return "BATCH-STORAGE-BOUNDED-CLEANUP-AUTHORIZATION-V1"
    return "BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1"
