"""Layered archive verification (Phase 10): structural, accounting,
scientific-parity, and manifest checks, returning exactly one
`VerificationState`. No failed state is ever treated as archive-ready --
`VERIFIED_DISPOSABLE` is the only state a caller may act on.
"""
from __future__ import annotations

from dataclasses import dataclass

VERIFIED_DISPOSABLE = "VERIFIED_DISPOSABLE"
FAILED_SCHEMA = "FAILED_SCHEMA"
FAILED_ACCOUNTING = "FAILED_ACCOUNTING"
FAILED_CONTENT_PARITY = "FAILED_CONTENT_PARITY"
FAILED_CHECKSUM = "FAILED_CHECKSUM"
FAILED_MANIFEST = "FAILED_MANIFEST"
FAILED_RESOURCE_LIMIT = "FAILED_RESOURCE_LIMIT"
FAILED_SOURCE_CHANGED = "FAILED_SOURCE_CHANGED"
FAILED_UNKNOWN = "FAILED_UNKNOWN"

VERIFICATION_STATES = frozenset({
    VERIFIED_DISPOSABLE, FAILED_SCHEMA, FAILED_ACCOUNTING, FAILED_CONTENT_PARITY,
    FAILED_CHECKSUM, FAILED_MANIFEST, FAILED_RESOURCE_LIMIT, FAILED_SOURCE_CHANGED, FAILED_UNKNOWN,
})

FAILED_STATES = VERIFICATION_STATES - {VERIFIED_DISPOSABLE}


@dataclass(frozen=True)
class VerificationResult:
    state: str
    reasons: tuple[str, ...]

    @property
    def is_verified(self) -> bool:
        return self.state == VERIFIED_DISPOSABLE


def verify_structural(*, parquet_readable: bool, schema_matches: bool, compression: str,
                       expected_compression: str, extra_columns: tuple[str, ...],
                       missing_columns: tuple[str, ...]) -> VerificationResult:
    if not parquet_readable:
        return VerificationResult(FAILED_SCHEMA, ("parquet file not readable",))
    reasons = []
    if not schema_matches:
        reasons.append("schema does not match archive contract")
    if compression != expected_compression:
        reasons.append(f"compression {compression!r} != expected {expected_compression!r}")
    if extra_columns:
        reasons.append(f"unexpected extra columns: {extra_columns}")
    if missing_columns:
        reasons.append(f"missing required columns: {missing_columns}")
    if reasons:
        return VerificationResult(FAILED_SCHEMA, tuple(reasons))
    return VerificationResult(VERIFIED_DISPOSABLE, ())


def verify_accounting(*, row_count: int, expected_row_count: int, min_id: int | None,
                       max_id: int | None, expected_min_id: int | None, expected_max_id: int | None,
                       duplicate_count: int, null_count_mismatches: tuple[str, ...],
                       watermark_value: int) -> VerificationResult:
    reasons = []
    if row_count != expected_row_count:
        reasons.append(f"row_count {row_count} != expected {expected_row_count}")
    if min_id != expected_min_id or max_id != expected_max_id:
        reasons.append(f"id range ({min_id},{max_id}) != expected ({expected_min_id},{expected_max_id})")
    if duplicate_count != 0:
        reasons.append(f"duplicate_count={duplicate_count}")
    if null_count_mismatches:
        reasons.append(f"null-count mismatches: {null_count_mismatches}")
    if max_id is not None and max_id > watermark_value:
        reasons.append(f"max_id {max_id} exceeds watermark {watermark_value}")
    if reasons:
        return VerificationResult(FAILED_ACCOUNTING, tuple(reasons))
    return VerificationResult(VERIFIED_DISPOSABLE, ())


def verify_scientific_parity(*, source_hash: str, parquet_hash: str, mismatch_count: int) -> VerificationResult:
    if source_hash != parquet_hash or mismatch_count != 0:
        return VerificationResult(
            FAILED_CONTENT_PARITY,
            (f"source_hash={source_hash} parquet_hash={parquet_hash} mismatch_count={mismatch_count}",))
    return VerificationResult(VERIFIED_DISPOSABLE, ())


def verify_checksum(*, expected_sha256: str, actual_sha256: str) -> VerificationResult:
    if expected_sha256 != actual_sha256:
        return VerificationResult(FAILED_CHECKSUM, (f"{expected_sha256} != {actual_sha256}",))
    return VerificationResult(VERIFIED_DISPOSABLE, ())


def verify_manifest(*, manifest: dict, expected_parquet_sha256: str, expected_scientific_hash: str,
                     expected_partition_id: str) -> VerificationResult:
    reasons = []
    if manifest.get("parquet_sha256") != expected_parquet_sha256:
        reasons.append("manifest parquet_sha256 mismatch")
    if manifest.get("ordered_scientific_content_hash") != expected_scientific_hash:
        reasons.append("manifest scientific-content-hash mismatch")
    if manifest.get("partition_id") != expected_partition_id:
        reasons.append("manifest partition_id mismatch")
    if manifest.get("production_status") != "DISPOSABLE_NOT_PRODUCTION":
        reasons.append("manifest production_status is not DISPOSABLE_NOT_PRODUCTION")
    if manifest.get("purge_authorization") != "PROHIBITED":
        reasons.append("manifest purge_authorization is not PROHIBITED")
    if reasons:
        return VerificationResult(FAILED_MANIFEST, tuple(reasons))
    return VerificationResult(VERIFIED_DISPOSABLE, ())


def verify_full(*results: VerificationResult) -> VerificationResult:
    """Combines a sequence of layer results into one final result --
    the first failure short-circuits and is returned; only if every
    layer independently returned VERIFIED_DISPOSABLE does the combined
    result do so too."""
    for r in results:
        if not r.is_verified:
            return r
    return VerificationResult(VERIFIED_DISPOSABLE, ())
