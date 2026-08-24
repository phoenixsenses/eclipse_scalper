"""Bounded storage archive tooling (BATCH-STORAGE-ROTATION-RETENTION-
BOUNDED-IMPLEMENTATION-V1). Production-quality, fail-closed, deterministic
Parquet/ZSTD archive rehearsal tooling for the three accepted archive-
eligible raw tables (`mark_prices`, `agg_trades`, `book_ticker`).

Production archival, scheduling, and purge are NOT implemented anywhere
in this package -- every operation is disposable-output-only, confined to
`.runtime_temp`/`.pytest_temp`, and every manifest hardcodes
`production_status=DISPOSABLE_NOT_PRODUCTION` / `purge_authorization=
PROHIBITED`.
"""
from ami.storage.policy import StoragePolicy, DEFAULT_POLICY, PolicyValidationError
from ami.storage.registry import SOURCE_TABLE_REGISTRY, get_table_spec, allowlisted_tables, UnknownTableError

__all__ = [
    "StoragePolicy", "DEFAULT_POLICY", "PolicyValidationError",
    "SOURCE_TABLE_REGISTRY", "get_table_spec", "allowlisted_tables", "UnknownTableError",
]
