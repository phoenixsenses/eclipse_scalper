"""Disposable archive catalog (Phase 11). Operates only beneath an
explicitly approved disposable root -- never a production archive
directory. Registers verified partitions, rejects duplicate-conflicting
identities, path escapes, unverified registrations, and any attempt to
mutate an already-verified entry (a repaired archive requires a new
version identity; prior history is never overwritten).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ami.storage.verifier import VERIFIED_DISPOSABLE


class CatalogPathEscapeError(Exception):
    """Raised when a registered archive/manifest path resolves outside
    the catalog's approved disposable root."""


class CatalogProductionPathError(Exception):
    """Raised when a registered path looks like a production archive
    location -- this catalog never accepts one."""


class CatalogConflictError(Exception):
    """Raised on a duplicate partition_id with a different content hash,
    or an attempt to overwrite an already-verified entry."""


class CatalogUnverifiedRegistrationError(Exception):
    """Raised when attempting to register a partition whose verification
    state is not VERIFIED_DISPOSABLE."""


@dataclass
class CatalogEntry:
    partition_id: str
    table: str
    symbol: str
    utc_year: int
    utc_month: int
    archive_path: str
    manifest_path: str
    verification_state: str
    scientific_content_hash: str
    parquet_sha256: str
    source_watermark_value: int
    unresolved_gap_count: int
    repair_status: str
    production_status: str
    purge_authorization: str
    version: int = 1


class DisposableArchiveCatalog:
    """In-memory catalog, one instance per disposable root. Not
    persisted to any production location by this batch."""

    def __init__(self, disposable_root: str, production_roots_to_reject: tuple[str, ...] = ()):
        self.disposable_root = os.path.normpath(os.path.abspath(disposable_root))
        self._production_roots = tuple(
            os.path.normpath(os.path.abspath(p)) for p in production_roots_to_reject)
        self._entries: dict[str, CatalogEntry] = {}
        self._history: dict[str, list[CatalogEntry]] = {}

    def _validate_path(self, path: str) -> None:
        norm = os.path.normpath(os.path.abspath(path))
        for prod in self._production_roots:
            if norm == prod or norm.startswith(prod + os.sep):
                raise CatalogProductionPathError(f"{path!r} resolves inside a production root {prod!r}")
        if not (norm == self.disposable_root or norm.startswith(self.disposable_root + os.sep)):
            raise CatalogPathEscapeError(f"{path!r} escapes the catalog's disposable root {self.disposable_root!r}")

    def register(self, entry: CatalogEntry) -> CatalogEntry:
        if entry.verification_state != VERIFIED_DISPOSABLE:
            raise CatalogUnverifiedRegistrationError(
                f"cannot register partition_id={entry.partition_id!r} with state {entry.verification_state!r}")
        if entry.purge_authorization != "PROHIBITED":
            raise CatalogConflictError("catalog will not accept a non-PROHIBITED purge_authorization")
        self._validate_path(entry.archive_path)
        self._validate_path(entry.manifest_path)

        existing = self._entries.get(entry.partition_id)
        if existing is not None:
            if existing.scientific_content_hash == entry.scientific_content_hash:
                return existing  # idempotent re-registration of the identical content
            raise CatalogConflictError(
                f"partition_id={entry.partition_id!r} already registered with a DIFFERENT content hash "
                f"({existing.scientific_content_hash} != {entry.scientific_content_hash}); "
                "a repaired archive requires a new version, not an overwrite")

        self._entries[entry.partition_id] = entry
        self._history.setdefault(entry.partition_id, []).append(entry)
        return entry

    def register_new_version(self, prior_partition_id: str, entry: CatalogEntry) -> CatalogEntry:
        """Explicit repaired-archive path: registers `entry` as a new
        version, preserving `prior_partition_id`'s history untouched."""
        if entry.partition_id == prior_partition_id:
            raise CatalogConflictError("a new version must have a distinct partition_id")
        prior_history = self._history.get(prior_partition_id, [])
        entry.version = len(prior_history) + 1 if prior_history else 1
        return self.register(entry)

    def get(self, partition_id: str) -> CatalogEntry | None:
        return self._entries.get(partition_id)

    def history(self, partition_id: str) -> tuple[CatalogEntry, ...]:
        return tuple(self._history.get(partition_id, ()))

    def all_entries(self) -> tuple[CatalogEntry, ...]:
        return tuple(self._entries.values())
