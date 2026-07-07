"""Closed-UTC-month partition identity and read-only partition planner
(Phases 4-5). `PartitionIdentity` is immutable and self-validating
(current/future/partial months are rejected at construction, not
discovered later); `plan_partition()` performs only bounded, indexed,
read-only queries against an already-open connection and returns an
`ArchivePlan` whose eligibility flags are computed, never asserted.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
from dataclasses import dataclass

from ami.storage.registry import SourceTableSpec, get_table_spec

ARCHIVE_CONTRACT_VERSION = "v1"


class PartitionValidationError(Exception):
    """Fail-closed: current month, future month, partial month, or a
    non-UTC/malformed boundary all raise this -- never silently coerced
    to the nearest valid month."""


@dataclass(frozen=True)
class PartitionIdentity:
    table: str
    symbol: str
    venue: str
    market_segment: str
    utc_year: int
    utc_month: int
    partition_start_ms: int
    partition_end_ms: int
    source_watermark_field: str
    source_watermark_value: int
    archive_contract_version: str = ARCHIVE_CONTRACT_VERSION
    schema_fingerprint: str = ""

    @property
    def partition_id(self) -> str:
        raw = (f"{self.table}|{self.symbol}|{self.venue}|{self.market_segment}|"
               f"{self.utc_year:04d}-{self.utc_month:02d}|{self.archive_contract_version}")
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    @property
    def archive_relative_path(self) -> str:
        return (f"{self.table}/{self.symbol}/{self.utc_year:04d}/{self.utc_month:02d}/"
                f"part-{self.partition_id}.parquet")


def build_partition_identity(*, table: str, symbol: str, utc_year: int, utc_month: int,
                              source_watermark_value: int, now: dt.datetime | None = None,
                              active_retention_days: int = 30,
                              schema_fingerprint: str = "") -> PartitionIdentity:
    """Constructs and validates a closed-month partition identity.
    Raises `PartitionValidationError` for current/future/partial months
    or a violation of the active-retention-horizon protection."""
    spec = get_table_spec(table)  # raises UnknownTableError if not allowlisted
    if not (1 <= utc_month <= 12):
        raise PartitionValidationError(f"malformed utc_month={utc_month}")
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        raise PartitionValidationError("`now` must be timezone-aware UTC")

    start = dt.datetime(utc_year, utc_month, 1, tzinfo=dt.timezone.utc)
    days_in_month = calendar.monthrange(utc_year, utc_month)[1]
    end = start + dt.timedelta(days=days_in_month)

    if start > now:
        raise PartitionValidationError("future UTC month is never archive-eligible")
    if end > now:
        # For calendar months this single half-open-period check covers
        # both "this is the current UTC month" and "this month has not
        # yet fully closed" -- they are the same condition (a month's end
        # boundary cannot be in the future without `now` being inside it).
        if (start.year, start.month) == (now.year, now.month):
            raise PartitionValidationError("current UTC month is never archive-eligible "
                                            "(partial UTC month, has not fully closed yet)")
        raise PartitionValidationError(f"partial UTC month rejected: {utc_year:04d}-{utc_month:02d} "
                                        "has not fully closed yet")

    active_horizon_start = now - dt.timedelta(days=active_retention_days)
    if end > active_horizon_start:
        raise PartitionValidationError(
            f"partition end {end.isoformat()} falls inside the {active_retention_days}-day "
            f"active retention horizon (starts {active_horizon_start.isoformat()}) -- not archive-eligible yet")

    return PartitionIdentity(
        table=table, symbol=symbol, venue=spec.venue, market_segment=spec.market_segment,
        utc_year=utc_year, utc_month=utc_month,
        partition_start_ms=int(start.timestamp() * 1000), partition_end_ms=int(end.timestamp() * 1000),
        source_watermark_field=spec.stable_ordering_field, source_watermark_value=source_watermark_value,
        schema_fingerprint=schema_fingerprint,
    )


# ---------------------------------------------------------------------------
# Planner (Phase 5)
# ---------------------------------------------------------------------------

PLAN_ARCHIVE_ELIGIBLE = "ARCHIVE_PLAN_ELIGIBLE"
PLAN_BLOCKED_BY_SCHEMA = "ARCHIVE_PLAN_BLOCKED_BY_SCHEMA"
PLAN_BLOCKED_BY_INDEX = "ARCHIVE_PLAN_BLOCKED_BY_INDEX"
PLAN_BLOCKED_BY_SOURCE_GAP = "ARCHIVE_PLAN_BLOCKED_BY_SOURCE_GAP"
PLAN_BLOCKED_BY_REPAIR = "ARCHIVE_PLAN_BLOCKED_BY_REPAIR"
PLAN_BLOCKED_BY_RESOURCE_LIMIT = "ARCHIVE_PLAN_BLOCKED_BY_RESOURCE_LIMIT"
PLAN_BLOCKED_BY_UNKNOWN_SOURCE_SEMANTICS = "ARCHIVE_PLAN_BLOCKED_BY_UNKNOWN_SOURCE_SEMANTICS"

PLAN_STATES = frozenset({
    PLAN_ARCHIVE_ELIGIBLE, PLAN_BLOCKED_BY_SCHEMA, PLAN_BLOCKED_BY_INDEX,
    PLAN_BLOCKED_BY_SOURCE_GAP, PLAN_BLOCKED_BY_REPAIR, PLAN_BLOCKED_BY_RESOURCE_LIMIT,
    PLAN_BLOCKED_BY_UNKNOWN_SOURCE_SEMANTICS,
})


@dataclass(frozen=True)
class ArchivePlan:
    partition: PartitionIdentity
    plan_state: str
    estimated_row_count: int | None
    estimated_source_bytes: int | None
    indexes_present: tuple[str, ...]
    unresolved_gap_count: int
    repair_status: str
    archive_rehearsal_eligible: bool
    production_activation_eligible: bool  # always False in this batch
    purge_eligible: bool  # always False in this batch
    blockers: tuple[str, ...]


def plan_partition(conn, *, table: str, symbol: str, utc_year: int, utc_month: int,
                    max_estimated_rows: int = 50_000_000,
                    max_estimated_bytes: int = 2 * 1024 ** 3,
                    now: dt.datetime | None = None) -> ArchivePlan:
    """Read-only. `conn` must already be opened read-only by the caller
    (this function never opens a connection itself -- see
    `ami.storage.source_access`). Every query here is bounded and
    index-backed (MIN/MAX/COUNT on indexed columns), never a full scan."""
    spec = get_table_spec(table)

    actual_indexes = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)).fetchall()}
    missing_indexes = [i for i in spec.expected_indexes if i not in actual_indexes]

    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        identity = build_partition_identity(
            table=table, symbol=symbol, utc_year=utc_year, utc_month=utc_month,
            source_watermark_value=0, now=now)
    except PartitionValidationError as exc:
        # Still return a structured plan (never raise out of the planner)
        dummy = PartitionIdentity(table=table, symbol=symbol, venue=spec.venue,
                                   market_segment=spec.market_segment, utc_year=utc_year,
                                   utc_month=utc_month, partition_start_ms=0, partition_end_ms=0,
                                   source_watermark_field=spec.stable_ordering_field,
                                   source_watermark_value=0)
        return ArchivePlan(partition=dummy, plan_state=PLAN_BLOCKED_BY_UNKNOWN_SOURCE_SEMANTICS,
                            estimated_row_count=None, estimated_source_bytes=None,
                            indexes_present=tuple(sorted(actual_indexes)), unresolved_gap_count=0,
                            repair_status="NOT_EVALUATED", archive_rehearsal_eligible=False,
                            production_activation_eligible=False, purge_eligible=False,
                            blockers=(str(exc),))

    row = conn.execute(
        f"SELECT COUNT(*), MAX({spec.stable_ordering_field}) FROM {table} "
        f"WHERE {spec.symbol_field}=? AND {spec.partition_ts_field}>=? AND {spec.partition_ts_field}<?",
        (symbol, identity.partition_start_ms, identity.partition_end_ms)).fetchone()
    row_count, watermark = row[0], row[1]
    identity = PartitionIdentity(
        **{**identity.__dict__, "source_watermark_value": watermark or 0})

    gap_count = 0
    if spec.gap_ledger_stream:
        gap_row = conn.execute(
            "SELECT COUNT(*) FROM gaps WHERE stream=? AND start_ts_ms<? AND "
            "(end_ts_ms IS NULL OR end_ts_ms>=?) AND resolved_bool=0",
            (spec.gap_ledger_stream, identity.partition_end_ms, identity.partition_start_ms)).fetchone()
        gap_count = gap_row[0]

    blockers: list[str] = []
    if missing_indexes:
        blockers.append(f"missing indexes: {missing_indexes}")
    est_bytes = row_count * 64  # conservative per-row upper-bound estimate
    if row_count > max_estimated_rows or est_bytes > max_estimated_bytes:
        blockers.append(f"resource estimate exceeds cap: rows={row_count} est_bytes={est_bytes}")

    if missing_indexes:
        plan_state = PLAN_BLOCKED_BY_INDEX
    elif row_count > max_estimated_rows or est_bytes > max_estimated_bytes:
        plan_state = PLAN_BLOCKED_BY_RESOURCE_LIMIT
    else:
        plan_state = PLAN_ARCHIVE_ELIGIBLE

    rehearsal_eligible = plan_state == PLAN_ARCHIVE_ELIGIBLE and row_count > 0

    return ArchivePlan(
        partition=identity, plan_state=plan_state,
        estimated_row_count=row_count, estimated_source_bytes=est_bytes,
        indexes_present=tuple(sorted(actual_indexes)), unresolved_gap_count=gap_count,
        repair_status=(spec.repair_layer or "NOT_APPLICABLE"),
        archive_rehearsal_eligible=rehearsal_eligible,
        production_activation_eligible=False,  # hard-coded false, this batch never sets true
        purge_eligible=False,  # hard-coded false, this batch never sets true
        blockers=tuple(blockers),
    )
