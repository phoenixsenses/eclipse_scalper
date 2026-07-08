"""Immutable event identity + real-vs-proxy source-quality (Protocol §8).

event_id is a deterministic hash of the natural key (symbol, event_family,
anchor_ts_ms, source_artifact_id). Re-ingesting the same underlying record
must always produce the same event_id -- IDs are never reassigned on rerun,
and no random/incrementing ID scheme is used anywhere in this module.

R-09 / CONFLICT-008: a real liquidation-triggered event and a purely
synthetic/approximated event (e.g. a matched random-time control) must never
be pooled into the same population without an explicit, visible label.
assert_not_pooled() enforces this as code, not just as documentation.
"""
from __future__ import annotations
import hashlib
from enum import Enum


class SourceQuality(str, Enum):
    REAL_LIQUIDATION = "REAL_LIQUIDATION"
    PROXY_CASCADE_6H_GAP = "PROXY_CASCADE_6H_GAP"
    PROXY_OTHER = "PROXY_OTHER"
    UNKNOWN = "UNKNOWN"


_PROXY_STATUSES = {SourceQuality.PROXY_CASCADE_6H_GAP, SourceQuality.PROXY_OTHER}


class PooledPopulationViolation(Exception):
    """Raised when REAL and PROXY events are combined into one population (R-09)."""


def generate_event_id(symbol: str, event_family: str, anchor_ts_ms: int, source_artifact_id: str) -> str:
    """Deterministic, immutable event_id. Same inputs -> same ID, always."""
    key = f"{symbol}|{event_family}|{anchor_ts_ms}|{source_artifact_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"EVT-{digest}"


def assert_not_pooled(source_qualities) -> None:
    """Raise if a population mixes REAL_LIQUIDATION with any PROXY_* status.

    source_qualities: iterable of SourceQuality (or their string values).
    UNKNOWN never triggers this guard by itself -- it is an unclassified
    data-quality gap, not an asserted proxy. Only an explicit PROXY_* status
    combined with REAL_LIQUIDATION is a violation.
    """
    values = {SourceQuality(v) if not isinstance(v, SourceQuality) else v for v in source_qualities}
    has_real = SourceQuality.REAL_LIQUIDATION in values
    has_proxy = bool(values & _PROXY_STATUSES)
    if has_real and has_proxy:
        raise PooledPopulationViolation(
            f"population mixes REAL_LIQUIDATION with proxy statuses {values & _PROXY_STATUSES} (R-09/CONFLICT-008)"
        )
