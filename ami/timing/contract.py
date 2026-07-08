"""Time and Lookahead Contract — Forward Observatory §6.

All timestamps are UTC epoch milliseconds (§6.5 — canonical storage is UTC;
local-time display is a UI concern outside this module's scope).

Required timestamps per feature/state (§6.1):
    event_ts          -- market time the underlying fact occurred
    available_at_ts    -- time the raw source became available to the system
    known_at_ts         -- earliest time the computed feature could validly be used

Mandatory rule (§6.2): known_at_ts <= observer_trigger_ts, else REJECT_REASON
= FUTURE_INFORMATION.

Partial candles (§6.3): an unfinished daily/weekly candle must be tagged
PARTIAL_CANDLE and must never be used/stored/displayed as a closed-candle
state.

Missing data (§6.4): missing data must never be converted to zero. This
module exposes the allowed quality states as an enum; it is the calling
engine's responsibility to store/propagate the explicit state rather than
substituting a numeric zero.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class DataQualityState(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    STALE = "STALE"
    GAPPED = "GAPPED"
    NOT_COLLECTED = "NOT_COLLECTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CandleState(str, Enum):
    CLOSED = "CLOSED"
    PARTIAL_CANDLE = "PARTIAL_CANDLE"


class LookaheadViolation(Exception):
    """Raised on any Time and Lookahead Contract violation (§6.2/§6.3)."""


def known_at_ok(known_at_ts: int, observer_trigger_ts: int) -> bool:
    """§6.2 mandatory rule, non-raising form."""
    return known_at_ts <= observer_trigger_ts


def enforce_known_at(known_at_ts: int, observer_trigger_ts: int) -> None:
    """§6.2 mandatory rule. Raises LookaheadViolation (FUTURE_INFORMATION) if violated."""
    if not known_at_ok(known_at_ts, observer_trigger_ts):
        raise LookaheadViolation(
            f"FUTURE_INFORMATION: known_at_ts={known_at_ts} > observer_trigger_ts={observer_trigger_ts}"
        )


def reject_if_partial_candle(candle_state: CandleState) -> None:
    """§6.3: a PARTIAL_CANDLE must never be used as a closed-candle state."""
    if candle_state == CandleState.PARTIAL_CANDLE:
        raise LookaheadViolation("PARTIAL_CANDLE must never be used as a closed-candle state")


@dataclass(frozen=True)
class TimestampedValue:
    """A single feature/state observation carrying the full §6.1 timestamp triple."""

    event_ts: int
    available_at_ts: int
    known_at_ts: int
    quality: DataQualityState = DataQualityState.AVAILABLE
    candle_state: CandleState = CandleState.CLOSED

    def __post_init__(self) -> None:
        if self.event_ts > self.available_at_ts:
            raise ValueError("event_ts must not be after available_at_ts (§6.1 ordering)")
        if self.available_at_ts > self.known_at_ts:
            raise ValueError("available_at_ts must not be after known_at_ts (§6.1 ordering)")

    def validate_for_use(self, observer_trigger_ts: int) -> None:
        """Full contract check an engine must call before consuming this value:
        timing (§6.2) + partial-candle (§6.3). Does not raise on §6.4 quality
        states -- the caller decides whether MISSING/STALE/etc. is acceptable
        for its own use case.
        """
        enforce_known_at(self.known_at_ts, observer_trigger_ts)
        reject_if_partial_candle(self.candle_state)
