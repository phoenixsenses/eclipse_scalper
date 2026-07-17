"""Freshness / trust evaluation for the canonical read-only operator dashboard.

Given a source timestamp, a read timestamp and a freshness threshold, decide the
:class:`~dashboard.backend.view_models.TrustState`. Centralised so every adapter
applies identical, testable rules:

- source missing / unparseable  -> MISSING / MALFORMED
- source in the future beyond a small skew allowance -> MALFORMED (clock skew)
- age <= threshold               -> CURRENT
- age  > threshold               -> STALE

The evaluator never raises on bad input; it returns a typed result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from dashboard.backend.view_models import ParseStatus, TrustState

# Allowance for benign clock skew between the writer of an artifact and this
# process before a future-dated timestamp is treated as malformed.
DEFAULT_CLOCK_SKEW_TOLERANCE_SEC = 120.0


@dataclass(frozen=True)
class FreshnessResult:
    trust_state: TrustState
    age_seconds: float | None
    reason: str


def evaluate_freshness(
    *,
    source_timestamp: float | None,
    freshness_threshold_seconds: float | None,
    read_timestamp: float | None = None,
    parse_status: ParseStatus = ParseStatus.OK,
    clock_skew_tolerance_sec: float = DEFAULT_CLOCK_SKEW_TOLERANCE_SEC,
) -> FreshnessResult:
    """Return a :class:`FreshnessResult` for one artifact.

    ``parse_status`` short-circuits: a MISSING/MALFORMED/EMPTY source cannot be
    CURRENT regardless of timestamps (fail-closed).
    """
    now = float(read_timestamp) if read_timestamp is not None else time.time()

    if parse_status == ParseStatus.MISSING:
        return FreshnessResult(TrustState.MISSING, None, "source_missing")
    if parse_status == ParseStatus.MALFORMED:
        return FreshnessResult(TrustState.MALFORMED, None, "source_malformed")
    if parse_status == ParseStatus.EMPTY:
        return FreshnessResult(TrustState.MISSING, None, "source_empty")

    if source_timestamp is None:
        # Parsed OK but no usable timestamp -> we cannot assert freshness.
        return FreshnessResult(TrustState.INFERRED, None, "no_source_timestamp")

    try:
        src = float(source_timestamp)
    except (TypeError, ValueError):
        return FreshnessResult(TrustState.MALFORMED, None, "unparseable_timestamp")

    if src <= 0:
        return FreshnessResult(TrustState.MALFORMED, None, "nonpositive_timestamp")

    age = now - src
    if age < -abs(clock_skew_tolerance_sec):
        # Source claims to be from the future beyond tolerance -> suspect clock.
        return FreshnessResult(TrustState.MALFORMED, age, "future_timestamp_clock_skew")

    # Clamp small negative ages (within skew tolerance) to 0 for display.
    if age < 0:
        age = 0.0

    if freshness_threshold_seconds is None:
        # No threshold configured: we know the age but not the policy -> inferred.
        return FreshnessResult(TrustState.INFERRED, age, "no_threshold_configured")

    if age <= float(freshness_threshold_seconds):
        return FreshnessResult(TrustState.CURRENT, age, "fresh")
    return FreshnessResult(TrustState.STALE, age, "age_exceeds_threshold")


def apply_freshness(vm, *, clock_skew_tolerance_sec: float = DEFAULT_CLOCK_SKEW_TOLERANCE_SEC) -> None:
    """Mutate a :class:`PanelViewModel` in place: compute age + trust_state from
    its ``source_timestamp`` / ``freshness_threshold_seconds`` / ``parse_status``.

    Adapters set the raw metadata; this applies the shared trust policy so the
    stale-GREEN prohibition is enforced uniformly. If the adapter already marked
    a hard trust failure (MISSING/MALFORMED), that is preserved.
    """
    res = evaluate_freshness(
        source_timestamp=vm.source_timestamp,
        freshness_threshold_seconds=vm.freshness_threshold_seconds,
        read_timestamp=vm.read_timestamp,
        parse_status=vm.parse_status,
        clock_skew_tolerance_sec=clock_skew_tolerance_sec,
    )
    vm.age_seconds = res.age_seconds
    vm.trust_state = res.trust_state
    vm.add_reason(res.reason)
