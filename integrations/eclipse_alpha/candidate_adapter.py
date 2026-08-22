"""The thinnest possible mapping from an E-DER V1 DETECTED event to the frozen
`eclipse.alpha.trade_candidate` contract.

This module is deliberately boring. It has no I/O, no transport, no state, no
clock and no dependency on the Master Center. It is one pure function.

**Why it is written as a snapshot rather than a wrapper.** In
`tools/e_der_v1_forward_shadow.py::run_cycle`, the DETECTED dict is stored by
reference in `state["pending"]` and then mutated in place by `mature()`, which
writes `gross_return_bps` and `net_return_bps` into it. `make_event` also emits
those keys up front as `None`. So an adapter that held a reference and published
later would publish a sealed arm's realised outcome. Copying at call time is the
whole defence, and the refusals below are the second one.

The ledger remains the record. This produces a notification, nothing more.
"""

from __future__ import annotations

from typing import Any, Mapping

from eclipse_shared.schemas import Direction, TradeCandidate

from . import manifest

MINUTE_MS = 60_000


class AdapterRefusal(Exception):
    """Base class. Refusing is always correct; guessing never is."""


class NotEligible(AdapterRefusal):
    """The event is not a fresh, in-window DETECTED event."""


class OutcomeLeak(AdapterRefusal):
    """An outcome field is populated, so this event has been matured.

    Raised rather than filtered: a populated outcome means the caller is holding
    a mutated object, and silently dropping the field would hide that.
    """


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def to_trade_candidate(event: Mapping[str, Any]) -> TradeCandidate:
    """Map one DETECTED event to a validated `TradeCandidate`.

    Raises `NotEligible` or `OutcomeLeak` rather than returning something
    partial. The returned model is frozen, and its context is copied — later
    mutation of *event* cannot reach it.
    """
    # 1. Only a fresh detection. A matured event is a different thing wearing
    #    the same dict.
    if event.get("event") != manifest.ELIGIBLE_EVENT:
        raise NotEligible(
            f"expected event={manifest.ELIGIBLE_EVENT!r}, got {event.get('event')!r}"
        )
    if event.get("status") != manifest.ELIGIBLE_STATUS:
        raise NotEligible(
            f"expected status={manifest.ELIGIBLE_STATUS!r}, got {event.get('status')!r}"
        )

    # 2. The seal. Present-and-None is the normal T0 shape; anything else means
    #    mature() has already run against this object.
    for field in sorted(manifest.OUTCOME_FIELDS):
        if event.get(field) is not None:
            raise OutcomeLeak(
                f"{field} is populated: this event has been matured and is sealed"
            )

    # 3. P2 — no backfill through the live path.
    anchor_ts = int(event["anchor_ts"])
    if anchor_ts < manifest.INTEGRATION_BOUNDARY_MS:
        raise NotEligible(
            f"anchor {anchor_ts} precedes the integration boundary "
            f"{manifest.INTEGRATION_BOUNDARY_MS}; backfill is not publishable"
        )

    # 4. Horizon from the event's own frozen timing. Both instants are T0 facts.
    entry_ms = int(event["entry_ms"])
    boundary_ms = int(event["boundary_ms"])
    horizon_minutes = (boundary_ms - entry_ms) / MINUTE_MS
    if horizon_minutes <= 0:
        raise NotEligible(f"non-positive horizon: entry={entry_ms} boundary={boundary_ms}")

    # 5. Context: closed whitelist, copied, stringified.
    context = {
        key: _stringify(event[key])
        for key in manifest.CONTEXT_WHITELIST
        if key in event and event[key] is not None
    }
    context["integration_contract"] = manifest.INTEGRATION_CONTRACT

    # 6. Validate against the frozen contract. extra="forbid" means a stray
    #    sizing field would raise here rather than travel.
    return TradeCandidate(
        candidate_id=str(event["event_id"]),
        arm=manifest.ARM,
        arm_version=manifest.ARM_VERSION,
        anchor_id=str(anchor_ts),
        symbol=str(event["symbol"]),
        direction=Direction(manifest.DIRECTION),
        horizon_minutes=horizon_minutes,
        context=context,
    )
