"""The thinnest possible mapping from an E-DER V1 DETECTED event to the frozen
`eclipse.alpha.trade_candidate` contract.

This module is deliberately boring. It has no I/O, no transport, no state, no
clock of its own and no dependency on the Master Center. It is one pure
function over a per-process boundary.

**Why it is written as a snapshot rather than a wrapper.** In
`tools/e_der_v1_forward_shadow.py::run_cycle`, the DETECTED dict is stored by
reference in `state["pending"]` and then mutated in place by `mature()`, which
writes `gross_return_bps` and `net_return_bps` into it. `make_event` also emits
those keys up front as `None`. So an adapter that held a reference and published
later would publish a sealed arm's realised outcome. Copying at call time is the
defence; the refusals below are the second one.

**Order of refusals matters** (review B3). Mutation is detected *before*
ordinary eligibility, so a real matured event reports the leak that actually
happened rather than a generic "wrong status". The refusal a reviewer sees
should name the real problem.

The ledger remains the record. This produces a notification, nothing more.
"""

from __future__ import annotations

from typing import Any, Mapping

from eclipse_shared.schemas import Direction, TradeCandidate

from . import manifest, publication_epoch

MINUTE_MS = 60_000


class AdapterRefusal(Exception):
    """Base class. Refusing is always correct; guessing never is."""


class OutcomeLeak(AdapterRefusal):
    """The event has been matured: an outcome or a mutation marker is present.

    Raised rather than filtered. A populated outcome means the caller is holding
    a mutated object, and silently dropping the field would hide that — the
    caller would keep publishing from a reference it should not have.
    """


class ProducerMismatch(AdapterRefusal):
    """Not the frozen V1 paper-shadow producer.

    Shape is not provenance (review B2). Before this check, any dict carrying
    the right `event` and `status` was labelled E-DER V1 — including one
    classified RETROSPECTIVE with `paper_only=False`.
    """


class NotEligible(AdapterRefusal):
    """A genuine V1 event, but not at a publishable point in its lifecycle."""


class NoPublicationEpoch(AdapterRefusal):
    """The publisher never started, so there is no no-backfill boundary.

    Fail closed: without an epoch there is no guarantee to make (review B1).
    """


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def to_trade_candidate(
    event: Mapping[str, Any],
    *,
    publication_epoch_ms: int | None = None,
) -> TradeCandidate:
    """Map one DETECTED event to a validated `TradeCandidate`.

    `publication_epoch_ms` lets the 03C caller supply its own runtime boundary;
    by default the process epoch from `publication_epoch.start()` is used.

    Raises an `AdapterRefusal` subclass rather than returning something partial.
    The returned model is frozen and its context is copied, so later mutation of
    *event* cannot reach it.
    """
    # ---- 1. Mutation first, so the refusal names the real problem (B3) -----
    leaked = sorted(
        field for field in manifest.REQUIRED_T0_MARKERS if event.get(field) is not None
    )
    if leaked:
        raise OutcomeLeak(
            "event has been matured; realised outcome present: "
            + ", ".join(leaked)
        )

    mutated = sorted(field for field in manifest.MUTATION_MARKERS if field in event)
    if mutated:
        raise OutcomeLeak(
            "event has been matured; mutation marker present: " + ", ".join(mutated)
        )

    # ---- 2. Provenance: is this the frozen V1 paper-shadow producer? (B2) --
    for field, expected in manifest.PRODUCER_IDENTITY.items():
        actual = event.get(field)
        if actual != expected or type(actual) is not type(expected):
            raise ProducerMismatch(
                f"{field}: expected {expected!r} from the frozen V1 producer, got {actual!r}"
            )

    missing = sorted(manifest.REQUIRED_T0_MARKERS - set(event))
    if missing:
        raise ProducerMismatch(
            "not the T0 shape; make_event emits these present-and-None, absent here: "
            + ", ".join(missing)
        )

    # ---- 3. Lifecycle: a real V1 event, but is it at T0? -------------------
    if event.get("event") != manifest.ELIGIBLE_EVENT:
        raise NotEligible(
            f"expected event={manifest.ELIGIBLE_EVENT!r}, got {event.get('event')!r}"
        )
    if event.get("status") != manifest.ELIGIBLE_STATUS:
        raise NotEligible(
            f"expected status={manifest.ELIGIBLE_STATUS!r}, got {event.get('status')!r}"
        )
    if event.get("data_quality_status") != manifest.T0_DATA_QUALITY:
        raise NotEligible(
            f"expected data_quality_status={manifest.T0_DATA_QUALITY!r}, "
            f"got {event.get('data_quality_status')!r}"
        )

    # ---- 4. P2: no backfill through the live path (B1) ---------------------
    boundary = publication_epoch_ms if publication_epoch_ms is not None else publication_epoch.current()
    if boundary is None:
        raise NoPublicationEpoch(
            "no publication epoch: call publication_epoch.start() at publisher "
            "startup before publishing"
        )
    anchor_ts = int(event["anchor_ts"])
    if anchor_ts < int(boundary):
        raise NotEligible(
            f"anchor {anchor_ts} precedes this process's publication epoch "
            f"{boundary}; backfill is not publishable"
        )

    # ---- 5. Horizon from the event's own frozen timing ---------------------
    entry_ms = int(event["entry_ms"])
    boundary_ms = int(event["boundary_ms"])
    horizon_minutes = (boundary_ms - entry_ms) / MINUTE_MS
    if horizon_minutes <= 0:
        raise NotEligible(f"non-positive horizon: entry={entry_ms} boundary={boundary_ms}")

    # ---- 6. Context: closed whitelist, copied, stringified -----------------
    context = {
        key: _stringify(event[key])
        for key in manifest.CONTEXT_WHITELIST
        if key in event and event[key] is not None
    }
    context["integration_contract"] = manifest.INTEGRATION_CONTRACT

    # ---- 7. Validate against the frozen contract --------------------------
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
