"""Regression tests for the independent Phase 03B review findings B1–B4.

Written before the fixes, against commit 47677f52. Each test fails there.

The theme across all four is the same: the adapter was checking *shape* where it
should have been checking *provenance and mutation*. A dict with the right two
keys was treated as a V1 T0 event; a matured event was refused for the wrong
reason; a static constant stood in for a runtime boundary.

Run:
    .venv\\Scripts\\python.exe -m pytest tests/test_eclipse_alpha_adapter_b_findings.py -q \\
        -p no:cacheprovider --basetemp=scratchpad/pytest_eclipse_alpha_b
"""

from __future__ import annotations

import pytest

from integrations.eclipse_alpha import manifest, publication_epoch
from integrations.eclipse_alpha.candidate_adapter import (
    NoPublicationEpoch,
    NotEligible,
    OutcomeLeak,
    ProducerMismatch,
    to_trade_candidate,
)

MINUTE_MS = 60_000
EPOCH = 1_800_000_000_000          # the process's publication epoch, in tests
ANCHOR = EPOCH + 5 * MINUTE_MS     # comfortably after it


@pytest.fixture(autouse=True)
def _fresh_process_epoch():
    """Each test starts like a freshly started process: no epoch yet."""
    publication_epoch.reset_for_tests()
    yield
    publication_epoch.reset_for_tests()


def detected_event(anchor: int = ANCHOR, **overrides) -> dict:
    """Exactly the shape `make_event` produces at T0 — including the four
    outcome keys present-and-None, and no `updated_at_utc`."""
    base_ms = (anchor // MINUTE_MS) * MINUTE_MS + MINUTE_MS
    event = {
        "event": "DETECTED",
        "protocol": manifest.ARM_VERSION,
        "classification": "PROSPECTIVE_FORWARD",
        "event_id": f"E:ETHUSDT:{anchor}",
        "symbol": "ETHUSDT",
        "anchor_ts": anchor,
        "base_ms": base_ms,
        "entry_ms": base_ms + 31 * MINUTE_MS,
        "boundary_ms": base_ms + 240 * MINUTE_MS,
        "parent_id": f"P:ETHUSDT:{anchor - 3 * MINUTE_MS}",
        "parent_ts_ms": anchor - 3 * MINUTE_MS,
        "echo_id": "ECHO:ETHUSDT:1",
        "q_parent": 1.5, "q_echo": 2.25, "prior_stress_count": 3,
        "multiscale_votes": {"i1_v30": -1, "i3_v30": -1, "i5_v30": -1, "i10_v30": -1},
        "multiscale_vote_sum": -4,
        "status": "AWAITING_ENTRY",
        "entry_open": None, "boundary_open": None,
        "gross_return_bps": None, "net_return_bps": None,
        "cost_bps": 10.0,
        "data_quality_status": "PENDING_EXACT_OPENS",
        "universe_version": "2026-08-21",
        "code_sha": "deadbeef", "contract_sha": "ABC123",
        "cascade_id": f"CASCADE:{anchor}",
        "product_cohort": "NATIVE_CRYPTO", "listing_age_days": 900.0,
        "session_state": "ALWAYS_OPEN",
        "paper_only": True, "real_order_sent": False,
        "created_at_utc": "2026-08-22T00:00:00Z",
    }
    event.update(overrides)
    return event


# ==========================================================================
# B1 — the no-backfill gate must be a per-process publication epoch
# ==========================================================================
def test_b1_refuses_when_no_publication_epoch_has_been_established():
    """Fail closed. Without a boundary there is no no-backfill guarantee."""
    with pytest.raises(NoPublicationEpoch):
        to_trade_candidate(detected_event())


def test_b1_publishes_an_anchor_at_or_after_the_process_epoch():
    publication_epoch.start(EPOCH)
    assert to_trade_candidate(detected_event(anchor=EPOCH)).anchor_id == str(EPOCH)
    assert to_trade_candidate(detected_event(anchor=EPOCH + 1)).anchor_id == str(EPOCH + 1)


def test_b1_refuses_an_anchor_before_the_process_epoch():
    publication_epoch.start(EPOCH)
    with pytest.raises(NotEligible, match="epoch"):
        to_trade_candidate(detected_event(anchor=EPOCH - 1))


def test_b1_a_restart_makes_previously_publishable_anchors_unpublishable():
    """The finding's real point: a static constant cannot survive a restart.

    First process publishes an anchor. The process restarts later — module state
    is fresh, a new epoch is captured — and that same anchor is now history and
    must not be republished through the live path.
    """
    publication_epoch.start(EPOCH)
    event = detected_event(anchor=EPOCH + MINUTE_MS)
    assert to_trade_candidate(event).anchor_id == str(EPOCH + MINUTE_MS)

    publication_epoch.reset_for_tests()             # process exits
    publication_epoch.start(EPOCH + 60 * MINUTE_MS)  # restarts an hour later
    with pytest.raises(NotEligible, match="epoch"):
        to_trade_candidate(event)


def test_b1_the_epoch_cannot_be_moved_once_established():
    publication_epoch.start(EPOCH)
    with pytest.raises(RuntimeError):
        publication_epoch.start(EPOCH - 60 * MINUTE_MS)   # backwards
    with pytest.raises(RuntimeError):
        publication_epoch.start(EPOCH + 60 * MINUTE_MS)   # forwards
    assert publication_epoch.current() == EPOCH


def test_b1_the_caller_may_supply_an_explicit_boundary():
    """03C may pass its own runtime boundary instead of the process epoch."""
    publication_epoch.start(EPOCH)
    with pytest.raises(NotEligible, match="epoch"):
        to_trade_candidate(detected_event(anchor=EPOCH + MINUTE_MS),
                           publication_epoch_ms=EPOCH + 10 * MINUTE_MS)


def test_b1_a_default_epoch_is_captured_from_the_clock_not_a_constant():
    import time
    before = int(time.time() * 1000)
    publication_epoch.start()
    after = int(time.time() * 1000)
    assert before <= publication_epoch.current() <= after


def test_b1_the_static_integration_boundary_constant_is_gone():
    """A dead constant that once looked like the gate invites re-use."""
    assert not hasattr(manifest, "INTEGRATION_BOUNDARY_MS")


# ==========================================================================
# B2 — the source must be the frozen V1 paper-shadow producer
# ==========================================================================
@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "SOME_OTHER_ARM_2026"),
        ("protocol", None),
        ("classification", "RETROSPECTIVE"),
        ("classification", "BACKTEST"),
        ("paper_only", False),
        ("real_order_sent", True),
    ],
)
def test_b2_refuses_a_foreign_producer(field, value):
    """The adapter labelled anything DETECTED/AWAITING_ENTRY as E-DER V1."""
    publication_epoch.start(EPOCH)
    with pytest.raises(ProducerMismatch, match=field):
        to_trade_candidate(detected_event(**{field: value}))


def test_b2_data_quality_status_is_lifecycle_not_provenance():
    """A correction to this test file, made before the fix and stated plainly.

    `data_quality_status` was first written into the B2 producer-identity list,
    which contradicted the UNAVAILABLE test below expecting NotEligible. It is a
    lifecycle marker — `mature()` moves it — not an immutable producer trait, so
    a non-T0 value is an eligibility failure. Both tests now agree.
    """
    publication_epoch.start(EPOCH)
    with pytest.raises(NotEligible, match="data_quality_status"):
        to_trade_candidate(detected_event(data_quality_status="ENTRY_EXACT_OPEN"))


def test_b2_refuses_a_dict_that_merely_has_the_right_event_and_status():
    publication_epoch.start(EPOCH)
    with pytest.raises(ProducerMismatch):
        to_trade_candidate({
            "event": "DETECTED", "status": "AWAITING_ENTRY",
            "event_id": "E:FAKE:1", "symbol": "FAKE", "anchor_ts": ANCHOR,
            "entry_ms": ANCHOR + MINUTE_MS, "boundary_ms": ANCHOR + 2 * MINUTE_MS,
        })


def test_b2_a_genuine_v1_event_still_publishes():
    publication_epoch.start(EPOCH)
    c = to_trade_candidate(detected_event())
    assert c.arm == manifest.ARM and c.arm_version == manifest.ARM_VERSION


# ==========================================================================
# B3 — a populated outcome must surface as OutcomeLeak, before eligibility
# ==========================================================================
def entry_shape(anchor: int = ANCHOR) -> dict:
    """Exactly what `mature()` produces at the entry boundary."""
    return detected_event(anchor=anchor, event="ENTRY", status="OPEN",
                          entry_open=2500.0,
                          data_quality_status="ENTRY_EXACT_OPEN",
                          updated_at_utc="2026-08-22T00:31:00Z")


def close_shape(anchor: int = ANCHOR) -> dict:
    """Exactly what `mature()` produces at the boundary — realised return."""
    return detected_event(anchor=anchor, event="CLOSE", status="CLOSED",
                          entry_open=2500.0, boundary_open=2530.0,
                          gross_return_bps=119.4, net_return_bps=109.4,
                          data_quality_status="COMPLETE_EXACT_OPENS",
                          updated_at_utc="2026-08-22T04:31:00Z")


def test_b3_a_real_close_shape_raises_outcome_leak_not_not_eligible():
    """The frozen requirement: a populated outcome is surfaced loudly."""
    publication_epoch.start(EPOCH)
    with pytest.raises(OutcomeLeak):
        to_trade_candidate(close_shape())


def test_b3_a_real_entry_shape_raises_outcome_leak():
    publication_epoch.start(EPOCH)
    with pytest.raises(OutcomeLeak):
        to_trade_candidate(entry_shape())


def test_b3_the_leak_names_the_field_that_leaked():
    publication_epoch.start(EPOCH)
    with pytest.raises(OutcomeLeak, match="gross_return_bps"):
        to_trade_candidate(close_shape())


def test_b3_seal_detection_precedes_every_other_refusal():
    """Even a foreign producer with no epoch must report the leak first."""
    with pytest.raises(OutcomeLeak):
        to_trade_candidate(close_shape(anchor=EPOCH - 10 * MINUTE_MS) |
                           {"protocol": "SOMETHING_ELSE", "paper_only": False})


def test_b3_an_unavailable_shape_with_no_outcome_is_merely_not_eligible():
    """Refusals must stay distinguishable: no outcome, so not a leak."""
    publication_epoch.start(EPOCH)
    with pytest.raises(NotEligible):
        to_trade_candidate(detected_event(event="ENTRY_UNAVAILABLE", status="UNAVAILABLE",
                                          data_quality_status="EXACT_ENTRY_OPEN_UNAVAILABLE"))


# ==========================================================================
# B4 — refuse rather than filter; require the real T0 shape
# ==========================================================================
def test_b4_a_mutation_marker_hard_fails_instead_of_being_filtered():
    """`updated_at_utc` is written only by mature(); its presence is evidence."""
    publication_epoch.start(EPOCH)
    with pytest.raises(OutcomeLeak, match="updated_at_utc"):
        to_trade_candidate(detected_event(updated_at_utc="2026-08-22T00:31:00Z"))


@pytest.mark.parametrize(
    "missing", ["entry_open", "boundary_open", "gross_return_bps", "net_return_bps"],
)
def test_b4_a_missing_t0_outcome_marker_is_refused(missing):
    """make_event emits all four as present-and-None. Absence is a foreign shape."""
    publication_epoch.start(EPOCH)
    event = detected_event()
    del event[missing]
    with pytest.raises(ProducerMismatch, match=missing):
        to_trade_candidate(event)


def test_b4_all_four_markers_present_and_none_is_the_accepted_shape():
    publication_epoch.start(EPOCH)
    c = to_trade_candidate(detected_event())
    assert c.candidate_id.startswith("E:ETHUSDT:")


def test_b4_the_forbidden_list_is_enforced_not_documented():
    """Every field the mapping calls a hard failure must actually hard-fail."""
    publication_epoch.start(EPOCH)
    for field in sorted(manifest.MUTATION_MARKERS):
        with pytest.raises(OutcomeLeak, match=field):
            to_trade_candidate(detected_event(**{field: "anything"}))
