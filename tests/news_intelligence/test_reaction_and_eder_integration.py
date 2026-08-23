"""Reverse causality, deferred capabilities, and the frozen arms.

The causality tests are the ones worth reading. The naive analysis — "price rose
after the headline, therefore the headline predicted it" — is wrong in a
specific and common way, and `classify_causality` is built to refuse it. A move
already underway before the item arrived is reported as price leading news no
matter how large the post-event return, because a sentiment feed reacting to a
move all afternoon will otherwise look like a forecast.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eclipse.news_intelligence import deferred
from eclipse.news_intelligence.errors import DeferredUntilPhase1Complete
from eclipse.news_intelligence.integration.eder_context import (
    FROZEN_ARMS,
    ArmModificationRefused,
    context_for_candidate,
    proposed_arm_name,
)
from eclipse.news_intelligence.reaction.contracts import (
    Causality,
    classify_causality,
)
from eclipse.news_intelligence.schemas.normalized import NormalizedEvent
from eclipse.news_intelligence.schemas.reaction import (
    HorizonMeasurement,
    MarketReaction,
    POST_EVENT_HORIZONS,
    PRE_EVENT_HORIZONS,
)
from eclipse.news_intelligence.schemas.relevance import AssetRelevance
from eclipse.news_intelligence.taxonomy.events import EventType

T0 = datetime(2026, 8, 23, 13, 44, tzinfo=timezone.utc)


def _reaction(pre_bps, post_bps, complete=True):
    return MarketReaction(
        event_id="evt_1",
        measured_at=T0 + timedelta(hours=5),
        measurements=(
            HorizonMeasurement("BTC", -15, return_bps=pre_bps, complete=complete,
                               missing_reason=None if complete else "feed gap"),
            HorizonMeasurement("BTC", 15, return_bps=post_bps, complete=complete,
                               missing_reason=None if complete else "feed gap"),
        ),
    )


# --- reverse causality -----------------------------------------------------

def test_a_move_that_starts_after_the_item_is_news_leading():
    verdict = classify_causality(_reaction(pre_bps=2.0, post_bps=45.0), "BTC")
    assert verdict.direction is Causality.NEWS_LEADS_PRICE


def test_a_move_already_underway_is_never_reported_as_news_leading():
    """The trap: a large post-event return on top of a move that had already
    begun. Sentiment feeds produce this all day."""
    verdict = classify_causality(_reaction(pre_bps=60.0, post_bps=55.0), "BTC")
    assert verdict.direction is Causality.PRICE_LEADS_NEWS
    assert "already underway" in verdict.reason


def test_a_move_that_stops_at_the_item_is_price_leading():
    verdict = classify_causality(_reaction(pre_bps=-40.0, post_bps=1.0), "BTC")
    assert verdict.direction is Causality.PRICE_LEADS_NEWS


def test_two_quiet_windows_are_no_relationship_not_a_weak_signal():
    verdict = classify_causality(_reaction(pre_bps=1.0, post_bps=-2.0), "BTC")
    assert verdict.direction is Causality.NO_RELATIONSHIP


def test_opposite_moves_are_reported_as_not_separable():
    verdict = classify_causality(_reaction(pre_bps=-50.0, post_bps=48.0), "BTC")
    assert verdict.direction is Causality.SIMULTANEOUS


def test_a_missing_window_is_undetermined_not_zero():
    """The failure this repository has already paid for: an absent measurement
    read as a calm market."""
    verdict = classify_causality(_reaction(pre_bps=None, post_bps=40.0, complete=False), "BTC")
    assert verdict.direction is Causality.UNDETERMINED
    assert "not a zero" in verdict.reason


def test_an_incomplete_measurement_must_say_why():
    with pytest.raises(ValueError):
        HorizonMeasurement("BTC", 15, return_bps=None, complete=False)


def test_horizons_cover_both_sides_of_the_event():
    assert max(PRE_EVENT_HORIZONS) < 0 < min(POST_EVENT_HORIZONS)
    assert -1 in PRE_EVENT_HORIZONS and 240 in POST_EVENT_HORIZONS


def test_a_reaction_is_complete_only_when_every_window_landed():
    assert _reaction(1.0, 1.0).is_complete
    assert not _reaction(None, 1.0, complete=False).is_complete


# --- deferred capabilities -------------------------------------------------

def test_every_heavy_capability_refuses_to_start():
    assert deferred.REGISTER, "the register must not be empty"
    for key in deferred.REGISTER:
        with pytest.raises(DeferredUntilPhase1Complete) as excinfo:
            deferred.start(key)
        assert deferred.MARKER in str(excinfo.value)


def test_an_unregistered_heavy_capability_also_refuses():
    with pytest.raises(DeferredUntilPhase1Complete):
        deferred.start("some_new_collector")


def test_a_real_source_adapter_refuses_rather_than_returning_nothing():
    """A collector that quietly yields nothing looks exactly like a quiet news
    day, which is the confusion this whole system is built to avoid."""
    from eclipse.news_intelligence.adapters.base import deferred_adapters

    adapters = deferred_adapters()
    assert "reuters" in adapters and "federal_reserve" in adapters
    with pytest.raises(DeferredUntilPhase1Complete):
        list(adapters["federal_reserve"].poll())


def test_the_register_reports_what_each_capability_would_cost():
    for row in deferred.register_report():
        assert row["status"] == deferred.MARKER
        assert row["resource"] and row["unblocks_when"]


# --- E-DER integration -----------------------------------------------------

def _event(entity, first_seen, weights):
    return NormalizedEvent(
        event_id=f"evt_{entity}_{first_seen.minute}",
        raw_event_id="raw",
        published_at=first_seen - timedelta(seconds=5),
        first_seen_at=first_seen,
        received_at=first_seen,
        entity=entity,
        event_type=EventType.TARIFF,
        asset_relevance=AssetRelevance(weights=weights, reasons={k: "test" for k in weights}),
    )


def test_context_uses_only_events_the_candidate_could_have_known():
    """A later event must not describe an earlier candidate, however relevant."""
    candidate_time = T0 + timedelta(minutes=10)
    earlier = _event("donald trump", T0 + timedelta(minutes=3), {"BTC": 0.6})
    later = _event("federal reserve", T0 + timedelta(minutes=30), {"BTC": 0.95})

    context = context_for_candidate(
        "cand_1", "E-DER-V1", candidate_time, (earlier, later), asset="BTC"
    )
    assert context.event_id == earlier.event_id
    assert context.high_impact_news is True
    assert context.event_age_minutes == pytest.approx(7.0, abs=0.01)


def test_no_recent_news_is_its_own_state_not_a_missing_value():
    context = context_for_candidate("cand_2", "E-DER-V3", T0, (), asset="BTC")
    assert context.global_context == "NO_RECENT_NEWS"
    assert context.high_impact_news is False
    assert context.event_id is None


def test_context_ranks_by_relevance_not_by_recency_alone():
    candidate_time = T0 + timedelta(minutes=20)
    weak_recent = _event("donald trump", T0 + timedelta(minutes=19), {"BTC": 0.2})
    strong_older = _event("federal reserve", T0 + timedelta(minutes=5), {"BTC": 0.9})
    context = context_for_candidate(
        "cand_3", "E-DER-A2", candidate_time, (weak_recent, strong_older), asset="BTC"
    )
    assert context.event_id == strong_older.event_id


def test_the_context_object_cannot_reach_into_an_arm():
    context = context_for_candidate("cand_4", "E-DER-V1", T0, (), asset="BTC")
    annotation = context.as_annotation()
    assert set(annotation) == {"candidate_id", "arm", "observed_at", "news_context"}
    assert not hasattr(context, "apply")
    assert not hasattr(context, "filter")


def test_combining_with_a_frozen_arm_produces_a_new_arm_name():
    name = proposed_arm_name("E-DER-V3", "risk_off")
    assert name == "E-DER-V3+NEWS_RISK_OFF"
    assert name not in FROZEN_ARMS


def test_a_combined_arm_may_not_take_a_frozen_arm_name():
    with pytest.raises(ArmModificationRefused):
        proposed_arm_name("SOMETHING_ELSE", "risk_off")
