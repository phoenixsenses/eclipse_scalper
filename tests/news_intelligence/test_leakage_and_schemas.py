"""The invariants that make this layer research-grade rather than decorative.

Every test here corresponds to a way a news pipeline is normally wrong. They are
written as regressions against those failures, not as coverage of the happy
path, so each one names the mistake it is guarding.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eclipse.news_intelligence.errors import (
    DeterministicFieldOverwrite,
    LookaheadError,
    OutcomeInFeatureSpace,
    RelevanceIsNotDirection,
    UnknownSource,
)
from eclipse.news_intelligence.normalization.annotation import (
    ModelAnnotation,
    apply_annotation,
)
from eclipse.news_intelligence.publishing.bus import (
    SUBJECTS,
    Envelope,
    InMemoryPublisher,
)
from eclipse.news_intelligence.relevance.graph import Edge, default_graph
from eclipse.news_intelligence.schemas.normalized import NormalizedEvent, Sentiment
from eclipse.news_intelligence.schemas.raw import RawEvent
from eclipse.news_intelligence.schemas.reaction import OUTCOME_FIELDS
from eclipse.news_intelligence.schemas.relevance import AssetRelevance
from eclipse.news_intelligence.schemas.snapshot import (
    FeatureSnapshot,
    Observation,
    ResearchLabel,
)
from eclipse.news_intelligence.taxonomy.events import EventType

T0 = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _event(**kwargs) -> NormalizedEvent:
    base = dict(
        event_id="evt_1",
        raw_event_id="raw_1",
        published_at=T0,
        first_seen_at=T0 + timedelta(seconds=30),
        received_at=T0 + timedelta(seconds=31),
        entity="donald trump",
        event_type=EventType.TARIFF,
    )
    base.update(kwargs)
    return NormalizedEvent(**base)


# --- lookahead ------------------------------------------------------------

def test_observation_from_after_the_decision_time_is_refused():
    """The failure this whole package is built around: a feature that could not
    have been known. Raised at construction, because by read time the caller no
    longer knows what went in."""
    with pytest.raises(LookaheadError) as excinfo:
        FeatureSnapshot(
            event_id="evt_1",
            decision_time=T0,
            observations=(Observation("btc_price", 100.0, T0 + timedelta(minutes=1)),),
        )
    assert "after the decision time" in str(excinfo.value)


def test_observation_from_before_the_decision_time_is_accepted():
    snapshot = FeatureSnapshot(
        event_id="evt_1",
        decision_time=T0,
        observations=(Observation("btc_price", 100.0, T0 - timedelta(minutes=5)),),
    )
    assert snapshot.observation("btc_price").value == 100.0


def test_an_outcome_cannot_enter_feature_space_even_when_timestamped_early():
    """A realised return carries no clock of its own, so the timestamp check
    cannot see it. Only a structural rule catches this one."""
    with pytest.raises(OutcomeInFeatureSpace):
        Observation("return_bps", 42.0, T0 - timedelta(minutes=5))

    with pytest.raises(OutcomeInFeatureSpace):
        FeatureSnapshot(event_id="e", decision_time=T0, context={"pnl": 3.0})


def test_snapshot_row_contains_no_outcome_columns():
    snapshot = FeatureSnapshot(
        event_id="evt_1",
        decision_time=T0,
        asset_relevance=AssetRelevance(weights={"BTC": 0.5}, reasons={"BTC": "test"}),
        observations=(Observation("vol_regime", "high", T0 - timedelta(minutes=1)),),
    )
    row = snapshot.as_row()
    assert not OUTCOME_FIELDS & set(row)
    assert row["relevance_BTC"] == 0.5
    assert row["obs_vol_regime"] == "high"


def test_a_label_cannot_carry_its_own_features():
    label = ResearchLabel(event_id="evt_1")
    assert not hasattr(label, "features")
    assert not hasattr(label, "snapshot")


# --- relevance is not direction -------------------------------------------

def test_negative_relevance_is_refused_as_a_directional_claim():
    with pytest.raises(RelevanceIsNotDirection):
        AssetRelevance(weights={"BTC": -0.4})
    with pytest.raises(RelevanceIsNotDirection):
        Edge("BTC", -0.2, "bearish")


def test_graph_edges_all_carry_a_reason():
    graph = default_graph()
    for entity in graph.entities():
        for edge in graph.edges_for(entity):
            assert edge.reason.strip(), f"{entity} -> {edge.asset} has no reason"
            assert 0.0 <= edge.weight <= 1.0


def test_a_relevance_threshold_includes_its_own_boundary():
    """Asking for SECONDARY must return the SECONDARY edges. A strictly-greater
    comparison drops the whole level being asked for, which emptied the
    relevance column for every tariff row the first time the demo was run."""
    relevance = AssetRelevance(
        weights={"DXY": 0.5, "BTC": 0.25, "NDX": 0.8},
        reasons={"DXY": "a", "BTC": "b", "NDX": "c"},
    )
    assert relevance.relevant(0.5) == ("NDX", "DXY")
    assert relevance.relevant(0.25) == ("NDX", "DXY", "BTC")
    assert relevance.relevant(0.9) == ()


def test_relevance_merges_by_maximum_not_by_sum():
    """Five weak mentions must not outrank one direct statement."""
    merged = AssetRelevance.merged(
        [
            AssetRelevance(weights={"BTC": 0.25}, reasons={"BTC": "a"}),
            AssetRelevance(weights={"BTC": 0.25}, reasons={"BTC": "b"}),
            AssetRelevance(weights={"BTC": 0.25}, reasons={"BTC": "c"}),
        ]
    )
    assert merged.weight("BTC") == 0.25


# --- deterministic fields --------------------------------------------------

def test_model_cannot_overwrite_a_timestamp():
    with pytest.raises(DeterministicFieldOverwrite):
        ModelAnnotation(
            model_id="m", prompt_version="p", produced_at=T0,
            values={"first_seen_at": T0}, confidences={"first_seen_at": 1.0},
        )


def test_model_cannot_emit_a_trading_decision():
    with pytest.raises(DeterministicFieldOverwrite):
        ModelAnnotation(
            model_id="m", prompt_version="p", produced_at=T0,
            values={"buy": True}, confidences={"buy": 0.9},
        )


def test_annotation_without_confidence_is_refused():
    with pytest.raises(ValueError):
        ModelAnnotation(
            model_id="m", prompt_version="p", produced_at=T0,
            values={"topic": "tariffs"}, confidences={},
        )


def test_annotation_records_its_provenance_and_leaves_the_original_alone():
    event = _event()
    annotated = apply_annotation(
        event,
        ModelAnnotation(
            model_id="gpt-test", prompt_version="v3", produced_at=T0,
            values={"topic": "semiconductor tariffs"},
            confidences={"topic": 0.72},
        ),
    )
    assert annotated.topic == "semiconductor tariffs"
    assert annotated.confidence_in("topic") == 0.72
    assert annotated.judgements["topic"].model_id == "gpt-test"
    assert annotated.judgements["topic"].prompt_version == "v3"
    assert event.topic == "", "annotation must not mutate the event it was given"


# --- timestamps and identity ----------------------------------------------

def test_naive_datetimes_are_refused():
    with pytest.raises(ValueError):
        _event(first_seen_at=datetime(2026, 8, 23, 12, 0))


def test_seeing_something_before_it_was_published_is_a_clock_error():
    with pytest.raises(ValueError) as excinfo:
        RawEvent(
            raw_event_id="r", source_id="mock", source_type="MOCK", source_authority="MOCK",
            source_ref="ref", published_at=T0, first_seen_at=T0 - timedelta(seconds=5),
            received_at=T0, raw_title="t", raw_text="x",
        )
    assert "clock disagreement" in str(excinfo.value)


def test_decision_time_is_when_we_could_know_not_when_it_was_published():
    event = _event()
    assert event.decision_time == event.first_seen_at
    assert event.decision_time > event.published_at


# --- the bus ---------------------------------------------------------------

def test_the_bus_refuses_an_outcome_even_when_nested():
    publisher = InMemoryPublisher()
    with pytest.raises(OutcomeInFeatureSpace):
        publisher.publish(
            Envelope(
                subject=SUBJECTS["news_normalized"],
                payload={"event_id": "e", "label": {"return_bps": 12.0}},
                published_at=T0,
                producer="test",
            )
        )
    assert publisher.published == []


def test_the_bus_accepts_a_candidate_shaped_payload():
    publisher = InMemoryPublisher()
    publisher.publish(
        Envelope(
            subject=SUBJECTS["news_high_impact"],
            payload={"event_id": "e", "entity": "federal reserve", "novelty": 0.9},
            published_at=T0,
            producer="test",
        )
    )
    assert publisher.subjects() == (SUBJECTS["news_high_impact"],)


# --- source registry -------------------------------------------------------

def test_an_unregistered_source_has_no_authority_rather_than_a_default():
    from eclipse.news_intelligence.adapters.base import default_registry

    with pytest.raises(UnknownSource):
        default_registry().authority_of("some_blog")


def test_sentiment_bounds_are_enforced():
    with pytest.raises(ValueError):
        Sentiment(polarity=1.4, strength=0.5)
    with pytest.raises(ValueError):
        Sentiment(polarity=0.0, strength=2.0)
