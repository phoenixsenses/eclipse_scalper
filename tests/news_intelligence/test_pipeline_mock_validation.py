"""Mock validation: the five scenarios, end to end, with nothing heavy.

The reprint scenario carries most of the weight. Three outlets republish one
tariff statement, and the suite asserts all three consequences together —
cluster count unchanged, novelty of the copies below the original, amplification
rising. Checking only one of the three would pass on a system that is quietly
wrong: a clusterer that groups everything satisfies the first, and a novelty
engine that always returns zero satisfies the second.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eclipse.news_intelligence.adapters.mock import MockAdapter, fixture_events
from eclipse.news_intelligence.clustering.clusterer import (
    ClusterInput,
    assert_outcome_blind,
)
from eclipse.news_intelligence.pipeline import NewsIntelligencePipeline
from eclipse.news_intelligence.publishing.bus import SUBJECTS
from eclipse.news_intelligence.research.api import ResearchStore
from eclipse.news_intelligence.taxonomy.events import EventType, is_scheduled


@pytest.fixture()
def processed():
    pipeline = NewsIntelligencePipeline()
    return pipeline, [pipeline.process(raw) for raw in MockAdapter().poll()]


def _by_entity(items, entity):
    return [p for p in items if p.event.entity == entity]


# --- the reprint scenario --------------------------------------------------

def test_four_articles_about_one_statement_are_one_independent_event(processed):
    pipeline, items = processed
    tariff = _by_entity(items, "donald trump")
    assert len(tariff) == 4, "fixture should carry the original plus three reprints"
    assert len({p.event.news_cluster_id for p in tariff}) == 1
    assert sum(1 for p in tariff if p.is_independent_observation) == 1


def test_the_reprints_are_less_novel_than_the_original(processed):
    _, items = processed
    tariff = sorted(_by_entity(items, "donald trump"), key=lambda p: p.event.first_seen_at)
    original, reprints = tariff[0], tariff[1:]
    assert original.novelty.novelty_score == 1.0
    for reprint in reprints:
        assert reprint.novelty.novelty_score < original.novelty.novelty_score
        assert reprint.novelty.nearest_previous_event_id is not None
        assert reprint.novelty.time_since_similar_seconds > 0


def test_amplification_rises_as_the_story_is_repeated(processed):
    _, items = processed
    tariff = sorted(_by_entity(items, "donald trump"), key=lambda p: p.event.first_seen_at)
    scores = [p.amplification.amplification_score for p in tariff]
    assert scores == sorted(scores), f"amplification should be non-decreasing, got {scores}"
    assert scores[0] == 0.0
    assert scores[-1] > 0.5
    assert tariff[-1].amplification.source_count == 4


def test_novelty_and_amplification_move_in_opposite_directions(processed):
    """The distinction the layer exists to make: the fiftieth article is not
    fifty times the news, but it is much more attention."""
    _, items = processed
    tariff = sorted(_by_entity(items, "donald trump"), key=lambda p: p.event.first_seen_at)
    assert tariff[0].novelty.novelty_score > tariff[-1].novelty.novelty_score
    assert tariff[0].amplification.amplification_score < tariff[-1].amplification.amplification_score


# --- classification --------------------------------------------------------

def test_each_scenario_lands_on_its_taxonomy_entry(processed):
    _, items = processed
    got = {(p.event.entity, p.event.event_type) for p in items}
    assert ("donald trump", EventType.TARIFF) in got
    assert ("federal reserve", EventType.RATE_POLICY) in got
    assert ("elon musk", EventType.SOCIAL_POST) in got
    assert ("nvidia", EventType.COMPANY_EARNINGS) in got


def test_the_speaker_outranks_anyone_merely_mentioned(processed):
    """The tariff wire copy names the White House in its body; the author is the
    speaker, and the speaker is the entity."""
    _, items = processed
    for item in _by_entity(items, "donald trump"):
        assert item.event.entity == "donald trump"


def test_scheduled_and_unscheduled_types_are_distinguished():
    assert is_scheduled(EventType.RATE_POLICY) is False
    assert is_scheduled(EventType.CENTRAL_BANK) is True
    assert is_scheduled(EventType.COMPANY_EARNINGS) is True
    assert is_scheduled(EventType.SOCIAL_POST) is False


# --- relevance -------------------------------------------------------------

def test_relevance_reaches_the_right_assets_without_claiming_a_direction(processed):
    _, items = processed
    musk = _by_entity(items, "elon musk")[0]
    assert "DOGE" in musk.event.asset_relevance.relevant(0.5)
    assert "TSLA" in musk.event.asset_relevance.relevant(0.5)
    assert all(w >= 0 for w in musk.event.asset_relevance.weights.values())

    fed = _by_entity(items, "federal reserve")[0]
    top = fed.event.asset_relevance.relevant(0.5)
    assert "US2Y" in top and "DXY" in top
    assert fed.event.asset_relevance.weight("BTC") < fed.event.asset_relevance.weight("US2Y")


def test_every_relevant_asset_can_explain_itself(processed):
    _, items = processed
    for item in items:
        for asset in item.event.asset_relevance.relevant(0.2):
            assert item.event.asset_relevance.explain(asset) != "no recorded reason"


# --- structure -------------------------------------------------------------

def test_cluster_input_is_structurally_outcome_blind():
    assert_outcome_blind(ClusterInput)
    fields = {f for f in ClusterInput.__dataclass_fields__}
    assert "reaction" not in fields and "label" not in fields


def test_snapshot_is_built_and_carries_no_future(processed):
    _, items = processed
    for item in items:
        assert item.snapshot.decision_time == item.event.first_seen_at
        row = item.snapshot.as_row()
        assert "label_return_bps" not in row


def test_reaction_request_is_anchored_to_first_seen_not_published(processed):
    _, items = processed
    item = _by_entity(items, "donald trump")[0]
    assert item.reaction_request.decision_time == item.event.first_seen_at
    assert item.reaction_request.decision_time > item.event.published_at
    assert item.reaction_request.earliest_needed < item.reaction_request.decision_time
    assert item.reaction_request.latest_needed > item.reaction_request.decision_time


def test_processing_is_deterministic():
    """Same raw items, same ids — otherwise nothing can be recomputed or joined."""
    first = [p.event.event_id for p in
             (NewsIntelligencePipeline().process(r) for r in fixture_events())]
    second = [p.event.event_id for p in
              (NewsIntelligencePipeline().process(r) for r in fixture_events())]
    assert first == second
    assert len(set(first)) == len(first)


# --- publishing ------------------------------------------------------------

def test_the_pipeline_publishes_candidates_and_never_an_outcome(processed):
    pipeline, items = processed
    published = pipeline.publisher.published
    assert {e.subject for e in published} <= set(SUBJECTS.values())
    assert SUBJECTS["news_raw"] in {e.subject for e in published}
    assert SUBJECTS["research_ready"] in {e.subject for e in published}
    for envelope in published:
        assert "return_bps" not in envelope.payload
        assert "pnl" not in envelope.payload


# --- counting --------------------------------------------------------------

def test_the_store_reports_raw_items_and_clusters_as_different_numbers(processed):
    _, items = processed
    store = ResearchStore()
    for item in items:
        store.add_snapshot(item.snapshot, high_impact=item.is_high_impact)
    counters = store.counters()
    assert counters.raw_items == 7
    assert counters.independent_clusters == 4
    assert counters.duplication_ratio == 1.75
    assert len(store.one_per_cluster()) == 4


def test_one_per_cluster_selects_the_first_not_the_best(processed):
    """Selecting by any score would condition the sample on a quantity the study
    is about to measure."""
    _, items = processed
    store = ResearchStore()
    for item in items:
        store.add_snapshot(item.snapshot)
    chosen = store.one_per_cluster()
    tariff = [p for p in items if p.event.entity == "donald trump"]
    earliest = min(tariff, key=lambda p: p.event.first_seen_at)
    assert earliest.event.event_id in {s.event_id for s in chosen}
