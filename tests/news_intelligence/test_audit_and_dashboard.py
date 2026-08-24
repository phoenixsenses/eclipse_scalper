"""Explainability, and the boundary around the private panel.

An event that cannot explain itself is not evidence, so `explain` has to produce
the full chain from the stored row back to the raw item — and it has to warn
about the things a reader would otherwise have to notice for themselves: low
confidence, a missing entity, an unclustered event that may not be an
independent observation.

The dashboard tests guard the other direction: an operator panel is one
screenshot from being public, so the payload builder refuses outcome-shaped keys
outright rather than trusting the template not to render them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eclipse.news_intelligence.adapters.mock import MockAdapter, fixture_events
from eclipse.news_intelligence.pipeline import NewsIntelligencePipeline
from eclipse.news_intelligence.relevance.graph import default_graph
from eclipse.news_intelligence.research.api import ResearchStore, research_frame
from eclipse.news_intelligence.research.dashboard_contract import (
    CROSS_ASSET_ROW,
    cross_asset_panel,
    dashboard_payload,
    recent_events_panel,
)
from eclipse.news_intelligence.schemas.reaction import HorizonMeasurement, MarketReaction
from eclipse.news_intelligence.schemas.snapshot import ResearchLabel
from eclipse.news_intelligence.validation.audit import explain

NOW = datetime(2026, 8, 23, 18, 0, tzinfo=timezone.utc)


@pytest.fixture()
def run():
    pipeline = NewsIntelligencePipeline()
    raws = list(MockAdapter().poll())
    processed = [pipeline.process(raw) for raw in raws]
    return pipeline, raws, processed


# --- audit -----------------------------------------------------------------

def test_an_event_can_explain_every_part_of_its_classification(run):
    _, raws, processed = run
    by_raw = {r.raw_event_id: r for r in raws}
    for item in processed:
        explanation = explain(item.event, by_raw[item.event.raw_event_id], default_graph())
        payload = explanation.as_dict()
        assert payload["source"]["source_id"]
        assert payload["timestamps"]["decision_time"]
        assert payload["classification"]["event_type"]
        assert payload["versions"]["taxonomy_version"] >= 1
        for name, judgement in payload["judgements"].items():
            assert judgement["model_id"], f"{name} judgement has no author"
            assert 0.0 <= judgement["confidence"] <= 1.0


def test_the_explanation_names_the_graph_reason_for_each_asset(run):
    _, raws, processed = run
    by_raw = {r.raw_event_id: r for r in raws}
    fed = next(p for p in processed if p.event.entity == "federal reserve")
    reasons = explain(fed.event, by_raw[fed.event.raw_event_id]).as_dict()["asset_relevance_reasons"]
    assert "US2Y" in reasons
    assert "policy rate" in reasons["US2Y"].lower()


def test_the_explanation_records_the_publication_lag(run):
    _, raws, processed = run
    by_raw = {r.raw_event_id: r for r in raws}
    item = processed[0]
    lag = explain(item.event, by_raw[item.event.raw_event_id]).publication_lag_seconds
    assert lag > 0, "the fixtures deliberately arrive after they are published"


def test_an_explanation_assembled_from_the_wrong_raw_item_is_refused(run):
    _, raws, processed = run
    with pytest.raises(ValueError):
        explain(processed[0].event, raws[1])


def test_low_confidence_and_missing_structure_surface_as_warnings():
    from dataclasses import replace

    pipeline = NewsIntelligencePipeline()
    item = pipeline.process(fixture_events()[2])  # the social post
    stripped = replace(item.event, news_cluster_id=None, entity="")
    warnings = explain(stripped, fixture_events()[2]).warnings
    assert any("no entity" in w for w in warnings)
    assert any("not clustered" in w for w in warnings)


# --- research frame --------------------------------------------------------

def test_the_frame_marks_which_columns_are_labels(run):
    _, _, processed = run
    store = ResearchStore()
    for item in processed:
        store.add_snapshot(item.snapshot)

    event_id = processed[0].event.event_id
    store.add_label(
        ResearchLabel(
            event_id=event_id,
            reaction=MarketReaction(
                event_id=event_id,
                measured_at=NOW,
                measurements=(HorizonMeasurement("BTC", 15, return_bps=12.5, complete=True),),
            ),
            resolved=True,
        )
    )
    rows = research_frame(store.one_per_cluster(), {event_id: store.label_for(event_id)}, "BTC", 15)
    labelled = [r for r in rows if r["event_id"] == event_id][0]
    assert labelled["label_return_bps"] == 12.5
    assert labelled["label_complete"] is True
    assert all(k.startswith("label_") for k in labelled if "return_bps" in k)


def test_unresolved_events_stay_in_the_frame_as_unresolved(run):
    """Dropping them would condition the sample on resolution, and events that
    resolve are not a random subset of events."""
    _, _, processed = run
    store = ResearchStore()
    for item in processed:
        store.add_snapshot(item.snapshot)
    rows = research_frame(store.one_per_cluster(), {}, "BTC", 15)
    assert len(rows) == 4
    assert all(row["label_complete"] is False for row in rows)
    assert all(row["label_return_bps"] is None for row in rows)


def test_a_label_without_its_feature_side_is_refused():
    store = ResearchStore()
    with pytest.raises(KeyError):
        store.add_label(ResearchLabel(event_id="never_seen"))


def test_complete_labels_are_counted_separately_from_events(run):
    _, _, processed = run
    store = ResearchStore()
    for item in processed:
        store.add_snapshot(item.snapshot, high_impact=item.is_high_impact)
    counters = store.counters()
    assert counters.events_with_complete_labels == 0
    assert counters.raw_items > counters.independent_clusters


# --- the private panel -----------------------------------------------------

def test_the_panel_never_carries_an_outcome(run):
    _, _, processed = run
    store = ResearchStore()
    for item in processed:
        store.add_snapshot(item.snapshot, high_impact=item.is_high_impact)
    payload = dashboard_payload(processed, store.counters(), {"BTC": "steady"}, NOW, ["BTC"])
    blob = repr(payload)
    for banned in ("return_bps", "pnl", "win_rate", "net_bps"):
        assert banned not in blob
    assert payload["visibility"] == "PRIVATE_MESH_ONLY"


def test_repeats_are_labelled_as_repeats_on_the_panel(run):
    _, _, processed = run
    rows = recent_events_panel(processed)["rows"]
    statuses = {row["status"] for row in rows}
    assert statuses == {"INDEPENDENT", "REPEAT"}
    assert sum(1 for row in rows if row["status"] == "INDEPENDENT") == 4


def test_an_asset_with_no_reading_renders_as_missing_not_as_zero():
    """The dashboard failure this repository already had once: a fabricated
    reading shown where the feed was dead."""
    panel = cross_asset_panel({"BTC": "steady", "VIX": "elevated"}, NOW, complete=["BTC"])
    rows = {row["asset"]: row for row in panel["rows"]}
    assert rows["BTC"]["state"] == "steady" and rows["BTC"]["known"] is True
    assert rows["VIX"]["state"] == "—" and rows["VIX"]["known"] is False
    assert set(panel["incomplete"]) == set(CROSS_ASSET_ROW) - {"BTC"}


def test_the_counters_publish_the_duplication_ratio(run):
    _, _, processed = run
    store = ResearchStore()
    for item in processed:
        store.add_snapshot(item.snapshot)
    payload = dashboard_payload(processed, store.counters(), {}, NOW)["research_counters"]
    assert payload["raw_news_items"] == 7
    assert payload["independent_news_clusters"] == 4
    assert payload["duplication_ratio"] == 1.75
    assert "different sample sizes" in payload["note"]
