"""What happens when the stream is not clean.

A collector retries. A process restarts and re-polls. A slow source delivers
something that happened earlier than what already arrived. None of these are
exotic — they are the normal life of a feed — and each of them corrupts a
different part of the record if the pipeline just processes whatever it is
handed.

The three failures below were found by feeding the pipeline a stream that was
not clean, not by reading the code.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eclipse.news_intelligence.adapters.mock import fixture_events
from eclipse.news_intelligence.errors import DuplicateDelivery, OutOfOrderDelivery
from eclipse.news_intelligence.pipeline import NewsIntelligencePipeline


# --- re-delivery -----------------------------------------------------------

def test_an_identical_re_delivery_is_refused_rather_than_counted_again():
    """A retry or a re-poll after a restart must not become a second observation.

    Measured before the fix: the same item processed twice pushed amplification
    from 0.00 to 0.17 and update_count from 1 to 2. Attention is a feature, and
    inventing it from a retry is a data-integrity defect, not a rounding error.
    """
    pipeline = NewsIntelligencePipeline()
    raw = fixture_events()[0]
    first = pipeline.process(raw)

    with pytest.raises(DuplicateDelivery) as excinfo:
        pipeline.process(raw)
    assert first.event.event_id in str(excinfo.value)


def test_a_collector_can_ask_whether_an_item_is_new_without_catching():
    """Re-delivery is normal operation, so the collector-facing call returns
    None rather than making every caller wrap process() in a try block."""
    pipeline = NewsIntelligencePipeline()
    raw = fixture_events()[0]
    assert pipeline.process_if_new(raw) is not None
    assert pipeline.process_if_new(raw) is None


def test_a_refused_duplicate_leaves_the_engines_untouched():
    """The refusal must happen before clustering and amplification see it —
    otherwise the counters are already wrong by the time it is raised."""
    pipeline = NewsIntelligencePipeline()
    raw = fixture_events()[0]
    first = pipeline.process(raw)
    before = (
        first.amplification.update_count,
        pipeline.clusterer.cluster_count(),
        pipeline.novelty.remembered(),
    )
    pipeline.process_if_new(raw)
    assert (
        first.amplification.update_count,
        pipeline.clusterer.cluster_count(),
        pipeline.novelty.remembered(),
    ) == before


def test_a_revision_of_an_item_is_not_a_duplicate():
    """A corrected article is genuinely new information about the same story."""
    from dataclasses import replace

    pipeline = NewsIntelligencePipeline()
    raw = fixture_events()[0]
    original = pipeline.process(raw)
    revised = replace(raw, raw_event_id="raw_001r", revision=1)
    processed = pipeline.process(revised)

    assert processed.event.event_id != original.event.event_id, (
        "a revision is a different row; the id carries the revision number"
    )
    assert not processed.is_independent_observation, "a revision belongs to the same cluster"
    assert processed.cluster.update_count == 2


# --- ordering --------------------------------------------------------------

def test_an_item_from_before_the_last_one_is_refused():
    """Out of order, the reprint became the cluster's first source and the
    original was recorded as a repeat of it. "Who was first" is one of the
    things this layer exists to measure, so a stream that would corrupt it is
    refused rather than silently accepted."""
    pipeline = NewsIntelligencePipeline()
    events = fixture_events()
    pipeline.process(events[4])  # the wire reprint, 13:48

    with pytest.raises(OutOfOrderDelivery) as excinfo:
        pipeline.process(events[0])  # the original, 13:44
    assert "sort" in str(excinfo.value).lower()


def test_a_batch_is_sorted_before_it_is_processed():
    """The correct path for a backfill or a multi-source poll: order the batch,
    then process. The cluster's first source is then the source that was
    actually first."""
    pipeline = NewsIntelligencePipeline()
    events = fixture_events()
    shuffled = [events[6], events[4], events[0], events[5]]  # all one story, scrambled

    processed = pipeline.process_batch(shuffled)

    assert [p.event.first_seen_at for p in processed] == sorted(
        p.event.first_seen_at for p in processed
    )
    first = processed[0]
    assert first.is_independent_observation
    assert first.cluster.first_source_id == "white_house"
    assert sum(1 for p in processed if p.is_independent_observation) == 1


def test_the_original_keeps_maximum_novelty_when_the_batch_is_ordered():
    pipeline = NewsIntelligencePipeline()
    events = fixture_events()
    processed = pipeline.process_batch([events[6], events[4], events[0], events[5]])
    assert processed[0].novelty.novelty_score == 1.0
    assert all(p.novelty.novelty_score < 1.0 for p in processed[1:])


def test_two_items_sharing_a_timestamp_are_not_out_of_order():
    """Equal is not earlier. Two sources landing in the same instant is normal."""
    from dataclasses import replace

    pipeline = NewsIntelligencePipeline()
    first = fixture_events()[0]
    same_instant = replace(
        fixture_events()[3],
        raw_event_id="raw_same",
        published_at=first.published_at,
        first_seen_at=first.first_seen_at,
        received_at=first.received_at,
    )
    pipeline.process(first)
    pipeline.process(same_instant)  # must not raise


# --- memory ----------------------------------------------------------------

def test_the_novelty_engine_forgets_what_is_older_than_its_memory():
    """Before the fix `_seen` grew forever and every item scanned all of it —
    quadratic in a stream that never ends."""
    from dataclasses import replace

    pipeline = NewsIntelligencePipeline()
    base = fixture_events()[0]
    for day in range(6):
        pipeline.process(
            replace(
                base,
                raw_event_id=f"raw_day{day}",
                source_ref=f"mock://white_house/day{day}",
                raw_payload={"day": day},
                published_at=base.published_at + timedelta(days=day),
                first_seen_at=base.first_seen_at + timedelta(days=day),
                received_at=base.received_at + timedelta(days=day),
            )
        )
    assert pipeline.novelty.remembered() <= 4, (
        "a three-day memory must not still be holding six days of items"
    )


def test_the_clusterer_forgets_clusters_that_can_no_longer_match():
    from dataclasses import replace

    pipeline = NewsIntelligencePipeline()
    base = fixture_events()[0]
    for day in range(6):
        pipeline.process(
            replace(
                base,
                raw_event_id=f"raw_c{day}",
                source_ref=f"mock://white_house/c{day}",
                raw_payload={"c": day},
                published_at=base.published_at + timedelta(days=day),
                first_seen_at=base.first_seen_at + timedelta(days=day),
                received_at=base.received_at + timedelta(days=day),
            )
        )
    assert pipeline.clusterer.cluster_count() <= 2, (
        "clusters outside the matching window are unreachable and must not be retained"
    )


# --- parameter fragility ---------------------------------------------------

def test_a_grouping_decided_close_to_the_threshold_is_flagged():
    """The sensitivity sweep put the chosen similarity threshold on the edge of
    the range that reproduces the intended grouping on the fixtures. Seven
    synthetic items are far too few to move the number on, so the fragility is
    measured instead: how often the threshold actually decided the answer."""
    pipeline = NewsIntelligencePipeline()
    processed = pipeline.process_batch(fixture_events())

    assert pipeline.clusterer.near_threshold_count() >= 1, (
        "on these fixtures at least one grouping sits within the margin"
    )
    flagged = [p for p in processed if p.cluster.near_threshold]
    assert flagged, "the row itself must carry the flag, not just the counter"
    assert all(p.event.entity == "donald trump" for p in flagged), (
        "the borderline case is the aggregator's short roundup of the tariff story"
    )


def test_the_gauge_stays_quiet_when_nothing_is_borderline():
    """A gauge that always reads high is not a gauge."""
    from eclipse.news_intelligence.clustering.clusterer import LexicalClusterer

    pipeline = NewsIntelligencePipeline()
    pipeline.clusterer = LexicalClusterer(threshold=0.32, margin=0.0)
    pipeline.process_batch(fixture_events())
    assert pipeline.clusterer.near_threshold_count() == 0
