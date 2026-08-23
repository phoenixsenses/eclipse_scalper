"""The whole path, in order, with nothing heavy in it.

raw -> normalize -> cluster -> novelty -> amplify -> snapshot -> publish

Every stage is a small object the caller could have wired itself; this module
exists so the *order* is written down once. Two orderings matter and are easy to
get wrong:

  **Cluster before novelty.** Novelty asks "have we seen this before?", and the
  answer depends on what counts as the same thing. Scoring novelty first makes
  every reprint look like a fresh item with a suspiciously familiar score.

  **Snapshot before any label exists.** The snapshot is built from the event
  alone. Nothing in this file has access to a reaction, and the reaction request
  it emits is a request — the measurement itself happens later, elsewhere, and
  never flows back into the snapshot object.

Processing one item touches no network, no database and no disk. That is what
makes it safe to run while the machine is busy, and it is why the expensive
parts live in `deferred` rather than behind a flag here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable

from .adapters.base import SourceRegistry, default_registry
from .amplification.engine import AmplificationEngine, AmplificationResult
from .clustering.clusterer import ClusterAssignment, ClusterInput, LexicalClusterer
from .normalization.normalizer import Normalizer
from .novelty.engine import LexicalNoveltyEngine, NoveltyResult
from .publishing.bus import SUBJECTS, Envelope, InMemoryPublisher, Publisher
from .reaction.contracts import build_request
from .relevance.graph import RelevanceGraph, default_graph
from .research.snapshot_builder import build_snapshot
from .schemas.normalized import NormalizedEvent
from .schemas.raw import RawEvent
from .schemas.reaction import ReactionRequest
from .schemas.snapshot import FeatureSnapshot, Observation

#: Relevance at or above which an event is worth telling the rest of the system
#: about. Not an impact prediction — a routing threshold.
HIGH_IMPACT_RELEVANCE = 0.7


@dataclass(frozen=True, slots=True)
class ProcessedEvent:
    event: NormalizedEvent
    snapshot: FeatureSnapshot
    reaction_request: ReactionRequest
    cluster: ClusterAssignment
    novelty: NoveltyResult
    amplification: AmplificationResult

    @property
    def is_independent_observation(self) -> bool:
        """True only for the first item in its cluster.

        The distinction between raw items and independent events, made
        available at the row level so a study cannot pool them by accident.
        """
        return self.cluster.is_new_cluster

    @property
    def is_high_impact(self) -> bool:
        weights = self.event.asset_relevance.weights
        return bool(weights) and max(weights.values()) >= HIGH_IMPACT_RELEVANCE


class NewsIntelligencePipeline:
    def __init__(
        self,
        registry: SourceRegistry | None = None,
        graph: RelevanceGraph | None = None,
        publisher: Publisher | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.graph = graph or default_graph()
        self.normalizer = Normalizer(self.registry, self.graph)
        self.clusterer = LexicalClusterer()
        self.novelty = LexicalNoveltyEngine()
        self.amplification = AmplificationEngine()
        self.publisher = publisher or InMemoryPublisher()

    def process(
        self,
        raw: RawEvent,
        observations: Iterable[Observation] = (),
    ) -> ProcessedEvent:
        event = self.normalizer.normalize(raw)

        cluster_input = ClusterInput.of(event, raw.raw_title, raw.raw_text, raw.source_id)
        assignment = self.clusterer.assign(cluster_input)
        novelty = self.novelty.score(cluster_input)
        amplification = self.amplification.observe(assignment, event.first_seen_at, raw.source_id)

        event = replace(
            event,
            news_cluster_id=assignment.cluster_id,
            novelty=novelty.novelty_score,
            attention_velocity=amplification.attention_velocity,
            amplification_score=amplification.amplification_score,
        )

        snapshot = build_snapshot(event, observations)
        request = build_request(event)

        self._publish(event, snapshot, raw)

        return ProcessedEvent(
            event=event,
            snapshot=snapshot,
            reaction_request=request,
            cluster=assignment,
            novelty=novelty,
            amplification=amplification,
        )

    def _publish(self, event: NormalizedEvent, snapshot: FeatureSnapshot, raw: RawEvent) -> None:
        now = datetime.now(timezone.utc)
        producer = "news-intelligence@0.1.0"

        self.publisher.publish(
            Envelope(
                subject=SUBJECTS["news_raw"],
                payload={
                    "raw_event_id": raw.raw_event_id,
                    "source_id": raw.source_id,
                    "source_ref": raw.source_ref,
                    "published_at": raw.published_at.isoformat(),
                    "first_seen_at": raw.first_seen_at.isoformat(),
                    "payload_digest": raw.payload_digest,
                },
                published_at=now,
                producer=producer,
            )
        )
        self.publisher.publish(
            Envelope(
                subject=SUBJECTS["news_normalized"],
                payload=snapshot.as_row(),
                published_at=now,
                producer=producer,
            )
        )
        weights = event.asset_relevance.weights
        if weights and max(weights.values()) >= HIGH_IMPACT_RELEVANCE:
            self.publisher.publish(
                Envelope(
                    subject=SUBJECTS["news_high_impact"],
                    payload={
                        "event_id": event.event_id,
                        "entity": event.entity,
                        "event_type": event.event_type.value,
                        "decision_time": event.decision_time.isoformat(),
                        "assets": list(event.asset_relevance.relevant(HIGH_IMPACT_RELEVANCE)),
                        "novelty": event.novelty,
                    },
                    published_at=now,
                    producer=producer,
                )
            )
        self.publisher.publish(
            Envelope(
                subject=SUBJECTS["research_ready"],
                payload={
                    "event_id": event.event_id,
                    "news_cluster_id": event.news_cluster_id,
                    "decision_time": event.decision_time.isoformat(),
                    "awaiting_horizons": list(build_request(event).horizons),
                },
                published_at=now,
                producer=producer,
            )
        )


__all__ = ["NewsIntelligencePipeline", "ProcessedEvent", "HIGH_IMPACT_RELEVANCE"]
