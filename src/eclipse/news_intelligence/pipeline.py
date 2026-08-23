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
from .errors import DuplicateDelivery, OutOfOrderDelivery
from .normalization.normalizer import Normalizer, event_id_for
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
        #: Ids already processed. The stream is a stream, so this is bounded by
        #: the same memory the novelty engine keeps: an item older than that can
        #: no longer influence any score, so remembering it buys nothing.
        self._seen_event_ids: dict[str, datetime] = {}
        self._last_first_seen: datetime | None = None

    def process(
        self,
        raw: RawEvent,
        observations: Iterable[Observation] = (),
    ) -> ProcessedEvent:
        """Process one item. Refuses a re-delivery and refuses to go backwards.

        Both checks run *before* any engine sees the item. A refusal raised
        after clustering and amplification have already counted it would be a
        message about a record that is already wrong.
        """
        event_id = event_id_for(raw)
        if event_id in self._seen_event_ids:
            raise DuplicateDelivery(
                f"{event_id} has already been processed "
                f"(source {raw.source_id!r}, ref {raw.source_ref!r}, revision {raw.revision}). "
                "A re-delivery is not a second observation; counting it would invent "
                "attention that did not happen. Use process_if_new() if re-delivery is "
                "expected."
            )
        if self._last_first_seen is not None and raw.first_seen_at < self._last_first_seen:
            raise OutOfOrderDelivery(
                f"{raw.source_id} item is from {raw.first_seen_at.isoformat()}, before the "
                f"last processed item at {self._last_first_seen.isoformat()}. Cluster "
                "identity depends on arrival order — sort the batch first, or use "
                "process_batch(), which sorts for you."
            )

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

        self._seen_event_ids[event.event_id] = event.first_seen_at
        self._last_first_seen = event.first_seen_at
        self._forget_old(event.first_seen_at)

        self._publish(event, snapshot, raw)

        return ProcessedEvent(
            event=event,
            snapshot=snapshot,
            reaction_request=request,
            cluster=assignment,
            novelty=novelty,
            amplification=amplification,
        )

    def process_if_new(
        self,
        raw: RawEvent,
        observations: Iterable[Observation] = (),
    ) -> ProcessedEvent | None:
        """Collector-facing call: returns None for a re-delivery.

        Re-delivery is ordinary operation, so the common path should not require
        every caller to wrap `process` in a try block — a try block that would,
        soon enough, be written to swallow the ordering refusal too.
        """
        try:
            return self.process(raw, observations)
        except DuplicateDelivery:
            return None

    def process_batch(
        self,
        raws: Iterable[RawEvent],
        observations: Iterable[Observation] = (),
    ) -> list[ProcessedEvent]:
        """Sort by arrival, then process. The correct path for a backfill or a
        multi-source poll, where delivery order says nothing about event order."""
        ordered = sorted(raws, key=lambda r: (r.first_seen_at, r.source_id, r.raw_event_id))
        processed = []
        for raw in ordered:
            result = self.process_if_new(raw, observations)
            if result is not None:
                processed.append(result)
        return processed

    def _forget_old(self, now: datetime) -> None:
        """Drop ids that can no longer be reached by any comparison.

        Bounded by the novelty engine's own memory: past it, an item cannot
        affect a score, so holding its id only grows the process.
        """
        horizon = now - self.novelty.memory
        if len(self._seen_event_ids) > 64:
            self._seen_event_ids = {
                event_id: seen
                for event_id, seen in self._seen_event_ids.items()
                if seen >= horizon
            }

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
