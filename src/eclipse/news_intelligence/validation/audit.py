"""Why was this event classified this way?

A question the system must be able to answer months later, about a row written
by a model version that no longer exists. If the only answer is "the classifier
said so", the row is not evidence — it is a number with a provenance-shaped hole
where its justification should be.

`explain` walks back from a normalized event to the raw item, the rules or model
that produced each judgement, the confidence attached, and the graph edges that
put each asset on the list. Nothing is recomputed: the explanation is assembled
from what was stored, so an explanation that cannot be produced is itself the
finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..relevance.graph import RelevanceGraph
from ..schemas.normalized import NormalizedEvent
from ..schemas.raw import RawEvent


@dataclass(frozen=True, slots=True)
class Explanation:
    event_id: str
    raw_event_id: str
    source_id: str
    source_authority: str
    source_ref: str

    published_at: str
    first_seen_at: str
    received_at: str
    decision_time: str
    publication_lag_seconds: float

    event_type: str
    entity: str
    secondary_entities: tuple[str, ...]
    topic: str

    judgements: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    asset_reasons: Mapping[str, str] = field(default_factory=dict)
    versions: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "raw_event_id": self.raw_event_id,
            "source": {
                "source_id": self.source_id,
                "authority": self.source_authority,
                "ref": self.source_ref,
            },
            "timestamps": {
                "published_at": self.published_at,
                "first_seen_at": self.first_seen_at,
                "received_at": self.received_at,
                "decision_time": self.decision_time,
                "publication_lag_seconds": self.publication_lag_seconds,
            },
            "classification": {
                "event_type": self.event_type,
                "entity": self.entity,
                "secondary_entities": list(self.secondary_entities),
                "topic": self.topic,
            },
            "judgements": {k: dict(v) for k, v in self.judgements.items()},
            "asset_relevance_reasons": dict(self.asset_reasons),
            "versions": dict(self.versions),
            "warnings": list(self.warnings),
        }


def explain(event: NormalizedEvent, raw: RawEvent, graph: RelevanceGraph | None = None) -> Explanation:
    if raw.raw_event_id != event.raw_event_id:
        raise ValueError(
            f"raw {raw.raw_event_id!r} does not belong to event {event.event_id!r}; "
            "an explanation assembled from the wrong source is worse than none"
        )

    judgements = {
        name: {
            "value": getattr(j.value, "value", j.value),
            "confidence": j.confidence,
            "model_id": j.model_id,
            "prompt_version": j.prompt_version,
            "produced_at": j.produced_at.isoformat() if j.produced_at else None,
        }
        for name, j in event.judgements.items()
    }

    warnings: list[str] = []
    if not event.entity:
        warnings.append("no entity was identified; asset relevance is empty by consequence")
    for name, j in event.judgements.items():
        if j.confidence < 0.5:
            warnings.append(f"low confidence on {name}: {j.confidence:.2f}")
    if event.news_cluster_id is None:
        warnings.append("not clustered; this event may not be an independent observation")

    return Explanation(
        event_id=event.event_id,
        raw_event_id=raw.raw_event_id,
        source_id=raw.source_id,
        source_authority=event.source_authority,
        source_ref=raw.source_ref,
        published_at=event.published_at.isoformat(),
        first_seen_at=event.first_seen_at.isoformat(),
        received_at=event.received_at.isoformat(),
        decision_time=event.decision_time.isoformat(),
        publication_lag_seconds=raw.publication_lag_seconds,
        event_type=event.event_type.value,
        entity=event.entity,
        secondary_entities=event.secondary_entities,
        topic=event.topic,
        judgements=judgements,
        asset_reasons={
            asset: event.asset_relevance.explain(asset)
            for asset in event.asset_relevance.relevant()
        },
        versions={
            "schema_version": event.schema_version,
            "taxonomy_version": event.taxonomy_version,
            "graph_version": event.asset_relevance.graph_version,
            "classifier_version": event.classifier_version,
            "model_version": event.model_version,
        },
        warnings=tuple(warnings),
    )


__all__ = ["explain", "Explanation"]
