"""The feature snapshot — everything that was knowable, and nothing else.

This is the object research is allowed to fit on. It is immutable, it is stamped
with a decision time, and it validates two different things at construction:

  1. **Every observation it carries is older than the decision time.** Each
     `Observation` brings its own `as_of`; one from the future is a
     `LookaheadError` at build time, not a puzzle six months later.

  2. **No field name belongs to outcome space.** A realised return carries no
     timestamp, so rule 1 cannot see it. Rule 2 is a structural check against
     `reaction.OUTCOME_FIELDS`, and it applies to the keys of the free-form
     context dict as well as to the dataclass fields.

The pairing object, `ResearchLabel`, deliberately has no `features` field and
this one deliberately has no `label` field. They are joined by `event_id` at
research time, by code that had to type the join out — which is the point. The
one thing that must never be possible is a convenience constructor that takes
both.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Mapping

from ..errors import LookaheadError, OutcomeInFeatureSpace
from ..version import SCHEMA_VERSION
from .reaction import OUTCOME_FIELDS, CrossAssetContext, MarketReaction
from .relevance import AssetRelevance


@dataclass(frozen=True, slots=True)
class Observation:
    """One measured quantity plus the moment it was true.

    The `as_of` is not bookkeeping. A volatility regime computed this morning
    and a volatility regime computed after the event are the same number with
    different meanings, and only the timestamp separates them.
    """

    name: str
    value: Any
    as_of: datetime
    source: str = ""

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError(f"observation {self.name!r} has a naive as_of")
        object.__setattr__(self, "as_of", self.as_of.astimezone(timezone.utc))
        if self.name in OUTCOME_FIELDS:
            raise OutcomeInFeatureSpace(
                f"observation {self.name!r} names an outcome; it cannot be a feature "
                "however early it is timestamped"
            )


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    event_id: str
    decision_time: datetime

    event_type: str = ""
    entity: str = ""
    topic: str = ""

    sentiment_polarity: float | None = None
    sentiment_strength: float | None = None
    novelty: float | None = None
    surprise: float | None = None
    credibility: float | None = None
    source_authority: str = "UNKNOWN"
    attention_velocity: float | None = None
    amplification_score: float | None = None

    asset_relevance: AssetRelevance = field(default_factory=AssetRelevance)

    #: Pre-event market state, volatility regime, liquidation pressure, current
    #: E-DER state and anything else a study wants — each carrying its own
    #: `as_of`, each checked against `decision_time`.
    observations: tuple[Observation, ...] = ()

    #: Free-form extras. Checked against outcome space by key, because a dict is
    #: exactly where an outcome sneaks in wearing a different name.
    context: Mapping[str, Any] = field(default_factory=dict)

    news_cluster_id: str | None = None
    taxonomy_version: int = 0
    graph_version: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        object.__setattr__(self, "decision_time", self.decision_time.astimezone(timezone.utc))

        for observation in self.observations:
            if observation.as_of > self.decision_time:
                raise LookaheadError(
                    f"observation {observation.name!r} is as of "
                    f"{observation.as_of.isoformat()}, after the decision time "
                    f"{self.decision_time.isoformat()}"
                )

        for key in self.context:
            if key in OUTCOME_FIELDS:
                raise OutcomeInFeatureSpace(
                    f"context key {key!r} names an outcome; put it on the ResearchLabel"
                )

        for f in fields(self):
            if f.name in OUTCOME_FIELDS:
                raise OutcomeInFeatureSpace(f"field {f.name!r} names an outcome")

    def observation(self, name: str) -> Observation | None:
        for o in self.observations:
            if o.name == name:
                return o
        return None

    def as_row(self) -> dict[str, Any]:
        """Flat mapping for a research frame. Outcome-free by construction."""
        row: dict[str, Any] = {
            "event_id": self.event_id,
            "decision_time": self.decision_time.isoformat(),
            "event_type": self.event_type,
            "entity": self.entity,
            "topic": self.topic,
            "sentiment_polarity": self.sentiment_polarity,
            "sentiment_strength": self.sentiment_strength,
            "novelty": self.novelty,
            "surprise": self.surprise,
            "credibility": self.credibility,
            "source_authority": self.source_authority,
            "attention_velocity": self.attention_velocity,
            "amplification_score": self.amplification_score,
            "news_cluster_id": self.news_cluster_id,
            "taxonomy_version": self.taxonomy_version,
            "graph_version": self.graph_version,
            "schema_version": self.schema_version,
        }
        for asset, weight in sorted(self.asset_relevance.weights.items()):
            row[f"relevance_{asset}"] = weight
        for observation in self.observations:
            row[f"obs_{observation.name}"] = observation.value
        for key, value in sorted(self.context.items()):
            row[f"ctx_{key}"] = value
        return row


@dataclass(frozen=True, slots=True)
class ResearchLabel:
    """The outcome side. Joined to a snapshot by `event_id`, never merged into it."""

    event_id: str
    reaction: MarketReaction | None = None
    cross_asset: tuple[CrossAssetContext, ...] = ()
    resolved: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name in {"features", "snapshot", "feature_snapshot"}:
                raise OutcomeInFeatureSpace(
                    "a label may not carry its own features; join them by event_id"
                )


__all__ = ["FeatureSnapshot", "Observation", "ResearchLabel"]
