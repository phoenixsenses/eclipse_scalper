"""The structured event.

Derived from exactly one `RawEvent` and always able to name it. Every judgement
on this object carries three things beside it: who made it, how sure they were,
and under which vocabulary version. A label without those is an opinion that has
lost its author.

Confidence is deliberately a separate field from the label rather than a
qualifier baked into it. `TARIFF@0.55` and `TARIFF@0.99` must be the same
category for grouping and different rows for filtering; fusing them produces a
taxonomy that quietly grows a new class every time a model hesitates.

There are no market fields on this object and there never will be. What the
price did afterwards lives in `schemas.reaction`, keyed by `event_id`, and the
two are joined only at research time by something that knows it is doing so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from ..taxonomy.events import EventType
from ..version import SCHEMA_VERSION, TAXONOMY_VERSION
from .relevance import AssetRelevance


@dataclass(frozen=True, slots=True)
class Judgement:
    """One labelled call plus its provenance.

    `model_id` is either a deterministic rule classifier or an LLM. Both are
    allowed; only one of them is reproducible, and the field is what lets a
    reader tell which kind of claim they are reading.
    """

    value: Any
    confidence: float
    model_id: str
    prompt_version: str | None = None
    produced_at: datetime | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")


@dataclass(frozen=True, slots=True)
class Sentiment:
    """Tone, kept small on purpose.

    Two fields, because "how positive" and "how strongly expressed" are
    different. A mild statement of good news and a furious statement of good
    news read the same on a single axis and behave differently.

    Tone is *not* the headline feature of this package. It is here because
    excluding it would prevent testing whether it matters, which is a different
    error from over-weighting it.
    """

    polarity: float  # -1 .. +1
    strength: float  # 0 .. 1

    def __post_init__(self) -> None:
        if not -1.0 <= self.polarity <= 1.0:
            raise ValueError(f"polarity {self.polarity} outside [-1, 1]")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength {self.strength} outside [0, 1]")


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    event_id: str
    raw_event_id: str

    published_at: datetime
    first_seen_at: datetime
    received_at: datetime

    entity: str
    secondary_entities: tuple[str, ...] = ()

    event_type: EventType = EventType.OTHER
    topic: str = ""
    subtopic: str = ""

    sentiment: Sentiment | None = None

    #: Scores produced by their own engines. Each is optional because an event
    #: can be stored before every engine has run — a partially scored event is
    #: honest, a defaulted one is not.
    novelty: float | None = None
    surprise: float | None = None
    credibility: float | None = None
    source_authority: str = "UNKNOWN"

    attention_velocity: float | None = None
    amplification_score: float | None = None
    expected_vs_unexpected: str | None = None

    asset_relevance: AssetRelevance = field(default_factory=AssetRelevance)
    country_relevance: tuple[str, ...] = ()
    sector_relevance: tuple[str, ...] = ()

    news_cluster_id: str | None = None

    judgements: Mapping[str, Judgement] = field(default_factory=dict)

    model_version: str = ""
    classifier_version: str = ""
    taxonomy_version: int = TAXONOMY_VERSION
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("published_at", "first_seen_at", "received_at"):
            value = getattr(self, name)
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(timezone.utc))

    @property
    def decision_time(self) -> datetime:
        """The moment this system could first have acted on the event.

        `first_seen_at`, never `published_at`. Anchoring to publication would
        credit the system with information it did not have, by exactly the
        delivery lag — which is largest precisely when the news matters most and
        everyone is hitting the same wire.
        """
        return self.first_seen_at

    def confidence_in(self, field_name: str) -> float | None:
        judgement = self.judgements.get(field_name)
        return None if judgement is None else judgement.confidence


__all__ = ["NormalizedEvent", "Judgement", "Sentiment"]
