"""News context beside an E-DER candidate — never inside one.

V1, A2 and V3 are frozen. This module cannot change them, cannot filter them and
cannot gate them, and the shape of the code says so: it produces a
`NewsContext`, a separate object keyed by candidate id. There is no function
here that takes an arm's rule set, and none will be added.

The reason is not deference, it is arithmetic. A frozen arm's forward record is
only evidence because its definition did not move while the sample accumulated.
The moment news state filters which candidates count, the arm under test is a
different arm with a sample of zero — and the old sample cannot be carried over
to it, however tempting the continuity looks.

So the honest path is: record the context, let it accumulate beside the existing
record, and when there is enough of it, *declare a new arm* and start its sample
at zero. `proposed_arm_name` exists to make that explicit rather than gradual.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping

from ..schemas.normalized import NormalizedEvent

#: Arms this layer may describe but never modify.
FROZEN_ARMS = frozenset({"E-DER-V1", "E-DER-A2", "E-DER-V3"})


class ArmModificationRefused(Exception):
    """Raised if anything tries to route news state into a frozen arm's decision."""


@dataclass(frozen=True, slots=True)
class NewsContext:
    """What the news layer knew at the moment a candidate formed.

    Attached to a candidate id, stored separately, and read only at research
    time. Nothing in the candidate's own path reads this object.
    """

    candidate_id: str
    arm: str
    observed_at: datetime

    high_impact_news: bool = False
    event_id: str | None = None
    event_type: str | None = None
    entity: str | None = None
    event_age_minutes: float | None = None
    novelty: float | None = None
    amplification: float | None = None
    source_authority: str | None = None
    relevance_to_arm_asset: float | None = None
    global_context: str = "UNCLASSIFIED"

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(timezone.utc))

    def as_annotation(self) -> Mapping[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "arm": self.arm,
            "observed_at": self.observed_at.isoformat(),
            "news_context": {
                "high_impact_news": self.high_impact_news,
                "event_id": self.event_id,
                "event_type": self.event_type,
                "entity": self.entity,
                "event_age_minutes": self.event_age_minutes,
                "novelty": self.novelty,
                "amplification": self.amplification,
                "source_authority": self.source_authority,
                "relevance": self.relevance_to_arm_asset,
                "global_context": self.global_context,
            },
        }


def context_for_candidate(
    candidate_id: str,
    arm: str,
    candidate_time: datetime,
    recent_events: tuple[NormalizedEvent, ...],
    asset: str = "BTC",
    max_age: timedelta = timedelta(minutes=60),
    high_impact_relevance: float = 0.5,
) -> NewsContext:
    """Summarise the news state at a candidate's moment, using only earlier events.

    Events with `first_seen_at` after the candidate are excluded, not because
    they are irrelevant but because including them would answer the question
    with information the candidate could not have had — the same lookahead the
    snapshot guards against, arriving through a side door.
    """
    known = [
        e
        for e in recent_events
        if e.first_seen_at <= candidate_time and candidate_time - e.first_seen_at <= max_age
    ]
    if not known:
        return NewsContext(
            candidate_id=candidate_id,
            arm=arm,
            observed_at=candidate_time,
            high_impact_news=False,
            global_context="NO_RECENT_NEWS",
        )

    # The most relevant recent event, ties broken by recency. Relevance, not
    # sentiment: the loudest headline is not the one most likely to matter.
    def rank(event: NormalizedEvent) -> tuple[float, datetime]:
        return (event.asset_relevance.weight(asset), event.first_seen_at)

    top = max(known, key=rank)
    relevance = top.asset_relevance.weight(asset)
    age_minutes = (candidate_time - top.first_seen_at).total_seconds() / 60.0

    return NewsContext(
        candidate_id=candidate_id,
        arm=arm,
        observed_at=candidate_time,
        high_impact_news=relevance >= high_impact_relevance,
        event_id=top.event_id,
        event_type=top.event_type.value,
        entity=top.entity,
        event_age_minutes=round(age_minutes, 3),
        novelty=top.novelty,
        amplification=top.amplification_score,
        source_authority=top.source_authority,
        relevance_to_arm_asset=relevance,
        global_context="RECENT_RELEVANT_NEWS" if relevance >= high_impact_relevance else "BACKGROUND_NEWS",
    )


def proposed_arm_name(base_arm: str, condition: str) -> str:
    """Name for a future arm that combines an existing one with a news condition.

    Refuses the frozen names outright. Combining is allowed; *renaming the
    combination back onto the frozen arm* is the thing that would destroy the
    forward record, and it is the mistake that feels most natural to make.
    """
    if base_arm not in FROZEN_ARMS:
        raise ArmModificationRefused(
            f"{base_arm!r} is not one of the frozen arms {sorted(FROZEN_ARMS)}"
        )
    candidate = f"{base_arm}+NEWS_{condition.upper()}"
    if candidate in FROZEN_ARMS:
        raise ArmModificationRefused("a combined arm may not take a frozen arm's name")
    return candidate


__all__ = [
    "NewsContext",
    "context_for_candidate",
    "proposed_arm_name",
    "FROZEN_ARMS",
    "ArmModificationRefused",
]
