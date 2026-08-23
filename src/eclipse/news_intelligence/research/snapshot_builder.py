"""Build the feature snapshot, and refuse to build a leaky one.

The builder takes a normalized event and a bag of market observations, and its
whole job is to say no. Every observation must carry the instant it was true;
anything stamped after the decision time raises rather than being dropped,
because a dropped observation is a silent change to the feature set and the
caller would never learn that the study they are about to run is not the study
they wrote.

There is deliberately no parameter here for outcomes, no optional label
argument, and no "include_future" flag. The absence is the design: a flag like
that gets set to True once, in a hurry, and nothing downstream can tell.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping

from ..errors import LookaheadError
from ..schemas.normalized import NormalizedEvent
from ..schemas.snapshot import FeatureSnapshot, Observation


def build_snapshot(
    event: NormalizedEvent,
    observations: Iterable[Observation] = (),
    context: Mapping[str, object] | None = None,
) -> FeatureSnapshot:
    """Assemble the knowable-at-the-time view of one event."""
    observations = tuple(observations)
    return FeatureSnapshot(
        event_id=event.event_id,
        decision_time=event.decision_time,
        event_type=event.event_type.value,
        entity=event.entity,
        topic=event.topic,
        sentiment_polarity=event.sentiment.polarity if event.sentiment else None,
        sentiment_strength=event.sentiment.strength if event.sentiment else None,
        novelty=event.novelty,
        surprise=event.surprise,
        credibility=event.credibility,
        source_authority=event.source_authority,
        attention_velocity=event.attention_velocity,
        amplification_score=event.amplification_score,
        asset_relevance=event.asset_relevance,
        observations=observations,
        context=dict(context or {}),
        news_cluster_id=event.news_cluster_id,
        taxonomy_version=event.taxonomy_version,
        graph_version=event.asset_relevance.graph_version,
    )


def market_state_observation(
    name: str,
    value: object,
    as_of: datetime,
    decision_time: datetime,
    source: str = "",
) -> Observation:
    """Build one pre-event observation, checking it against the decision time here too.

    The snapshot checks this as well. The duplication is on purpose: the caller
    site is where the mistake is made and where the error message is most
    useful, and a guard that only fires at the end tells you a batch is bad
    without telling you which line built it.
    """
    observation = Observation(name=name, value=value, as_of=as_of, source=source)
    if observation.as_of > decision_time.astimezone(observation.as_of.tzinfo):
        raise LookaheadError(
            f"{name!r} is as of {observation.as_of.isoformat()}, after the decision time "
            f"{decision_time.isoformat()}; it cannot be a feature of this event"
        )
    return observation


__all__ = ["build_snapshot", "market_state_observation"]
