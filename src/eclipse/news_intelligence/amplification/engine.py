"""How loudly the same thing is being repeated — which is not how new it is.

Novelty and attention move in opposite directions on the same story and must
never be collapsed into one "impact" number:

  13:44  a statement lands            novelty high, amplification none
  14:05  fifty outlets carry it       novelty near zero, amplification high

Both are real and they are different features. Repetition is what tells you the
market has *seen* it; the original is what tells you the market could not have.
A single score that mixes them cannot express either, and a study using it will
be unable to tell "nobody noticed" from "everybody already knew".

Amplification is computed per cluster, because the unit being amplified is the
underlying event and not any one article.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..clustering.clusterer import ClusterAssignment


@dataclass(frozen=True, slots=True)
class AmplificationResult:
    cluster_id: str
    source_count: int
    update_count: int
    mention_velocity_per_min: float
    attention_velocity: float
    amplification_score: float
    window_seconds: float


class AmplificationEngine:
    """Track repetition per cluster over a rolling window.

    Distinct sources are weighted above repeat updates from the same source:
    one outlet filing five updates is one outlet paying attention, while five
    outlets filing once each is the story spreading. Treating those the same
    lets a single chatty feed look like market-wide interest.
    """

    def __init__(self, window: timedelta = timedelta(minutes=30)) -> None:
        self.window = window
        self._mentions: dict[str, list[tuple[datetime, str]]] = {}

    def observe(self, assignment: ClusterAssignment, at: datetime, source_id: str) -> AmplificationResult:
        mentions = self._mentions.setdefault(assignment.cluster_id, [])
        mentions.append((at, source_id))
        cutoff = at - self.window
        recent = [(t, s) for t, s in mentions if t >= cutoff]
        self._mentions[assignment.cluster_id] = recent

        elapsed = max((at - recent[0][0]).total_seconds(), 60.0)
        distinct_sources = len({s for _, s in recent})
        mention_velocity = len(recent) / (elapsed / 60.0)
        attention_velocity = distinct_sources / (elapsed / 60.0)

        # Bounded and monotonic in both counts. Deliberately crude: the shape of
        # this curve is not knowledge we have, so it should not pretend to be.
        amplification = 1.0 - 1.0 / (1.0 + 0.6 * (distinct_sources - 1) + 0.2 * (len(recent) - 1))

        return AmplificationResult(
            cluster_id=assignment.cluster_id,
            source_count=distinct_sources,
            update_count=len(recent),
            mention_velocity_per_min=round(mention_velocity, 6),
            attention_velocity=round(attention_velocity, 6),
            amplification_score=round(max(0.0, amplification), 6),
            window_seconds=elapsed,
        )


__all__ = ["AmplificationEngine", "AmplificationResult"]
