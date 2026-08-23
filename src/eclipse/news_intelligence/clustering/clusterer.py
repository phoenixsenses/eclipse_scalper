"""One real-world event, however many articles describe it.

Four outlets covering the same tariff announcement are one observation, not
four. Getting this wrong is not a rounding error: sample size is the denominator
of every significance test this system will ever run, and counting reprints as
independent events inflates it in exactly the direction that makes noise look
like a finding.

**The clusterer is structurally outcome-blind.** It accepts `ClusterInput`, a
projection that carries text, entities, type and time and physically cannot
carry a reaction. That is stronger than a rule saying "don't look at outcomes":
if grouping could depend on what happened next, the number of independent
observations would become a function of the result, which is the most efficient
way to manufacture significance ever devised.

The matcher is lexical and cheap on purpose. An embedding index is the right
long-term answer and the wrong thing to start while the machine is busy — see
`deferred`. The interface is the same either way.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Protocol, runtime_checkable

from ..errors import OutcomeAwareClustering
from ..schemas.normalized import NormalizedEvent
from ..schemas.reaction import OUTCOME_FIELDS

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """a an the of to in on for and or is are was were be been as at by from with that this
    it its will would say says said after before over under new more than about""".split()
)


@dataclass(frozen=True, slots=True)
class ClusterInput:
    """Everything the clusterer is allowed to see.

    Built by `ClusterInput.of`, which reads a `NormalizedEvent` and drops
    everything else. There is no field here that could hold a return.
    """

    event_id: str
    first_seen_at: datetime
    entity: str
    event_type: str
    topic: str
    title: str
    text: str
    source_id: str

    @staticmethod
    def of(event: NormalizedEvent, title: str, text: str, source_id: str) -> "ClusterInput":
        return ClusterInput(
            event_id=event.event_id,
            first_seen_at=event.first_seen_at,
            entity=event.entity,
            event_type=event.event_type.value,
            topic=event.topic,
            title=title,
            text=text,
            source_id=source_id,
        )

    def tokens(self) -> frozenset[str]:
        words = _WORD.findall(f"{self.title} {self.text}".lower())
        return frozenset(w for w in words if w not in _STOP and len(w) > 2)


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    cluster_id: str
    is_new_cluster: bool
    similarity: float
    matched_event_id: str | None
    source_count: int
    update_count: int
    first_source_id: str
    first_seen_at: datetime


@runtime_checkable
class Clusterer(Protocol):
    def assign(self, item: ClusterInput) -> ClusterAssignment:
        ...


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LexicalClusterer:
    """Group by shared vocabulary within a time window.

    Two guards beyond similarity: the entity must match, and the item must
    arrive within `window`. Without the window a story that recurs monthly
    collapses into one eternal cluster and its recurrence — which is a real
    signal about novelty — becomes invisible.
    """

    def __init__(self, threshold: float = 0.32, window: timedelta = timedelta(hours=6)) -> None:
        self.threshold = threshold
        self.window = window
        self._clusters: dict[str, dict] = {}

    def assign(self, item: ClusterInput) -> ClusterAssignment:
        for field_name in ("reaction", "label", "outcome"):
            if hasattr(item, field_name):
                raise OutcomeAwareClustering(
                    f"cluster input exposes {field_name!r}; grouping must not be able to "
                    "see what happened next"
                )

        tokens = item.tokens()
        best_id, best_similarity, best_event = None, 0.0, None
        for cluster_id, cluster in self._clusters.items():
            if cluster["entity"] != item.entity:
                continue
            if item.first_seen_at - cluster["last_seen_at"] > self.window:
                continue
            similarity = jaccard(tokens, cluster["tokens"])
            if similarity > best_similarity:
                best_id, best_similarity, best_event = cluster_id, similarity, cluster["first_event_id"]

        if best_id is not None and best_similarity >= self.threshold:
            cluster = self._clusters[best_id]
            cluster["update_count"] += 1
            cluster["sources"].add(item.source_id)
            cluster["last_seen_at"] = item.first_seen_at
            cluster["tokens"] = cluster["tokens"] | tokens
            return ClusterAssignment(
                cluster_id=best_id,
                is_new_cluster=False,
                similarity=best_similarity,
                matched_event_id=best_event,
                source_count=len(cluster["sources"]),
                update_count=cluster["update_count"],
                first_source_id=cluster["first_source_id"],
                first_seen_at=cluster["first_seen_at"],
            )

        cluster_id = "cl_" + hashlib.sha256(
            f"{item.entity}|{item.event_type}|{item.first_seen_at.isoformat()}|{item.event_id}".encode()
        ).hexdigest()[:20]
        self._clusters[cluster_id] = {
            "entity": item.entity,
            "tokens": tokens,
            "first_event_id": item.event_id,
            "first_source_id": item.source_id,
            "first_seen_at": item.first_seen_at,
            "last_seen_at": item.first_seen_at,
            "sources": {item.source_id},
            "update_count": 1,
        }
        return ClusterAssignment(
            cluster_id=cluster_id,
            is_new_cluster=True,
            similarity=best_similarity,
            matched_event_id=best_event if best_similarity else None,
            source_count=1,
            update_count=1,
            first_source_id=item.source_id,
            first_seen_at=item.first_seen_at,
        )

    def cluster_count(self) -> int:
        return len(self._clusters)

    def state_of(self, cluster_id: str) -> dict:
        cluster = self._clusters[cluster_id]
        return {
            "first_source": cluster["first_source_id"],
            "first_seen": cluster["first_seen_at"],
            "source_count": len(cluster["sources"]),
            "update_count": cluster["update_count"],
        }


def assert_outcome_blind(cls: type) -> None:
    """Fail if a cluster input type ever grows an outcome field."""
    from dataclasses import fields as dc_fields

    for f in dc_fields(cls):
        if f.name in OUTCOME_FIELDS:
            raise OutcomeAwareClustering(f"{cls.__name__}.{f.name} is an outcome field")


__all__ = [
    "ClusterInput",
    "ClusterAssignment",
    "Clusterer",
    "LexicalClusterer",
    "jaccard",
    "assert_outcome_blind",
]
