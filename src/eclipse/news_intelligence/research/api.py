"""Read-only research surface, and the counters that keep it honest.

Two design choices carry the weight here.

**Raw items and independent events are counted separately, always.** Every
counter that could be quoted as a sample size reports both, because the gap
between them is the whole reason clustering exists. A store that reports only
`raw_items` invites a study to use it as N, and N is the denominator of every
significance test that follows.

**Features and labels are joined by an explicit call.** `research_frame` takes
both sides and returns rows that say plainly which half is which, prefixing
outcome columns with `label_`. There is no accessor that hands back a merged
row with no marking, because that is exactly the object someone drops into a
model and fits on its own answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..schemas.snapshot import FeatureSnapshot, ResearchLabel


@dataclass(frozen=True, slots=True)
class ResearchCounters:
    raw_items: int
    independent_clusters: int
    high_impact_events: int
    events_with_complete_labels: int
    events_matched_to_eder: int

    @property
    def duplication_ratio(self) -> float:
        """Raw items per independent event.

        Published rather than hidden: it is the single number that says how much
        of the incoming stream is echo, and a study that quotes `raw_items` as
        its sample size is overstating N by exactly this factor.
        """
        if self.independent_clusters == 0:
            return 0.0
        return round(self.raw_items / self.independent_clusters, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_news_items": self.raw_items,
            "independent_news_clusters": self.independent_clusters,
            "high_impact_events": self.high_impact_events,
            "events_with_complete_market_labels": self.events_with_complete_labels,
            "events_matched_to_eder": self.events_matched_to_eder,
            "duplication_ratio": self.duplication_ratio,
        }


class ResearchStore:
    """In-memory store with the shape a database will later have.

    Deliberately not backed by anything. While the machine is busy the correct
    persistence layer is none, and the interface is what the later Postgres or
    Parquet implementation has to satisfy.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, FeatureSnapshot] = {}
        self._labels: dict[str, ResearchLabel] = {}
        self._clusters: set[str] = set()
        self._raw_count = 0
        self._high_impact: set[str] = set()
        self._eder_matched: set[str] = set()

    def add_snapshot(self, snapshot: FeatureSnapshot, high_impact: bool = False) -> None:
        self._snapshots[snapshot.event_id] = snapshot
        self._raw_count += 1
        if snapshot.news_cluster_id:
            self._clusters.add(snapshot.news_cluster_id)
        if high_impact:
            self._high_impact.add(snapshot.event_id)

    def add_label(self, label: ResearchLabel) -> None:
        if label.event_id not in self._snapshots:
            raise KeyError(
                f"label for unknown event {label.event_id!r}; a label without its "
                "feature side cannot be joined and would silently be dropped"
            )
        self._labels[label.event_id] = label

    def mark_eder_match(self, event_id: str) -> None:
        self._eder_matched.add(event_id)

    def counters(self) -> ResearchCounters:
        complete = sum(
            1
            for label in self._labels.values()
            if label.resolved and label.reaction is not None and label.reaction.is_complete
        )
        return ResearchCounters(
            raw_items=self._raw_count,
            independent_clusters=len(self._clusters),
            high_impact_events=len(self._high_impact),
            events_with_complete_labels=complete,
            events_matched_to_eder=len(self._eder_matched),
        )

    def snapshots(self) -> tuple[FeatureSnapshot, ...]:
        return tuple(self._snapshots.values())

    def label_for(self, event_id: str) -> ResearchLabel | None:
        return self._labels.get(event_id)

    def one_per_cluster(self) -> tuple[FeatureSnapshot, ...]:
        """The first snapshot of each cluster, in decision-time order.

        The projection a study should almost always use. Selecting the *first*
        rather than the best-scoring one keeps the choice outcome-blind; picking
        by any score would select on a quantity the study is about to measure.
        """
        best: dict[str, FeatureSnapshot] = {}
        for snapshot in sorted(self._snapshots.values(), key=lambda s: s.decision_time):
            key = snapshot.news_cluster_id or snapshot.event_id
            best.setdefault(key, snapshot)
        return tuple(sorted(best.values(), key=lambda s: s.decision_time))


def research_frame(
    snapshots: Sequence[FeatureSnapshot],
    labels: Mapping[str, ResearchLabel],
    asset: str,
    horizon_minutes: int,
) -> list[dict[str, Any]]:
    """Join one label column onto the feature rows, marked as a label.

    Rows whose label is missing or incomplete are returned with `label_*` set to
    None and `label_complete` False rather than being dropped. Dropping them
    would silently condition the sample on resolution, and events that resolve
    are not a random subset of events.
    """
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        row = snapshot.as_row()
        label = labels.get(snapshot.event_id)
        measurement = (
            label.reaction.get(asset, horizon_minutes)
            if label is not None and label.reaction is not None
            else None
        )
        row["label_asset"] = asset
        row["label_horizon_minutes"] = horizon_minutes
        row["label_return_bps"] = measurement.return_bps if measurement and measurement.complete else None
        row["label_complete"] = bool(measurement and measurement.complete)
        rows.append(row)
    return rows


__all__ = ["ResearchStore", "ResearchCounters", "research_frame"]
