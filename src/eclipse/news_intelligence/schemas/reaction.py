"""What the market did — labels, never features.

Everything in this module is knowable only after the fact, so nothing in it may
appear in a feature object. That rule is enforced structurally in
`schemas.snapshot` against `OUTCOME_FIELDS` below, because a timestamp check
alone cannot catch it: a realised return is a number with no clock attached, and
it will pass any "is this from the future?" test you write.

The pre-event windows are here rather than in the snapshot on purpose. They are
used for one job — asking whether the price had already moved before the news
arrived — and that job is reverse-causality analysis, which is research, not
signal construction. Putting them in the label object keeps the question
answerable without letting the answer leak into a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from ..version import SCHEMA_VERSION

#: Minutes relative to the decision time. Negative is before.
PRE_EVENT_HORIZONS: tuple[int, ...] = (-30, -15, -5, -1)
POST_EVENT_HORIZONS: tuple[int, ...] = (1, 5, 15, 30, 60, 240)
ALL_HORIZONS: tuple[int, ...] = PRE_EVENT_HORIZONS + POST_EVENT_HORIZONS

#: Field names that can only be known after the decision time. The snapshot
#: refuses any of these, whatever they are called on the way in.
OUTCOME_FIELDS = frozenset(
    {
        "return_bps",
        "realised_return",
        "forward_return",
        "volume_change",
        "realised_volatility",
        "spread_after",
        "liquidation_pressure_after",
        "flow_imbalance_after",
        "relative_strength_after",
        "reaction",
        "reactions",
        "label",
        "labels",
        "outcome",
        "outcomes",
        "pnl",
        "win",
    }
)


@dataclass(frozen=True, slots=True)
class HorizonMeasurement:
    """One asset, one horizon, one set of measured quantities.

    `complete` is not decoration. A horizon whose data was missing must be
    distinguishable from one that was measured as zero; without the flag those
    two collapse and a feed gap starts looking like a calm market — the failure
    this repository has already paid for once.
    """

    asset: str
    horizon_minutes: int
    return_bps: float | None = None
    volume_change: float | None = None
    realised_volatility: float | None = None
    spread: float | None = None
    liquidation_pressure: float | None = None
    flow_imbalance: float | None = None
    relative_strength: float | None = None
    complete: bool = False
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.complete and self.missing_reason is None:
            raise ValueError(
                "an incomplete measurement must say why; an unexplained gap is "
                "indistinguishable from a quiet market"
            )


@dataclass(frozen=True, slots=True)
class MarketReaction:
    """The full set of measurements attached to one event."""

    event_id: str
    measured_at: datetime
    measurements: tuple[HorizonMeasurement, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.measured_at.tzinfo is None:
            raise ValueError("measured_at must be timezone-aware")
        object.__setattr__(self, "measured_at", self.measured_at.astimezone(timezone.utc))

    def get(self, asset: str, horizon_minutes: int) -> HorizonMeasurement | None:
        for m in self.measurements:
            if m.asset == asset and m.horizon_minutes == horizon_minutes:
                return m
        return None

    @property
    def is_complete(self) -> bool:
        """True only when every requested measurement landed.

        A partially observed event is usable for some questions and not others,
        and the research counter on the private dashboard reports the complete
        count separately for exactly that reason.
        """
        return bool(self.measurements) and all(m.complete for m in self.measurements)


@dataclass(frozen=True, slots=True)
class CrossAssetContext:
    """Synchronised moves across the universe around one event.

    Stored without any claim about who moved first. Lead, lag, simultaneity and
    no-relationship are four hypotheses, and this object is the input to testing
    them rather than a record of one having been assumed.
    """

    event_id: str
    horizon_minutes: int
    moves: Mapping[str, float] = field(default_factory=dict)
    complete_assets: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ReactionRequest:
    """What the reaction engine is being asked to measure, and from when.

    A request, not a measurement: building it requires no market data, so a
    pipeline can be fully exercised — and tested — with no access to any price
    store at all. Filling it in is the deferred, expensive half.
    """

    event_id: str
    decision_time: datetime
    assets: tuple[str, ...]
    horizons: tuple[int, ...] = ALL_HORIZONS

    @property
    def earliest_needed(self) -> datetime:
        from datetime import timedelta

        return self.decision_time + timedelta(minutes=min(self.horizons))

    @property
    def latest_needed(self) -> datetime:
        from datetime import timedelta

        return self.decision_time + timedelta(minutes=max(self.horizons))

    def ready_at(self) -> datetime:
        """When every requested horizon has closed.

        A label read before this is a partial label. Research that treats it as
        final gets a survivorship effect for free: the events that resolve early
        are not a random sample of events.
        """
        return self.latest_needed


__all__ = [
    "HorizonMeasurement",
    "MarketReaction",
    "CrossAssetContext",
    "ReactionRequest",
    "PRE_EVENT_HORIZONS",
    "POST_EVENT_HORIZONS",
    "ALL_HORIZONS",
    "OUTCOME_FIELDS",
]
