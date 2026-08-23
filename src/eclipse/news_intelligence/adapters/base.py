"""The source adapter contract, and the registry of who is allowed to speak.

An adapter's only job is to produce `RawEvent`s. It does not classify, score,
cluster or judge — those are separable concerns that improve at different rates,
and an adapter that does them cannot be replaced without re-deriving history.

**Authority is a property of the source, not of the item.** A central bank
publishing a note about its cafeteria is still TIER 1; the note is still
irrelevant. Conflating the two is the most common way a news system talks itself
into treating a prestigious byline as a signal, so authority and relevance are
computed by different objects here and never multiplied together silently.

Every real adapter is deferred while the current research phase holds the
machine: a live collector is a network process with a queue and a retry loop,
which is exactly the kind of thing that must not be started casually. The mock
adapter is not a placeholder for a real one — it is the fixture that makes the
rest of the pipeline testable without a network at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Protocol, runtime_checkable

from ..errors import DeferredUntilPhase1Complete, UnknownSource
from ..schemas.raw import RawEvent


class SourceAuthority(str, Enum):
    """Who is speaking, ranked by how directly they know.

    Tiers describe proximity to the fact, not reliability of judgement and not
    market impact. A TIER_4 aggregator can be first, and being first is a
    separate, measurable property.
    """

    TIER_1_OFFICIAL = "TIER_1_OFFICIAL"
    TIER_2_PROFESSIONAL = "TIER_2_PROFESSIONAL"
    TIER_3_VERIFIED_INDIVIDUAL = "TIER_3_VERIFIED_INDIVIDUAL"
    TIER_4_AGGREGATOR = "TIER_4_AGGREGATOR"
    UNKNOWN = "UNKNOWN"


class SourceType(str, Enum):
    OFFICIAL = "OFFICIAL"
    NEWS = "NEWS"
    SOCIAL = "SOCIAL"
    MARKET = "MARKET"
    MOCK = "MOCK"


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    source_id: str
    display_name: str
    source_type: SourceType
    authority: SourceAuthority
    #: Whether pulling from this source needs credentials we do not hold, or
    #: touches a service whose terms forbid automated collection. Recorded so
    #: that "not implemented" and "not permitted" never look the same.
    requires_credentials: bool = False
    permitted: bool = True
    notes: str = ""


@runtime_checkable
class SourceAdapter(Protocol):
    """Produce raw items. Nothing else."""

    descriptor: SourceDescriptor

    def poll(self, since: datetime | None = None) -> Iterable[RawEvent]:
        ...


class DeferredAdapter:
    """A real adapter's shape with its engine removed.

    Refuses rather than returning an empty iterator. A collector that silently
    yields nothing is indistinguishable from a quiet news day, and this
    repository has already learned what it costs to confuse those two.
    """

    def __init__(self, descriptor: SourceDescriptor, reason: str = "") -> None:
        self.descriptor = descriptor
        self.reason = reason or "DEFERRED_UNTIL_PHASE1_COMPLETE"

    def poll(self, since: datetime | None = None) -> Iterable[RawEvent]:
        raise DeferredUntilPhase1Complete(
            f"{self.descriptor.source_id}: {self.reason}. The interface is complete; "
            "starting the collector is a separate, deliberate act."
        )


class SourceRegistry:
    """The set of sources whose items are allowed to enter the pipeline."""

    def __init__(self, descriptors: Iterable[SourceDescriptor] = ()) -> None:
        self._by_id: dict[str, SourceDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: SourceDescriptor) -> None:
        if descriptor.source_id in self._by_id:
            raise ValueError(f"source {descriptor.source_id!r} already registered")
        self._by_id[descriptor.source_id] = descriptor

    def get(self, source_id: str) -> SourceDescriptor:
        try:
            return self._by_id[source_id]
        except KeyError:
            raise UnknownSource(
                f"{source_id!r} is not registered. An unregistered source has no "
                "authority — not a default one."
            ) from None

    def authority_of(self, source_id: str) -> SourceAuthority:
        return self.get(source_id).authority

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(sorted(self._by_id.values(), key=lambda d: d.source_id))


#: The sources this layer is designed around. Everything real is deferred; the
#: descriptors exist now so that authority, permission and credential state are
#: recorded before any collector is written against them.
DEFAULT_SOURCES: tuple[SourceDescriptor, ...] = (
    SourceDescriptor("federal_reserve", "Federal Reserve", SourceType.OFFICIAL,
                     SourceAuthority.TIER_1_OFFICIAL, notes="statements, minutes, decisions"),
    SourceDescriptor("sec_edgar", "SEC EDGAR", SourceType.OFFICIAL,
                     SourceAuthority.TIER_1_OFFICIAL, notes="filings"),
    SourceDescriptor("bls", "Bureau of Labor Statistics", SourceType.OFFICIAL,
                     SourceAuthority.TIER_1_OFFICIAL, notes="CPI, NFP"),
    SourceDescriptor("us_treasury", "US Treasury", SourceType.OFFICIAL,
                     SourceAuthority.TIER_1_OFFICIAL),
    SourceDescriptor("white_house", "White House", SourceType.OFFICIAL,
                     SourceAuthority.TIER_1_OFFICIAL),
    SourceDescriptor("company_ir", "Company investor relations", SourceType.OFFICIAL,
                     SourceAuthority.TIER_1_OFFICIAL),
    SourceDescriptor("reuters", "Reuters", SourceType.NEWS,
                     SourceAuthority.TIER_2_PROFESSIONAL, requires_credentials=True,
                     notes="licensed feed required"),
    SourceDescriptor("bloomberg", "Bloomberg", SourceType.NEWS,
                     SourceAuthority.TIER_2_PROFESSIONAL, requires_credentials=True,
                     notes="licensed feed required"),
    SourceDescriptor("cnbc", "CNBC", SourceType.NEWS,
                     SourceAuthority.TIER_2_PROFESSIONAL),
    SourceDescriptor("x_public_accounts", "X — public accounts", SourceType.SOCIAL,
                     SourceAuthority.TIER_3_VERIFIED_INDIVIDUAL, requires_credentials=True,
                     notes="official API only; no scraping of protected endpoints"),
    SourceDescriptor("aggregator", "Secondary aggregator", SourceType.SOCIAL,
                     SourceAuthority.TIER_4_AGGREGATOR),
    SourceDescriptor("market_tape", "Market tape", SourceType.MARKET,
                     SourceAuthority.TIER_1_OFFICIAL,
                     notes="price and flow events; supplied by the existing collector"),
    SourceDescriptor("mock", "Synthetic fixtures", SourceType.MOCK,
                     SourceAuthority.TIER_4_AGGREGATOR, notes="tests only"),
)


def default_registry() -> SourceRegistry:
    return SourceRegistry(DEFAULT_SOURCES)


def deferred_adapters() -> dict[str, DeferredAdapter]:
    """Every non-mock source, as an adapter that refuses to run."""
    return {
        d.source_id: DeferredAdapter(d)
        for d in DEFAULT_SOURCES
        if d.source_type is not SourceType.MOCK
    }


__all__ = [
    "SourceAdapter",
    "SourceAuthority",
    "SourceType",
    "SourceDescriptor",
    "SourceRegistry",
    "DeferredAdapter",
    "DEFAULT_SOURCES",
    "default_registry",
    "deferred_adapters",
]
