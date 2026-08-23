"""Entity to asset relevance, versioned and unsigned.

The graph answers one question: *if this entity is in the news, which assets are
worth measuring?* It does not answer which way they move, and the type system
below makes the wrong answer unrepresentable — `Edge` refuses a negative weight
and there is nowhere to put a direction.

That refusal is the whole design. A graph that said "Musk -> DOGE, bullish"
would let every downstream study inherit the conclusion as an assumption and
then rediscover it; the study would look like evidence and be a tautology. What
the graph may encode is the mechanical channel — *why* the asset is worth
looking at — and that reason is stored so an auditor can disagree with it.

Weights are coarse on purpose. 0.9 versus 0.85 is false precision on a
hand-built graph; three or four levels is all the resolution the evidence
supports, and pretending otherwise invites fitting the graph rather than the
market.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..errors import RelevanceIsNotDirection
from ..schemas.relevance import ASSET_SET, AssetRelevance
from ..version import GRAPH_VERSION

#: Coarse levels, named so that a reader can see the claim being made.
DIRECT = 1.0        # the asset *is* the entity
PRIMARY = 0.8       # first-order: the entity's actions move this asset directly
SECONDARY = 0.5     # a channel exists and is well understood
CONTEXTUAL = 0.25   # plausible, indirect, worth measuring but easily spurious


@dataclass(frozen=True, slots=True)
class Edge:
    asset: str
    weight: float
    reason: str

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise RelevanceIsNotDirection(
                f"edge to {self.asset} has weight {self.weight}; a negative weight is a "
                "directional claim, and direction is what research is for"
            )
        if self.weight > 1:
            raise ValueError(f"edge to {self.asset} has weight {self.weight} > 1")
        if self.asset not in ASSET_SET:
            raise ValueError(f"unknown asset {self.asset!r}")
        if not self.reason.strip():
            raise ValueError(
                f"edge to {self.asset} has no reason; an unexplained edge cannot be "
                "audited or argued with"
            )


class RelevanceGraph:
    def __init__(self, edges: Mapping[str, tuple[Edge, ...]], version: int = GRAPH_VERSION) -> None:
        self._edges = {k.lower(): v for k, v in edges.items()}
        self.version = version

    def entities(self) -> tuple[str, ...]:
        return tuple(sorted(self._edges))

    def edges_for(self, entity: str) -> tuple[Edge, ...]:
        return self._edges.get(entity.lower(), ())

    def relevance_for(self, entities: Iterable[str]) -> AssetRelevance:
        """Relevance for one event, which may name several entities.

        Combined with `max`, not with a sum — see `AssetRelevance.merged`. An
        event that mentions five minor names must not outweigh one that quotes
        the central bank.
        """
        parts = []
        for entity in entities:
            edges = self.edges_for(entity)
            if not edges:
                continue
            parts.append(
                AssetRelevance(
                    weights={e.asset: e.weight for e in edges},
                    reasons={e.asset: f"{entity}: {e.reason}" for e in edges},
                    graph_version=self.version,
                )
            )
        if not parts:
            return AssetRelevance(graph_version=self.version)
        return AssetRelevance.merged(parts)


def _edges(*items: tuple[str, float, str]) -> tuple[Edge, ...]:
    return tuple(Edge(asset, weight, reason) for asset, weight, reason in items)


#: The starting graph. Every edge names its channel. Where the channel is a
#: narrative rather than a mechanism, the reason says so — a narrative link is
#: still worth measuring and is much more likely to be spurious.
DEFAULT_EDGES: dict[str, tuple[Edge, ...]] = {
    "elon musk": _edges(
        ("TSLA", DIRECT, "chief executive; statements move the equity directly"),
        ("DOGE", PRIMARY, "repeatedly and publicly associated with the asset"),
        ("CRYPTO_BASKET", CONTEXTUAL, "attention spillover from DOGE; narrative channel"),
        ("NDX", CONTEXTUAL, "TSLA is an index constituent"),
    ),
    "nvidia": _edges(
        ("NVDA", DIRECT, "the issuer"),
        ("NDX", PRIMARY, "large index weight"),
        ("SPX", SECONDARY, "index weight, smaller"),
        ("CRYPTO_BASKET", CONTEXTUAL, "shared risk-appetite and AI narrative; not mechanical"),
    ),
    "apple": _edges(
        ("AAPL", DIRECT, "the issuer"),
        ("NDX", PRIMARY, "large index weight"),
        ("SPX", SECONDARY, "index weight"),
    ),
    "tesla": _edges(
        ("TSLA", DIRECT, "the issuer"),
        ("NDX", SECONDARY, "index constituent"),
    ),
    "federal reserve": _edges(
        ("US2Y", PRIMARY, "policy rate expectations are priced here first"),
        ("US10Y", PRIMARY, "term structure responds to the path"),
        ("DXY", PRIMARY, "rate differentials drive the dollar"),
        ("SPX", SECONDARY, "discount rate channel"),
        ("NDX", SECONDARY, "duration-sensitive equity"),
        ("GOLD", SECONDARY, "real-rate channel"),
        ("BTC", CONTEXTUAL, "liquidity and risk-appetite channel; indirect"),
        ("ETH", CONTEXTUAL, "same channel as BTC, weaker"),
    ),
    "donald trump": _edges(
        ("DXY", SECONDARY, "trade and fiscal policy affect the dollar"),
        ("SPX", SECONDARY, "policy risk premium"),
        ("NDX", SECONDARY, "policy risk premium, supply-chain exposed"),
        ("GOLD", CONTEXTUAL, "haven demand under policy uncertainty"),
        ("OIL", CONTEXTUAL, "sanctions and production policy"),
        ("BTC", CONTEXTUAL, "regulation and haven narratives; direction untested"),
        ("US10Y", CONTEXTUAL, "fiscal path"),
    ),
    "sec": _edges(
        ("BTC", SECONDARY, "regulatory perimeter for crypto assets"),
        ("ETH", SECONDARY, "regulatory perimeter"),
        ("CRYPTO_BASKET", SECONDARY, "perimeter applies across the basket"),
        ("SPX", CONTEXTUAL, "enforcement affects listed issuers"),
    ),
    "bureau of labor statistics": _edges(
        ("US2Y", PRIMARY, "employment and inflation prints reprice the front end"),
        ("US10Y", SECONDARY, "term premium"),
        ("DXY", SECONDARY, "rate differentials"),
        ("SPX", SECONDARY, "growth and discount rate"),
        ("BTC", CONTEXTUAL, "liquidity channel, indirect"),
    ),
    "opec": _edges(
        ("OIL", DIRECT, "supply policy"),
        ("DXY", CONTEXTUAL, "terms of trade"),
        ("SPX", CONTEXTUAL, "input costs"),
    ),
}


def default_graph() -> RelevanceGraph:
    return RelevanceGraph(DEFAULT_EDGES, GRAPH_VERSION)


__all__ = [
    "RelevanceGraph",
    "Edge",
    "default_graph",
    "DEFAULT_EDGES",
    "DIRECT",
    "PRIMARY",
    "SECONDARY",
    "CONTEXTUAL",
]
