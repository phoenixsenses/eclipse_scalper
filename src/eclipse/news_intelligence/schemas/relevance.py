"""Which assets an event could plausibly touch — and emphatically not how.

`AssetRelevance` is a magnitude in [0, 1] per asset. There is no sign, and the
type refuses one. The reason is not stylistic:

  If the graph says "Trump statement -> BTC negative", then every study built on
  the graph inherits that claim as an assumption, and the study can only ever
  confirm it. The direction is the question. Encoding it in the input answers it
  by fiat and produces a result that looks like evidence.

Relevance says: *this asset is one of the places to look*. Nothing more. A high
relevance with a measured reaction of zero is a perfectly good finding, and one
this package must be able to express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ..errors import RelevanceIsNotDirection

#: The asset universe research may ask about. Kept explicit rather than free
#: text so a typo becomes an error instead of a silently empty column.
ASSETS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "DOGE",
    "CRYPTO_BASKET",
    "SPX",
    "NDX",
    "AAPL",
    "NVDA",
    "TSLA",
    "DXY",
    "US2Y",
    "US10Y",
    "VIX",
    "GOLD",
    "OIL",
)

ASSET_SET = frozenset(ASSETS)


@dataclass(frozen=True, slots=True)
class AssetRelevance:
    """Per-asset relevance magnitudes plus the reason each one is there."""

    weights: Mapping[str, float] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)
    graph_version: int = 0

    def __post_init__(self) -> None:
        for asset, weight in self.weights.items():
            if asset not in ASSET_SET:
                raise ValueError(f"unknown asset {asset!r}; extend ASSETS deliberately")
            if weight < 0:
                raise RelevanceIsNotDirection(
                    f"{asset} weight {weight} is negative. Relevance is a magnitude; "
                    "a sign here would encode the direction that research exists to test"
                )
            if weight > 1:
                raise ValueError(f"{asset} weight {weight} exceeds 1")

    def relevant(self, threshold: float = 0.0) -> tuple[str, ...]:
        """Assets at or above a threshold, in descending weight then alphabetical order.

        Inclusive at the boundary, which is the reading of "threshold" a caller
        expects and the one the graph's own levels assume: the levels are named
        constants, so asking for `SECONDARY` and being handed everything
        *strictly* above it silently drops the entire level you asked for. That
        is not hypothetical — it emptied the relevance column for every tariff
        row in the first demo run.

        Ordering is deterministic so two runs produce byte-identical rows; ties
        break alphabetically rather than by dict insertion, which would leak the
        order the graph happened to be walked in.
        """
        above = [a for a, w in self.weights.items() if w >= threshold]
        return tuple(sorted(above, key=lambda a: (-self.weights[a], a)))

    def weight(self, asset: str) -> float:
        return float(self.weights.get(asset, 0.0))

    def explain(self, asset: str) -> str:
        return self.reasons.get(asset, "no recorded reason")

    @staticmethod
    def merged(parts: Iterable["AssetRelevance"]) -> "AssetRelevance":
        """Combine relevance from several entities in one event.

        Takes the maximum per asset rather than the sum. Two entities that each
        point weakly at BTC do not make a strong claim about BTC; summing would
        let an event mentioning many minor names outrank a direct statement.
        """
        weights: dict[str, float] = {}
        reasons: dict[str, str] = {}
        version = 0
        for part in parts:
            version = max(version, part.graph_version)
            for asset, weight in part.weights.items():
                if weight > weights.get(asset, -1.0):
                    weights[asset] = weight
                    reasons[asset] = part.reasons.get(asset, "")
        return AssetRelevance(weights=weights, reasons=reasons, graph_version=version)


__all__ = ["AssetRelevance", "ASSETS", "ASSET_SET"]
