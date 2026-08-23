"""Asking for a measurement, and reading one honestly.

No market data is touched here. This module builds the *request* — which asset,
which horizons, anchored to which instant — so the whole pipeline can be built
and tested with no price store, and so that filling the request in later is a
separate, schedulable, interruptible job.

The second half is the part that matters more. `classify_causality` exists
because the obvious analysis is wrong in a specific way: if a price moved before
the news arrived and continued after it, a naive post-event return will report
the news as predictive when the news was the *consequence*. Sentiment feeds are
especially prone to this — commentary follows price all day. So the pre-event
window is not decoration; it is the control, and the classifier refuses to call
anything predictive when the move was already underway.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..schemas.normalized import NormalizedEvent
from ..schemas.reaction import ALL_HORIZONS, MarketReaction, ReactionRequest


def build_request(
    event: NormalizedEvent,
    threshold: float = 0.2,
    horizons: tuple[int, ...] = ALL_HORIZONS,
) -> ReactionRequest:
    """Measure the assets the graph says are worth measuring, and only those.

    Anchored to `decision_time` — when the system could know — not to
    `published_at`. Anchoring to publication would measure a window the system
    could not have traded and would flatter every result by the delivery lag.
    """
    assets = event.asset_relevance.relevant(threshold)
    return ReactionRequest(
        event_id=event.event_id,
        decision_time=event.decision_time,
        assets=assets,
        horizons=horizons,
    )


class Causality(str, Enum):
    NEWS_LEADS_PRICE = "NEWS_LEADS_PRICE"
    PRICE_LEADS_NEWS = "PRICE_LEADS_NEWS"
    SIMULTANEOUS = "SIMULTANEOUS"
    NO_RELATIONSHIP = "NO_RELATIONSHIP"
    UNDETERMINED = "UNDETERMINED"


@dataclass(frozen=True, slots=True)
class CausalityVerdict:
    direction: Causality
    pre_move_bps: float | None
    post_move_bps: float | None
    reason: str


def classify_causality(
    reaction: MarketReaction,
    asset: str,
    pre_horizon: int = -15,
    post_horizon: int = 15,
    move_threshold_bps: float = 10.0,
) -> CausalityVerdict:
    """Read one asset's reaction without assuming the news caused it.

    Four outcomes and an honest fifth. The rules are deliberately conservative
    in one direction: a move that had already happened before the item arrived
    is never reported as the news leading, however large the post-event return.
    Being wrong about that is how a feed of commentary gets mistaken for a feed
    of information.
    """
    pre = reaction.get(asset, pre_horizon)
    post = reaction.get(asset, post_horizon)

    if pre is None or post is None or not pre.complete or not post.complete:
        return CausalityVerdict(
            Causality.UNDETERMINED,
            pre.return_bps if pre else None,
            post.return_bps if post else None,
            "a required window is missing or incomplete; an absent measurement is not a zero",
        )

    pre_move = pre.return_bps or 0.0
    post_move = post.return_bps or 0.0
    pre_big = abs(pre_move) >= move_threshold_bps
    post_big = abs(post_move) >= move_threshold_bps

    if not pre_big and not post_big:
        return CausalityVerdict(Causality.NO_RELATIONSHIP, pre_move, post_move,
                                "neither window moved beyond the threshold")
    if pre_big and not post_big:
        return CausalityVerdict(Causality.PRICE_LEADS_NEWS, pre_move, post_move,
                                "the move preceded the item and did not continue after it")
    if not pre_big and post_big:
        return CausalityVerdict(Causality.NEWS_LEADS_PRICE, pre_move, post_move,
                                "the move begins after the item was knowable")
    same_sign = (pre_move > 0) == (post_move > 0)
    if same_sign:
        return CausalityVerdict(
            Causality.PRICE_LEADS_NEWS, pre_move, post_move,
            "the move was already underway in the same direction; continuation is not evidence "
            "that the item caused it",
        )
    return CausalityVerdict(Causality.SIMULTANEOUS, pre_move, post_move,
                            "both windows moved, in opposite directions; direction is not separable here")


__all__ = ["build_request", "classify_causality", "Causality", "CausalityVerdict"]
