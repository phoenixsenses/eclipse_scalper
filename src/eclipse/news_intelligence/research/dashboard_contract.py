"""The data contract for the private Master Center panel.

Private mesh only. The public site may describe that this layer exists; it may
not carry its rows, and this module is the boundary that makes the distinction
enforceable rather than remembered.

Two things are deliberately absent from every payload here:

  **No realised returns.** The panel shows what was known and what is pending.
  A reaction column would put outcome data on an operator screen, and an
  operator screen is one screenshot away from being a public one.

  **No arm-level aggregates.** `events_matched_to_eder` is a count of matches,
  never a summary of how those matches performed. The counter exists so an
  operator can see the research pipeline filling up, not how it is doing.

Everything below is a plain dict so the transport can be anything later.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from ..pipeline import ProcessedEvent
from ..research.api import ResearchCounters

#: Columns of the recent-events panel, in display order.
RECENT_EVENT_COLUMNS = (
    "time",
    "source",
    "entity",
    "type",
    "topic",
    "novelty",
    "surprise",
    "relevance",
    "status",
)

#: Assets shown in the cross-asset context strip. Context, not performance:
#: these are the market's state, not any arm's result.
CROSS_ASSET_ROW = ("BTC", "ETH", "NDX", "DXY", "US10Y", "VIX")

FORBIDDEN_KEYS = frozenset(
    {"return_bps", "pnl", "win_rate", "net_bps", "outcome", "label", "reaction"}
)


def _assert_clean(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in FORBIDDEN_KEYS:
                    raise ValueError(
                        f"{key!r} would put outcome data on the operator panel; the panel "
                        "shows what is known and what is pending, never how it did"
                    )
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)
    return payload


def recent_events_panel(processed: Sequence[ProcessedEvent], limit: int = 20) -> dict[str, Any]:
    rows = []
    for item in sorted(processed, key=lambda p: p.event.first_seen_at, reverse=True)[:limit]:
        event = item.event
        top_assets = event.asset_relevance.relevant(0.5)
        rows.append(
            {
                "time": event.first_seen_at.strftime("%H:%M"),
                "source": event.source_authority,
                "entity": event.entity or "—",
                "type": event.event_type.value,
                "topic": event.topic or "—",
                "novelty": None if event.novelty is None else round(event.novelty, 2),
                "surprise": event.surprise,
                "relevance": list(top_assets[:3]),
                "status": (
                    "INDEPENDENT" if item.is_independent_observation else "REPEAT"
                ),
                "cluster": event.news_cluster_id,
            }
        )
    return _assert_clean({"columns": list(RECENT_EVENT_COLUMNS), "rows": rows})


def cross_asset_panel(
    state: Mapping[str, str], as_of: datetime, complete: Iterable[str] = ()
) -> dict[str, Any]:
    """Current market context, with missing assets shown as missing.

    An asset with no reading renders as `—` and is listed in `incomplete`.
    Rendering it as a zero or as a stale carry-forward would put a fabricated
    reading on an operator's screen — the same fault the dashboard review caught
    once already, when freshness was displayed as green while the feed was dead.
    """
    complete_set = set(complete)
    rows = [
        {
            "asset": asset,
            "state": state.get(asset, "—") if asset in complete_set else "—",
            "known": asset in complete_set,
        }
        for asset in CROSS_ASSET_ROW
    ]
    return _assert_clean(
        {
            "as_of": as_of.isoformat(),
            "rows": rows,
            "incomplete": [a for a in CROSS_ASSET_ROW if a not in complete_set],
        }
    )


def research_counters_panel(
    counters: ResearchCounters, near_threshold: int | None = None
) -> dict[str, Any]:
    payload = counters.as_dict()
    if near_threshold is not None:
        payload["groupings_decided_near_the_threshold"] = near_threshold
    payload["note"] = (
        "raw items and independent clusters are different sample sizes; "
        "the ratio between them is published so neither can be quoted as the other"
    )
    return _assert_clean(payload)


def dashboard_payload(
    processed: Sequence[ProcessedEvent],
    counters: ResearchCounters,
    market_state: Mapping[str, str],
    as_of: datetime,
    complete_assets: Iterable[str] = (),
    near_threshold: int | None = None,
) -> dict[str, Any]:
    return {
        "panel": "NEWS_INTELLIGENCE",
        "visibility": "PRIVATE_MESH_ONLY",
        "generated_at": as_of.isoformat(),
        "recent_events": recent_events_panel(processed),
        "cross_asset_context": cross_asset_panel(market_state, as_of, complete_assets),
        "research_counters": research_counters_panel(counters, near_threshold),
    }


__all__ = [
    "dashboard_payload",
    "recent_events_panel",
    "cross_asset_panel",
    "research_counters_panel",
    "RECENT_EVENT_COLUMNS",
    "CROSS_ASSET_ROW",
]
