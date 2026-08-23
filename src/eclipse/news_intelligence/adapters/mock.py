"""Synthetic fixtures — the only adapter that runs today.

These five scenarios are not filler. Each one exists to make a specific
invariant fail loudly if it ever stops holding:

  1. a tariff statement            entity, type, relevance across several assets
  2. a rate decision               scheduled event, official source, rate complex
  3. a DOGE post                   person -> asset via a narrative channel
  4. an earnings surprise          company event, index spillover
  5-7. three reprints of (1)       the same real event from three outlets

The reprints are the important ones. They are what turns "we handle duplicates"
from a claim into a test: after them the cluster count must not have moved, the
novelty of the copies must be below the original's, and amplification must have
risen. All three together, because any one of them alone can be satisfied by a
system that is quietly wrong.

No network, no credentials, no files. Deterministic timestamps so two runs
produce identical ids.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from ..schemas.raw import RawEvent
from .base import SourceAuthority, SourceDescriptor, SourceType

BASE = datetime(2026, 8, 23, 13, 44, 0, tzinfo=timezone.utc)


def _raw(
    idx: int,
    source_id: str,
    title: str,
    text: str,
    published_offset_s: int,
    seen_lag_s: int,
    author: str | None = None,
    source_ref: str | None = None,
) -> RawEvent:
    published = BASE + timedelta(seconds=published_offset_s)
    first_seen = published + timedelta(seconds=seen_lag_s)
    return RawEvent(
        raw_event_id=f"raw_{idx:03d}",
        source_id=source_id,
        source_type="MOCK",
        source_authority="MOCK",
        source_ref=source_ref or f"mock://{source_id}/{idx}",
        published_at=published,
        first_seen_at=first_seen,
        received_at=first_seen + timedelta(milliseconds=120),
        raw_title=title,
        raw_text=text,
        raw_payload={"idx": idx, "source": source_id},
        author=author,
    )


def fixture_events() -> tuple[RawEvent, ...]:
    return (
        _raw(
            1,
            "white_house",
            "President announces new tariffs on imported semiconductors",
            "The White House said it will impose tariffs of an unspecified level on imported "
            "semiconductors, citing national security. Officials said further duties on other "
            "categories remain under consideration.",
            published_offset_s=0,
            seen_lag_s=6,
            author="Donald Trump",
        ),
        _raw(
            2,
            "federal_reserve",
            "FOMC holds the target range steady",
            "The Federal Reserve said the committee will hold rates, noting that inflation has "
            "moved closer to the objective while the labour market remains solid.",
            published_offset_s=900,
            seen_lag_s=3,
        ),
        _raw(
            3,
            "x_public_accounts",
            "much wow",
            "Elon Musk posted on X about DOGE again this afternoon.",
            published_offset_s=1500,
            seen_lag_s=11,
            author="Elon Musk",
        ),
        _raw(
            4,
            "company_ir",
            "NVIDIA reports quarterly results above guidance",
            "NVIDIA reported revenue above its prior guidance, and said demand for its data "
            "centre products beat internal forecasts.",
            published_offset_s=2100,
            seen_lag_s=2,
        ),
        _raw(
            5,
            "reuters",
            "US to impose tariffs on imported semiconductors, White House says",
            "The United States will impose tariffs on imported semiconductors, the White House "
            "said, citing national security. Further duties on other categories are under "
            "consideration, officials said.",
            published_offset_s=240,
            seen_lag_s=4,
            author="Donald Trump",
        ),
        _raw(
            6,
            "cnbc",
            "White House announces semiconductor tariffs",
            "The White House announced tariffs on imported semiconductors on Friday, citing "
            "national security concerns. Officials said duties on further categories remain "
            "under consideration.",
            published_offset_s=420,
            seen_lag_s=5,
            author="Donald Trump",
        ),
        _raw(
            7,
            "aggregator",
            "Trump: tariffs on imported semiconductors",
            "Roundup: the White House will impose tariffs on imported semiconductors, citing "
            "national security; more duties under consideration.",
            published_offset_s=600,
            seen_lag_s=9,
            author="Donald Trump",
        ),
    )


class MockAdapter:
    descriptor = SourceDescriptor(
        source_id="mock",
        display_name="Synthetic fixtures",
        source_type=SourceType.MOCK,
        authority=SourceAuthority.TIER_4_AGGREGATOR,
        notes="tests only",
    )

    def __init__(self, events: tuple[RawEvent, ...] | None = None) -> None:
        self._events = events if events is not None else fixture_events()

    def poll(self, since: datetime | None = None) -> Iterable[RawEvent]:
        for event in sorted(self._events, key=lambda e: e.first_seen_at):
            if since is None or event.first_seen_at > since:
                yield event


__all__ = ["MockAdapter", "fixture_events", "BASE"]
