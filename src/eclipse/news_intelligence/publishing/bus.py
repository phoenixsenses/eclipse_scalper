"""Event bus contracts. No transport, and one hard rule.

The subjects below mirror the namespaces the platform already reserves, so this
layer joins the existing bus rather than inventing a parallel one. NATS is not
required to be running: `InMemoryPublisher` satisfies the same protocol, which
is what makes the pipeline testable today and swappable later.

**The rule the platform already settled: publish candidates, never outcomes.**
The bus is a fan-out. Anything on it reaches subscribers that have no idea which
arm is sealed, and a sealed arm's realised result crossing the bus is a leak
that no downstream discipline can undo. So the publisher inspects every payload
and refuses the ones carrying outcome-shaped keys — a refusal, not a filter,
because a filtered publish leaves the caller believing it published what it
meant to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, runtime_checkable

from ..errors import OutcomeInFeatureSpace
from ..schemas.reaction import OUTCOME_FIELDS
from ..version import SCHEMA_VERSION

#: Subjects this layer owns. Every one is additive to the existing namespaces.
SUBJECTS = {
    "news_raw": "eclipse.news.raw",
    "news_normalized": "eclipse.news.normalized",
    "news_high_impact": "eclipse.news.high_impact",
    "social_raw": "eclipse.social.raw",
    "social_normalized": "eclipse.social.normalized",
    "macro_scheduled": "eclipse.macro.scheduled",
    "macro_released": "eclipse.macro.released",
    "cross_asset_context": "eclipse.market.cross_asset_context",
    "research_ready": "eclipse.research.news_event_ready",
}

#: Subjects on which an outcome would be a leak. Every subject this layer
#: publishes, in other words — listed explicitly so that adding one forces a
#: decision rather than inheriting a default.
OUTCOME_FORBIDDEN = frozenset(SUBJECTS.values())


@dataclass(frozen=True, slots=True)
class Envelope:
    subject: str
    payload: Mapping[str, Any]
    published_at: datetime
    producer: str
    schema_version: int = SCHEMA_VERSION
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        object.__setattr__(self, "published_at", self.published_at.astimezone(timezone.utc))


@runtime_checkable
class Publisher(Protocol):
    def publish(self, envelope: Envelope) -> None:
        ...


def assert_no_outcome(subject: str, payload: Mapping[str, Any]) -> None:
    """Refuse a payload carrying anything only knowable after the fact.

    Checks nested mappings too. The realistic mistake is not a top-level
    `return_bps`; it is a tidy `{"event": {...}, "label": {...}}` envelope built
    by someone joining two objects for convenience.
    """
    if subject not in OUTCOME_FORBIDDEN:
        return

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in OUTCOME_FIELDS:
                    raise OutcomeInFeatureSpace(
                        f"{path}{key!r} is an outcome and may not cross {subject}. "
                        "The bus publishes candidates; results stay in the ledger"
                    )
                walk(value, f"{path}{key}.")
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, path)

    walk(payload, "")


class InMemoryPublisher:
    """Records what would have been published. Enforces the same rule as the real one."""

    def __init__(self) -> None:
        self.published: list[Envelope] = []

    def publish(self, envelope: Envelope) -> None:
        assert_no_outcome(envelope.subject, envelope.payload)
        self.published.append(envelope)

    def subjects(self) -> tuple[str, ...]:
        return tuple(e.subject for e in self.published)

    def payloads_for(self, subject: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(e.payload for e in self.published if e.subject == subject)


__all__ = [
    "SUBJECTS",
    "Envelope",
    "Publisher",
    "InMemoryPublisher",
    "assert_no_outcome",
    "OUTCOME_FORBIDDEN",
]
