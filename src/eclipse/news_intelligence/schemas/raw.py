"""The raw item, exactly as it arrived.

This is the only object in the package that is allowed to be messy, and the only
one that is never rewritten. Everything downstream is derived and can be
recomputed when a classifier improves; the raw row is the thing that makes that
recomputation possible. A pipeline that normalises in place can never answer
"what did we actually receive?" after the first model change.

Three timestamps, not one, and they are routinely different:

  published_at   when the world says it happened
  first_seen_at  when this system could first have known
  received_at    when this process took delivery

`first_seen_at` is the one research anchors to. `published_at` minus
`first_seen_at` is itself a measurement — a source that is consistently late is
a source whose "news" the market has already traded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from ..version import SCHEMA_VERSION


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware; naive datetimes are ambiguous")
    return value.astimezone(timezone.utc)


def payload_hash(payload: Mapping[str, Any]) -> str:
    """Stable digest of the original payload.

    Sorted keys and a compact separator so the same payload always hashes the
    same way regardless of how the adapter happened to build the dict. Used to
    recognise a re-delivery of an item we already hold, which is a different
    thing from a *revision* of it.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RawEvent:
    raw_event_id: str
    source_id: str
    source_type: str
    source_authority: str
    source_ref: str

    published_at: datetime
    first_seen_at: datetime
    received_at: datetime

    raw_title: str
    raw_text: str
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    author: str | None = None
    language: str = "en"
    revision: int = 0
    processed_at: datetime | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "published_at", _utc(self.published_at, "published_at"))
        object.__setattr__(self, "first_seen_at", _utc(self.first_seen_at, "first_seen_at"))
        object.__setattr__(self, "received_at", _utc(self.received_at, "received_at"))
        if self.processed_at is not None:
            object.__setattr__(self, "processed_at", _utc(self.processed_at, "processed_at"))
        if self.first_seen_at < self.published_at:
            # Seeing something before it was published means one of the two
            # clocks is wrong. Guessing which would corrupt every downstream
            # latency measurement, so the item is refused instead.
            raise ValueError(
                f"first_seen_at {self.first_seen_at.isoformat()} precedes published_at "
                f"{self.published_at.isoformat()} — clock disagreement, not a scoop"
            )

    @property
    def payload_digest(self) -> str:
        return payload_hash(self.raw_payload)

    @property
    def publication_lag_seconds(self) -> float:
        """How long the world had this before we did. A feature in its own right."""
        return (self.first_seen_at - self.published_at).total_seconds()


__all__ = ["RawEvent", "payload_hash"]
