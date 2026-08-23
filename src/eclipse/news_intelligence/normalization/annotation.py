"""What a model is allowed to write, and what it may never touch.

An LLM is useful here: classification, entity extraction, topic naming,
summarising, suggesting which assets are worth measuring. It is not allowed
anywhere near the parts of a row that must be reproducible — timestamps, source
identity, payload digests, ids — and it is never allowed to emit a trading
decision.

The rule is enforced rather than documented. `apply_annotation` refuses to write
a protected field, so a prompt change cannot quietly turn provenance into model
output. That is a real failure mode: the fields most useful to a model are
exactly the ones that make a row auditable, and a model asked to "fill in the
event" will happily invent a publication time.

Every annotation carries its model id, prompt version, confidence and processing
timestamp. Two runs of the same pipeline over the same raw item must be
distinguishable when the model changed underneath them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

from ..errors import DeterministicFieldOverwrite
from ..schemas.normalized import Judgement, NormalizedEvent

#: Fields that come from the wire or the clock. A model may read them; it may
#: never write them.
PROTECTED_FIELDS = frozenset(
    {
        "event_id",
        "raw_event_id",
        "published_at",
        "first_seen_at",
        "received_at",
        "source_authority",
        "schema_version",
        "taxonomy_version",
        "news_cluster_id",
    }
)

#: Fields a model may propose. Anything outside both sets is rejected as unknown
#: rather than silently ignored, so a renamed field fails loudly.
ANNOTATABLE_FIELDS = frozenset(
    {
        "entity",
        "secondary_entities",
        "event_type",
        "topic",
        "subtopic",
        "sentiment",
        "surprise",
        "credibility",
        "expected_vs_unexpected",
        "country_relevance",
        "sector_relevance",
        "asset_relevance",
    }
)

#: Things a model must never produce in this system at all, under any field name.
FORBIDDEN_OUTPUTS = frozenset({"buy", "sell", "long", "short", "position_size", "order"})


@dataclass(frozen=True, slots=True)
class ModelAnnotation:
    model_id: str
    prompt_version: str
    produced_at: datetime
    values: Mapping[str, Any]
    confidences: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.produced_at.tzinfo is None:
            raise ValueError("produced_at must be timezone-aware")
        object.__setattr__(self, "produced_at", self.produced_at.astimezone(timezone.utc))
        for key in self.values:
            if key in FORBIDDEN_OUTPUTS:
                raise DeterministicFieldOverwrite(
                    f"model tried to emit {key!r}. This layer produces research features; "
                    "a trading decision is not one of them"
                )
            if key in PROTECTED_FIELDS:
                raise DeterministicFieldOverwrite(
                    f"{key!r} is deterministic — it comes from the source and the clock. "
                    "A model may annotate beside it, never over it"
                )
            if key not in ANNOTATABLE_FIELDS:
                raise DeterministicFieldOverwrite(
                    f"{key!r} is not an annotatable field. Add it to ANNOTATABLE_FIELDS "
                    "deliberately, or the schema has drifted"
                )
            if key not in self.confidences:
                raise ValueError(
                    f"{key!r} was annotated without a confidence; a label whose "
                    "uncertainty is unknown cannot be filtered later"
                )


def apply_annotation(event: NormalizedEvent, annotation: ModelAnnotation) -> NormalizedEvent:
    """Return a new event with the annotation applied and recorded.

    Never mutates: the input event stays exactly as it was, so the pre- and
    post-annotation rows can both be kept when a model is being evaluated.
    """
    judgements = dict(event.judgements)
    for field_name, value in annotation.values.items():
        judgements[field_name] = Judgement(
            value=value,
            confidence=annotation.confidences[field_name],
            model_id=annotation.model_id,
            prompt_version=annotation.prompt_version,
            produced_at=annotation.produced_at,
        )
    updated = replace(
        event,
        **dict(annotation.values),
        judgements=judgements,
        model_version=annotation.model_id,
    )
    return updated


__all__ = [
    "ModelAnnotation",
    "apply_annotation",
    "PROTECTED_FIELDS",
    "ANNOTATABLE_FIELDS",
    "FORBIDDEN_OUTPUTS",
]
