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
from ..schemas.normalized import Judgement, NormalizedEvent, Sentiment
from ..schemas.relevance import AssetRelevance
from ..taxonomy.events import EventType

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

#: The type each annotatable field must arrive as. `dataclasses.replace` does no
#: type checking, so without this a model could write `{"polarity": 9.0}` where a
#: `Sentiment` belongs and skip its bounds check — and, worse, write
#: `{"BTC": -0.9}` into `asset_relevance`, walking straight past the refusal that
#: keeps direction out of the relevance graph. The headline invariant had a side
#: door, and this is the door.
ANNOTATION_TYPES: dict[str, type | tuple[type, ...]] = {
    "entity": str,
    "secondary_entities": tuple,
    "event_type": EventType,
    "topic": str,
    "subtopic": str,
    "sentiment": Sentiment,
    "surprise": (int, float),
    "credibility": (int, float),
    "expected_vs_unexpected": str,
    "country_relevance": tuple,
    "sector_relevance": tuple,
    "asset_relevance": AssetRelevance,
}

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
            expected = ANNOTATION_TYPES[key]
            if not isinstance(self.values[key], expected):
                names = (
                    expected.__name__
                    if isinstance(expected, type)
                    else " or ".join(t.__name__ for t in expected)
                )
                raise DeterministicFieldOverwrite(
                    f"{key!r} must be annotated as {names}, not "
                    f"{type(self.values[key]).__name__}. Passing a raw mapping would "
                    "reach the field without running that type's own validation — "
                    "which is where the bounds and the refusal of a signed relevance "
                    "live."
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
    "ANNOTATION_TYPES",
]
