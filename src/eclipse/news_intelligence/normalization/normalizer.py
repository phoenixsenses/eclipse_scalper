"""Raw item to structured event.

Deterministic by default. Everything this module produces can be recomputed from
the raw row with the same code and the same versions, which is what makes a
later model comparison meaningful: without a reproducible baseline, "the model
improved" is unfalsifiable.

Credibility here is a property of the *source*, mapped from its authority tier,
and nothing else. It is emphatically not "how believable this claim sounds" —
that judgement belongs to a model, gets a confidence, and sits beside this one
rather than replacing it.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from ..adapters.base import SourceAuthority, SourceRegistry
from ..relevance.graph import RelevanceGraph
from ..schemas.normalized import Judgement, NormalizedEvent, Sentiment
from ..schemas.raw import RawEvent
from ..version import RULE_CLASSIFIER_ID, TAXONOMY_VERSION
from .extraction import Extractor, RuleExtractor

#: Authority tier to a credibility number, so that downstream code can do
#: arithmetic without inventing its own mapping. Coarse, and deliberately not
#: 1.0 even for an official source: primary sources are wrong sometimes, and a
#: model that cannot represent that will never notice when it happens.
CREDIBILITY_BY_AUTHORITY = {
    SourceAuthority.TIER_1_OFFICIAL: 0.95,
    SourceAuthority.TIER_2_PROFESSIONAL: 0.8,
    SourceAuthority.TIER_3_VERIFIED_INDIVIDUAL: 0.6,
    SourceAuthority.TIER_4_AGGREGATOR: 0.35,
    SourceAuthority.UNKNOWN: 0.1,
}

_POSITIVE = ("beat", "beats", "surge", "record", "approve", "approved", "boost", "strong", "cut rates")
_NEGATIVE = ("miss", "misses", "fall", "plunge", "ban", "halt", "reject", "weak", "tariff", "sanction")


def _event_id(raw: RawEvent) -> str:
    """Stable id derived from provenance, not from a counter or a clock.

    Two runs over the same raw item must produce the same event id, or every
    downstream join becomes run-dependent and nothing can be recomputed.
    """
    material = f"{raw.source_id}|{raw.source_ref}|{raw.revision}|{raw.payload_digest}"
    return "evt_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _naive_sentiment(blob: str) -> Sentiment:
    """A lexicon floor, not a sentiment model.

    Present so the field is populated reproducibly and so tests can assert that
    tone is *not* what drives anything downstream. A real model annotates over
    it and records that it did.
    """
    text = blob.lower()
    hits_pos = sum(text.count(w) for w in _POSITIVE)
    hits_neg = sum(text.count(w) for w in _NEGATIVE)
    total = hits_pos + hits_neg
    if total == 0:
        return Sentiment(polarity=0.0, strength=0.0)
    polarity = (hits_pos - hits_neg) / total
    strength = min(1.0, total / 4.0)
    return Sentiment(polarity=polarity, strength=strength)


class Normalizer:
    def __init__(
        self,
        registry: SourceRegistry,
        graph: RelevanceGraph,
        extractor: Extractor | None = None,
    ) -> None:
        self.registry = registry
        self.graph = graph
        self.extractor = extractor or RuleExtractor()

    def normalize(self, raw: RawEvent) -> NormalizedEvent:
        descriptor = self.registry.get(raw.source_id)
        authority = descriptor.authority
        extraction = self.extractor.extract(raw.raw_title, raw.raw_text, raw.author)

        entities = [extraction.entity, *extraction.secondary_entities]
        relevance = self.graph.relevance_for([e for e in entities if e])

        judgements = {
            "event_type": Judgement(
                value=extraction.event_type,
                confidence=extraction.confidence,
                model_id=extraction.model_id,
                produced_at=datetime.now(timezone.utc),
            ),
            "entity": Judgement(
                value=extraction.entity,
                confidence=extraction.confidence if extraction.entity else 0.0,
                model_id=extraction.model_id,
            ),
            "asset_relevance": Judgement(
                value=relevance.relevant(),
                confidence=1.0 if relevance.weights else 0.0,
                model_id=f"relevance-graph@{self.graph.version}",
            ),
        }

        return NormalizedEvent(
            event_id=_event_id(raw),
            raw_event_id=raw.raw_event_id,
            published_at=raw.published_at,
            first_seen_at=raw.first_seen_at,
            received_at=raw.received_at,
            entity=extraction.entity,
            secondary_entities=extraction.secondary_entities,
            event_type=extraction.event_type,
            topic=extraction.topic,
            subtopic=extraction.subtopic,
            sentiment=_naive_sentiment(f"{raw.raw_title} {raw.raw_text}"),
            credibility=CREDIBILITY_BY_AUTHORITY[authority],
            source_authority=authority.value,
            asset_relevance=relevance,
            judgements=judgements,
            classifier_version=RULE_CLASSIFIER_ID,
            model_version=RULE_CLASSIFIER_ID,
            taxonomy_version=TAXONOMY_VERSION,
        )


__all__ = ["Normalizer", "CREDIBILITY_BY_AUTHORITY"]
