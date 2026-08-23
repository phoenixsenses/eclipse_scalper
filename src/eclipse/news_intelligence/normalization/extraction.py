"""Entity and topic extraction.

Two implementations behind one protocol, because they fail differently and the
difference matters. The rule extractor is exhaustive about the entities it knows
and blind to everything else; a model extractor is the reverse. A pipeline that
uses only the second cannot be replayed, and one that uses only the first cannot
grow — so the contract lets them be composed, with the rule result taking
precedence on anything it is certain about.

The rule extractor is not a stand-in for the model. It is the reproducible floor
that makes it possible to notice when the model starts drifting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from ..taxonomy.events import EventType
from ..version import RULE_CLASSIFIER_ID


@dataclass(frozen=True, slots=True)
class Extraction:
    entity: str
    secondary_entities: tuple[str, ...]
    event_type: EventType
    topic: str
    subtopic: str
    confidence: float
    model_id: str
    matched_on: tuple[str, ...] = ()


@runtime_checkable
class Extractor(Protocol):
    model_id: str

    def extract(self, title: str, text: str, author: str | None = None) -> Extraction:
        ...


#: Canonical entity names and the surface forms that map to them. Kept as data
#: rather than as a model so that "why did this become Donald Trump?" has an
#: answer a person can read.
ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "donald trump": ("trump", "president trump", "donald trump", "white house"),
    "elon musk": ("elon", "musk", "elon musk"),
    "federal reserve": ("fed", "federal reserve", "fomc", "powell", "the committee"),
    "nvidia": ("nvidia", "nvda"),
    "apple": ("apple", "aapl"),
    "tesla": ("tesla", "tsla"),
    "sec": ("sec", "securities and exchange commission"),
    "bureau of labor statistics": ("bls", "bureau of labor statistics", "cpi", "nonfarm", "payrolls"),
    "opec": ("opec", "opec+"),
}

#: Ordered: the first rule that matches wins, so the more specific patterns come
#: first. Order is part of the definition, not an implementation detail.
TYPE_RULES: tuple[tuple[EventType, str, tuple[str, ...]], ...] = (
    (EventType.TARIFF, "tariffs", ("tariff", "tariffs", "import duty", "duties on")),
    (EventType.SANCTIONS, "sanctions", ("sanction", "sanctions", "export ban")),
    (EventType.TRADE_POLICY, "trade", ("trade deal", "trade policy", "trade war", "quota")),
    (EventType.RATE_POLICY, "rates", ("rate decision", "basis point", "cut rates", "raise rates", "hold rates")),
    (EventType.CENTRAL_BANK, "central bank", ("fomc", "federal reserve", "central bank", "monetary policy")),
    (EventType.INFLATION, "inflation", ("cpi", "inflation", "pce", "consumer price")),
    (EventType.EMPLOYMENT, "employment", ("nonfarm", "payrolls", "unemployment", "jobless")),
    (EventType.CRYPTO_REGULATION, "crypto regulation", ("crypto regulation", "digital asset rule", "bitcoin etf", "crypto framework")),
    (EventType.FINANCIAL_REGULATION, "regulation", ("regulation", "rulemaking", "enforcement action")),
    (EventType.COMPANY_EARNINGS, "earnings", ("earnings", "quarterly results", "reported revenue", "eps")),
    (EventType.COMPANY_GUIDANCE, "guidance", ("guidance", "outlook for the quarter", "forecasts revenue")),
    (EventType.SEC_FILING, "filing", ("8-k", "10-q", "10-k", "s-1", "13f")),
    (EventType.MERGER_ACQUISITION, "m&a", ("acquire", "acquisition", "merger", "takeover")),
    (EventType.PRODUCT, "product", ("launches", "unveils", "product line")),
    (EventType.LEGAL, "legal", ("lawsuit", "court", "settlement", "indictment")),
    (EventType.SECURITY_INCIDENT, "security", ("hack", "exploit", "breach", "stolen funds")),
    (EventType.EXCHANGE_INCIDENT, "exchange", ("exchange halt", "withdrawals paused", "outage")),
    (EventType.COMMODITY_SHOCK, "commodity", ("output cut", "production cut", "supply disruption")),
    (EventType.GEOPOLITICAL, "geopolitics", ("strike", "conflict", "ceasefire", "military")),
    (EventType.SOCIAL_POST, "social", ("posted on x", "tweeted")),
)


class RuleExtractor:
    """Deterministic, auditable, and honest about what it does not know.

    Returns `OTHER` with low confidence rather than reaching for the nearest
    category. A classifier that always produces a confident label is a
    classifier whose confidence means nothing.
    """

    model_id = RULE_CLASSIFIER_ID

    def __init__(self, aliases: dict[str, tuple[str, ...]] | None = None) -> None:
        self._aliases = aliases or ENTITY_ALIASES

    def _entities(self, blob: str) -> tuple[list[str], list[str]]:
        hits: list[tuple[int, str]] = []
        for canonical, forms in self._aliases.items():
            for form in forms:
                match = re.search(rf"\b{re.escape(form)}\b", blob)
                if match:
                    hits.append((match.start(), canonical))
                    break
        hits.sort()
        ordered: list[str] = []
        for _, canonical in hits:
            if canonical not in ordered:
                ordered.append(canonical)
        return ordered[:1], ordered[1:]

    def extract(self, title: str, text: str, author: str | None = None) -> Extraction:
        blob = f"{title}\n{text}".lower()
        primary, secondary = self._entities(blob)

        # An author who is themselves a known entity is the speaker, and the
        # speaker outranks anyone merely mentioned in the text.
        if author:
            author_primary, _ = self._entities(author.lower())
            if author_primary:
                speaker = author_primary[0]
                secondary = [e for e in (primary + secondary) if e != speaker]
                primary = [speaker]

        matched: list[str] = []
        event_type = EventType.OTHER
        topic = ""
        for candidate, candidate_topic, needles in TYPE_RULES:
            for needle in needles:
                if needle in blob:
                    event_type, topic = candidate, candidate_topic
                    matched.append(needle)
                    break
            if matched:
                break

        if event_type is EventType.OTHER and primary:
            # A known person speaking about nothing we recognise is still a
            # statement, and the distinction between a person and an institution
            # is worth keeping.
            event_type = (
                EventType.POLITICAL_STATEMENT
                if primary[0] in {"donald trump"}
                else EventType.PERSON_STATEMENT
            )
            topic = "statement"

        confidence = 0.85 if matched else (0.4 if primary else 0.1)
        return Extraction(
            entity=primary[0] if primary else "",
            secondary_entities=tuple(secondary),
            event_type=event_type,
            topic=topic,
            subtopic=matched[0] if matched else "",
            confidence=confidence,
            model_id=self.model_id,
            matched_on=tuple(matched),
        )


__all__ = ["Extractor", "RuleExtractor", "Extraction", "ENTITY_ALIASES", "TYPE_RULES"]
