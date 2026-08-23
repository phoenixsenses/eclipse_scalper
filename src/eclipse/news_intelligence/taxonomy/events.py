"""The event vocabulary.

Deliberately not a sentiment scale. "Positive" and "negative" describe how a
headline reads, and how a headline reads is the least informative thing about
it: a tariff announcement and an earnings beat can both read positive and have
nothing in common mechanically. The categories below are about *what kind of
thing happened*, because that is what determines which assets could plausibly
respond and through what channel.

Sentiment still exists — see `schemas.normalized.Sentiment` — but it is one
field among many rather than the organising axis.

`OTHER` is a real answer. A classifier that never returns it is a classifier
that is guessing.
"""

from __future__ import annotations

from enum import Enum

from ..version import TAXONOMY_VERSION


class EventType(str, Enum):
    # --- macro and policy -------------------------------------------------
    MACRO_RELEASE = "MACRO_RELEASE"
    CENTRAL_BANK = "CENTRAL_BANK"
    RATE_POLICY = "RATE_POLICY"
    INFLATION = "INFLATION"
    EMPLOYMENT = "EMPLOYMENT"

    # --- trade, sanctions, geopolitics ------------------------------------
    TRADE_POLICY = "TRADE_POLICY"
    TARIFF = "TARIFF"
    SANCTIONS = "SANCTIONS"
    GEOPOLITICAL = "GEOPOLITICAL"

    # --- regulation --------------------------------------------------------
    CRYPTO_REGULATION = "CRYPTO_REGULATION"
    FINANCIAL_REGULATION = "FINANCIAL_REGULATION"

    # --- company -----------------------------------------------------------
    COMPANY_EARNINGS = "COMPANY_EARNINGS"
    COMPANY_GUIDANCE = "COMPANY_GUIDANCE"
    PRODUCT = "PRODUCT"
    MERGER_ACQUISITION = "M&A"
    LEGAL = "LEGAL"
    SEC_FILING = "SEC_FILING"

    # --- people ------------------------------------------------------------
    PERSON_STATEMENT = "PERSON_STATEMENT"
    POLITICAL_STATEMENT = "POLITICAL_STATEMENT"
    SOCIAL_POST = "SOCIAL_POST"

    # --- infrastructure and market plumbing --------------------------------
    SECURITY_INCIDENT = "SECURITY_INCIDENT"
    EXCHANGE_INCIDENT = "EXCHANGE_INCIDENT"
    MARKET_STRUCTURE = "MARKET_STRUCTURE"

    # --- commodities -------------------------------------------------------
    COMMODITY_SHOCK = "COMMODITY_SHOCK"

    # --- the honest default -------------------------------------------------
    OTHER = "OTHER"


#: Types whose release time is known in advance. The distinction matters more
#: than it looks: for a scheduled event the market has already priced an
#: expectation, so the informative quantity is the *surprise* against consensus,
#: not the level. For an unscheduled one there is no consensus to surprise.
SCHEDULED_TYPES = frozenset(
    {
        EventType.MACRO_RELEASE,
        EventType.CENTRAL_BANK,
        EventType.INFLATION,
        EventType.EMPLOYMENT,
        EventType.COMPANY_EARNINGS,
    }
)


def is_scheduled(event_type: EventType) -> bool:
    return event_type in SCHEDULED_TYPES


__all__ = ["EventType", "SCHEDULED_TYPES", "is_scheduled", "TAXONOMY_VERSION"]
