"""Refusals.

Every failure mode in this package is a refusal rather than a repair. The
alternative — filtering the bad field, estimating the missing window, defaulting
the unknown label — produces a row that looks like every other row and carries a
defect no downstream reader can see. A raised exception is loud once; a silently
mended row is wrong forever.
"""

from __future__ import annotations


class NewsIntelligenceError(Exception):
    """Base class."""


class LookaheadError(NewsIntelligenceError):
    """An observation from after the decision time was offered as a feature.

    The most expensive class of bug this package can have, so it is the loudest.
    Raised at construction time, not at use time: by the time a snapshot is read
    the caller no longer knows what went into it.
    """


class OutcomeInFeatureSpace(NewsIntelligenceError):
    """A field that can only be known afterwards appeared in a feature object.

    Distinct from `LookaheadError`: that one is about a timestamp, this one is
    about a *kind* of field. A realised return carries no timestamp of its own,
    so only a structural check catches it.
    """


class DeterministicFieldOverwrite(NewsIntelligenceError):
    """A model tried to write a field that must stay deterministic.

    Timestamps, source identity, payload hashes and ids come from the wire and
    from the clock. A classifier may add labels beside them; it may never
    replace them. Without this, an event's provenance becomes a model output.
    """


class OutcomeAwareClustering(NewsIntelligenceError):
    """Clustering was offered information about what happened next.

    Clusters decide what counts as one independent observation. If the outcome
    can influence the grouping, the sample size becomes a function of the result
    — the single most effective way to manufacture significance.
    """


class RelevanceIsNotDirection(NewsIntelligenceError):
    """A signed or directional weight was offered to the relevance graph.

    "Trump is relevant to tariffs" is an input. "Trump is bearish for BTC" is
    the conclusion of a study that has not been run.
    """


class DeferredUntilPhase1Complete(NewsIntelligenceError):
    """A capability that would compete for CPU, RAM, disk or network.

    The interface exists so the system can be built and tested; the
    implementation refuses to start while the current research phase holds the
    machine. Deliberately not a silent no-op — a collector that quietly does
    nothing looks exactly like a collector that is working.
    """


class UnknownSource(NewsIntelligenceError):
    """An item arrived from a source that is not in the registry.

    Authority is a property of a registered source. An unregistered one has no
    authority rather than a default authority.
    """
