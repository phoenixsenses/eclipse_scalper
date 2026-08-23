"""Eclipse News Intelligence — a research data layer, not a trading system.

This package turns unstructured world information into structured, timestamped,
auditable events so that a *later* research programme can ask whether any of it
predicts anything. It does not answer that question and it must not pretend to.

Three properties are load-bearing, and every module here exists to protect one
of them:

**A feature is what was knowable, an outcome is what happened.** They live in
two different objects that cannot be merged (`schemas.snapshot`). The snapshot
validates every observation it carries against its own decision time and
refuses the ones that come from the future. This repository has burned samples
before by letting a hindsight-selected gate into a feature set; that failure is
structural, so the defence is structural too.

**The decision time is when *we* could know, not when the world published.** A
statement made at 13:44 and received at 13:51 is actionable at 13:51. Every
snapshot is anchored to `first_seen_at`, and `published_at` is kept beside it so
that the gap itself can be studied.

**Relevance is not direction.** The entity graph says the Federal Reserve is
relevant to the two-year yield. It does not say which way. Direction is the
thing under test; a graph that encodes it has answered the research question by
assumption. `relevance.graph` refuses signed weights.

Nothing in this package places an order, reads a sealed arm, mutates E-DER V1 /
A2 / V3, or opens a network connection. The heavy parts — collectors, embedding
indexes, historical backfill, batch LLM classification — exist as interfaces
whose implementations refuse to run; see `deferred`.
"""

from __future__ import annotations

from .version import (
    GRAPH_VERSION,
    PACKAGE_VERSION,
    SCHEMA_VERSION,
    TAXONOMY_VERSION,
)

__all__ = [
    "PACKAGE_VERSION",
    "SCHEMA_VERSION",
    "TAXONOMY_VERSION",
    "GRAPH_VERSION",
]
