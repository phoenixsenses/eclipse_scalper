"""Versions for everything that can change the meaning of a stored row.

Four version numbers rather than one, because they move for different reasons
and a research result has to be able to say which of them it was computed
under. A taxonomy revision changes what `TRADE_POLICY` means; a graph revision
changes which assets an entity touches; a schema revision changes the shape of
the row. Collapsing them into a single "version" would make a result that
survives one revision indistinguishable from one that survives all three.

Bumping any of these is a governance act: existing rows keep the version they
were written with, and a study that pools rows across a bump has to say so.
"""

from __future__ import annotations

PACKAGE_VERSION = "0.1.0"

#: Shape of the stored rows (raw, normalized, snapshot, label).
SCHEMA_VERSION = 1

#: Meaning of the event-type vocabulary in `taxonomy.events`.
TAXONOMY_VERSION = 1

#: Contents of the entity -> asset relevance graph in `relevance.graph`.
GRAPH_VERSION = 1

#: Every deterministic classifier in this package identifies itself with this.
#: An LLM-assisted annotation carries its own model id instead; see
#: `normalization.annotation`.
RULE_CLASSIFIER_ID = "rule-classifier@1"
