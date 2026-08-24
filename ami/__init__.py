"""AMI — Artificial Market Intelligence.

Canonical implementation of AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md
(Appendix F build brief). Phase 0-5 foundation:

- ami.enums          : canonical statuses, permissions, claim types, state families
- ami.constitution   : constitutional principles (machine-checkable where possible)
- ami.knowledge      : KnowledgeObject + governed knowledge store (SQLite)
- ami.governance     : EpistemicGovernor (permissions, promotion/demotion, circuit breakers)
- ami.states         : multi-timeframe StateObject engine + structure phases
- ami.lifecycle      : trade lifecycle states + post-entry snapshots
- ami.research       : question/hypothesis/experiment registry, failure archive, marketplace
- ami.decision       : DecisionTrace + authorization flow

GUARDRAILS (Appendix F — binding):
- Never modifies live executor, .env, leverage, sizing.
- Never sends exchange orders.
- Never promotes research findings on its own; promotion gates are code, not narrative.
- Main market DB is opened read-only. AMI stores live in data/ami/.
"""
__version__ = "0.1.0"
