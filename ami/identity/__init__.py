"""Event/cycle identity foundation (Phase 3).

Immutable event_id generation and real-vs-proxy source-quality
classification (Protocol §8, R-09/CONFLICT-008). Canonical cycle
resolution (ami_cycles population) is intentionally NOT part of this
package -- it is BLOCKED_PENDING_OPERATOR_DECISION(OD-003). See
cooldown_sensitivity.py for the non-canonical sensitivity views that
remain safe to compute without that decision.
"""
from ami.identity.event_identity import (
    PooledPopulationViolation,
    SourceQuality,
    assert_not_pooled,
    generate_event_id,
)

__all__ = [
    "PooledPopulationViolation",
    "SourceQuality",
    "assert_not_pooled",
    "generate_event_id",
]
