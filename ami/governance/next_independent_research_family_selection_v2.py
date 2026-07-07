"""BATCH-NEXT-INDEPENDENT-RESEARCH-FAMILY-SELECTION-V2.

Governance-only, outcome-blind portfolio reconciliation. Determines
whether any family in the authoritative roadmap
(`reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md`,
commit `0c976e21`) is currently eligible to begin a new readiness gate,
given everything closed/blocked/parked since that document was written.

This module reads no outcome value, builds no feature, creates no
experiment/nullifier/gate-receipt, and touches no canonical/knowledge
table. It is a pure, deterministic reconciliation of already-recorded
statuses -- every status string below is copied verbatim from an accepted
report/commit, never re-derived from outcome behavior.

RESULT: all candidates in the V1 roadmap are currently ineligible
(closed, blocked, graveyarded, or parked) -- disposition
`NO_CURRENTLY_ELIGIBLE_INDEPENDENT_FAMILY`.
"""
from __future__ import annotations

ELIGIBLE_STATUSES = frozenset({"UNTOUCHED_ELIGIBLE"})

# Every other enum value from the gate's own status vocabulary is treated
# as not-currently-selectable.
INELIGIBLE_STATUSES = frozenset({
    "ACTIVE_GATE_IN_PROGRESS",
    "PARKED_FOR_SAMPLE_GROWTH",
    "BLOCKED_BY_COVERAGE",
    "BLOCKED_BY_SOURCE_QUALITY",
    "DEFINITION_AMBIGUOUS",
    "MIGRATED_AWAITING_GATE",
    "PREREGISTERED_NOT_EXECUTED",
    "SCIENTIFICALLY_CLOSED",
    "GRAVEYARDED",
    "RETRY_CONDITION_UNMET",
    "DUPLICATE_OR_NONINDEPENDENT",
    "SUPERSEDED",
    "GOVERNANCE_AMBIGUOUS",
})

# ---------------------------------------------------------------------------
# Candidate portfolio, reconciled from the V1 roadmap (Phase 1/2/4) plus
# every accepted commit that has landed since. Every `evidence` value is a
# commit hash, OD-number, or graveyard/failure-archive id already on record
# -- nothing here is newly computed from outcome data.
# ---------------------------------------------------------------------------
CANDIDATES: tuple[dict, ...] = (
    {
        "family_id": "FAM_CVD_WINDOWED_TAKER_FLOW", "rank": None, "score": None,
        "status": "SCIENTIFICALLY_CLOSED",
        "reason": "NO_RELIABLE_ASSOCIATION, G2 governed execution",
        "evidence": "60c3e26f",
    },
    {
        "family_id": "FAM_CVD_WINDOWED_TAKER_FLOW_ALT_WINDOWS", "rank": None, "score": None,
        "status": "DUPLICATE_OR_NONINDEPENDENT",
        "reason": "explicitly forbidden follow-up of the closed CVD child; same predictor family",
        "evidence": "S34_CVD_NEXT_BATCHES_PLAN_2026-07-06.md (plan only, never built)",
    },
    {
        "family_id": "OFI_MOMENTUM", "rank": None, "score": None,
        "status": "GRAVEYARDED",
        "reason": "S34_ORDERFLOW_LEAD, standalone all-timestamp OFI-quantile momentum, dead on all 3 symbols",
        "evidence": "graveyard_slash_fingerprints (ofi momentum / ofi-momentum)",
    },
    {
        "family_id": "OFI_EVENT_ANCHORED_NONMOMENTUM", "rank": None, "score": None,
        "status": "GRAVEYARDED",
        "reason": "shares the identical graveyard fingerprint family as OFI_MOMENTUM; deprioritized not cleared",
        "evidence": "V1 Phase 2",
    },
    {
        "family_id": "PULL_REFILL_LIQUIDITY", "rank": None, "score": None,
        "status": "GRAVEYARDED",
        "reason": "failure_archive id=6 (bk_refill universal separator, REGIME_LIMITED), retry unmet; also an open-recompute Knowledge Object",
        "evidence": "FAILURE_ARCHIVE.md id=6; K-S34-BOOK-PULL-001",
    },
    {
        "family_id": "FUNDING_LEVEL_VELOCITY", "rank": None, "score": None,
        "status": "BLOCKED_BY_SOURCE_QUALITY",
        "reason": "both funding collectors dead (last write 2026-04-13/2026-05-12); no live producer",
        "evidence": "OD-006 (OPEN)",
    },
    {
        "family_id": "OPEN_INTEREST", "rank": None, "score": None,
        "status": "BLOCKED_BY_COVERAGE",
        "reason": "raw collector live but eligible-anchor coverage last measured 38/252 (15%), two disconnected windows; retry not re-satisfied since",
        "evidence": "OD-012 (2026-07-04, IMPLEMENTED/deferred)",
    },
    {
        "family_id": "CROSS_ASSET_TRANSFER", "rank": None, "score": None,
        "status": "GRAVEYARDED",
        "reason": "NO_EDGE, no retry condition offered",
        "evidence": "graveyard id=9",
    },
    {
        "family_id": "CROSS_EXCHANGE_NEW_COLLECTOR", "rank": None, "score": None,
        "status": "BLOCKED_BY_SOURCE_QUALITY",
        "reason": "no collector exists; Phase 5, operator-gated",
        "evidence": "S34_MECHANISM_RESEARCH_PLAN.md",
    },
    {
        "family_id": "LONG_ANCHOR_EVENT_ASYMMETRY", "rank": None, "score": None,
        "status": "BLOCKED_BY_COVERAGE",
        "reason": "ami_events structurally 100% SELL-cascade (252/252 REAL_LIQUIDATION); 0 LONG-anchor population exists",
        "evidence": "OD-017 (OPEN)",
    },
    {
        "family_id": "ENTRY_TIMING_HOLD_EXIT_MFE_MAE_REVERSAL_REGIME", "rank": None, "score": None,
        "status": "SCIENTIFICALLY_CLOSED",
        "reason": "W3/W4/W7A/W8/W10A already answered; sub-cuts individually graveyarded (id=13,14,15) where tested",
        "evidence": "W3/W4/W7A/W8/W10A reports",
    },
    {
        "family_id": "FAILED_BREAKOUTS_SWEEP_RETEST_EXHAUSTION", "rank": None, "score": None,
        "status": "BLOCKED_BY_SOURCE_QUALITY",
        "reason": "NOT_IMPLEMENTED infra; breakout/volume-profile engine does not exist; exhaustion has zero computed feature",
        "evidence": "OD-014 (OPEN)",
    },
    {
        "family_id": "FORWARD_SHADOW_VALIDATION", "rank": None, "score": None,
        "status": "ACTIVE_GATE_IN_PROGRESS",
        "reason": "E-HOUR17-FWD-001 / E-CONVCOMP-FWD-001 already preregistered and accumulating; explicitly leave alone",
        "evidence": "AMI_S34_RESEARCH_BACKLOG.md",
    },
    {
        "family_id": "BIRTH_TRUNCATED_CASCADE_GEOMETRY", "rank": None, "score": None,
        "status": "BLOCKED_BY_SOURCE_QUALITY",
        "reason": "INFERENTIAL_PARKED_SOURCE_DEAD, excluded per operator instruction",
        "evidence": "V1 Phase 2",
    },
    {
        "family_id": "PRE_CASCADE_DIP_RECOVERY", "rank": None, "score": None,
        "status": "GRAVEYARDED",
        "reason": "NO_EDGE; retry (>=6mo more data + single pre-fixed config) unmet",
        "evidence": "failure_archive id=22",
    },
    # --- V1 shortlist (candidates 1-3), all now resolved since V1 was written ---
    {
        "family_id": "FAM_CASCADE_ABSORPTION_IMPACT", "rank": 1, "score": 51,
        "status": "SCIENTIFICALLY_CLOSED",
        "reason": "NO_RELIABLE_INCREMENTAL_ASSOCIATION, governed execution",
        "evidence": "execution 5e9e2e33, closure ba3ab906",
    },
    {
        "family_id": "FAM_SPOT_PERP_BASIS_REVERSAL", "rank": 2, "score": 47,
        "status": "BLOCKED_BY_COVERAGE",
        "reason": "SPOT_PERP_BASIS_BLOCKED_BY_COVERAGE, 54/324 valid aligned anchors, 38 independent cycles",
        "evidence": "1630f0a1",
    },
    {
        "family_id": "FAM_BOOK_SPREAD_DYNAMICS", "rank": 3, "score": 47,
        "status": "PARKED_FOR_SAMPLE_GROWTH",
        "reason": (
            "LONG child H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1 parked: TEST=18<20 "
            "(need eligible n>=67); mixed-direction child H-BOOK-SPREAD-CHANGE-BPS-W300-V1 "
            "closed INCOMPLETE (direction-mixed sign unresolvable, not retryable without a "
            "new authorized scope); SHORT not authorized"
        ),
        "evidence": "M-0036 5267a15a; mixed a4722117; LONG 93b7296d",
    },
)


def build_portfolio_matrix() -> tuple[dict, ...]:
    """Returns the full candidate matrix with an `eligible` flag derived
    purely from `status` membership in `ELIGIBLE_STATUSES` -- no outcome
    value or new computation is involved."""
    return tuple({**c, "eligible": c["status"] in ELIGIBLE_STATUSES} for c in CANDIDATES)


def select_next_family(matrix: tuple[dict, ...] | None = None) -> dict:
    """Selects the highest-ranked eligible candidate, or returns the
    NO_CURRENTLY_ELIGIBLE_INDEPENDENT_FAMILY disposition if none exists.
    Ranking uses only the frozen `rank` field from the V1 roadmap (lower
    rank = higher priority); candidates without a rank (excluded in V1
    itself) are never selectable regardless of status."""
    matrix = matrix if matrix is not None else build_portfolio_matrix()
    eligible = [c for c in matrix if c["eligible"] and c["rank"] is not None]
    eligible.sort(key=lambda c: c["rank"])
    if eligible:
        return {"disposition": "FAMILY_SELECTED", "selected": eligible[0],
                "all_candidates": matrix}
    return {
        "disposition": "NO_CURRENTLY_ELIGIBLE_INDEPENDENT_FAMILY",
        "selected": None, "all_candidates": matrix,
        "retry_conditions": _retry_conditions(matrix),
    }


def _retry_conditions(matrix: tuple[dict, ...]) -> tuple[dict, ...]:
    """For every currently-ineligible candidate, the exact condition that
    would need to change before it becomes selectable again -- copied from
    each candidate's own accepted closure/block record, not invented."""
    conditions = {
        "FAM_CASCADE_ABSORPTION_IMPACT": "none -- scientifically closed for this child; no rescue permitted",
        "FAM_SPOT_PERP_BASIS_REVERSAL": "sufficient new forward spot-price coverage to raise aligned-anchor count past the frozen minimum",
        "FAM_BOOK_SPREAD_DYNAMICS": "LONG eligible independent-cycle population reaches >=67 (9 more eligible LONG cycles) under the frozen 70/30 split, or a new authorized split-ratio scoping decision",
        "OPEN_INTEREST": "a dedicated, separately-authorized anchor-coverage recheck (forbidden inside a selection-only batch) showing coverage materially above the last-measured 15%",
        "FUNDING_LEVEL_VELOCITY": "a live funding-rate collector is (re)activated (operator decision pending, OD-006)",
        "LONG_ANCHOR_EVENT_ASYMMETRY": "a LONG-anchor event population is structurally created (currently 0/252)",
        "FAILED_BREAKOUTS_SWEEP_RETEST_EXHAUSTION": "the breakout/volume-profile chart-native engine is built (NOT_IMPLEMENTED)",
        "CROSS_EXCHANGE_NEW_COLLECTOR": "a cross-exchange collector is built (none exists)",
        "PULL_REFILL_LIQUIDITY": "forward data accumulation resolves the open recompute flag (unconfirmed)",
        "PRE_CASCADE_DIP_RECOVERY": ">=6 months of additional data plus a single pre-fixed config prereg",
        "OFI_MOMENTUM": "none offered by the graveyard record",
        "OFI_EVENT_ANCHORED_NONMOMENTUM": "would need to be argued independently distinct from the OFI graveyard fingerprint, not attempted here",
        "CROSS_ASSET_TRANSFER": "none offered by the graveyard record",
        "BIRTH_TRUNCATED_CASCADE_GEOMETRY": "source revival (currently dead)",
        "FORWARD_SHADOW_VALIDATION": "not a retry condition -- already active, leave running",
        "ENTRY_TIMING_HOLD_EXIT_MFE_MAE_REVERSAL_REGIME": "none -- already answered; a genuinely novel sub-question would need its own justification",
        "FAM_CVD_WINDOWED_TAKER_FLOW": "none -- scientifically closed, no rescue permitted",
        "FAM_CVD_WINDOWED_TAKER_FLOW_ALT_WINDOWS": "none -- explicitly forbidden as a disguised rescue",
    }
    return tuple({"family_id": c["family_id"], "condition": conditions.get(c["family_id"], "unspecified")}
                 for c in matrix if not c["eligible"])
