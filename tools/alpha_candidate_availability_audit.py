"""Candidate-level entry-predicate availability audit.

BATCH-ALPHA-CANDIDATE-AVAILABILITY-AUDIT-V1.

Companion to `tools/s34_feature_availability.py` (which audits individual
*features* against `FeatureClass` + `knowable_at_ts`). This module audits
whole *candidates*: given the predicate feature names a candidate's entry
rule depends on and the timestamps involved, it certifies whether the
entry decision could actually have known those features, and assigns one
of a fixed set of dispositions. It does not compute performance, does not
search for new candidates, and does not run any DB query -- callers supply
timestamps/rows explicitly (see `compute_threshold_cross_ts`).

Core invariant (see reports/governance/alpha/ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1.md):

    entry_decision_ts_ms >= feature_available_ts_ms
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
# NOTE: ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1.md is the hand-authored governance
# doc (invariant, rules, checklist) -- this tool must never write there. Its
# auto-generated output uses the _FAMILY_RECORD suffix instead.
OUT_MD = ROOT / "reports" / "governance" / "alpha" / "ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1_FAMILY_RECORD.md"
OUT_JSON = ROOT / "reports" / "governance" / "alpha" / "ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1.json"

# ---------------------------------------------------------------------------
# Dispositions (exact strings; see governance doc section 6)
# ---------------------------------------------------------------------------
PASS_AVAILABILITY_AUDIT = "PASS_AVAILABILITY_AUDIT"
BLOCKED_BY_AVAILABILITY_UNKNOWN = "BLOCKED_BY_AVAILABILITY_UNKNOWN"
REJECT_ENTRY_PREDICATE_LOOKAHEAD = "REJECT_ENTRY_PREDICATE_LOOKAHEAD"
REJECT_NO_EDGE_AFTER_AVAILABILITY_CORRECTION = "REJECT_NO_EDGE_AFTER_AVAILABILITY_CORRECTION"
HOLD_FOR_NEW_MECHANISM_CLAIM = "HOLD_FOR_NEW_MECHANISM_CLAIM"
HOLD_FOR_FORWARD_PREREGISTRATION = "HOLD_FOR_FORWARD_PREREGISTRATION"
OBSERVATION_ONLY = "OBSERVATION_ONLY"
NOT_AN_ALPHA_CANDIDATE = "NOT_AN_ALPHA_CANDIDATE"

# Rule 2: features requiring full cluster membership -- never available before
# cluster_end_ts_ms, regardless of how early entry_decision_ts_ms is.
COMPLETED_CLUSTER_AGGREGATE_FEATURES = frozenset({
    "cluster_notional",
    "cluster_liq_count",
    "cluster_count",
    "cluster_duration_s",
    "cluster_duration_sec",
    "max_single_liq_share",
    "cluster_max_notional",
    "frontloaded_ratio",
    "backloaded_ratio",
    "shape_label",
})

# Rule 5: post-entry outcome/tempo features -- diagnostics only, never predicates.
POST_ENTRY_DIAGNOSTIC_FEATURES = frozenset({
    "time_to_mfe_s", "time_to_mae_s", "time_to_tp_s", "time_to_sl_s",
    "time_to_be_s", "time_to_exit_s",
    "first_1m_net_bps", "first_5m_net_bps", "first_15m_net_bps", "first_30m_net_bps",
    "mfe_bps", "mae_bps", "net_bps", "gross_bps", "exit_reason",
    "outcome_tempo_state", "mfe_speed_bps_per_min", "mae_speed_bps_per_min",
    "tempo_edge_ratio",
})


@dataclass(frozen=True)
class CandidateAvailabilityRecord:
    family_id: str
    candidate_id: str
    predicate_features: tuple[str, ...]
    event_ts_ms: int
    entry_decision_ts_ms: int
    feature_available_ts_ms: int | None
    feature_available_rule: str
    disposition: str = ""
    reason: str = ""


def compute_threshold_cross_ts(events: Sequence[tuple[int, float]], threshold_notional: float) -> int | None:
    """Rule 3 helper. `events` is a `(ts_ms, notional)` row sequence for a single
    cluster/side (e.g. liquidation prints); need not be pre-sorted. Returns the
    ts_ms of the first row at which the *running cumulative* notional first
    reaches/exceeds `threshold_notional` -- this is `feature_available_ts_ms`
    for any predicate of the form `running_cluster_notional >= threshold_notional`.
    Deterministic, no DB. Returns None if the threshold is never crossed."""
    cumulative = 0.0
    for ts_ms, notional in sorted(events, key=lambda row: row[0]):
        cumulative += float(notional)
        if cumulative >= float(threshold_notional):
            return int(ts_ms)
    return None


def compute_count_cross_ts(events: Sequence[tuple[int, float]], threshold_count: int) -> int | None:
    """Rule 3 helper for running-count predicates (e.g. cluster_liq_count >= 22).
    Same shape/semantics as `compute_threshold_cross_ts`; the notional value in
    each row is unused for the count itself but kept for a uniform row shape."""
    ordered = sorted(events, key=lambda row: row[0])
    if len(ordered) < threshold_count:
        return None
    return int(ordered[threshold_count - 1][0])


def completed_cluster_feature_available_ts(cluster_end_ts_ms: int) -> int:
    """Rule 2: any completed-cluster aggregate is available no earlier than
    cluster_end_ts_ms."""
    return int(cluster_end_ts_ms)


def audit_candidate_availability(record: CandidateAvailabilityRecord) -> CandidateAvailabilityRecord:
    """Applies rules 1, 5, 6, 7, 8 to a single candidate and returns a NEW
    record with `disposition`/`reason` filled in. Never mutates the input,
    never touches a DB, never computes performance."""

    diagnostic_hits = sorted(f for f in record.predicate_features if f in POST_ENTRY_DIAGNOSTIC_FEATURES)
    if diagnostic_hits:
        return _decide(
            record, REJECT_ENTRY_PREDICATE_LOOKAHEAD,
            f"rule 5: post-entry diagnostic feature(s) {diagnostic_hits} used as entry "
            f"predicate; diagnostics are explanation-only and can never be predicates")

    completed_hits = sorted(f for f in record.predicate_features if f in COMPLETED_CLUSTER_AGGREGATE_FEATURES)
    if completed_hits and record.feature_available_rule == "cluster_start_entry":
        return _decide(
            record, REJECT_ENTRY_PREDICATE_LOOKAHEAD,
            f"rule 8: event_ts_ms == cluster_start_ts_ms but predicate uses completed-cluster "
            f"aggregate(s) {completed_hits}, which finalize only at cluster_end_ts_ms")

    if record.feature_available_ts_ms is None:
        return _decide(
            record, BLOCKED_BY_AVAILABILITY_UNKNOWN,
            "rule 6: feature_available_ts_ms is not provided/derivable; availability cannot be certified")

    if int(record.entry_decision_ts_ms) < int(record.feature_available_ts_ms):
        return _decide(
            record, REJECT_ENTRY_PREDICATE_LOOKAHEAD,
            f"rule 1/7: entry_decision_ts_ms={record.entry_decision_ts_ms} < "
            f"feature_available_ts_ms={record.feature_available_ts_ms}")

    return _decide(
        record, PASS_AVAILABILITY_AUDIT,
        f"entry_decision_ts_ms={record.entry_decision_ts_ms} >= "
        f"feature_available_ts_ms={record.feature_available_ts_ms} under rule "
        f"'{record.feature_available_rule}'")


def _decide(record: CandidateAvailabilityRecord, disposition: str, reason: str) -> CandidateAvailabilityRecord:
    return replace(record, disposition=disposition, reason=reason)


# ---------------------------------------------------------------------------
# FAM_ETH_BUY_LIQ_CONTINUATION -- current rejected family audit record
# (DISPOSABLE_ALPHA_CANDIDATE_PROMOTION_REHEARSAL_V1 finding, codified here)
# ---------------------------------------------------------------------------

_FAM_ETH_BUY_LIQ_CONTINUATION_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CAND_ETH_BUY_CONT_500K_DAYTREND_D0_TP40_SL50_BE20", ("cluster_notional", "day_trend_bps")),
    ("CAND_ETH_BUY_CONT_1M_DAYTREND_D0_TP40_SL50_BE20", ("cluster_notional", "day_trend_bps")),
    ("CAND_ETH_BUY_CONT_500K_GEOM_COUNT22_D0_TP40_SL50_BE20", ("cluster_notional", "cluster_liq_count")),
    ("CAND_ETH_BUY_CONT_500K_CASCADE_P15_109K_D0_TP40_SL50_BE20", ("cluster_notional", "prior15_buy_liq_notional")),
    ("CAND_ETH_BUY_CONT_500K_DAYTREND_GEOM_CASCADE_D0_TP40_SL50_BE20",
     ("cluster_notional", "day_trend_bps", "cluster_liq_count", "prior15_buy_liq_notional")),
)


def build_family_audit_records() -> list[CandidateAvailabilityRecord]:
    """Audited records for FAM_ETH_BUY_LIQ_CONTINUATION exactly as it was
    rehearsed: `event_ts_ms == cluster_start_ts_ms` (delay0), so every
    candidate is scored under `feature_available_rule="cluster_start_entry"`.
    Timestamps are representative, not the exact per-event values -- the
    disposition depends only on the completed-cluster-feature/rule mismatch
    (rule 8), not on the specific numbers."""
    event_ts_ms = 1_780_000_000_000  # == cluster_start_ts_ms by construction
    records = []
    for candidate_id, predicate_features in _FAM_ETH_BUY_LIQ_CONTINUATION_CANDIDATES:
        record = CandidateAvailabilityRecord(
            family_id="FAM_ETH_BUY_LIQ_CONTINUATION",
            candidate_id=candidate_id,
            predicate_features=predicate_features,
            event_ts_ms=event_ts_ms,
            entry_decision_ts_ms=event_ts_ms,
            feature_available_ts_ms=None,
            feature_available_rule="cluster_start_entry",
        )
        records.append(audit_candidate_availability(record))
    return records


def build_family_summary() -> dict[str, Any]:
    return {
        "family_id": "FAM_ETH_BUY_LIQ_CONTINUATION",
        "family_disposition": REJECT_ENTRY_PREDICATE_LOOKAHEAD,
        "promotion_disposition": "REJECT_SPURIOUS_OR_DATA_PATH_SUSPECT",
        "reason": [
            "event_ts_ms == cluster_start_ts_ms for all universe events",
            "predicate uses completed-cluster aggregate(s) (cluster_notional, "
            "cluster_liq_count) not available until cluster_end_ts_ms, or running-"
            "threshold facts not available until threshold_cross_ts_ms",
            "honest threshold-cross control loses the edge (DISPOSABLE_ALPHA_CANDIDATE_"
            "PROMOTION_REHEARSAL_V1: mark median ~-8bps vs lookahead-entry ~+33bps)",
        ],
        "canonical_alpha_gate_eligible": False,
        "permitted_future_work": [
            "new preregistered mechanism claim using feature_available_ts_ms / "
            "threshold_cross_ts_ms as the entry anchor (see "
            "tools/research_s34_knowable_anchor_continuation.py for the repo's "
            "established knowable-anchor pattern)",
        ],
        "banned_future_work": [
            "reuse of cluster_start (delay0) entry combined with a completed-cluster "
            "predicate for this or any structurally identical family",
        ],
    }


def build_registry_payload() -> dict[str, Any]:
    records = build_family_audit_records()
    summary = build_family_summary()
    return {
        "invariant": "entry_decision_ts_ms >= feature_available_ts_ms",
        "family_summary": summary,
        "candidate_records": [
            {
                "family_id": r.family_id,
                "candidate_id": r.candidate_id,
                "predicate_features": list(r.predicate_features),
                "event_ts_ms": r.event_ts_ms,
                "entry_decision_ts_ms": r.entry_decision_ts_ms,
                "feature_available_ts_ms": r.feature_available_ts_ms,
                "feature_available_rule": r.feature_available_rule,
                "disposition": r.disposition,
                "reason": r.reason,
            }
            for r in records
        ],
    }


def write_registry(out_md: Path = OUT_MD, out_json: Path = OUT_JSON) -> dict[str, Any]:
    payload = build_registry_payload()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = payload["family_summary"]
    lines = [
        "# Alpha Candidate Availability Audit -- Family Record (auto-generated)",
        "",
        f"- family_id: `{summary['family_id']}`",
        f"- family_disposition: `{summary['family_disposition']}`",
        f"- promotion_disposition: `{summary['promotion_disposition']}`",
        f"- canonical_alpha_gate_eligible: `{summary['canonical_alpha_gate_eligible']}`",
        "",
        "| candidate_id | predicate_features | disposition | reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["candidate_records"]:
        lines.append(
            f"| `{row['candidate_id']}` | {', '.join(row['predicate_features'])} | "
            f"`{row['disposition']}` | {row['reason']} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write alpha candidate availability audit registry.")
    parser.add_argument("--fail-on-pass-missing", action="store_true",
                        help="exit 1 if any FAM_ETH_BUY_LIQ_CONTINUATION candidate is not "
                             "REJECT_ENTRY_PREDICATE_LOOKAHEAD (regression guard)")
    args = parser.parse_args()
    payload = write_registry()
    dispositions = {row["disposition"] for row in payload["candidate_records"]}
    print(f"family_disposition={payload['family_summary']['family_disposition']}")
    print(f"candidate_dispositions={sorted(dispositions)}")
    print(f"md={OUT_MD}")
    print(f"json={OUT_JSON}")
    if args.fail_on_pass_missing and dispositions != {REJECT_ENTRY_PREDICATE_LOOKAHEAD}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
