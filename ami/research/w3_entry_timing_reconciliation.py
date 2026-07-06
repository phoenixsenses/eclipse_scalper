"""BATCH-P6-004: W3 (entry timing) due-diligence reconciliation.

Before designing any new W3 population/prereg, this reconciles every existing
entry-timing report in reports/research/s34/ against data/ami/knowledge.sqlite
:failure_archive, per the same due-diligence discipline that caught the W2
graveyard collision (SYSTEM_STATE §53 / OD-012).

This is NOT an alpha-search wave -- no new population is frozen, no new
threshold is fit, no new trade is scored. It is a read-only synthesis of
PRIOR results, recorded in canonical SQL (experiment_registry/experiment_
results) instead of only in a new Markdown file, per the operator's
token-efficient documentation policy and explicit instruction for this batch.

Per-report verdict taxonomy (operator-specified):
    EXACT_HYPOTHESIS_ALREADY_TESTED, SCIENTIFICALLY_REJECTED,
    ECONOMICALLY_REJECTED, RETRY_CONDITION_NOT_MET, RETRY_CONDITION_MET,
    DISTINCT_MECHANISM, UNANSWERED_RESEARCH_GAP, SUPERSEDED,
    INSUFFICIENT_SAMPLE
"""
from __future__ import annotations
import hashlib
import time

from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates

EXPERIMENT_ID = "E-W3-ENTRY-TIMING-RECONCILIATION-001"

# Each entry: report file reviewed -> (verdict(s), one-line reason, matched
# failure_archive id(s) if any). Verdicts are read-only findings from this
# reconciliation, not new experiment results on fresh data.
RECONCILIATION = [
    {
        "report": "S34_ABSORPTION_TIMING.md",
        "verdict": "UNANSWERED_RESEARCH_GAP",
        "reason": "N=36, ETH SELL deep-V 28-40bps: whether pre-cross order-book "
                  "absorption state is a point-in-time-valid permission feature "
                  "(vs T+5/T+30 confirmation-only) was never formally resolved "
                  "(no freeze, no TRAIN/TEST, no MC, own 'Read' section is "
                  "conditional/undecided). Distinct from every delay/confirm-"
                  "entry graveyard idea -- this is about FEATURE point-in-time "
                  "validity, not entry-delay optimization.",
        "matched_failure_archive_ids": [],
    },
    {
        "report": "S34_ABSORPTION_TIMING_BROAD.md",
        "verdict": "UNANSWERED_RESEARCH_GAP",
        "reason": "N=51, broader route (28bps-inf), same open question as above, "
                  "same undecided conclusion. Companion population, not an "
                  "independent gap.",
        "matched_failure_archive_ids": [],
    },
    {
        "report": "S34_CONFIRMATION_ENTRY.md",
        "verdict": "EXACT_HYPOTHESIS_ALREADY_TESTED;ECONOMICALLY_REJECTED",
        "reason": "'Wait for reclaim, then fade' -- confirm_5m/10m/15m all show "
                  "weaker or negative calibration-period performance vs "
                  "baseline_no_wait. This IS the buy-side-fade-with-delayed-"
                  "confirmation mechanism.",
        "matched_failure_archive_ids": [8, 16],
    },
    {
        "report": "S34_EARLY_BUILD_ENTRY.md",
        "verdict": "EXACT_HYPOTHESIS_ALREADY_TESTED;ECONOMICALLY_REJECTED;RETRY_CONDITION_NOT_MET",
        "reason": "Enter at Kth same-side liquidation within W seconds, ALL "
                  "symbols/directions/K/W combos net-negative both cal and "
                  "holdout, 'Leads: none'. Exact match to failure_archive #1. "
                  "Retry condition ('much stronger trigger precision') not "
                  "satisfied by anything built this session.",
        "matched_failure_archive_ids": [1],
    },
    {
        "report": "S34_ENTRY_OFFSET_DECOMPOSITION.md",
        "verdict": "SUPERSEDED",
        "reason": "Self-labeled 'execution-realism diagnostic, not a new rule' -- "
                  "measures fill-latency sensitivity of an existing route concept, "
                  "not a new hypothesis. Knowable-anchor entry (threshold_cross/"
                  "cluster_end) is markedly worse than event_ts_ms-anchored "
                  "version, a lookahead-sensitivity finding already superseded "
                  "by the known-at/point-in-time discipline built into every "
                  "Phase 4+ ami/chart/* object (known_at_ts=close_ts_ms).",
        "matched_failure_archive_ids": [],
    },
    {
        "report": "S34_ENTRY_OFFSET_DECOMPOSITION_50K_TP120.md",
        "verdict": "SUPERSEDED",
        "reason": "Same diagnostic, different threshold/TP config (50K/TP120 vs "
                  "500K/TP60). Same conclusion.",
        "matched_failure_archive_ids": [],
    },
    {
        "report": "S34_MICRO_ENTRY_SCALP.md",
        "verdict": "EXACT_HYPOTHESIS_ALREADY_TESTED;ECONOMICALLY_REJECTED",
        "reason": "Confirm/pullback wait entries across SOL/ETH/BTC buckets -- "
                  "every row verdict is 'thin' or 'reject_negative_median', no "
                  "robust candidate. Matches failure_archive #2 (broad micro-"
                  "timing optimization, NO_EDGE, no retry condition given).",
        "matched_failure_archive_ids": [2],
    },
    {
        "report": "S34_SELL_DELAYED_LONG_SCAN.md",
        "verdict": "EXACT_HYPOTHESIS_ALREADY_TESTED;ECONOMICALLY_REJECTED;RETRY_CONDITION_NOT_MET",
        "reason": "SELL cascade -> delayed LONG, DELAY600 grid: TRAIN weak-"
                  "positive, TEST negative/fragile (cum -38.66, top3-removed "
                  "-156.10). Report's own 'Read' section: 'any promotion would "
                  "need a separate pre-registration' -- never promoted. This is "
                  "the precursor scan folded into failure_archive #16, whose own "
                  "reason field cites 'val_delay600_untouched=-12.3' -- the same "
                  "route family. #16's retry condition ('forward shadow'da "
                  "delay-grid izle; yeni prereg olmadan tekrar tarama YASAK') is "
                  "NOT met -- no new prereg exists.",
        "matched_failure_archive_ids": [16],
    },
    {
        "report": "S34_V_ENGINE_CONFIRMATION_DELAY_SWEEP.md",
        "verdict": "EXACT_HYPOTHESIS_ALREADY_TESTED;INSUFFICIENT_SAMPLE",
        "reason": "N=22 baseline, confirmation-delay sweep on a different live "
                  "route config (O20_W300_O5_C1) -- every delayed-entry cell's "
                  "post-deterioration T3R is below baseline. Same mechanism "
                  "question as #2/#16 (does waiting for confirmation help), "
                  "same negative answer, but N=22 is too small to stand alone "
                  "as a fresh, independently powered verdict.",
        "matched_failure_archive_ids": [2, 16],
    },
    {
        "report": "S34_V02_ENTRY_QUALITY_NAVIGATION.md",
        "verdict": "INSUFFICIENT_SAMPLE",
        "reason": "N=11, self-labeled 'navigation labels for the current v0.2 "
                  "alpha, not new entry filters... tags are for dashboard/shadow "
                  "observation only'. Operational monitoring of an existing live "
                  "route, not a research hypothesis test.",
        "matched_failure_archive_ids": [],
    },
]

CONCLUSION = (
    "9/10 reports either exactly re-test a graveyarded entry-timing mechanism "
    "(delay/confirm/pullback/wait-for-reclaim -- #1, #2, #8, #16) with the same "
    "negative economic verdict, are execution-realism diagnostics superseded by "
    "the existing known-at discipline, or are operational dashboard monitoring "
    "with N too small to be a hypothesis test. The ONE genuine open question "
    "(absorption-timing point-in-time permission-feature validity, N=36-51) is "
    "real and distinct from every graveyarded idea, but is currently "
    "underpowered by the same order of magnitude that just blocked W2's OI "
    "retry-condition (OD-012) -- freezing a full W3 prereg on it now would "
    "likely also yield INSUFFICIENT_SAMPLE. Per operator instruction, W3 is NOT "
    "forced open on this basis; the reconciliation is recorded and the next "
    "independent research wave (W4: post-event path taxonomy + structural "
    "location + event geometry, precondition P3 paths -- already satisfied) is "
    "the next candidate."
)


def freeze_and_record(conn, provenance: str = "batch-p6-004-w3-reconciliation") -> dict:
    now = int(time.time() * 1000)
    frozen_population = (
        "10 reports/research/s34/*.md entry-timing files (S34_ABSORPTION_TIMING, "
        "S34_ABSORPTION_TIMING_BROAD, S34_CONFIRMATION_ENTRY, S34_EARLY_BUILD_ENTRY, "
        "S34_ENTRY_OFFSET_DECOMPOSITION[+50K_TP120], S34_MICRO_ENTRY_SCALP, "
        "S34_SELL_DELAYED_LONG_SCAN, S34_V_ENGINE_CONFIRMATION_DELAY_SWEEP, "
        "S34_V02_ENTRY_QUALITY_NAVIGATION) x data/ami/knowledge.sqlite:failure_archive (21 rows)"
    )
    # reconciliation of static, already-existing artifacts -- hash the reconciliation
    # table itself (not a live dataset snapshot) so a re-run with unchanged verdicts
    # produces the same hash.
    dataset_hash = hashlib.sha256(
        "|".join(f"{i['report']}={i['verdict']}" for i in RECONCILIATION).encode("utf-8")
    ).hexdigest()

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary (graveyard-checked; no_test_split=True matches
    # this module's own "N/A (no train/test split, static artifact review)").
    results = [
        (item["report"],
         f"{item['verdict']} | matched_failure_archive_ids={item['matched_failure_archive_ids']} | "
         f"{item['reason']}")
        for item in RECONCILIATION
    ] + [("CONCLUSION", CONCLUSION)]
    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_ENTRY_TIMING",
            "hypothesis_id": "H-W3-RECONCILIATION", "preregistered_at": now,
            "frozen_population": frozen_population,
            "frozen_features": "N/A (reconciliation, not a feature-based test)",
            "frozen_target": "determine whether W3 has a genuine unanswered research gap distinct "
                              "from graveyarded entry-timing hypotheses",
            "frozen_thresholds": "N/A (no threshold search)",
            "frozen_splits": "N/A (no train/test split, static artifact review)",
            "frozen_economic_gate": "N/A (reconciliation, not an alpha route)",
            "frozen_statistical_gate": "N/A (reconciliation, not an alpha route)",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "NO_NEW_TESTABLE_HYPOTHESIS",
            "mutation_test_count": 0, "mutation_test_passed": 1, "supersedes_experiment_id": None,
            "report_artifact_id": None, "schema_version": 7, "provenance": provenance,
            "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in results],
        results_schema_version=7, results_provenance=provenance, results_created_ms=now,
        no_test_split=True,
    )
    return {"n_reports_reconciled": len(RECONCILIATION), "conclusion": CONCLUSION}


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        result = freeze_and_record(conn)
        print(f"reconciled {result['n_reports_reconciled']} entry-timing reports; "
              f"W3 NOT forced open -- see CONCLUSION in experiment_results")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
