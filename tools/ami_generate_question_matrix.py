"""Generate QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv (Phase 0, read-only audit artifact).

Sources:
- Whitepaper v0.3 Appendix O: family ranges Q001-Q866 (family-level canonical, NO verbatim text)
- Chart-Native Extension section 23: Q867-Q1058 verbatim question texts
- Reconstruction Protocol section 16: Q396-Q730 claimed explicit but NOT found in repo
  (see CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER.md CONFLICT-001)

Rule: do NOT fabricate missing canonical text -> text_status=MISSING_CANONICAL_TEXT.
Reproducible: rerun any time; output deterministic given the two source documents.
"""
import csv
import re
import sys
from datetime import date

CHART_DOC = "AMI_CHART_NATIVE_PRICE_STRUCTURE_INTELLIGENCE_EXTENSION_v1.0_COMPLETE.md"
OUT = "QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv"

# Appendix O family ranges (whitepaper v0.3). implementation_phase per MASTER_ROADMAP.md.
FAMILIES = [
    (1, 24, "BUYFADE_EVIDENCE_STRUCTURAL_THESIS", "WHITEPAPER_v0.3_AppendixO", 6),
    (25, 85, "PRE_EVENT_LONG_GENESIS_MATURITY", "WHITEPAPER_v0.3_AppendixO", 6),
    (86, 147, "SHORT_ENTRY_TIMING", "WHITEPAPER_v0.3_AppendixO", 6),
    (148, 175, "SILENCE_ONSET_MATURITY_BREAKDOWN", "WHITEPAPER_v0.3_AppendixO", 6),
    (176, 218, "SHORT_HORIZON_EXIT_MANAGEMENT", "WHITEPAPER_v0.3_AppendixO", 7),
    (219, 243, "STOP_TAXONOMY_SHORT_REENTRY", "WHITEPAPER_v0.3_AppendixO", 7),
    (244, 299, "LONG_SHORT_TRANSITIONS", "WHITEPAPER_v0.3_AppendixO", 7),
    (300, 335, "MULTI_TF_REGIME_OPPOSITE_LIQ", "WHITEPAPER_v0.3_AppendixO", 6),
    (336, 395, "REPLICATION_FORWARD_GOVERNANCE", "WHITEPAPER_v0.3_AppendixO", 2),
    (396, 534, "POSITION_AWARE_CYCLE_PATH_MECHANISM", "WHITEPAPER_v0.3_AppendixO", 3),
    (535, 730, "SIGNAL_AGING_CLOCK_ROUTE_HOLD_EXECUTION", "WHITEPAPER_v0.3_AppendixO", 7),
    (731, 866, "EVIDENCE_INDEPENDENCE_CAUSAL_OOD_META", "WHITEPAPER_v0.3_AppendixO", 2),
]

COLS = [
    "question_id", "canonical_parent", "question_family", "source_document",
    "question_text", "text_status", "current_status", "existing_evidence",
    "evidence_layer", "required_data", "required_features", "experiment_engine",
    "minimum_event_n", "minimum_cycle_n", "minimum_day_n", "required_regimes",
    "required_sessions", "required_forward_duration", "historical_testable",
    "forward_required", "blocked_by", "implementation_phase", "retry_condition",
    "permission_ceiling", "final_verdict", "last_updated",
]

today = date.today().isoformat()


def base_row(qid, family, source, phase):
    return {
        "question_id": f"Q{qid:03d}",
        "canonical_parent": f"FAM_{family}",
        "question_family": family,
        "source_document": source,
        "question_text": "",
        "text_status": "MISSING_CANONICAL_TEXT",
        "current_status": "FUTURE_PHASE",
        "existing_evidence": "",
        "evidence_layer": "",
        "required_data": "TBD_AT_FAMILY_TRIAGE",
        "required_features": "TBD_AT_FAMILY_TRIAGE",
        "experiment_engine": "TBD_AT_FAMILY_TRIAGE",
        "minimum_event_n": "",
        "minimum_cycle_n": "",
        "minimum_day_n": "",
        "required_regimes": "",
        "required_sessions": "",
        "required_forward_duration": "",
        "historical_testable": "UNKNOWN",
        "forward_required": "UNKNOWN",
        "blocked_by": "FAMILY_TRIAGE_PENDING",
        "implementation_phase": str(phase),
        "retry_condition": "",
        "permission_ceiling": "RESEARCH_ONLY",
        "final_verdict": "",
        "last_updated": today,
    }


def parse_chart_native():
    text = open(CHART_DOC, encoding="utf-8").read()
    fam_pat = re.compile(r"^## Q(\d+)–Q(\d+) — (.+)$", re.M)
    q_pat = re.compile(r"\*\*Q(\d+)\.\*\*\s*(.+?)(?:\s{2,}|\n|$)")
    fams = [(int(m.group(1)), int(m.group(2)), m.group(3).strip(), m.start())
            for m in fam_pat.finditer(text)]
    questions = {}
    for i, (lo, hi, title, start) in enumerate(fams):
        end = fams[i + 1][3] if i + 1 < len(fams) else len(text)
        block = text[start:end]
        fam_slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").upper()
        for m in q_pat.finditer(block):
            qid = int(m.group(1))
            if lo <= qid <= hi:
                questions[qid] = (fam_slug, m.group(2).strip(), lo)
    return questions


def main():
    chart_q = parse_chart_native()
    rows = []
    for lo, hi, fam, src, phase in FAMILIES:
        for qid in range(lo, hi + 1):
            rows.append(base_row(qid, fam, src, phase))
    for qid in range(867, 1059):
        if qid in chart_q:
            fam, qtext, lo = chart_q[qid]
            r = base_row(qid, fam, "CHART_NATIVE_EXT_v1.0_sec23", 4)
            r["canonical_parent"] = f"FAM_CHART_{fam}"
            r["question_text"] = qtext
            r["text_status"] = "CANONICAL_TEXT_PRESENT_PROPOSED"
            r["current_status"] = "FUTURE_PHASE"
            r["blocked_by"] = "PHASE4_CHART_OBJECT_FOUNDATION"
            r["historical_testable"] = "PARTIAL"
            r["forward_required"] = "TRUE_FOR_OBSERVER_FAMILIES"
        else:
            r = base_row(qid, "CHART_NATIVE_UNRESOLVED", "CHART_NATIVE_EXT_v1.0_sec23", 4)
        rows.append(r)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    n_text = sum(1 for r in rows if r["text_status"].startswith("CANONICAL_TEXT"))
    n_missing = sum(1 for r in rows if r["text_status"] == "MISSING_CANONICAL_TEXT")
    print(f"rows={len(rows)} with_text={n_text} missing_text={n_missing}")
    assert len(rows) == 1058, len(rows)


if __name__ == "__main__":
    sys.exit(main())
