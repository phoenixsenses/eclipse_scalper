"""BATCH-P1-004: contradiction_registry + operator_decision_queue + artifact_lineage seed.

Sources (read-only, repo-root Phase 0 artifacts, already reconciled by a human
+ Fable 5 in Phase 0 — this batch loads them into the warehouse verbatim, it
does not re-derive or re-judge them):
  - CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER.md (CONFLICT-001..010, document-level)
  - CONTRADICTION_REGISTER.md (CT-001..004, evidence-level)
  - OPERATOR_DECISION_QUEUE.md (OD-001..010)

artifact_lineage is seeded only with relations that are already explicitly
recorded elsewhere (no new supersession judgement is made here — full
canonical-selection/duplicate-detection is later Protocol §5 work):
  - whitepaper v0.2 <-> v0.3_COMPLETE: relation=UNDER_RECONCILIATION (CONFLICT-002)

Writes only to data/ami/canonical.sqlite. No existing store or process touched.
"""
from __future__ import annotations
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFLICT_REGISTER_MD = REPO_ROOT / "CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER.md"
CONTRADICTION_REGISTER_MD = REPO_ROOT / "CONTRADICTION_REGISTER.md"
OPERATOR_DECISION_QUEUE_MD = REPO_ROOT / "OPERATOR_DECISION_QUEUE.md"

_CONFLICT_HEADER = re.compile(
    r"^### (CONFLICT-\d+) — (.+?) · \*{0,2}SEVERITY: (\w+)\*{0,2} · STATUS: ([A-Z_]+)(.*)$", re.M
)


def parse_conflict_register(path: Path = CONFLICT_REGISTER_MD) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    matches = list(_CONFLICT_HEADER.finditer(text))
    out = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        out.append({
            "contradiction_id": m.group(1),
            "left_ref": "canonical_documents",
            "right_ref": m.group(2).strip(),
            "description": f"[DOCUMENT-LEVEL] {body[:2000]}",
            "status": m.group(4).strip() + m.group(5).strip(),
            "resolution": body,
        })
    return out


def parse_contradiction_register(path: Path = CONTRADICTION_REGISTER_MD) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("| CT-"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        cid, conflict, parties, status = cells[0], cells[1], cells[2], cells[3]
        out.append({
            "contradiction_id": cid,
            "left_ref": "evidence_artifacts",
            "right_ref": parties,
            "description": f"[EVIDENCE-LEVEL] {conflict}",
            "status": status,
            "resolution": status,
        })
    return out


def parse_operator_decision_queue(path: Path = OPERATOR_DECISION_QUEUE_MD) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("| OD-"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        did, question, evidence, recommendation, risk, status = cells[:6]
        out.append({
            "decision_id": did,
            "question": question,
            "current_evidence": evidence,
            "recommended_options": recommendation,
            "risk_of_each_option": risk,
            "required_approval": "OPERATOR",
            "status": status,
        })
    return out


def derive_artifact_lineage() -> list[dict]:
    return [{
        "artifact_id": "AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE.md",
        "predecessor_id": "AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md",
        "relation": "UNDER_RECONCILIATION",
    }]


def seed(conn, provenance: str = "batch-p1-004-registry-seed") -> tuple[int, int, int]:
    now = int(time.time() * 1000)

    contradictions = parse_conflict_register() + parse_contradiction_register()
    for c in contradictions:
        conn.execute(
            "INSERT INTO contradiction_registry (contradiction_id, left_ref, right_ref, description, "
            "status, resolution, schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(contradiction_id) DO UPDATE SET description=excluded.description, "
            "status=excluded.status, resolution=excluded.resolution, updated_ms=excluded.updated_ms",
            (c["contradiction_id"], c["left_ref"], c["right_ref"], c["description"],
             c["status"], c["resolution"], 1, provenance, now, now),
        )

    decisions = parse_operator_decision_queue()
    for d in decisions:
        conn.execute(
            "INSERT INTO operator_decision_queue (decision_id, question, current_evidence, "
            "recommended_options, risk_of_each_option, required_approval, status, operator_response, "
            "schema_version, provenance, created_ms, implemented_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(decision_id) DO UPDATE SET question=excluded.question, status=excluded.status, "
            "current_evidence=excluded.current_evidence",
            (d["decision_id"], d["question"], d["current_evidence"], d["recommended_options"],
             d["risk_of_each_option"], d["required_approval"], d["status"], None, 1, provenance, now, None),
        )

    lineage = derive_artifact_lineage()
    for l in lineage:
        exists = conn.execute(
            "SELECT 1 FROM artifact_lineage WHERE artifact_id=? AND predecessor_id=? AND relation=?",
            (l["artifact_id"], l["predecessor_id"], l["relation"]),
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO artifact_lineage (artifact_id, predecessor_id, relation, schema_version, "
                "provenance, created_ms) VALUES (?,?,?,?,?,?)",
                (l["artifact_id"], l["predecessor_id"], l["relation"], 1, provenance, now),
            )

    conn.commit()
    return len(contradictions), len(decisions), len(lineage)


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n_c, n_d, n_l = seed(conn)
        print(f"seeded {n_c} contradictions, {n_d} operator decisions, {n_l} lineage rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
