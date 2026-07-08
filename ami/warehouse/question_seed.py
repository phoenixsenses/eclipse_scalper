"""BATCH-P1-003: question registry seed.

Sources (read-only):
  - QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv (Phase 0 artifact; 1058 rows)
  - data/ami/research.sqlite `questions` table (14 existing slug-ID S34 research
    questions — a distinct ID namespace, not renumbered/merged into Q001-Q1058)

Writes only to data/ami/canonical.sqlite (question_families, question_registry).
No fabrication: rows whose canonical text is unknown keep text_status =
MISSING_CANONICAL_TEXT (carried over verbatim from the CSV).
"""
from __future__ import annotations
import csv
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv"
DEFAULT_RESEARCH_DB = REPO_ROOT / "data" / "ami" / "research.sqlite"

LEGACY_SLUG_FAMILY_ID = "FAM_S34_LEGACY_SLUG_QUESTIONS"


def _read_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def derive_families(rows: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    q_ids: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        fam = row["question_family"]
        qid = int(row["question_id"][1:])
        q_ids[fam].append(qid)
        groups.setdefault(fam, {
            "family_id": row["canonical_parent"],
            "title": fam,
            "source_document": row["source_document"],
            "implementation_phase": row["implementation_phase"],
        })
    families = []
    for fam, meta in groups.items():
        ids = q_ids[fam]
        families.append({
            "family_id": meta["family_id"],
            "title": meta["title"],
            "q_range_lo": min(ids),
            "q_range_hi": max(ids),
            "source_document": meta["source_document"],
            "implementation_phase": meta["implementation_phase"],
        })
    families.append({
        "family_id": LEGACY_SLUG_FAMILY_ID,
        "title": "S34_LEGACY_SLUG_QUESTIONS",
        "q_range_lo": None,
        "q_range_hi": None,
        "source_document": "data/ami/research.sqlite:questions",
        "implementation_phase": None,
    })
    return families


def load_legacy_slug_questions(research_db: Path = DEFAULT_RESEARCH_DB) -> list[dict]:
    if not research_db.exists():
        return []
    conn = sqlite3.connect(f"file:{research_db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT question_id, payload, status FROM questions").fetchall()
    finally:
        conn.close()
    out = []
    for qid, payload_json, status in rows:
        payload = json.loads(payload_json)
        out.append({
            "question_id": qid,
            "canonical_parent": LEGACY_SLUG_FAMILY_ID,
            "family_id": LEGACY_SLUG_FAMILY_ID,
            "question_text": payload.get("question", ""),
            "text_status": "CANONICAL_TEXT_PRESENT_LEGACY_SLUG",
            "current_status": status,
            "evidence_layer": "",
            "blocked_by": "",
            "implementation_phase": None,
            "permission_ceiling": "RESEARCH_ONLY",
            "final_verdict": "",
        })
    return out


def seed(conn, csv_path: Path = DEFAULT_CSV, research_db: Path = DEFAULT_RESEARCH_DB,
          provenance: str = "batch-p1-003-question-seed") -> tuple[int, int]:
    now = int(time.time() * 1000)
    rows = _read_csv_rows(csv_path)
    families = derive_families(rows)

    for fam in families:
        conn.execute(
            "INSERT INTO question_families (family_id, title, q_range_lo, q_range_hi, source_document, "
            "implementation_phase, schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(family_id) DO UPDATE SET title=excluded.title, q_range_lo=excluded.q_range_lo, "
            "q_range_hi=excluded.q_range_hi, updated_ms=excluded.updated_ms",
            (fam["family_id"], fam["title"], fam["q_range_lo"], fam["q_range_hi"],
             fam["source_document"], fam["implementation_phase"], 1, provenance, now, now),
        )

    n_q = 0
    for row in rows:
        conn.execute(
            "INSERT INTO question_registry (question_id, canonical_parent, family_id, question_text, "
            "text_status, current_status, evidence_layer, blocked_by, implementation_phase, "
            "permission_ceiling, final_verdict, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(question_id) DO UPDATE SET family_id=excluded.family_id, "
            "question_text=excluded.question_text, text_status=excluded.text_status, "
            "current_status=excluded.current_status, updated_ms=excluded.updated_ms",
            # FABLE-REVIEW-A follow-up (found by enabling FK enforcement, F4):
            # family_id must be canonical_parent ("FAM_..." prefixed), matching
            # question_families.family_id -- question_family (unprefixed) was a
            # latent FK mismatch, silently unenforced before FK pragma was on.
            # family_id is in the UPDATE SET list so a re-run repairs
            # already-stored rows, not just fresh inserts.
            (row["question_id"], row["canonical_parent"], row["canonical_parent"], row["question_text"],
             row["text_status"], row["current_status"], row["existing_evidence"], row["blocked_by"],
             row["implementation_phase"] or None, row["permission_ceiling"], row["final_verdict"],
             1, provenance, now, now),
        )
        n_q += 1

    legacy = load_legacy_slug_questions(research_db)
    for q in legacy:
        conn.execute(
            "INSERT INTO question_registry (question_id, canonical_parent, family_id, question_text, "
            "text_status, current_status, evidence_layer, blocked_by, implementation_phase, "
            "permission_ceiling, final_verdict, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(question_id) DO UPDATE SET question_text=excluded.question_text, "
            "current_status=excluded.current_status, updated_ms=excluded.updated_ms",
            (q["question_id"], q["canonical_parent"], q["family_id"], q["question_text"],
             q["text_status"], q["current_status"], q["evidence_layer"], q["blocked_by"],
             q["implementation_phase"], q["permission_ceiling"], q["final_verdict"],
             1, provenance, now, now),
        )
        n_q += 1

    conn.commit()
    return len(families), n_q


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n_fam, n_q = seed(conn)
        print(f"seeded {n_fam} families, {n_q} questions")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
