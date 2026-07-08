"""BATCH-P1-002: read-only artifact discovery -> artifact_registry ingest.

Scope (deliberately bounded — full repo-wide discovery of research reports/code
is later work, not this batch; see docstring bottom for rationale):
  - root-level canonical *.md documents
  - docs/ami/**  (roadmap, changelog, decision records, schema/state docs)
  - docs/protocols/** (master execution protocol)

Reads files only. Writes only to data/ami/canonical.sqlite (artifact_registry,
namespace_registry). Touches no other store and no running process.

AMI acronym-collision gate (Reconstruction Protocol §2.2): this repo's "AMI"
means Artificial Market Intelligence. Files are scanned for competing meanings
(e.g. "Advanced Metering Infrastructure") near the acronym; any hit is
quarantined into namespace_registry with is_ami_artificial_market_intelligence=0
and excluded from canonical_status=CANONICAL classification.
"""
from __future__ import annotations
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_COLLISION_PATTERNS = re.compile(
    r"advanced\s+metering\s+infrastructure|metering\s+infrastructure|smart\s+meter",
    re.IGNORECASE,
)
_AMI_SELF_ID_PATTERN = re.compile(
    r"artificial\s+market\s+intelligence|\bS34\b|AMI\s*[×x]\s*S34|AMI-S34", re.IGNORECASE
)

# Files whose canonical status cannot be auto-resolved by this batch because an
# operator decision is pending (CONFLICT-002 / OD-002 — whitepaper v0.2 vs v0.3).
_UNDER_RECONCILIATION = {
    "AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md",
    "AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE.md",
}

_ROOT_MD_ROLE = {
    "SYSTEM_STATE.md": "state_doc",
    "CLAUDE.md": "operating_doctrine",
}


@dataclass
class ArtifactRecord:
    artifact_id: str
    path: str
    content_hash: str
    role: str
    canonical_status: str
    namespace: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _classify_role(rel_path: str) -> str:
    if rel_path in _ROOT_MD_ROLE:
        return _ROOT_MD_ROLE[rel_path]
    if rel_path.startswith("docs/protocols/"):
        return "master_protocol"
    if rel_path.startswith("docs/ami/AMI_DECISION_RECORDS/"):
        return "decision_record"
    if "CHANGELOG" in rel_path:
        return "changelog"
    if "ROADMAP" in rel_path:
        return "roadmap"
    if rel_path.startswith("docs/ami/"):
        return "ami_doc"
    if rel_path.count("/") == 0 and rel_path.endswith(".md"):
        return "canonical_spec_or_audit_artifact"
    return "other"


def _classify_canonical_status(rel_path: str, is_collision: bool) -> str:
    if is_collision:
        return "NAMESPACE_QUARANTINED"
    name = Path(rel_path).name
    if name in _UNDER_RECONCILIATION:
        return "UNDER_RECONCILIATION"
    return "CANONICAL"


def detect_ami_collision(text: str) -> bool:
    """True if text uses 'AMI' in a sense other than Artificial Market Intelligence.

    A document that mentions the metering-infrastructure phrase while also
    self-identifying as Artificial Market Intelligence is treated as a
    self-aware discussion of the collision (e.g. Reconstruction Protocol
    §2.2/15.2/18, which instructs to quarantine a *separate* external
    document — not itself a collision) rather than an actual namespace hit.
    """
    if not _COLLISION_PATTERNS.search(text):
        return False
    return not _AMI_SELF_ID_PATTERN.search(text)


def discover(repo_root: Path = REPO_ROOT) -> list[ArtifactRecord]:
    roots = [
        (repo_root, "*.md", False),          # root-level only, non-recursive
        (repo_root / "docs" / "ami", "*.md", True),
        (repo_root / "docs" / "protocols", "*.md", True),
    ]
    seen: set[str] = set()
    records: list[ArtifactRecord] = []
    for base, pattern, recursive in roots:
        if not base.exists():
            continue
        it = base.rglob(pattern) if recursive else base.glob(pattern)
        for path in sorted(it):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            text = path.read_text(encoding="utf-8", errors="replace")
            collision = detect_ami_collision(text)
            records.append(
                ArtifactRecord(
                    artifact_id=rel,
                    path=rel,
                    content_hash=_sha256(path),
                    role=_classify_role(rel),
                    canonical_status=_classify_canonical_status(rel, collision),
                    namespace="ami_s34",
                )
            )
    return records


def ingest(conn, records: list[ArtifactRecord], provenance: str = "batch-p1-002-artifact-discovery") -> int:
    """Idempotent upsert by artifact_id (= repo-relative path, stable across runs)."""
    now = int(time.time() * 1000)
    n = 0
    for r in records:
        conn.execute(
            "INSERT INTO artifact_registry (artifact_id, path, content_hash, role, canonical_status, "
            "namespace, schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(artifact_id) DO UPDATE SET content_hash=excluded.content_hash, "
            "role=excluded.role, canonical_status=excluded.canonical_status, updated_ms=excluded.updated_ms",
            (r.artifact_id, r.path, r.content_hash, r.role, r.canonical_status,
             r.namespace, 1, provenance, now, now),
        )
        conn.execute(
            "INSERT INTO namespace_registry (namespace, meaning, is_ami_artificial_market_intelligence, "
            "schema_version, provenance, created_ms) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(namespace) DO NOTHING",
            (r.namespace, "Artificial Market Intelligence (AMI) x S34 research programme", 1, 1, provenance, now),
        )
        n += 1
    conn.commit()
    return n


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        records = discover()
        n = ingest(conn, records)
        quarantined = [r for r in records if r.canonical_status == "NAMESPACE_QUARANTINED"]
        print(f"ingested {n} artifacts; namespace_quarantined={len(quarantined)}")
        for r in quarantined:
            print("  QUARANTINED:", r.path)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
