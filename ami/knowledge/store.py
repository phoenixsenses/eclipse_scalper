"""KnowledgeStore — governed SQLite knowledge graph + audit log + failure archive.

Part XIV §78: KNOWLEDGE GRAPH, FAILURE ARCHIVE, AUDIT LOG.
Never deletes: REJECT keeps the row with terminal status (constitution #7).
"""
from __future__ import annotations
import json, sqlite3, time
from pathlib import Path
from typing import Any, Iterable

from ami.constitution import ConstitutionViolation
from ami.enums import FailureType, KnowledgeStatus
from ami.knowledge.objects import KnowledgeObject

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "ami" / "knowledge.sqlite"

# [BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1 TEST-ISOLATION CLOSURE]
# Root cause this closes: unlike ami/warehouse/schema.py's `connect()`,
# `KnowledgeStore.__init__` had no test-isolation guard at all, AND its
# `path=DEFAULT_PATH` parameter default was bound at function-definition
# time (not re-read per call) -- so tests/conftest.py could never have
# redirected bare `KnowledgeStore()` calls even if it tried to. Discovered
# when this batch's `register_legacy_snapshot_with_gates()` (default
# `knowledge_db_path=None` -> resolves `ami.knowledge.store.DEFAULT_PATH` at
# CALL time, correctly) got invoked by several pre-existing
# `ami/research/w*.py` tests that open the REAL canonical.sqlite via the
# schema.py isolation copy but never told the knowledge side to redirect --
# the M-0033/M-0034 additive schema (2 new empty tables) landed on the REAL
# `data/ami/knowledge.sqlite` as an unintended side effect. Rolled back
# (tables dropped) before this closure was written; the fix below prevents
# recurrence for the whole test session, same discipline as
# tests/conftest.py's `_isolate_real_canonical_db` fixture.
REAL_KNOWLEDGE_PATH_IMMUTABLE = DEFAULT_PATH
_TEST_ISOLATION_ACTIVE = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    knowledge_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL, rel TEXT NOT NULL, dst TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY (src, rel, dst)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    knowledge_id TEXT,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS failure_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    idea TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    reason TEXT,
    data_period TEXT,
    regimes_tested TEXT,
    retry_condition TEXT,
    related TEXT
);
"""

# Compatibility relations (§21)
VALID_RELS = {"SUPPORTS", "CONTRADICTS", "DEPENDS_ON", "REQUIRES", "INVALIDATES",
              "RESTRICTS", "SUPERSEDES", "CAN_COEXIST_WITH", "CANNOT_COEXIST_WITH",
              "DERIVED_FROM"}


class KnowledgeStore:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = DEFAULT_PATH  # resolved at CALL time, not bound at def time -- see module note above
        self.path = Path(path)
        if _TEST_ISOLATION_ACTIVE and self.path.resolve() == Path(REAL_KNOWLEDGE_PATH_IMMUTABLE).resolve():
            raise RuntimeError(
                "TEST_ISOLATION_SAFETY_BLOCKER: refusing to open the REAL knowledge.sqlite "
                f"({REAL_KNOWLEDGE_PATH_IMMUTABLE}) during a test run. Tests must use a disposable "
                "copy -- see tests/conftest.py's session-scoped _isolate_real_knowledge_db fixture, "
                "which redirects ami.knowledge.store.DEFAULT_PATH automatically for any test that "
                "imports/passes it. If this fired from a test that hardcodes the real path as a "
                "literal string, fix the test to use DEFAULT_PATH instead.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ---- audit ----
    def _audit(self, actor: str, action: str, kid: str | None = None, detail: str = "") -> None:
        self.conn.execute("INSERT INTO audit_log (ts_ms, actor, action, knowledge_id, detail) VALUES (?,?,?,?,?)",
                          (int(time.time() * 1000), actor, action, kid, detail))

    # ---- knowledge CRUD (no silent delete) ----
    def put(self, ko: KnowledgeObject, actor: str = "ami") -> None:
        ko.validate()
        self.conn.execute(
            "INSERT INTO knowledge (knowledge_id, status, claim_type, version, payload, updated_ms) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(knowledge_id) DO UPDATE SET "
            "status=excluded.status, claim_type=excluded.claim_type, version=excluded.version, "
            "payload=excluded.payload, updated_ms=excluded.updated_ms",
            (ko.knowledge_id, ko.status.value, ko.claim_type.value, ko.version,
             json.dumps(ko.to_dict(), default=str), int(time.time() * 1000)))
        self._audit(actor, "PUT", ko.knowledge_id, f"status={ko.status.value} v{ko.version}")
        self.conn.commit()

    def get(self, knowledge_id: str) -> KnowledgeObject | None:
        r = self.conn.execute("SELECT payload FROM knowledge WHERE knowledge_id=?", (knowledge_id,)).fetchone()
        return KnowledgeObject.from_dict(json.loads(r[0])) if r else None

    def delete(self, knowledge_id: str) -> None:
        raise ConstitutionViolation("No failed idea is silently deleted.")

    def all(self, status: KnowledgeStatus | None = None) -> list[KnowledgeObject]:
        q = "SELECT payload FROM knowledge"
        args: tuple = ()
        if status:
            q += " WHERE status=?"; args = (status.value,)
        return [KnowledgeObject.from_dict(json.loads(r[0])) for r in self.conn.execute(q, args)]

    # ---- graph ----
    def link(self, src: str, rel: str, dst: str, actor: str = "ami") -> None:
        if rel not in VALID_RELS:
            raise ValueError(f"invalid relation {rel}")
        self.conn.execute("INSERT OR IGNORE INTO edges (src, rel, dst, created_ms) VALUES (?,?,?,?)",
                          (src, rel, dst, int(time.time() * 1000)))
        self._audit(actor, "LINK", src, f"{rel} -> {dst}")
        # contradiction edges must surface on the object itself (constitution #8)
        if rel == "CONTRADICTS":
            for kid, other in ((src, dst), (dst, src)):
                ko = self.get(kid)
                if ko and other not in ko.contradictions:
                    ko.contradictions.append(other)
                    self.put(ko, actor=actor)
        self.conn.commit()

    def edges(self, src: str | None = None, rel: str | None = None) -> list[tuple[str, str, str]]:
        q = "SELECT src, rel, dst FROM edges WHERE 1=1"; args: list = []
        if src: q += " AND src=?"; args.append(src)
        if rel: q += " AND rel=?"; args.append(rel)
        return [tuple(r) for r in self.conn.execute(q, args)]

    def conflicting_pairs(self) -> list[tuple[str, str]]:
        return [(s, d) for s, r, d in self.edges(rel="CONTRADICTS")]

    # ---- failure archive (§62) ----
    def archive_failure(self, idea: str, failure_type: FailureType, reason: str = "",
                        data_period: str = "", regimes: str = "", retry: str = "",
                        related: Iterable[str] = ()) -> None:
        self.conn.execute(
            "INSERT INTO failure_archive (ts_ms, idea, failure_type, reason, data_period, regimes_tested, retry_condition, related) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (int(time.time() * 1000), idea, failure_type.value, reason, data_period, regimes, retry,
             json.dumps(list(related))))
        self._audit("ami", "ARCHIVE_FAILURE", None, idea[:120])
        self.conn.commit()

    def failures(self) -> list[dict[str, Any]]:
        cols = ["id", "ts_ms", "idea", "failure_type", "reason", "data_period",
                "regimes_tested", "retry_condition", "related"]
        return [dict(zip(cols, r)) for r in self.conn.execute(
            "SELECT id, ts_ms, idea, failure_type, reason, data_period, regimes_tested, retry_condition, related FROM failure_archive")]

    def is_known_failure(self, idea_substr: str) -> bool:
        r = self.conn.execute("SELECT COUNT(*) FROM failure_archive WHERE idea LIKE ?",
                              (f"%{idea_substr}%",)).fetchone()
        return bool(r and r[0])

    def audit_tail(self, n: int = 20) -> list[tuple]:
        return list(self.conn.execute(
            "SELECT ts_ms, actor, action, knowledge_id, detail FROM audit_log ORDER BY id DESC LIMIT ?", (n,)))

    def close(self) -> None:
        self.conn.close()
