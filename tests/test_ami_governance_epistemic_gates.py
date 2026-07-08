"""BATCH-EPISTEMIC-NULLIFIER-GATES-V1 — tests for
ami/governance/epistemic_gates.py (graveyard slash-set gate + TEST-evidence
nullifier). Design transfer: validator-unlinkability P2 (slash-by-nullifier /
no-double-vote) mechanized as fail-closed research-governance gates.

DISPOSABLE_DB_ONLY: real knowledge.sqlite is only ever COPIED to tmp_path;
real canonical.sqlite is opened strictly mode=ro (retro-audit smoke).

Run: pytest tests/test_ami_governance_epistemic_gates.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import inspect
import shutil
import sqlite3

import pytest

from ami.governance import epistemic_gates as gates

REAL_KNOWLEDGE_PATH = "D:/eclipse_scalper/data/ami/knowledge.sqlite"
REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    gates.init_gates_schema(conn)
    return conn


def _seeded_conn():
    conn = _fresh_conn()
    # minimal failure_archive so retry_condition lookups resolve
    conn.execute("CREATE TABLE IF NOT EXISTS failure_archive (id INTEGER PRIMARY KEY,"
                 " ts_ms INTEGER, idea TEXT, failure_type TEXT, reason TEXT, data_period TEXT,"
                 " regimes_tested TEXT, retry_condition TEXT, related TEXT)")
    conn.execute("INSERT INTO failure_archive (id, ts_ms, idea, failure_type, retry_condition)"
                 " VALUES (4, 0, 'partial exits', 'NO_EDGE', 'none registered')")
    gates.seed_slash_fingerprints(conn)
    return conn


# ---------------------------------------------------------------------------
# structural guards
# ---------------------------------------------------------------------------

def test_not_called_automatically_no_module_level_connect():
    for fn in (gates.init_gates_schema, gates.seed_slash_fingerprints,
               gates.assert_not_graveyard, gates.consume_test_evidence,
               gates.retro_audit_experiment_registry):
        body = inspect.getsource(fn)
        assert "sqlite3.connect(" not in body


def test_schema_is_additive_and_idempotent():
    conn = _fresh_conn()
    gates.init_gates_schema(conn)  # second call must be a no-op
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"graveyard_slash_fingerprints", "epistemic_test_nullifiers"} <= tables


# ---------------------------------------------------------------------------
# nullifier derivation
# ---------------------------------------------------------------------------

def test_nullifier_deterministic_and_order_invariant():
    a = gates.derive_test_nullifier("FAM_X", "canonical-v1", ["c3", "c1", "c2"])
    b = gates.derive_test_nullifier("FAM_X", "canonical-v1", ["c1", "c2", "c3"])
    assert a == b


def test_nullifier_sensitive_to_family_split_and_set():
    base = gates.derive_test_nullifier("FAM_X", "canonical-v1", ["c1", "c2"])
    assert gates.derive_test_nullifier("FAM_Y", "canonical-v1", ["c1", "c2"]) != base
    assert gates.derive_test_nullifier("FAM_X", "canonical-v2", ["c1", "c2"]) != base
    assert gates.derive_test_nullifier("FAM_X", "canonical-v1", ["c1", "c2", "c3"]) != base


# ---------------------------------------------------------------------------
# consumption law
# ---------------------------------------------------------------------------

def test_first_consumption_then_idempotent_rerun():
    conn = _fresh_conn()
    kw = dict(family_id="FAM_X", split_version="canonical-v1",
              test_cycle_ids=["c1", "c2"], experiment_id="E-001")
    assert gates.consume_test_evidence(conn, **kw) == "CONSUMED"
    assert gates.consume_test_evidence(conn, **kw) == "NOOP_IDENTICAL"


def test_second_experiment_same_family_same_set_blocked():
    conn = _fresh_conn()
    gates.consume_test_evidence(conn, family_id="FAM_X", split_version="canonical-v1",
                                test_cycle_ids=["c1", "c2"], experiment_id="E-001")
    with pytest.raises(gates.TestEvidenceReuseBlocked):
        gates.consume_test_evidence(conn, family_id="FAM_X", split_version="canonical-v1",
                                    test_cycle_ids=["c1", "c2"], experiment_id="E-002")


def test_supersession_token_allows_and_records():
    conn = _fresh_conn()
    gates.consume_test_evidence(conn, family_id="FAM_X", split_version="canonical-v1",
                                test_cycle_ids=["c1", "c2"], experiment_id="E-001")
    r = gates.consume_test_evidence(conn, family_id="FAM_X", split_version="canonical-v1",
                                    test_cycle_ids=["c1", "c2"], experiment_id="E-002",
                                    supersession_token="OP-2026-07-06-corrected-rerun")
    assert r == "CONSUMED_WITH_SUPERSESSION"
    rows = conn.execute("SELECT consumed_by_experiment_id, supersession_token"
                        " FROM epistemic_test_nullifiers ORDER BY consumed_ms").fetchall()
    assert len(rows) == 2
    assert rows[1] == ("E-002", "OP-2026-07-06-corrected-rerun")


def test_forward_extended_set_is_a_new_nullifier_and_free():
    conn = _fresh_conn()
    gates.consume_test_evidence(conn, family_id="FAM_X", split_version="canonical-v1",
                                test_cycle_ids=["c1", "c2"], experiment_id="E-001")
    # genuinely new evidence (extra forward cycle) never needs a token
    assert gates.consume_test_evidence(
        conn, family_id="FAM_X", split_version="canonical-v1",
        test_cycle_ids=["c1", "c2", "c3"], experiment_id="E-002") == "CONSUMED"


def test_different_family_same_set_is_independent():
    conn = _fresh_conn()
    gates.consume_test_evidence(conn, family_id="FAM_X", split_version="canonical-v1",
                                test_cycle_ids=["c1", "c2"], experiment_id="E-001")
    assert gates.consume_test_evidence(
        conn, family_id="FAM_Y", split_version="canonical-v1",
        test_cycle_ids=["c1", "c2"], experiment_id="E-003") == "CONSUMED"


# ---------------------------------------------------------------------------
# graveyard gate
# ---------------------------------------------------------------------------

def test_graveyard_hit_blocks_and_names_family_and_retry_condition():
    conn = _seeded_conn()
    with pytest.raises(gates.GraveyardRetestBlocked) as e:
        gates.assert_not_graveyard(conn, "New idea: scale out via PARTIAL EXIT at +100bps")
    assert "partial exits" in str(e.value)
    assert "none registered" in str(e.value)  # retry_condition surfaced


def test_graveyard_hit_with_retry_token_passes_and_audits():
    conn = _seeded_conn()
    conn.execute("CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 " ts_ms INTEGER, actor TEXT, action TEXT, knowledge_id TEXT, detail TEXT)")
    r = gates.assert_not_graveyard(conn, "partial exit revisited",
                                   retry_token="OP-RETRY-2026-07-06")
    assert r["result"] == "PASSED_WITH_RETRY_TOKEN"
    n = conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='GRAVEYARD_RETRY_TOKEN_USED'").fetchone()[0]
    assert n == 1


def test_clean_spec_passes():
    conn = _seeded_conn()
    r = gates.assert_not_graveyard(
        conn, "pre-birth windowed taker flow stratification of frozen path outcomes")
    assert r["result"] == "CLEAN"


def test_matching_is_case_and_whitespace_insensitive():
    conn = _seeded_conn()
    hits = gates.match_graveyard(conn, "  Cross-Asset   TRANSFER  of SOL edge ")
    assert any(h["keyword"] == "cross-asset transfer" for h in hits)


def test_seed_is_idempotent():
    conn = _seeded_conn()
    before = conn.execute("SELECT COUNT(*) FROM graveyard_slash_fingerprints").fetchone()[0]
    r = gates.seed_slash_fingerprints(conn)
    assert r["inserted"] == 0
    assert r["total"] == before


# ---------------------------------------------------------------------------
# real-data smoke (disposable copy of knowledge.sqlite; canonical mode=ro)
# ---------------------------------------------------------------------------

def test_real_knowledge_disposable_seed_and_block(tmp_path):
    disposable = tmp_path / "knowledge_disposable.sqlite"
    shutil.copy2(REAL_KNOWLEDGE_PATH, disposable)
    conn = sqlite3.connect(disposable)
    gates.init_gates_schema(conn)
    r = gates.seed_slash_fingerprints(conn)
    assert r["total"] >= 25
    # the real archive's own family must block, with its real retry_condition surfaced
    with pytest.raises(gates.GraveyardRetestBlocked):
        gates.assert_not_graveyard(conn, "retry the BUY-SIDE FADE with new filters")
    # OFI-momentum (report-level graveyard) must also block
    with pytest.raises(gates.GraveyardRetestBlocked):
        gates.assert_not_graveyard(conn, "adaptive-quantile OFI momentum lead at 30s")
    conn.close()


def test_retro_audit_runs_read_only_on_real_canonical(tmp_path):
    disposable = tmp_path / "knowledge_disposable.sqlite"
    shutil.copy2(REAL_KNOWLEDGE_PATH, disposable)
    kconn = sqlite3.connect(disposable)
    gates.init_gates_schema(kconn)
    gates.seed_slash_fingerprints(kconn)
    canonical_ro = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        report = gates.retro_audit_experiment_registry(kconn, canonical_ro)
    finally:
        canonical_ro.close()
    assert len(report) >= 22  # every registered experiment audited (22 as of M-0030)
    for row in report:
        assert set(row) == {"experiment_id", "would_block", "hits"}
    kconn.close()
