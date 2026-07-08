"""BATCH-EPISTEMIC-NULLIFIER-ENFORCEMENT-WIRING-V1 (M-0033) -- tests for
`ami/warehouse/experiment_ledger.py:register_experiment_with_gates()`, the
new mandatory gated entry point that wires BATCH-EPISTEMIC-NULLIFIER-GATES-V1
(graveyard slash-set + TEST-evidence nullifier) into the experiment
registration path, plus the new authorization-token model and family/split
identity adapters added to `ami/governance/epistemic_gates.py`.

DISPOSABLE_DB_ONLY: canonical-like and knowledge-like databases used here are
always fresh tmp_path files (canonical schema built via
`ami.warehouse.schema.init_schema` on a throwaway path, never the real one).
Real canonical.sqlite/knowledge.sqlite are opened strictly mode=ro / via a
disposable copy (real-data smoke section only), matching repo convention
(tests/conftest.py session-scoped isolation + BATCH-EPISTEMIC-NULLIFIER-
GATES-V1's own test file).

Run: pytest tests/test_ami_epistemic_nullifier_enforcement_wiring.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import hashlib
import inspect
import shutil
import sqlite3
from unittest.mock import patch

import pytest

import ami.warehouse.schema as schema_mod
from ami.constitution import ConstitutionViolation
from ami.governance import epistemic_gates as gates
from ami.knowledge.store import KnowledgeStore
from ami.warehouse import experiment_ledger as ledger

REAL_KNOWLEDGE_PATH = "D:/eclipse_scalper/data/ami/knowledge.sqlite"
REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_CANONICAL_SHA256 = "0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def canonical_conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "canonical_test.sqlite"))
    schema_mod.init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture()
def knowledge_path(tmp_path):
    path = tmp_path / "knowledge_test.sqlite"
    KnowledgeStore(path).close()  # pre-creates audit_log/failure_archive/knowledge/edges
    kconn = sqlite3.connect(str(path))
    gates.init_gates_schema(kconn)
    gates.seed_slash_fingerprints(kconn)
    kconn.close()
    return path


def _registry_values(experiment_id, *, hypothesis_id="H-TEST-CLEAN", question_ids="FAM_TEST_CLEAN",
                      frozen_splits="chronological 70/30 by signal_birth_ts, never randomized",
                      frozen_population="pop-v1", supersedes_experiment_id=None) -> dict:
    now = 1_800_000_000_000
    return {
        "experiment_id": experiment_id, "question_ids": question_ids, "hypothesis_id": hypothesis_id,
        "preregistered_at": now, "frozen_population": frozen_population, "frozen_features": "f1,f2",
        "frozen_target": "target-v1", "frozen_thresholds": "MIN_N=20",
        "frozen_splits": frozen_splits, "frozen_economic_gate": "N/A", "frozen_statistical_gate": "bootstrap",
        "code_commit": None, "dataset_hash": "hash-v1", "started_at": now, "completed_at": now,
        "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
        "mutation_test_count": 0, "mutation_test_passed": 1,
        "supersedes_experiment_id": supersedes_experiment_id, "report_artifact_id": None,
        "schema_version": 12, "provenance": "test-enforcement-wiring", "created_ms": now, "updated_ms": now,
    }


def _register(canonical_conn, knowledge_path, experiment_id, **kwargs):
    defaults = dict(
        registry_values=_registry_values(experiment_id, **{
            k: v for k, v in kwargs.items()
            if k in ("hypothesis_id", "question_ids", "frozen_splits", "frozen_population",
                      "supersedes_experiment_id")
        }),
        results=[("metric_a", "1.0")],
        results_schema_version=12, results_provenance="test", results_created_ms=1_800_000_000_000,
        test_cycle_ids=["c1", "c2", "c3"],
        knowledge_db_path=str(knowledge_path),
    )
    for k in ("hypothesis_id", "question_ids", "frozen_splits", "frozen_population",
              "supersedes_experiment_id"):
        kwargs.pop(k, None)
    defaults.update(kwargs)
    return ledger.register_experiment_with_gates(canonical_conn, **defaults)


def _registry_count(canonical_conn, experiment_id) -> int:
    return canonical_conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (experiment_id,)).fetchone()[0]


def _results_count(canonical_conn, experiment_id) -> int:
    return canonical_conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (experiment_id,)).fetchone()[0]


def _audit_actions(knowledge_path) -> list[str]:
    conn = sqlite3.connect(str(knowledge_path))
    try:
        return [r[0] for r in conn.execute("SELECT action FROM audit_log ORDER BY ts_ms")]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. Clean new family, first TEST use — allowed
# ---------------------------------------------------------------------------

def test_01_clean_new_family_first_test_use_allowed(canonical_conn, knowledge_path):
    result = _register(canonical_conn, knowledge_path, "E-001")
    assert result["registry_result"] == "INSERTED"
    assert result["nullifier_result"] == "CONSUMED"
    assert _registry_count(canonical_conn, "E-001") == 1
    assert _results_count(canonical_conn, "E-001") == 1


# ---------------------------------------------------------------------------
# 2. Graveyard family without retry token — blocked
# ---------------------------------------------------------------------------

def test_02_graveyard_family_without_retry_token_blocked(canonical_conn, knowledge_path):
    with pytest.raises(gates.GraveyardRetestBlocked):
        _register(canonical_conn, knowledge_path, "E-002", hypothesis_id="H-BUYFADE-RETEST")
    assert _registry_count(canonical_conn, "E-002") == 0
    assert _results_count(canonical_conn, "E-002") == 0


# ---------------------------------------------------------------------------
# 3. Graveyard family with invalid token — blocked
# ---------------------------------------------------------------------------

def test_03_graveyard_family_with_invalid_token_blocked(canonical_conn, knowledge_path):
    with pytest.raises(gates.AuthorizationInvalid):
        _register(canonical_conn, knowledge_path, "E-003", hypothesis_id="H-BUYFADE-RETEST",
                   retry_token="not-a-real-issued-token")
    assert _registry_count(canonical_conn, "E-003") == 0


# ---------------------------------------------------------------------------
# 4. Graveyard family with valid scoped token — allowed and audited
# ---------------------------------------------------------------------------

def test_04_graveyard_family_with_valid_token_allowed_and_audited(canonical_conn, knowledge_path):
    family_id = gates.resolve_canonical_family_id("FAM_TEST_04", "H-BUYFADE-RETEST-04")
    kconn = sqlite3.connect(str(knowledge_path))
    gates.init_gates_schema(kconn)
    token = gates.issue_retry_authorization(
        kconn, canonical_family_id=family_id, approver="operator", justification="retry condition met",
        related_experiment_id="E-BUYFADE-ORIGINAL-001", retry_condition_satisfied="condition satisfied text")
    kconn.close()

    result = _register(canonical_conn, knowledge_path, "E-004", hypothesis_id="H-BUYFADE-RETEST-04",
                        question_ids="FAM_TEST_04", retry_token=token)
    assert result["registry_result"] == "INSERTED"
    actions = _audit_actions(knowledge_path)
    assert "GRAVEYARD_RETRY_TOKEN_USED" in actions
    assert "AUTHORIZATION_CONSUMED" in actions

    kconn = sqlite3.connect(str(knowledge_path))
    row = kconn.execute(
        "SELECT consumed, resulting_experiment_id FROM epistemic_authorization_tokens"
        " WHERE canonical_family_id=?", (family_id,)).fetchone()
    kconn.close()
    assert row == (1, "E-004")


# ---------------------------------------------------------------------------
# 5. Retry token reuse — blocked
# ---------------------------------------------------------------------------

def test_05_retry_token_reuse_blocked(canonical_conn, knowledge_path):
    family_id = gates.resolve_canonical_family_id("FAM_TEST_05", "H-BUYFADE-RETEST-05")
    kconn = sqlite3.connect(str(knowledge_path))
    gates.init_gates_schema(kconn)
    token = gates.issue_retry_authorization(
        kconn, canonical_family_id=family_id, approver="operator", justification="j",
        related_experiment_id="E-ORIG", retry_condition_satisfied="met")
    kconn.close()

    _register(canonical_conn, knowledge_path, "E-005a", hypothesis_id="H-BUYFADE-RETEST-05",
              question_ids="FAM_TEST_05", retry_token=token, test_cycle_ids=["c1", "c2"])
    with pytest.raises(gates.AuthorizationInvalid):
        _register(canonical_conn, knowledge_path, "E-005b", hypothesis_id="H-BUYFADE-RETEST-05",
                   question_ids="FAM_TEST_05", retry_token=token, test_cycle_ids=["c3", "c4"])
    assert _registry_count(canonical_conn, "E-005b") == 0


# ---------------------------------------------------------------------------
# 6. Same frozen experiment rerun — NOOP_IDENTICAL
# ---------------------------------------------------------------------------

def test_06_same_frozen_experiment_rerun_is_noop_identical(canonical_conn, knowledge_path):
    r1 = _register(canonical_conn, knowledge_path, "E-006")
    r2 = _register(canonical_conn, knowledge_path, "E-006")
    assert r1["registry_result"] == "INSERTED"
    assert r2["registry_result"] == "NOOP_IDENTICAL"
    assert r2["nullifier_result"] == "NOOP_IDENTICAL"
    assert _registry_count(canonical_conn, "E-006") == 1


# ---------------------------------------------------------------------------
# 7. Same family/split/TEST set with new experiment ID — blocked
# ---------------------------------------------------------------------------

def test_07_same_family_split_test_set_new_experiment_id_blocked(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-007a", hypothesis_id="H-007", question_ids="FAM-007",
              test_cycle_ids=["c1", "c2"])
    with pytest.raises(gates.TestEvidenceReuseBlocked):
        _register(canonical_conn, knowledge_path, "E-007b", hypothesis_id="H-007", question_ids="FAM-007",
                  test_cycle_ids=["c1", "c2"])
    assert _registry_count(canonical_conn, "E-007b") == 0
    assert _results_count(canonical_conn, "E-007b") == 0


# ---------------------------------------------------------------------------
# 8. Same family with forward-expanded TEST set — new nullifier, allowed
# ---------------------------------------------------------------------------

def test_08_forward_expanded_test_set_new_nullifier_allowed(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-008a", hypothesis_id="H-008", question_ids="FAM-008",
              test_cycle_ids=["c1", "c2"])
    r2 = _register(canonical_conn, knowledge_path, "E-008b", hypothesis_id="H-008", question_ids="FAM-008",
                    test_cycle_ids=["c1", "c2", "c3"])
    assert r2["registry_result"] == "INSERTED"
    assert r2["nullifier_result"] == "CONSUMED"
    assert r2["nullifier"] != gates.derive_test_nullifier(
        gates.resolve_canonical_family_id("FAM-008", "H-008"),
        gates.resolve_split_version("chronological 70/30 by signal_birth_ts, never randomized"),
        ["c1", "c2"])


# ---------------------------------------------------------------------------
# 9. Same cycles in different input order — same normalized nullifier
# ---------------------------------------------------------------------------

def test_09_same_cycles_different_order_same_nullifier():
    fam, split = "FAMv1:x", "SPLITv1:y"
    n1 = gates.derive_test_nullifier(fam, split, ["c3", "c1", "c2"])
    n2 = gates.derive_test_nullifier(fam, split, ["c1", "c2", "c3"])
    assert n1 == n2


def test_09b_same_cycles_different_order_rerun_is_noop(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-009", hypothesis_id="H-009", question_ids="FAM-009",
              test_cycle_ids=["c1", "c2", "c3"])
    r2 = _register(canonical_conn, knowledge_path, "E-009", hypothesis_id="H-009", question_ids="FAM-009",
                    test_cycle_ids=["c3", "c2", "c1"])
    assert r2["nullifier_result"] == "NOOP_IDENTICAL"


# ---------------------------------------------------------------------------
# 10. Duplicate TEST cycle IDs — blocked
# ---------------------------------------------------------------------------

def test_10_duplicate_test_cycle_ids_blocked(canonical_conn, knowledge_path):
    with pytest.raises(ValueError):
        _register(canonical_conn, knowledge_path, "E-010", test_cycle_ids=["c1", "c1", "c2"])
    assert _registry_count(canonical_conn, "E-010") == 0


# ---------------------------------------------------------------------------
# 11. TRAIN-cycle contamination in TEST set — blocked
# ---------------------------------------------------------------------------

def test_11_train_cycle_contamination_in_test_set_blocked(canonical_conn, knowledge_path):
    with pytest.raises(ConstitutionViolation):
        _register(canonical_conn, knowledge_path, "E-011", test_cycle_ids=["c1", "c2"],
                  train_cycle_ids=["c0", "c2"])
    assert _registry_count(canonical_conn, "E-011") == 0


# ---------------------------------------------------------------------------
# 12. Family alias/rename resolving to same canonical family — blocked from reuse
# ---------------------------------------------------------------------------

def test_12_family_alias_case_whitespace_resolves_same_family_blocked(canonical_conn, knowledge_path):
    assert gates.resolve_canonical_family_id("FAM-012", "H-012") == \
        gates.resolve_canonical_family_id("  fam-012  ", "  h-012  ")
    _register(canonical_conn, knowledge_path, "E-012a", hypothesis_id="H-012", question_ids="FAM-012",
              test_cycle_ids=["c1", "c2"])
    with pytest.raises(gates.TestEvidenceReuseBlocked):
        _register(canonical_conn, knowledge_path, "E-012b", hypothesis_id="  H-012  ",
                  question_ids="fam-012", test_cycle_ids=["c1", "c2"])


# ---------------------------------------------------------------------------
# 13. Valid supersession authorization — allowed and audited
# ---------------------------------------------------------------------------

def test_13_valid_supersession_authorization_allowed_and_audited(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-013a", hypothesis_id="H-013", question_ids="FAM-013",
              test_cycle_ids=["c1", "c2"])
    family_id = gates.resolve_canonical_family_id("FAM-013", "H-013")
    split_version = gates.resolve_split_version("chronological 70/30 by signal_birth_ts, never randomized")
    _, test_set_hash, _ = gates._normalized_test_set_hash(["c1", "c2"])  # noqa: SLF001
    nullifier = gates.derive_test_nullifier(family_id, split_version, ["c1", "c2"])

    kconn = sqlite3.connect(str(knowledge_path))
    token = gates.issue_supersession_authorization(
        kconn, canonical_family_id=family_id, approver="operator", justification="corrected rerun",
        related_nullifier=nullifier, split_version=split_version, test_set_hash=test_set_hash)
    kconn.close()

    result = _register(canonical_conn, knowledge_path, "E-013b", hypothesis_id="H-013", question_ids="FAM-013",
                        test_cycle_ids=["c1", "c2"], supersession_token=token)
    assert result["nullifier_result"] == "CONSUMED_WITH_SUPERSESSION"
    assert "AUTHORIZATION_CONSUMED" in _audit_actions(knowledge_path)


# ---------------------------------------------------------------------------
# 14. Supersession token reuse — blocked
# ---------------------------------------------------------------------------

def test_14_supersession_token_reuse_blocked(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-014a", hypothesis_id="H-014", question_ids="FAM-014",
              test_cycle_ids=["c1", "c2"])
    family_id = gates.resolve_canonical_family_id("FAM-014", "H-014")
    split_version = gates.resolve_split_version("chronological 70/30 by signal_birth_ts, never randomized")
    _, test_set_hash, _ = gates._normalized_test_set_hash(["c1", "c2"])  # noqa: SLF001
    nullifier = gates.derive_test_nullifier(family_id, split_version, ["c1", "c2"])

    kconn = sqlite3.connect(str(knowledge_path))
    token = gates.issue_supersession_authorization(
        kconn, canonical_family_id=family_id, approver="operator", justification="corrected rerun",
        related_nullifier=nullifier, split_version=split_version, test_set_hash=test_set_hash)
    kconn.close()

    _register(canonical_conn, knowledge_path, "E-014b", hypothesis_id="H-014", question_ids="FAM-014",
              test_cycle_ids=["c1", "c2"], supersession_token=token)
    with pytest.raises(gates.AuthorizationInvalid):
        _register(canonical_conn, knowledge_path, "E-014c", hypothesis_id="H-014", question_ids="FAM-014",
                  test_cycle_ids=["c1", "c2"], supersession_token=token)


# ---------------------------------------------------------------------------
# 15. Wrong family token — blocked
# ---------------------------------------------------------------------------

def test_15_wrong_family_token_blocked(canonical_conn, knowledge_path):
    family_a = gates.resolve_canonical_family_id("FAM-015-A", "H-BUYFADE-RETEST-15A")
    kconn = sqlite3.connect(str(knowledge_path))
    gates.init_gates_schema(kconn)
    token = gates.issue_retry_authorization(
        kconn, canonical_family_id=family_a, approver="operator", justification="j",
        related_experiment_id="E-ORIG", retry_condition_satisfied="met")
    kconn.close()

    with pytest.raises(gates.AuthorizationInvalid):
        _register(canonical_conn, knowledge_path, "E-015", hypothesis_id="H-BUYFADE-RETEST-15B",
                   question_ids="FAM-015-B", retry_token=token)


# ---------------------------------------------------------------------------
# 16. Wrong split-version token — blocked
# ---------------------------------------------------------------------------

def test_16_wrong_split_version_token_blocked(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-016a", hypothesis_id="H-016", question_ids="FAM-016",
              test_cycle_ids=["c1", "c2"])
    family_id = gates.resolve_canonical_family_id("FAM-016", "H-016")
    _, test_set_hash, _ = gates._normalized_test_set_hash(["c1", "c2"])  # noqa: SLF001

    kconn = sqlite3.connect(str(knowledge_path))
    token = gates.issue_supersession_authorization(
        kconn, canonical_family_id=family_id, approver="operator", justification="j",
        related_nullifier="irrelevant", split_version="SPLITv1:wrong-split", test_set_hash=test_set_hash)
    kconn.close()

    with pytest.raises(gates.AuthorizationInvalid):
        _register(canonical_conn, knowledge_path, "E-016b", hypothesis_id="H-016", question_ids="FAM-016",
                  test_cycle_ids=["c1", "c2"], supersession_token=token)


# ---------------------------------------------------------------------------
# 17. Wrong TEST-set token — blocked
# ---------------------------------------------------------------------------

def test_17_wrong_test_set_token_blocked(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-017a", hypothesis_id="H-017", question_ids="FAM-017",
              test_cycle_ids=["c1", "c2"])
    family_id = gates.resolve_canonical_family_id("FAM-017", "H-017")
    split_version = gates.resolve_split_version("chronological 70/30 by signal_birth_ts, never randomized")

    kconn = sqlite3.connect(str(knowledge_path))
    token = gates.issue_supersession_authorization(
        kconn, canonical_family_id=family_id, approver="operator", justification="j",
        related_nullifier="irrelevant", split_version=split_version, test_set_hash="wrong-test-set-hash")
    kconn.close()

    with pytest.raises(gates.AuthorizationInvalid):
        _register(canonical_conn, knowledge_path, "E-017b", hypothesis_id="H-017", question_ids="FAM-017",
                  test_cycle_ids=["c1", "c2"], supersession_token=token)


# ---------------------------------------------------------------------------
# 18. Crash before experiment registration — nullifier not consumed
# ---------------------------------------------------------------------------

def test_18_crash_before_registration_nullifier_not_consumed(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-018", hypothesis_id="H-018", question_ids="FAM-018",
              frozen_population="pop-v1", test_cycle_ids=["c1", "c2"])
    # Same experiment_id, DIFFERENT content -> ImmutableExperimentConflict raised
    # inside record_experiment_registry, AFTER the gate decision was persisted
    # but BEFORE the nullifier consumption step -- proves the whole thing
    # rolls back together (nullifier for a fresh, different id below still free).
    with pytest.raises(ledger.ImmutableExperimentConflict):
        _register(canonical_conn, knowledge_path, "E-018", hypothesis_id="H-018", question_ids="FAM-018",
                  frozen_population="pop-v2-DIFFERENT", test_cycle_ids=["c1", "c2"])
    # nullifier for this family/split/set is still only consumed by E-018 (first call) -- a
    # NEW experiment id could not slip through using the interrupted attempt's non-commit.
    nullifier = gates.derive_test_nullifier(
        gates.resolve_canonical_family_id("FAM-018", "H-018"),
        gates.resolve_split_version("chronological 70/30 by signal_birth_ts, never randomized"),
        ["c1", "c2"])
    kconn = sqlite3.connect(str(knowledge_path))
    rows = kconn.execute(
        "SELECT consumed_by_experiment_id FROM epistemic_test_nullifiers WHERE nullifier=?",
        (nullifier,)).fetchall()
    kconn.close()
    assert [r[0] for r in rows] == ["E-018"]


# ---------------------------------------------------------------------------
# 19. Crash after registration within transaction — atomic rollback
# ---------------------------------------------------------------------------

def test_19_crash_after_registration_before_commit_atomic_rollback(canonical_conn, knowledge_path):
    with patch.object(gates, "consume_test_evidence", side_effect=RuntimeError("simulated crash")):
        with pytest.raises(RuntimeError):
            _register(canonical_conn, knowledge_path, "E-019", test_cycle_ids=["c1", "c2"])
    # record_experiment_registry/record_experiment_results DID execute (in-transaction)
    # before the simulated crash, but must have been rolled back with everything else.
    assert _registry_count(canonical_conn, "E-019") == 0
    assert _results_count(canonical_conn, "E-019") == 0


# ---------------------------------------------------------------------------
# 20. Concurrent double-consumption attempt — only one succeeds (DB-level backstop)
# ---------------------------------------------------------------------------

def test_20_concurrent_double_consumption_only_one_succeeds_at_db_level(knowledge_path):
    kconn = sqlite3.connect(str(knowledge_path))
    gates.init_gates_schema(kconn)
    nullifier = "race-nullifier-abc"
    kconn.execute(
        "INSERT INTO epistemic_test_nullifiers (nullifier, family_id, split_version, test_set_hash,"
        " consumed_by_experiment_id, supersession_token, consumed_ms) VALUES (?,?,?,?,?,NULL,?)",
        (nullifier, "fam", "split", "hash", "E-RACE-WINNER", 1))
    kconn.commit()
    # a second, concurrent "first consumption" (supersession_token IS NULL) of the
    # SAME nullifier by a different experiment_id must be rejected by the partial
    # unique index at the database level, independent of any Python pre-check.
    with pytest.raises(sqlite3.IntegrityError):
        kconn.execute(
            "INSERT INTO epistemic_test_nullifiers (nullifier, family_id, split_version, test_set_hash,"
            " consumed_by_experiment_id, supersession_token, consumed_ms) VALUES (?,?,?,?,?,NULL,?)",
            (nullifier, "fam", "split", "hash", "E-RACE-LOSER", 2))
    kconn.close()


def test_20b_application_level_race_translated_to_reuse_blocked(canonical_conn, knowledge_path):
    _register(canonical_conn, knowledge_path, "E-020a", hypothesis_id="H-020", question_ids="FAM-020",
              test_cycle_ids=["c1", "c2"])
    with pytest.raises(gates.TestEvidenceReuseBlocked):
        _register(canonical_conn, knowledge_path, "E-020b", hypothesis_id="H-020", question_ids="FAM-020",
                  test_cycle_ids=["c1", "c2"])


# ---------------------------------------------------------------------------
# 21. Direct experiment registry write through normal application path cannot bypass gates
# ---------------------------------------------------------------------------

def test_21_gated_entry_point_has_no_bypass_switch():
    sig = inspect.signature(ledger.register_experiment_with_gates)
    for name, param in sig.parameters.items():
        if param.default is True and "enforce" in name.lower():
            pytest.fail(f"found an enforce-gates-style bypass switch defaulting True: {name}")
        assert name != "enforce_gates", "must not expose an enforce_gates toggle at all"


# ---------------------------------------------------------------------------
# 22. Legacy/CLI path bypass — honest canary, not a false "closed" claim
# ---------------------------------------------------------------------------

def test_22_known_legacy_bypass_surface_is_exactly_the_documented_set():
    """UPDATED by BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: the 10
    legacy modules (candidate_universe/w1/w3/w4/w5a/w6/w6rs_confirmation/
    w6rs_confound_resolution/w7a/w10a) that this test originally flagged as
    an open bypass now all route through
    ami.warehouse.experiment_ledger.register_legacy_snapshot_with_gates
    (see reports/governance/EPISTEMIC_NULLIFIER_LEGACY_BYPASS_CLOSURE_V1_
    STATE_TRANSITION_PROOF.md) -- the offending set is now empty. This canary
    still fails loudly if inline registry/result SQL reappears in
    ami/research/ without a matching transition-proof update."""
    import pathlib
    research_dir = pathlib.Path(__file__).resolve().parents[1] / "ami" / "research"
    offenders = set()
    for f in research_dir.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        if "INSERT INTO experiment_registry" in text:
            offenders.add(f.name)
    expected: set[str] = set()
    assert offenders == expected, (
        f"legacy bypass surface changed: now={sorted(offenders)} expected={sorted(expected)} -- "
        "update reports/governance/EPISTEMIC_NULLIFIER_LEGACY_BYPASS_CLOSURE_V1_STATE_TRANSITION_PROOF.md"
        " if this is an intentional, reviewed change.")


# ---------------------------------------------------------------------------
# 23. Blocked attempt writes no experiment result
# ---------------------------------------------------------------------------

def test_23_blocked_attempt_writes_no_experiment_result(canonical_conn, knowledge_path):
    with pytest.raises(gates.GraveyardRetestBlocked):
        _register(canonical_conn, knowledge_path, "E-023", hypothesis_id="H-BUYFADE-RETEST-23")
    assert _results_count(canonical_conn, "E-023") == 0


# ---------------------------------------------------------------------------
# 24/25/26. Real-data smoke: historical immutability, retro-audit, canonical hash/version
# ---------------------------------------------------------------------------

def test_24_existing_22_historical_experiments_remain_unchanged():
    """Baseline was 22 at this batch's freeze point (2026-07-06); two
    independent, already-accepted governed executions since then
    (G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1, FAM_CASCADE_ABSORPTION_IMPACT
    execution) each added one experiment_registry row, giving 24. See
    MIGRATION_LOG.md M-0035's own regression-waiver note for the identical
    pattern. Not a violation of this batch's own historical-immutability
    claim -- this batch never wrote to experiment_registry itself."""
    conn = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        count = conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
    finally:
        conn.close()
    assert count == 24


def test_25_retro_audit_remains_0_of_22(tmp_path):
    disposable = tmp_path / "knowledge_retro.sqlite"
    shutil.copyfile(REAL_KNOWLEDGE_PATH, disposable)
    kconn = sqlite3.connect(str(disposable))
    gates.init_gates_schema(kconn)
    canonical_ro = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        results = gates.retro_audit_experiment_registry(kconn, canonical_ro)
    finally:
        kconn.close()
        canonical_ro.close()
    assert len(results) == 24
    assert sum(1 for r in results if r["would_block"]) == 0


def test_26_canonical_schema_version_and_hash_unchanged():
    conn = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        version = conn.execute(
            "SELECT version FROM schema_versions WHERE component='canonical_warehouse'").fetchone()[0]
    finally:
        conn.close()
    assert version == 14
    h = hashlib.sha256()
    with open(REAL_CANONICAL_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    assert h.hexdigest() == REAL_CANONICAL_SHA256


# ---------------------------------------------------------------------------
# 27. Protected subsystem delta = 0 (proxy: protected counts unchanged, real DB, read-only)
# ---------------------------------------------------------------------------

def test_27_protected_counts_unchanged():
    conn = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("ami_events", "ami_signal_lifecycle", "ami_cycles",
                      "ami_birth_truncated_cascade_geometry")
        }
    finally:
        conn.close()
    assert counts == {
        "ami_events": 252, "ami_signal_lifecycle": 324, "ami_cycles": 167,
        "ami_birth_truncated_cascade_geometry": 220,
    }
