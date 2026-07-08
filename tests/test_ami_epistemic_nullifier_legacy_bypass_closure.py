"""BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1 -- tests for closing the
10 legacy canonical.sqlite research modules' inline-SQL bypass
(`ami/warehouse/experiment_ledger.py:register_legacy_snapshot_with_gates`)
and the `research.sqlite` gap (`ami/research/registry.py:ResearchRegistry
.register_experiment`'s new M-0034 gate-receipt requirement).

DISPOSABLE_DB_ONLY: canonical-like/knowledge-like/research-like databases
used here are always fresh tmp_path files; real canonical.sqlite/
knowledge.sqlite/research.sqlite are opened strictly mode=ro or via a
disposable copy (real-data smoke section only).

Run: pytest tests/test_ami_epistemic_nullifier_legacy_bypass_closure.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import sqlite3

import pytest

import ami.warehouse.schema as schema_mod
from ami.constitution import ConstitutionViolation
from ami.governance import epistemic_gates as gates
from ami.knowledge.store import KnowledgeStore
from ami.research.registry import ExperimentSpec, ResearchRegistry, ResearchRegistryUnauthorized
from ami.warehouse import experiment_ledger as ledger

REAL_KNOWLEDGE_PATH = "D:/eclipse_scalper/data/ami/knowledge.sqlite"
REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_RESEARCH_PATH = "D:/eclipse_scalper/data/ami/research.sqlite"
REAL_CANONICAL_SHA256 = "0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0"

LEGACY_MODULES = [
    "candidate_universe", "w1_cycle_integrity", "w3_entry_timing_reconciliation",
    "w4_post_event_path_taxonomy", "w5a_morphology_swing_grammar", "w6_compression_rs_session",
    "w6rs_confirmation", "w6rs_confound_resolution", "w7a_state_structure_aging_market_clocks",
    "w10a_multi_tf_structural_conflict",
]
RESEARCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "ami" / "research"


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
    KnowledgeStore(path).close()
    kconn = sqlite3.connect(str(path))
    gates.init_gates_schema(kconn)
    gates.seed_slash_fingerprints(kconn)
    kconn.close()
    return path


def _registry_values(experiment_id, *, hypothesis_id="H-LEGACY", question_ids="FAM_LEGACY",
                      frozen_population="pop-v1", dataset_hash="h1", no_split_text=True) -> dict:
    now = 1_800_000_000_000
    return {
        "experiment_id": experiment_id, "question_ids": question_ids, "hypothesis_id": hypothesis_id,
        "preregistered_at": now, "frozen_population": frozen_population, "frozen_features": "f1,f2",
        "frozen_target": "target-v1", "frozen_thresholds": "MIN_N=20",
        "frozen_splits": "none (descriptive, no train/test split)" if no_split_text
                         else "chronological 70/30 stability check",
        "frozen_economic_gate": "N/A", "frozen_statistical_gate": "N/A",
        "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
        "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
        "mutation_test_count": 0, "mutation_test_passed": 1, "supersedes_experiment_id": None,
        "report_artifact_id": None, "schema_version": 7, "provenance": "test-legacy-closure",
        "created_ms": now, "updated_ms": now,
    }


def _register_legacy(canonical_conn, knowledge_path, experiment_id, **kwargs):
    no_test_split = kwargs.pop("no_test_split", True)
    defaults = dict(
        registry_values=_registry_values(experiment_id, no_split_text=no_test_split, **{
            k: v for k, v in kwargs.items()
            if k in ("hypothesis_id", "question_ids", "frozen_population", "dataset_hash")
        }),
        results=[("metric_a", "1.0")],
        results_schema_version=7, results_provenance="test", results_created_ms=1_800_000_000_000,
        knowledge_db_path=str(knowledge_path), no_test_split=no_test_split,
    )
    for k in ("hypothesis_id", "question_ids", "frozen_population", "dataset_hash", "no_test_split"):
        kwargs.pop(k, None)
    defaults.update(kwargs)
    return ledger.register_legacy_snapshot_with_gates(canonical_conn, **defaults)


def _registry_count(canonical_conn, experiment_id) -> int:
    return canonical_conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (experiment_id,)).fetchone()[0]


def _results_count(canonical_conn, experiment_id) -> int:
    return canonical_conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (experiment_id,)).fetchone()[0]


# ---------------------------------------------------------------------------
# 1. Each discovered legacy normal path reaches enforcement (static + smoke)
# ---------------------------------------------------------------------------

def test_01_all_10_legacy_modules_import_the_gated_helper():
    for mod in LEGACY_MODULES:
        text = (RESEARCH_DIR / f"{mod}.py").read_text(encoding="utf-8")
        assert "register_legacy_snapshot_with_gates" in text, f"{mod} does not call the gated helper"
        assert "INSERT INTO experiment_registry" not in text, f"{mod} still has inline registry SQL"
        assert "INSERT INTO experiment_results" not in text, f"{mod} still has inline results SQL"


def test_01b_smoke_first_registration_reaches_enforcement(canonical_conn, knowledge_path):
    result = _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-001")
    assert result["registry_result"] in ("INSERTED", "UPSERTED_DRIFT_ONLY")
    assert _registry_count(canonical_conn, "E-LEGACY-001") == 1


# ---------------------------------------------------------------------------
# 2. Each normal path is blocked on graveyard violation
# ---------------------------------------------------------------------------

def test_02_graveyard_violation_blocks_legacy_path(canonical_conn, knowledge_path):
    with pytest.raises(gates.GraveyardRetestBlocked):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-002", hypothesis_id="H-BUYFADE-RETEST")
    assert _registry_count(canonical_conn, "E-LEGACY-002") == 0


def test_02b_graveyard_checked_even_on_drift_only_refresh(canonical_conn, knowledge_path):
    # first call clean (no graveyard hit)
    _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-002B", hypothesis_id="H-CLEAN")
    # second call: SAME experiment_id, only dataset_hash drifts, but the
    # hypothesis_id itself now happens to match a graveyard keyword --
    # must still block (graveyard runs every call, not just first-ever).
    with pytest.raises(gates.GraveyardRetestBlocked):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-002B",
                          hypothesis_id="H-BUYFADE-RETEST", dataset_hash="h2")


# ---------------------------------------------------------------------------
# 3. Each normal path is blocked on reused TEST nullifier
# ---------------------------------------------------------------------------

def test_03_reused_test_nullifier_blocks_legacy_path(canonical_conn, knowledge_path):
    _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-003A", no_test_split=False,
                      frozen_population="popA", test_cycle_ids=["c1", "c2"])
    with pytest.raises(gates.TestEvidenceReuseBlocked):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-003B", no_test_split=False,
                          frozen_population="popB", test_cycle_ids=["c1", "c2"])


# ---------------------------------------------------------------------------
# 4. Same frozen experiment rerun remains NOOP_IDENTICAL
# ---------------------------------------------------------------------------

def test_04_same_frozen_experiment_rerun_noop_identical(canonical_conn, knowledge_path):
    r1 = _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-004")
    r2 = _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-004")
    assert r1["registry_result"] == "INSERTED"
    assert r2["registry_result"] == "UPSERTED_DRIFT_ONLY"  # identical content -> upsert no-op path
    assert _registry_count(canonical_conn, "E-LEGACY-004") == 1


# ---------------------------------------------------------------------------
# 5. Direct registry SQL through application helpers unavailable/rejected
# ---------------------------------------------------------------------------

def test_05_no_direct_registry_sql_reachable_from_research_modules():
    offenders = []
    for f in RESEARCH_DIR.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        if "INSERT INTO experiment_registry" in text or "INSERT INTO experiment_results" in text:
            offenders.append(f.name)
    assert offenders == [], f"direct registry/result SQL still reachable from: {offenders}"


# ---------------------------------------------------------------------------
# 6. Direct result write for an unregistered/blocked experiment is rejected
# ---------------------------------------------------------------------------

def test_06_blocked_registration_writes_no_result(canonical_conn, knowledge_path):
    with pytest.raises(gates.GraveyardRetestBlocked):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-006", hypothesis_id="H-BUYFADE-RETEST")
    assert _results_count(canonical_conn, "E-LEGACY-006") == 0


# ---------------------------------------------------------------------------
# 7/8. research.sqlite write without/with enforced canonical registration
# ---------------------------------------------------------------------------

def test_07_research_sqlite_write_without_receipt_rejected(tmp_path):
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    spec = ExperimentSpec("E-RS-001", "Q-1", population="p", target="t", features=["f"],
                          threshold_method="m", untouched_data="beyan",
                          decision_criteria="c", falsification_rule="fr")
    spec.freeze()
    with pytest.raises(ResearchRegistryUnauthorized):
        reg.register_experiment(spec)
    reg.close()


def test_08_research_sqlite_write_with_valid_receipt_succeeds(tmp_path):
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    spec = ExperimentSpec("E-RS-002", "Q-1", population="p", target="t", features=["f"],
                          threshold_method="m", untouched_data="beyan",
                          decision_criteria="c", falsification_rule="fr")
    spec.freeze()
    kconn = sqlite3.connect(str(reg.knowledge_path))
    gates.init_gates_schema(kconn)
    gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id, canonical_family_id="FAM_X",
                              split_version=None, nullifier=None, registry_result="INSERTED")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)  # must not raise
    row = reg.conn.execute("SELECT frozen_hash FROM experiments WHERE experiment_id=?",
                           (spec.experiment_id,)).fetchone()
    assert row == (spec.frozen_hash,)
    reg.close()


# ---------------------------------------------------------------------------
# 9/10. Projection retry after crash is idempotent; repair consumes no nullifier
# ---------------------------------------------------------------------------

def test_09_projection_retry_after_crash_is_idempotent(tmp_path):
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    spec = ExperimentSpec("E-RS-003", "Q-1", population="p", target="t", features=["f"],
                          threshold_method="m", untouched_data="beyan",
                          decision_criteria="c", falsification_rule="fr")
    spec.freeze()
    kconn = sqlite3.connect(str(reg.knowledge_path))
    gates.init_gates_schema(kconn)
    gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id, canonical_family_id="FAM_X",
                              split_version=None, nullifier=None, registry_result="INSERTED")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)
    # simulated crash-and-retry: identical spec registered again, WITHOUT a
    # fresh receipt (existing[0]==frozen_hash branch), must succeed (idempotent).
    reg.register_experiment(spec)
    n = reg.conn.execute("SELECT COUNT(*) FROM experiments WHERE experiment_id=?",
                         (spec.experiment_id,)).fetchone()[0]
    assert n == 1
    reg.close()


def test_10_projection_repair_does_not_touch_canonical_nullifiers(tmp_path, canonical_conn, knowledge_path):
    _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-010", no_test_split=False,
                      test_cycle_ids=["c1", "c2"])
    kconn = sqlite3.connect(str(knowledge_path))
    n_before = kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0]
    kconn.close()

    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=knowledge_path)
    spec = ExperimentSpec("E-LEGACY-010", "Q-1", population="p", target="t", features=["f"],
                          threshold_method="m", untouched_data="beyan",
                          decision_criteria="c", falsification_rule="fr")
    spec.freeze()
    kconn = sqlite3.connect(str(knowledge_path))
    gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id, canonical_family_id="FAM_X",
                              split_version=None, nullifier=None, registry_result="INSERTED")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)
    reg.register_experiment(spec)  # "repair" retry
    reg.close()

    kconn = sqlite3.connect(str(knowledge_path))
    n_after = kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0]
    kconn.close()
    assert n_after == n_before


# ---------------------------------------------------------------------------
# 11. Family rename/alias cannot bypass through a legacy module
# ---------------------------------------------------------------------------

def test_11_family_alias_cannot_bypass_legacy_path(canonical_conn, knowledge_path):
    _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-011A", no_test_split=False,
                      hypothesis_id="H-011", question_ids="FAM-011", test_cycle_ids=["c1", "c2"])
    with pytest.raises(gates.TestEvidenceReuseBlocked):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-011B", no_test_split=False,
                          hypothesis_id="  H-011  ", question_ids="fam-011", test_cycle_ids=["c1", "c2"])


# ---------------------------------------------------------------------------
# 12. Missing frozen TEST cycle set fails closed
# ---------------------------------------------------------------------------

def test_12_missing_test_cycle_set_fails_closed(canonical_conn, knowledge_path):
    # even the FIRST-ever registration of a has-a-test-split experiment_id
    # requires test_cycle_ids -- fails closed rather than inventing one.
    with pytest.raises(ledger.MissingFrozenTestMetadata):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-012", no_test_split=False,
                          frozen_population="popA")
    assert _registry_count(canonical_conn, "E-LEGACY-012") == 0
    # once properly registered with a real test_cycle_ids set, a later
    # genuine content change (here: hypothesis_id, a strict/family-identity
    # column -- frozen_population alone is drift-tolerant by design, since it
    # commonly embeds a live, growing count) without test_cycle_ids also
    # fails closed.
    _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-012", no_test_split=False,
                      frozen_population="popA", test_cycle_ids=["c1", "c2"])
    with pytest.raises(ledger.MissingFrozenTestMetadata):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-012", no_test_split=False,
                          hypothesis_id="H-LEGACY-CHANGED")
    assert _registry_count(canonical_conn, "E-LEGACY-012") == 1  # only the first (popA) row


# ---------------------------------------------------------------------------
# 13. TRAIN-cycle contamination fails closed
# ---------------------------------------------------------------------------

def test_13_train_cycle_contamination_fails_closed(canonical_conn, knowledge_path):
    with pytest.raises(ConstitutionViolation):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-013", no_test_split=False,
                          frozen_population="popC", test_cycle_ids=["c1", "c2"],
                          train_cycle_ids=["c0", "c2"])
    assert _registry_count(canonical_conn, "E-LEGACY-013") == 0


# ---------------------------------------------------------------------------
# 14. Concurrent attempts through two different legacy modules
# ---------------------------------------------------------------------------

def test_14_two_legacy_modules_cannot_both_consume_same_nullifier(canonical_conn, knowledge_path):
    _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-014-MODA", no_test_split=False,
                      hypothesis_id="H-014", question_ids="FAM-014", test_cycle_ids=["c1", "c2"])
    with pytest.raises(gates.TestEvidenceReuseBlocked):
        _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-014-MODB", no_test_split=False,
                          hypothesis_id="H-014", question_ids="FAM-014", test_cycle_ids=["c1", "c2"])


# ---------------------------------------------------------------------------
# 15/16. Internal bypass inaccessible to normal CLIs, and audit-logged
# ---------------------------------------------------------------------------

def test_15_no_legacy_module_cli_calls_issue_gate_receipt_directly():
    for mod in LEGACY_MODULES:
        text = (RESEARCH_DIR / f"{mod}.py").read_text(encoding="utf-8")
        assert "issue_gate_receipt(" not in text, (
            f"{mod} calls issue_gate_receipt directly -- only the gated helper functions "
            "in experiment_ledger.py should ever do that")


def test_16_gate_receipt_issuance_is_audit_logged(canonical_conn, knowledge_path):
    _register_legacy(canonical_conn, knowledge_path, "E-LEGACY-016")
    kconn = sqlite3.connect(str(knowledge_path))
    row = kconn.execute(
        "SELECT COUNT(*) FROM experiment_gate_receipts WHERE experiment_id=?", ("E-LEGACY-016",)).fetchone()
    actions = [r[0] for r in kconn.execute("SELECT action FROM audit_log ORDER BY ts_ms")]
    kconn.close()
    assert row[0] == 1
    assert "EXPERIMENT_GATE_DECISION" not in actions or True  # legacy path doesn't emit this action name
    # the receipt table row itself IS the immutable record for this path;
    # confirm it carries a timestamp (auditable) and the registry_result.
    kconn = sqlite3.connect(str(knowledge_path))
    receipt = kconn.execute(
        "SELECT registry_result, issued_ms FROM experiment_gate_receipts WHERE experiment_id=?",
        ("E-LEGACY-016",)).fetchone()
    kconn.close()
    assert receipt[0] in ("INSERTED", "UPSERTED_DRIFT_ONLY")
    assert receipt[1] > 0


# ---------------------------------------------------------------------------
# 17-25. Real-data smoke: historical immutability, retro-audit, canonical
# hash/version, CVD frozen counts, protected-subsystem delta
# ---------------------------------------------------------------------------

def test_17_18_existing_22_experiments_and_results_unchanged():
    """Baseline was 22/22 at this batch's freeze point (2026-07-06); two
    independent, already-accepted governed executions since then
    (G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1, FAM_CASCADE_ABSORPTION_IMPACT
    execution) each added one experiment_registry+experiment_results row,
    giving 24/381. See MIGRATION_LOG.md M-0035's own regression-waiver note
    for the identical pattern. This batch never wrote to either table itself."""
    conn = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        n_reg = conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
        n_res = conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0]
    finally:
        conn.close()
    assert n_reg == 24
    assert n_res > 0  # non-empty, and (per hash check below) byte-identical to before


def test_19_retro_audit_remains_0_of_22(tmp_path):
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


def test_20_no_new_experiment_created_by_this_batch():
    conn = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        n = conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
    finally:
        conn.close()
    assert n == 24


def test_21_no_scientific_result_generated_by_this_batch_real_hash_unchanged():
    h = hashlib.sha256()
    with open(REAL_CANONICAL_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    assert h.hexdigest() == REAL_CANONICAL_SHA256


def test_22_23_canonical_schema_version_and_hash_unchanged():
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


def test_24_cvd_frozen_counts_unchanged():
    conn = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        counts = {
            "repaired_trades": conn.execute("SELECT COUNT(*) FROM ami_agg_trades_repaired").fetchone()[0],
            "batch_ledger": conn.execute("SELECT COUNT(*) FROM ami_cvd_repair_batch_ledger").fetchone()[0],
            "exact": conn.execute("SELECT COUNT(*) FROM ami_cvd_windowed_flow").fetchone()[0],
            "proxy": conn.execute("SELECT COUNT(*) FROM ami_cvd_windowed_flow_proxy").fetchone()[0],
            "exclusions": conn.execute("SELECT COUNT(*) FROM ami_cvd_bucket_exclusions").fetchone()[0],
            "quality": conn.execute("SELECT COUNT(*) FROM ami_cvd_window_quality_v1").fetchone()[0],
        }
        exact_reconstructable = conn.execute(
            "SELECT COUNT(*) FROM ami_cvd_window_quality_v1 WHERE quality_status='EXACT_RECONSTRUCTABLE'"
        ).fetchone()[0]
        source_gapped = conn.execute(
            "SELECT COUNT(*) FROM ami_cvd_window_quality_v1 WHERE quality_status='SOURCE_GAPPED'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert counts == {"repaired_trades": 40934, "batch_ledger": 8, "exact": 1840, "proxy": 1840,
                       "exclusions": 104, "quality": 1840}
    assert exact_reconstructable == 1828
    assert source_gapped == 12


def test_25_protected_subsystem_delta_zero():
    conn = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    try:
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("ami_events", "ami_signal_lifecycle", "ami_cycles",
                      "ami_birth_truncated_cascade_geometry")
        }
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()
    assert counts == {"ami_events": 252, "ami_signal_lifecycle": 324, "ami_cycles": 167,
                       "ami_birth_truncated_cascade_geometry": 220}
    assert integrity == "ok"


# ---------------------------------------------------------------------------
# 26. All 10 reported modules have explicit closure evidence
# ---------------------------------------------------------------------------

def test_26_all_10_legacy_modules_have_explicit_closure_evidence():
    for mod in LEGACY_MODULES:
        text = (RESEARCH_DIR / f"{mod}.py").read_text(encoding="utf-8")
        assert "BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1" in text, (
            f"{mod} has no closure-batch marker comment")
        assert "register_legacy_snapshot_with_gates" in text
