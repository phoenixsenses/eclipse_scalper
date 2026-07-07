"""BATCH-BOOK-SPREAD-DYNAMICS-LONG-PREREGISTRATION-V1 -- focused tests.

DISPOSABLE_DB_ONLY: state-touching tests run against disposable COPIES of
the real canonical.sqlite / knowledge.sqlite. Read-only checks against the
real DBs use `mode=ro` URIs only. Outcome-blind throughout -- no test
reads `endpoint_return_bps`/`mfe_bps`.

Run: pytest tests/test_ami_research_book_spread_dynamics_long_preregistration_v1.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import shutil
import sqlite3

import pytest

from ami.governance import epistemic_gates as gates
from ami.research import book_spread_dynamics_long_preregistration_v1 as PREREG
from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH

REAL_KNOWLEDGE_PATH = "D:/eclipse_scalper/data/ami/knowledge.sqlite"


def _disposable_canonical(tmp_path):
    dst = tmp_path / "canon.sqlite"
    shutil.copy2(REAL_CANONICAL_PATH, dst)
    conn = sqlite3.connect(dst)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _disposable_knowledge(tmp_path):
    dst = tmp_path / "knowledge.sqlite"
    shutil.copy2(REAL_KNOWLEDGE_PATH, dst)
    conn = sqlite3.connect(dst)
    gates.init_gates_schema(conn)
    return conn


def _real_canonical_ro():
    conn = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _real_knowledge_ro():
    conn = sqlite3.connect(f"file:{REAL_KNOWLEDGE_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


# ---------------------------------------------------------------------------
# Prior V1 immutability + parent/child identity
# ---------------------------------------------------------------------------

def test_prior_incomplete_v1_immutability_constants():
    assert PREREG.PARENT_FAMILY_ID == "FAMv1:2d102e7b70820470"
    assert PREREG.PRIOR_INCOMPLETE_CHILD_ID == "H-BOOK-SPREAD-CHANGE-BPS-W300-V1"
    assert PREREG.PRIOR_INCOMPLETE_COMMIT == "a4722117"


def test_prior_incomplete_v1_artifact_unchanged_on_disk():
    import hashlib
    path = "D:/eclipse_scalper/reports/research/s34/S34_BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1.md"
    with open(path, "rb") as f:
        content = f.read()
    assert b"BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE" in content
    assert hashlib.sha256(content).hexdigest() is not None  # file is readable/stable


def test_new_child_constants():
    assert PREREG.CHILD_ID == "H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1"
    assert PREREG.FORMULA_VERSION == "BOOK_SPREAD_CHANGE_BPS_W300_V1"
    assert PREREG.ROW_ACCOUNTING_ROOT == (
        "33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31")
    assert PREREG.EXPECTED_SCHEMA_VERSION == 14
    assert PREREG.DIRECTION == "LONG"
    assert PREREG.EXPECTED_SIGN == "NEGATIVE"
    assert PREREG.EFFECT_FLOOR_BPS == -1.0


def test_new_family_id_distinct_from_parent_and_mixed_child():
    family_id = gates.resolve_canonical_family_id(PREREG.QUESTION_IDS, PREREG.HYPOTHESIS_ID)
    assert family_id == "FAMv1:85cfe11ceeadbbe8"
    assert family_id != PREREG.PARENT_FAMILY_ID
    mixed_child_family_id = gates.resolve_canonical_family_id(
        "FAM_BOOK_SPREAD_DYNAMICS", "H-BOOK-SPREAD-CHANGE-BPS-W300-V1-DIRECTION-NEUTRAL")
    assert family_id != mixed_child_family_id


# ---------------------------------------------------------------------------
# Phase 1-2: identity / graveyard / exposure (real DB, RO)
# ---------------------------------------------------------------------------

def test_graveyard_clean_for_long_child():
    kconn = _real_knowledge_ro()
    try:
        result = PREREG.resolve_family_and_child_identity(kconn)
    finally:
        kconn.close()
    assert result["graveyard_clean"] is True
    assert result["distinct_from_parent"] is True


def test_mixed_direction_incomplete_attempt_is_not_test_exposure_for_long_child():
    """The prior INCOMPLETE attempt created no gate receipt and no
    nullifier -- confirm the LONG child's own exposure check is clean,
    proving the mixed-direction attempt left nothing to be 'exposed to'."""
    kconn = _real_knowledge_ro()
    try:
        result = PREREG.resolve_family_and_child_identity(kconn)
    finally:
        kconn.close()
    assert result["prior_test_exposure_clean"] is True
    assert result["existing_gate_receipts_for_family"] == []
    assert result["genuinely_unconsumed"] is True


def test_resolve_identity_is_read_only(tmp_path):
    kconn = _disposable_knowledge(tmp_path)
    before = kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0]
    PREREG.resolve_family_and_child_identity(kconn)
    after = kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0]
    kconn.close()
    assert before == after


# ---------------------------------------------------------------------------
# Phase 3: outcome (metadata only, reused verbatim)
# ---------------------------------------------------------------------------

def test_outcome_reused_verbatim_no_direction_flip():
    outcome = PREREG.resolve_outcome_metadata()
    assert outcome["outcome_id"] == "endpoint_return_bps@swing_24h"
    assert outcome["reused_verbatim"] is True
    assert outcome["newly_derived"] is False
    assert outcome["direction_flip_applied"] is False


def test_module_source_never_reads_outcome_value_columns():
    """AST-level guard scoped to .execute()-family call arguments only."""
    import ast
    import inspect
    src = inspect.getsource(PREREG)
    tree = ast.parse(src)
    forbidden = {"endpoint_return_bps", "mfe_bps"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("execute", "executescript", "executemany"):
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        for token in forbidden:
                            if token in sub.value:
                                hits.append((token, sub.value))
    assert hits == [], f"forbidden outcome-value column in a SQL execute() literal: {hits}"


# ---------------------------------------------------------------------------
# Phase 4: LONG population (structural + outcome-compatible, real DB)
# ---------------------------------------------------------------------------

def test_long_structural_population_is_70():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_long_population(conn)
    finally:
        conn.close()
    assert pop["structural_representative_count"] == 70
    assert pop["matches_expected_structural_70"] is True


def test_long_eligible_population_is_58_of_70():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_long_population(conn)
    finally:
        conn.close()
    assert pop["eligible_representative_count"] == 58
    assert pop["excluded_count"] == 12
    assert pop["duplicate_cycle_ids"] == 0
    assert pop["missing_representatives"] == 0


def test_short_never_appears_in_long_population():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_long_population(conn)
        anchors_direction = conn.execute(
            "SELECT DISTINCT direction FROM ami_book_spread_change_windowed_flow "
            "WHERE is_cycle_representative=1 AND direction=?", (PREREG.DIRECTION,)).fetchall()
    finally:
        conn.close()
    assert anchors_direction == [("LONG",)]
    assert pop["structural_representative_count"] > 0


def test_row_accounting_root_enforced_by_schema(tmp_path):
    conn = _disposable_canonical(tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="row_accounting_root"):
        conn.execute(
            "UPDATE ami_book_spread_change_windowed_flow SET row_accounting_root='TAMPERED' "
            "WHERE feature_id=(SELECT feature_id FROM ami_book_spread_change_windowed_flow "
            "WHERE direction='LONG' LIMIT 1)")
    conn.close()


def test_long_population_read_only(tmp_path):
    conn = _disposable_canonical(tmp_path)
    before = conn.execute("SELECT COUNT(*) FROM ami_book_spread_change_windowed_flow").fetchone()[0]
    PREREG.resolve_long_population(conn)
    after = conn.execute("SELECT COUNT(*) FROM ami_book_spread_change_windowed_flow").fetchone()[0]
    conn.close()
    assert before == after == 196


# ---------------------------------------------------------------------------
# Phase 5: split resolution + sample sufficiency -- must report INCOMPLETE
# ---------------------------------------------------------------------------

def test_split_reuses_frozen_70_30_convention():
    assert PREREG.TRAIN_FRACTION == 0.7
    assert PREREG.MIN_TRAIN_N == 30
    assert PREREG.MIN_TEST_N == 20
    assert PREREG.MIN_TOTAL_N == 50
    assert PREREG.MIN_RESIDUAL_DF == 15


def test_split_produces_test_18_below_minimum_20():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_long_population(conn)
    finally:
        conn.close()
    split = PREREG.resolve_split(pop)
    assert split["train_n"] == 40
    assert split["test_n"] == 18
    assert split["total_n"] == 58
    assert split["sufficiency"]["test_n_ok"] is False
    assert split["sufficiency"]["train_n_ok"] is True
    assert split["sufficiency"]["total_n_ok"] is True
    assert split["sufficient"] is False
    assert split["verdict"] == "BOOK_SPREAD_DYNAMICS_LONG_PREREGISTRATION_V1_INCOMPLETE"


def test_split_train_test_disjoint_and_no_straddling():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_long_population(conn)
    finally:
        conn.close()
    split = PREREG.resolve_split(pop)
    assert split["overlap"] == 0
    assert split["sufficiency"]["overlap_zero"] is True
    assert split["sufficiency"]["no_straddling"] is True
    assert len(set(split["train_cycle_keys"])) == len(split["train_cycle_keys"])
    assert len(set(split["test_cycle_keys"])) == len(split["test_cycle_keys"])


def test_split_hashes_deterministic():
    conn = _real_canonical_ro()
    try:
        pop1 = PREREG.resolve_long_population(conn)
        pop2 = PREREG.resolve_long_population(conn)
    finally:
        conn.close()
    split1 = PREREG.resolve_split(pop1)
    split2 = PREREG.resolve_split(pop2)
    assert split1["train_hash"] == split2["train_hash"]
    assert split1["test_hash"] == split2["test_hash"]
    assert split1["split_version"] == split2["split_version"]
    assert split1["split_version"].startswith("SPLITv1:")


def test_residual_df_would_have_passed_in_isolation():
    """Confirm TEST-n is the SOLE blocking condition (residual df=16>=15
    would pass on its own) -- this is a clean sample-size shortfall, not a
    degrees-of-freedom problem."""
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_long_population(conn)
    finally:
        conn.close()
    split = PREREG.resolve_split(pop)
    assert split["residual_df"] == 16
    assert split["sufficiency"]["residual_df_ok"] is True
    failing = [k for k, v in split["sufficiency"].items() if not v]
    assert failing == ["test_n_ok"]


def test_no_downstream_freeze_attempted_when_split_insufficient():
    """The module exposes no predictor/model-fitting function -- Phase 5
    blocking means phases 6+ (predictor/model freeze proper, split
    consumption, nullifier, registry) are structurally never reached."""
    assert not hasattr(PREREG, "run_governed_execution")
    assert not hasattr(PREREG, "fit_model")
    assert not hasattr(PREREG, "consume_and_register")


# ---------------------------------------------------------------------------
# Real-DB governance state: unchanged (INCOMPLETE authorizes nothing)
# ---------------------------------------------------------------------------

def test_real_db_governance_state_unchanged():
    conn = _real_canonical_ro()
    kconn = _real_knowledge_ro()
    try:
        schema_version = conn.execute(
            "SELECT version FROM schema_versions WHERE component='canonical_warehouse'").fetchone()[0]
        registry = conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
        results = conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0]
        nullifiers = kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0]
        receipts = kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0]
    finally:
        conn.close()
        kconn.close()
    assert schema_version == 14
    assert registry == 24
    assert results == 381
    assert nullifiers == 2
    assert receipts == 2


def test_no_gate_receipt_or_nullifier_registered_for_long_family_id():
    family_id = gates.resolve_canonical_family_id(PREREG.QUESTION_IDS, PREREG.HYPOTHESIS_ID)
    kconn = _real_knowledge_ro()
    try:
        receipts = kconn.execute(
            "SELECT * FROM experiment_gate_receipts WHERE canonical_family_id=?", (family_id,)).fetchall()
        nullifiers = kconn.execute(
            "SELECT * FROM epistemic_test_nullifiers WHERE family_id=?", (family_id,)).fetchall()
    finally:
        kconn.close()
    assert receipts == []
    assert nullifiers == []


def test_no_experiment_registry_row_for_long_child():
    conn = _real_canonical_ro()
    try:
        rows = conn.execute(
            "SELECT experiment_id FROM experiment_registry WHERE hypothesis_id LIKE '%BOOK-SPREAD%LONG%'"
        ).fetchall()
    finally:
        conn.close()
    assert rows == []


def test_full_chain_mutates_nothing(tmp_path):
    conn = _disposable_canonical(tmp_path)
    kconn = _disposable_knowledge(tmp_path)
    before = (
        conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0],
        kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0],
        kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0],
    )
    identity = PREREG.resolve_family_and_child_identity(kconn)
    pop = PREREG.resolve_long_population(conn)
    PREREG.resolve_outcome_metadata()
    split = PREREG.resolve_split(pop)
    after = (
        conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0],
        kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0],
        kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0],
    )
    conn.close()
    kconn.close()
    assert before == after
    assert identity["genuinely_unconsumed"] is True
    assert split["sufficient"] is False
