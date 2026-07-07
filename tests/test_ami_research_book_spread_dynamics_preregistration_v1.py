"""BATCH-BOOK-SPREAD-DYNAMICS-PREREGISTRATION-V1 -- focused tests.

DISPOSABLE_DB_ONLY: every test that touches state runs against disposable
COPIES of the real canonical.sqlite / knowledge.sqlite (never the real
paths). Read-only checks against the real DBs use `mode=ro` URIs only.
Outcome-blind throughout -- no test reads `endpoint_return_bps`/`mfe_bps`.

Run: pytest tests/test_ami_research_book_spread_dynamics_preregistration_v1.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import shutil
import sqlite3

import pytest

from ami.governance import epistemic_gates as gates
from ami.research import book_spread_dynamics_preregistration_v1 as PREREG
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
# Identity / constant enforcement
# ---------------------------------------------------------------------------

def test_family_and_child_constants():
    assert PREREG.FAMILY_NAME == "FAM_BOOK_SPREAD_DYNAMICS"
    assert PREREG.CHILD_ID == "H-BOOK-SPREAD-CHANGE-BPS-W300-V1"
    assert PREREG.FORMULA_VERSION == "BOOK_SPREAD_CHANGE_BPS_W300_V1"
    assert PREREG.SPECIFICATION_HASH == (
        "ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212")
    assert PREREG.ROW_ACCOUNTING_ROOT == (
        "33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31")
    assert PREREG.EXPECTED_SCHEMA_VERSION == 14


def test_family_id_deterministic():
    a = gates.resolve_canonical_family_id(PREREG.QUESTION_IDS, PREREG.HYPOTHESIS_ID)
    b = gates.resolve_canonical_family_id(PREREG.QUESTION_IDS, PREREG.HYPOTHESIS_ID)
    assert a == b
    assert a.startswith("FAMv1:")


# ---------------------------------------------------------------------------
# Phase 1: family/child identity, graveyard, prior exposure (real DB, RO)
# ---------------------------------------------------------------------------

def test_graveyard_clean_against_real_knowledge_db():
    kconn = _real_knowledge_ro()
    try:
        result = PREREG.resolve_family_and_child_identity(kconn)
    finally:
        kconn.close()
    assert result["graveyard_clean"] is True
    assert result["graveyard_hits"] == []


def test_prior_exposure_clean_against_real_knowledge_db():
    kconn = _real_knowledge_ro()
    try:
        result = PREREG.resolve_family_and_child_identity(kconn)
    finally:
        kconn.close()
    assert result["prior_test_exposure_clean"] is True
    assert result["prior_test_exposure"] == []
    assert result["existing_gate_receipts_for_family"] == []
    assert result["genuinely_unconsumed"] is True


def test_resolve_family_and_child_identity_read_only(tmp_path):
    """Disposable knowledge copy: prove the resolution function performs no
    write (graveyard/nullifier/receipt tables unchanged)."""
    kconn = _disposable_knowledge(tmp_path)
    before = {
        "fingerprints": kconn.execute("SELECT COUNT(*) FROM graveyard_slash_fingerprints").fetchone()[0],
        "nullifiers": kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0],
        "receipts": kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0],
    }
    PREREG.resolve_family_and_child_identity(kconn)
    after = {
        "fingerprints": kconn.execute("SELECT COUNT(*) FROM graveyard_slash_fingerprints").fetchone()[0],
        "nullifiers": kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0],
        "receipts": kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0],
    }
    kconn.close()
    assert before == after


# ---------------------------------------------------------------------------
# Population resolution (real DB, RO) -- exact canonical accounting
# ---------------------------------------------------------------------------

def test_population_matches_frozen_196_97():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_population(conn)
    finally:
        conn.close()
    assert pop["feature_row_count"] == 196
    assert pop["representative_count"] == 97
    assert pop["matches_frozen_196_97"] is True


def test_population_is_direction_mixed():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_population(conn)
    finally:
        conn.close()
    assert pop["is_direction_mixed"] is True
    assert pop["direction_breakdown_representatives"] == {"LONG": 70, "SHORT": 27}
    assert sum(pop["direction_breakdown_representatives"].values()) == 97
    assert pop["direction_breakdown_all"] == {"LONG": 120, "SHORT": 76}
    assert sum(pop["direction_breakdown_all"].values()) == 196


def test_population_row_accounting_root_enforced(tmp_path):
    """A row with a drifted row_accounting_root must be rejected -- defense
    in depth: the schema's own CHECK constraint fires first (stronger than
    the module-level guard), so the UPDATE itself is what raises."""
    conn = _disposable_canonical(tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="row_accounting_root"):
        conn.execute(
            "UPDATE ami_book_spread_change_windowed_flow SET row_accounting_root='TAMPERED' "
            "WHERE feature_id=(SELECT feature_id FROM ami_book_spread_change_windowed_flow LIMIT 1)")
    conn.close()


def test_population_formula_version_drift_detected_by_module_guard():
    """The module's own drift guard (`resolve_population`) is exercised
    directly with an in-memory synthetic row set, independent of the
    schema's CHECK constraint (which the prior test already proves fires
    first on the real table)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE ami_book_spread_change_windowed_flow (anchor_id TEXT, direction TEXT, "
        "cycle_id TEXT, is_cycle_representative INTEGER, formula_version TEXT, row_accounting_root TEXT)")
    conn.execute(
        "INSERT INTO ami_book_spread_change_windowed_flow VALUES ('A','LONG','C1',1,'WRONG_VERSION',?)",
        (PREREG.ROW_ACCOUNTING_ROOT,))
    with pytest.raises(PREREG.PreregistrationIncomplete):
        PREREG.resolve_population(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Phase 2: outcome-ID resolution (metadata only, no outcome VALUE read)
# ---------------------------------------------------------------------------

def test_outcome_reused_verbatim_not_derived():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_population(conn)
        reps = [r[0] for r in conn.execute(
            "SELECT anchor_id FROM ami_book_spread_change_windowed_flow "
            "WHERE is_cycle_representative=1").fetchall()]
        outcome = PREREG.resolve_outcome_metadata(conn, reps)
    finally:
        conn.close()
    assert outcome["outcome_id"] == "endpoint_return_bps@swing_24h"
    assert outcome["reused_verbatim"] is True
    assert outcome["newly_derived"] is False
    assert outcome["dependent_variable_type"] == "continuous"
    assert outcome["structurally_compatible"] is True
    assert pop["representative_count"] == len(reps)


def test_outcome_metadata_reads_no_value_columns():
    """AST-level guard, scoped to `.execute()`-family call arguments only
    (not docstrings/comments, which legitimately name these columns when
    explaining what is NOT read): this module's SQL literals must never
    SELECT endpoint_return_bps or mfe_bps."""
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
    assert hits == [], f"forbidden outcome-value column referenced in a SQL execute() literal: {hits}"


def test_outcome_coverage_uses_effective_status_not_raw_values(tmp_path):
    """The coverage query only ever groups by observation_status -- confirm
    no row in the returned dict is a numeric return value."""
    conn = _disposable_canonical(tmp_path)
    reps = [r[0] for r in conn.execute(
        "SELECT anchor_id FROM ami_book_spread_change_windowed_flow "
        "WHERE is_cycle_representative=1").fetchall()]
    outcome = PREREG.resolve_outcome_metadata(conn, reps)
    conn.close()
    assert set(outcome["coverage"].keys()) <= {
        "OK", "MISSING_INTERNAL_GAP", "EXCLUDED_NO_HORIZON_DATA", "SAME_CANDLE_UNKNOWN"}


# ---------------------------------------------------------------------------
# Phase 3: direction/sign resolution -- must report INCOMPLETE, not invent one
# ---------------------------------------------------------------------------

def test_direction_and_sign_reports_incomplete_for_mixed_population():
    conn = _real_canonical_ro()
    try:
        pop = PREREG.resolve_population(conn)
    finally:
        conn.close()
    result = PREREG.resolve_direction_and_sign(pop)
    assert result["resolved"] is False
    assert result["verdict"] == "BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE"
    assert "path_1_restrict_population_blocked" in result
    assert "path_2_flip_outcome_blocked" in result
    assert "path_3_interaction_or_subgroup_blocked" in result


def test_direction_and_sign_resolves_trivially_for_single_direction_population():
    """Sanity check on the function's own logic branch (not a claim about
    the real population, which IS mixed): a single-direction population
    must not be reported as unresolved."""
    fake_pop = {
        "is_direction_mixed": False,
        "direction_breakdown_representatives": {"LONG": 97},
    }
    result = PREREG.resolve_direction_and_sign(fake_pop)
    assert result["resolved"] is True


def test_direction_and_sign_does_not_mutate_any_database(tmp_path):
    """Phase-3 resolution is pure computation on an already-fetched dict --
    no connection is even passed to it. Confirm no disposable DB write
    occurs anywhere in the family/child/population/outcome/direction chain."""
    conn = _disposable_canonical(tmp_path)
    kconn = _disposable_knowledge(tmp_path)
    before_reg = conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
    before_res = conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0]
    before_null = kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0]
    before_receipt = kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0]

    identity = PREREG.resolve_family_and_child_identity(kconn)
    pop = PREREG.resolve_population(conn)
    reps = [r[0] for r in conn.execute(
        "SELECT anchor_id FROM ami_book_spread_change_windowed_flow "
        "WHERE is_cycle_representative=1").fetchall()]
    PREREG.resolve_outcome_metadata(conn, reps)
    direction_result = PREREG.resolve_direction_and_sign(pop)

    after_reg = conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
    after_res = conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0]
    after_null = kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0]
    after_receipt = kconn.execute("SELECT COUNT(*) FROM experiment_gate_receipts").fetchone()[0]
    conn.close()
    kconn.close()

    assert (before_reg, before_res, before_null, before_receipt) == (
        after_reg, after_res, after_null, after_receipt)
    assert identity["genuinely_unconsumed"] is True
    assert direction_result["resolved"] is False


# ---------------------------------------------------------------------------
# Real-DB governance state: unchanged by this batch (no nullifier/receipt/
# registry row is authorized to be created for an INCOMPLETE preregistration)
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


def test_no_family_id_registered_for_this_attempt():
    """Confirm this INCOMPLETE attempt left no gate receipt / nullifier
    behind under its own resolved family_id -- an INCOMPLETE closure must
    be indistinguishable, governance-wise, from never having attempted it."""
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
