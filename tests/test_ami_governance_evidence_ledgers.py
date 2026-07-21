"""Tests — Phase A evidence-safety ledger writers (gap-analysis A1 + A3).

Covers the classification law itself, the idempotency contract, the refusal to
fabricate un-establishable fields, and the end-to-end wiring through
`register_experiment_with_gates` (including the rollback invariant: a blocked
registration must leave NO ledger row).
"""
from __future__ import annotations

import sqlite3

import pytest

from ami.governance import evidence_ledgers as EL
from ami.warehouse.schema import connect, init_schema


# --------------------------------------------------------------------------
# classify_evidence_status — the law, in isolation
# --------------------------------------------------------------------------

def _c(**kw):
    base = dict(prior_consumption_count=0, is_rerun_of_self=False, supersession_used=False,
                hypothesis_origin_split=None, split_version="split-v1")
    base.update(kw)
    return EL.classify_evidence_status(**base)


def test_first_look_is_independent():
    status, fresh, ceiling = _c()
    assert status == EL.INDEPENDENT_EVIDENCE
    assert fresh is False
    assert ceiling == EL.CEILING_CONFIRMATORY


def test_prior_consumption_by_another_experiment_is_reuse():
    status, fresh, ceiling = _c(prior_consumption_count=1, supersession_used=True)
    assert status == EL.REUSED_EVIDENCE
    assert fresh is True
    assert ceiling == EL.CEILING_NO_UPGRADE


def test_rerun_of_same_experiment_stays_independent():
    """Idempotent replay of one frozen experiment is a NOOP, not a second look."""
    status, _fresh, _c2 = _c(prior_consumption_count=1, is_rerun_of_self=True)
    assert status == EL.INDEPENDENT_EVIDENCE


def test_hypothesis_born_on_the_split_it_is_confirmed_on_is_contaminated():
    status, fresh, ceiling = _c(hypothesis_origin_split="split-v1", split_version="split-v1")
    assert status == EL.CONTAMINATED_FOR_CONFIRMATION
    assert fresh is True
    assert ceiling == EL.CEILING_FORWARD_ONLY


def test_origin_contamination_beats_first_look():
    """Being the first to consume the split does NOT repair being born on it."""
    status, _f, _c2 = _c(prior_consumption_count=0, hypothesis_origin_split="split-v1",
                         split_version="split-v1")
    assert status == EL.CONTAMINATED_FOR_CONFIRMATION


def test_born_on_split_and_split_already_spent_requires_fresh_forward():
    status, fresh, ceiling = _c(prior_consumption_count=1, supersession_used=True,
                                hypothesis_origin_split="split-v1", split_version="split-v1")
    assert status == EL.FORWARD_ONLY_CONFIRMATION_REQUIRED
    assert fresh is True
    assert ceiling == EL.CEILING_FORWARD_ONLY


def test_hypothesis_born_on_a_different_split_is_not_contaminated():
    status, _f, _c2 = _c(hypothesis_origin_split="split-v0", split_version="split-v1")
    assert status == EL.INDEPENDENT_EVIDENCE


def test_undeclared_origin_is_not_silently_treated_as_clean_or_dirty():
    """UNDECLARED must not fabricate a contamination verdict either way: the
    status falls back to what IS knowable (nullifier history)."""
    clean, _f, _c1 = _c(hypothesis_origin_split=EL.UNDECLARED_ORIGIN_SPLIT)
    dirty, _f2, _c2 = _c(hypothesis_origin_split=EL.UNDECLARED_ORIGIN_SPLIT,
                         prior_consumption_count=1, supersession_used=True)
    assert clean == EL.INDEPENDENT_EVIDENCE
    assert dirty == EL.REUSED_EVIDENCE


def test_all_four_spec_statuses_are_reachable():
    """Whitepaper §70.1 names four values; a writer that can only ever emit
    two would be a checkbox, not a ledger."""
    reached = {
        _c()[0],
        _c(prior_consumption_count=1, supersession_used=True)[0],
        _c(hypothesis_origin_split="split-v1")[0],
        _c(prior_consumption_count=1, supersession_used=True, hypothesis_origin_split="split-v1")[0],
    }
    assert reached == set(EL.ALL_EVIDENCE_STATUSES)


# --------------------------------------------------------------------------
# bonferroni
# --------------------------------------------------------------------------

def test_bonferroni_tightens_with_trials():
    assert EL.bonferroni_alpha(1) == pytest.approx(0.05)
    assert EL.bonferroni_alpha(5) == pytest.approx(0.01)
    assert EL.bonferroni_alpha(0) == pytest.approx(0.05), "must not divide by zero"


# --------------------------------------------------------------------------
# writers against a real schema
# --------------------------------------------------------------------------

@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "canonical.sqlite")
    init_schema(c)
    yield c
    c.close()


def _add_experiment(conn, experiment_id: str, family_id: str, hypothesis_id: str = "H-1") -> None:
    """Minimal experiment_registry row satisfying its full NOT NULL contract
    (frozen_population, dataset_hash, software_verdict, scientific_verdict,
    schema_version, provenance, created_ms, updated_ms)."""
    conn.execute(
        "INSERT INTO experiment_registry (experiment_id, question_ids, hypothesis_id,"
        " frozen_population, dataset_hash, software_verdict, scientific_verdict,"
        " schema_version, provenance, created_ms, updated_ms)"
        " VALUES (?,?,?,'pop','ds-1','PASSED','ANSWERED_SUPPORTED',1,'t',1,1)",
        (experiment_id, family_id, hypothesis_id))


def test_contamination_row_is_idempotent_upsert_not_append(conn):
    for _ in range(3):
        EL.record_evidence_contamination(
            conn, hypothesis_id="H-1", family_id="FAM_X", split_version="s1",
            prior_consumption_count=0, is_rerun_of_self=False, supersession_used=False)
    n = conn.execute("SELECT COUNT(*) FROM evidence_contamination").fetchone()[0]
    assert n == 1, "same hypothesis+split must not accumulate duplicate verdicts"


def test_contamination_upsert_preserves_created_ms_and_moves_updated_ms(conn):
    EL.record_evidence_contamination(
        conn, hypothesis_id="H-1", family_id="FAM_X", split_version="s1",
        prior_consumption_count=0, is_rerun_of_self=False, supersession_used=False, now_ms=1000)
    EL.record_evidence_contamination(
        conn, hypothesis_id="H-1", family_id="FAM_X", split_version="s1",
        prior_consumption_count=1, is_rerun_of_self=False, supersession_used=True, now_ms=2000)
    row = conn.execute(
        "SELECT created_ms, updated_ms, evidence_status FROM evidence_contamination").fetchone()
    assert row[0] == 1000 and row[1] == 2000
    assert row[2] == EL.REUSED_EVIDENCE, "status must reflect the newer facts"


def test_contamination_records_undeclared_origin_explicitly(conn):
    r = EL.record_evidence_contamination(
        conn, hypothesis_id="H-1", family_id="FAM_X", split_version="s1",
        prior_consumption_count=0, is_rerun_of_self=False, supersession_used=False)
    assert r["hypothesis_origin_split"] == EL.UNDECLARED_ORIGIN_SPLIT
    assert r["origin_declared"] is False
    stored = conn.execute("SELECT hypothesis_origin_split FROM evidence_contamination").fetchone()[0]
    assert stored == EL.UNDECLARED_ORIGIN_SPLIT, "must be recorded, not left NULL/guessed"


def test_mt_family_refuses_to_fabricate_unestablishable_fields(conn):
    _add_experiment(conn, "E-1", "FAM_X")
    r = EL.record_family_variant(conn, question_ids="FAM_X", experiment_id="E-1", is_rerun_of_self=False)
    row = conn.execute(
        "SELECT threshold_stability, researcher_freedom_score, minimum_economic_effect"
        " FROM mt_family_registry WHERE family_id=?", (EL.resolve_mt_family_id("FAM_X"),)).fetchone()
    assert row == (None, None, None), "un-establishable fields must stay NULL, never guessed"
    assert set(r["refused_fields"]) == {
        "threshold_stability", "researcher_freedom_score", "minimum_economic_effect"}


def test_mt_family_counts_distinct_experiments_and_tightens_alpha(conn):
    for eid in ("E-1", "E-2", "E-3", "E-4", "E-5"):
        _add_experiment(conn, eid, "FAM_X")
    r = EL.record_family_variant(conn, question_ids="FAM_X", experiment_id="E-5", is_rerun_of_self=False)
    assert r["variants_tested"] == 5
    assert r["effective_trials"] == 5
    assert r["family_adjusted_significance"] == pytest.approx(0.01)


def test_mt_family_recount_is_idempotent_not_incremented(conn):
    """Replaying the same registration must not inflate the family's trial
    count -- the whole point of a multiple-testing ledger is an honest k."""
    _add_experiment(conn, "E-1", "FAM_X")
    for _ in range(4):
        r = EL.record_family_variant(conn, question_ids="FAM_X", experiment_id="E-1", is_rerun_of_self=True)
    assert r["variants_tested"] == 1
    assert conn.execute("SELECT COUNT(*) FROM mt_family_registry").fetchone()[0] == 1


def test_mt_family_does_not_count_other_families(conn):
    for eid, fam in (("E-1", "FAM_X"), ("E-2", "FAM_Y"), ("E-3", "FAM_X")):
        _add_experiment(conn, eid, fam, hypothesis_id="H")
    r = EL.record_family_variant(conn, question_ids="FAM_X", experiment_id="E-3", is_rerun_of_self=False)
    assert r["variants_tested"] == 2


# --------------------------------------------------------------------------
# Regressions for two defects the UNIT tests missed and only an end-to-end
# run through the real gate exposed. Both were dead-code-in-production bugs:
# the feature looked implemented and could never fire.
# --------------------------------------------------------------------------

def test_raw_origin_split_is_resolved_before_comparison(conn):
    """DEFECT 1: the gate compares RESOLVED split identity ("SPLITv1:<hash>"),
    but callers declare the origin split as prose. Comparing the two directly
    can never match, so CONTAMINATED_FOR_CONFIRMATION was unreachable in
    production while every unit test passed (they used raw strings on both
    sides)."""
    from ami.governance.epistemic_gates import resolve_split_version
    prose = "chronological 70/30 by signal_birth_ts"
    r = EL.record_evidence_contamination(
        conn, hypothesis_id="H-1", family_id="FAM_X",
        split_version=resolve_split_version(prose),   # what the gate passes
        prior_consumption_count=0, is_rerun_of_self=False, supersession_used=False,
        hypothesis_origin_split=prose)                # what the caller declares
    assert r["evidence_status"] == EL.CONTAMINATED_FOR_CONFIRMATION
    assert r["fresh_forward_required"] is True


def test_already_resolved_origin_is_not_double_resolved(conn):
    from ami.governance.epistemic_gates import resolve_split_version
    resolved = resolve_split_version("chronological 70/30 by signal_birth_ts")
    r = EL.record_evidence_contamination(
        conn, hypothesis_id="H-1", family_id="FAM_X", split_version=resolved,
        prior_consumption_count=0, is_rerun_of_self=False, supersession_used=False,
        hypothesis_origin_split=resolved, origin_split_is_resolved=True)
    assert r["evidence_status"] == EL.CONTAMINATED_FOR_CONFIRMATION


def test_mt_family_is_question_ids_only_not_the_gate_family(conn):
    """DEFECT 2: the gate's family_id hashes question_ids AND hypothesis_id, so
    every hypothesis is its own family -> variants_tested permanently 1 and
    Bonferroni never tightens. The MT family must key on question_ids alone."""
    from ami.governance.epistemic_gates import resolve_canonical_family_id
    gate_a = resolve_canonical_family_id("FAM_X", "H-1")
    gate_b = resolve_canonical_family_id("FAM_X", "H-2")
    assert gate_a != gate_b, "precondition: gate families fork per hypothesis"
    assert EL.resolve_mt_family_id("FAM_X") == EL.resolve_mt_family_id("FAM_X")

    for eid, hyp in (("E-1", "H-1"), ("E-2", "H-2"), ("E-3", "H-3")):
        _add_experiment(conn, eid, "FAM_X", hypothesis_id=hyp)
        r = EL.record_family_variant(conn, question_ids="FAM_X", experiment_id=eid,
                                     is_rerun_of_self=False)
    assert r["variants_tested"] == 3, "three hypotheses in one FAM_* = three trials"
    assert r["family_adjusted_significance"] == pytest.approx(0.05 / 3)
    assert conn.execute("SELECT COUNT(*) FROM mt_family_registry").fetchone()[0] == 1


def test_mt_family_does_not_substring_merge_related_family_names(conn):
    """FAM_LONG must not absorb FAM_LONG_SHORT_TRANSITIONS' trials."""
    _add_experiment(conn, "E-1", "FAM_LONG")
    _add_experiment(conn, "E-2", "FAM_LONG_SHORT_TRANSITIONS")
    r = EL.record_family_variant(conn, question_ids="FAM_LONG", experiment_id="E-1",
                                 is_rerun_of_self=False)
    assert r["variants_tested"] == 1


def test_mt_family_id_is_whitespace_and_case_stable(conn):
    assert EL.resolve_mt_family_id("  FAM_X  ") == EL.resolve_mt_family_id("fam_x")
