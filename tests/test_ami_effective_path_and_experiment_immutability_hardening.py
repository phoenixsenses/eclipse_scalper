"""BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING --
cross-cutting integration proof (GOAL D items 9 and 10).

This batch is a SAFETY HARDENING batch, not a research wave: it adds a
fail-closed path-version contract (GOAL A) and an immutable-write guard
(GOAL B) around the six existing, already-completed W8 experiments (GOAL C),
and must prove -- against the real, already-repaired canonical.sqlite (via
conftest.py's session-scoped isolated copy, never the real file) -- that
none of their stored content changed as a result.

NO_OUTCOME_ANALYSIS / NO_NEW_EXPERIMENT_RESULT: calling each module's
freeze_and_record() below recomputes each experiment's cells/medians/verdict
in Python memory (unavoidable -- that recomputation IS the mechanism that
proves byte-identical equality against the already-stored row, via the new
ami.warehouse.experiment_ledger.record_experiment_registry/
record_experiment_results NOOP_IDENTICAL path introduced by GOAL B). No
newly-computed value is ever interpreted, published, or stored under a new
identity here -- every one of these calls must resolve to a no-op write
(proven explicitly below), exactly the same "real-data smoke +
idempotent rerun" pattern each module's own existing test file already
established before this batch.

Run: pytest tests/test_ami_effective_path_and_experiment_immutability_hardening.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import inspect

_FORBIDDEN_TERMS = ("p_value", "win_rate", "alpha_claim", "economic_gate_change", "threshold_sweep")


def test_no_outcome_or_economic_terms_in_hardening_module_source():
    import ami.warehouse.experiment_ledger as ledger_mod

    src = inspect.getsource(ledger_mod).lower()
    hits = [t for t in _FORBIDDEN_TERMS if t in src]
    assert hits == [], f"forbidden terms found in experiment_ledger.py: {hits}"


_ALL_EXPERIMENT_IDS = (
    "E-W8-HOLD-BASELINE-001",
    "E-W8-VOL-NORMALIZED-BASELINE-001",
    "E-W8-HOLD-BASELINE-002-SHORT-EXPANDED",
    "E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED",
    "E-W8-LONG-TIMING-STRUCTURE-001",
    "E-W8-LONG-NESTED-PATH-ACCUMULATION-001",
    "E-W8-LONG-NESTED-PATH-ACCUMULATION-002-CANDLE-REPAIR",
)


def _snapshot(conn, experiment_id):
    reg = conn.execute(
        "SELECT * FROM experiment_registry WHERE experiment_id=?", (experiment_id,)
    ).fetchone()
    results = conn.execute(
        "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? ORDER BY metric_name",
        (experiment_id,),
    ).fetchall()
    return (reg, tuple(results))


# E-W8-HOLD-BASELINE-001 and E-W8-VOL-NORMALIZED-BASELINE-001 are the two experiment_ids where a
# GENUINE, unrelated population drift exists: both span direction=SHORT with no setup_id filter, so
# BATCH-SHORT-NOISY-V1-CANON-BACKFILL's 54 new SHORT signals (added AFTER these two were originally
# frozen) inflate raw_signal_n on every fresh recompute, regardless of the GOAL C path-version pin
# (that pin only controls which PATH rows are read; the SIGNAL population itself grew). For these
# two, "byte-identical" is proven by the fact that freeze_and_record() now correctly REFUSES to
# write (ImmutableExperimentConflict) rather than silently absorbing the drift -- see
# tests/test_ami_research_w8_hold_baseline.py::test_freeze_and_record_fails_closed_on_real_population_drift
# and the analogous test in test_ami_research_w8_vol_normalized_baseline.py.
_DRIFTED_EXPERIMENT_IDS = ("E-W8-HOLD-BASELINE-001", "E-W8-VOL-NORMALIZED-BASELINE-001")
_REPRODUCIBLE_EXPERIMENT_IDS = tuple(eid for eid in _ALL_EXPERIMENT_IDS if eid not in _DRIFTED_EXPERIMENT_IDS)


def test_reproducible_experiment_ids_byte_identical_across_hardened_rerun():
    """GOAL D item 9 for the five experiment_ids whose population has NOT
    drifted since freezing (SHORT-EXPANDED's own population already included
    the SHORT_NOISY_V1 backfill from inception; the three LONG-only
    experiments were never affected by a SHORT-only signal backfill).
    Snapshots every one of these rows BEFORE re-invoking each module's
    freeze_and_record() against the real (isolated-copy) canonical.sqlite,
    then AFTER, and asserts byte-for-byte equality -- proving GOAL A's
    path-version pins (GOAL C) plus GOAL B's immutable-write guard together
    reproduce every frozen result exactly, with zero silent drift."""
    import ami.warehouse.schema as schema_mod
    import ami.research.w8_short_expanded_baseline as w8se
    import ami.research.w8_long_timing_structure as w8lt
    import ami.research.w8_long_nested_path_accumulation as w8np
    import ami.research.w8_long_nested_path_accumulation_002_candle_repair as w8np2

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        before = {eid: _snapshot(conn, eid) for eid in _REPRODUCIBLE_EXPERIMENT_IDS}

        # dependency order: LONG-NESTED-PATH-ACCUMULATION-001 before its -002-CANDLE-REPAIR (reads it back)
        w8se.freeze_and_record(conn)
        w8lt.freeze_and_record(conn)
        w8np.freeze_and_record(conn)
        w8np2.freeze_and_record(conn)

        after = {eid: _snapshot(conn, eid) for eid in _REPRODUCIBLE_EXPERIMENT_IDS}

        for eid in _REPRODUCIBLE_EXPERIMENT_IDS:
            assert before[eid][0] is not None, f"{eid} was expected to already exist in the real DB"
            assert after[eid] == before[eid], f"{eid} content changed across the hardened rerun -- NOT byte-identical"
    finally:
        conn.close()


def test_drifted_experiment_ids_fail_closed_and_remain_byte_identical():
    """GOAL D item 9 for the two drifted experiment_ids: freeze_and_record()
    must raise ImmutableExperimentConflict (never silently write), and the
    stored row must be byte-identical before and after the failed attempt."""
    import ami.warehouse.schema as schema_mod
    import ami.research.w8_hold_baseline as w8h
    import ami.research.w8_vol_normalized_baseline as w8v
    from ami.warehouse.experiment_ledger import ImmutableExperimentConflict

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        before = {eid: _snapshot(conn, eid) for eid in _DRIFTED_EXPERIMENT_IDS}

        for module in (w8h, w8v):
            try:
                module.freeze_and_record(conn)
                raised = False
            except ImmutableExperimentConflict:
                raised = True
            assert raised, f"{module.__name__}.freeze_and_record() was expected to fail closed on real drift"

        after = {eid: _snapshot(conn, eid) for eid in _DRIFTED_EXPERIMENT_IDS}
        for eid in _DRIFTED_EXPERIMENT_IDS:
            assert before[eid][0] is not None, f"{eid} was expected to already exist in the real DB"
            assert after[eid] == before[eid], f"{eid} content changed despite the write being refused"
    finally:
        conn.close()
