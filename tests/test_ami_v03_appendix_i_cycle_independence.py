"""v0.3 Appendix I — cycle-independence enforcement mutations (I-3 / I-4 / I-12).

FIRST TEST BATCH of the v0.3 gap programme (docs/ami/AMI_GAP_ANALYSIS.md §6.1).
Chosen because it sits at the centre of Appendix J P0 ("cycle deduplication;
overlap and censoring") and is testable against engines that ALREADY EXIST --
it does NOT wait for the unbuilt Phase E forward observatory.

WHAT THIS FILE PROVES (and deliberately does not fix)
-----------------------------------------------------
The repo HAS the capability to keep cycles independent
(`ami/identity/split_utils.chronological_group_split`, whitepaper §65.1:
"the same cycle must never be split across train and validation") and it is
unit-tested. But the CANONICAL leakage guard that experiments actually call --
`ami/research/registry.assert_no_overlap` -- is **cycle-blind**: it compares
event-id sets only. Two DIFFERENT events belonging to the SAME cycle may land
in train and test and nothing raises.

So the gap is not "no algorithm", it is "no gate":

    capability EXISTS  +  nothing FORCES its use  =  Appendix I-4 unmet

This is precisely the shape the repo already solved once, for a different
discipline -- see `ami/governance/epistemic_gates.py:20`:
    "`researcher_exposure_ledger` RECORDED exposure but nothing BLOCKED a
     family from re-reading the same TEST cycles under a new experiment id."
-- which was closed by the M-0033 enforcement wiring. The precedent and its
solution template therefore already exist in-repo.

XFAIL CONTRACT — and its honest limits
--------------------------------------
Tests named `*__I{n}_spec_requirement` are `xfail(strict=True)`. They encode
what Appendix I REQUIRES, which the system does not yet do.

LIMIT (independent review F-2): these three tests are SELF-CONTAINED -- they
hardcode the absence of a guard (`guard = None`, a literal `cycle_of` dict, a
literal `reported` dict) instead of calling a real repo API, because the API
they would call DOES NOT EXIST yet. Consequence: if someone wires a cycle-aware
guard elsewhere in the codebase, these tests will NOT auto-XPASS -- they will
keep evaluating their own hardcoded logic and stay xfail forever. So this is
NOT an automatic drift alarm. Closing the gap REQUIRES rewriting these test
bodies to call the new API, jointly with the gap analysis. The `reason=` string
on each marker names what to wire; treat it as the instruction, not as a
tripwire that will fire on its own.

`strict=True` is still worth keeping: it prevents a silent XPASS from being
ignored once the bodies DO call a real API.

Tests named `*__I{n}_current_gap` assert the CURRENT (non-compliant) behaviour
so that an accidental silent change is caught too.

READ-ONLY / NO SIDE EFFECTS: no store is written, no canonical verdict or
KnowledgeObject is touched, no process is started. The single real-data test
opens canonical.sqlite with `mode=ro` and skips if absent.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ami.constitution import ConstitutionViolation
from ami.identity.split_utils import chronological_group_split
from ami.research.registry import assert_no_overlap

CANONICAL_DB = Path("data/ami/canonical.sqlite")

# Two events that belong to ONE cycle, plus one event from a later cycle.
# Straddle construction: e1 and e2 share cycle C1 but are far apart in time,
# so a naive chronological (event-level) split puts them on opposite sides.
_HOUR = 3_600_000
EVENTS = [
    {"event_id": "E1", "cycle_id": "C1", "ts": 1_000 * _HOUR},
    {"event_id": "E2", "cycle_id": "C1", "ts": 1_010 * _HOUR},
    {"event_id": "E3", "cycle_id": "C2", "ts": 1_020 * _HOUR},
    {"event_id": "E4", "cycle_id": "C3", "ts": 1_030 * _HOUR},
]


def _by_cycle(r):
    return r["cycle_id"]


def _by_ts(r):
    return r["ts"]


# --------------------------------------------------------------------------
# Positive control: the capability exists and works when it is actually used.
# --------------------------------------------------------------------------

def test_group_split_never_straddles_a_cycle__positive_control():
    """§65.1 holds *when* chronological_group_split is the splitter."""
    res = chronological_group_split(
        EVENTS, group_key_fn=_by_cycle, timestamp_fn=_by_ts,
        val_ratio=0.34, test_ratio=0.34,
    )
    placements: dict[str, set[str]] = {}
    for split_name, rows in (("train", res.train), ("val", res.val), ("test", res.test)):
        for r in rows:
            placements.setdefault(r["cycle_id"], set()).add(split_name)

    straddlers = {c: s for c, s in placements.items() if len(s) > 1}
    assert straddlers == {}, f"cycle straddled splits: {straddlers}"
    assert len(placements) == 3, placements

    # C1's two events must have travelled together into the same split, even
    # though 10h separates them and a naive chronological cut would split them.
    landed = {r["event_id"]: name
              for name, rows in (("train", res.train), ("val", res.val), ("test", res.test))
              for r in rows}
    assert landed["E1"] == landed["E2"], landed
    assert set(landed) == {"E1", "E2", "E3", "E4"}, landed


# --------------------------------------------------------------------------
# I-4 — "splitting one cycle across train and validation"
# --------------------------------------------------------------------------

def test_assert_no_overlap_is_cycle_blind__I4_current_gap():
    """CURRENT behaviour: the canonical guard passes a cycle-straddling split.

    E1 and E2 are the same cycle (C1). Putting E1 in train and E2 in test is
    exactly the Appendix I-4 violation. `assert_no_overlap` does not raise,
    because it only diffs event-id sets and knows nothing about cycles.
    """
    train_ids = {"E1"}
    test_ids = {"E2", "E3"}

    assert not (train_ids & test_ids), "precondition: no event-id overlap"
    # Does NOT raise -- this is the gap, asserted so it cannot change silently.
    assert_no_overlap(train_ids, test_ids)


def test_assert_no_overlap_still_catches_identical_event_ids():
    """Non-regression: the guard's *existing* (event-level) contract holds."""
    with pytest.raises(ConstitutionViolation):
        assert_no_overlap({"E1", "E2"}, {"E2", "E3"})


@pytest.mark.xfail(strict=True, reason="Appendix I-4: no cycle-aware leakage gate exists "
                                       "(gap analysis §4). Wire a cycle-aware guard to close.")
def test_assert_no_overlap_rejects_cycle_straddle__I4_spec_requirement():
    """REQUIRED by Appendix I-4: the system must reject or detect a split that
    puts two events of one cycle on opposite sides -- given the cycle mapping.

    There is currently no API that accepts the mapping at all, which is the
    finding. Closing this means a guard that consumes (ids -> cycle_id).
    """
    cycle_of = {"E1": "C1", "E2": "C1", "E3": "C2"}
    train_ids, test_ids = {"E1"}, {"E2", "E3"}

    shared = {cycle_of[i] for i in train_ids} & {cycle_of[i] for i in test_ids}
    if shared:
        raise AssertionError(
            f"cycle(s) {sorted(shared)} straddle train/test, but no repo gate raised. "
            "assert_no_overlap is cycle-blind; split_utils is optional and was not called."
        )


# --------------------------------------------------------------------------
# I-3 — "counting multiple events from one cycle as independent"
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="Appendix I-3: no independent-N guard exists. "
                                       "W1 MEASURED the funnel (252 event -> 167 cycle) but "
                                       "nothing blocks an experiment reporting event-N as "
                                       "independent-N. See OD-011 / CT-005.")
def test_independent_n_must_not_equal_event_n__I3_spec_requirement():
    """REQUIRED by Appendix I-3: reporting N=4 events from 3 cycles as 4
    independent observations must be rejected or flagged.

    OD-011 records the real-world damage of this gap: 8 KnowledgeObjects were
    built on cycle-unadjusted N (anchor_to_cycle_ratio=0.66, ~1.5x inflation);
    7 are still RECOMPUTE_REQUIRED.
    """
    event_n = len(EVENTS)
    cycle_n = len({e["cycle_id"] for e in EVENTS})
    assert event_n != cycle_n, "precondition: this sample IS inflated"

    guard = None  # no such callable exists in the repo
    if guard is None:
        raise AssertionError(
            f"event_n={event_n} vs independent cycle_n={cycle_n}: no guard rejects "
            "reporting the inflated count as independent evidence."
        )


# --------------------------------------------------------------------------
# I-12 — "hiding cycle_N while displaying event_N"
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="Appendix I-12: no schema/API requires cycle_N to be "
                                       "reported alongside event_N. experiment_results carries "
                                       "free-form metrics; nothing enforces the pair.")
def test_reporting_event_n_requires_cycle_n__I12_spec_requirement():
    """REQUIRED by Appendix I-12: a result that displays event_N must not be
    able to omit cycle_N."""
    reported = {"event_n": 252}  # a result payload omitting cycle_n
    if "cycle_n" not in reported:
        raise AssertionError(
            "result reports event_n without cycle_n and no validator rejected it."
        )


# --------------------------------------------------------------------------
# Real-data corroboration (read-only, skips if the store is absent).
# --------------------------------------------------------------------------

def test_real_canonical_store_confirms_event_to_cycle_inflation():
    """The gap is not hypothetical: the live canonical store carries a real
    event->cycle deflation, which is exactly what an event-level N over-counts.
    """
    if not CANONICAL_DB.exists():
        pytest.skip("canonical.sqlite not present")

    conn = sqlite3.connect(f"file:{CANONICAL_DB.as_posix()}?mode=ro", uri=True)
    try:
        event_n = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
        cycle_n = conn.execute("SELECT COUNT(*) FROM ami_cycles").fetchone()[0]
    finally:
        conn.close()

    if event_n == 0 or cycle_n == 0:
        pytest.skip("canonical store not seeded")

    assert cycle_n < event_n, (event_n, cycle_n)
    ratio = cycle_n / event_n
    # W1 (E-W1-CYCLE-INTEGRITY-001) froze this funnel at 252 -> 167 (~0.66).
    # A wide band: this asserts the inflation is real, not the exact figure.
    assert 0.4 < ratio < 0.95, f"unexpected anchor_to_cycle_ratio={ratio:.3f}"
