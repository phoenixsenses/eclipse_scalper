"""PHASE 7A.1: tests for ami/lifecycle/canonical_schema.py (identity
algorithm, append-only transition ledger, validators, deterministic
rebuild). DISPOSABLE_DB_ONLY: every test here uses an in-memory or tmp_path
sqlite connection, never data/ami/canonical.sqlite.

Run: pytest tests/test_ami_lifecycle_canonical_schema.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import sqlite3

import pytest

from ami.lifecycle.canonical_schema import (
    LifecycleIntegrityViolation,
    classify_direction_from_setup_id,
    count_effective_closed_signals,
    effective_lifecycle_status,
    generate_signal_id,
    init_lifecycle_schema,
    insert_transition,
    rebuild_current_state,
    rollback_lifecycle_schema,
    validate_transition,
)

PROV = "test"


def _conn():
    conn = sqlite3.connect(":memory:")
    init_lifecycle_schema(conn)
    return conn


# ---- deterministic signal identity ----

def test_signal_id_deterministic_same_inputs_same_id():
    a = generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG", source_event_id="EVT-1")
    b = generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG", source_event_id="EVT-1")
    assert a == b


def test_signal_id_never_uses_row_id_or_pid():
    # calling twice with identical logical inputs must be stable regardless
    # of call order/process -- no os.getpid()/uuid4() call anywhere in the
    # module (checking for actual usage patterns, not the bare substring,
    # since the module's own docstrings legitimately discuss these concepts)
    import ami.lifecycle.canonical_schema as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert "uuid4(" not in src
    assert "import uuid" not in src
    assert "getpid(" not in src


# ---- restart identity stability ----

def test_signal_id_stable_across_fresh_calls():
    ids = {generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG", source_event_id="EVT-1")
           for _ in range(5)}
    assert len(ids) == 1


# ---- setup-version isolation ----

def test_signal_id_changes_with_setup_version():
    a = generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG", source_event_id="EVT-1")
    b = generate_signal_id("LONG_SILENCE", "setup-v2", "ETHUSDT", "LONG", source_event_id="EVT-1")
    assert a != b


# ---- source-event/cycle separation ----

def test_signal_id_does_not_depend_on_cycle_id():
    # independent_cycle_id is not even a parameter of generate_signal_id --
    # this test locks that identity is anchored on source_event_id only,
    # never conflated with the cycle grouping layer.
    import inspect
    sig = inspect.signature(generate_signal_id)
    assert "independent_cycle_id" not in sig.parameters


# ---- event-less signal identity ----

def test_event_less_signal_identity_uses_birth_ts():
    a = generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG", signal_birth_ts=1000)
    b = generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG", signal_birth_ts=1000)
    assert a == b
    c = generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG", source_event_id="EVT-1")
    assert a != c  # event-anchored vs event-less must never collide


def test_event_less_signal_without_birth_ts_raises():
    with pytest.raises(ValueError):
        generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG")


# ---- LONG/SHORT identity symmetry ----

def test_direction_classification_long_short_symmetry():
    assert classify_direction_from_setup_id("LONG_SILENCE") == "LONG"
    assert classify_direction_from_setup_id("SHORT_NOISY_BTC1M") == "SHORT"
    assert classify_direction_from_setup_id("BUY_FADE_SHORT_H45_SL75") == "SHORT"
    assert classify_direction_from_setup_id("SOMETHING_ELSE") == "UNKNOWN"


def test_signal_id_differs_for_long_vs_short_same_event():
    long_id = generate_signal_id("LONG_SILENCE", "setup-v1", "ETHUSDT", "LONG", source_event_id="EVT-1")
    short_id = generate_signal_id("SHORT_NEITHER", "setup-v1", "ETHUSDT", "SHORT", source_event_id="EVT-1")
    assert long_id != short_id


# ---- timestamp ordering constraints (schema CHECK enforcement) ----

def _insert_signal(conn, signal_id="SIG-1", **overrides):
    row = {
        "signal_id": signal_id, "setup_id": "LONG_SILENCE", "setup_version": "setup-v1",
        "source_event_id": "EVT-1", "independent_cycle_id": "CYC-1", "symbol": "ETHUSDT",
        "direction": "LONG", "timeframe": None, "route_version": "LONG_SILENCE",
        "signal_birth_ts": 1000, "first_known_ts": None, "first_executable_ts": None,
        "last_confirmation_ts": None, "invalidation_ts": None, "terminal_ts": None,
        "lifecycle_status": "OPEN", "lifecycle_reason_code": "SIGNAL_BIRTH",
        "observation_mode": "HISTORICAL_REPLAY", "evidence_layer": "REAL", "is_proxy": 0,
        "executability_status": "FORWARD_ONLY", "identity_version": "signal-identity-v1",
        "schema_version": 1, "source_hash": "h", "code_commit": "test", "provenance": PROV,
        "created_at": 0, "updated_ms": 0,
    }
    row.update(overrides)
    cols = list(row.keys())
    conn.execute(
        f"INSERT INTO ami_signal_lifecycle ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
        [row[c] for c in cols],
    )


def test_first_known_ts_before_birth_ts_rejected():
    conn = _conn()
    with pytest.raises(sqlite3.IntegrityError):
        _insert_signal(conn, signal_birth_ts=1000, first_known_ts=500)


def test_first_executable_before_first_known_rejected():
    conn = _conn()
    with pytest.raises(sqlite3.IntegrityError):
        _insert_signal(conn, first_known_ts=1000, first_executable_ts=500)


def test_invalidation_before_birth_rejected():
    conn = _conn()
    with pytest.raises(sqlite3.IntegrityError):
        _insert_signal(conn, signal_birth_ts=1000, invalidation_ts=500)


def test_terminal_before_birth_rejected():
    conn = _conn()
    with pytest.raises(sqlite3.IntegrityError):
        _insert_signal(conn, signal_birth_ts=1000, terminal_ts=500)


def test_direction_outside_allowed_set_rejected():
    conn = _conn()
    with pytest.raises(sqlite3.IntegrityError):
        _insert_signal(conn, direction="SIDEWAYS")


def test_evidence_layer_is_proxy_mismatch_rejected():
    conn = _conn()
    with pytest.raises(sqlite3.IntegrityError):
        _insert_signal(conn, evidence_layer="REAL", is_proxy=1)


def test_valid_signal_row_accepted():
    conn = _conn()
    _insert_signal(conn)  # must not raise
    n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
    assert n == 1


# ---- known-at / no-lookahead ----

def test_known_at_before_transition_ts_rejected():
    conn = _conn()
    with pytest.raises(LifecycleIntegrityViolation):
        insert_transition(conn, "SIG-1", None, "OPEN", transition_ts=1000, known_at_ts=500,
                          reason_code="SIGNAL_BIRTH", observation_mode="HISTORICAL_REPLAY", provenance=PROV)


def test_known_at_schema_check_enforced_directly():
    conn = _conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_lifecycle_transitions (transition_id, signal_id, previous_status, new_status, "
            "transition_ts, known_at_ts, reason_code, transition_version, observation_mode, schema_version, "
            "provenance, created_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("TRN-x", "SIG-1", None, "OPEN", 1000, 500, "SIGNAL_BIRTH", 1, "HISTORICAL_REPLAY", 1, PROV, 0),
        )


# ---- invalid lifecycle status / reason code rejection ----

def test_invalid_new_status_rejected():
    conn = _conn()
    with pytest.raises(LifecycleIntegrityViolation):
        insert_transition(conn, "SIG-1", None, "NOT_A_REAL_STATUS", transition_ts=1000, known_at_ts=1000,
                          reason_code="SIGNAL_BIRTH", observation_mode="HISTORICAL_REPLAY", provenance=PROV)


def test_invalid_reason_code_rejected():
    conn = _conn()
    with pytest.raises(LifecycleIntegrityViolation):
        insert_transition(conn, "SIG-1", None, "OPEN", transition_ts=1000, known_at_ts=1000,
                          reason_code="MADE_UP_REASON", observation_mode="HISTORICAL_REPLAY", provenance=PROV)


# ---- invalid transition sequence rejection ----

def test_no_op_transition_rejected():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    with pytest.raises(LifecycleIntegrityViolation):
        validate_transition(conn, "SIG-1", "OPEN", "OPEN")


def test_broken_chain_rejected():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    with pytest.raises(LifecycleIntegrityViolation):
        # previous_status claims CLOSED but the signal's actual latest status is OPEN
        insert_transition(conn, "SIG-1", "CLOSED", "INVALIDATED", 2000, 2000, "TERMINAL_CLOSE",
                          "HISTORICAL_REPLAY", provenance=PROV)


def test_transition_from_terminal_status_rejected():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    with pytest.raises(LifecycleIntegrityViolation):
        insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 3000, 3000, "HISTORICAL_RECONSTRUCTION",
                          "HISTORICAL_REPLAY", provenance=PROV)


def test_duplicate_genesis_rejected():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    with pytest.raises(LifecycleIntegrityViolation):
        insert_transition(conn, "SIG-1", None, "CLOSED", 500, 500, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)


def test_non_genesis_first_transition_rejected():
    conn = _conn()
    with pytest.raises(LifecycleIntegrityViolation):
        insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 1000, 1000, "TERMINAL_CLOSE",
                          "HISTORICAL_REPLAY", provenance=PROV)


# ---- append-only update rejection ----

def test_no_update_or_delete_statements_in_schema_module():
    import ami.lifecycle.canonical_schema as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert "UPDATE ami_lifecycle_transitions" not in src
    assert "DELETE FROM ami_lifecycle_transitions" not in src


# ---- correction/supersession behavior ----

def test_correction_uses_higher_transition_version_not_update():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    original_tid = insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE",
                                     "HISTORICAL_REPLAY", provenance=PROV)
    # correction: re-assert the SAME nominal transition (OPEN->CLOSED at ts=2000)
    # under transition_version=2, bypassing the normal chain check (validate=False)
    # because CLOSED is already terminal by then. [round 3] correction_of is now
    # mandatory for validate=False -- it must reference the transition being revised.
    insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2500, "CORRECTION", "HISTORICAL_REPLAY",
                      transition_version=2, correction_of=original_tid, provenance=PROV, validate=False)
    n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions WHERE signal_id='SIG-1'").fetchone()[0]
    assert n == 3  # original 2 rows + 1 correction row, nothing overwritten


# ---- [PHASE 7A-P1 semantic closure, round 3] validate=False fail-closed
# safety contract: only reachable via an explicit correction/supersession,
# never by a normal transition writer ----

def test_validate_false_requires_correction_of():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    with pytest.raises(LifecycleIntegrityViolation, match="correction_of"):
        insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 2000, 2500, "CORRECTION", "HISTORICAL_REPLAY",
                          transition_version=2, correction_of=None, provenance=PROV, validate=False)


def test_validate_false_requires_reason_code_correction():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    tid = insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    with pytest.raises(LifecycleIntegrityViolation, match="reason_code=CORRECTION"):
        insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 2000, 2500, "DATA_GAP", "HISTORICAL_REPLAY",
                          transition_version=2, correction_of=tid, provenance=PROV, validate=False)


def test_validate_false_fails_closed_on_nonexistent_correction_target():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    with pytest.raises(LifecycleIntegrityViolation, match="does not reference an existing transition"):
        insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 2000, 2500, "CORRECTION", "HISTORICAL_REPLAY",
                          transition_version=2, correction_of="TRN-does-not-exist", provenance=PROV, validate=False)


def test_same_transition_cannot_be_corrected_twice():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    tid = insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 2000, 2500, "CORRECTION", "HISTORICAL_REPLAY",
                      transition_version=2, correction_of=tid, provenance=PROV, validate=False)
    # a SECOND, DIFFERENT correction attempt against the same original (different
    # known_at_ts -> different transition_id, not caught by the idempotent-tuple check)
    with pytest.raises(LifecycleIntegrityViolation, match="already been corrected"):
        insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 2000, 9999, "CORRECTION", "HISTORICAL_REPLAY",
                          transition_version=3, correction_of=tid, provenance=PROV, validate=False)


def test_identical_correction_resubmission_remains_idempotent_noop():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    tid = insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 2000, 2500, "CORRECTION", "HISTORICAL_REPLAY",
                      transition_version=2, correction_of=tid, provenance=PROV, validate=False)
    # re-submitting the EXACT same correction tuple again must still be a
    # silent no-op (caught before the new guards even run)
    insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 2000, 2500, "CORRECTION", "HISTORICAL_REPLAY",
                      transition_version=2, correction_of=tid, provenance=PROV, validate=False)
    n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions WHERE signal_id='SIG-1'").fetchone()[0]
    assert n == 3  # genesis + TERMINAL_CLOSE + 1 correction (not 2)


def test_normal_writer_cannot_reach_validate_false_bypass():
    # backfill_lifecycle-style genesis writes always use validate=True (the
    # default) -- this documents/locks the contract that a normal writer,
    # simply by never passing validate=False, structurally cannot bypass
    # chain/terminal checks (see the two guards above for the OTHER half:
    # even if it tried, correction_of/reason_code enforcement blocks it).
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    with pytest.raises(LifecycleIntegrityViolation, match="terminal status rejected"):
        insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 3000, 3000, "HISTORICAL_RECONSTRUCTION",
                          "HISTORICAL_REPLAY", provenance=PROV)  # validate defaults to True


# ---- [PHASE 7A-P1 semantic closure, round 3] effective ledger: superseded +
# pure-reversal correction pairs excluded; a genuine (non-reversal) correction
# is NOT excluded ----

def test_effective_view_excludes_pure_reversal_pair():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    tid = insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 2000, 2500, "CORRECTION", "HISTORICAL_REPLAY",
                      transition_version=2, correction_of=tid, provenance=PROV, validate=False)

    raw_n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions WHERE signal_id='SIG-1'").fetchone()[0]
    effective_rows = conn.execute(
        "SELECT reason_code, new_status FROM ami_lifecycle_effective_transitions WHERE signal_id='SIG-1'"
    ).fetchall()
    assert raw_n == 3  # raw ledger: immutable, all 3 rows preserved
    assert effective_rows == [("SIGNAL_BIRTH", "OPEN")]  # effective: only genesis survives

    status = effective_lifecycle_status(conn, "SIG-1")
    assert status["current_status"] == "OPEN"
    assert status["n_effective_transitions"] == 1  # no fake CLOSED interval


def test_effective_view_keeps_genuine_non_reversal_correction():
    # a correction that does NOT reverse status (re-asserts the SAME
    # OPEN->CLOSED claim under a corrected timestamp/metadata) represents a
    # REAL transition that genuinely happened -- it must NOT be excluded.
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    tid = insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2500, "CORRECTION", "HISTORICAL_REPLAY",
                      transition_version=2, correction_of=tid, provenance=PROV, validate=False)

    effective_rows = conn.execute(
        "SELECT reason_code, new_status FROM ami_lifecycle_effective_transitions WHERE signal_id='SIG-1' "
        "ORDER BY transition_ts, transition_version"
    ).fetchall()
    # original TERMINAL_CLOSE excluded (superseded); the metadata-only
    # CORRECTION (same OPEN->CLOSED direction) is kept -- a real close happened
    assert effective_rows == [("SIGNAL_BIRTH", "OPEN"), ("CORRECTION", "CLOSED")]

    status = effective_lifecycle_status(conn, "SIG-1")
    assert status["current_status"] == "CLOSED"
    assert count_effective_closed_signals(conn) == 1


def test_effective_view_agrees_with_rebuild_current_state_when_no_correction():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    raw = rebuild_current_state(conn, "SIG-1")
    effective = effective_lifecycle_status(conn, "SIG-1")
    assert raw["current_status"] == effective["current_status"] == "OPEN"


def test_effective_view_survives_init_lifecycle_schema_rerun():
    conn = _conn()
    init_lifecycle_schema(conn)  # rerun must not drop/duplicate the view
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()}
    assert "ami_lifecycle_effective_transitions" in tables


def test_naive_raw_ledger_duration_query_is_contaminated_but_effective_is_not():
    # proves the exact failure mode the operator flagged: a downstream
    # researcher who naively computes "hold duration" by joining OPEN/CLOSED
    # rows off the RAW ledger gets a fake, invalid interval; the SAME query
    # against the effective view correctly finds no interval at all.
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    tid = insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 5000, 5000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "CLOSED", "OPEN", 5000, 5500, "CORRECTION", "HISTORICAL_REPLAY",
                      transition_version=2, correction_of=tid, provenance=PROV, validate=False)

    naive_duration = conn.execute(
        "SELECT (SELECT transition_ts FROM ami_lifecycle_transitions WHERE signal_id='SIG-1' "
        "AND new_status='CLOSED' ORDER BY transition_ts LIMIT 1) - "
        "(SELECT transition_ts FROM ami_lifecycle_transitions WHERE signal_id='SIG-1' "
        "AND new_status='OPEN' ORDER BY transition_ts LIMIT 1)"
    ).fetchone()[0]
    assert naive_duration == 4000  # the exact, invalid "hold duration" a raw-ledger query would fabricate

    effective_closed_row = conn.execute(
        "SELECT transition_ts FROM ami_lifecycle_effective_transitions WHERE signal_id='SIG-1' AND new_status='CLOSED'"
    ).fetchone()
    assert effective_closed_row is None  # no CLOSED row survives -> no interval computable, no fake duration
    assert count_effective_closed_signals(conn) == 0


# ---- duplicate transition suppression ----

def test_duplicate_identical_transition_is_idempotent_noop():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
    assert n == 1


# ---- migration rerun idempotency ----

def test_init_lifecycle_schema_rerun_idempotent():
    conn = _conn()
    init_lifecycle_schema(conn)  # second call, must not raise
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"ami_signal_lifecycle", "ami_lifecycle_transitions"} <= tables


def test_rollback_then_reapply_is_clean():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    rollback_lifecycle_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "ami_lifecycle_transitions" not in tables
    init_lifecycle_schema(conn)
    n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
    assert n == 0  # rolled back, not restored -- reapply starts clean


# ---- current-state rebuild equality ----

def test_rebuild_current_state_matches_manual_fold():
    conn = _conn()
    insert_transition(conn, "SIG-1", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, "SIG-1", "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    rebuilt = rebuild_current_state(conn, "SIG-1")
    assert rebuilt["current_status"] == "CLOSED"
    assert rebuilt["n_transitions"] == 2


def test_rebuild_current_state_none_when_no_transitions():
    conn = _conn()
    assert rebuild_current_state(conn, "SIG-UNKNOWN") is None
