"""PHASE 7A-P1: tests for ami/lifecycle/canonical_field_provenance.py.
DISPOSABLE_DB_ONLY: every test uses :memory:/tmp_path fixtures, never the
real data/ami/canonical.sqlite for writes.

Run: pytest tests/test_ami_lifecycle_canonical_field_provenance.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import sqlite3

import pytest

from ami.lifecycle.canonical_field_provenance import (
    FIELD_PROVENANCE_SPECS,
    FieldProvenanceViolation,
    PROVENANCE_VERSION,
    backfill_field_provenance,
    init_field_provenance_schema,
    rollback_field_provenance_schema,
    validate_field_provenance_complete,
)
from ami.lifecycle.canonical_schema import init_lifecycle_schema, insert_transition

PROV = "test"


def _conn_with_signal(signal_id="SIG-1"):
    conn = sqlite3.connect(":memory:")
    init_lifecycle_schema(conn)
    init_field_provenance_schema(conn)
    conn.execute(
        "INSERT INTO ami_signal_lifecycle (signal_id, setup_id, setup_version, source_event_id, "
        "independent_cycle_id, symbol, direction, timeframe, route_version, signal_birth_ts, "
        "first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts, terminal_ts, "
        "lifecycle_status, lifecycle_reason_code, observation_mode, evidence_layer, is_proxy, "
        "executability_status, identity_version, schema_version, source_hash, code_commit, "
        "provenance, created_at, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (signal_id, "LONG_SILENCE", "setup-v1", "EVT-1", "CYC-1", "ETHUSDT", "LONG", None,
         "LONG_SILENCE", 1000, None, None, None, None, 2000, "CLOSED", "TERMINAL_CLOSE",
         "HISTORICAL_REPLAY", "REAL", 0, "FORWARD_ONLY", "signal-identity-v1", 1, "h", "test", PROV, 0, 0),
    )
    insert_transition(conn, signal_id, None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    insert_transition(conn, signal_id, "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE", "HISTORICAL_REPLAY", provenance=PROV)
    conn.commit()
    return conn


# ---- existing provenance equivalent reconciliation (documented, not re-tested here --
# see canonical_field_provenance.py module docstring for the reconciliation record) ----

def test_no_second_truth_layer_created():
    # the new table never stores a competing copy of core identity fields
    conn = _conn_with_signal()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ami_lifecycle_field_provenance)").fetchall()}
    assert "source_event_id" not in cols
    assert "independent_cycle_id" not in cols
    assert "lifecycle_status" not in cols


# ---- every non-null route-derived direction has exactly one applicable provenance record ----

def test_direction_provenance_written_for_every_signal():
    conn = _conn_with_signal("SIG-1")
    conn.execute(
        "INSERT INTO ami_signal_lifecycle (signal_id, setup_id, setup_version, source_event_id, "
        "independent_cycle_id, symbol, direction, timeframe, route_version, signal_birth_ts, "
        "first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts, terminal_ts, "
        "lifecycle_status, lifecycle_reason_code, observation_mode, evidence_layer, is_proxy, "
        "executability_status, identity_version, schema_version, source_hash, code_commit, "
        "provenance, created_at, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("SIG-2", "SHORT_NEITHER", "setup-v1", "EVT-2", None, "ETHUSDT", "SHORT", None,
         "SHORT_NEITHER", 3000, None, None, None, None, None, "OPEN", "SIGNAL_BIRTH",
         "HISTORICAL_REPLAY", "REAL", 0, "FORWARD_ONLY", "signal-identity-v1", 1, "h", "test", PROV, 0, 0),
    )
    insert_transition(conn, "SIG-2", None, "OPEN", 3000, 3000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY", provenance=PROV)
    conn.commit()

    backfill_field_provenance(conn, ["SIG-1", "SIG-2"])
    rows = conn.execute(
        "SELECT signal_id, COUNT(*) FROM ami_lifecycle_field_provenance WHERE field_name='direction' GROUP BY signal_id"
    ).fetchall()
    assert dict(rows) == {"SIG-1": 1, "SIG-2": 1}  # exactly one applicable record each


# ---- direction classification is HISTORICAL_PROXY / is_proxy=true at field level ----

def test_direction_field_classification_and_is_proxy():
    conn = _conn_with_signal("SIG-1")
    backfill_field_provenance(conn, ["SIG-1"])
    row = conn.execute(
        "SELECT field_classification, is_proxy FROM ami_lifecycle_field_provenance "
        "WHERE signal_id='SIG-1' AND field_name='direction'"
    ).fetchone()
    assert row == ("HISTORICAL_PROXY", 1)


def test_row_level_is_proxy_false_does_not_imply_field_level_real():
    # ami_signal_lifecycle.is_proxy=0 (REAL anchor) coexists with
    # direction's field-level is_proxy=1 -- the two axes are independent.
    conn = _conn_with_signal("SIG-1")
    row_level_is_proxy = conn.execute(
        "SELECT is_proxy FROM ami_signal_lifecycle WHERE signal_id='SIG-1'"
    ).fetchone()[0]
    backfill_field_provenance(conn, ["SIG-1"])
    field_level_is_proxy = conn.execute(
        "SELECT is_proxy FROM ami_lifecycle_field_provenance WHERE signal_id='SIG-1' AND field_name='direction'"
    ).fetchone()[0]
    assert row_level_is_proxy == 0
    assert field_level_is_proxy == 1
    assert row_level_is_proxy != field_level_is_proxy  # genuinely independent axes


# ---- no direction field is exposed as REAL_OBSERVED (canonical view contract) ----

def test_direction_view_always_carries_classification():
    conn = _conn_with_signal("SIG-1")
    backfill_field_provenance(conn, ["SIG-1"])
    row = conn.execute(
        "SELECT direction, direction_classification, direction_is_proxy FROM ami_lifecycle_direction_view "
        "WHERE signal_id='SIG-1'"
    ).fetchone()
    assert row == ("LONG", "HISTORICAL_PROXY", 1)


def test_direction_view_shows_missing_provenance_as_null_not_real():
    # if provenance was never backfilled, the view surfaces NULL classification
    # (never silently implying REAL/DETERMINISTIC_HISTORICAL_SAFE)
    conn = _conn_with_signal("SIG-1")
    row = conn.execute(
        "SELECT direction, direction_classification FROM ami_lifecycle_direction_view WHERE signal_id='SIG-1'"
    ).fetchone()
    assert row == ("LONG", None)


# ---- duplicate provenance suppression ----

def test_duplicate_backfill_does_not_grow_row_count():
    conn = _conn_with_signal("SIG-1")
    backfill_field_provenance(conn, ["SIG-1"])
    n1 = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]
    backfill_field_provenance(conn, ["SIG-1"])
    n2 = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]
    assert n1 == n2 == len(FIELD_PROVENANCE_SPECS)


def test_unique_constraint_backstops_application_level_dedup():
    conn = _conn_with_signal("SIG-1")
    backfill_field_provenance(conn, ["SIG-1"])
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_lifecycle_field_provenance (provenance_id, signal_id, field_name, "
            "field_classification, is_proxy, derivation_method, source_reference, provenance_version, "
            "schema_version, created_at) VALUES ('OTHER-ID','SIG-1','direction','HISTORICAL_PROXY',1,"
            "'x','y',?,1,0)",
            (PROVENANCE_VERSION,),
        )


# ---- missing provenance fails validation rather than silently defaulting REAL ----

def test_missing_provenance_raises_not_silently_real():
    conn = _conn_with_signal("SIG-1")
    # deliberately do NOT backfill
    with pytest.raises(FieldProvenanceViolation):
        validate_field_provenance_complete(conn, ["SIG-1"])


def test_partial_provenance_for_one_signal_among_many_raises():
    conn = _conn_with_signal("SIG-1")
    backfill_field_provenance(conn, ["SIG-1"])
    with pytest.raises(FieldProvenanceViolation):
        validate_field_provenance_complete(conn, ["SIG-1", "SIG-UNBACKFILLED"])


def test_complete_provenance_passes_validation():
    conn = _conn_with_signal("SIG-1")
    backfill_field_provenance(conn, ["SIG-1"])
    validate_field_provenance_complete(conn, ["SIG-1"])  # must not raise


# ---- rollback + rerun idempotency ----

def test_rollback_then_reapply_clean():
    conn = _conn_with_signal("SIG-1")
    backfill_field_provenance(conn, ["SIG-1"])
    rollback_field_provenance_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()}
    assert "ami_lifecycle_field_provenance" not in tables
    assert "ami_lifecycle_direction_view" not in tables
    init_field_provenance_schema(conn)
    n = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]
    assert n == 0  # rolled back, not restored


def test_init_schema_rerun_idempotent():
    conn = _conn_with_signal("SIG-1")
    init_field_provenance_schema(conn)  # second call must not raise
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()}
    assert {"ami_lifecycle_field_provenance", "ami_lifecycle_direction_view"} <= tables


# ---- lifecycle rows never modified by this closure ----

def test_lifecycle_and_transition_rows_untouched_by_provenance_backfill():
    conn = _conn_with_signal("SIG-1")
    before = conn.execute("SELECT * FROM ami_signal_lifecycle WHERE signal_id='SIG-1'").fetchone()
    before_trn = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
    backfill_field_provenance(conn, ["SIG-1"])
    after = conn.execute("SELECT * FROM ami_signal_lifecycle WHERE signal_id='SIG-1'").fetchone()
    after_trn = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
    assert before == after
    assert before_trn == after_trn


# ---- all 16 FIELD_PROVENANCE_SPECS entries are well-formed (no None source_reference) ----

def test_all_field_specs_have_non_null_source_reference():
    for field, spec in FIELD_PROVENANCE_SPECS.items():
        assert spec["source_reference"] is not None, f"{field} has a None source_reference"
        assert spec["field_classification"] in {
            "DETERMINISTIC_HISTORICAL_SAFE", "HISTORICAL_PROXY", "FORWARD_ONLY",
            "NOT_IMPLEMENTED", "BLOCKED_BY_DATA",
        }
