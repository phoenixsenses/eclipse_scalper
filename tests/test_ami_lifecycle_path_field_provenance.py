"""PHASE 7B-0.1: tests for ami/lifecycle/path_field_provenance.py (per-field
DETERMINISTIC_HISTORICAL_SAFE/HISTORICAL_PROXY mapping for
ami_lifecycle_path_observations, written into the EXISTING
ami_lifecycle_field_provenance table). DISPOSABLE_DB_ONLY: every test here
uses an in-memory sqlite connection, never data/ami/canonical.sqlite.

Run: pytest tests/test_ami_lifecycle_path_field_provenance.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import sqlite3
import time

import pytest

from ami.lifecycle.canonical_field_provenance import init_field_provenance_schema
from ami.lifecycle.canonical_schema import init_lifecycle_schema
from ami.lifecycle.path_field_provenance import (
    FIELD_NAME_PREFIX,
    PATH_FIELD_PROVENANCE_SPECS,
    PATH_PROVENANCE_VERSION,
    PathFieldProvenanceDowngradeViolation,
    backfill_path_field_provenance,
    rollback_path_field_provenance,
)
from ami.lifecycle.path_schema import init_path_schema

PROV = "test"


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    init_lifecycle_schema(conn)
    init_path_schema(conn)
    init_field_provenance_schema(conn)
    return conn


def _insert_dummy_signal(conn, signal_id):
    conn.execute(
        "INSERT INTO ami_signal_lifecycle (signal_id, setup_id, setup_version, source_event_id, "
        "independent_cycle_id, symbol, direction, timeframe, route_version, signal_birth_ts, "
        "first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts, terminal_ts, "
        "lifecycle_status, lifecycle_reason_code, observation_mode, evidence_layer, is_proxy, "
        "executability_status, identity_version, schema_version, source_hash, code_commit, "
        "provenance, created_at, updated_ms) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (signal_id, "LONG_SILENCE", None, "EVT-1", "CYC-1", "ETHUSDT", "LONG", None, "LONG_SILENCE",
         1_000_000, None, None, None, None, None, "OPEN", "SIGNAL_BIRTH", "HISTORICAL_REPLAY", "REAL", 0,
         "FORWARD_ONLY", "signal-identity-v1", 3, None, None, PROV, 0, 0),
    )
    conn.commit()


# ---- classification completeness / no unknown-default ----

def test_every_field_has_valid_classification():
    valid = {"DETERMINISTIC_HISTORICAL_SAFE", "HISTORICAL_PROXY"}
    for field, spec in PATH_FIELD_PROVENANCE_SPECS.items():
        assert spec["field_classification"] in valid, field
        assert spec["derivation_method"], field
        assert spec["source_reference"], field


def test_23_fields_tracked():
    assert len(PATH_FIELD_PROVENANCE_SPECS) == 23


def test_direction_dependent_fields_are_proxy():
    for field in ("mfe_bps", "mae_bps", "time_to_mfe_ms", "time_to_mae_ms",
                  "intrabar_order_status", "mfe_anchor_vol_units", "mae_anchor_vol_units"):
        assert PATH_FIELD_PROVENANCE_SPECS[field]["field_classification"] == "HISTORICAL_PROXY"


def test_realized_vol_at_anchor_is_deterministic_not_proxy():
    # does not depend on reference_price or direction -- the one vol-adjacent field that is NOT proxy
    assert PATH_FIELD_PROVENANCE_SPECS["realized_vol_at_anchor"]["field_classification"] == \
        "DETERMINISTIC_HISTORICAL_SAFE"


def test_endpoint_fields_proxy_but_direction_independent_reason_documented():
    for field in ("endpoint_return_bps", "horizon_outcome_class", "endpoint_return_anchor_vol_units"):
        spec = PATH_FIELD_PROVENANCE_SPECS[field]
        assert spec["field_classification"] == "HISTORICAL_PROXY"
        assert "direction" not in spec["limitations"].lower() or "independent" in spec["limitations"].lower()


# ---- backfill: row counts / idempotency / completeness ----

def test_backfill_writes_expected_row_count():
    conn = _conn()
    _insert_dummy_signal(conn, "SIG-1")
    _insert_dummy_signal(conn, "SIG-2")
    result = backfill_path_field_provenance(conn, ["SIG-1", "SIG-2"])
    assert result["provenance_rows_expected_total"] == 2 * len(PATH_FIELD_PROVENANCE_SPECS)
    assert result["provenance_rows_actual_total"] == result["provenance_rows_expected_total"]
    assert result["provenance_rows_missing"] == 0
    assert result["provenance_rows_duplicate_groups"] == 0


def test_backfill_idempotent():
    conn = _conn()
    _insert_dummy_signal(conn, "SIG-1")
    r1 = backfill_path_field_provenance(conn, ["SIG-1"])
    r2 = backfill_path_field_provenance(conn, ["SIG-1"])
    assert r1["provenance_rows_actual_total"] == r2["provenance_rows_actual_total"]
    assert r2["provenance_rows_missing"] == 0
    assert r2["provenance_rows_duplicate_groups"] == 0


def test_field_name_prefix_applied_to_every_row():
    conn = _conn()
    _insert_dummy_signal(conn, "SIG-1")
    backfill_path_field_provenance(conn, ["SIG-1"])
    rows = conn.execute(
        "SELECT field_name FROM ami_lifecycle_field_provenance WHERE provenance_version=?",
        (PATH_PROVENANCE_VERSION,),
    ).fetchall()
    assert rows
    assert all(r[0].startswith(FIELD_NAME_PREFIX) for r in rows)


def test_is_proxy_consistent_with_classification():
    conn = _conn()
    _insert_dummy_signal(conn, "SIG-1")
    backfill_path_field_provenance(conn, ["SIG-1"])
    rows = conn.execute(
        "SELECT field_classification, is_proxy FROM ami_lifecycle_field_provenance WHERE provenance_version=?",
        (PATH_PROVENANCE_VERSION,),
    ).fetchall()
    for classification, is_proxy in rows:
        if classification == "HISTORICAL_PROXY":
            assert is_proxy == 1
        else:
            assert is_proxy == 0


# ---- no-downgrade guard ----

def test_no_silent_proxy_to_safe_downgrade():
    conn = _conn()
    _insert_dummy_signal(conn, "SIG-1")
    # simulate a PRIOR run (different provenance_version) that classified
    # "path_observations.realized_vol_at_anchor" as HISTORICAL_PROXY --
    # current spec says DETERMINISTIC_HISTORICAL_SAFE, so backfilling now must refuse.
    conn.execute(
        "INSERT INTO ami_lifecycle_field_provenance (provenance_id, signal_id, field_name, "
        "field_classification, is_proxy, derivation_method, source_reference, limitations, "
        "provenance_version, schema_version, code_commit, source_hash, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("FPR-FAKE-PRIOR", "SIG-1", "path_observations.realized_vol_at_anchor", "HISTORICAL_PROXY", 1,
         "old_method", "old_source", None, "path-observations-field-provenance-v0", 1, None, None,
         int(time.time() * 1000)),
    )
    conn.commit()
    with pytest.raises(PathFieldProvenanceDowngradeViolation):
        backfill_path_field_provenance(conn, ["SIG-1"])


def test_no_downgrade_guard_does_not_fire_for_consistent_reruns():
    conn = _conn()
    _insert_dummy_signal(conn, "SIG-1")
    backfill_path_field_provenance(conn, ["SIG-1"])  # writes HISTORICAL_PROXY for e.g. mfe_bps
    backfill_path_field_provenance(conn, ["SIG-1"])  # rerun with the SAME spec -- must not raise


# ---- rollback: only path_observations.* rows removed ----

def test_rollback_removes_only_path_prefixed_rows_preserves_existing():
    conn = _conn()
    _insert_dummy_signal(conn, "SIG-1")
    # a pre-existing, unrelated field-provenance row (e.g. "direction", Phase 7A-P1 style)
    conn.execute(
        "INSERT INTO ami_lifecycle_field_provenance (provenance_id, signal_id, field_name, "
        "field_classification, is_proxy, derivation_method, source_reference, limitations, "
        "provenance_version, schema_version, code_commit, source_hash, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("FPR-DIRECTION-1", "SIG-1", "direction", "HISTORICAL_PROXY", 1, "route_name_prefix_heuristic_v1",
         "ami_signal_lifecycle.setup_id", None, "field-provenance-v1", 1, None, None, int(time.time() * 1000)),
    )
    conn.commit()
    backfill_path_field_provenance(conn, ["SIG-1"])

    deleted = rollback_path_field_provenance(conn)
    assert deleted == len(PATH_FIELD_PROVENANCE_SPECS)

    remaining = conn.execute("SELECT field_name FROM ami_lifecycle_field_provenance").fetchall()
    assert remaining == [("direction",)]  # the pre-existing row survives untouched


# ---- real-data smoke test (disposable copy only, via conftest isolation) ----

def test_real_data_smoke_backfill_completeness():
    import ami.warehouse.schema as schema_mod
    from ami.lifecycle.path_metrics import fetch_signals

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        init_path_schema(conn)
        signal_ids = [s["signal_id"] for s in fetch_signals(conn)]
        result = backfill_path_field_provenance(conn, signal_ids)
        # 324 = 270 original + 54 SHORT_NOISY_BTC200K_CONFIRMED_V1 (BATCH-SHORT-NOISY-V1-CANON-BACKFILL)
        assert result["signals_covered"] == 324
        assert result["provenance_rows_expected_total"] == 324 * 23
        assert result["provenance_rows_actual_total"] == result["provenance_rows_expected_total"]
        assert result["provenance_rows_missing"] == 0
        assert result["provenance_rows_duplicate_groups"] == 0

        result2 = backfill_path_field_provenance(conn, signal_ids)
        assert result2["provenance_rows_actual_total"] == result["provenance_rows_actual_total"]
    finally:
        conn.close()
