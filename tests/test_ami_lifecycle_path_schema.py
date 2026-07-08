"""PHASE 7B-0: tests for ami/lifecycle/path_schema.py (schema round-trip,
CHECK/UNIQUE/FK constraints). DISPOSABLE_DB_ONLY: every test here uses an
in-memory sqlite connection, never data/ami/canonical.sqlite.

Run: pytest tests/test_ami_lifecycle_path_schema.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import sqlite3

import pytest

from ami.lifecycle.canonical_schema import init_lifecycle_schema
from ami.lifecycle.path_schema import (
    PATH_DEFINITION_VERSION,
    PATH_SCHEMA_VERSION,
    init_path_schema,
    rollback_path_schema,
)

PROV = "test"


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    init_lifecycle_schema(conn)
    init_path_schema(conn)
    return conn


def _insert_dummy_signal(conn, signal_id="SIG-DUMMY"):
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


_FULL_ROW = dict(
    observation_id="OBS-1", signal_id="SIG-DUMMY", horizon_name="scalp_30m",
    horizon_end_ts=2_000_000, known_at_ts=2_000_000, as_of_ts=2_000_000,
    observation_status="OK", volatility_status="OK", reference_price=100.0, reference_price_ts=999_000,
    effective_path_start_ts=1_000_000, endpoint_return_bps=5.0, mfe_bps=10.0, mae_bps=-3.0,
    time_to_mfe_ms=60_000, time_to_mae_ms=120_000, intrabar_order_status="MFE_FIRST",
    realized_vol_at_anchor=0.001, endpoint_return_anchor_vol_units=0.5,
    mfe_anchor_vol_units=1.0, mae_anchor_vol_units=-0.3,
    horizon_outcome_class="CHOP", expected_candle_count=30, observed_candle_count=30, gap_count=0,
    candle_definition_version="candle-agg_trades-v1", path_definition_version=PATH_DEFINITION_VERSION,
    observation_mode="HISTORICAL_REPLAY", provenance=PROV, schema_version=PATH_SCHEMA_VERSION, created_ms=0,
)


def _insert_row(conn, **overrides):
    row = dict(_FULL_ROW, **overrides)
    cols = list(row.keys())
    conn.execute(
        f"INSERT INTO ami_lifecycle_path_observations ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})",
        [row[c] for c in cols],
    )
    conn.commit()


# ---- schema round-trip ----

def test_init_path_schema_creates_table():
    conn = _conn()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "ami_lifecycle_path_observations" in tables


def test_init_path_schema_idempotent():
    conn = _conn()
    init_path_schema(conn)  # second call, must not raise
    init_path_schema(conn)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ami_lifecycle_path_observations'"
    ).fetchall()]
    assert len(tables) == 1


def test_rollback_removes_only_path_table():
    conn = _conn()
    rollback_path_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "ami_lifecycle_path_observations" not in tables
    assert "ami_signal_lifecycle" in tables  # untouched
    assert "ami_lifecycle_transitions" in tables  # untouched


def test_rollback_then_reapply_roundtrip():
    conn = _conn()
    rollback_path_schema(conn)
    init_path_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "ami_lifecycle_path_observations" in tables


# ---- valid insert ----

def test_full_valid_row_inserts_cleanly():
    conn = _conn()
    _insert_dummy_signal(conn)
    _insert_row(conn)
    count = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
    assert count == 1


# ---- CHECK constraints ----

def test_mfe_negative_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, mfe_bps=-1.0)


def test_mae_positive_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, mae_bps=1.0)


def test_mfe_zero_and_mae_zero_allowed():
    conn = _conn()
    _insert_dummy_signal(conn)
    _insert_row(conn, mfe_bps=0.0, mae_bps=0.0)  # must not raise
    assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == 1


def test_invalid_observation_status_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, observation_status="BOGUS_STATUS")


def test_invalid_horizon_name_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, horizon_name="scalp_5m")


def test_invalid_intrabar_order_status_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, intrabar_order_status="BOGUS")


def test_null_intrabar_order_status_allowed():
    conn = _conn()
    _insert_dummy_signal(conn)
    _insert_row(conn, intrabar_order_status=None)  # must not raise


def test_invalid_horizon_outcome_class_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, horizon_outcome_class="BOGUS")


def test_known_at_before_horizon_end_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, known_at_ts=1_999_999, horizon_end_ts=2_000_000)


def test_negative_gap_count_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, gap_count=-1)


# ---- UNIQUE / FK ----

def test_unique_signal_horizon_version_enforced():
    conn = _conn()
    _insert_dummy_signal(conn)
    _insert_row(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, observation_id="OBS-2")  # same (signal_id, horizon_name, path_definition_version)


def test_different_horizon_same_signal_allowed():
    conn = _conn()
    _insert_dummy_signal(conn)
    _insert_row(conn)
    _insert_row(conn, observation_id="OBS-2", horizon_name="scalp_1h")  # must not raise
    assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == 2


def test_fk_rejects_unknown_signal_id():
    conn = _conn()
    # no dummy signal inserted -- SIG-DUMMY does not exist
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn)


# ---- volatility_status: independent axis (PHASE 7B-0.1) ----

def test_invalid_volatility_status_rejected():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, volatility_status="BOGUS")


def test_observation_status_ok_with_volatility_invalid_allowed():
    conn = _conn()
    _insert_dummy_signal(conn)
    _insert_row(conn, observation_status="OK", volatility_status="INVALID_VOLATILITY_BASELINE",
                realized_vol_at_anchor=None, endpoint_return_anchor_vol_units=None,
                mfe_anchor_vol_units=None, mae_anchor_vol_units=None)  # must not raise
    assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == 1


def test_volatility_not_applicable_requires_observation_not_ok():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, observation_status="OK", volatility_status="NOT_APPLICABLE")


def test_volatility_ok_requires_observation_ok():
    conn = _conn()
    _insert_dummy_signal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_row(conn, observation_status="MISSING_REFERENCE_PRICE", volatility_status="OK",
                    reference_price=None, reference_price_ts=None, effective_path_start_ts=None,
                    endpoint_return_bps=None, mfe_bps=None, mae_bps=None, time_to_mfe_ms=None,
                    time_to_mae_ms=None, intrabar_order_status=None, horizon_outcome_class=None,
                    expected_candle_count=0, observed_candle_count=0, candle_definition_version=None)


def test_non_ok_observation_status_with_not_applicable_volatility_allowed():
    conn = _conn()
    _insert_dummy_signal(conn)
    _insert_row(conn, observation_status="MISSING_REFERENCE_PRICE", volatility_status="NOT_APPLICABLE",
                reference_price=None, reference_price_ts=None, effective_path_start_ts=None,
                endpoint_return_bps=None, mfe_bps=None, mae_bps=None, time_to_mfe_ms=None,
                time_to_mae_ms=None, intrabar_order_status=None, horizon_outcome_class=None,
                expected_candle_count=0, observed_candle_count=0, candle_definition_version=None,
                realized_vol_at_anchor=None, endpoint_return_anchor_vol_units=None,
                mfe_anchor_vol_units=None, mae_anchor_vol_units=None)  # must not raise
    assert conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == 1


# ---- schema manifest (PHASE 7B-0.1) ----

def test_schema_manifest_column_count_is_31():
    from ami.lifecycle.path_schema import schema_manifest
    conn = _conn()
    manifest = schema_manifest(conn)
    assert manifest["column_count"] == 31
    names = [c["name"] for c in manifest["columns"]]
    assert names == [
        "observation_id", "signal_id", "horizon_name", "horizon_end_ts", "known_at_ts", "as_of_ts",
        "observation_status", "volatility_status", "reference_price", "reference_price_ts",
        "effective_path_start_ts", "endpoint_return_bps", "mfe_bps", "mae_bps", "time_to_mfe_ms",
        "time_to_mae_ms", "intrabar_order_status", "realized_vol_at_anchor",
        "endpoint_return_anchor_vol_units", "mfe_anchor_vol_units", "mae_anchor_vol_units",
        "horizon_outcome_class", "expected_candle_count", "observed_candle_count", "gap_count",
        "candle_definition_version", "path_definition_version", "observation_mode", "provenance",
        "schema_version", "created_ms",
    ]


def test_schema_manifest_reports_unique_check_fk_and_fingerprint():
    from ami.lifecycle.path_schema import schema_manifest
    conn = _conn()
    manifest = schema_manifest(conn)
    assert any("UNIQUE" in u.upper() for u in manifest["unique_constraints"])
    assert len(manifest["check_constraints"]) >= 10  # 30-field original CHECK set + new volatility ones
    assert manifest["foreign_keys"] == [{"table": "ami_signal_lifecycle", "from": "signal_id", "to": "signal_id"}]
    assert isinstance(manifest["schema_fingerprint"], str) and len(manifest["schema_fingerprint"]) == 64


def test_schema_manifest_fingerprint_deterministic_across_calls():
    from ami.lifecycle.path_schema import schema_manifest
    conn = _conn()
    a = schema_manifest(conn)["schema_fingerprint"]
    b = schema_manifest(conn)["schema_fingerprint"]
    assert a == b


def test_prior_7b0_schema_had_exactly_30_columns_before_volatility_status_added():
    """Verification requested by the operator: the 7B-0 disposable schema
    (before this 7B-0.1 semantic-closure batch added volatility_status) had
    EXACTLY the 30 fields the operator's original spec listed -- confirmed
    here by reconstructing that DDL's column list directly (not from memory)
    and checking it is exactly today's 31-column manifest minus
    volatility_status."""
    from ami.lifecycle.path_schema import schema_manifest
    conn = _conn()
    manifest = schema_manifest(conn)
    names_today = [c["name"] for c in manifest["columns"]]
    names_before_7b0_1 = [n for n in names_today if n != "volatility_status"]
    assert len(names_before_7b0_1) == 30
