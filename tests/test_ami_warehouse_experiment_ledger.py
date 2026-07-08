"""BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING (GOAL B/D).

Tests for ami/warehouse/experiment_ledger.py -- the mandatory, fail-closed
write path for experiment_registry/experiment_results that replaced every
research module's own "INSERT ... ON CONFLICT DO UPDATE" + blind
"DELETE FROM experiment_results" pattern.

Run: pytest tests/test_ami_warehouse_experiment_ledger.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import pytest

from ami.warehouse.experiment_ledger import (
    REGISTRY_COLUMNS,
    ImmutableExperimentConflict,
    record_experiment_registry,
    record_experiment_results,
)
from ami.warehouse.schema import connect, init_schema


def _db(tmp_path):
    conn = connect(tmp_path / "ledger_test.sqlite")
    init_schema(conn)
    return conn


def _base_registry_values(experiment_id="E-TEST-001", **overrides) -> dict:
    values = {
        "experiment_id": experiment_id, "question_ids": "FAM_TEST", "hypothesis_id": "H-TEST",
        "preregistered_at": 1000, "frozen_population": "pop-desc", "frozen_features": "f1,f2",
        "frozen_target": "target-desc", "frozen_thresholds": "thresh-desc", "frozen_splits": "split-desc",
        "frozen_economic_gate": "N/A", "frozen_statistical_gate": "stat-desc", "code_commit": None,
        "dataset_hash": "abc123", "started_at": 1000, "completed_at": 2000,
        "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
        "mutation_test_count": 0, "mutation_test_passed": 1, "supersedes_experiment_id": None,
        "report_artifact_id": None, "schema_version": 10, "provenance": "test", "created_ms": 1000,
        "updated_ms": 1000,
    }
    values.update(overrides)
    return values


def test_new_experiment_id_inserts_successfully(tmp_path):
    conn = _db(tmp_path)
    status = record_experiment_registry(conn, _base_registry_values())
    conn.commit()
    assert status == "INSERTED"
    row = conn.execute(
        "SELECT dataset_hash FROM experiment_registry WHERE experiment_id=?", ("E-TEST-001",)
    ).fetchone()
    assert row[0] == "abc123"
    conn.close()


def test_registry_missing_required_column_raises_value_error(tmp_path):
    conn = _db(tmp_path)
    incomplete = _base_registry_values()
    del incomplete["dataset_hash"]
    with pytest.raises(ValueError, match="missing required column"):
        record_experiment_registry(conn, incomplete)
    conn.close()


def test_byte_identical_rerun_is_registry_noop(tmp_path):
    conn = _db(tmp_path)
    values = _base_registry_values()
    assert record_experiment_registry(conn, values) == "INSERTED"
    conn.commit()

    # a second call with IDENTICAL content (but different bookkeeping timestamps, exactly like a
    # real rerun computed at a different wall-clock time) must be a no-op, never a write
    rerun_values = dict(values)
    rerun_values.update({"preregistered_at": 9999, "started_at": 9999, "completed_at": 9999,
                         "created_ms": 9999, "updated_ms": 9999})
    status = record_experiment_registry(conn, rerun_values)
    conn.commit()
    assert status == "NOOP_IDENTICAL"

    # the STORED row's bookkeeping timestamps were never rewritten by the no-op
    row = conn.execute(
        "SELECT started_at, completed_at, created_ms, updated_ms FROM experiment_registry WHERE experiment_id=?",
        ("E-TEST-001",),
    ).fetchone()
    assert row == (1000, 2000, 1000, 1000)
    conn.close()


def test_content_difference_fails_closed_with_immutable_conflict(tmp_path):
    conn = _db(tmp_path)
    values = _base_registry_values()
    record_experiment_registry(conn, values)
    conn.commit()

    changed = dict(values)
    changed["dataset_hash"] = "different-hash-because-population-changed"
    with pytest.raises(ImmutableExperimentConflict, match="IMMUTABLE_EXPERIMENT_CONFLICT"):
        record_experiment_registry(conn, changed)

    # the stored row must be completely unaffected by the failed attempt
    row = conn.execute(
        "SELECT dataset_hash FROM experiment_registry WHERE experiment_id=?", ("E-TEST-001",)
    ).fetchone()
    assert row[0] == "abc123"
    conn.close()


def test_scientific_verdict_difference_also_fails_closed(tmp_path):
    """Any content column, not just dataset_hash, triggers the conflict --
    e.g. a rerun that would have produced a different verdict string."""
    conn = _db(tmp_path)
    record_experiment_registry(conn, _base_registry_values())
    conn.commit()
    changed = _base_registry_values(scientific_verdict="ANSWERED_REGIME_DEPENDENT_BASELINE")
    with pytest.raises(ImmutableExperimentConflict):
        record_experiment_registry(conn, changed)
    conn.close()


def test_results_new_insert_then_identical_noop_then_conflict(tmp_path):
    conn = _db(tmp_path)
    record_experiment_registry(conn, _base_registry_values())
    conn.commit()

    results = [("metric_a", "1.0"), ("metric_b", "{'x': 2}")]
    status1 = record_experiment_results(conn, "E-TEST-001", results, schema_version=10,
                                         provenance="test", created_ms=1000)
    conn.commit()
    assert status1 == "INSERTED"
    n1 = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", ("E-TEST-001",)
    ).fetchone()[0]
    assert n1 == 2

    # identical content, different bookkeeping timestamp -> no-op, no new rows
    status2 = record_experiment_results(conn, "E-TEST-001", results, schema_version=10,
                                         provenance="test-rerun", created_ms=9999)
    conn.commit()
    assert status2 == "NOOP_IDENTICAL"
    n2 = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", ("E-TEST-001",)
    ).fetchone()[0]
    assert n2 == 2

    # a genuinely different result set (e.g. a rerun over a changed population) fails closed
    different_results = [("metric_a", "1.0"), ("metric_b", "{'x': 3}")]
    with pytest.raises(ImmutableExperimentConflict, match="IMMUTABLE_EXPERIMENT_CONFLICT"):
        record_experiment_results(conn, "E-TEST-001", different_results, schema_version=10,
                                   provenance="test", created_ms=1000)
    # stored results completely unaffected by the failed attempt
    n3 = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", ("E-TEST-001",)
    ).fetchone()[0]
    assert n3 == 2
    stored = dict(conn.execute(
        "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=?", ("E-TEST-001",)
    ).fetchall())
    assert stored["metric_b"] == "{'x': 2}"
    conn.close()


def test_no_on_conflict_do_update_or_delete_in_module_source():
    """Structural guard against regressing back to the old silent-overwrite
    pattern: the two write functions themselves must never use
    ON CONFLICT DO UPDATE for these two tables, and must never DELETE
    experiment_results wholesale (the module docstring quotes that old SQL
    verbatim for documentation purposes, so this checks only the function
    bodies, not the whole module source)."""
    import inspect
    import ami.warehouse.experiment_ledger as ledger_mod

    for fn in (ledger_mod.record_experiment_registry, ledger_mod.record_experiment_results):
        src = inspect.getsource(fn)
        # "ON CONFLICT(" (with the opening paren, as actual SQL syntax requires) rather than the
        # bare phrase "ON CONFLICT DO UPDATE" -- the latter also appears in this function's own
        # explanatory docstring describing the OLD pattern it replaces
        assert "ON CONFLICT(" not in src, f"{fn.__name__} must not use ON CONFLICT DO UPDATE"
        assert "DELETE FROM experiment_results" not in src, f"{fn.__name__} must not bulk-delete results"


def test_registry_columns_matches_schema_table_definition():
    """REGISTRY_COLUMNS must stay in sync with experiment_registry's actual
    column set (excluding nothing, adding nothing) -- a schema drift here
    would silently break every INSERT built from this tuple."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    actual_columns = [row[1] for row in conn.execute("PRAGMA table_info(experiment_registry)")]
    assert list(REGISTRY_COLUMNS) == actual_columns
    conn.close()
