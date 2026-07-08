"""BATCH-P1-001 + BATCH-P2-001 + BATCH-P3-001: canonical warehouse skeleton tests.

Run: pytest tests/test_ami_warehouse_schema.py --basetemp <scratchpad> -p no:cacheprovider
"""
import sqlite3

import pytest

from ami.warehouse.schema import CANONICAL_SCHEMA_VERSION, _add_column_if_missing, connect, init_schema

EXPECTED_TABLES = {
    "schema_versions",
    "artifact_registry",
    "artifact_lineage",
    "question_families",
    "question_registry",
    "contradiction_registry",
    "operator_decision_queue",
    "namespace_registry",
}

EXPECTED_TABLES_PHASE2 = {
    "evidence_contamination",
    "researcher_exposure_ledger",
    "mt_family_registry",
    "causal_assumption_registry",
    "data_quality_events",
    "market_structure_versions",
}

EXPECTED_TABLES_PHASE3 = {
    "ami_events",
    "ami_cycles",
    "event_cycle_membership",
}

EXPECTED_TABLES_PHASE4 = {
    "ami_candles",
    "ami_candle_morphology",
    "ami_swings",
    "ami_levels",
    "ami_pushes",
}

EXPECTED_TABLES_PHASE6 = {
    "experiment_registry",
    "experiment_results",
    "ami_candidate_universe",
}


def test_init_schema_creates_all_tables(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert EXPECTED_TABLES.issubset(tables)
    assert EXPECTED_TABLES_PHASE2.issubset(tables)
    assert EXPECTED_TABLES_PHASE3.issubset(tables)
    assert EXPECTED_TABLES_PHASE4.issubset(tables)
    assert EXPECTED_TABLES_PHASE6.issubset(tables)


def test_ami_cycles_empty_after_schema_init_alone(tmp_path):
    # init_schema() never writes data by itself -- ami_cycles population only
    # happens via ami/identity/cycle_resolver.py's seed() (BATCH-P3-005,
    # OD-003 approved A2+B2+C2), never as a side effect of schema creation.
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    n = conn.execute("SELECT COUNT(*) FROM ami_cycles").fetchone()[0]
    conn.close()
    assert n == 0


def test_foreign_keys_enforced_on_writable_connection(tmp_path):
    # FABLE-REVIEW-A F4: FK constraints must actually be enforced, not just declared.
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO event_cycle_membership (event_id, candidate_cycle_key, cycle_definition_version, "
            "is_canonical, schema_version, provenance, created_ms) VALUES (?,?,?,0,?,?,?)",
            ("NONEXISTENT-EVENT-ID", "key", "v1", 3, "test", 0),
        )
    conn.close()


def test_schema_versions_note_reflects_current_version(tmp_path):
    # FABLE-REVIEW-A F5: note must not stay hardcoded to an old batch label.
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    note = conn.execute(
        "SELECT note FROM schema_versions WHERE component='canonical_warehouse'"
    ).fetchone()[0]
    conn.close()
    assert str(CANONICAL_SCHEMA_VERSION) in note


def test_phase2_evidence_contamination_round_trip(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    conn.execute(
        "INSERT INTO evidence_contamination (contamination_id, hypothesis_id, hypothesis_birth_ts, "
        "hypothesis_origin_split, evidence_status, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("EC-001", "H-TEST-001", 0, "train", "INDEPENDENT_EVIDENCE", 2, "test", 0, 0),
    )
    conn.commit()
    row = conn.execute(
        "SELECT evidence_status FROM evidence_contamination WHERE contamination_id='EC-001'"
    ).fetchone()
    conn.close()
    assert row == ("INDEPENDENT_EVIDENCE",)


def test_init_schema_is_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    init_schema(conn)  # second call must be a no-op, not an error
    n = conn.execute("SELECT COUNT(*) FROM schema_versions WHERE component='canonical_warehouse'").fetchone()[0]
    version = conn.execute("SELECT version FROM schema_versions WHERE component='canonical_warehouse'").fetchone()[0]
    conn.close()
    assert n == 1
    assert version == CANONICAL_SCHEMA_VERSION


def test_round_trip_insert_select(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    conn.execute(
        "INSERT INTO artifact_registry (artifact_id, path, content_hash, role, canonical_status, "
        "namespace, schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("ART-001", "SYSTEM_STATE.md", "deadbeef", "state_doc", "CANONICAL",
         "ami_s34", 1, "phase0_audit", 1751500000000, 1751500000000),
    )
    conn.commit()
    row = conn.execute("SELECT path, canonical_status FROM artifact_registry WHERE artifact_id=?",
                        ("ART-001",)).fetchone()
    conn.close()
    assert row == ("SYSTEM_STATE.md", "CANONICAL")


def test_sql_dump_round_trip(tmp_path):
    """Phase 1 DoD: warehouse must survive a full .dump -> reload cycle (Protocol §23.1)."""
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    conn.execute(
        "INSERT INTO namespace_registry (namespace, meaning, schema_version, provenance, created_ms) "
        "VALUES ('ami_s34','test',1,'p',0)"
    )
    conn.commit()
    dump_sql = "\n".join(conn.iterdump())
    conn.close()

    restored_db = tmp_path / "restored.sqlite"
    restored = sqlite3.connect(restored_db)
    restored.executescript(dump_sql)
    tables = {r[0] for r in restored.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ns_row = restored.execute("SELECT meaning FROM namespace_registry WHERE namespace='ami_s34'").fetchone()
    restored.close()

    assert EXPECTED_TABLES.issubset(tables)
    assert ns_row == ("test",)


def test_read_only_open_after_init(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    conn.close()

    ro_conn = connect(db, read_only=True)
    tables = {r[0] for r in ro_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED_TABLES.issubset(tables)
    try:
        ro_conn.execute("INSERT INTO namespace_registry (namespace, meaning, schema_version, provenance, created_ms) "
                         "VALUES ('x','y',1,'p',0)")
        ro_conn.commit()
        assert False, "read-only connection must reject writes"
    except sqlite3.OperationalError:
        pass
    finally:
        ro_conn.close()


def test_ami_levels_has_touch_stats_point_in_time_column(tmp_path):
    # BATCH-P6-000 F-B2: column must exist and default to 0 for fresh DBs.
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ami_levels)")}
    conn.close()
    assert "touch_stats_point_in_time" in cols


def test_add_column_if_missing_is_idempotent_on_pre_existing_table(tmp_path):
    # Simulates the real canonical.sqlite that already had ami_levels before
    # touch_stats_point_in_time existed (F-B2 remediation must not crash on
    # an already-created table missing the new column).
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    conn.execute("CREATE TABLE demo_table (id INTEGER PRIMARY KEY, existing_col TEXT)")
    _add_column_if_missing(conn, "demo_table", "new_col", "new_col INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, "demo_table", "new_col", "new_col INTEGER NOT NULL DEFAULT 0")  # 2nd call: no-op
    cols = {row[1] for row in conn.execute("PRAGMA table_info(demo_table)")}
    conn.close()
    assert "new_col" in cols
