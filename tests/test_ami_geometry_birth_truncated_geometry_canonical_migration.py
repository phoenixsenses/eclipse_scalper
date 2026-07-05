"""AMI BIRTH-TRUNCATED CASCADE GEOMETRY -- tests for
ami/geometry/birth_truncated_geometry_canonical_migration.py: the controlled
canonical migration/backfill entry point (Goal: composes the already-tested
geometry backfill + field-quality-v2 backfill into ONE auditable, idempotent
call, exactly the code path the real migration uses).

DISPOSABLE_DB_ONLY here too: every test runs against a disposable COPY of the
real canonical.sqlite (never the real path itself) + real data/microstructure.db
opened ONLY mode=ro. This module's own NOT_CALLED_AUTOMATICALLY contract means
`run_canonical_migration()` is never invoked as an import side effect -- these
tests call it explicitly, the same way the one-off real-migration script will.

Run: pytest tests/test_ami_geometry_birth_truncated_geometry_canonical_migration.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect
import shutil
import sqlite3

from ami.geometry import birth_truncated_cascade_geometry as geo
from ami.geometry import birth_truncated_geometry_canonical_migration as migration
from ami.geometry import liquidation_source_quality_contract_v2 as v2
from ami.warehouse import schema as wschema
from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH

MICROSTRUCTURE_DB_PATH = "D:/eclipse_scalper/data/microstructure.db"


def test_not_called_automatically_no_module_level_connect():
    """Structural guard: importing this module must never open any database
    -- no module-level call to ami.warehouse.schema.connect/DEFAULT_PATH.
    Checks the two function bodies (the only executable code here), not the
    module's own docstring, which legitimately DISCUSSES this contract in
    prose."""
    for fn in (migration.run_canonical_migration, migration._reconstruct_anchors_fn):
        body = inspect.getsource(fn)
        assert "connect(DEFAULT_PATH)" not in body
        assert "schema.connect(" not in body


def _prepare_disposable(tmp_path):
    disposable = tmp_path / "disposable.sqlite"
    shutil.copy2(REAL_CANONICAL_PATH, disposable)
    conn = sqlite3.connect(disposable)
    conn.execute("PRAGMA foreign_keys=ON")
    wschema.init_schema(conn)  # additive-only; matches the real pre-migration step
    return conn


def test_run_canonical_migration_against_disposable_copy_real_data(tmp_path):
    conn = _prepare_disposable(tmp_path)
    conn_liq = sqlite3.connect(f"file:{MICROSTRUCTURE_DB_PATH}?mode=ro", uri=True)
    try:
        result = migration.run_canonical_migration(conn, conn_liq, provenance="test-canonical-migration")
    finally:
        conn_liq.close()

    assert result["geometry"]["candidate_n"] == result["geometry"]["accepted_n"] + result["geometry"]["rejected_n"]
    assert result["field_quality"]["accepted_n"] == result["geometry"]["accepted_n"] * len(geo._FEATURE_FIELDS)
    assert geo.row_counts(conn)["geometry"] == result["geometry"]["accepted_n"]
    assert v2.row_counts(conn)["field_quality_v2"] == result["field_quality"]["accepted_n"]
    conn.close()


def test_run_canonical_migration_is_idempotent_on_rerun(tmp_path):
    conn = _prepare_disposable(tmp_path)
    conn_liq = sqlite3.connect(f"file:{MICROSTRUCTURE_DB_PATH}?mode=ro", uri=True)
    try:
        migration.run_canonical_migration(conn, conn_liq, provenance="test-run1")
        hash_1 = geo.content_hash(conn)
        quality_hash_1 = v2.content_hash(conn)

        result2 = migration.run_canonical_migration(conn, conn_liq, provenance="test-run2")
        hash_2 = geo.content_hash(conn)
        quality_hash_2 = v2.content_hash(conn)
    finally:
        conn_liq.close()

    assert hash_1 == hash_2
    assert quality_hash_1 == quality_hash_2
    assert result2["geometry"]["rejected_n"] == 0
    conn.close()
