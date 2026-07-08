"""AMI CVD REPAIR + WINDOWED TAKER-FLOW -- tests for
ami/cvd/cvd_canonical_migration.py: the controlled canonical migration/
backfill entry point (Goal: composes the already-rehearsal-tested schema +
frozen-package backfill into ONE auditable, idempotent call, exactly the
code path the real migration uses).

DISPOSABLE_DB_ONLY here too: every test runs against a disposable COPY of the
real canonical.sqlite (never the real path itself) + a READ-ONLY connection
to the frozen disposable rehearsal database (never written). This module's
own NOT_CALLED_AUTOMATICALLY contract means `run_canonical_migration()` is
never invoked as an import side effect -- these tests call it explicitly, the
same way the one-off real-migration script will.

Run: pytest tests/test_ami_cvd_canonical_migration.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import inspect
import shutil
import sqlite3

from ami.cvd import cvd_canonical_migration as migration
from ami.warehouse import schema as wschema
from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH

FROZEN_SOURCE_PATH = (
    "D:/eclipse_scalper/data/ami/cvd_rehearsal_disposable_20260705/"
    "cvd_rehearsal_disposable.sqlite"
)

EXPECTED_ROW_COUNTS = {
    "ami_agg_trades_repaired": 40934,
    "ami_cvd_repair_batch_ledger": 8,
    "ami_cvd_windowed_flow": 1840,
    "ami_cvd_windowed_flow_proxy": 1840,
    "ami_cvd_bucket_exclusions": 104,
    "ami_cvd_window_quality_v1": 1840,
}


def test_not_called_automatically_no_module_level_connect():
    """Structural guard: importing this module must never open any database
    -- no module-level call to ami.warehouse.schema.connect/DEFAULT_PATH."""
    for fn in (migration.run_canonical_migration, migration.content_hashes, migration.row_counts):
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


def _open_frozen_source_ro():
    conn = sqlite3.connect(f"file:{FROZEN_SOURCE_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def test_run_canonical_migration_against_disposable_copy_reproduces_rehearsal_counts(tmp_path):
    """Branch-aware (birth-truncated-geometry-migration precedent,
    `test_schema_fingerprint_changes_only_by_addition`): once the real
    canonical migration has actually been applied to REAL_CANONICAL_PATH, any
    disposable COPY of it already contains every CVD row -- a fresh
    `run_canonical_migration()` call against it is then correctly an
    all-NOOP idempotent replay, not a fresh insert. Both branches assert the
    same accounting identity (inserted + noop_identical == expected)."""
    conn = _prepare_disposable(tmp_path)
    pre_counts = migration.row_counts(conn)
    real_migration_already_applied = pre_counts == EXPECTED_ROW_COUNTS

    source_ro = _open_frozen_source_ro()
    try:
        result = migration.run_canonical_migration(conn, source_ro, provenance="test-canonical-migration")
    finally:
        source_ro.close()

    for table, expected in EXPECTED_ROW_COUNTS.items():
        assert result[table]["inserted"] + result[table]["noop_identical"] == expected
        if real_migration_already_applied:
            assert result[table]["noop_identical"] == expected
        else:
            assert result[table]["inserted"] == expected
            assert result[table]["noop_identical"] == 0

    counts = migration.row_counts(conn)
    assert counts == EXPECTED_ROW_COUNTS
    conn.close()


def test_run_canonical_migration_is_idempotent_on_rerun(tmp_path):
    conn = _prepare_disposable(tmp_path)
    source_ro = _open_frozen_source_ro()
    try:
        migration.run_canonical_migration(conn, source_ro, provenance="test-run1")
        hashes_1 = migration.content_hashes(conn)
        counts_1 = migration.row_counts(conn)

        result2 = migration.run_canonical_migration(conn, source_ro, provenance="test-run2")
        hashes_2 = migration.content_hashes(conn)
        counts_2 = migration.row_counts(conn)
    finally:
        source_ro.close()

    assert hashes_1 == hashes_2
    assert counts_1 == counts_2
    for table, expected in EXPECTED_ROW_COUNTS.items():
        assert result2[table]["inserted"] == 0
        assert result2[table]["noop_identical"] == expected
    conn.close()


def test_conflicting_content_under_same_identity_raises(tmp_path):
    conn = _prepare_disposable(tmp_path)
    source_ro = _open_frozen_source_ro()
    try:
        migration.run_canonical_migration(conn, source_ro, provenance="test-run1")
    finally:
        source_ro.close()

    # Corrupt one already-migrated row's content in the canonical copy, then
    # attempt to re-migrate from the (unmodified) frozen source -- must raise,
    # never silently overwrite.
    row = conn.execute("SELECT feature_id FROM ami_cvd_windowed_flow LIMIT 1").fetchone()
    conn.execute("UPDATE ami_cvd_windowed_flow SET cvd_qty = cvd_qty + 999.0 WHERE feature_id=?", row)
    conn.commit()

    source_ro = _open_frozen_source_ro()
    try:
        raised = False
        try:
            migration.run_canonical_migration(conn, source_ro, provenance="test-run2")
        except migration.FrozenSourceRowConflict:
            raised = True
        assert raised
    finally:
        source_ro.close()
    conn.close()


def test_protected_invariants_unchanged_by_migration(tmp_path):
    conn = _prepare_disposable(tmp_path)
    pre = {
        "ami_events": conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0],
        "ami_signal_lifecycle": conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0],
        "ami_cycles": conn.execute("SELECT COUNT(*) FROM ami_cycles").fetchone()[0],
        "ami_birth_truncated_cascade_geometry": conn.execute(
            "SELECT COUNT(*) FROM ami_birth_truncated_cascade_geometry").fetchone()[0],
    }
    source_ro = _open_frozen_source_ro()
    try:
        migration.run_canonical_migration(conn, source_ro, provenance="test-invariants")
    finally:
        source_ro.close()
    post = {
        "ami_events": conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0],
        "ami_signal_lifecycle": conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0],
        "ami_cycles": conn.execute("SELECT COUNT(*) FROM ami_cycles").fetchone()[0],
        "ami_birth_truncated_cascade_geometry": conn.execute(
            "SELECT COUNT(*) FROM ami_birth_truncated_cascade_geometry").fetchone()[0],
    }
    assert pre == post
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    assert fk_violations == []
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok"
    conn.close()
