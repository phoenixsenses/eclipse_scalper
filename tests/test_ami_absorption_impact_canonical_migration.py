"""AMI FAM_CASCADE_ABSORPTION_IMPACT -- tests for
ami/absorption/cascade_absorption_impact_canonical_migration.py: the
controlled canonical migration/backfill entry point (schema 12->13, M-0035).

DISPOSABLE_DB_ONLY here too: every test runs against a disposable COPY of the
real canonical.sqlite (never the real path itself) + a READ-ONLY connection
to the frozen retained rehearsal database (never written). This module's own
NOT_CALLED_AUTOMATICALLY contract means `run_canonical_migration()` is never
invoked as an import side effect -- these tests call it explicitly, the same
way the one-off real-migration script does.

Run: pytest tests/test_ami_absorption_impact_canonical_migration.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import inspect
import shutil
import sqlite3

from ami.absorption import cascade_absorption_impact_canonical_migration as migration
from ami.warehouse import schema as wschema
from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH

FROZEN_SOURCE_PATH = (
    "D:/eclipse_scalper/.runtime_temp/absorption_impact_rehearsal_v1/"
    "rehearsal_run1.sqlite"
)

EXPECTED_ROW_COUNTS = {
    "ami_absorption_impact_windowed_flow": 1619,
    "ami_absorption_impact_window_quality_v1": 1620,
    "ami_absorption_impact_exclusions": 1,
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
    """Branch-aware (CVD/geometry migration precedent,
    `test_schema_fingerprint_changes_only_by_addition`): once the real
    canonical migration has actually been applied to REAL_CANONICAL_PATH, any
    disposable COPY of it already contains every absorption/impact row -- a
    fresh `run_canonical_migration()` call against it is then correctly an
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

    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    assert fk_violations == []
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok"
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


def test_content_hashes_match_frozen_row_accounting_freeze_values(tmp_path):
    """The row-accounting freeze (commit 931cd3dd) froze three content
    hashes for the disposable rehearsal's tables. This test proves the
    canonical migration's copy -- under the operator's renamed
    ami_absorption_impact_* tables -- produces byte-identical content hashes,
    since content_hashes() hashes the same declared columns in the same
    order, only the table name differs."""
    conn = _prepare_disposable(tmp_path)
    source_ro = _open_frozen_source_ro()
    try:
        migration.run_canonical_migration(conn, source_ro, provenance="test-hash-parity")
    finally:
        source_ro.close()

    frozen_hashes = {
        "ami_absorption_impact_windowed_flow": "f7c834cc8ebe90708e308629f1921a050d58520ad5560422b09406a7d1ca8942",
        "ami_absorption_impact_window_quality_v1": "5d1a205c7f79ca1b269307e34750c0d46dc104c8a799e9b4d01c862d307d7ba0",
        "ami_absorption_impact_exclusions": "5e3ae2e524fcdbd5d045698a5a14bd397ae2c21bf0ff9ae2f54f2502c35a3ff7",
    }
    assert migration.content_hashes(conn) == frozen_hashes
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
    row = conn.execute(
        "SELECT feature_id FROM ami_absorption_impact_windowed_flow LIMIT 1").fetchone()
    conn.execute(
        "UPDATE ami_absorption_impact_windowed_flow SET signed_notional = signed_notional + 999.0 "
        "WHERE feature_id=?", row)
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


def test_exclusion_row_never_also_a_feature_row(tmp_path):
    """The single W3600 source-gapped signal must remain excluded, never
    imputed into the feature table."""
    conn = _prepare_disposable(tmp_path)
    source_ro = _open_frozen_source_ro()
    try:
        migration.run_canonical_migration(conn, source_ro, provenance="test-exclusion-identity")
    finally:
        source_ro.close()

    excl = conn.execute(
        "SELECT signal_id, window_id, reason_code FROM ami_absorption_impact_exclusions").fetchall()
    assert excl == [("SIG-e03382b4d82720185dfc870a", "W3600", "CONFIRMED_GAP_OVERLAP")]

    both = conn.execute("""
        SELECT COUNT(*) FROM (
          SELECT signal_id, window_id FROM ami_absorption_impact_windowed_flow
          INTERSECT
          SELECT signal_id, window_id FROM ami_absorption_impact_exclusions
        )
    """).fetchone()[0]
    assert both == 0
    conn.close()


def test_protected_invariants_unchanged_by_migration(tmp_path):
    conn = _prepare_disposable(tmp_path)
    pre = {
        "ami_events": conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0],
        "ami_signal_lifecycle": conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0],
        "ami_cycles": conn.execute("SELECT COUNT(*) FROM ami_cycles").fetchone()[0],
        "ami_birth_truncated_cascade_geometry": conn.execute(
            "SELECT COUNT(*) FROM ami_birth_truncated_cascade_geometry").fetchone()[0],
        "experiment_registry": conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0],
        "experiment_results": conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0],
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
        "experiment_registry": conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0],
        "experiment_results": conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0],
    }
    assert pre == post
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    assert fk_violations == []
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok"
    conn.close()
