"""[PHASE 7A-P TEST ISOLATION SAFETY CLOSURE]

Regression test: the REAL data/ami/canonical.sqlite must never change
(hash/mtime/schema fingerprint) as a result of running the test suite --
this is the exact failure mode that caused the SCHEMA_DRIFT_BLOCKER finding
(running the full test suite after editing ami/warehouse/schema.py applied
the new DDL/version bump to the real file via test_ami_research_w*.py's
real-data smoke tests calling connect(DEFAULT_PATH)/init_schema(conn)).

Run: pytest tests/test_test_isolation_safety.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import hashlib
import os

import pytest

import ami.warehouse.schema as schema_mod
from ami.lifecycle.migration_rehearsal import schema_fingerprint


def _file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_default_path_is_redirected_away_from_real_file_during_tests():
    # the session-scoped autouse fixture in conftest.py must have already
    # redirected DEFAULT_PATH by the time any test body runs
    assert schema_mod.DEFAULT_PATH != schema_mod.REAL_CANONICAL_PATH_IMMUTABLE
    assert schema_mod._TEST_ISOLATION_ACTIVE is True


def test_writable_connect_to_real_path_is_fail_closed_rejected():
    with pytest.raises(RuntimeError, match="TEST_ISOLATION_SAFETY_BLOCKER"):
        schema_mod.connect(schema_mod.REAL_CANONICAL_PATH_IMMUTABLE, read_only=False)


def test_readonly_connect_to_real_path_still_permitted():
    # read-only access to the real file must remain possible (postflight
    # verification queries, existing mode=ro smoke tests) -- only WRITABLE
    # connections are fail-closed rejected
    conn = schema_mod.connect(schema_mod.REAL_CANONICAL_PATH_IMMUTABLE, read_only=True)
    try:
        row = conn.execute(
            "SELECT version FROM schema_versions WHERE component='canonical_warehouse'"
        ).fetchone()
        assert row is not None
    finally:
        conn.close()


def test_real_canonical_db_hash_mtime_unchanged_so_far_this_session(
    real_canonical_hash_and_mtime_at_session_start,
):
    # explicit, discoverable regression check (in addition to the fixture's
    # own teardown assertion, which runs once at the very end of the whole
    # session): whatever tests have already run in this pytest invocation,
    # up to and including this one, must not have touched the real file.
    real_path = schema_mod.REAL_CANONICAL_PATH_IMMUTABLE
    expected_hash, expected_mtime = real_canonical_hash_and_mtime_at_session_start
    assert os.path.getmtime(real_path) == expected_mtime
    assert _file_hash(real_path) == expected_hash


def test_real_canonical_db_schema_fingerprint_unchanged_so_far_this_session(
    real_canonical_fingerprint_and_version_at_session_start,
):
    expected_fp, expected_version = real_canonical_fingerprint_and_version_at_session_start
    conn = schema_mod.connect(schema_mod.REAL_CANONICAL_PATH_IMMUTABLE, read_only=True)
    try:
        fp = schema_fingerprint(conn)
        version = conn.execute(
            "SELECT version FROM schema_versions WHERE component='canonical_warehouse'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert fp == expected_fp, "real canonical.sqlite schema fingerprint changed mid-session"
    assert version == expected_version, "real canonical.sqlite schema_versions row changed mid-session"
