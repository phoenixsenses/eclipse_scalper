"""[PHASE 7A-P TEST ISOLATION SAFETY CLOSURE]

Root cause this closes: several existing "real data smoke test" functions
(test_ami_research_w1/w3/w4/w5a/w6/w6rs/w7a/w10a_*.py) open a WRITABLE
connection to the REAL data/ami/canonical.sqlite via
`ami.warehouse.schema.connect(DEFAULT_PATH)` -- some of them additionally
call `init_schema(conn)` on that connection. Running the full test suite
after `ami/warehouse/schema.py` was edited (CANONICAL_SCHEMA_VERSION bumped,
new DDL added) caused init_schema()'s idempotent-but-real DDL/version-bump
to land on the REAL file as an unintended side effect, before any deliberate,
approved canonical migration had run (SCHEMA_DRIFT_BLOCKER, discovered and
reported before any data was touched).

This fixture makes that structurally impossible for the whole test session:
DEFAULT_PATH is redirected to a disposable copy (created once, real data,
free to read+write), and ami.warehouse.schema.connect() itself fail-closed-
rejects any WRITABLE connection to the REAL path while this isolation is
active (backstop for a test that hardcodes the real path as a literal
string instead of importing DEFAULT_PATH).

Session-scoped + autouse: applies to every test in this directory without
requiring any individual test file to opt in or change its own code.
"""
from __future__ import annotations
import hashlib
import os
import shutil

import pytest

import ami.knowledge.store as _knowledge_mod
import ami.warehouse.schema as _schema_mod
from ami.lifecycle.migration_rehearsal import schema_fingerprint as _schema_fingerprint


def _file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# Populated by _isolate_real_canonical_db's setup (BEFORE the copy is made),
# read by both the teardown assertion below and the explicit
# real_canonical_hash_and_mtime_at_session_start fixture -- a single source
# of truth for "what the real file looked like at the very start of the
# session", never recomputed lazily (which could silently adopt an
# already-drifted state as its baseline if first read late).
_SESSION_START_SNAPSHOT: dict = {}


@pytest.fixture(scope="session", autouse=True)
def _isolate_real_canonical_db(tmp_path_factory):
    real_path = _schema_mod.REAL_CANONICAL_PATH_IMMUTABLE

    # regression guard: the REAL file's hash/mtime must be byte-identical
    # before this fixture copies it and after the ENTIRE test session
    # completes -- if it isn't, isolation failed somewhere and a test wrote
    # to the real file despite the guards above.
    _SESSION_START_SNAPSHOT["hash"] = _file_hash(real_path)
    _SESSION_START_SNAPSHOT["mtime"] = os.path.getmtime(real_path)
    _conn_fp = _schema_mod.connect(real_path, read_only=True)
    try:
        _SESSION_START_SNAPSHOT["fingerprint"] = _schema_fingerprint(_conn_fp)
        _SESSION_START_SNAPSHOT["schema_version"] = _conn_fp.execute(
            "SELECT version FROM schema_versions WHERE component='canonical_warehouse'"
        ).fetchone()[0]
    finally:
        _conn_fp.close()

    isolated_dir = tmp_path_factory.mktemp("real_canonical_test_copy")
    isolated_path = isolated_dir / "canonical.sqlite"
    shutil.copy2(real_path, isolated_path)

    original_default_path = _schema_mod.DEFAULT_PATH
    _schema_mod.DEFAULT_PATH = isolated_path
    _schema_mod._TEST_ISOLATION_ACTIVE = True
    try:
        yield isolated_path
    finally:
        _schema_mod.DEFAULT_PATH = original_default_path
        _schema_mod._TEST_ISOLATION_ACTIVE = False

        hash_after_session = _file_hash(real_path)
        mtime_after_session = os.path.getmtime(real_path)
        assert mtime_after_session == _SESSION_START_SNAPSHOT["mtime"], (
            "TEST_ISOLATION_SAFETY_BLOCKER: real canonical.sqlite mtime changed during this "
            "test session -- a test wrote to the real file despite isolation")
        assert hash_after_session == _SESSION_START_SNAPSHOT["hash"], (
            "TEST_ISOLATION_SAFETY_BLOCKER: real canonical.sqlite content changed during this "
            "test session -- a test wrote to the real file despite isolation")
        _conn_fp_after = _schema_mod.connect(real_path, read_only=True)
        try:
            fp_after = _schema_fingerprint(_conn_fp_after)
            version_after = _conn_fp_after.execute(
                "SELECT version FROM schema_versions WHERE component='canonical_warehouse'"
            ).fetchone()[0]
        finally:
            _conn_fp_after.close()
        assert fp_after == _SESSION_START_SNAPSHOT["fingerprint"], (
            "TEST_ISOLATION_SAFETY_BLOCKER: real canonical.sqlite schema fingerprint changed "
            "during this test session -- a test applied DDL to the real file despite isolation")
        assert version_after == _SESSION_START_SNAPSHOT["schema_version"], (
            "TEST_ISOLATION_SAFETY_BLOCKER: real canonical.sqlite schema_versions row changed "
            "during this test session -- a test applied a migration to the real file despite isolation")


@pytest.fixture(scope="session")
def real_canonical_test_copy_path(_isolate_real_canonical_db):
    """Explicit accessor for the disposable, real-data-populated copy tests
    are redirected to -- identical to ami.warehouse.schema.DEFAULT_PATH for
    the duration of the session, exposed by name for tests that want it
    directly rather than importing DEFAULT_PATH themselves."""
    return _isolate_real_canonical_db


@pytest.fixture(scope="session")
def real_canonical_hash_and_mtime_at_session_start(_isolate_real_canonical_db):
    """Snapshot of the REAL (untouched) canonical.sqlite's hash/mtime, taken
    before this session's isolation copy was made -- exposed so an explicit
    test (see test_test_isolation_safety.py) can assert no drift occurred,
    independent of the fixture-teardown assertion above."""
    return _SESSION_START_SNAPSHOT["hash"], _SESSION_START_SNAPSHOT["mtime"]


@pytest.fixture(scope="session")
def real_canonical_fingerprint_and_version_at_session_start(_isolate_real_canonical_db):
    """Snapshot of the REAL canonical.sqlite's schema fingerprint + schema_versions
    row, taken before this session's isolation copy was made."""
    return _SESSION_START_SNAPSHOT["fingerprint"], _SESSION_START_SNAPSHOT["schema_version"]


# ---------------------------------------------------------------------------
# [BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1 TEST-ISOLATION CLOSURE]
# Same discipline as _isolate_real_canonical_db above, for
# data/ami/knowledge.sqlite -- see ami/knowledge/store.py's module-level note
# for the incident this closes (M-0033/M-0034 additive schema accidentally
# landed on the real file via KnowledgeStore()'s previously-unguarded,
# bound-at-def-time default path).
# ---------------------------------------------------------------------------

_KNOWLEDGE_SESSION_START_SNAPSHOT: dict = {}


@pytest.fixture(scope="session", autouse=True)
def _isolate_real_knowledge_db(tmp_path_factory):
    real_path = _knowledge_mod.REAL_KNOWLEDGE_PATH_IMMUTABLE

    _KNOWLEDGE_SESSION_START_SNAPSHOT["hash"] = _file_hash(real_path)
    _KNOWLEDGE_SESSION_START_SNAPSHOT["mtime"] = os.path.getmtime(real_path)

    isolated_dir = tmp_path_factory.mktemp("real_knowledge_test_copy")
    isolated_path = isolated_dir / "knowledge.sqlite"
    shutil.copy2(real_path, isolated_path)

    original_default_path = _knowledge_mod.DEFAULT_PATH
    _knowledge_mod.DEFAULT_PATH = isolated_path
    _knowledge_mod._TEST_ISOLATION_ACTIVE = True
    try:
        yield isolated_path
    finally:
        _knowledge_mod.DEFAULT_PATH = original_default_path
        _knowledge_mod._TEST_ISOLATION_ACTIVE = False

        hash_after_session = _file_hash(real_path)
        mtime_after_session = os.path.getmtime(real_path)
        assert mtime_after_session == _KNOWLEDGE_SESSION_START_SNAPSHOT["mtime"], (
            "TEST_ISOLATION_SAFETY_BLOCKER: real knowledge.sqlite mtime changed during this "
            "test session -- a test wrote to the real file despite isolation")
        assert hash_after_session == _KNOWLEDGE_SESSION_START_SNAPSHOT["hash"], (
            "TEST_ISOLATION_SAFETY_BLOCKER: real knowledge.sqlite content changed during this "
            "test session -- a test wrote to the real file despite isolation")


@pytest.fixture(scope="session")
def real_knowledge_test_copy_path(_isolate_real_knowledge_db):
    """Explicit accessor for the disposable, real-data-populated knowledge.sqlite
    copy tests are redirected to for the duration of the session."""
    return _isolate_real_knowledge_db
