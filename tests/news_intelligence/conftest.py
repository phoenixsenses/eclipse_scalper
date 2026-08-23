"""Local test setup for the news intelligence layer.

Two jobs, and the second one needs justifying.

**1. Put `src/` on the path.** The package lives at
`src/eclipse/news_intelligence/` and the repository has no installed
distribution, so the path is set here rather than in every test file.

**2. Override the session-wide canonical-database isolation fixture.** The root
`tests/conftest.py` hashes and copies a 213 MB SQLite file at session start so
that no test can write to the real one. That guard is right for the tests it was
written for and pointless here: nothing in this subtree imports `ami`, opens a
database, or touches disk at all. Paying ~400 MB of I/O to protect a file these
tests cannot reach is a real cost on a machine that is already busy with
research jobs.

Overriding a safety fixture deserves a guard of its own, so this one is
fail-closed rather than merely absent: `_no_database_access` records every call to
`sqlite3.connect` while these tests run and asserts there were none. If someone
later adds a test here that opens a database, the override stops being safe and
the suite says so instead of silently running unprotected.

The first version of this guard asserted that no database *module* was imported,
which could never pass: the parent conftest imports `ami.warehouse.schema` and
`sqlite3` at collection time, before this file has any say. Counting calls tests
the property that actually matters.
"""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope="session", autouse=True)
def _isolate_real_canonical_db():
    """Replaces the root fixture for this directory only. Deliberately does nothing."""
    yield None


@pytest.fixture(scope="session", autouse=True)
def _no_database_access():
    """Fail-closed proof that skipping the isolation fixture was safe.

    Counts calls rather than imports. `sqlite3` is already imported by the
    parent conftest before this subtree is collected, so its presence proves
    nothing; a call to `connect` during these tests would prove the opposite.
    """
    import sqlite3

    original = sqlite3.connect
    calls: list[tuple] = []

    def recording_connect(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    sqlite3.connect = recording_connect
    try:
        yield
    finally:
        sqlite3.connect = original

    assert not calls, (
        f"news intelligence tests opened {len(calls)} database connection(s): {calls[:3]}. "
        "Skipping the canonical-database isolation fixture is only safe while this "
        "subtree touches no database — remove the dependency or delete the override."
    )
