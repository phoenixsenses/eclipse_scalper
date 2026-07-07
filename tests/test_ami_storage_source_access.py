"""Focused tests: ami.storage.source_access -- read-only enforcement.

All rejection tests run against disposable in-memory/temp-file fixtures,
never the live database.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from ami.storage import source_access as SA


def _make_fixture_db(tmp_path):
    path = tmp_path / "fixture.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'a')")
    conn.commit()
    conn.close()
    return path


def test_open_read_only_uses_mode_ro_uri(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    assert log == []
    rows = conn.execute("SELECT * FROM t").fetchall()
    assert rows == [(1, "a")]
    conn.close()


def test_query_only_pragma_rejects_insert(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    with pytest.raises(sqlite3.Error):
        conn.execute("INSERT INTO t VALUES (2, 'b')")
    conn.close()


def test_authorizer_denies_insert(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    # query_only=ON already blocks this at the pragma level; the
    # authorizer is the second, independent layer -- verify the log
    # captures a denial when query_only is disabled for this check only.
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute("INSERT INTO t VALUES (3, 'c')")
    assert len(log) >= 1
    conn.close()


def test_authorizer_denies_update(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute("UPDATE t SET v='z' WHERE id=1")
    assert len(log) >= 1
    conn.close()


def test_authorizer_denies_delete(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute("DELETE FROM t WHERE id=1")
    assert len(log) >= 1
    conn.close()


def test_authorizer_denies_create_table(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute("CREATE TABLE u (id INTEGER)")
    assert len(log) >= 1
    conn.close()


def test_authorizer_denies_drop_table(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute("DROP TABLE t")
    assert len(log) >= 1
    conn.close()


def test_authorizer_denies_alter_table(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute("ALTER TABLE t ADD COLUMN w TEXT")
    assert len(log) >= 1
    conn.close()


def test_authorizer_denies_reindex(tmp_path):
    path = tmp_path / "fixture_idx.db"
    setup = sqlite3.connect(str(path))
    setup.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    setup.execute("CREATE INDEX idx_v ON t(v)")
    setup.execute("INSERT INTO t VALUES (1, 'a')")
    setup.commit()
    setup.close()
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute("REINDEX idx_v")
    assert len(log) >= 1
    conn.close()


def test_authorizer_denies_attach(tmp_path):
    path = _make_fixture_db(tmp_path)
    other = tmp_path / "other.db"
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute(f"ATTACH DATABASE '{other}' AS other")
    assert len(log) >= 1
    conn.close()


def test_writable_pragma_denied(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    with pytest.raises(sqlite3.Error):
        conn.execute("PRAGMA journal_mode=DELETE")
    assert len(log) >= 1
    conn.close()


def test_clean_read_only_session_has_empty_log(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    conn.execute("SELECT * FROM t").fetchall()
    conn.execute("SELECT COUNT(*) FROM t").fetchone()
    SA.assert_read_only_session_clean(log)  # must not raise
    conn.close()


def test_assert_clean_raises_when_denials_occurred(tmp_path):
    path = _make_fixture_db(tmp_path)
    conn, log = SA.open_read_only(path)
    conn.execute("PRAGMA query_only=OFF")
    try:
        conn.execute("DELETE FROM t")
    except sqlite3.Error:
        pass
    with pytest.raises(SA.SourceMutationRejected):
        SA.assert_read_only_session_clean(log)
    conn.close()


def test_default_source_path_points_at_microstructure_db():
    assert SA.DEFAULT_SOURCE_PATH.name == "microstructure.db"
    assert "data" in SA.DEFAULT_SOURCE_PATH.parts
