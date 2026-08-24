"""A collector must never create a database nothing will ever read.

Before Phase-4, every collector's hardcoded `data/microstructure.db` default was a
real file carrying `attrib +R` from the cutover, so a hand-launch without
`--db-path` failed loudly with "attempt to write a readonly database". Deleting
that file removed the loud failure and left the silent one: SQLite happily creates
an empty database at a free path and writes a live feed into it, while every reader
goes on reading the file named in rotation_state.json.

Split-brain feed, on the only asset that cannot be re-collected, with no error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ami.storage import union_reader as UR


@pytest.fixture()
def estate(tmp_path):
    live = tmp_path / "microstructure_02.db"
    live.write_bytes(b"")
    state = tmp_path / "rotation_state.json"
    state.write_text(json.dumps({
        "live_db_path": str(live), "cutoff_ms": 1, "frozen_segments": [],
    }), encoding="utf-8")
    return {"live": live, "state": state, "root": tmp_path}


def _resolve(estate, explicit=None):
    return UR.resolve_writer_db_path(explicit, rotation_state_path=estate["state"])


def test_no_argument_resolves_to_the_live_database(estate):
    assert _resolve(estate) == estate["live"]


def test_the_deleted_pre_rotation_path_is_refused(estate):
    """The exact hand-launch mistake Phase-4 turned from loud into silent."""
    gone = estate["root"] / "microstructure.db"
    assert not gone.exists()
    with pytest.raises(UR.WriterTargetError, match="second, unread feed"):
        _resolve(estate, str(gone))


def test_a_first_ever_run_may_still_create_the_live_database(estate):
    """The rule is not "must exist" -- bootstrapping has to work."""
    estate["live"].unlink()
    assert _resolve(estate) == estate["live"]


def test_an_existing_alternate_database_is_allowed(estate):
    """Someone who set one up deliberately is not making this mistake."""
    other = estate["root"] / "deliberate.db"
    other.write_bytes(b"")
    assert _resolve(estate, str(other)) == other


def test_the_refusal_names_both_paths(estate):
    """An operator must be able to see what they asked for and what is live."""
    gone = estate["root"] / "microstructure.db"
    with pytest.raises(UR.WriterTargetError) as exc:
        _resolve(estate, str(gone))
    assert "microstructure.db" in str(exc.value)
    assert "microstructure_02.db" in str(exc.value)


def test_every_collector_routes_its_db_path_through_the_guard():
    """The guard is worthless if a writer bypasses it."""
    writers = {
        "data/bookticker_collector.py": "resolve_writer_db_path(args.db_path)",
        "data/oi_spot_poller.py": "resolve_writer_db_path(args.db)",
        "data/microstructure_collector.py": "resolve_writer_db_path(args.db_path)",
    }
    for rel, needle in writers.items():
        body = Path(rel).read_text(encoding="utf-8")
        assert needle in body, f"{rel} does not pass its db path through the guard"
