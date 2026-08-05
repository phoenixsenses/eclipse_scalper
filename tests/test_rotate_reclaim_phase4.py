"""Phase-4 reclaim: the irreversible step, so its refusals are the real product.

Two properties carry the safety here and both are pinned below:

  * nothing is deleted unless every table in the segment is provably reproduced
    elsewhere -- this is the last moment that comparison is possible at all;
  * rotation_state.json is updated BEFORE the unlink. Unlink-first would leave
    every `open_union_ro` caller raising against a segment that no longer exists.
    Drop-first degrades to "readers see live only", which is recoverable while the
    bytes are still on disk.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

import scripts.rotate_microstructure_db as RC


def _mkdb(path: Path, tables: dict[str, int]) -> None:
    conn = sqlite3.connect(str(path))
    for name, rows in tables.items():
        conn.execute(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY, ts_ms INTEGER)")
        conn.executemany(f"INSERT INTO {name}(ts_ms) VALUES (?)",
                         [(1_700_000_000_000 + i,) for i in range(rows)])
    conn.commit()
    conn.close()


def _manifest(root: Path, table: str, rows: int) -> None:
    d = root / table
    d.mkdir(parents=True, exist_ok=True)
    (d / "_manifest.jsonl").write_text(json.dumps({"key": "k", "rows": rows}) + "\n",
                                       encoding="utf-8")


@pytest.fixture()
def estate(tmp_path, monkeypatch):
    """A miniature of the real estate: frozen segment + keeper + parquet manifests."""
    frozen = tmp_path / "microstructure.db"
    keeper = tmp_path / "keeper.db"
    parquet = tmp_path / "parquet"
    state = tmp_path / "rotation_state.json"

    _mkdb(frozen, {"book_ticker": 3, "liquidations": 5, "spot_prices": 2})
    _mkdb(keeper, {"liquidations": 5, "spot_prices": 2})
    _manifest(parquet, "book_ticker", 3)

    state.write_text(json.dumps({
        "live_db_path": str(tmp_path / "live.db"),
        "cutoff_ms": 1_700_000_000_000,
        "frozen_segments": [{"path": str(frozen), "start_ms": 1, "end_ms": 2}],
    }), encoding="utf-8")

    monkeypatch.setattr(RC, "STATE_PATH", state)
    monkeypatch.setattr(RC, "KEEPER_DB", keeper)
    monkeypatch.setattr(RC, "PARQUET_ROOT", parquet)
    monkeypatch.setattr(RC, "PARQUET_TABLES", {"book_ticker": 3})
    return {"frozen": frozen, "keeper": keeper, "parquet": parquet, "state": state}


def _args(**kw):
    base = dict(reclaim=True, confirm=False, attach_keeper=False,
                abandon_small_table_history=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _segments(state: Path):
    return [s["path"] for s in json.loads(state.read_text(encoding="utf-8"))["frozen_segments"]]


# ---------------------------------------------------------------------------
# preflight refusals -- the last moment the comparison is possible
# ---------------------------------------------------------------------------


def test_preflight_clean_estate_has_no_blockers(estate):
    assert RC.reclaim_preflight(estate["frozen"]) == []


def test_blocks_a_table_that_exists_nowhere_else(estate):
    conn = sqlite3.connect(str(estate["frozen"]))
    conn.execute("CREATE TABLE orphan_table (id INTEGER PRIMARY KEY, ts_ms INTEGER)")
    conn.commit()
    conn.close()
    blockers = RC.reclaim_preflight(estate["frozen"])
    assert any("orphan_table" in b for b in blockers)


def test_blocks_when_the_archive_row_count_disagrees_with_the_census(estate):
    _manifest(estate["parquet"], "book_ticker", 2)  # one row short
    blockers = RC.reclaim_preflight(estate["frozen"])
    assert any("book_ticker" in b and "census" in b for b in blockers)


def test_blocks_when_keeper_lost_rows(estate):
    conn = sqlite3.connect(str(estate["keeper"]))
    conn.execute("DELETE FROM liquidations WHERE id = 1")
    conn.commit()
    conn.close()
    blockers = RC.reclaim_preflight(estate["frozen"])
    assert any("liquidations" in b and "keeper" in b for b in blockers)


def test_blocks_a_missing_parquet_manifest(estate):
    (estate["parquet"] / "book_ticker" / "_manifest.jsonl").unlink()
    assert any("book_ticker" in b for b in RC.reclaim_preflight(estate["frozen"]))


def test_a_blocked_preflight_deletes_nothing(estate):
    (estate["parquet"] / "book_ticker" / "_manifest.jsonl").unlink()
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True, attach_keeper=True))
    assert estate["frozen"].exists()
    assert _segments(estate["state"]) == [str(estate["frozen"])]


# ---------------------------------------------------------------------------
# the keeper decision must be made, not defaulted
# ---------------------------------------------------------------------------


def test_neither_keeper_flag_is_refused(estate):
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True))
    assert estate["frozen"].exists()


def test_both_keeper_flags_are_refused(estate):
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True, attach_keeper=True,
                            abandon_small_table_history=True))
    assert estate["frozen"].exists()


def test_dry_run_mutates_nothing(estate):
    assert RC.do_reclaim(_args(attach_keeper=True)) == 0
    assert estate["frozen"].exists()
    assert _segments(estate["state"]) == [str(estate["frozen"])]


# ---------------------------------------------------------------------------
# the load-bearing order: state first, unlink second
# ---------------------------------------------------------------------------


def test_attach_keeper_swaps_the_segment_and_deletes(estate):
    assert RC.do_reclaim(_args(confirm=True, attach_keeper=True)) == 0
    assert not estate["frozen"].exists()
    assert _segments(estate["state"]) == [str(estate["keeper"])]


def test_abandon_leaves_no_segments_and_deletes(estate):
    assert RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True)) == 0
    assert not estate["frozen"].exists()
    assert _segments(estate["state"]) == []


def test_state_is_updated_even_when_the_unlink_fails(estate, monkeypatch):
    """A role still holding the file open must not leave readers pointing at it."""
    def _locked(self, *a, **k):
        raise OSError(32, "The process cannot access the file because it is being used")

    monkeypatch.setattr(Path, "unlink", _locked)
    assert RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True)) == 0
    assert estate["frozen"].exists()          # still on disk
    assert _segments(estate["state"]) == []   # but no reader will ask for it


def test_refuses_when_there_is_nothing_to_reclaim(estate):
    estate["state"].write_text(json.dumps({
        "live_db_path": str(estate["frozen"].parent / "live.db"),
        "cutoff_ms": None, "frozen_segments": [],
    }), encoding="utf-8")
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True))
