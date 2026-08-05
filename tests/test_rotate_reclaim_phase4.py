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


def _manifest(root: Path, table: str, rows: int, *, source_fp: str = "",
              verified: bool = True, mode: str = "parquet+sqlite") -> None:
    """A complete archive: manifest, a real part file, and a proof-gate record."""
    d = root / table / "sym=X" / "dt=2026-01-01"
    d.mkdir(parents=True, exist_ok=True)
    part = d / "part-0000.parquet"
    part.write_bytes(b"PAR1" * 8)
    rel = str(part.relative_to(root))
    rec = {"key": f"{table}|X|2026-01-01", "rows": rows, "digest": "d" * 64,
           "path": rel, "file_bytes": part.stat().st_size}
    (root / table / "_manifest.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    if verified:
        (root / table / "_verified.jsonl").write_text(json.dumps({
            "key": rec["key"], "digest": rec["digest"], "rows": rows,
            "file_bytes": rec["file_bytes"], "mode": mode, "source_fp": source_fp,
        }) + "\n", encoding="utf-8")


@pytest.fixture()
def estate(tmp_path, monkeypatch):
    """A miniature of the real estate: frozen segment + keeper + parquet manifests."""
    frozen = tmp_path / "microstructure.db"
    keeper = tmp_path / "keeper.db"
    parquet = tmp_path / "parquet"
    state = tmp_path / "rotation_state.json"

    live = tmp_path / "live.db"
    _mkdb(frozen, {"book_ticker": 3, "liquidations": 5, "spot_prices": 2})
    _mkdb(keeper, {"liquidations": 5, "spot_prices": 2})
    _mkdb(live, {"liquidations": 0, "open_interest": 0, "spot_prices": 0})
    _manifest(parquet, "book_ticker", 3, source_fp=RC.FX.source_fingerprint(frozen))

    state.write_text(json.dumps({
        "live_db_path": str(live),
        "cutoff_ms": 1_800_000_000_000,
        "frozen_segments": [{"path": str(frozen), "start_ms": 1, "end_ms": 2}],
        "rotated_utc": "2026-07-23T20:40:41.215144+00:00",
    }), encoding="utf-8")

    monkeypatch.setattr(RC, "STATE_PATH", state)
    monkeypatch.setattr(RC, "KEEPER_DB", keeper)
    monkeypatch.setattr(RC, "PARQUET_ROOT", parquet)
    monkeypatch.setattr(RC, "PARQUET_TABLES", {"book_ticker": 3})
    monkeypatch.setattr(RC, "KEEPER_UNION_TABLES", ("liquidations", "spot_prices"))
    return {"frozen": frozen, "keeper": keeper, "parquet": parquet,
            "state": state, "live": live}


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


def test_the_state_is_already_dropped_at_the_moment_unlink_is_called(estate, monkeypatch):
    """Pins the ORDER, not just the end state.

    Asserting only 'the state ended up updated' is satisfied by a reversal that
    unlinks first and updates afterwards -- and that reversal is the dangerous one:
    a crash between the two leaves every open_union_ro caller raising against a
    segment that is gone. So the assertion happens INSIDE unlink.
    """
    observed = {}

    def _spy(self, *a, **k):
        observed["segments"] = _segments(estate["state"])
        raise OSError(32, "The process cannot access the file because it is being used")

    monkeypatch.setattr(Path, "unlink", _spy)
    rc = RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True))
    assert observed["segments"] == [], "unlink ran while readers still pointed at the segment"
    assert rc == 3, "a failed unlink must not report success"


def test_a_failed_unlink_reports_a_distinct_non_zero_code(estate, monkeypatch):
    """Exit 0 here would let a chained caller record the reclaim as done."""
    monkeypatch.setattr(Path, "unlink",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError(32, "locked")))
    assert RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True)) == 3
    assert estate["frozen"].exists()


# ---------------------------------------------------------------------------
# the proof gate, not the writer's own row count
# ---------------------------------------------------------------------------


def test_blocks_a_table_that_never_passed_the_proof_gate(estate):
    (estate["parquet"] / "book_ticker" / "_verified.jsonl").unlink()
    blockers = RC.reclaim_preflight(estate["frozen"])
    assert any("never passed the proof gate" in b for b in blockers)


def test_blocks_a_parquet_only_verification(estate):
    _manifest(estate["parquet"], "book_ticker", 3,
              source_fp=RC.FX.source_fingerprint(estate["frozen"]), mode="parquet-only")
    blockers = RC.reclaim_preflight(estate["frozen"])
    assert any("never re-queried from SQLite" in b for b in blockers)


def test_blocks_a_verification_made_against_a_different_source_state(estate):
    _manifest(estate["parquet"], "book_ticker", 3, source_fp="db:1:1")
    blockers = RC.reclaim_preflight(estate["frozen"])
    assert any("DIFFERENT" in b for b in blockers)


def test_blocks_when_the_parquet_files_are_gone_but_the_manifest_remains(estate):
    """The manifest is a description; deleting the data must not read as covered."""
    for p in (estate["parquet"] / "book_ticker").rglob("*.parquet"):
        p.unlink()
    blockers = RC.reclaim_preflight(estate["frozen"])
    assert any("absent" in b for b in blockers)


def test_blocks_when_a_parquet_file_changed_size(estate):
    part = next((estate["parquet"] / "book_ticker").rglob("*.parquet"))
    part.write_bytes(b"x")
    blockers = RC.reclaim_preflight(estate["frozen"])
    assert any("changed size" in b for b in blockers)


# ---------------------------------------------------------------------------
# the keeper must not approve its own deletion
# ---------------------------------------------------------------------------


def test_the_keeper_can_never_be_the_segment_being_reclaimed(estate):
    """After --attach-keeper the keeper IS the segment; a second run must refuse."""
    blockers = RC.reclaim_preflight(estate["keeper"])
    assert blockers and "only remaining copy" in blockers[0]


def test_a_second_reclaim_after_attach_keeper_refuses(estate):
    assert RC.do_reclaim(_args(confirm=True, attach_keeper=True)) == 0
    assert _segments(estate["state"]) == [str(estate["keeper"])]
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True))
    assert estate["keeper"].exists(), "the keeper deleted itself by its own approval"


# ---------------------------------------------------------------------------
# installing the keeper as a segment
# ---------------------------------------------------------------------------


def test_refuses_a_keeper_missing_a_column_the_live_table_has(estate):
    conn = sqlite3.connect(str(estate["live"]))
    conn.execute("ALTER TABLE liquidations ADD COLUMN side TEXT")
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True, attach_keeper=True))
    assert estate["frozen"].exists()


def test_refuses_a_keeper_that_overlaps_the_live_file(estate):
    conn = sqlite3.connect(str(estate["live"]))
    conn.execute("INSERT INTO liquidations(ts_ms) VALUES (1700000000000)")  # older than keeper max
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True, attach_keeper=True))
    assert estate["frozen"].exists()


# ---------------------------------------------------------------------------
# the state file must survive its own rewrite
# ---------------------------------------------------------------------------


def test_a_corrupt_state_write_is_rolled_back_and_nothing_is_deleted(estate, monkeypatch):
    real = Path.replace

    def _truncating_replace(self, target):
        real(self, target)
        Path(target).write_text("{ this is not json", encoding="utf-8")

    monkeypatch.setattr(Path, "replace", _truncating_replace)
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True))
    assert estate["frozen"].exists()
    assert _segments(estate["state"]) == [str(estate["frozen"])]


def test_the_cutover_timestamp_is_carried_through(estate):
    assert RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True)) == 0
    doc = json.loads(estate["state"].read_text(encoding="utf-8"))
    assert doc["rotated_utc"] == "2026-07-23T20:40:41.215144+00:00"
    assert doc["reclaimed_utc"] != doc["rotated_utc"]


def test_refuses_when_there_is_nothing_to_reclaim(estate):
    estate["state"].write_text(json.dumps({
        "live_db_path": str(estate["frozen"].parent / "live.db"),
        "cutoff_ms": None, "frozen_segments": [],
    }), encoding="utf-8")
    with pytest.raises(SystemExit):
        RC.do_reclaim(_args(confirm=True, abandon_small_table_history=True))
