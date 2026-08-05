"""Microstructure DB rotation cutover — Phase 2 orchestrator (gated, human-invoked).

Freezes the live `data/microstructure.db` (grown to ~837GB) and starts a fresh
`data/microstructure_02.db`, so the old file can later be archived to Parquet and
deleted WHOLE (Phase 3/4) — reclaiming ~800GB with NO vacuum and no lasting
interruption to forward collection. The readers were made rotation-aware in Phase 1
(`ami/storage/union_reader.py`: `open_union_ro` unions live+frozen, `current_live_db_path`
resolves the live file). This script performs the WRITE-side cutover.

SAFETY MODEL — this script NEVER acts by default:
  * default mode = --dry-run: prints the full plan + read-only preflight, touches nothing.
  * every mutating phase requires an explicit sub-command AND --confirm.
  * the seconds-scale writer switch is split around a HUMAN stop/start:
        1) operator: stop_eclipse.ps1              (writers down)
        2) this:     --cutover --confirm           (checkpoint+freeze+flip state) [only op that touches the big file, RW, once]
        3) operator: start_eclipse.ps1             (writers resolve the fresh file via current_live_db_path)
        4) this:     --canary                       (verify fresh file is receiving rows)
  * fail-closed: any preflight failure aborts before mutation; never guesses.
  * rollback: on canary failure it PRINTS exact rollback steps (does not auto-rollback).

Timing: run at a UTC month boundary (so the frozen file = whole closed months → clean
Phase-4 whole-file delete). --allow-mid-month overrides with a loud warning.

Usage:
  python -m scripts.rotate_microstructure_db                       # dry-run plan + preflight (default)
  python -m scripts.rotate_microstructure_db --preseed  --confirm  # create+schema-seed the fresh DB (idempotent-guarded)
  python -m scripts.rotate_microstructure_db --cutover  --confirm  # WRITERS-MUST-BE-STOPPED: checkpoint+freeze+flip state
  python -m scripts.rotate_microstructure_db --canary              # after start_eclipse: verify fresh file is live
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ami.storage.union_reader import (  # noqa: E402
    DEFAULT_LIVE_PATH, DEFAULT_ROTATION_STATE_PATH, UNION_TABLES,
    load_rotation_state, current_live_db_path,
)

LIVE_DB = DEFAULT_LIVE_PATH                                  # data/microstructure.db
FRESH_DB = REPO_ROOT / "data" / "microstructure_02.db"      # the next live file
STATE_PATH = DEFAULT_ROTATION_STATE_PATH                    # data/rotation_state.json
GB = 1024 ** 3
CANARY_TABLES = ("book_ticker", "mark_prices", "open_interest", "spot_prices")  # the 4 live-writer targets
CANARY_WAIT_SEC = 300
CANARY_POLL_SEC = 10


def _ro(conn_path: Path) -> sqlite3.Connection:
    uri = "file:" + str(conn_path).replace("\\", "/") + "?mode=ro"
    c = sqlite3.connect(uri, uri=True, timeout=10.0)
    c.execute("PRAGMA query_only=ON")
    return c


def _fail(msg: str) -> None:
    print(f"  [ABORT] {msg}")
    raise SystemExit(2)


def _full_schema(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """(type, name, sql) for every table+index+trigger in the live DB (sqlite internal
    tables + autoindexes excluded — those have NULL sql)."""
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
        "ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END, name"
    ).fetchall()
    return [(t, n, s) for (t, n, s) in rows]


def _table_max_ts(conn: sqlite3.Connection, table: str) -> int | None:
    try:
        r = conn.execute(f"SELECT MAX(ts_ms) FROM {table}").fetchone()
        return int(r[0]) if r and r[0] is not None else None
    except sqlite3.Error:
        return None


def _is_month_boundary(now: dt.datetime) -> bool:
    # within the first 6h of a UTC month = "at the boundary" (operator runs it just after 00:00 UTC day-1)
    return now.day == 1 and now.hour < 6


def preflight(args) -> dict:
    print("=" * 78)
    print("MICROSTRUCTURE DB ROTATION — PHASE 2 CUTOVER  (preflight, read-only)")
    print("=" * 78)
    state = load_rotation_state(STATE_PATH)
    now = dt.datetime.now(dt.timezone.utc)

    # 1) not already rotated
    if not state.is_pre_rotation:
        print(f"  rotation_state.json ALREADY lists {len(state.frozen_segments)} frozen segment(s):")
        for s in state.frozen_segments:
            print(f"      frozen: {s.path}")
        print(f"  current live = {state.live_db_path}")
        _fail("state is not pre-rotation. This orchestrator does the FIRST cutover only; "
              "a re-rotation is a separate reviewed step. Nothing to do.")

    # 2) live db present + is the default single-file world
    if not LIVE_DB.exists():
        _fail(f"live DB not found: {LIVE_DB}")
    if str(current_live_db_path()) != str(LIVE_DB):
        _fail(f"current_live_db_path()={current_live_db_path()} != {LIVE_DB}; unexpected state.")
    live_gb = round(LIVE_DB.stat().st_size / GB, 2)
    ro = _ro(LIVE_DB)

    # 3) schema capture (FULL — every writer's table must exist in the fresh file)
    schema = _full_schema(ro)
    tables = [n for (t, n, _s) in schema if t == "table"]
    missing_union = [t for t in UNION_TABLES if t not in tables]
    if missing_union:
        ro.close()
        _fail(f"live DB missing expected union tables {missing_union}; refusing.")

    # 4) per-table cutoff (the frozen segment's end = live's current max ts_ms)
    per_table_max = {t: _table_max_ts(ro, t) for t in UNION_TABLES}
    per_table_min = {}
    for t in UNION_TABLES:
        try:
            r = ro.execute(f"SELECT MIN(ts_ms) FROM {t}").fetchone()
            per_table_min[t] = int(r[0]) if r and r[0] is not None else None
        except sqlite3.Error:
            per_table_min[t] = None
    ro.close()
    seg_end = max([v for v in per_table_max.values() if v is not None], default=None)
    seg_start = min([v for v in per_table_min.values() if v is not None], default=None)

    # 5) timing + disk + fresh-file collision
    month_ok = _is_month_boundary(now)
    fresh_exists = FRESH_DB.exists()
    import shutil
    free_gb = round(shutil.disk_usage(str(REPO_ROOT.anchor)).free / GB, 2)

    non_union_tables = [t for t in tables if t not in UNION_TABLES]

    print(f"  now (UTC)            : {now.isoformat()}")
    print(f"  live DB              : {LIVE_DB}  ({live_gb} GB)")
    print(f"  fresh DB (target)    : {FRESH_DB}  {'*** ALREADY EXISTS ***' if fresh_exists else '(will be created)'}")
    print(f"  disk free            : {free_gb} GB")
    print(f"  union tables (6)     : all present; frozen segment ts range "
          f"[{seg_start} .. {seg_end}]")
    print(f"  schema objects       : {len(schema)} (tables+indexes+triggers) to clone into fresh DB")
    print(f"  month-boundary       : {'YES' if month_ok else 'NO -> needs --allow-mid-month'}")
    if non_union_tables:
        print(f"  NON-UNION tables ({len(non_union_tables)}) will be RE-CREATED EMPTY in the fresh file "
              f"(readers union only the 6; these read live-only post-rotation):")
        print(f"      {', '.join(non_union_tables)}")
        print("      -> continuity-stateful tables (vol_state/detector_*/gaps) reset to empty; "
              "acceptable only if dormant. VERIFY before --cutover.")

    print("\n  PLAN (each step is a separate gated invocation):")
    print("   1. operator: stop_eclipse.ps1                         (writers down)")
    print("   2. this:     --preseed --confirm                      (create fresh DB + clone schema)")
    print("   3. this:     --cutover --confirm  [--allow-mid-month] (checkpoint live WAL, freeze old +R, flip rotation_state.json)")
    print("   4. operator: start_eclipse.ps1                        (writers resolve fresh file)")
    print("   5. this:     --canary                                 (verify fresh file receiving rows)")
    print("=" * 78)
    return {"state": state, "now": now, "live_gb": live_gb, "schema": schema,
            "seg_start": seg_start, "seg_end": seg_end, "month_ok": month_ok,
            "fresh_exists": fresh_exists, "non_union_tables": non_union_tables}


def do_preseed(pf: dict, args) -> None:
    if not args.confirm:
        _fail("--preseed requires --confirm.")
    if FRESH_DB.exists():
        _fail(f"fresh DB already exists ({FRESH_DB}); refusing to overwrite. Delete it deliberately if re-seeding.")
    schema = pf["schema"]
    print(f"  creating fresh DB {FRESH_DB} + cloning {len(schema)} schema objects ...")
    conn = sqlite3.connect(str(FRESH_DB), timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        for (typ, name, sql) in schema:
            conn.execute(sql)
        conn.commit()
        got = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    finally:
        conn.close()
    want = {n for (t, n, _s) in schema if t == "table"}
    missing = want - got
    if missing:
        _fail(f"pre-seed verification FAILED: fresh DB missing tables {missing}")
    for t in UNION_TABLES:
        if t not in got:
            _fail(f"pre-seed verification FAILED: union table {t} absent in fresh DB")
    print(f"  [OK] fresh DB seeded, WAL on, {len(want)} tables present (incl all 6 union tables + OI/spot).")
    print("  NEXT: operator stop_eclipse.ps1, then --cutover --confirm.")


def _live_is_locked() -> bool:
    """Best-effort: can we get a RESERVED lock on the live DB (i.e. is a writer active)?
    Opens RW briefly with a short busy timeout and tries BEGIN IMMEDIATE."""
    try:
        c = sqlite3.connect(str(LIVE_DB), timeout=2.0)
        try:
            c.execute("PRAGMA busy_timeout=1500")
            c.execute("BEGIN IMMEDIATE")
            c.execute("ROLLBACK")
            return False
        finally:
            c.close()
    except sqlite3.OperationalError:
        return True


def do_cutover(pf: dict, args) -> None:
    if not args.confirm:
        _fail("--cutover requires --confirm.")
    if not FRESH_DB.exists():
        _fail("fresh DB not seeded yet; run --preseed --confirm first.")
    if not pf["month_ok"] and not args.allow_mid_month:
        _fail("not a UTC month boundary; frozen file would hold a partial month. "
              "Re-run at 00:00 UTC on day 1, or pass --allow-mid-month to override deliberately.")

    # writers MUST be stopped: the live file must not be write-locked. (Advisory; the authoritative
    # machine gate is the wal_checkpoint(TRUNCATE) busy-guard below, which fails if any writer is attached.)
    if _live_is_locked():
        _fail("live DB is WRITE-LOCKED (a collector is still running). "
              "Run stop_eclipse.ps1 first, confirm no python writer holds microstructure.db, then retry.")
    print("  [OK] live DB is not write-locked (writers appear stopped).")

    # 1) checkpoint the WAL fully so the frozen .db is self-contained (TRUNCATE empties -wal). This is
    #    ALSO the authoritative "no writer attached" gate: busy != 0 => a writer is still live => abort.
    print("  checkpointing live WAL (TRUNCATE) so the frozen file is self-contained ...")
    cp = sqlite3.connect(str(LIVE_DB), timeout=30.0)
    try:
        cp.execute("PRAGMA busy_timeout=30000")
        res = cp.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        cp.commit()
    finally:
        cp.close()
    print(f"    wal_checkpoint(TRUNCATE) -> {res}  (busy,log,checkpointed)")
    if res and res[0] != 0:
        _fail("WAL checkpoint did not complete (busy != 0) — a writer is still attached. Abort.")

    # 2) recompute the frozen segment's TRUE ts-range from the now-quiescent, checkpointed live DB
    #    (review note 5: preflight's seg_end was snapshotted while writers were still running, so it
    #    understated the real max; here writers are stopped + WAL is truncated => this is authoritative).
    ro = _ro(LIVE_DB)
    try:
        maxes = [_table_max_ts(ro, t) for t in UNION_TABLES]
        mins = []
        for t in UNION_TABLES:
            try:
                r = ro.execute(f"SELECT MIN(ts_ms) FROM {t}").fetchone()
                mins.append(int(r[0]) if r and r[0] is not None else None)
            except sqlite3.Error:
                mins.append(None)
    finally:
        ro.close()
    seg_end = max([v for v in maxes if v is not None], default=None)
    seg_start = min([v for v in mins if v is not None], default=None)

    # 3) flip rotation_state.json FIRST, then freeze (review note 2: flip-then-freeze is the strictly
    #    safer order — if the freeze in step 4 fails, writers already resolve the FRESH file via the
    #    flipped state, and the still-writable old file is simply left unwritten, not a wedge that
    #    points restarted writers at a read-only file).
    doc = {
        "live_db_path": str(FRESH_DB),
        "cutoff_ms": seg_end,
        "frozen_segments": [
            {"path": str(LIVE_DB), "start_ms": seg_start, "end_ms": seg_end}
        ],
        "rotated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)
    ns = load_rotation_state(STATE_PATH)   # re-load through the real loader to prove it parses (fail-closed)
    if ns.is_pre_rotation or str(ns.live_db_path) != str(FRESH_DB):
        _fail("rotation_state.json did not flip correctly after write — inspect immediately.")
    print(f"  [OK] rotation_state.json flipped: live -> {FRESH_DB.name}, frozen += {LIVE_DB.name} "
          f"(ts [{seg_start} .. {seg_end}])")

    # 4) freeze the old file read-only (attrib +R). State is already flipped, so a failure here is
    #    SAFE (writers target fresh); we surface it loudly and leave the -R retryable.
    print("  freezing old file read-only (attrib +R) ...")
    r = subprocess.run(["attrib", "+R", str(LIVE_DB)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] attrib +R failed: {r.stderr.strip()}")
        print("  State is already flipped (writers -> fresh, SAFE). Re-run `attrib +R` on the old file "
              "manually to enforce read-only, then continue to --canary.")
        return
    try:  # verify read-only by attempting a RW open (should fail)
        t = sqlite3.connect(str(LIVE_DB), timeout=2.0)
        t.execute("PRAGMA busy_timeout=1000")
        t.execute("BEGIN IMMEDIATE"); t.execute("ROLLBACK"); t.close()
        print("  [WARN] post-freeze the live file is STILL writable — attrib +R did not take. "
              "State is flipped (SAFE); enforce +R manually before archiving.")
    except sqlite3.OperationalError:
        print("    [OK] old file verified read-only (RW open refused).")

    print("\n  CUTOVER FILE-OPS COMPLETE. NEXT: operator runs start_eclipse.ps1, then this --canary.")
    print("  ROLLBACK (only if start/canary fails): attrib -R the old file, delete data/rotation_state.json "
          "(back to pre-rotation), delete the empty fresh DB, restart.")


def do_canary(args) -> None:
    state = load_rotation_state(STATE_PATH)
    if state.is_pre_rotation:
        _fail("rotation_state.json is still pre-rotation; cutover has not run.")
    live = Path(state.live_db_path)
    if not live.exists():
        _fail(f"current live file missing: {live}")
    print(f"  canary: watching {live.name} for fresh rows in {CANARY_TABLES} (up to {CANARY_WAIT_SEC}s) ...")
    deadline = time.time() + CANARY_WAIT_SEC
    seen = {t: None for t in CANARY_TABLES}
    while time.time() < deadline:
        c = _ro(live)
        try:
            for t in CANARY_TABLES:
                seen[t] = _table_max_ts(c, t)
        finally:
            c.close()
        now_ms = int(time.time() * 1000)
        fresh = {t: (v is not None and now_ms - v < 180_000) for t, v in seen.items()}
        line = "  ".join(f"{t}:{'OK' if fresh[t] else 'wait'}" for t in CANARY_TABLES)
        print(f"    {line}")
        if all(fresh.values()):
            print("  [OK] all four live-writer tables are receiving fresh rows in the new file. Rotation LIVE.")
            print("  Phase 3 (archive frozen -> Parquet) and Phase 4 (whole-file delete) remain separately gated.")
            return
        time.sleep(CANARY_POLL_SEC)
    _fail(f"canary TIMEOUT: not all of {CANARY_TABLES} are fresh in {live.name}. "
          "Writers may not have picked up the fresh file (check launcher repoints / start_eclipse). "
          "Consider ROLLBACK (see --cutover output).")


# Row counts from reports/governance/storage/frozen_db_size_census.json (2026-07-30).
# The frozen segment has been read-only and quiescent since the 2026-07-23 cutover,
# so these are the totals an archive must match exactly before it may replace the file.
PARQUET_TABLES = {
    "book_ticker": 5_723_357_020,
    "agg_trades": 427_185_688,
    "mark_prices": 24_441_427,
    "detector_heartbeat": 551_629_265,
}
KEEPER_DB = REPO_ROOT / "data" / "keeper_frozen_smalltables.db"
PARQUET_ROOT = REPO_ROOT / "data" / "archives" / "parquet_v1"


def _parquet_rows(table: str) -> int | None:
    """Rows recorded in the archive manifest for `table`, or None if unusable."""
    manifest = PARQUET_ROOT / table / "_manifest.jsonl"
    if not manifest.exists():
        return None
    total = 0
    try:
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += int(json.loads(line).get("rows", 0))
    except (OSError, ValueError):
        return None
    return total


def reclaim_preflight(seg_path: Path) -> list[str]:
    """READ-ONLY. Returns the reasons this frozen segment may NOT be deleted.

    Every table in the segment must be provably reproduced somewhere else. The two
    routes are the Parquet archive (the four big tables) and the keeper DB (the small
    ones). This is the last moment the comparison is possible at all -- after the
    unlink the source of truth is gone, so nothing here may be deferred.
    """
    blockers: list[str] = []

    for table, expected in PARQUET_TABLES.items():
        rows = _parquet_rows(table)
        if rows is None:
            blockers.append(f"{table}: no readable Parquet manifest at {PARQUET_ROOT / table}")
        elif rows != expected:
            blockers.append(
                f"{table}: archive holds {rows:,} rows, census says {expected:,} "
                f"({expected - rows:+,})")

    if not KEEPER_DB.exists():
        blockers.append(f"keeper DB missing at {KEEPER_DB}")
        return blockers
    if not seg_path.exists():
        blockers.append(f"frozen segment already absent at {seg_path}")
        return blockers

    src = _ro(seg_path)
    dst = _ro(KEEPER_DB)
    try:
        frozen_tables = {r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        keeper_tables = {r[0] for r in dst.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        for table in sorted(frozen_tables - set(PARQUET_TABLES)):
            if table not in keeper_tables:
                blockers.append(f"{table}: in the frozen segment but in neither Parquet nor keeper")
                continue
            a = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            b = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if a != b:
                blockers.append(f"{table}: frozen has {a:,} rows, keeper has {b:,}")
    finally:
        src.close()
        dst.close()
    return blockers


def do_reclaim(args) -> int:
    """Phase 4: drop the frozen segment from rotation_state, THEN unlink it.

    Order is load-bearing and mirrors the flip-then-freeze reasoning in do_cutover.
    Unlink-first would leave every `open_union_ro` caller raising RotationStateError
    against a segment that no longer exists -- total breakage. Drop-first degrades
    to "readers see live only", which is recoverable by restoring the state file
    while the data is still on disk.
    """
    state = load_rotation_state(STATE_PATH)
    if state.is_pre_rotation:
        _fail("rotation_state.json lists no frozen segments; nothing to reclaim.")
    if len(state.frozen_segments) != 1:
        _fail(f"expected exactly one frozen segment, found {len(state.frozen_segments)}; "
              "reclaiming several at once is not supported.")
    seg = state.frozen_segments[0]
    seg_path = Path(seg.path)

    print(f"\nPHASE 4 - RECLAIM  segment={seg_path}")
    size_gib = seg_path.stat().st_size / (1024 ** 3) if seg_path.exists() else 0.0
    print(f"  size={size_gib:,.2f} GiB   (irreversible)")

    print("  preflight (read-only): proving every table survives elsewhere ...")
    blockers = reclaim_preflight(seg_path)
    if blockers:
        print(f"  [BLOCKED] {len(blockers)} unmet precondition(s):")
        for b in blockers:
            print(f"    - {b}")
        _fail("refusing to delete: the segment holds data that is not provably reproduced.")
    print("    [OK] all frozen tables reproduced in Parquet or keeper (counts match exactly).")

    # The small tables live in the keeper, which is NOT currently a segment. Dropping
    # the frozen file without attaching it silently ends liq/OI/spot history for every
    # union reader. Neither outcome is a safe default, so the operator must say which.
    if args.attach_keeper == args.abandon_small_table_history:
        _fail("choose exactly one of --attach-keeper / --abandon-small-table-history "
              "(the keeper holds liquidations/open_interest/spot_prices history that "
              "the union reader loses when the frozen segment goes away).")

    if args.attach_keeper:
        ro = _ro(KEEPER_DB)
        try:
            lo = ro.execute("SELECT MIN(ts_ms) FROM liquidations").fetchone()[0]
            hi = ro.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
        finally:
            ro.close()
        new_segments = [{"path": str(KEEPER_DB), "start_ms": lo, "end_ms": hi}]
        print(f"    keeper will replace it as the frozen segment (ts [{lo} .. {hi}])")
    else:
        new_segments = []
        print("    liq/OI/spot history will NOT be visible to the union reader after this.")

    if not args.confirm:
        print("\n  (dry-run: nothing mutated. Re-run with --confirm to act.)")
        return 0

    doc = {
        "live_db_path": state.live_db_path,
        "cutoff_ms": state.cutoff_ms,
        "frozen_segments": new_segments,
        "rotated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reclaimed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reclaimed_segment": str(seg_path),
    }
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)
    ns = load_rotation_state(STATE_PATH)  # re-load through the real loader (fail-closed)
    if any(str(Path(s.path)) == str(seg_path) for s in ns.frozen_segments):
        _fail("rotation_state.json still lists the segment after the write - NOT deleting.")
    print(f"  [OK] rotation_state.json updated: segment dropped, "
          f"{len(ns.frozen_segments)} segment(s) remain.")

    print("  unlinking ...")
    subprocess.run(["attrib", "-R", str(seg_path)], capture_output=True, text=True)
    try:
        seg_path.unlink()
    except OSError as exc:
        print(f"  [WARN] unlink failed: {exc}")
        print("  State is ALREADY updated, so readers are consistent (live-only) and this is SAFE.")
        print("  A running role almost certainly still holds the file open: stop_eclipse.ps1, "
              "delete the file manually, then start_eclipse.ps1.")
        return 0
    print(f"  [OK] deleted {seg_path.name} - {size_gib:,.2f} GiB reclaimed.")
    print("\n  NEXT: restart the stack so every reader reopens against the new state.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Microstructure DB rotation cutover (Phase 2, gated).")
    ap.add_argument("--dry-run", action="store_true", help="preflight + plan only (default if no phase given)")
    ap.add_argument("--preseed", action="store_true", help="create fresh DB + clone schema (needs --confirm)")
    ap.add_argument("--cutover", action="store_true", help="checkpoint+freeze+flip (writers must be stopped; needs --confirm)")
    ap.add_argument("--canary", action="store_true", help="verify fresh file receiving rows (after start_eclipse)")
    ap.add_argument("--reclaim", action="store_true",
                    help="Phase 4: drop the frozen segment from rotation_state, then delete it "
                         "(irreversible; needs --confirm)")
    ap.add_argument("--attach-keeper", action="store_true",
                    help="reclaim: keep liq/OI/spot history by making the keeper DB the frozen segment")
    ap.add_argument("--abandon-small-table-history", action="store_true",
                    help="reclaim: accept that the union reader loses liq/OI/spot history")
    ap.add_argument("--confirm", action="store_true", help="required for mutating phases")
    ap.add_argument("--allow-mid-month", action="store_true", help="override the UTC month-boundary guard")
    args = ap.parse_args()

    if args.reclaim:
        # Deliberately does NOT run the Phase-2 preflight: that one inspects the live
        # DB for a cutover that already happened, and its month-boundary guard is
        # meaningless here.
        return do_reclaim(args)

    if args.canary and not (args.preseed or args.cutover):
        return do_canary(args) or 0

    pf = preflight(args)   # always runs (read-only)
    if args.preseed:
        do_preseed(pf, args)
    elif args.cutover:
        do_cutover(pf, args)
    elif not args.canary:
        print("\n  (dry-run: nothing mutated. Add a phase flag + --confirm to act.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
