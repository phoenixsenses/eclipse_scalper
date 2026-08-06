"""Union read-only connection factory for the microstructure DB rotation
(Phase 1a). Lets every existing reader keep its exact SQL while transparently
spanning the live SQLite file plus one or more FROZEN (rotated-out, read-only)
SQLite files.

Mechanism (no query rewrite required):
  * open the live DB `?mode=ro`,
  * `ATTACH` each frozen DB `?mode=ro` as `frozen0`, `frozen1`, ...,
  * create a TEMP VIEW per table that `UNION ALL`s the live + frozen copies,
    using SCHEMA-qualified names (`main.T`, `frozenN.T`) so the view never
    recurses into itself.

Because SQLite resolves an unqualified name against the TEMP schema first, a
`FROM mark_prices` in a caller's untouched SQL binds to the union view, not to
`main.mark_prices`. Aggregates (MAX/MIN/SUM/COUNT/AVG) and range scans therefore
span both files with no change to the caller, and at the expected cost: a bounded
`WHERE ts_ms BETWEEN ...` gets an index SEARCH per branch (an ordered range even
plans as a `MERGE`).

COST CAVEAT -- AS-OF LOOKUPS (`ORDER BY ts_ms DESC LIMIT 1`) THROUGH THE VIEW ARE A
PLANNER COIN-FLIP. They are always CORRECT, but the cost hinges on one easily-missed
detail: SQLite can only merge an ordered compound when the ORDER BY term is a RESULT
COLUMN of that compound.

    SELECT mark_price        ... ORDER BY ts_ms DESC LIMIT 1  -> CO-ROUTINE + COMPOUND
                                                                 + SCAN + TEMP B-TREE
    SELECT ts_ms, mark_price ... ORDER BY ts_ms DESC LIMIT 1  -> MERGE (UNION ALL)

In the first form the whole compound is materialised and sorted -- every row
satisfying the predicate, in every file -- to return one. Measured 2026-07-26 on the
production estate (live + one 836 GB frozen segment): 0.001 s -> 7.560 s, ~224M VM
steps, for an identical answer: 7560x. Standing 5-second pollers written in the first
form saturated the disk continuously.

Selecting `ts_ms` would restore the merge, but that leaves correctness of the COST
resting on a planner heuristic that no test in the caller would notice regressing.
Prefer `as_of_row()` below: it asks each schema separately, so every branch is an
index seek by construction, independent of what the planner decides. Frozen and live are disjoint in time by construction
(frozen holds ts_ms < cutoff, live holds ts_ms >= cutoff), so `UNION ALL`
never double-counts.

Row identity: the per-file `id` PRIMARY KEY is blanked to `NULL` inside the
union view (frozen and live both number ids from 1, so a raw union would
collide them). The factory is therefore safe for **time-keyed** (`ts_ms`) reads
only; an id-keyed reader must not use it as-is -- an id filter over the view
matches nothing (fails loud) instead of silently mapping a row from the wrong
file. Schema drift (a frozen segment missing a column the live table has) is
refused at open with a `RotationStateError`, never a half-built view.

Read-only guarantee: every physical file is opened `?mode=ro`, so SQLite
itself refuses any write to a real DB file; `query_only=ON` is set as a second
lock after the temp views exist. TEMP views live only in this connection's
in-memory temp schema and vanish on close.

PRE-ROTATION (the state today): `rotation_state.json` is absent or lists zero
frozen segments -> `open_union_ro` is a pure pass-through that opens only the
live DB read-only. Callers then see the base tables directly, byte-identical to
opening `microstructure.db` themselves. This is what makes Phase 1 safe to land
before any rotation happens.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_PATH = REPO_ROOT / "data" / "microstructure.db"
DEFAULT_ROTATION_STATE_PATH = REPO_ROOT / "data" / "rotation_state.json"

# The six tables that live in microstructure.db (the four collector tables plus
# the two the oi_spot_poller depends on). Allowlist: view creation only ever
# touches these names.
UNION_TABLES = ("agg_trades", "book_ticker", "mark_prices", "liquidations",
                "open_interest", "spot_prices")


class RotationStateError(Exception):
    """Raised fail-closed on a malformed rotation_state.json or a missing
    frozen-segment file -- never guesses a safe default for a corrupt state."""


@dataclass(frozen=True)
class FrozenSegment:
    path: str
    start_ms: int | None
    end_ms: int | None


@dataclass(frozen=True)
class RotationState:
    live_db_path: str
    cutoff_ms: int | None
    frozen_segments: tuple[FrozenSegment, ...]

    @property
    def is_pre_rotation(self) -> bool:
        return not self.frozen_segments


def default_pre_rotation_state() -> RotationState:
    """The state before any rotation: live DB only, no frozen segments."""
    return RotationState(live_db_path=str(DEFAULT_LIVE_PATH), cutoff_ms=None, frozen_segments=())


def load_rotation_state(path: str | Path = DEFAULT_ROTATION_STATE_PATH) -> RotationState:
    """Reads `rotation_state.json`. Absent file -> pre-rotation pass-through
    default (so nothing has to exist for today's single-file world to work).
    A present-but-malformed file fails closed."""
    if not os.path.exists(path):
        return default_pre_rotation_state()
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RotationStateError(f"unreadable rotation_state at {path!r}: {exc}") from exc
    if not isinstance(doc, dict):
        raise RotationStateError(f"rotation_state at {path!r} is not a JSON object")

    live = doc.get("live_db_path") or str(DEFAULT_LIVE_PATH)
    cutoff = doc.get("cutoff_ms")
    if cutoff is not None and not isinstance(cutoff, int):
        raise RotationStateError(f"cutoff_ms must be int or null, got {cutoff!r}")

    raw_segments = doc.get("frozen_segments") or []
    if not isinstance(raw_segments, list):
        raise RotationStateError("frozen_segments must be a list")
    segments: list[FrozenSegment] = []
    for i, seg in enumerate(raw_segments):
        if not isinstance(seg, dict) or "path" not in seg:
            raise RotationStateError(f"frozen_segments[{i}] must be an object with a 'path'")
        segments.append(FrozenSegment(path=str(seg["path"]),
                                       start_ms=seg.get("start_ms"), end_ms=seg.get("end_ms")))
    return RotationState(live_db_path=str(live), cutoff_ms=cutoff, frozen_segments=tuple(segments))


def _ro_uri(path: str) -> str:
    # Forward slashes are the portable form for a file: URI, including on Windows.
    return "file:" + str(path).replace("\\", "/") + "?mode=ro"


def _table_columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    """Column names of `schema.table`, in declared order. Empty list if the
    table does not exist in that schema."""
    rows = conn.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
    return [r[1] for r in rows]


def open_union_ro(state: RotationState | None = None, *,
                  rotation_state_path: str | Path = DEFAULT_ROTATION_STATE_PATH,
                  timeout: float = 10.0) -> sqlite3.Connection:
    """Open a read-only connection that transparently unions the live DB with
    every frozen segment. Pre-rotation (no frozen segments) this is a plain
    read-only open of the live DB. The caller uses the returned connection
    exactly like a normal `sqlite3` connection and its existing SQL is
    unchanged."""
    if state is None:
        state = load_rotation_state(rotation_state_path)

    if not os.path.exists(state.live_db_path):
        raise RotationStateError(f"live_db_path does not exist: {state.live_db_path!r}")

    conn = sqlite3.connect(_ro_uri(state.live_db_path), uri=True, timeout=timeout)

    if state.is_pre_rotation:
        conn.execute("PRAGMA query_only=ON")
        return conn

    # Multi-file: attach frozen segments, then shadow each base table with a
    # union view. Views are created BEFORE query_only=ON (temp-schema DDL).
    #
    # Everything below runs under a close-on-failure guard. The explicit raises used
    # to close the handle themselves, but an error from the ATTACH or the view DDL --
    # a truncated or mid-replace segment fails there as a plain sqlite3 error --
    # propagated with the connection still open. That was harmless only while such a
    # failure killed the process; callers now survive it in a 30s loop, which would
    # otherwise leak a handle per cycle.
    try:
        return _build_union(conn, state)
    except BaseException:
        conn.close()
        raise


def _build_union(conn: sqlite3.Connection, state: RotationState) -> sqlite3.Connection:
    for i, seg in enumerate(state.frozen_segments):
        if not os.path.exists(seg.path):
            raise RotationStateError(f"frozen segment file does not exist: {seg.path!r}")
        conn.execute(f"ATTACH DATABASE ? AS frozen{i}", (_ro_uri(seg.path),))

    n = len(state.frozen_segments)
    for table in UNION_TABLES:
        main_cols = _table_columns(conn, "main", table)
        if not main_cols:
            continue  # table not present in the live DB; nothing to shadow
        # `id` is a per-file AUTOINCREMENT PRIMARY KEY: frozen and live both start
        # it at 1, so a raw UNION ALL yields COLLIDING ids across files. Blank it to
        # NULL in the union view so any id-keyed query (WHERE id IN (...), MIN/MAX(id),
        # id->row maps) fails LOUD (NULL/empty) rather than silently mapping the wrong
        # row from the other file. Time-keyed (ts_ms) readers are unaffected. The
        # column is kept (as NULL) so `SELECT id` still resolves instead of erroring.
        select_list = ", ".join("NULL AS id" if c == "id" else c for c in main_cols)
        parts = [f"SELECT {select_list} FROM main.{table}"]
        for i in range(n):
            frozen_cols = _table_columns(conn, f"frozen{i}", table)
            if not frozen_cols:
                continue  # table absent in this frozen segment; nothing to union
            missing = [c for c in main_cols if c not in frozen_cols]
            if missing:
                raise RotationStateError(
                    f"schema drift: frozen segment {i} table {table!r} is missing "
                    f"column(s) {missing} present in the live DB; refusing to build "
                    f"an inconsistent union view")
            parts.append(f"SELECT {select_list} FROM frozen{i}.{table}")
        conn.execute(f"CREATE TEMP VIEW {table} AS " + " UNION ALL ".join(parts))

    conn.execute("PRAGMA query_only=ON")
    return conn


class InsufficientHistoryError(Exception):
    """Raised when a caller needs more history than the estate physically holds.

    This is the failure mode that replaces the frozen segment once it is deleted.
    A query reaching before the earliest stored row does NOT error in SQLite -- it
    quietly returns whatever exists, so a 30-day percentile silently becomes a
    12-day percentile and every downstream number looks normal. Truncation has to
    be turned into an exception by something, because SQL will not do it."""


def history_floor_ms(conn: sqlite3.Connection, table: str) -> int | None:
    """Earliest `ts_ms` held for `table`, across every attached schema.

    Asks each schema separately (`main`, `frozen0`, ...) rather than going through
    the union view: schema-qualified `MIN(ts_ms)` is an index seek per branch, while
    the same query against the compound view materialises it. Same reasoning as the
    as-of caveat in this module's docstring.

    Scoped to `union_schemas()` -- the schemas the union view actually spans -- NOT
    every schema on the connection. Those differ the moment anything else is
    ATTACHed (which SQLite still permits under `query_only=ON`), and then the floor
    would be read off a database the caller's `FROM <table>` cannot see: the guard
    would report deep history while the query returned shallow, which is the exact
    inversion of what it exists to prevent.

    Returns None when the table holds no rows anywhere, which callers must treat as
    "no history at all", not as "no limit".
    """
    if not _IDENT_RE.fullmatch(table):
        raise ValueError(f"unsafe table identifier: {table!r}")
    floors: list[int] = []
    for schema in union_schemas(conn):
        if not _table_columns(conn, schema, table):
            continue
        row = conn.execute(f"SELECT MIN(ts_ms) FROM {schema}.{table}").fetchone()
        if row and row[0] is not None:
            floors.append(int(row[0]))
    return min(floors) if floors else None


def assert_history_covers(conn: sqlite3.Connection, table: str, earliest_ms: int,
                          *, what: str = "") -> int:
    """Fail loud unless `table` actually reaches back to `earliest_ms`.

    Deliberately explicit rather than automatic. Sniffing bound parameters for
    things that look like epoch-ms would be guessing: it cannot distinguish a lower
    bound from an upper bound, and a caller may legitimately pass a timestamp older
    than the data (e.g. a forward-only cutoff constant used as a filter). A guard
    that fires on healthy roles is worse than no guard, so coverage is asserted
    where the required depth is actually known -- next to the lookback constant.

    SCOPE -- LEFT EDGE ONLY. This proves a row exists at or before `earliest_ms`. It
    says nothing about holes INSIDE the window (this estate has documented ones: a
    39-day `liquidations` outage in 2026-04/06 and a ~5.3-day `book_ticker` gap) or
    about the data being current. A window landing inside an outage passes this check
    and is still computed on a fraction of the rows. Gap and freshness coverage are
    separate guards and are not built.

    Returns the measured floor so callers can report it.
    """
    if not isinstance(earliest_ms, int) or isinstance(earliest_ms, bool):
        raise InsufficientHistoryError(
            f"{what or table}: earliest_ms must be an int epoch-ms, got {earliest_ms!r}")
    known = set(UNION_TABLES)
    if table not in known:
        # a typo would otherwise surface as "holds no rows", sending the operator to
        # hunt for missing data that was never missing
        raise InsufficientHistoryError(
            f"{what or table}: {table!r} is not a union table {sorted(known)}")

    floor = history_floor_ms(conn, table)
    label = what or table
    if floor is None:
        raise InsufficientHistoryError(
            f"{label}: {table!r} holds no rows in any attached segment")
    if floor > earliest_ms:
        short = floor - earliest_ms
        amount = (f"{short / 86_400_000.0:.2f} days" if short >= 86_400_000
                  else f"{short / 3_600_000.0:.2f} hours" if short >= 3_600_000
                  else f"{short:,} ms")
        raise InsufficientHistoryError(
            f"{label}: needs {table!r} back to {earliest_ms} but the estate starts "
            f"at {floor} -- {amount} short; the answer would be silently computed "
            f"on a shorter window than it claims")
    return floor


def current_live_db_path(state: RotationState | None = None, *,
                         rotation_state_path: str | Path = DEFAULT_ROTATION_STATE_PATH) -> str:
    """Path of the CURRENTLY-LIVE db file. Pre-rotation this is
    `data/microstructure.db`; AFTER a rotation it is the fresh file
    (e.g. `microstructure_02.db`), because the original is frozen. Freshness /
    "is the collector writing now?" probes MUST resolve their path through this,
    never a hard-coded `data/microstructure.db`, or post-rotation they would open
    the frozen file and report perpetually-stale (rotation plan catch b)."""
    if state is None:
        state = load_rotation_state(rotation_state_path)
    return state.live_db_path


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def union_schemas(conn: sqlite3.Connection) -> list[str]:
    """The physical schemas this connection spans, in union order: `['main']`
    pre-rotation, `['main', 'frozen0', ...]` after. Read off `PRAGMA
    database_list` so it reflects the CONNECTION rather than a rotation-state
    file that may have been re-read since the connection was opened."""
    names = [r[1] for r in conn.execute("PRAGMA database_list").fetchall()]
    return [n for n in names if n == "main" or n.startswith("frozen")]


def as_of_row(conn: sqlite3.Connection, table: str, select: str, *,
              where: str = "", params: tuple = (), ts_col: str = "ts_ms"):
    """The newest row at-or-before some point in time, WITHOUT the union view's
    sort blow-up. Returns `(<ts_col>, <select ...>)` -- the timestamp is always
    the first element -- or `None` when no schema has a matching row.

    Issues `SELECT <ts_col>, <select> FROM <schema>.<table> WHERE <where>
    ORDER BY <ts_col> DESC LIMIT 1` against EACH schema separately and keeps the
    row with the greatest `ts_col`. Every branch is an index seek BY CONSTRUCTION
    -- there is no compound for the planner to mis-handle -- so the cost is
    O(number of files), not O(history), and it cannot silently regress the way the
    view form does when a caller drops `ts_ms` from its select list (see the module
    docstring). The answer is identical to the same query run over the union view:
    `UNION ALL` of disjoint-in-time files means the global newest row is
    necessarily the newest of the per-file newest rows.

    `where` is the caller's own predicate WITHOUT the `ORDER BY`/`LIMIT` (e.g.
    `"symbol=? AND ts_ms<=?"`); `params` binds it and is re-bound per schema.
    `table` must be one of `UNION_TABLES` and `ts_col` a bare identifier: both are
    interpolated into SQL, so both are validated rather than trusted.

    Pre-rotation (a single `main` schema) this is exactly the caller's original
    query, one seek, no behaviour change."""
    if table not in UNION_TABLES:
        raise ValueError(f"table {table!r} is not one of UNION_TABLES {UNION_TABLES}")
    if not _IDENT_RE.match(ts_col):
        raise ValueError(f"ts_col {ts_col!r} is not a bare identifier")
    clause = f" WHERE {where}" if where else ""
    best = None
    for schema in union_schemas(conn):
        row = conn.execute(
            f"SELECT {ts_col}, {select} FROM {schema}.{table}{clause} "
            f"ORDER BY {ts_col} DESC LIMIT 1", params).fetchone()
        if row is not None and row[0] is not None and (best is None or row[0] > best[0]):
            best = row
    return best


def open_live_ro(state: RotationState | None = None, *,
                 rotation_state_path: str | Path = DEFAULT_ROTATION_STATE_PATH,
                 timeout: float = 10.0) -> sqlite3.Connection:
    """Read-only connection to the CURRENT LIVE db file ONLY (no union, no frozen
    segments). For freshness / latest probes that rely on base-table fast-paths --
    `ORDER BY rowid DESC LIMIT 1`, index-backed `MAX(ts_ms)` -- which a UNION view
    would defeat or (for `rowid`) not expose. History-spanning readers use
    `open_union_ro` instead. Read-only via `?mode=ro` + `query_only=ON`."""
    if state is None:
        state = load_rotation_state(rotation_state_path)
    live = state.live_db_path
    if not os.path.exists(live):
        raise RotationStateError(f"live_db_path does not exist: {live!r}")
    conn = sqlite3.connect(_ro_uri(live), uri=True, timeout=timeout)
    conn.execute("PRAGMA query_only=ON")
    return conn
