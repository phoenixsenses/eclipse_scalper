"""Frozen-DB -> Parquet archival exporter (DB rotation Phase-3).

SYSTEM_STATE §225 fixed the sequence for the frozen segment:

    1. write Parquet  (additive; nothing is ever deleted here)
    2. verify
    3. PROOF GATE: the same data read through both paths must be identical
    4. delete            <- NOT this tool, and NOT authorized

This module implements steps 1-3 only. It is strictly read-only on the source:
the frozen DB is opened with ``mode=ro`` and the live DB named in
``data/rotation_state.json`` is refused outright.

Integrity model
---------------
"Verified" here does not mean "row counts look similar". Every batch of rows is
packed into a canonical byte encoding (declared-type driven, so SQLite's dynamic
typing cannot smuggle a float into an INTEGER column unnoticed) and hashed. Batch
digests are chained, so the partition digest covers *values and order*. One batch
is written as exactly one Parquet row group, which lets ``--verify`` re-read the
Parquet file row group by row group and reproduce the identical chain.

The gate therefore closes in both directions:

* Parquet re-read digest  == manifest digest   (the archive is intact)
* SQLite re-query digest  == manifest digest   (the archive matches the source)

Layout
------
    <out-root>/<table>/symbol=<SYMBOL>/dt=<YYYY-MM-DD>/part-0000.parquet

Hive-style partitioning so a future DuckDB reader gets predicate pushdown on the
two columns every consumer filters on (symbol, ts_ms).

Resumability
------------
Frozen-DB scans in this repo have been killed mid-run before, so progress is
appended to a JSONL manifest and fsynced per partition. Re-running skips every
partition already recorded. Foreground runs only.

Usage
-----
    python -m tools.frozen_db_parquet_export --table mark_prices --plan
    python -m tools.frozen_db_parquet_export --table mark_prices
    python -m tools.frozen_db_parquet_export --table mark_prices --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FROZEN_DB = REPO_ROOT / "data" / "microstructure.db"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "archives" / "parquet_v1"
ROTATION_STATE = REPO_ROOT / "data" / "rotation_state.json"

DEFAULT_BATCH_ROWS = 200_000
# v2 changed the canonical digest from row-major Python packing to column-major
# Arrow buffers. v1 records are not comparable and must be re-exported.
MANIFEST_VERSION = 2

# Canonical type tags. Never renumber: manifest digests depend on them.
_TAG_INT = b"\x01"
_TAG_FLOAT = b"\x02"
_TAG_TEXT = b"\x03"
_TAG_BLOB = b"\x04"


class ExportError(RuntimeError):
    """Raised for any condition that must stop the export rather than degrade it."""


# --------------------------------------------------------------------------
# source safety
# --------------------------------------------------------------------------


def live_db_path() -> Path | None:
    """Path of the currently live DB segment, or None if rotation state is absent."""
    try:
        state = json.loads(ROTATION_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    raw = state.get("live_db_path")
    if not raw:
        return None
    return Path(raw)


def assert_source_is_not_live(db_path: Path) -> None:
    live = live_db_path()
    if live is None:
        return
    try:
        same = db_path.resolve() == live.resolve()
    except OSError:
        same = str(db_path).lower() == str(live).lower()
    if same:
        raise ExportError(
            f"refusing to export the LIVE db ({db_path}); this tool archives frozen segments only"
        )


def open_source_ro(db_path: Path) -> sqlite3.Connection:
    """Open the frozen segment read-only. Any write attempt must fail at the driver."""
    if not db_path.exists():
        raise ExportError(f"source db not found: {db_path}")
    assert_source_is_not_live(db_path)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=60.0)
    con.execute("PRAGMA query_only = 1")
    return con


# --------------------------------------------------------------------------
# schema + canonical packing
# --------------------------------------------------------------------------


def declared_types(con: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    """[(column_name, normalized_declared_type)] in table order."""
    rows = list(con.execute(f"PRAGMA table_info({table})"))
    if not rows:
        raise ExportError(f"table not found in source: {table}")
    out: list[tuple[str, str]] = []
    for row in rows:
        name = str(row[1])
        decl = str(row[2] or "").upper()
        if "INT" in decl:
            kind = "INTEGER"
        elif "REAL" in decl or "FLOA" in decl or "DOUB" in decl:
            kind = "REAL"
        elif "CHAR" in decl or "TEXT" in decl or "CLOB" in decl:
            kind = "TEXT"
        elif "BLOB" in decl or decl == "":
            kind = "BLOB"
        else:
            kind = "TEXT"
        out.append((name, kind))
    return out


def arrow_schema(cols: Sequence[tuple[str, str]]) -> pa.Schema:
    mapping = {
        "INTEGER": pa.int64(),
        "REAL": pa.float64(),
        "TEXT": pa.string(),
        "BLOB": pa.binary(),
    }
    return pa.schema([pa.field(name, mapping[kind]) for name, kind in cols])


_KIND_TAG = {"INTEGER": _TAG_INT, "REAL": _TAG_FLOAT, "TEXT": _TAG_TEXT, "BLOB": _TAG_BLOB}


def rows_to_arrow(
    rows: Sequence[Sequence[Any]], cols: Sequence[tuple[str, str]], schema: pa.Schema
) -> pa.Table:
    """Build an Arrow table from SQLite rows, letting Arrow do the type checking.

    SQLite is dynamically typed, so a column declared INTEGER can physically hold a
    float. Arrow's safe cast refuses the lossy direction (1.5 -> int64) while
    accepting the lossless one (3000 -> 3000.0 in a REAL column), which is exactly
    the normalization both sides of the proof gate need. Doing it here rather than
    per value in Python keeps it at C speed.
    """
    if not rows:
        return schema.empty_table()
    columns = list(zip(*rows))
    arrays: list[pa.Array] = []
    for idx, (name, _kind) in enumerate(cols):
        target = schema.field(idx).type
        try:
            # Infer first, then cast with safe=True. Constructing directly at the
            # target type lets Arrow TRUNCATE a stray 1.5 in an INTEGER column to 1
            # -- and because export and verify would both truncate identically, the
            # proof gate would pass on data that had silently changed.
            arrays.append(pa.array(columns[idx]).cast(target, safe=True))
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
            raise ExportError(f"column {name} does not fit its declared type: {exc}") from exc
    return pa.Table.from_arrays(arrays, schema=schema)


def _as_array(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(arr, pa.ChunkedArray):
        if arr.num_chunks == 1:
            return arr.chunk(0)
        if arr.num_chunks == 0:
            return pa.array([], type=arr.type)
        return pa.concat_arrays(list(arr.chunks))
    return arr


def digest_batch(table: pa.Table, cols: Sequence[tuple[str, str]]) -> bytes:
    """Canonical digest of one batch, computed from Arrow buffers.

    Column-major rather than row-major, because that is the layout Arrow already
    holds in contiguous memory -- hashing it is a memcpy plus a SHA pass instead of
    a Python loop over every value. The encoding is still fully determined by the
    values: type tag, row count, an explicit null mask, then the values with nulls
    filled to a fixed sentinel so that Arrow's undefined null slots can never leak
    into the hash. Identical values therefore always hash identically, whether the
    batch came from SQLite rows or from a Parquet row group.
    """
    hasher = hashlib.sha256()
    for index, (name, kind) in enumerate(cols):
        arr = _as_array(table.column(index))
        hasher.update(_KIND_TAG[kind])
        hasher.update(struct.pack("<Q", len(arr)))
        if arr.null_count == 0:
            hasher.update(b"\x00")
        else:
            hasher.update(b"\x01")
            hasher.update(arr.is_valid().to_numpy(zero_copy_only=False).tobytes())
        if kind == "INTEGER":
            filled = arr.fill_null(0).to_numpy(zero_copy_only=False)
            hasher.update(filled.astype("<i8", copy=False).tobytes())
        elif kind == "REAL":
            filled = arr.fill_null(0.0).to_numpy(zero_copy_only=False)
            hasher.update(filled.astype("<f8", copy=False).tobytes())
        elif kind == "TEXT":
            encoded = arr.fill_null("").dictionary_encode()
            # dictionary ids follow first-appearance order, so this is deterministic
            for value in encoded.dictionary.to_pylist():
                raw = value.encode("utf-8")
                hasher.update(struct.pack("<I", len(raw)) + raw)
            hasher.update(b"\xff")
            indices = _as_array(encoded.indices).fill_null(0).to_numpy(zero_copy_only=False)
            hasher.update(indices.astype("<i4", copy=False).tobytes())
        else:
            for value in arr.to_pylist():
                raw = b"" if value is None else bytes(value)
                hasher.update(struct.pack("<I", len(raw)) + raw)
    return hasher.digest()


def chain_digest(previous_hex: str, batch_digest: bytes) -> str:
    """Fold one batch into the running partition digest (covers values AND order)."""
    return hashlib.sha256(bytes.fromhex(previous_hex) + batch_digest).hexdigest()


EMPTY_DIGEST = hashlib.sha256(b"frozen_db_parquet_export/v2").hexdigest()


# --------------------------------------------------------------------------
# partition planning
# --------------------------------------------------------------------------

_MS_PER_DAY = 86_400_000


def day_floor_ms(ts_ms: int) -> int:
    return (ts_ms // _MS_PER_DAY) * _MS_PER_DAY


def day_label(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def discover_symbols(con: sqlite3.Connection, table: str) -> list[str]:
    """Distinct symbols via a loose index scan: one seek per symbol, not a table scan.

    ``SELECT DISTINCT symbol`` would walk all 5.7B entries of book_ticker's
    (symbol, ts_ms) index. Repeatedly asking for the smallest symbol greater than
    the last one hits SQLite's single-aggregate min/max index shortcut instead, so
    the whole enumeration costs (distinct symbols + 1) seeks. That makes it cheap
    enough to always run -- which matters, because a symbol nobody remembered would
    otherwise be skipped in silence and only surface as a row-count shortfall after
    a multi-hour export.
    """
    symbols: list[str] = []
    row = con.execute(f"SELECT MIN(symbol) FROM {table}").fetchone()
    current = row[0] if row else None
    while current is not None:
        symbols.append(str(current))
        row = con.execute(
            f"SELECT MIN(symbol) FROM {table} WHERE symbol > ?", (current,)
        ).fetchone()
        current = row[0] if row else None
    return symbols


def symbol_bounds(con: sqlite3.Connection, table: str, symbol: str) -> tuple[int, int] | None:
    """(min_ts_ms, max_ts_ms) for a symbol, as two index seeks.

    Deliberately NOT ``SELECT MIN(ts_ms), MAX(ts_ms) ... WHERE symbol = ?``: SQLite
    applies its min/max index shortcut to a single aggregate only, so asking for both
    at once degrades to a full scan of the (symbol, ts_ms) index. Measured on
    agg_trades that is 1.76s vs 0.00s per symbol warm, and far worse cold -- and this
    cost is paid again on every resume, so on book_ticker (5.7B rows) it would dwarf
    the export itself.
    """
    lo = con.execute(
        f"SELECT ts_ms FROM {table} WHERE symbol = ? ORDER BY ts_ms ASC LIMIT 1", (symbol,)
    ).fetchone()
    if lo is None or lo[0] is None:
        return None
    hi = con.execute(
        f"SELECT ts_ms FROM {table} WHERE symbol = ? ORDER BY ts_ms DESC LIMIT 1", (symbol,)
    ).fetchone()
    return int(lo[0]), int(hi[0])


def plan_partitions(
    con: sqlite3.Connection, table: str, symbols: Sequence[str]
) -> list[dict[str, Any]]:
    """One partition per (symbol, UTC day) across each symbol's observed range."""
    plan: list[dict[str, Any]] = []
    for symbol in symbols:
        bounds = symbol_bounds(con, table, symbol)
        if bounds is None:
            continue
        lo, hi = bounds
        start = day_floor_ms(lo)
        end = day_floor_ms(hi)
        cursor = start
        while cursor <= end:
            plan.append(
                {
                    "symbol": symbol,
                    "dt": day_label(cursor),
                    "start_ms": cursor,
                    "end_ms": cursor + _MS_PER_DAY,
                }
            )
            cursor += _MS_PER_DAY
    return plan


def partition_key(table: str, symbol: str, dt: str) -> str:
    return f"{table}|{symbol}|{dt}"


def partition_path(out_root: Path, table: str, symbol: str, dt: str) -> Path:
    # The hive key is `sym`, not `symbol`, on purpose: `symbol` is also a real column
    # in every exported table, and a dataset reader that infers hive partitioning
    # would then see the same name twice (partition dictionary vs file column) and
    # refuse to merge the schema. Renaming the key keeps the full row intact in the
    # file while still giving DuckDB a pushdown predicate.
    return out_root / table / f"sym={symbol}" / f"dt={dt}" / "part-0000.parquet"


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def manifest_path(out_root: Path, table: str) -> Path:
    return out_root / table / "_manifest.jsonl"


def lock_path(out_root: Path, table: str) -> Path:
    return out_root / table / "_export.lock"


def verified_path(out_root: Path, table: str) -> Path:
    return out_root / table / "_verified.jsonl"


def verification_carries(record: dict[str, Any], checkpoint: dict[str, Any], want_source: bool) -> bool:
    """Whether a past verification of this partition can stand in for redoing it.

    A full pass over book_ticker takes hours and has already died once at 246/305,
    so the result of each partition is checkpointed. Carrying a result forward is
    only sound while nothing it depended on has moved, hence all four conditions:
    the digest it agreed with, the file it read (size and mtime), and the mode --
    a parquet-only pass must never satisfy a request that includes the source.
    """
    if checkpoint.get("digest") != record.get("digest"):
        return False
    if want_source and checkpoint.get("mode") != "parquet+sqlite":
        return False
    if checkpoint.get("file_bytes") != record.get("file_bytes"):
        return False
    if checkpoint.get("file_mtime_ns") != record.get("_file_mtime_ns"):
        return False
    return True


def parquet_stat(out_root: Path, record: dict[str, Any]) -> int | None:
    if not record.get("path"):
        return None
    target = out_root / record["path"]
    try:
        return target.stat().st_mtime_ns
    except OSError:
        return None


class ExportLock:
    """Single-writer lock over one table's archive directory.

    Two exporters appending to the same manifest and racing on the same partition
    tmp file would interleave silently -- this repo has already been bitten once by
    a second background writer nobody knew was alive. Long runs get interrupted and
    resumed by hand here, so an operator relaunching while a run is still going is a
    realistic accident, not a theoretical one.

    A stale lock (holder no longer running) is reclaimed rather than requiring
    manual cleanup, because the common case is exactly a killed run.
    """

    def __init__(self, out_root: Path, table: str) -> None:
        self.path = lock_path(out_root, table)
        self.acquired = False

    def _holder_alive(self, pid: int) -> bool:
        # No self-exclusion: a live pid is a live pid. Treating "same pid as me" as
        # dead would let a process reclaim a lock it is already holding, which is the
        # one case where reclaiming is definitely wrong.
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        except Exception:
            return True
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(held.get("pid", -1))
            except (OSError, ValueError, TypeError):
                pid = -1
            if pid > 0 and self._holder_alive(pid):
                raise ExportError(
                    f"another export is already running for this table (pid {pid}); "
                    f"stop it first or delete {self.path}"
                )
            self.path.unlink(missing_ok=True)
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid()}, fh)
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "ExportLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            key = record.get("key")
            if key:
                entries[key] = record
    return entries


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def iter_batches(
    con: sqlite3.Connection,
    table: str,
    cols: Sequence[tuple[str, str]],
    symbol: str,
    start_ms: int,
    end_ms: int,
    batch_rows: int,
) -> Iterator[list[tuple[Any, ...]]]:
    col_sql = ", ".join(name for name, _ in cols)
    sql = (
        f"SELECT {col_sql} FROM {table} "
        "WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ?"
    )
    cur = con.execute(sql, (symbol, start_ms, end_ms))
    try:
        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                return
            yield rows
    finally:
        cur.close()


def export_partition(
    con: sqlite3.Connection,
    table: str,
    cols: Sequence[tuple[str, str]],
    schema: pa.Schema,
    part: dict[str, Any],
    out_root: Path,
    batch_rows: int,
    compression: str,
) -> dict[str, Any]:
    """Write one (symbol, day) partition and return its manifest record."""
    symbol = part["symbol"]
    dt = part["dt"]
    target = partition_path(out_root, table, symbol, dt)
    tmp = target.with_suffix(".parquet.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    started = time.time()
    digest = EMPTY_DIGEST
    rows_total = 0
    batches = 0
    writer: pq.ParquetWriter | None = None

    try:
        for rows in iter_batches(
            con, table, cols, symbol, part["start_ms"], part["end_ms"], batch_rows
        ):
            arrow_table = rows_to_arrow(rows, cols, schema)
            digest = chain_digest(digest, digest_batch(arrow_table, cols))
            if writer is None:
                writer = pq.ParquetWriter(tmp, schema, compression=compression)
            # one batch == one row group, so --verify can reproduce the chain exactly
            writer.write_table(arrow_table, row_group_size=len(rows))
            rows_total += len(rows)
            batches += 1
    finally:
        if writer is not None:
            writer.close()

    if rows_total == 0:
        # Empty days are real (feed outages). Record them so coverage is provable.
        if tmp.exists():
            tmp.unlink()
        if target.exists():
            target.unlink()
        file_bytes = 0
    else:
        os.replace(tmp, target)
        file_bytes = target.stat().st_size

    return {
        "manifest_version": MANIFEST_VERSION,
        "key": partition_key(table, symbol, dt),
        "table": table,
        "symbol": symbol,
        "dt": dt,
        "start_ms": part["start_ms"],
        "end_ms": part["end_ms"],
        "rows": rows_total,
        "batches": batches,
        "batch_rows": batch_rows,
        "digest": digest,
        "path": str(target.relative_to(out_root)) if rows_total else None,
        "file_bytes": file_bytes,
        "compression": compression,
        "elapsed_sec": round(time.time() - started, 3),
        "columns": [{"name": name, "type": kind} for name, kind in cols],
    }


def run_status(db_path: Path, table: str, out_root: Path, expect_rows: int | None) -> int:
    """Report progress from the manifest alone -- no DB scan, safe to poll.

    A multi-hour export is meant to run detached from any session, so checking on it
    must not cost anything or disturb it.
    """
    done = {
        key: rec
        for key, rec in read_manifest(manifest_path(out_root, table)).items()
        if rec.get("manifest_version") == MANIFEST_VERSION
    }
    rows = sum(rec["rows"] for rec in done.values())
    file_bytes = sum(rec["file_bytes"] for rec in done.values())
    elapsed = sum(rec.get("elapsed_sec", 0.0) for rec in done.values())

    lock = lock_path(out_root, table)
    holder = "none"
    if lock.exists():
        try:
            holder = f"pid {json.loads(lock.read_text(encoding='utf-8')).get('pid')}"
        except (OSError, ValueError):
            holder = "unreadable"

    total = None
    try:
        con = open_source_ro(db_path)
        try:
            total = len(plan_partitions(con, table, discover_symbols(con, table)))
        finally:
            con.close()
    except ExportError:
        pass

    print(f"table={table}")
    print(f"partitions_done={len(done)}" + (f"/{total}" if total else ""))
    print(f"rows={rows:,}  parquet_bytes={file_bytes:,} ({file_bytes / 1e9:.2f} GB)")
    if elapsed > 0:
        print(f"export_time={elapsed / 3600:.2f}h  rate={rows / elapsed:,.0f} rows/s")
    if total and len(done) and len(done) < total and elapsed > 0:
        remaining = (total - len(done)) / len(done) * elapsed
        print(f"eta_remaining={remaining / 3600:.1f}h (at the observed rate)")
    if expect_rows:
        pct = 100.0 * rows / expect_rows if expect_rows else 0.0
        print(f"coverage={pct:.2f}% of {expect_rows:,} source rows")
    print(f"running={holder}")
    return 0


def run_export(
    db_path: Path,
    table: str,
    out_root: Path,
    symbols: Sequence[str] | None,
    batch_rows: int,
    compression: str,
    max_partitions: int | None,
    plan_only: bool,
) -> int:
    con = open_source_ro(db_path)
    try:
        cols = declared_types(con, table)
        schema = arrow_schema(cols)
        # Always enumerate the truth, even when the caller named symbols: an omitted
        # symbol must be an explicit, printed decision rather than a silent gap.
        present = discover_symbols(con, table)
        if not present:
            raise ExportError(f"no symbols found for {table}")
        if symbols:
            unknown = sorted(set(symbols) - set(present))
            if unknown:
                raise ExportError(f"--symbols names symbols absent from {table}: {unknown}")
            excluded = sorted(set(present) - set(symbols))
            syms = [s for s in present if s in set(symbols)]
        else:
            excluded = []
            syms = present
        plan = plan_partitions(con, table, syms)

        mpath = manifest_path(out_root, table)
        done = {
            key: rec
            for key, rec in read_manifest(mpath).items()
            if rec.get("manifest_version") == MANIFEST_VERSION
        }
        pending = [p for p in plan if partition_key(table, p["symbol"], p["dt"]) not in done]

        print(f"table={table} source={db_path}")
        print(f"symbols_in_table={','.join(present)}")
        print(f"symbols_exported={','.join(syms)}")
        if excluded:
            print(f"SYMBOLS_EXCLUDED={','.join(excluded)} <- these rows will NOT be archived")
        print(f"partitions_total={len(plan)} done={len(plan) - len(pending)} pending={len(pending)}")
        if plan_only:
            for part in pending[:20]:
                print(f"  PENDING {part['symbol']} {part['dt']}")
            if len(pending) > 20:
                print(f"  ... {len(pending) - 20} more")
            return 0

        if max_partitions is not None:
            pending = pending[:max_partitions]
            print(f"limited to {len(pending)} partitions this run")

        rows_written = 0
        bytes_written = 0
        wall = time.time()
        with ExportLock(out_root, table):
            for index, part in enumerate(pending, start=1):
                record = export_partition(
                    con, table, cols, schema, part, out_root, batch_rows, compression
                )
                append_manifest(mpath, record)
                rows_written += record["rows"]
                bytes_written += record["file_bytes"]
                rate = record["rows"] / record["elapsed_sec"] if record["elapsed_sec"] > 0 else 0.0
                print(
                    f"[{index}/{len(pending)}] {record['symbol']} {record['dt']} "
                    f"rows={record['rows']:,} bytes={record['file_bytes']:,} "
                    f"{record['elapsed_sec']:.1f}s ({rate:,.0f} rows/s)"
                )

        elapsed = time.time() - wall
        print(
            f"EXPORT_DONE partitions={len(pending)} rows={rows_written:,} "
            f"bytes={bytes_written:,} elapsed={elapsed:.1f}s"
        )
        return 0
    finally:
        con.close()


# --------------------------------------------------------------------------
# proof gate
# --------------------------------------------------------------------------


def digest_parquet(path: Path, cols: Sequence[tuple[str, str]]) -> tuple[str, int]:
    """Recompute the chained digest from the Parquet file, row group by row group."""
    pfile = pq.ParquetFile(path)
    names = [name for name, _ in cols]
    digest = EMPTY_DIGEST
    rows_total = 0
    for group in range(pfile.num_row_groups):
        table = pfile.read_row_group(group, columns=names)
        if table.num_rows == 0:
            continue
        digest = chain_digest(digest, digest_batch(table, cols))
        rows_total += table.num_rows
    return digest, rows_total


def digest_sqlite(
    con: sqlite3.Connection,
    table: str,
    cols: Sequence[tuple[str, str]],
    symbol: str,
    start_ms: int,
    end_ms: int,
    batch_rows: int,
) -> tuple[str, int]:
    schema = arrow_schema(cols)
    digest = EMPTY_DIGEST
    rows_total = 0
    for rows in iter_batches(con, table, cols, symbol, start_ms, end_ms, batch_rows):
        digest = chain_digest(digest, digest_batch(rows_to_arrow(rows, cols, schema), cols))
        rows_total += len(rows)
    return digest, rows_total


def run_verify(
    db_path: Path,
    table: str,
    out_root: Path,
    max_partitions: int | None,
    skip_source: bool,
    expect_rows: int | None = None,
    reverify: bool = False,
) -> int:
    """Close the §225 proof gate: Parquet == manifest == live re-query of SQLite."""
    mpath = manifest_path(out_root, table)
    done = read_manifest(mpath)
    if not done:
        print(f"VERIFY_FAIL no manifest entries for {table} at {mpath}")
        return 2

    stale = sorted(
        {
            rec.get("manifest_version")
            for rec in done.values()
            if rec.get("manifest_version") != MANIFEST_VERSION
        },
        key=lambda v: (v is None, v),
    )
    if stale:
        # A v1 digest cannot be compared against a v2 one; failing loudly beats
        # reporting a mismatch that looks like data corruption.
        print(
            f"VERIFY_FAIL manifest holds version(s) {stale} but this tool writes "
            f"v{MANIFEST_VERSION}; re-export those partitions"
        )
        return 2

    records = sorted(done.values(), key=lambda r: (r["symbol"], r["dt"]))
    if max_partitions is not None:
        records = records[:max_partitions]

    vpath = verified_path(out_root, table)
    checkpoint = {} if reverify else read_manifest(vpath)

    con = open_source_ro(db_path)
    try:
        cols = declared_types(con, table)
        failures: list[str] = []
        rows_checked = 0
        checked_now = 0
        carried = 0

        for index, record in enumerate(records, start=1):
            label = f"{record['symbol']} {record['dt']}"
            expected = record["digest"]
            batch_rows = int(record.get("batch_rows", DEFAULT_BATCH_ROWS))
            record["_file_mtime_ns"] = parquet_stat(out_root, record)

            prior = checkpoint.get(record["key"])
            if prior and verification_carries(record, prior, not skip_source):
                rows_checked += record["rows"]
                carried += 1
                continue

            if record["rows"] == 0:
                if record.get("path"):
                    failures.append(f"{label}: manifest says 0 rows but names a file")
                parquet_digest, parquet_rows = EMPTY_DIGEST, 0
            else:
                ppath = out_root / record["path"]
                if not ppath.exists():
                    failures.append(f"{label}: parquet missing at {ppath}")
                    continue
                parquet_digest, parquet_rows = digest_parquet(ppath, cols)

            if parquet_rows != record["rows"]:
                failures.append(
                    f"{label}: parquet rows {parquet_rows} != manifest {record['rows']}"
                )
            if parquet_digest != expected:
                failures.append(f"{label}: parquet digest mismatch")

            if not skip_source:
                src_digest, src_rows = digest_sqlite(
                    con,
                    table,
                    cols,
                    record["symbol"],
                    int(record["start_ms"]),
                    int(record["end_ms"]),
                    batch_rows,
                )
                if src_rows != record["rows"]:
                    failures.append(
                        f"{label}: sqlite rows {src_rows} != manifest {record['rows']}"
                    )
                if src_digest != expected:
                    failures.append(f"{label}: sqlite digest mismatch")

            rows_checked += record["rows"]
            checked_now += 1
            passed = not failures or not failures[-1].startswith(label)
            if passed:
                # Checkpoint only what actually passed, so an interrupted run never
                # carries forward a partition it had not finished agreeing with.
                append_manifest(
                    vpath,
                    {
                        "key": record["key"],
                        "digest": expected,
                        "rows": record["rows"],
                        "file_bytes": record["file_bytes"],
                        "file_mtime_ns": record["_file_mtime_ns"],
                        "mode": "parquet-only" if skip_source else "parquet+sqlite",
                        "verified_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            print(
                f"[{index}/{len(records)}] {label} rows={record['rows']:,} "
                f"{'OK' if passed else 'FAIL'}"
            )

        if failures:
            print(f"VERIFY_FAIL partitions={len(records)} failures={len(failures)}")
            for line in failures[:40]:
                print(f"  {line}")
            return 2

        if expect_rows is not None and max_partitions is None:
            # Partition-level digests prove every exported row is faithful, but they
            # say nothing about rows that were never enumerated -- a symbol missing
            # from --symbols would be skipped in complete silence. This is the only
            # check that closes that hole.
            if rows_checked != expect_rows:
                print(
                    f"VERIFY_FAIL coverage: archive holds {rows_checked:,} rows but the "
                    f"source table is {expect_rows:,} ({expect_rows - rows_checked:+,})"
                )
                return 2
            print(f"COVERAGE_OK rows={rows_checked:,} == source table count")

        mode = "parquet-only" if skip_source else "parquet+sqlite"
        # checked_now vs carried is stated explicitly: a 30-second VERIFY_OK that was
        # almost entirely carried forward must not read like a full pass.
        print(
            f"VERIFY_OK table={table} partitions={len(records)} rows={rows_checked:,} "
            f"mode={mode} checked_now={checked_now} carried={carried}"
        )
        return 0
    finally:
        con.close()


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", required=True)
    parser.add_argument("--db", default=str(DEFAULT_FROZEN_DB))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--symbols", default="", help="comma separated; avoids a DISTINCT scan")
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--max-partitions", type=int, default=None)
    parser.add_argument("--plan", action="store_true", help="show pending partitions, write nothing")
    parser.add_argument("--verify", action="store_true", help="run the proof gate")
    parser.add_argument(
        "--status", action="store_true", help="report progress from the manifest; touches nothing"
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="verify Parquet against the manifest only (does not close the gate)",
    )
    parser.add_argument(
        "--expect-rows",
        type=int,
        default=None,
        help="source table row count; verify fails unless the archive matches it exactly",
    )
    parser.add_argument(
        "--reverify",
        action="store_true",
        help="ignore the verification checkpoint and re-check every partition",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    db_path = Path(args.db)
    out_root = Path(args.out_root)

    try:
        if args.status:
            return run_status(db_path, args.table, out_root, args.expect_rows)
        if args.verify:
            return run_verify(
                db_path,
                args.table,
                out_root,
                args.max_partitions,
                args.skip_source_check,
                args.expect_rows,
                args.reverify,
            )
        return run_export(
            db_path,
            args.table,
            out_root,
            symbols or None,
            args.batch_rows,
            args.compression,
            args.max_partitions,
            args.plan,
        )
    except ExportError as exc:
        print(f"EXPORT_ERROR {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
