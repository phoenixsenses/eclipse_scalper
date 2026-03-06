from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TS_CANDIDATES = [
    "ts",
    "timestamp",
    "time",
    "t",
    "T",
    "E",
    "ts_ms",
    "time_ms",
    "event_time",
    "eventTime",
    "transactTime",
    "tradeTime",
    "created_at",
    "createdAt",
]
SYMBOL_CANDIDATES = ["symbol", "s", "pair", "instrument", "ticker"]


@dataclass(frozen=True)
class TableDiag:
    table: str
    columns: List[str]
    ts_col: Optional[str]
    symbol_col: Optional[str]
    max_ts: Optional[float]
    age_sec: Optional[int]
    error: Optional[str] = None


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [str(r[0]) for r in rows if r and str(r[0]) and not str(r[0]).startswith("sqlite_")]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows if len(r) > 1]


def detect_ts_col(columns: List[str]) -> Optional[str]:
    lower_to_actual = {str(c).lower(): str(c) for c in columns}
    for cand in TS_CANDIDATES:
        found = lower_to_actual.get(cand.lower())
        if found:
            return found
    return None


def _detect_symbol_col(columns: List[str]) -> Optional[str]:
    lower_to_actual = {str(c).lower(): str(c) for c in columns}
    for cand in SYMBOL_CANDIDATES:
        found = lower_to_actual.get(cand.lower())
        if found:
            return found
    return None


def normalize_ts_to_seconds(mx: int | float) -> float:
    x = float(mx)
    if x > 1e12:
        return x / 1000.0
    return x


def compute_age_sec(mx: int | float, now: float) -> int:
    ts_sec = normalize_ts_to_seconds(mx)
    age = int(max(0.0, float(now) - ts_sec))
    return age


def _safe_max_ts(conn: sqlite3.Connection, table: str, ts_col: str) -> Optional[float]:
    try:
        row = conn.execute(f"SELECT MAX({ts_col}) FROM {table}").fetchone()
    except sqlite3.DatabaseError:
        return None
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except Exception:
        return None


def _latest_symbol_ts(conn: sqlite3.Connection, table_diags: List[TableDiag], symbol: str) -> Optional[float]:
    best: Optional[float] = None
    symbol_u = str(symbol or "").upper()
    for d in table_diags:
        if not d.ts_col or not d.symbol_col:
            continue
        try:
            row = conn.execute(
                f"SELECT MAX({d.ts_col}) FROM {d.table} WHERE {d.symbol_col} = ?",
                (symbol_u,),
            ).fetchone()
        except sqlite3.DatabaseError:
            continue
        if not row or row[0] is None:
            continue
        try:
            ts_val = float(row[0])
        except Exception:
            continue
        if best is None or ts_val > best:
            best = ts_val
    return best


def inspect_tables(conn: sqlite3.Connection, now: Optional[float] = None) -> List[TableDiag]:
    now_ts = float(now if now is not None else time.time())
    out: List[TableDiag] = []
    for table in list_tables(conn):
        try:
            cols = table_columns(conn, table)
        except sqlite3.DatabaseError as exc:
            out.append(
                TableDiag(
                    table=table,
                    columns=[],
                    ts_col=None,
                    symbol_col=None,
                    max_ts=None,
                    age_sec=None,
                    error=f"PRAGMA_FAILED:{exc}",
                )
            )
            continue
        ts_col = detect_ts_col(cols)
        symbol_col = _detect_symbol_col(cols)
        if ts_col is None:
            out.append(
                TableDiag(
                    table=table,
                    columns=cols,
                    ts_col=None,
                    symbol_col=symbol_col,
                    max_ts=None,
                    age_sec=None,
                    error=f"NO_TS_COL columns={','.join(cols)}",
                )
            )
            continue
        mx = _safe_max_ts(conn, table, ts_col)
        age = compute_age_sec(mx, now_ts) if mx is not None else None
        out.append(
            TableDiag(
                table=table,
                columns=cols,
                ts_col=ts_col,
                symbol_col=symbol_col,
                max_ts=mx,
                age_sec=age,
            )
        )
    return out


def check_db_fresh(
    conn: sqlite3.Connection,
    symbols: List[str],
    fresh_sec: int,
    now: Optional[float] = None,
) -> Tuple[bool, Dict[str, str], List[TableDiag]]:
    details: Dict[str, str] = {}
    now_ts = float(now if now is not None else time.time())
    table_diags = inspect_tables(conn, now=now_ts)
    if not table_diags:
        details["db"] = "no user tables in sqlite database"
        return False, details, table_diags
    for d in table_diags:
        details[f"table:{d.table}"] = (
            f"ts_col={d.ts_col or '-'} max_ts={d.max_ts if d.max_ts is not None else '-'} "
            f"age_sec={d.age_sec if d.age_sec is not None else '-'}"
            + (f" err={d.error}" if d.error else "")
        )
    key_tables = [d for d in table_diags if d.ts_col and d.symbol_col]
    if not key_tables:
        details["db"] = "no key tables with symbol+timestamp columns"
        return False, details, table_diags

    fresh_symbols = 0
    for sym in symbols:
        ts = _latest_symbol_ts(conn, key_tables, sym)
        if ts is None:
            details[f"symbol:{sym}"] = "no rows found in key tables"
            continue
        age = compute_age_sec(ts, now_ts)
        details[f"symbol:{sym}"] = f"last_ts={ts} age_sec={age}"
        if age <= int(max(1, fresh_sec)):
            fresh_symbols += 1
    if fresh_symbols <= 0:
        details["db"] = f"stale: no symbol had rows within fresh_sec={fresh_sec}"
        return False, details, table_diags

    freshest_table_age = min((d.age_sec for d in key_tables if d.age_sec is not None), default=None)
    if freshest_table_age is None:
        details["db"] = "no timestamp values found in key tables"
        return False, details, table_diags
    if freshest_table_age > int(max(1, fresh_sec)):
        details["db"] = f"stale: freshest_table_age_sec={freshest_table_age} > fresh_sec={fresh_sec}"
        return False, details, table_diags

    details["db"] = "ready"
    return True, details, table_diags


def check_ready(
    db_path: Path,
    csv_path: Path,
    symbols: List[str],
    fresh_sec: int,
) -> Tuple[bool, Dict[str, str]]:
    details: Dict[str, str] = {}
    if not db_path.exists():
        details["db"] = f"missing: {db_path}"
        return False, details
    if not csv_path.exists():
        details["csv"] = f"missing: {csv_path}"
        return False, details
    age = max(0.0, time.time() - csv_path.stat().st_mtime)
    if age > max(1, int(fresh_sec)):
        details["csv"] = f"stale: age_sec={age:.1f} fresh_sec={fresh_sec}"
        return False, details
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except Exception as exc:
        details["db"] = f"open_failed: {exc}"
        return False, details
    try:
        ok_db, db_details, _ = check_db_fresh(conn, symbols=symbols, fresh_sec=fresh_sec)
        details.update(db_details)
        if not ok_db:
            return False, details
        details["ok"] = "ready"
        details["tables"] = ",".join(list_tables(conn))
        return True, details
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check microstructure data readiness.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--csv", default="data/event_diary.csv")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--fresh-sec", type=int, default=120)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        ok, details = check_ready(
            db_path=Path(str(args.db)),
            csv_path=Path(str(args.csv)),
            symbols=_parse_symbols(args.symbols),
            fresh_sec=int(args.fresh_sec),
        )
        if ok:
            print("READY check_data_ready")
            for k, v in details.items():
                print(f"{k}={v}")
            return 0
        print("NOT_READY check_data_ready")
        for k, v in details.items():
            print(f"{k}={v}")
        return 2
    except Exception as exc:
        print(f"ERROR check_data_ready {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
