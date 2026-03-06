from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tools.check_data_ready import detect_ts_col, list_tables, normalize_ts_to_seconds, table_columns


@dataclass
class IngestionResult:
    rows_before: int
    rows_after: int
    rows_delta: int
    latest_ts_by_symbol: Dict[str, Optional[float]]
    lag_sec: Optional[int]
    verdict: str
    reason: str


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _pick_table(conn: sqlite3.Connection) -> tuple[str, str, Optional[str]]:
    tables = list_tables(conn)
    for t in ("agg_trades", "mark_prices", "liquidations"):
        if t in tables:
            cols = table_columns(conn, t)
            ts_col = detect_ts_col(cols)
            if ts_col:
                sym = None
                lower = [c.lower() for c in cols]
                if "symbol" in lower:
                    sym = cols[lower.index("symbol")]
                return t, ts_col, sym
    for t in tables:
        cols = table_columns(conn, t)
        ts_col = detect_ts_col(cols)
        if ts_col:
            sym = None
            lower = [c.lower() for c in cols]
            if "symbol" in lower:
                sym = cols[lower.index("symbol")]
            return t, ts_col, sym
    raise RuntimeError("no timestamped table found in database")


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0] or 0) if row else 0


def _latest_ts(
    conn: sqlite3.Connection,
    table: str,
    ts_col: str,
    sym_col: Optional[str],
    symbols: Iterable[str],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    syms = list(symbols)
    if sym_col and syms:
        for s in syms:
            row = conn.execute(f"SELECT MAX({ts_col}) FROM {table} WHERE {sym_col}=?", (s,)).fetchone()
            out[s] = normalize_ts_to_seconds(float(row[0])) if row and row[0] is not None else None
    else:
        row = conn.execute(f"SELECT MAX({ts_col}) FROM {table}").fetchone()
        out["ALL"] = normalize_ts_to_seconds(float(row[0])) if row and row[0] is not None else None
    return out


def run_ingestion_check(db: Path, symbols: List[str], window_sec: int, max_lag_sec: int) -> IngestionResult:
    if not db.exists():
        raise FileNotFoundError(str(db))
    conn = sqlite3.connect(str(db))
    try:
        table, ts_col, sym_col = _pick_table(conn)
        now = time.time()
        before = _count_rows(conn, table)
        time.sleep(max(1, int(window_sec)))
        after_now = time.time()
        after = _count_rows(conn, table)
        latest = _latest_ts(conn, table, ts_col, sym_col, symbols)
        latest_values = [v for v in latest.values() if v is not None]
        lag = None if not latest_values else int(max(0.0, after_now - max(latest_values)))
        delta = int(after - before)
        verdict = "OK"
        reason = ""
        if delta <= 0:
            verdict = "DEGRADED"
            reason = "rows_delta_zero"
        if lag is None or lag > int(max_lag_sec):
            verdict = "DEGRADED"
            reason = "lag_exceeded" if lag is not None else "lag_missing"
        return IngestionResult(
            rows_before=before,
            rows_after=after,
            rows_delta=delta,
            latest_ts_by_symbol=latest,
            lag_sec=lag,
            verdict=verdict,
            reason=reason,
        )
    finally:
        conn.close()


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "null"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(ts)))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="10-second ingestion proof checker for microstructure DB.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT")
    p.add_argument("--window-sec", type=int, default=10)
    p.add_argument("--max-lag-sec", type=int, default=5)
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        res = run_ingestion_check(
            db=Path(str(args.db)),
            symbols=_parse_symbols(args.symbols),
            window_sec=max(1, int(args.window_sec)),
            max_lag_sec=max(0, int(args.max_lag_sec)),
        )
    except FileNotFoundError as e:
        print(f"ingestion_check ERROR missing_db={e}")
        return 2
    except Exception as e:
        print(f"ingestion_check ERROR runtime={type(e).__name__}:{e}")
        return 2

    print(
        f"ingestion_check verdict={res.verdict} reason={res.reason} "
        f"rows_before={res.rows_before} rows_after={res.rows_after} rows_delta={res.rows_delta} lag_sec={res.lag_sec}"
    )
    for sym in sorted(res.latest_ts_by_symbol):
        print(f"- {sym}: latest_ts_utc={_fmt_ts(res.latest_ts_by_symbol[sym])}")

    if res.verdict == "OK":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
