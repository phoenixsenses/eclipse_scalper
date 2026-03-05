from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tools.check_data_ready import detect_ts_col, list_tables, table_columns
from tools.health_state import write_component_health, write_overall_health, utc_now_iso


def _parse_iso_utc(text: str) -> float:
    s = str(text or "").strip()
    if not s:
        raise ValueError("empty datetime")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _table_specs(conn: sqlite3.Connection) -> List[Tuple[str, str, str | None]]:
    out: List[Tuple[str, str, str | None]] = []
    for t in ("agg_trades", "mark_prices", "liquidations"):
        if t not in list_tables(conn):
            continue
        cols = table_columns(conn, t)
        ts_col = detect_ts_col(cols)
        if not ts_col:
            continue
        lower = [c.lower() for c in cols]
        sym_col = cols[lower.index("symbol")] if "symbol" in lower else None
        out.append((t, ts_col, sym_col))
    return out


def _fetch_rows(
    conn: sqlite3.Connection,
    specs: List[Tuple[str, str, str | None]],
    symbols: Iterable[str],
    start_ts: float,
    end_ts: float,
) -> List[Tuple[float, str, str]]:
    rows: List[Tuple[float, str, str]] = []
    syms = list(symbols)
    for table, ts_col, sym_col in specs:
        base = f"SELECT {ts_col}, {sym_col if sym_col else 'NULL'} FROM {table} WHERE {ts_col} >= ? AND {ts_col} <= ?"
        params: list[object] = [start_ts, end_ts]
        if sym_col and syms:
            placeholders = ",".join("?" for _ in syms)
            base += f" AND {sym_col} IN ({placeholders})"
            params.extend(syms)
        cur = conn.execute(base, tuple(params))
        for ts, sym in cur.fetchall():
            try:
                v = float(ts)
                if v > 1e12:
                    v = v / 1000.0
                s = str(sym).upper() if sym is not None else "ALL"
                rows.append((v, table, s))
            except Exception:
                continue
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return rows


def _write_replay_health(status: str, last_progress_ts: str | None, replayed_events: int, reason: str = "") -> None:
    comp = {
        "status": status,
        "connected": True if status == "ok" else False if status == "halted" else None,
        "last_progress_ts_utc": last_progress_ts,
        "progress_lag_sec": 0 if last_progress_ts else None,
        "reconnects_last_5m": 0,
        "errors_last_5m": 0,
        "replayed_events": int(replayed_events),
        "reason": reason,
    }
    write_component_health("replay", comp)
    overall = {
        "ts_utc": utc_now_iso(),
        "mode": "paper",
        "state": "ok" if status == "ok" else "degraded",
        "reason": reason,
        "components": {
            "replay": comp,
        },
    }
    write_overall_health(overall)


def run_replay(
    db: Path,
    symbols: List[str],
    start_iso: str,
    end_iso: str,
    speed: float,
    progress_every: int = 1000,
) -> int:
    if not db.exists():
        print(f"replay_slice error missing_db={db}")
        return 2
    start_ts = _parse_iso_utc(start_iso)
    end_ts = _parse_iso_utc(end_iso)
    if end_ts <= start_ts:
        print("replay_slice error invalid_range")
        return 2
    conn = sqlite3.connect(str(db))
    try:
        specs = _table_specs(conn)
        if not specs:
            print("replay_slice error no_supported_tables")
            return 2
        rows = _fetch_rows(conn, specs, symbols, start_ts, end_ts)
    finally:
        conn.close()

    print(f"replay_slice rows={len(rows)} start={start_iso} end={end_iso}")
    replayed = 0
    prev_ts = None
    for ts, table, sym in rows:
        replayed += 1
        if prev_ts is not None and speed > 0:
            dt = max(0.0, float(ts - prev_ts))
            sleep_sec = dt / max(1e-6, float(speed))
            if sleep_sec > 0:
                time.sleep(min(0.25, sleep_sec))
        prev_ts = ts
        if replayed % max(1, int(progress_every)) == 0:
            cur_iso = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))
            print(f"replay_slice progress replayed_events={replayed} current_ts={cur_iso} table={table} symbol={sym}")
            _write_replay_health("ok", utc_now_iso(), replayed)

    _write_replay_health("ok", utc_now_iso(), replayed)
    print(f"replay_slice done replayed_events={replayed}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deterministic offline replay skeleton for microstructure DB slices.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--speed", type=float, default=50.0)
    p.add_argument("--progress-every", type=int, default=1000)
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        return run_replay(
            db=Path(str(args.db)),
            symbols=_parse_symbols(args.symbols),
            start_iso=str(args.start),
            end_iso=str(args.end),
            speed=float(args.speed),
            progress_every=max(1, int(args.progress_every)),
        )
    except Exception as e:
        print(f"replay_slice error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

