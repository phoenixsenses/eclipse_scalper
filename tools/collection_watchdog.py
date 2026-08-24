"""DEPRECATED. Retired as an always-on process by a canonical operational-topology
decision; kept as an importable module for the health surfaces that still call it.

This module used to run a persistent loop that wrote both
logs/health/watchdog.json and logs/health/overall.json under the same
component name/filenames that tools/heartbeat_watchdog.py also writes,
racing against it (and, until 2026-07-03, against
data/microstructure_collector.py's own blind-overwrite of overall.json).
It was never launched by start_eclipse.ps1 in the first place.

The one genuinely unique responsibility here -- research-fitness readiness
(analyze_research_fitness) -- now lives in tools/research_fitness_report.py
as a one-shot, dedicated-output tool. This file is kept only as a thin,
non-looping compatibility shim for anyone still invoking it by name; it
cannot write any operational-health file and cannot loop.

The staleness-detection helpers below (get_latest_timestamp, _is_stale,
_pick_table) are kept as plain, side-effect-free utilities -- still exercised
by tests/test_collection_watchdog.py -- but are no longer wired into any
alerting or health-file-writing path.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from logging.handlers import RotatingFileHandler
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tools.check_data_ready import detect_ts_col, list_tables, normalize_ts_to_seconds, table_columns
from tools.research_fitness_report import build_report, _atomic_write_json, _parse_symbols


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("collection_watchdog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    h = RotatingFileHandler(str(log_path), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
    return logger


def _pick_table(conn: sqlite3.Connection, requested: str = "") -> Optional[Tuple[str, str]]:
    req = str(requested or "").strip()
    if req:
        cols = table_columns(conn, req)
        ts_col = detect_ts_col(cols)
        return (req, ts_col) if ts_col else None
    tables = list_tables(conn)
    for t in ("mark_prices", "agg_trades", "liquidations"):
        if t in tables:
            cols = table_columns(conn, t)
            ts_col = detect_ts_col(cols)
            if ts_col:
                return t, ts_col
    for t in tables:
        cols = table_columns(conn, t)
        ts_col = detect_ts_col(cols)
        if ts_col:
            return t, ts_col
    return None


def _symbol_col(conn: sqlite3.Connection, table: str) -> Optional[str]:
    cols = table_columns(conn, table)
    lower = [c.lower() for c in cols]
    if "symbol" in lower:
        return cols[lower.index("symbol")]
    return None


def get_latest_timestamp(
    db_path: Path,
    symbols: List[str],
    table: str = "",
) -> Dict[str, Optional[float]]:
    if not db_path.exists():
        raise FileNotFoundError(str(db_path))
    conn = sqlite3.connect(str(db_path))
    try:
        picked = _pick_table(conn, requested=table)
        if not picked:
            raise RuntimeError("no timestamped table found")
        table_name, ts_col = picked
        sym_col = _symbol_col(conn, table_name)
        out: Dict[str, Optional[float]] = {"_table": table_name, "_ts_col": ts_col}
        if sym_col and symbols:
            for sym in symbols:
                row = conn.execute(
                    f"SELECT MAX({ts_col}) FROM {table_name} WHERE {sym_col}=?",
                    (sym,),
                ).fetchone()
                out[sym] = (normalize_ts_to_seconds(float(row[0])) if row and row[0] is not None else None)
        else:
            row = conn.execute(f"SELECT MAX({ts_col}) FROM {table_name}").fetchone()
            out["ALL"] = (normalize_ts_to_seconds(float(row[0])) if row and row[0] is not None else None)
        return out
    finally:
        conn.close()


def _is_stale(latest_ts_sec: Optional[float], stale_threshold_sec: int, now_sec: Optional[float] = None) -> bool:
    if latest_ts_sec is None:
        return True
    now = float(time.time() if now_sec is None else now_sec)
    return (now - float(latest_ts_sec)) > float(stale_threshold_sec)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="DEPRECATED -- delegates to tools.research_fitness_report (one-shot, no operational-health writes)."
    )
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--csv", default="data/event_diary.csv")
    p.add_argument("--fresh-sec", type=int, default=120)
    p.add_argument("--out", default="logs/health/research_fitness.json")
    # Accepted for backward-compatible argument parsing only; a persistent
    # loop is intentionally no longer possible from this entry point.
    p.add_argument("--check-interval-sec", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--stale-threshold-sec", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--table", default="", help=argparse.SUPPRESS)
    p.add_argument("--log", default=None, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    print(
        "DEPRECATED tools.collection_watchdog no longer runs a persistent loop or writes "
        "logs/health/watchdog.json or logs/health/overall.json. Delegating once to "
        "tools.research_fitness_report; use that module directly going forward.",
        file=sys.stderr,
    )

    report = build_report(
        db_path=Path(str(args.db)),
        csv_path=Path(str(args.csv)),
        symbols=_parse_symbols(args.symbols),
        fresh_sec=int(args.fresh_sec),
        stale_after_sec=3600,
    )
    _atomic_write_json(Path(str(args.out)), report)
    print(f"research_fitness status={report['status']} contract_tier={report['contract_tier']}")
    return {"ready": 0, "limited": 1, "blocked": 2}.get(report["status"], 3)


if __name__ == "__main__":
    sys.exit(main())
