from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.check_data_ready import detect_ts_col, list_tables, normalize_ts_to_seconds, table_columns


@dataclass(frozen=True)
class GapRecord:
    symbol: str
    start_ts_sec: float
    end_ts_sec: float
    gap_sec: float
    critical: bool


def _fmt_utc(ts_sec: float) -> str:
    return datetime.fromtimestamp(float(ts_sec), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _pick_table(conn: sqlite3.Connection, requested: str = "") -> Optional[Tuple[str, str]]:
    req = str(requested or "").strip()
    if req:
        cols = table_columns(conn, req)
        ts_col = detect_ts_col(cols)
        return (req, ts_col) if ts_col else None
    tables = list_tables(conn)
    preferred = ["mark_prices", "agg_trades", "liquidations"]
    for name in preferred:
        if name in tables:
            cols = table_columns(conn, name)
            ts_col = detect_ts_col(cols)
            if ts_col:
                return name, ts_col
    for t in tables:
        cols = table_columns(conn, t)
        ts_col = detect_ts_col(cols)
        if ts_col:
            return t, ts_col
    return None


def _get_symbol_col(conn: sqlite3.Connection, table: str) -> Optional[str]:
    cols = [c.lower() for c in table_columns(conn, table)]
    if "symbol" in cols:
        idx = cols.index("symbol")
        return table_columns(conn, table)[idx]
    return None


def _load_stream_rows(
    conn: sqlite3.Connection,
    table: str,
    ts_col: str,
    symbol_col: Optional[str],
    symbols: List[str],
) -> List[Tuple[str, float]]:
    if symbol_col and symbols:
        q_marks = ",".join("?" for _ in symbols)
        sql = f"SELECT {symbol_col}, {ts_col} FROM {table} WHERE {symbol_col} IN ({q_marks}) ORDER BY {symbol_col}, {ts_col}"
        rows = conn.execute(sql, tuple(symbols)).fetchall()
        return [(str(r[0]), normalize_ts_to_seconds(float(r[1]))) for r in rows if r[1] is not None]
    if symbol_col:
        sql = f"SELECT {symbol_col}, {ts_col} FROM {table} ORDER BY {symbol_col}, {ts_col}"
        rows = conn.execute(sql).fetchall()
        return [(str(r[0]), normalize_ts_to_seconds(float(r[1]))) for r in rows if r[1] is not None]
    sql = f"SELECT {ts_col} FROM {table} ORDER BY {ts_col}"
    rows = conn.execute(sql).fetchall()
    return [("ALL", normalize_ts_to_seconds(float(r[0]))) for r in rows if r[0] is not None]


def analyze_collection_health(
    db_path: Path,
    symbols: List[str],
    gap_threshold_sec: int = 60,
    alert_threshold_sec: int = 300,
    table: str = "",
) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        picked = _pick_table(conn, requested=table)
        if not picked:
            raise RuntimeError("no timestamped table found")
        table_name, ts_col = picked
        symbol_col = _get_symbol_col(conn, table_name)
        rows = _load_stream_rows(conn, table_name, ts_col, symbol_col, symbols)
        if len(rows) < 2:
            return {
                "status": "insufficient_data",
                "table": table_name,
                "ts_col": ts_col,
                "symbol_col": symbol_col,
                "rows": len(rows),
                "gap_records": [],
            }
        by_symbol: Dict[str, List[float]] = {}
        for sym, ts in rows:
            by_symbol.setdefault(sym, []).append(float(ts))
        gaps: List[GapRecord] = []
        for sym, ts_list in by_symbol.items():
            for i in range(1, len(ts_list)):
                dt = float(ts_list[i] - ts_list[i - 1])
                if dt > float(gap_threshold_sec):
                    gaps.append(
                        GapRecord(
                            symbol=sym,
                            start_ts_sec=float(ts_list[i - 1]),
                            end_ts_sec=float(ts_list[i]),
                            gap_sec=dt,
                            critical=dt >= float(alert_threshold_sec),
                        )
                    )
        all_ts = [ts for vals in by_symbol.values() for ts in vals]
        start_ts = min(all_ts)
        end_ts = max(all_ts)
        span_sec = max(1.0, end_ts - start_ts)
        total_gap_sec = sum(g.gap_sec for g in gaps)
        uptime_pct = max(0.0, 100.0 * (1.0 - (total_gap_sec / span_sec)))
        rows_per_day = (len(all_ts) / max(1.0, span_sec / 86400.0))
        longest_gap = max((g.gap_sec for g in gaps), default=0.0)
        gaps_by_hour: Dict[int, int] = {}
        for g in gaps:
            h = datetime.fromtimestamp(g.start_ts_sec, tz=timezone.utc).hour
            gaps_by_hour[h] = int(gaps_by_hour.get(h, 0) + 1)
        per_symbol_summary: Dict[str, Dict[str, Any]] = {}
        for sym, ts_list in sorted(by_symbol.items()):
            per_symbol_summary[sym] = {
                "rows": len(ts_list),
                "first_ts_utc": _fmt_utc(min(ts_list)),
                "last_ts_utc": _fmt_utc(max(ts_list)),
            }
        return {
            "status": "ok",
            "table": table_name,
            "ts_col": ts_col,
            "symbol_col": symbol_col,
            "symbols": sorted(by_symbol.keys()),
            "rows_total": len(all_ts),
            "rows_per_day_avg": rows_per_day,
            "date_range": {"start_utc": _fmt_utc(start_ts), "end_utc": _fmt_utc(end_ts), "span_sec": span_sec},
            "uptime_pct": uptime_pct,
            "total_gap_sec": total_gap_sec,
            "gap_count": len(gaps),
            "critical_gap_count": sum(1 for g in gaps if g.critical),
            "longest_gap_sec": longest_gap,
            "longest_gap": (
                None
                if not gaps
                else {
                    "symbol": max(gaps, key=lambda x: x.gap_sec).symbol,
                    "start_utc": _fmt_utc(max(gaps, key=lambda x: x.gap_sec).start_ts_sec),
                    "end_utc": _fmt_utc(max(gaps, key=lambda x: x.gap_sec).end_ts_sec),
                    "gap_sec": max(gaps, key=lambda x: x.gap_sec).gap_sec,
                }
            ),
            "gaps_by_hour_utc": {str(k): int(v) for k, v in sorted(gaps_by_hour.items())},
            "per_symbol": per_symbol_summary,
            "gap_records": [
                {
                    "symbol": g.symbol,
                    "start_utc": _fmt_utc(g.start_ts_sec),
                    "end_utc": _fmt_utc(g.end_ts_sec),
                    "gap_sec": g.gap_sec,
                    "critical": g.critical,
                }
                for g in gaps
            ],
        }
    finally:
        conn.close()


def write_markdown(report: Dict[str, Any], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["# COLLECTION_HEALTH", ""]
    if report.get("status") != "ok":
        lines.append(f"status={report.get('status')}")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.extend(
        [
            f"table={report.get('table')} ts_col={report.get('ts_col')} symbol_col={report.get('symbol_col')}",
            "",
            f"rows_total={int(report.get('rows_total', 0))}",
            f"rows_per_day_avg={float(report.get('rows_per_day_avg', 0.0)):.2f}",
            f"uptime_pct={float(report.get('uptime_pct', 0.0)):.4f}",
            f"gap_count={int(report.get('gap_count', 0))}",
            f"critical_gap_count={int(report.get('critical_gap_count', 0))}",
            f"longest_gap_sec={float(report.get('longest_gap_sec', 0.0)):.2f}",
            f"date_start_utc={report.get('date_range', {}).get('start_utc', '-')}",
            f"date_end_utc={report.get('date_range', {}).get('end_utc', '-')}",
            "",
            "## Gaps By Hour UTC",
        ]
    )
    for h, n in sorted((report.get("gaps_by_hour_utc", {}) or {}).items()):
        lines.append(f"- hour={h}: count={n}")
    lines.append("")
    lines.append("## First 50 Gap Records")
    lines.append("")
    lines.append("| symbol | start_utc | end_utc | gap_sec | critical |")
    lines.append("|---|---|---|---:|---|")
    for row in (report.get("gap_records", []) or [])[:50]:
        lines.append(
            f"| {row.get('symbol')} | {row.get('start_utc')} | {row.get('end_utc')} | {float(row.get('gap_sec', 0.0)):.2f} | {bool(row.get('critical'))} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collection health report for microstructure sqlite stream.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--gap-threshold-sec", type=int, default=60)
    p.add_argument("--alert-threshold-sec", type=int, default=300)
    p.add_argument("--table", default="", help="Optional source table override (default picks mark_prices first).")
    p.add_argument("--out-md", default="reports/COLLECTION_HEALTH.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    db_path = Path(str(args.db))
    if not db_path.exists():
        print(f"ERROR collection_health missing_db path={db_path}")
        return 2
    print(f"[scan] db={db_path}")
    print(f"[scan] symbols={_parse_symbols(args.symbols)}")
    report = analyze_collection_health(
        db_path=db_path,
        symbols=_parse_symbols(args.symbols),
        gap_threshold_sec=int(args.gap_threshold_sec),
        alert_threshold_sec=int(args.alert_threshold_sec),
        table=str(args.table),
    )
    out_md = Path(str(args.out_md))
    write_markdown(report, out_md)
    print(f"status={report.get('status')}")
    print(f"table={report.get('table')}")
    print(f"rows_total={report.get('rows_total')}")
    print(f"uptime_pct={float(report.get('uptime_pct', 0.0)):.4f}")
    print(f"gap_count={report.get('gap_count')}")
    print(f"critical_gap_count={report.get('critical_gap_count')}")
    print(f"longest_gap_sec={float(report.get('longest_gap_sec', 0.0)):.2f}")
    print(f"out_md={out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
