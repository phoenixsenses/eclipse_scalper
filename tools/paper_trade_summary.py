from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from tools.run_summary import build_run_summary
from utils.env_profile import load_dotenv_best_effort


load_dotenv_best_effort(root=Path.cwd(), cwd_fallback=True)


def _connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, days: int = 0) -> List[sqlite3.Row]:
    if days and days > 0:
        now = datetime.now(timezone.utc).timestamp()
        start = now - float(days) * 86400.0
        return conn.execute("SELECT * FROM trades WHERE exit_time>=? ORDER BY exit_time ASC", (start,)).fetchall()
    return conn.execute("SELECT * FROM trades ORDER BY exit_time ASC").fetchall()


def _pct(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def _daily(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    by: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        ts = float(r["exit_time"] or 0.0)
        if ts <= 0:
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        by.setdefault(day, []).append(r)
    out: List[Dict[str, Any]] = []
    for d in sorted(by):
        rs = by[d]
        pnl = [float(x["pnl_bps"] or 0.0) for x in rs]
        scratches = sum(1 for x in rs if "scratch" in str(x["exit_reason"] or "").lower())
        wins = sum(1 for x in pnl if x > 0.0)
        eq = 1.0
        peak = 1.0
        dd = 0.0
        for p in pnl:
            eq *= (1.0 + p / 10000.0)
            if eq > peak:
                peak = eq
            cur = (peak - eq) / peak if peak > 0 else 0.0
            if cur > dd:
                dd = cur
        out.append(
            {
                "date": d,
                "trades": len(rs),
                "win_rate": _pct(wins, len(rs)),
                "mean_pnl_bps": mean(pnl) if pnl else 0.0,
                "total_pnl_bps": sum(pnl),
                "scratch_rate": _pct(scratches, len(rs)),
                "max_drawdown_bps": dd * 10000.0,
            }
        )
    return out


def _rolling7(daily: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(len(daily)):
        w = daily[max(0, i - 6) : i + 1]
        pnls = [float(x["total_pnl_bps"]) for x in w]
        wins = [float(x["win_rate"]) for x in w]
        out.append(
            {
                "date": daily[i]["date"],
                "roll7_total_pnl_bps": sum(pnls),
                "roll7_mean_win_rate": mean(wins) if wins else 0.0,
                "roll7_mean_daily_pnl_bps": mean(pnls) if pnls else 0.0,
            }
        )
    return out


def generate_summary(db: str, days: int = 0) -> Dict[str, Any]:
    conn = _connect(db)
    try:
        rs = _rows(conn, days=days)
    finally:
        conn.close()
    pnls = [float(r["pnl_bps"] or 0.0) for r in rs]
    wins = sum(1 for p in pnls if p > 0.0)
    scratches = sum(1 for r in rs if "scratch" in str(r["exit_reason"] or "").lower())
    regime_up = sum(1 for r in rs if str(r["regime"] or "").upper() == "UP")
    regime_down = sum(1 for r in rs if str(r["regime"] or "").upper() == "DOWN")
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for p in pnls:
        eq *= (1.0 + p / 10000.0)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    exit_types: Dict[str, int] = {}
    for r in rs:
        t = str(r["exit_type"] or "")
        exit_types[t] = int(exit_types.get(t, 0)) + 1
    by_day = _daily(rs)
    return {
        "total_trades": len(rs),
        "win_rate": _pct(wins, len(rs)),
        "mean_pnl_bps": mean(pnls) if pnls else 0.0,
        "total_pnl_bps": sum(pnls),
        "scratch_rate": _pct(scratches, len(rs)),
        "regime_up_pct": _pct(regime_up, len(rs)),
        "regime_down_pct": _pct(regime_down, len(rs)),
        "max_drawdown_bps": max_dd * 10000.0,
        "exit_types": dict(sorted(exit_types.items())),
        "daily": by_day,
        "rolling7": _rolling7(by_day),
    }


def _write_md(path: Path, s: Dict[str, Any], compare_backtest: str = "") -> None:
    lines: List[str] = []
    lines.append("# PAPER_TRADE_SUMMARY")
    lines.append("")
    lines.append(
        f"total_trades={int(s['total_trades'])} win_rate={float(s['win_rate']):.2%} "
        f"mean_pnl_bps={float(s['mean_pnl_bps']):+.3f} total_pnl_bps={float(s['total_pnl_bps']):+.3f}"
    )
    lines.append(
        f"scratch_rate={float(s['scratch_rate']):.2%} max_drawdown_bps={float(s['max_drawdown_bps']):.2f} "
        f"regime_up_pct={float(s['regime_up_pct']):.2%} regime_down_pct={float(s['regime_down_pct']):.2%}"
    )
    lines.append("")
    lines.append("## Daily")
    lines.append("| date | trades | win_rate | mean_pnl_bps | total_pnl_bps | scratch_rate | max_drawdown_bps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in s["daily"]:
        lines.append(
            f"| {d['date']} | {int(d['trades'])} | {float(d['win_rate']):.2%} | {float(d['mean_pnl_bps']):+.3f} | "
            f"{float(d['total_pnl_bps']):+.3f} | {float(d['scratch_rate']):.2%} | {float(d['max_drawdown_bps']):.2f} |"
        )
    lines.append("")
    lines.append("## Exit Types")
    for k, v in sorted((s.get("exit_types") or {}).items()):
        lines.append(f"- {k}: {int(v)}")
    lines.append("")
    if compare_backtest:
        lines.append(f"compare_backtest={compare_backtest}")
    if isinstance(s.get("run_summary"), dict):
        lines.append("")
        lines.append("## Run Summary")
        lines.append(f"- `{s['run_summary']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize paper trade performance from data/paper_trades.db.")
    p.add_argument("--db", default="data/paper_trades.db")
    p.add_argument("--out-md", default="reports/PAPER_TRADE_SUMMARY.md")
    p.add_argument("--out-json", default="reports/PAPER_TRADE_SUMMARY.json")
    p.add_argument("--days", type=int, default=0, help="Limit to last N days; 0 means all.")
    p.add_argument("--compare-backtest", default="")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    db_path = Path(str(args.db))
    if not db_path.exists():
        print(f"paper_trade_summary skip_missing_db db={db_path}")
        return 0
    s = generate_summary(str(db_path), days=int(args.days))
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    s["run_summary"] = build_run_summary(
        run_type="paper_trade_summary",
        inputs={
            "db": str(args.db),
            "days": int(args.days),
            "compare_backtest": str(args.compare_backtest or ""),
        },
        metrics={
            "total_trades": int(s.get("total_trades", 0)),
            "win_rate": float(s.get("win_rate", 0.0)),
            "total_pnl_bps": float(s.get("total_pnl_bps", 0.0)),
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    _write_md(out_md, s, compare_backtest=str(args.compare_backtest or ""))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(s, ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        f"paper_trade_summary trades={int(s['total_trades'])} win_rate={float(s['win_rate']):.2%} "
        f"total_pnl_bps={float(s['total_pnl_bps']):+.3f} out_md={out_md} out_json={out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
