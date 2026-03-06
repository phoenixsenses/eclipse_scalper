from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _load_dotenv_best_effort() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    try:
        env_paper = Path(".env.paper")
        env_default = Path(".env")
        if env_paper.exists():
            load_dotenv(dotenv_path=env_paper, override=False)
        elif env_default.exists():
            load_dotenv(dotenv_path=env_default, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        return


_load_dotenv_best_effort()


def _connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _fmt_dur(sec: float) -> str:
    s = int(max(0, sec))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d > 0:
        return f"{d}d {h}h {m}m"
    return f"{h}h {m}m {s}s"


def main() -> int:
    p = argparse.ArgumentParser(description="Quick paper trading status from paper_trades.db.")
    p.add_argument("--db", default="data/paper_trades.db")
    args = p.parse_args()
    db = Path(str(args.db))
    if not db.exists():
        print("Eclipse Scalper Paper Trading Status\n  DB missing: data/paper_trades.db")
        return 0
    conn = _connect(str(db))
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] or 0)
        last = conn.execute("SELECT * FROM trades ORDER BY exit_time DESC LIMIT 1").fetchone()
        if total <= 0 or last is None:
            print("Eclipse Scalper Paper Trading Status\n  No trades yet.")
            return 0
        now = datetime.now(timezone.utc).timestamp()
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        today = conn.execute(
            "SELECT COUNT(*) as n, COALESCE(SUM(pnl_bps),0) as pnl, "
            "COALESCE(AVG(CASE WHEN pnl_bps>0 THEN 1.0 ELSE 0.0 END),0) as win_rate "
            "FROM trades WHERE exit_time>=?",
            (day_start,),
        ).fetchone()
        total_pnl = float(conn.execute("SELECT COALESCE(SUM(pnl_bps),0) FROM trades").fetchone()[0] or 0.0)
        total_wr = float(
            conn.execute("SELECT COALESCE(AVG(CASE WHEN pnl_bps>0 THEN 1.0 ELSE 0.0 END),0) FROM trades").fetchone()[0] or 0.0
        )
        first_ts = float(conn.execute("SELECT MIN(entry_time) FROM trades").fetchone()[0] or 0.0)
        running = _fmt_dur(now - first_ts) if first_ts > 0 else "n/a"
        print("Eclipse Scalper Paper Trading Status")
        print(f"  Running since: {datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ({running})")
        print(
            f"  Today: {int(today['n'])} trades | {float(today['pnl']):+.2f} bps | "
            f"{float(today['win_rate']):.0%} win rate"
        )
        print(f"  Total: {total} trades | {total_pnl:+.2f} bps | {total_wr:.0%} win rate")
        print(
            f"  Last trade: {datetime.fromtimestamp(float(last['exit_time'] or 0.0), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"| {str(last['side'] or '').lower()} | {float(last['pnl_bps'] or 0.0):+.2f} bps | {str(last['exit_type'] or '')}"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
