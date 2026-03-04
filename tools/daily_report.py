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
    root = Path(__file__).resolve().parents[1]
    env_paper = root / ".env.paper"
    env_default = root / ".env"
    if env_paper.exists():
        load_dotenv(dotenv_path=env_paper, override=False)
    elif env_default.exists():
        load_dotenv(dotenv_path=env_default, override=False)


_load_dotenv_best_effort()


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate daily paper trading report.")
    p.add_argument("--db", default="data/paper_trades.db")
    p.add_argument("--out-dir", default="reports/daily")
    p.add_argument("--push", action="store_true")
    return p.parse_args()


def _summary(conn: sqlite3.Connection) -> dict:
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    today = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(pnl_bps),0) pnl, "
        "COALESCE(AVG(CASE WHEN pnl_bps>0 THEN 1.0 ELSE 0.0 END),0) wr "
        "FROM trades WHERE exit_time>=?",
        (day_start,),
    ).fetchone()
    total = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(pnl_bps),0) pnl, "
        "COALESCE(AVG(CASE WHEN pnl_bps>0 THEN 1.0 ELSE 0.0 END),0) wr "
        "FROM trades"
    ).fetchone()
    return {
        "today_n": int(today[0] or 0),
        "today_pnl": float(today[1] or 0.0),
        "today_wr": float(today[2] or 0.0),
        "total_n": int(total[0] or 0),
        "total_pnl": float(total[1] or 0.0),
        "total_wr": float(total[2] or 0.0),
    }


def main() -> int:
    args = _args()
    db = Path(args.db)
    if not db.exists():
        print(f"daily_report: missing db {db}")
        return 2
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        s = _summary(conn)
    finally:
        conn.close()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{today}.md"
    anomaly = []
    if s["today_n"] > 0 and s["today_wr"] < 0.35:
        anomaly.append("low_win_rate")
    if s["today_pnl"] < -50.0:
        anomaly.append("deep_negative_day")
    md = "\n".join(
        [
            f"# Daily Report ({today} UTC)",
            "",
            f"- Today trades: {s['today_n']}",
            f"- Today pnl_bps: {s['today_pnl']:+.2f}",
            f"- Today win_rate: {s['today_wr']*100.0:.1f}%",
            f"- Total trades: {s['total_n']}",
            f"- Total pnl_bps: {s['total_pnl']:+.2f}",
            f"- Total win_rate: {s['total_wr']*100.0:.1f}%",
            f"- Anomalies: {', '.join(anomaly) if anomaly else 'none'}",
            "",
        ]
    )
    out.write_text(md, encoding="utf-8")
    print(f"daily_report: wrote {out}")
    if args.push:
        try:
            import os
            import asyncio
            from notifications.telegram import Notifier  # type: ignore

            token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN")
            chat = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID")
            if token and chat:
                asyncio.run(Notifier(token=token, chat_id=chat).speak(md, priority="normal", silent=True))
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

