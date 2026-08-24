from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.performance_monitor import DailyMetrics, compute_daily_metrics, detect_anomalies, weekly_digest_markdown


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
    p = argparse.ArgumentParser(description="Generate daily/weekly paper trading reports.")
    p.add_argument("--db", default="data/paper_trades.db")
    p.add_argument("--out-dir", default="reports/daily")
    p.add_argument("--push", action="store_true")
    p.add_argument("--schedule", action="store_true", help="Run forever and emit at --at-utc each day.")
    p.add_argument("--at-utc", default="00:05", help="Daily schedule time in UTC, HH:MM.")
    p.add_argument("--weekly", action="store_true", help="Also emit weekly digest (Sunday UTC).")
    p.add_argument("--expected-pnl-bps", type=float, default=0.0)
    p.add_argument("--expected-sigma-bps", type=float, default=15.0)
    return p.parse_args()


def _summary(conn: sqlite3.Connection, days: int = 1) -> dict:
    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - max(1, int(days)) * 86400.0
    row = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(pnl_bps),0) pnl, "
        "COALESCE(AVG(CASE WHEN pnl_bps>0 THEN 1.0 ELSE 0.0 END),0) wr, "
        "COALESCE(AVG(CASE WHEN ABS(pnl_bps)<=1.0 THEN 1.0 ELSE 0.0 END),0) scratch "
        "FROM trades WHERE exit_time>=?",
        (cutoff,),
    ).fetchone()
    return {
        "n": int((row[0] if row else 0) or 0),
        "pnl": float((row[1] if row else 0.0) or 0.0),
        "wr": float((row[2] if row else 0.0) or 0.0),
        "scratch": float((row[3] if row else 0.0) or 0.0),
    }


def _daily_markdown(metrics: DailyMetrics, anomalies: list[str], rolling7: dict) -> str:
    reg_total = metrics.regime_up_sec + metrics.regime_down_sec + metrics.regime_unknown_sec
    reg_up = (metrics.regime_up_sec / reg_total * 100.0) if reg_total > 0 else 0.0
    reg_down = (metrics.regime_down_sec / reg_total * 100.0) if reg_total > 0 else 0.0
    reg_unk = (metrics.regime_unknown_sec / reg_total * 100.0) if reg_total > 0 else 0.0
    return "\n".join(
        [
            f"# Daily Report ({metrics.day_utc} UTC)",
            "",
            "## Performance",
            f"- trade_count: {metrics.trade_count}",
            f"- win_rate: {metrics.win_rate*100.0:.1f}%",
            f"- pnl_bps: {metrics.pnl_bps:+.2f}",
            f"- scratch_rate: {metrics.scratch_rate*100.0:.1f}%",
            "",
            "## Rolling",
            f"- pnl_7d_bps: {float(rolling7.get('pnl', 0.0)):+.2f}",
            f"- trades_7d: {int(rolling7.get('n', 0))}",
            f"- win_rate_7d: {float(rolling7.get('wr', 0.0))*100.0:.1f}%",
            "",
            "## Regime Distribution (from health history)",
            f"- up_pct: {reg_up:.1f}%",
            f"- down_pct: {reg_down:.1f}%",
            f"- unknown_pct: {reg_unk:.1f}%",
            "",
            "## Risk/Data",
            f"- blocked_signals: {metrics.blocked_count}",
            "",
            f"## Anomalies\n- {', '.join(anomalies) if anomalies else 'none'}",
            f"Anomalies: {', '.join(anomalies) if anomalies else 'none'}",
            "",
        ]
    )


def _send_telegram(md: str) -> None:
    try:
        import os
        from notifications.telegram import Notifier  # type: ignore

        token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN")
        chat = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID")
        if not token or not chat:
            return
        asyncio.run(Notifier(token=token, chat_id=chat).speak(md, priority="normal", silent=True))
    except Exception:
        return


def _collect_weekly(out_dir: Path, end_day_utc: str, max_days: int = 7) -> list[DailyMetrics]:
    days: list[DailyMetrics] = []
    end_dt = datetime.strptime(end_day_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    for i in range(max_days):
        d = (end_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        path = out_dir / f"{d}.json"
        if not path.exists():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            days.append(
                DailyMetrics(
                    day_utc=str(obj.get("day_utc") or d),
                    trade_count=int(obj.get("trade_count") or 0),
                    win_rate=float(obj.get("win_rate") or 0.0),
                    pnl_bps=float(obj.get("pnl_bps") or 0.0),
                    scratch_rate=float(obj.get("scratch_rate") or 0.0),
                    blocked_count=int(obj.get("blocked_count") or 0),
                    regime_up_sec=float(obj.get("regime_up_sec") or 0.0),
                    regime_down_sec=float(obj.get("regime_down_sec") or 0.0),
                    regime_unknown_sec=float(obj.get("regime_unknown_sec") or 0.0),
                )
            )
        except Exception:
            continue
    return sorted(days, key=lambda x: x.day_utc)


def run_once(args: argparse.Namespace) -> int:
    db = Path(args.db)
    if not db.exists():
        print(f"daily_report: missing db {db}")
        return 2
    metrics = compute_daily_metrics(trades_db=db)
    anomalies = detect_anomalies(
        metrics,
        history_db=db,
        expected_pnl_bps=float(args.expected_pnl_bps),
        expected_sigma_bps=float(args.expected_sigma_bps),
    )
    # Backward-compat anomaly names expected by existing tests
    if metrics.trade_count > 0 and metrics.win_rate < 0.35 and "low_win_rate" not in anomalies:
        anomalies.append("low_win_rate")
    if metrics.pnl_bps < -50.0 and "deep_negative_day" not in anomalies:
        anomalies.append("deep_negative_day")

    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        rolling7 = _summary(conn, days=7)
    finally:
        conn.close()
    md = _daily_markdown(metrics, anomalies, rolling7)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{metrics.day_utc}.md"
    md_path.write_text(md, encoding="utf-8")
    json_path = out_dir / f"{metrics.day_utc}.json"
    json_path.write_text(
        json.dumps(
            {
                "day_utc": metrics.day_utc,
                "trade_count": metrics.trade_count,
                "win_rate": metrics.win_rate,
                "pnl_bps": metrics.pnl_bps,
                "scratch_rate": metrics.scratch_rate,
                "blocked_count": metrics.blocked_count,
                "regime_up_sec": metrics.regime_up_sec,
                "regime_down_sec": metrics.regime_down_sec,
                "regime_unknown_sec": metrics.regime_unknown_sec,
                "anomalies": anomalies,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"daily_report: wrote {md_path}")
    if args.push:
        _send_telegram(md)

    # Sunday UTC weekly digest
    if bool(args.weekly):
        dt = datetime.strptime(metrics.day_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if dt.weekday() == 6:
            dailies = _collect_weekly(out_dir, metrics.day_utc, max_days=7)
            weekly_md = weekly_digest_markdown(dailies)
            weekly_path = out_dir / f"weekly_{metrics.day_utc}.md"
            weekly_path.write_text(weekly_md, encoding="utf-8")
            print(f"daily_report: wrote {weekly_path}")
            if args.push:
                _send_telegram(weekly_md)
    return 0


def _seconds_until_hhmm_utc(hhmm: str) -> float:
    hh, mm = [int(x) for x in str(hhmm).split(":", 1)]
    now = datetime.now(timezone.utc)
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return max(1.0, (target - now).total_seconds())


def run_schedule(args: argparse.Namespace) -> int:
    print(f"daily_report: schedule mode at UTC {args.at_utc}")
    while True:
        sleep_for = _seconds_until_hhmm_utc(str(args.at_utc))
        time.sleep(sleep_for)
        try:
            run_once(args)
        except Exception as e:
            print(f"daily_report: scheduled run error: {type(e).__name__}: {e}")


def main() -> int:
    args = _args()
    if args.schedule:
        return run_schedule(args)
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
