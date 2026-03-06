from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set


def _load_dotenv_best_effort() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    try:
        env_paper = ".env.paper"
        env_default = ".env"
        if os.path.exists(env_paper):
            load_dotenv(dotenv_path=env_paper, override=False)
        elif os.path.exists(env_default):
            load_dotenv(dotenv_path=env_default, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        return


_load_dotenv_best_effort()
from telegram import Bot
from core.chart_generator import build_equity_curve_png

from monitoring.status_snapshot import (
    collect_config_flags,
    collect_diag,
    collect_last_decisions,
    collect_open_positions,
    collect_pnl,
    render_status_text,
)


def _allowed_chat_ids() -> Set[str]:
    raw = os.getenv("ALLOWED_CHAT_IDS", "").strip()
    if not raw:
        fallback = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID") or ""
        raw = fallback
    return {x.strip() for x in raw.replace(";", ",").split(",") if x.strip()}


def _is_allowed(chat_id: int | str, allowed: Set[str]) -> bool:
    if not allowed:
        return True
    return str(chat_id) in allowed


def _render_pnl() -> str:
    p = collect_pnl()
    if not p.get("ok"):
        return "PnL unavailable (paper_trades.db missing)"
    p7 = collect_pnl_rolling(days=7)
    return (
        "PnL Snapshot\n"
        f"Today: {float(p.get('today_pnl_bps', 0.0)):+.2f} bps | Trades: {int(p.get('today_trades', 0))} | Win: {float(p.get('today_win_rate', 0.0))*100.0:.1f}%\n"
        f"7d: {float(p7.get('pnl_bps', 0.0)):+.2f} bps | Trades: {int(p7.get('trades', 0))} | Win: {float(p7.get('win_rate', 0.0))*100.0:.1f}%\n"
        f"Total: {float(p.get('total_pnl_bps', 0.0)):+.2f} bps | Trades: {int(p.get('total_trades', 0))} | Win: {float(p.get('total_win_rate', 0.0))*100.0:.1f}%\n"
        f"Max DD: -{float(p.get('max_drawdown_bps', 0.0)):.2f} bps"
    )


def _render_last(limit: int = 5) -> str:
    rows = collect_last_decisions(limit=limit)
    if not rows:
        return "No recent decisions in execution_journal."
    out = ["Last decisions:"]
    for r in rows:
        out.append(f"- {r.get('event')} {r.get('symbol','')} reason={r.get('reason','')} ts={r.get('ts','')}")
    return "\n".join(out)


def _render_open() -> str:
    p = collect_open_positions()
    if not p.get("ok"):
        return "Open positions unavailable (state/brain.json missing or invalid)"
    if int(p.get("count", 0)) <= 0:
        return "Open positions: 0"
    lines = [f"Open positions: {int(p.get('count', 0))}"]
    for row in p.get("positions", []):
        lines.append(
            f"- {row.get('symbol')} {str(row.get('side','')).upper()} qty={float(row.get('qty',0.0)):.6f} entry={float(row.get('entry_price',0.0)):.4f}"
        )
    return "\n".join(lines)


def _render_config() -> str:
    c = collect_config_flags()
    lines = ["Config flags:"]
    for k in sorted(c):
        lines.append(f"- {k}={c[k]}")
    return "\n".join(lines)


def _render_diag() -> str:
    d = collect_diag()
    if not d.get("ok"):
        return "Diag: microstructure feed unavailable"
    return (
        "Diagnostics\n"
        f"- micro_db: {d.get('micro_db')}\n"
        f"- last_tick: {d.get('last_tick_utc')}\n"
        f"- feed_age_sec: {int(d.get('feed_age_sec', 0))}"
    )


def collect_pnl_rolling(days: int = 7, db_path: str = "data/paper_trades.db") -> Dict[str, float]:
    p = Path(str(db_path))
    if not p.exists():
        return {"trades": 0, "pnl_bps": 0.0, "win_rate": 0.0}
    cutoff = time.time() - max(1, int(days)) * 86400.0
    conn = sqlite3.connect(str(p), check_same_thread=False)
    try:
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(pnl_bps),0) pnl, "
            "COALESCE(AVG(CASE WHEN pnl_bps>0 THEN 1.0 ELSE 0.0 END),0) wr "
            "FROM trades WHERE exit_time>=?",
            (cutoff,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {"trades": 0, "pnl_bps": 0.0, "win_rate": 0.0}
    return {"trades": int(row[0] or 0), "pnl_bps": float(row[1] or 0.0), "win_rate": float(row[2] or 0.0)}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _feature_health(symbol: str = "ETHUSDT", db_path: str = "data/microstructure.db") -> Dict[str, float]:
    db = Path(db_path)
    if not db.exists():
        return {"ok": 0.0}
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 30_000
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        mark = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        trades_30 = conn.execute(
            "SELECT COUNT(*) n FROM agg_trades WHERE symbol=? AND ts_ms>=?",
            (symbol, start_ms),
        ).fetchone()
    finally:
        conn.close()
    if not mark:
        return {"ok": 0.0}
    age_sec = max(0.0, (float(now_ms) - float(mark["ts_ms"] or now_ms)) / 1000.0)
    return {
        "ok": 1.0,
        "age_sec": age_sec,
        "mark_price": float(mark["mark_price"] or 0.0),
        "trades_30s": float((trades_30[0] if trades_30 else 0) or 0.0),
    }


def _regime_history_stats(path: Path = Path("logs/health/history.jsonl")) -> Dict[str, float]:
    if not path.exists():
        return {"ok": 0.0}
    now = time.time()
    start = now - 86400.0
    entries: list[tuple[float, str]] = []
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            ts = float(obj.get("ts") or 0.0)
            if ts < start:
                continue
            reg = str(obj.get("regime") or obj.get("regime_state") or "UNKNOWN").upper()
            entries.append((ts, reg))
    except Exception:
        return {"ok": 0.0}
    if not entries:
        return {"ok": 0.0}
    transitions = 0
    durations: list[float] = []
    prev_reg = entries[0][1]
    prev_ts = entries[0][0]
    for ts, reg in entries[1:]:
        if reg != prev_reg:
            transitions += 1
            durations.append(max(0.0, ts - prev_ts))
            prev_ts = ts
            prev_reg = reg
    if entries:
        durations.append(max(0.0, now - prev_ts))
    avg_dur = (sum(durations) / len(durations)) if durations else 0.0
    return {
        "ok": 1.0,
        "transitions": float(transitions),
        "avg_duration_sec": avg_dur,
        "current_regime": float(0),  # placeholder for compatibility
    }


def _render_regime() -> str:
    p = Path("logs/health/overall.json")
    if not p.exists():
        return "Regime info unavailable (logs/health/overall.json missing)"
    obj = _read_json(p)
    if not obj:
        return "Regime info unavailable (invalid health json)"
    reg = str(obj.get("regime") or obj.get("regime_state") or "UNKNOWN")
    rr = float(obj.get("rolling_return_1h", obj.get("rolling_return", 0.0)) or 0.0)
    age = float(obj.get("regime_age_sec", 0.0) or 0.0)
    conf = min(1.0, abs(rr) / 0.01)  # 1% 1h move ~= high regime confidence
    hist = _regime_history_stats()
    return (
        "Regime\n"
        f"- state: {reg}\n"
        f"- rolling_return_1h: {rr*100.0:+.3f}%\n"
        f"- age_sec: {int(age)}\n"
        f"- confidence_proxy: {conf:.2f}\n"
        f"- transitions_today: {int(hist.get('transitions', 0))}\n"
        f"- avg_regime_duration_sec: {int(hist.get('avg_duration_sec', 0.0))}"
    )


def _render_health() -> str:
    p = Path("logs/health/overall.json")
    if not p.exists():
        return "Health unavailable (overall.json missing)"
    obj = _read_json(p)
    if not obj:
        return "Health unavailable (invalid json)"
    collector = _read_json(Path("logs/collector_heartbeat.json"))
    risk = _read_json(Path("data/live/risk_snapshot.json"))
    live_hb = _read_json(Path("data/live/heartbeat.json"))
    free_gb = float(shutil.disk_usage(str(Path(".").resolve())).free) / (1024.0 ** 3)
    micro_db = Path("data/microstructure.db")
    wal = Path("data/microstructure.db-wal")
    db_size_mb = (float(micro_db.stat().st_size) / (1024.0 ** 2)) if micro_db.exists() else 0.0
    wal_size_mb = (float(wal.stat().st_size) / (1024.0 ** 2)) if wal.exists() else 0.0
    feature = _feature_health(symbol=str(os.getenv("MICRO_SIGNAL_SYMBOL", "ETHUSDT")))
    proc_uptime_sec = -1.0
    try:
        import psutil  # type: ignore

        proc_uptime_sec = max(0.0, time.time() - float(psutil.Process(os.getpid()).create_time()))
    except Exception:
        proc_uptime_sec = -1.0
    last_trade_age_sec = -1.0
    tdb = Path("data/paper_trades.db")
    if tdb.exists():
        conn = sqlite3.connect(str(tdb), check_same_thread=False)
        try:
            row = conn.execute("SELECT MAX(exit_time) FROM trades").fetchone()
        finally:
            conn.close()
        if row and row[0]:
            last_trade_age_sec = max(0.0, time.time() - float(row[0]))
    st = str(obj.get("state") or "unknown")
    reason = str(obj.get("reason") or "")
    ts = str(obj.get("ts_utc") or "")
    comps = obj.get("components") if isinstance(obj.get("components"), dict) else {}
    lines = [f"Health\n- state: {st}\n- reason: {reason}\n- ts_utc: {ts}"]
    for k in sorted(comps):
        c = comps.get(k) if isinstance(comps.get(k), dict) else {}
        lines.append(
            f"- {k}: status={c.get('status')} lag={c.get('progress_lag_sec')} "
            f"errors_5m={c.get('errors_last_5m')} reconnects_5m={c.get('reconnects_last_5m')}"
        )
    lines.append(
        f"- collector_connected={int(bool(collector.get('connected', False)))} "
        f"collector_backoff_sec={float(collector.get('current_backoff_seconds', 0.0) or 0.0):.2f}"
    )
    lines.append(f"- micro_db_size_mb={db_size_mb:.2f} wal_size_mb={wal_size_mb:.2f} disk_free_gb={free_gb:.2f}")
    lines.append(f"- feature_age_sec={int(feature.get('age_sec', 0.0))} trades_30s={int(feature.get('trades_30s', 0.0))}")
    if last_trade_age_sec >= 0:
        lines.append(f"- last_trade_age_sec={int(last_trade_age_sec)}")
    if risk:
        lines.append(
            f"- risk_drawdown_pct={float(risk.get('drawdown_pct', 0.0) or 0.0):.4f} "
            f"kill_active={int(bool(risk.get('kill_active', False)))}"
        )
    if live_hb:
        lines.append(f"- live_heartbeat_ts={live_hb.get('ts_utc', '')}")
    if proc_uptime_sec >= 0:
        lines.append(f"- process_uptime_sec={int(proc_uptime_sec)}")
    return "\n".join(lines)


def _make_chart_png(path: Path) -> bool:
    return bool(build_equity_curve_png("data/paper_trades.db", path, days=7))


def _render_status_plus() -> str:
    base = render_status_text()
    pnl = _render_pnl()
    regime = _render_regime()
    health = _render_health()
    last = _render_last(limit=5)
    return "\n\n".join([base, pnl, regime, health, last])


def _dispatch(cmd: str) -> str:
    c = str(cmd or "").strip().lower()
    if c.startswith("/status"):
        return _render_status_plus()
    if c.startswith("/pnl"):
        return _render_pnl()
    if c.startswith("/last"):
        return _render_last(limit=5)
    if c.startswith("/open"):
        return _render_open()
    if c.startswith("/config"):
        return _render_config()
    if c.startswith("/diag"):
        return _render_diag()
    if c.startswith("/regime"):
        return _render_regime()
    if c.startswith("/health"):
        return _render_health()
    if c.startswith("/help"):
        return "Commands: /status /pnl /last /open /config /diag /regime /health /chart /help"
    return "Unknown command. Use /help."


async def run_bot(poll_sec: float = 2.0) -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN")
    if not token:
        print("telegram_bot: missing TELEGRAM_BOT_TOKEN/TELEGRAM_TOKEN/ECLIPSE_TG_BOT_TOKEN")
        return 2
    bot = Bot(token=token)
    allowed = _allowed_chat_ids()
    offset: Optional[int] = None
    print("telegram_bot: online (long polling)")
    while True:
        try:
            updates = await bot.get_updates(offset=offset, timeout=20)
            for u in updates:
                offset = int(u.update_id) + 1
                msg = getattr(u, "message", None)
                if msg is None:
                    continue
                text = str(getattr(msg, "text", "") or "").strip()
                chat = getattr(msg, "chat", None)
                chat_id = getattr(chat, "id", None)
                if chat_id is None:
                    continue
                if not _is_allowed(chat_id, allowed):
                    continue
                if not text.startswith("/"):
                    continue
                if text.lower().startswith("/chart"):
                    png = Path("reports") / "telegram_chart.png"
                    ok = _make_chart_png(png)
                    if not ok:
                        await bot.send_message(chat_id=chat_id, text="<pre>Chart unavailable.</pre>", parse_mode="HTML")
                    else:
                        with png.open("rb") as fh:
                            await bot.send_photo(chat_id=chat_id, photo=fh, caption="Paper equity chart")
                    continue
                reply = _dispatch(text)
                await bot.send_message(chat_id=chat_id, text=f"<pre>{reply}</pre>", parse_mode="HTML")
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(f"telegram_bot: loop error: {type(exc).__name__}: {exc}")
            await asyncio.sleep(max(0.5, float(poll_sec)))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Telegram command bot for Eclipse status dashboard.")
    p.add_argument("--poll-sec", type=float, default=2.0, help="Backoff sleep on polling errors.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(run_bot(poll_sec=float(args.poll_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
