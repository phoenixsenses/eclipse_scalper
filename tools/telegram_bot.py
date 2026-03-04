from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
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
    return (
        "PnL Snapshot\n"
        f"Today: {float(p.get('today_pnl_bps', 0.0)):+.2f} bps | Trades: {int(p.get('today_trades', 0))} | Win: {float(p.get('today_win_rate', 0.0))*100.0:.1f}%\n"
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


def _dispatch(cmd: str) -> str:
    c = str(cmd or "").strip().lower()
    if c.startswith("/status"):
        return render_status_text()
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
    if c.startswith("/help"):
        return "Commands: /status /pnl /last /open /config /diag /help"
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
