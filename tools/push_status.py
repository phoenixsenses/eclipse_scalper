from __future__ import annotations

import argparse
import asyncio
import os
import sys
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
from monitoring.status_snapshot import render_status_text
from notifications.telegram import Notifier


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Send one-shot status snapshot to Telegram.")
    p.add_argument("--silent", action="store_true", help="Send silent notification.")
    return p


async def _run(silent: bool) -> int:
    token = (
        os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_TOKEN")
        or os.getenv("ECLIPSE_TG_BOT_TOKEN")
    )
    chat = os.getenv("TELEGRAM_CHAT_ID")
    text = render_status_text()
    if not chat:
        print("push_status: TELEGRAM_CHAT_ID is required for push mode")
        print(text)
        return 2
    if not token:
        print("push_status: missing TELEGRAM_TOKEN/TELEGRAM_BOT_TOKEN")
        print(text)
        return 2
    n = Notifier(token=token, chat_id=chat)
    ok = await n.speak(text, priority="normal", silent=bool(silent))
    if not ok:
        print("push_status: telegram send failed")
        return 3
    print("status_sent")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_run(silent=bool(args.silent)))


if __name__ == "__main__":
    raise SystemExit(main())
