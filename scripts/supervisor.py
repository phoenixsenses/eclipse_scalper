from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def _load_dotenv_best_effort() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    root = Path(__file__).resolve().parents[1]
    env_paper = root / ".env.paper"
    env_default = root / ".env"
    try:
        if env_paper.exists():
            load_dotenv(dotenv_path=env_paper, override=False)
        elif env_default.exists():
            load_dotenv(dotenv_path=env_default, override=False)
    except Exception:
        return


_load_dotenv_best_effort()


def _send_alert(msg: str) -> None:
    try:
        from notifications.telegram import Notifier  # type: ignore
    except Exception:
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        import asyncio

        asyncio.run(Notifier(token=token, chat_id=chat_id).speak(msg, priority="critical"))
    except Exception:
        return


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-trading supervisor with restart guard.")
    p.add_argument("--cmd", default=f"{sys.executable} -m execution.bootstrap")
    p.add_argument("--restart-delay-sec", type=float, default=30.0)
    p.add_argument("--max-restarts-per-hour", type=int, default=5)
    p.add_argument("--cwd", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--log", default="logs/supervisor.log")
    return p.parse_args()


def main() -> int:
    args = _args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    restart_hist: deque[float] = deque()
    cmd = str(args.cmd)
    print(f"[supervisor] cmd={cmd}")
    while True:
        start = time.time()
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start: {cmd}\n")
        proc = subprocess.Popen(cmd, cwd=args.cwd, shell=True)
        rc = proc.wait()
        end = time.time()
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} exit rc={rc} runtime_sec={end-start:.1f}\n")
        _send_alert(f"SUPERVISOR: bootstrap exited rc={rc} runtime={end-start:.1f}s")
        now = time.time()
        while restart_hist and (now - restart_hist[0]) > 3600.0:
            restart_hist.popleft()
        restart_hist.append(now)
        if len(restart_hist) > int(max(1, args.max_restarts_per_hour)):
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} halt: max restarts exceeded\n")
            print("[supervisor] max restarts/hour exceeded; exiting")
            return 2
        time.sleep(max(1.0, float(args.restart_delay_sec)))


if __name__ == "__main__":
    raise SystemExit(main())

