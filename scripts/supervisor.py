from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from utils.env_profile import load_dotenv_best_effort, paper_profile_active


load_dotenv_best_effort(root=Path(__file__).resolve().parents[1])


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
    if not paper_profile_active():
        print("[supervisor] refusing to run without paper profile / dry-run")
        return 2
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    restart_hist: deque[float] = deque()
    cmd = str(args.cmd)
    print(f"[supervisor] cmd={cmd}")
    last_shutdown = Path(args.cwd) / "logs" / "last_shutdown.json"
    while True:
        start = time.time()
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start: {cmd}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=args.cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out, _ = proc.communicate()
        rc = proc.returncode
        end = time.time()
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} exit rc={rc} runtime_sec={end-start:.1f}\n")
            if out:
                fh.write(out[-4000:])
                if not out.endswith("\n"):
                    fh.write("\n")
            if last_shutdown.exists():
                try:
                    payload = json.loads(last_shutdown.read_text(encoding="utf-8"))
                    fh.write(f"shutdown_meta={json.dumps(payload, ensure_ascii=True)}\n")
                except Exception:
                    fh.write("shutdown_meta=unreadable\n")
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
