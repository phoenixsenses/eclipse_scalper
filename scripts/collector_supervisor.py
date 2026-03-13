"""Eclipse Scalper -- Collector Supervisor

Manages microstructure_collector and event_diary as subprocesses.
Restarts automatically on crash. Sends Telegram alert on each restart.

Usage:
    python scripts/collector_supervisor.py
    python scripts/collector_supervisor.py --cwd "C:\\path\\to\\eclipse_scalper"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
import json
from logging.handlers import RotatingFileHandler
import logging
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Managed process definitions
# ---------------------------------------------------------------------------

PROCS: List[Dict] = [
    {
        "name": "MicroCollector",
        "module": "data.microstructure_collector",
        "args": ["--symbols", "BTCUSDT,ETHUSDT",
                 "--db-path", "data/microstructure.db",
                 "--stats-interval", "300"],
        "restart_log": "logs/collector_restarts.log",
    },
    {
        "name": "EventDiary",
        "module": "data.event_diary",
        "args": ["--db-path", "data/microstructure.db",
                 "--csv-path", "data/event_diary.csv"],
        "restart_log": "logs/diary_restarts.log",
    },
]

RESTART_DELAY_SEC = 5
FAST_RESTART_WINDOW_SEC = 120   # seconds
FAST_RESTART_MAX = 5            # restarts within window = storm
BACKOFF_DELAY_SEC = 60
CHECK_INTERVAL_SEC = 10


def _pid_dir(cwd: Path) -> Path:
    p = cwd / "logs" / "pids"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pid_file(cwd: Path, module: str) -> Path:
    name = "microstructure_collector.pid" if module == "data.microstructure_collector" else "event_diary.pid"
    return _pid_dir(cwd) / name


def _meta_file(cwd: Path, module: str) -> Path:
    name = "microstructure_collector.json" if module == "data.microstructure_collector" else "event_diary.json"
    return _pid_dir(cwd) / name


def _write_json_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_pid_state(*, cwd: Path, module: str, pid: int, launcher: str, python: str, args: List[str]) -> None:
    pid_path = _pid_file(cwd, module)
    meta_path = _meta_file(cwd, module)
    pid_path.write_text(str(int(pid)), encoding="ascii")
    _write_json_atomic(
        meta_path,
        {
            "role": "data_layer",
            "pid": int(pid),
            "module": str(module),
            "cmdline_sig": " ".join([str(python), "-W", "ignore", "-u", "-m", str(module)] + list(args)),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "launcher": str(launcher),
            "cwd": str(cwd),
        },
    )


def _clear_pid_state_if_matches(*, cwd: Path, module: str, pid: Optional[int]) -> None:
    if not pid:
        return
    pid_path = _pid_file(cwd, module)
    meta_path = _meta_file(cwd, module)
    try:
        raw = pid_path.read_text(encoding="utf-8", errors="ignore").strip()
        if raw and int(raw) == int(pid):
            pid_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(payload, dict) and int(payload.get("pid") or 0) == int(pid):
            meta_path.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("collector_supervisor")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = RotatingFileHandler(str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ---------------------------------------------------------------------------
# Telegram helper
# ---------------------------------------------------------------------------

def _tg_token_chat():
    token = os.getenv("ECLIPSE_TG_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("ECLIPSE_TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    return token, chat_id


async def _send_telegram(text: str) -> None:
    token, chat_id = _tg_token_chat()
    if not token or not chat_id:
        return
    try:
        from notifications.telegram import Notifier
        await Notifier(token=token, chat_id=chat_id).speak(text, priority="critical")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Managed process
# ---------------------------------------------------------------------------

class ManagedProcess:
    def __init__(self, cfg: Dict, python: str, cwd: Path, log: logging.Logger):
        self.name: str = cfg["name"]
        self.module: str = cfg["module"]
        self.extra_args: List[str] = cfg["args"]
        self.restart_log: Path = cwd / cfg["restart_log"]
        self.python = python
        self.cwd = cwd
        self.log = log
        self.proc: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.restart_times: List[float] = []

    def _cmd(self) -> List[str]:
        return [self.python, "-W", "ignore", "-u", "-m", self.module] + self.extra_args

    def start(self) -> None:
        self.restart_log.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(self._cmd(), cwd=str(self.cwd))
        self.restart_count += 1
        now = time.time()
        self.restart_times.append(now)
        self.restart_times = [t for t in self.restart_times
                              if now - t <= FAST_RESTART_WINDOW_SEC]
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.restart_log.open("a", encoding="utf-8") as f:
            f.write(f"{self.name} started at {ts} (restart #{self.restart_count})\n")
        _write_pid_state(
            cwd=self.cwd,
            module=self.module,
            pid=int(self.proc.pid),
            launcher="collector_supervisor",
            python=self.python,
            args=self.extra_args,
        )
        self.log.info(f"[{self.name}] started pid={self.proc.pid}")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self) -> Optional[int]:
        return self.proc.poll() if self.proc else None

    def is_crash_looping(self) -> bool:
        now = time.time()
        recent = [t for t in self.restart_times if now - t <= FAST_RESTART_WINDOW_SEC]
        return len(recent) >= FAST_RESTART_MAX


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(cwd: Path) -> None:
    venv_py = cwd / ".venv" / "Scripts" / "python.exe"
    python = str(venv_py) if venv_py.exists() else sys.executable

    log = _setup_logger(cwd / "logs" / "collector_supervisor.log")
    log.info(f"Collector supervisor started cwd={cwd} python={python}")

    # Load .env.paper if present
    env_file = cwd / ".env.paper"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    procs = [ManagedProcess(cfg, python, cwd, log) for cfg in PROCS]

    # Start all
    for mp in procs:
        mp.start()
        await asyncio.sleep(1)

    pids = " | ".join(f"{mp.name}={mp.proc.pid}" for mp in procs)
    await _send_telegram(f"[SUPERVISOR] Collector supervisor started\n{pids}")

    while True:
        await asyncio.sleep(CHECK_INTERVAL_SEC)
        for mp in procs:
            if not mp.is_alive():
                code = mp.exit_code()
                log.warning(f"[{mp.name}] DIED exit_code={code} total_restarts={mp.restart_count}")

                if mp.is_crash_looping():
                    msg = (
                        f"[SUPERVISOR] {mp.name} crash-loop detected "
                        f"({FAST_RESTART_MAX}+ deaths in {FAST_RESTART_WINDOW_SEC}s). "
                        f"Backing off {BACKOFF_DELAY_SEC}s."
                    )
                    log.error(msg)
                    await _send_telegram(msg)
                    await asyncio.sleep(BACKOFF_DELAY_SEC)
                else:
                    msg = f"[SUPERVISOR] {mp.name} died (exit={code}). Restarting in {RESTART_DELAY_SEC}s..."
                    log.warning(msg)
                    await _send_telegram(msg)
                    await asyncio.sleep(RESTART_DELAY_SEC)

                mp.start()
                await _send_telegram(f"[SUPERVISOR] {mp.name} restarted (restart #{mp.restart_count})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cwd", default=str(Path(__file__).resolve().parent.parent))
    args = p.parse_args()
    cwd = Path(args.cwd).resolve()
    # Add cwd to sys.path so project modules are importable
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    try:
        asyncio.run(run(cwd))
    except KeyboardInterrupt:
        print("Collector supervisor stopped.")
    finally:
        for cfg in PROCS:
            try:
                _clear_pid_state_if_matches(cwd=cwd, module=str(cfg["module"]), pid=None)
            except Exception:
                pass


if __name__ == "__main__":
    main()
