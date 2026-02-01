#!/usr/bin/env python3
"""
Telemetry dashboard notifier.

Runs the dashboard helper, prints the snapshot for the workflow log, and
sends the same summary over Telegram when `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID`
are configured (e.g., via GitHub Secrets).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TELEMETRY_SCRIPT = ROOT / "eclipse_scalper" / "tools" / "telemetry_dashboard.py"


def _build_notifier():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return None
    try:
        from eclipse_scalper.notifications.telegram import Notifier

        return Notifier(token=token, chat_id=chat_id)
    except Exception:
        return None


def _send_alert(notifier, text: str) -> None:
    if notifier is None or not text.strip():
        return
    try:
        asyncio.run(notifier.speak(text, "critical"))
    except Exception:
        pass


def _truncate(text: str, max_len: int = 3800) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    suffix = " …"
    return text[: max_len - len(suffix)] + suffix


def _run_dashboard(path: Path, limit: int, codes_per_symbol: bool, codes_top: int) -> tuple[str, int]:
    cmd = [
        sys.executable or "python",
        str(TELEMETRY_SCRIPT),
        "--path",
        str(path),
        "--limit",
        str(limit),
    ]
    if codes_per_symbol:
        cmd.append("--codes-per-symbol")
        cmd.extend(["--codes-top", str(codes_top)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    return stdout, result.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry dashboard notifier")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--limit", type=int, default=100, help="Max events to load")
    parser.add_argument("--codes-per-symbol", action="store_true", help="Include per-symbol code summary")
    parser.add_argument("--codes-top", type=int, default=4, help="Top per-symbol codes when enabled")
    parser.add_argument("--no-notify", action="store_true", help="Skip sending the Telegram notification")
    args = parser.parse_args(argv)

    telemetry_path = Path(args.path)
    stdout, rc = _run_dashboard(telemetry_path, max(1, args.limit), args.codes_per_symbol, max(1, args.codes_top))
    if args.no_notify:
        return rc

    notifier = _build_notifier()
    if notifier and stdout:
        header = f"Telemetry snapshot: {telemetry_path.name}"
        max_body_len = max(0, 3600 - len(header))
        body = _truncate(stdout, max_len=max_body_len)
        _send_alert(notifier, f"{header}\n{body}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
