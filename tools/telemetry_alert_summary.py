#!/usr/bin/env python3
"""
Telemetry alert summary reporter.

Scans the generated telemetry artifacts, produces a short summary, and notifies
the Telegram channel whenever anomalies or pause states appear.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _build_message(signal_lines, core_lines, anomaly_lines) -> tuple[str, bool]:
    warn = False
    parts = []
    if anomaly_lines:
        parts.append("Anomaly report:")
        parts.extend(anomaly_lines)
        warn = warn or any("Anomalies detected" in l for l in anomaly_lines)
        warn = warn or any("pause" in l.lower() for l in anomaly_lines)
    if signal_lines:
        parts.append("\nSignal data health (top lines):")
        parts.extend(signal_lines[:4])
    if core_lines:
        parts.append("\nCore health (summary):")
        parts.extend(core_lines[:4])
    if not parts:
        parts.append("Telemetry artifacts were empty.")
    return "\n".join(parts), warn


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


def _send_alert(message: str, notifier) -> None:
    if notifier is None or not message.strip():
        return
    try:
        asyncio.run(notifier.speak(message, "critical"))
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry alert summary (artifact -> Telegram)")
    parser.add_argument("--signal", default="logs/signal_data_health.txt")
    parser.add_argument("--core", default="logs/core_health.txt")
    parser.add_argument("--anomaly", default="logs/telemetry_anomaly.txt")
    parser.add_argument("--output", default="logs/telemetry_alert_summary.txt")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args(argv)

    signal_lines = _read_lines(Path(args.signal))
    core_lines = _read_lines(Path(args.core))
    anomaly_lines = _read_lines(Path(args.anomaly))

    message, warn = _build_message(signal_lines, core_lines, anomaly_lines)
    Path(args.output).write_text(message + "\n", encoding="utf-8")
    print(message)

    if warn and not args.no_notify:
        notifier = _build_notifier()
        _send_alert(message, notifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
