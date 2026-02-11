#!/usr/bin/env python3
"""
Telemetry alert summary reporter.

Scans the generated telemetry artifacts, produces a short summary, and notifies
the Telegram channel whenever anomalies or pause states appear.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from collections import Counter


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _load_actions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_exit_quality_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            events.append(json.loads(s))
        except Exception:
            continue
    return events


def _guard_reason_lines(events: list[dict], limit: int = 6) -> list[str]:
    guard_events = {
        "entry.blocked",
        "exit.telemetry_guard",
        "order.create.retry_alert",
        "entry.partial_fill_escalation",
    }
    counts: Counter[str] = Counter()
    for ev in events:
        name = str(ev.get("event") or "")
        if name not in guard_events:
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        reason = (
            str(data.get("reason") or data.get("guard") or data.get("code") or "").strip().lower()
        )
        label = reason if reason else name
        counts[label] += 1
    if not counts:
        return []
    return [f"- {label}: {cnt}" for label, cnt in counts.most_common(limit)]


def _guard_symbol_lines(events: list[dict], limit: int = 6) -> list[str]:
    guard_events = {
        "entry.blocked",
        "exit.telemetry_guard",
        "order.create.retry_alert",
        "entry.partial_fill_escalation",
    }
    counts: Counter[str] = Counter()
    for ev in events:
        name = str(ev.get("event") or "")
        if name not in guard_events:
            continue
        symbol = str(ev.get("symbol") or "")
        if not symbol:
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            symbol = str(data.get("symbol") or "")
        symbol = symbol.strip().upper() if symbol else "UNKNOWN"
        counts[symbol] += 1
    if not counts:
        return []
    return [f"- {symbol}: {cnt}" for symbol, cnt in counts.most_common(limit)]


def _build_message(signal_lines, core_lines, anomaly_lines, signal_exit_lines) -> tuple[str, bool]:
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
    if signal_exit_lines:
        parts.append("\nSignal/exit context:")
        parts.extend(signal_exit_lines[:4])
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
    parser.add_argument("--exit-quality", default="logs/exit_quality.txt")
    parser.add_argument("--exit-quality-json", default="logs/exit_quality_summary.json")
    parser.add_argument("--exit-quality-win-drop", type=float, default=0.15)
    parser.add_argument("--exit-quality-pnl-drop", type=float, default=0.15)
    parser.add_argument("--telemetry", default="logs/telemetry.jsonl")
    parser.add_argument("--guard-reasons-top", type=int, default=6)
    parser.add_argument("--guard-symbols-top", type=int, default=6)
    parser.add_argument("--output", default="logs/telemetry_alert_summary.txt")
    parser.add_argument(
        "--actions-path",
        default="logs/telemetry_anomaly_actions.json",
        help="Path to anomaly mitigation metadata",
    )
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--signal-exit", default="logs/signal_exit_notify.txt")
    args = parser.parse_args(argv)

    signal_lines = _read_lines(Path(args.signal))
    core_lines = _read_lines(Path(args.core))
    anomaly_lines = _read_lines(Path(args.anomaly))
    exit_quality_lines = _read_lines(Path(args.exit_quality))
    exit_quality_json = _load_exit_quality_json(Path(args.exit_quality_json))
    signal_exit_lines = _read_lines(Path(args.signal_exit))
    actions = _load_actions(Path(args.actions_path))
    telemetry_events = _load_jsonl(Path(args.telemetry))

    message, warn = _build_message(signal_lines, core_lines, anomaly_lines, signal_exit_lines)
    if exit_quality_lines:
        message += "\n\nExit quality (top lines):\n" + "\n".join(exit_quality_lines[:6])
    if exit_quality_json:
        w24 = exit_quality_json.get("window_24h", {})
        w7 = exit_quality_json.get("window_7d", {})
        message += "\n\nExit window:"
        message += f"\n- 24h win rate {w24.get('win_rate', 0.0):.0%} avg pnl {w24.get('avg_pnl', 0.0):+.2f}"
        message += f"\n- 7d win rate {w7.get('win_rate', 0.0):.0%} avg pnl {w7.get('avg_pnl', 0.0):+.2f}"
        drift_notes = []
        win_drop = (w7.get("win_rate", 0.0) or 0.0) - (w24.get("win_rate", 0.0) or 0.0)
        if w7.get("win_rate", 0.0) > 0 and win_drop >= args.exit_quality_win_drop:
            drift_notes.append(f"win rate down {win_drop:.0%}")
            warn = True
        pnl7 = w7.get("avg_pnl", 0.0)
        pnl_drop = (pnl7 or 0.0) - (w24.get("avg_pnl", 0.0) or 0.0)
        if pnl7 > 0 and (pnl_drop / pnl7) >= args.exit_quality_pnl_drop:
            drift_notes.append(f"pnl down {pnl_drop:.2f}")
            warn = True
        if drift_notes:
            message += "\n" + " | ".join(drift_notes)
    guard_reason_lines = _guard_reason_lines(telemetry_events, max(1, args.guard_reasons_top))
    if guard_reason_lines:
        message += "\n\nTop guard reasons:\n" + "\n".join(guard_reason_lines)
    guard_symbol_lines = _guard_symbol_lines(telemetry_events, max(1, args.guard_symbols_top))
    if guard_symbol_lines:
        message += "\n\nGuard hits by symbol:\n" + "\n".join(guard_symbol_lines)
    action_notes = []
    now = time.time()
    pause_until = float(actions.get("pause_until", 0) or 0)
    if pause_until and pause_until > now:
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(pause_until))
        reason = str(actions.get("pause_reason") or "").strip()
        note = f"auto-pause until {ts}"
        if reason:
            note += f" ({reason})"
        action_notes.append(note)
    exit_mult = float(actions.get("classifier_exit_multiplier", 1.0))
    conf_mult = float(actions.get("classifier_confidence_multiplier", 1.0))
    if exit_mult != 1.0:
        action_notes.append(f"classifier exit mult x{exit_mult:.2f}")
    if conf_mult != 1.0:
        action_notes.append(f"classifier confidence mult x{conf_mult:.2f}")
    if actions.get("anomaly_messages"):
        msgs = actions.get("anomaly_messages")[:3]
        action_notes.append("anomalies: " + "; ".join(str(m) for m in msgs))
    if action_notes:
        message += "\n\nAuto mitigation:\n" + "\n".join(action_notes)
    Path(args.output).write_text(message + "\n", encoding="utf-8")
    print(message)

    if warn and not args.no_notify:
        notifier = _build_notifier()
        _send_alert(message, notifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
