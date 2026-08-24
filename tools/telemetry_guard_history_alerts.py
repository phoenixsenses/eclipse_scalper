#!/usr/bin/env python3
"""
Telemetry guard history alerts.

Scans the guard history CSV for partial-fill / retry overrides over a recent window
and raises a notification (console + optional Telegram) when the count exceeds a threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path
from typing import Any


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


async def _send_alert(text: str, severity: str) -> None:
    notifier = _build_notifier()
    if notifier is None or not text.strip():
        return
    try:
        await notifier.speak(text, severity)
    except Exception:
        pass


def _keyword_hit(row: dict[str, str], keywords: tuple[str, ...]) -> bool:
    for col in ("override_reason", "recent_guard_hit", "signal_context"):
        value = (row.get(col) or "").lower()
        if any(k in value for k in keywords):
            return True
    return False


def _read_rows(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows[-limit:]


def _emit_event(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _write_actions(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Guard history partial/retry alert")
    parser.add_argument("--path", default="logs/telemetry_guard_history.csv", help="Guard history CSV")
    parser.add_argument("--window", type=int, default=8, help="Rows to inspect")
    parser.add_argument("--threshold", type=int, default=3, help="Trigger when hits exceed this")
    parser.add_argument("--hit-rate", type=float, default=0.3, help="Trigger when hit ratio >= this")
    parser.add_argument(
        "--keywords",
        default="partial_fill,retry_alert",
        help="Comma-separated keywords to search within guard columns",
    )
    parser.add_argument("--emit-event", action="store_true", help="Append telemetry.guard_history_spike event")
    parser.add_argument(
        "--event-path",
        default="logs/telemetry_guard_history_events.jsonl",
        help="Path to append guard history spike events",
    )
    parser.add_argument(
        "--actions-path",
        default="logs/telemetry_guard_history_actions.json",
        help="Write adaptive guard actions JSON",
    )
    parser.add_argument("--conf-delta", type=float, default=0.1, help="Min confidence delta on trigger")
    parser.add_argument("--conf-duration", type=float, default=900, help="Min confidence override duration (sec)")
    parser.add_argument("--leverage-scale", type=float, default=0.85, help="Leverage multiplier on trigger")
    parser.add_argument("--leverage-duration", type=float, default=900, help="Leverage override duration (sec)")
    parser.add_argument("--notional-scale", type=float, default=0.8, help="Notional multiplier on trigger")
    parser.add_argument("--notional-duration", type=float, default=900, help="Notional override duration (sec)")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification")
    args = parser.parse_args(argv)

    keywords = tuple(kw.strip().lower() for kw in args.keywords.split(",") if kw.strip())
    rows = _read_rows(Path(args.path), args.window)
    hits = [row for row in rows if _keyword_hit(row, keywords)]

    summary_lines = [
        f"Telemetry guard history window last {len(rows)} rows:",
        f"- partial/retry hits: {len(hits)} (threshold {args.threshold})",
        f"- hit rate: {len(rows) and len(hits)/len(rows):.2%} threshold {args.hit_rate:.0%}",
        f"- keywords: {', '.join(keywords)}",
    ]
    if hits:
        summary_lines.append("- recent overrides:")
        for row in hits[-3:]:
            summary_lines.append(
                f"  - {row.get('timestamp')} override={row.get('override_reason')} recent={row.get('recent_guard_hit')}"
            )

    summary = "\n".join(summary_lines)
    print(summary)

    rate = (len(rows) and len(hits) / len(rows)) or 0.0
    triggered = len(hits) >= args.threshold or rate >= args.hit_rate
    if triggered:
        now = time.time()
        action_payload = {
            "ts": now,
            "event": "telemetry.guard_history_spike",
            "hit_count": len(hits),
            "hit_rate": rate,
            "window": len(rows),
            "threshold": args.threshold,
            "hit_rate_threshold": args.hit_rate,
            "keywords": list(keywords),
            "confidence_delta": float(args.conf_delta),
            "confidence_duration": float(args.conf_duration),
            "leverage_scale": float(args.leverage_scale),
            "leverage_duration": float(args.leverage_duration),
            "notional_scale": float(args.notional_scale),
            "notional_duration": float(args.notional_duration),
        }
        _write_actions(Path(args.actions_path), action_payload)
        if args.emit_event:
            _emit_event(
                Path(args.event_path),
                {"ts": now, "event": "telemetry.guard_history_spike", "data": action_payload},
            )
        if args.notify:
            asyncio.run(_send_alert(summary, "warning"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
