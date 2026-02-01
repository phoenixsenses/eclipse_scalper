#!/usr/bin/env python3
"""
Exit telemetry threshold alerts with notifier support.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_scalper.tools import telemetry_threshold_alerts as tta


def _build_notifier():
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None
    try:
        from notifications.telegram import Notifier

        return Notifier(token=token, chat_id=chat_id)
    except Exception:
        return None


def _send_alert(notifier, text: str) -> None:
    if notifier is None:
        return
    try:
        asyncio.run(notifier.speak(text, "critical"))
    except Exception:
        pass


def _format_summary(
    counts: Dict[str, Dict[str, int]],
    triggered: list[Tuple[str, int, int]],
    since_min: int,
) -> str:
    lines = [f"Exit thresholds (last {since_min} min):"]
    for name, data in sorted(counts.items(), key=lambda kv: kv[1].get("total", 0), reverse=True):
        total = data.get("total", 0)
        symbols = ", ".join(f"{sym}={cnt}" for sym, cnt in sorted(data.items()) if sym not in ("total",) and cnt > 0)
        lines.append(f"- {name}: total={total} {symbols or ''}")
    if triggered:
        lines.append("Alerts triggered:")
        for name, total, threshold in triggered:
            lines.append(f"  * {name}: {total} >= {threshold}")
    return "\n".join(lines)


def _parse_thresholds(raw: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not raw:
        return out
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if "=" not in part:
            continue
        name, val = part.split("=", 1)
        try:
            out[name.strip()] = max(1, int(float(val.strip())))
        except Exception:
            continue
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Exit telemetry notifier (threshold alerts)")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--limit", type=int, default=5000, help="Max events to load")
    parser.add_argument("--since-min", type=int, default=30, help="Window in minutes to scan")
    parser.add_argument("--thresholds", default="", help="Override preset thresholds")
    parser.add_argument("--events", default="", help="Comma-separated events to watch (optional override)")
    args = parser.parse_args(argv)

    merged_thresholds: Dict[str, int] = dict(tta.PRESET_THRESHOLDS.get("exit", {}))
    overrides = _parse_thresholds(args.thresholds)
    merged_thresholds.update(overrides)
    monitor: set[str] = set(merged_thresholds.keys())
    if args.events.strip():
        monitor = {e.strip() for e in args.events.split(",") if e.strip()}

    counts, triggered = tta.scan_thresholds(
        path=Path(args.path),
        limit=max(1, int(args.limit)),
        since_min=args.since_min,
        thresholds=merged_thresholds,
        monitor=monitor,
    )

    if not counts:
        print("No matching telemetry events for exit preset")
        return 0

    summary = _format_summary(counts, triggered, args.since_min)
    print(summary)
    if triggered:
        notifier = _build_notifier()
        _send_alert(notifier, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
