#!/usr/bin/env python3
"""
Telemetry alert classifier.

Tails the latest telemetry snapshot, tracks state between runs, and notifies
via Telegram when exit counts spike, previously unseen error/exit codes appear,
or the averaged confidence falls sharply.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_DEFAULT = {
    "known_codes": [],
    "last_exit_total": 0,
    "avg_confidence": None,
    "confidence_samples": 0,
    "last_run_ts": 0,
}


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return STATE_DEFAULT.copy()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return STATE_DEFAULT.copy()
    out = STATE_DEFAULT.copy()
    out.update(data)
    return out


def _persist_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def _append_health_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ts",
        "exit_total",
        "avg_confidence",
        "confidence_samples",
        "multiplier",
        "exit_threshold",
        "confidence_drop_pct",
        "new_codes_count",
        "new_codes",
        "triggered",
    ]
    exists = path.exists()
    try:
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass


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
    if notifier is None:
        return
    try:
        asyncio.run(notifier.speak(text, "critical"))
    except Exception:
        pass


def _collect_events(path: Path, since: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    now = time.time()
    cutoff = now - since
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ts = float(ev.get("ts") or 0)
            if ts < cutoff:
                continue
            out.append(ev)
    return out


def _summarize_confidence(events: list[dict[str, Any]]) -> tuple[float, int]:
    values: list[float] = []
    for ev in events:
        data = ev.get("data") or {}
        for key in ("confidence", "conf"):
            if key not in data:
                continue
            try:
                values.append(float(data[key]))
            except Exception:
                continue
    if not values:
        return 0.0, 0
    return sum(values) / len(values), len(values)


def _adaptive_multiplier(
    events: list[dict[str, Any]],
    watch: set[str],
    step: float,
    minimum: float,
) -> tuple[float, Counter[str]]:
    counts = Counter(ev.get("event", "") for ev in events if ev.get("event") in watch)
    total = sum(counts.values())
    multiplier = max(minimum, 1 - total * step)
    return multiplier, counts


def _classify(
    events: list[dict[str, Any]],
    state: dict[str, Any],
    exit_threshold: int,
    drop_pct: float,
) -> tuple[list[str], dict[str, Any], list[str], dict[str, Any]]:
    exit_events = [ev for ev in events if str(ev.get("event", "")).startswith("exit.")]
    exit_total = len(exit_events)
    exit_counter = Counter(str(ev.get("event", "")) for ev in exit_events if ev.get("event"))

    codes = set(
        str(ev.get("data", {}).get("code"))
        for ev in events
        if ev.get("data") and "code" in ev["data"] and ev["data"]["code"] is not None
    )
    codes.discard("None")
    known_codes = set(state.get("known_codes") or [])
    new_codes = sorted(codes - known_codes)

    confidence_avg, confidence_samples = _summarize_confidence(events)
    prev_confidence = state.get("avg_confidence")
    lines: list[str] = []
    triggers: list[str] = []

    lines.append(f"Window exit.total={exit_total} (threshold={exit_threshold})")
    if exit_counter:
        exit_summary = ", ".join(f"{name}={cnt}" for name, cnt in exit_counter.items())
        lines.append(f"  → by event: {exit_summary}")
    if exit_total >= exit_threshold and exit_total > int(state.get("last_exit_total", 0)):
        triggers.append(f"exit spike: {exit_total} events (prev {state.get('last_exit_total', 0)})")

    if new_codes:
        lines.append("New codes: " + ", ".join(new_codes))
        triggers.append("new telemetry codes: " + ", ".join(new_codes))
    else:
        lines.append("New codes: none")

    if confidence_samples:
        lines.append(f"Avg confidence={confidence_avg:.2f} over {confidence_samples} samples")
        if (
            prev_confidence
            and confidence_avg < float(prev_confidence) * (1 - drop_pct)
            and confidence_samples >= 3
        ):
            drop = (prev_confidence - confidence_avg) / prev_confidence if prev_confidence else 0.0
            triggers.append(f"confidence drop {drop * 100:.1f}% (< {confidence_avg:.2f})")
    else:
        lines.append("Avg confidence=no samples")

    new_state = {
        "known_codes": sorted(k for k in known_codes.union(codes) if k),
        "last_exit_total": exit_total,
        "avg_confidence": confidence_avg if confidence_samples else prev_confidence,
        "confidence_samples": confidence_samples,
        "last_run_ts": int(time.time()),
    }
    metrics = {
        "exit_total": exit_total,
        "confidence_avg": confidence_avg,
        "confidence_samples": confidence_samples,
        "new_codes": new_codes,
    }
    return lines if lines else ["No telemetry data"], new_state, triggers, metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry alert classifier")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Path to telemetry JSONL")
    parser.add_argument("--state-path", default="logs/telemetry_classifier_state.json", help="State file location")
    parser.add_argument("--since-min", type=int, default=30, help="Minutes to include in the snapshot")
    parser.add_argument("--exit-threshold", type=int, default=5, help="Exit event count threshold to alert")
    parser.add_argument(
        "--confidence-drop-pct",
        type=float,
        default=0.15,
        help="Fractional drop (e.g., 0.15 for 15%) vs previous avg to trigger",
    )
    parser.add_argument("--no-notify", action="store_true", help="Skip Telegram notification")
    parser.add_argument(
        "--health-path",
        default="logs/telemetry_health.csv",
        help="CSV path to append the health history",
    )
    parser.add_argument(
        "--adaptive-events",
        default="entry.blocked,data.quality.roll_alert",
        help="Comma-separated events that reduce sensitivity",
    )
    parser.add_argument(
        "--adaptive-step",
        type=float,
        default=0.1,
        help="Per-event multiplier reduction (1 - total*step, min=adaptive-min)",
    )
    parser.add_argument(
        "--adaptive-min",
        type=float,
        default=0.4,
        help="Minimum adaptive multiplier (0-1)",
    )
    args = parser.parse_args(argv)

    telemetry_path = Path(args.path)
    state_path = Path(args.state_path)
    state = _load_state(state_path)
    events = _collect_events(telemetry_path, max(1, args.since_min) * 60)

    watch_events = {e.strip() for e in args.adaptive_events.split(",") if e.strip()}
    step = max(0.0, args.adaptive_step)
    minimum = min(1.0, max(0.0, args.adaptive_min))
    if watch_events:
        multiplier, adaptive_counts = _adaptive_multiplier(events, watch_events, step, minimum)
    else:
        multiplier = 1.0
        adaptive_counts = Counter()

    adjusted_exit_threshold = max(1, math.ceil(args.exit_threshold * multiplier))
    adjusted_drop_pct = max(0.01, args.confidence_drop_pct * multiplier)

    lines, new_state, triggers, metrics = _classify(
        events,
        state,
        adjusted_exit_threshold,
        adjusted_drop_pct,
    )

    adaptive_summary = (
        f"Adaptive multiplier={multiplier:.2f} (counts="
        + ", ".join(f"{name}={adaptive_counts.get(name,0)}" for name in sorted(watch_events))
        + ") -> exit threshold="
        + str(adjusted_exit_threshold)
        + f", conf-drop={adjusted_drop_pct:.2%}"
    )
    lines.insert(0, adaptive_summary)

    summary = "Telemetry alert classifier summary:\n" + "\n".join("  " + l for l in lines)
    print(summary)
    _persist_state(state_path, new_state)

    health_path = Path(args.health_path)
    triggered_flag = bool(triggers)
    _append_health_row(
        health_path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "exit_total": metrics.get("exit_total", 0),
            "avg_confidence": metrics.get("confidence_avg", 0.0),
            "confidence_samples": metrics.get("confidence_samples", 0),
            "multiplier": multiplier,
            "exit_threshold": adjusted_exit_threshold,
            "confidence_drop_pct": adjusted_drop_pct,
            "new_codes_count": len(metrics.get("new_codes", [])),
            "new_codes": ";".join(metrics.get("new_codes", [])),
            "triggered": triggered_flag,
        },
    )

    if args.no_notify:
        return 0

    notifier = _build_notifier()
    if notifier and triggers:
        _send_alert(notifier, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
