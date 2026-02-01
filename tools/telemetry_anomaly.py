#!/usr/bin/env python3
"""
Telemetry anomaly detector.

Compares exposures, confidence, and risk events against the previous run and
notifies via Telegram when thresholds trigger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DEFAULT = {
    "exposures": 0.0,
    "avg_confidence": 0.0,
    "risk_total": 0,
    "exit_codes": [],
    "last_ts": 0,
}


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return STATE_DEFAULT.copy()
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        out = STATE_DEFAULT.copy()
        out.update(data)
        return out
    except Exception:
        return STATE_DEFAULT.copy()


def _persist_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
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
    if notifier is None or not text.strip():
        return
    try:
        asyncio.run(notifier.speak(text, "critical"))
    except Exception:
        pass


def _load_events(path: Path, since_min: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = time.time() - max(1, since_min) * 60
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ts = float(ev.get("ts") or 0.0)
            if ts < cutoff:
                continue
            out.append(ev)
    return out


def _summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    exposures_total = 0.0
    confidences = []
    risk_total = 0
    exit_codes = set()
    exit_events = 0

    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") or {}
        if name == "order.create":
            amount = float(data.get("amount") or 0.0)
            price = float(data.get("price") or data.get("px") or 1.0)
            exposures_total += abs(amount) * max(price, 1.0)
        elif name == "entry.submitted":
            try:
                confidences.append(float(data.get("confidence") or 0.0))
            except Exception:
                pass
        elif name.startswith("risk.") or name.startswith("kill_switch"):
            risk_total += 1
        elif name.startswith("exit.") and data:
            exit_events += 1
            code = str(data.get("code") or "").strip()
            if code:
                exit_codes.add(code)

    avg_conf = statistics.mean(confidences) if confidences else 0.0
    return {
        "exposures": exposures_total,
        "avg_confidence": avg_conf,
        "risk_total": risk_total,
        "exit_events": exit_events,
        "exit_codes": sorted(exit_codes),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry anomaly detector")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--state", default="logs/telemetry_anomaly_state.json", help="State file")
    parser.add_argument("--since-min", type=int, default=60, help="Minutes to include")
    parser.add_argument("--exposure-thresh", type=float, default=0.5, help="Relative spike threshold")
    parser.add_argument("--confidence-drop", type=float, default=0.15, help="Absolute drop fraction")
    parser.add_argument("--risk-thresh", type=int, default=3, help="Risk event count threshold")
    parser.add_argument("--no-notify", action="store_true", help="Skip Telegram notification")
    parser.add_argument("--output", default="logs/telemetry_anomaly.txt", help="Report file")

    args = parser.parse_args(argv)
    events = _load_events(Path(args.path), args.since_min)
    metrics = _summarize(events)
    state_path = Path(args.state)
    prev = _load_state(state_path)

    anomalies: list[str] = []

    prev_exposures = float(prev.get("exposures") or 0.0)
    curr_exposures = metrics["exposures"]
    if prev_exposures > 0:
        delta = (curr_exposures - prev_exposures) / prev_exposures
        if delta >= args.exposure_thresh:
            anomalies.append(
                f"Exposure spike: {curr_exposures:.0f} (Δ {delta:+.0%} vs prev {prev_exposures:.0f})"
            )

    prev_conf = float(prev.get("avg_confidence") or 0.0)
    curr_conf = metrics["avg_confidence"]
    if prev_conf > 0 and curr_conf < prev_conf * (1 - args.confidence_drop):
        drop_pct = (prev_conf - curr_conf) / prev_conf
        anomalies.append(f"Confidence drop: {curr_conf:.2f} (-{drop_pct:.0%} from {prev_conf:.2f})")

    if metrics["risk_total"] >= args.risk_thresh and metrics["risk_total"] > prev.get("risk_total", 0):
        anomalies.append(f"Risk events: {metrics['risk_total']} (threshold {args.risk_thresh})")

    new_codes = set(metrics["exit_codes"]) - set(prev.get("exit_codes") or [])
    if new_codes:
        anomalies.append(f"New exit codes: {', '.join(sorted(new_codes))}")

    report = [
        f"Telemetry anomaly report ({len(events)} events, last {args.since_min} min)",
        f"- exposures: {curr_exposures:.0f}",
        f"- avg confidence: {curr_conf:.2f}",
        f"- risk events: {metrics['risk_total']}",
        f"- exit events: {metrics['exit_events']} codes={','.join(metrics['exit_codes']) or 'none'}",
    ]
    if anomalies:
        report.append("\nAnomalies detected:")
        report.extend(f"  • {item}" for item in anomalies)
    else:
        report.append("\nNo anomalies detected.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    if anomalies and not args.no_notify:
        notifier = _build_notifier()
        _send_alert(notifier, "\n".join(report))

    new_state = {
        "exposures": curr_exposures,
        "avg_confidence": curr_conf,
        "risk_total": metrics["risk_total"],
        "exit_codes": metrics["exit_codes"],
        "last_ts": int(time.time()),
    }
    _persist_state(state_path, new_state)
    for line in report:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
