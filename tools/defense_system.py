#!/usr/bin/env python3
"""
Defense system / strategy tuning helper.

Reads the guard history, anomaly actions, and guard timeline to produce a simple
defense score and actionable tuning suggestions (reduce leverage, raise confidence,
pause entries) when telemetry trends start to deteriorate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


def _read_csv(path: Path, rows: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        data = list(reader)
        if rows > 0:
            return data[-rows:]
        return data


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_timeline(path: Path) -> dict[str, Any]:
    out = {"override_active": False, "override_reason": "", "recent_hits": []}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("Recovery override active:"):
            out["override_active"] = "yes" in line.lower()
        if line.startswith("Override:"):
            out["override_reason"] = line.split(":", 1)[1].strip()
        if line.startswith("- "):
            out["recent_hits"].append(line[2:].strip())
    return out


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def compute_defense(rows: list[dict[str, str]], actions: dict[str, Any], timeline: dict[str, Any]) -> tuple[str, int]:
    partial_hits = [int(row.get("partial_retry_hits") or 0) for row in rows if row.get("partial_retry_hits")]
    total_rows = len(rows)
    avg_hits = statistics.mean(partial_hits) if partial_hits else 0
    recent_hits = partial_hits[-1] if partial_hits else 0
    hit_rate = (sum(partial_hits) / total_rows) if total_rows else 0
    exposures = float(actions.get("exposures") or 0.0)
    override_active = timeline.get("override_active", False)
    override_reason = timeline.get("override_reason", "")
    score = 0
    suggestions = []
    if hit_rate >= 0.25 or avg_hits >= 2 or recent_hits >= 2:
        score += 2
        suggestions.append("raise ENTRY_MIN_CONFIDENCE by 0.05")
        suggestions.append("reduce FIXED_NOTIONAL_USDT by ~20% or drop LEVERAGE by 1")
    if override_active:
        score += 1
        suggestions.append(f"keep entries paused until override reason clears ({override_reason})")
    if exposures > 0:
        exposure_pct = min(1.0, exposures / 1000.0)
        score += 1 if exposure_pct > 0.5 else 0
        suggestions.append(f"keep per-symbol leverage low; exposures currently {int(exposures)} USDT")
    severity = max(0, min(score, 4))
    summary = [
        f"Guard history rows: {total_rows}",
        f"Avg partial/retry hits: {avg_hits:.2f}",
        f"Recent hits: {recent_hits}, hit rate: {hit_rate:.2%}",
        f"Override active: {override_active} ({override_reason})",
        f"Exposures: {int(exposures)} USDT",
    ]
    if suggestions:
        summary.append("Suggested tuning:")
        summary.extend(f"- {line}" for line in suggestions[:3])
    else:
        summary.append("Defensive posture: normal")
    return "\n".join(summary), severity


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


async def _notify(message: str, severity: str) -> None:
    notifier = _build_notifier()
    if notifier is None:
        return
    try:
        await notifier.speak(message, severity)
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Guard defense/strategy tuning helper")
    parser.add_argument("--history", default="logs/telemetry_guard_history.csv", help="Guard history CSV")
    parser.add_argument("--rows", type=int, default=12, help="Rows to ingest")
    parser.add_argument("--actions", default="logs/telemetry_anomaly_actions.json", help="Anomaly actions JSON")
    parser.add_argument("--timeline", default="logs/telemetry_guard_timeline.txt", help="Guard timeline text")
    parser.add_argument("--notify", action="store_true", help="Send notification via Telegram when severity >=2")
    args = parser.parse_args(argv)

    rows = _read_csv(Path(args.history), args.rows)
    actions = _read_json(Path(args.actions))
    timeline = _parse_timeline(Path(args.timeline))
    summary, severity = compute_defense(rows, actions, timeline)
    print(summary)
    print(f"Defense severity (0–4): {severity}")
    if args.notify and severity >= 2:
        import asyncio

        asyncio.run(_notify(summary, "warning"))
    return 0 if severity < 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
