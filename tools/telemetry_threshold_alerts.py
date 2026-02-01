#!/usr/bin/env python3
"""
Summarize telemetry events against thresholds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict


PRESET_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "exit": {
        "exit.blocked": 3,
        "exit_loop:error": 1,
        "exit_loop_symbol_error": 1,
    },
}


PRESET_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "exit": {
        "exit.blocked": 3,
        "exit_loop:error": 1,
        "exit_loop_symbol_error": 1,
    },
}


def _load_jsonl(path: Path, limit: int = 5000):
    if not path.exists():
        return []
    out = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
                if len(out) >= limit:
                    break
    except Exception:
        return out
    return out


def _parse_thresholds(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not raw:
        return out
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        event, val = part.split("=", 1)
        try:
            out[event.strip()] = max(1, int(float(val.strip())))
        except Exception:
            continue
    return out


def _plan_filters(events: list[dict], since_ts: float, names: set[str]) -> list[dict]:
    out = []
    for ev in events:
        ts = float(ev.get("ts") or 0.0)
        if since_ts and ts < since_ts:
            continue
        name = str(ev.get("event") or "")
        if names and name not in names:
            continue
        out.append(ev)
    return out


def _symkey(ev: dict) -> str:
    sym = ev.get("symbol") or (ev.get("data") or {}).get("symbol")
    return str(sym or "").upper()


def scan_thresholds(
    path: Path,
    limit: int,
    since_min: int,
    thresholds: Dict[str, int],
    monitor: set[str] | None = None,
) -> tuple[Dict[str, Dict[str, int]], list[tuple[str, int, int]]]:
    events = _load_jsonl(path, limit=limit)
    since_ts = 0.0
    if since_min > 0:
        since_ts = time.time() - since_min * 60

    monitor_set = monitor or set()
    if not monitor_set and thresholds:
        monitor_set = set(thresholds.keys())

    filtered = _plan_filters(events, since_ts, monitor_set)

    counts: Dict[str, Dict[str, int]] = {}
    triggered: list[tuple[str, int, int]] = []
    for ev in filtered:
        name = str(ev.get("event") or "")
        sym = _symkey(ev)
        entry = counts.setdefault(name, {})
        entry["total"] = entry.get("total", 0) + 1
        if sym:
            entry[sym] = entry.get(sym, 0) + 1

    for name, data in counts.items():
        total = data.get("total", 0)
        threshold = thresholds.get(name)
        if threshold and total >= threshold:
            triggered.append((name, total, threshold))
    return counts, triggered



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry threshold alert helper")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--limit", type=int, default=5000, help="Max events to load")
    parser.add_argument("--since-min", type=int, default=30, help="Lookback window in minutes")
    parser.add_argument("--thresholds", default="", help="Comma-separated event=threshold")
    parser.add_argument("--events", default="", help="Comma-separated events to monitor (default: those in thresholds)")
    parser.add_argument("--preset", default="", help="Apply named preset thresholds (exit)")
    args = parser.parse_args(argv)

    path = Path(args.path)
    thresholds = _parse_thresholds(args.thresholds)
    preset_thresholds: Dict[str, int] = {}
    preset_name = str(args.preset or "").strip().lower()
    if preset_name and preset_name in PRESET_THRESHOLDS:
        preset_thresholds = dict(PRESET_THRESHOLDS[preset_name])
    merged_thresholds: Dict[str, int] = {**preset_thresholds, **thresholds}
    monitor = {e for e in merged_thresholds} if merged_thresholds else set()
    if args.events.strip():
        monitor = {e.strip() for e in args.events.split(",") if e.strip()}

    counts, triggered = scan_thresholds(
        path=path,
        limit=max(1, int(args.limit)),
        since_min=args.since_min,
        thresholds=merged_thresholds,
        monitor=monitor,
    )

    if not counts:
        print("No matching telemetry events")
        return 0

    print(f"Telemetry threshold scan (since last {args.since_min} min):")
    for name, data in sorted(counts.items(), key=lambda kv: kv[1].get("total", 0), reverse=True):
        total = data.get("total", 0)
        symbols = ", ".join(f"{sym}={cnt}" for sym, cnt in sorted(data.items()) if sym not in ("total",) and cnt > 0)
        print(f"- {name}: total={total} {symbols or ''}")
        if merged_thresholds and name in merged_thresholds:
            threshold = merged_thresholds[name]
            if total >= threshold:
                print(f"  ALERT {name}: {total} >= {threshold}")
    if triggered:
        print("- Alerts triggered:")
        for name, total, threshold in triggered:
            print(f"  * {name}: {total} >= {threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
