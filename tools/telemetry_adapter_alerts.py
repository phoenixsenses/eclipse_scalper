#!/usr/bin/env python3
"""
Adapter-aware kill-switch telemetry alert helper.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


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


def _parse_thresholds(raw: str) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    if not raw:
        return out
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        try:
            event, adapter = (key.split(".", 1) + [""])[:2]
            event = event.strip()
            adapter = adapter.strip() or "any"
            out[(event, adapter)] = max(1, int(float(val.strip())))
        except Exception:
            continue
    return out


DEFAULT_EVENTS = [
    "kill_switch.halt",
    "kill_switch.clear",
    "kill_switch.escalate_flat",
    "kill_switch.escalate_shutdown",
    "kill_switch.evaluate_error",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Adapter-aware kill-switch telemetry alerts")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Path to telemetry.jsonl")
    parser.add_argument("--limit", type=int, default=5000, help="Max events to load (per script run)")
    parser.add_argument("--since-min", type=int, default=60, help="Lookback window in minutes")
    parser.add_argument("--thresholds", default="", help="Comma-separated event.adapter=threshold (adapter optional)")
    parser.add_argument("--events", default=",".join(DEFAULT_EVENTS), help="Comma list of events to monitor")
    parser.add_argument("--alert-any", action="store_true", help="Trigger alert when any monitored event exceeds its threshold")
    args = parser.parse_args(argv)

    path = Path(args.path)
    events = _load_jsonl(path, limit=max(1, int(args.limit)))
    cutoff = 0.0
    if args.since_min > 0:
        cutoff = time.time() - args.since_min * 60

    monitor = {e.strip() for e in args.events.split(",") if e.strip()}
    thresholds = _parse_thresholds(args.thresholds)

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    triggered = []
    for ev in events:
        ts = float(ev.get("ts") or 0.0)
        if cutoff and ts < cutoff:
            continue
        name = str(ev.get("event") or "")

        if monitor and name not in monitor:
            continue
        data = ev.get("data") or {}
        adapter = str(data.get("adapter") or "unknown")
        counts[name][adapter] += 1
        total[name] += 1

        # threshold check
        for (event, adapter_key), threshold in thresholds.items():
            if event and event != name:
                continue
            if adapter_key not in ("", "any") and adapter_key != adapter:
                continue
            if counts[name][adapter] >= threshold:
                triggered.append((name, adapter, counts[name][adapter], threshold))

    if not total:
        print("No monitored kill-switch events found.")
        return 0

    print(f"Kill-switch telemetry (last {args.since_min} min):")
    for name, adapters in counts.items():
        print(f"- {name}: total={total[name]}")
        for adapter, cnt in sorted(adapters.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  - {adapter}: {cnt}")

    if triggered:
        print("Alerts triggered:")
        for name, adapter, value, threshold in triggered:
            print(f"  * {name} ({adapter}): {value} >= {threshold}")
        if args.alert_any:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
