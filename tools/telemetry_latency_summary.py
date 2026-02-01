#!/usr/bin/env python3
"""
Summarize telemetry.latency events (per symbol/stage).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import time


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


def _now() -> float:
    return time.time()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize telemetry.latency events")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=60, help="Window minutes (0 = all)")
    args = parser.parse_args(argv)

    path = Path(args.path)
    events = _load_jsonl(path, limit=10000)
    cutoff = 0.0
    if args.since_min > 0:
        cutoff = _now() - args.since_min * 60

    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ev in events:
        if str(ev.get("event")) != "telemetry.latency":
            continue
        ts = float(ev.get("ts") or 0.0)
        if cutoff and ts < cutoff:
            continue
        data = ev.get("data") or {}
        sym = str(data.get("symbol") or "").upper() or "UNKNOWN"
        stage = str(data.get("stage") or "unknown")
        duration = float(data.get("duration_ms") or 0.0)
        if duration <= 0:
            continue
        stats[sym][stage].append(duration)

    if not stats:
        print("No telemetry.latency data in window.")
        return 0

    print(f"Telemetry latency summary (last {args.since_min} min):")
    for sym, stages in sorted(stats.items()):
        print(f"- {sym}:")
        for stage, values in sorted(stages.items()):
            avg = sum(values) / len(values)
            print(f"    {stage}: count={len(values)} avg={avg:.1f}ms max={max(values):.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
