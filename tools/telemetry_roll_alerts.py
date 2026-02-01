#!/usr/bin/env python3
"""
Summarize recent data.quality.roll_alert events.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_jsonl(path: Path, limit: int = 10000):
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


def _symkey(sym: str) -> str:
    return str(sym or "").upper()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize data.quality.roll_alert events")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=60, help="Only include alerts from last N minutes")
    args = parser.parse_args(argv)

    path = Path(args.path)
    events = _load_jsonl(path, limit=5000)
    since = time.time() - max(0, int(args.since_min)) * 60
    agg: dict[str, dict] = {}

    for ev in events:
        if str(ev.get("event")) != "data.quality.roll_alert":
            continue
        ts = float(ev.get("ts") or 0.0)
        if args.since_min > 0 and ts < since:
            continue
        sym = _symkey(ev.get("symbol") or (ev.get("data") or {}).get("symbol"))
        if not sym:
            continue
        info = agg.setdefault(sym, {"count": 0, "last": 0.0, "max_roll": 0.0})
        info["count"] += 1
        info["last"] = max(info["last"], ts)
        roll = float((ev.get("data") or {}).get("roll") or 0.0)
        info["max_roll"] = max(info["max_roll"], roll)

    if not agg:
        print("No data.quality.roll_alert events found in window.")
        return 0

    print(f"data.quality.roll_alert (last {max(0, int(args.since_min))} min):")
    for sym, info in sorted(agg.items(), key=lambda kv: kv[1]["count"], reverse=True):
        last_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(info["last"]))
        print(f"- {sym}: count={info['count']} last={last_ts} max_roll={info['max_roll']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
