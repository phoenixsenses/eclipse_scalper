#!/usr/bin/env python3
"""
Signal data health report.

Summarizes `data.quality`, `data.stale`, and missing-data telemetry so you can
spot symbols with bad freshness/quality before they hit the entry loop.
"""

from __future__ import annotations

import argparse
import json
import statistics
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any


def _load_events(path: Path, since_min: int):
    now = time.time()
    cutoff = max(0.0, now - since_min * 60)
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
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


def _analyze_data_health(events: list[dict[str, Any]]):
    quality: dict[str, dict[str, Any]] = {}
    stale: Counter[str] = Counter()
    stale_age: dict[str, float] = {}
    missing: Counter[str] = Counter()

    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") or {}
        sym = str(data.get("symbol") or ev.get("symbol") or "").upper()

        if name == "data.quality":
            info = quality.setdefault(sym or "unknown", {"scores": [], "rolls": [], "count": 0})
            score = float(data.get("score") or 0.0)
            roll = float(data.get("roll") or 0.0)
            info["scores"].append(score)
            info["rolls"].append(roll)
            info["count"] += 1
        elif name == "data.stale":
            stale[sym or "unknown"] += 1
            age = float(data.get("age_sec") or 0.0)
            stale_age[sym] = max(stale_age.get(sym, 0.0), age)
        elif name in ("data.ticker_missing", "data.ohlcv_missing"):
            missing[name] += 1

    return quality, stale, stale_age, missing


def _fmt_table(rows, headers):
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))]
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Signal data health report")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=60, help="Minutes to include")
    parser.add_argument("--top", type=int, default=6, help="Top worst-quality symbols")
    args = parser.parse_args()

    path = Path(args.path)
    events = _load_events(path, max(1, args.since_min))
    quality, stale, stale_age, missing = _analyze_data_health(events)

    print(f"Signal data health (last {args.since_min} min over {len(events)} events)")

    if quality:
        scored = []
        for sym, info in quality.items():
            avg = statistics.mean(info["scores"]) if info["scores"] else 0.0
            roll = statistics.mean(info["rolls"]) if info["rolls"] else 0.0
            scored.append((sym, avg, roll, info["count"]))
        scored.sort(key=lambda x: (x[1], x[2]))  # worst score first
        rows = []
        for sym, avg, roll, cnt in scored[: args.top]:
            rows.append((sym, f"{avg:.1f}", f"{roll:.1f}", cnt))
        print("\nWorst data quality (score, roll, samples):")
        print(_fmt_table(rows, ("Symbol", "Score", "Roll", "Samples")))
    else:
        print("\nNo `data.quality` telemetry found in the window.")

    if stale:
        rows = [
            (sym, count, f"{stale_age.get(sym, 0.0):.1f}")
            for sym, count in stale.items()
        ]
        rows.sort(key=lambda x: x[1], reverse=True)
        print("\nStaleness events:")
        print(_fmt_table(rows, ("Symbol", "Count", "Max age (sec)")))
    else:
        print("\nNo staleness events in the window.")

    if missing:
        print("\nMissing data events:")
        for name, count in missing.items():
            print(f"- {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
