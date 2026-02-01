#!/usr/bin/env python3
"""
Quick summary of entry.blocked reasons from logs/telemetry.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from typing import Any


def _safe_load(line: str) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize entry.blocked telemetry.")
    parser.add_argument(
        "--path",
        default=os.getenv("TELEMETRY_PATH", "") or os.path.join("logs", "telemetry.jsonl"),
        help="Path to telemetry jsonl file (default: logs/telemetry.jsonl or TELEMETRY_PATH).",
    )
    parser.add_argument(
        "--symbol",
        default="",
        help="Optional symbol filter (e.g., DOGEUSDT).",
    )
    parser.add_argument(
        "--event",
        default="entry.blocked",
        help="Telemetry event to summarize (default: entry.blocked).",
    )
    parser.add_argument(
        "--since",
        type=float,
        default=0.0,
        help="Only include events from the last N minutes (default: 0, no filter).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of reasons to show (default: 20).",
    )
    parser.add_argument(
        "--sort",
        default="count",
        choices=("count", "alpha"),
        help="Sort reasons by count or alphabetically (default: count).",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=1,
        help="Only show reasons with count >= N (default: 1).",
    )
    parser.add_argument(
        "--reason-contains",
        default="",
        help="Only include reasons containing this text (case-insensitive).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the header and guard bucket summary.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path for reasons and counts.",
    )
    return parser.parse_args()


def _rec_symbol(rec: dict[str, Any]) -> str:
    data = rec.get("data") or {}
    return str(rec.get("symbol") or data.get("symbol") or data.get("k") or "")


def main() -> int:
    args = _parse_args()
    path = args.path
    if not os.path.exists(path):
        print(f"missing telemetry file: {path}")
        return 1

    reasons = Counter()
    corr_blocks = Counter()
    total = 0
    since_ts = 0.0
    if args.since and args.since > 0:
        since_ts = time.time() - float(args.since) * 60.0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = _safe_load(line)
            if not isinstance(rec, dict):
                continue
            if rec.get("event") != args.event:
                continue
            if since_ts > 0:
                ts = float(rec.get("ts") or rec.get("timestamp") or 0.0)
                if ts and ts < since_ts:
                    continue
            if args.symbol:
                sym = _rec_symbol(rec).upper()
                if sym != args.symbol.upper():
                    continue
            data = rec.get("data") or {}
            reason = data.get("reason") or "<unknown>"
            reason_str = str(reason)
            reasons[reason_str] += 1
            if reason_str.lower().startswith("group "):
                corr_blocks[reason_str] += 1
            total += 1

    scope = f"symbol={args.symbol.upper()}" if args.symbol else "all symbols"
    print(f"{args.event} total: {total} ({scope})")
    if total:
        buckets = {
            "cooldown": 0,
            "data": 0,
            "correlation": 0,
        }
        for reason, count in reasons.items():
            r = reason.lower()
            if "cooldown" in r:
                buckets["cooldown"] += count
            if "dataframe" in r or "no dataframe" in r or "insufficient bars" in r or "missing ohlc" in r:
                buckets["data"] += count
            if r.startswith("group "):
                buckets["correlation"] += count
        print(
            "top guard buckets: "
            f"cooldown={buckets['cooldown']} | "
            f"data={buckets['data']} | "
            f"correlation={buckets['correlation']}"
        )
    if args.sort == "alpha":
        reason_items = sorted(reasons.items(), key=lambda kv: kv[0])
    else:
        reason_items = reasons.most_common()
    if args.reason_contains:
        needle = args.reason_contains.lower()
        reason_items = [(r, c) for r, c in reason_items if needle in r.lower()]
    if args.min > 1:
        reason_items = [(r, c) for r, c in reason_items if c >= args.min]
    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["event", "reason", "count"])
                for reason, count in reason_items:
                    w.writerow([args.event, reason, count])
        except Exception as e:
            print(f"csv write failed: {e}")
    if not args.summary_only:
        for reason, count in reason_items[: max(1, args.top)]:
            print(f"{reason}: {count}")
    if corr_blocks:
        if not args.summary_only:
            print("\ncorrelation guard blocks:")
            if args.sort == "alpha":
                corr_items = sorted(corr_blocks.items(), key=lambda kv: kv[0])
            else:
                corr_items = corr_blocks.most_common()
            if args.reason_contains:
                needle = args.reason_contains.lower()
                corr_items = [(r, c) for r, c in corr_items if needle in r.lower()]
            if args.min > 1:
                corr_items = [(r, c) for r, c in corr_items if c >= args.min]
            for reason, count in corr_items[: max(1, args.top)]:
                print(f"{reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
