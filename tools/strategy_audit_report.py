#!/usr/bin/env python3
"""
Summarize strategy audit CSV (blocker counts and signal counts).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strategy audit CSV.")
    parser.add_argument(
        "--path",
        default=os.getenv("SCALPER_AUDIT_PATH", "") or os.path.join("logs", "strategy_audit.csv"),
        help="Path to audit CSV file (default: logs/strategy_audit.csv or SCALPER_AUDIT_PATH).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print totals and unique counts.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of items to show (default: 20).",
    )
    parser.add_argument(
        "--sort",
        default="count",
        choices=("count", "alpha"),
        help="Sort items by count or alphabetically (default: count).",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=1,
        help="Only show items with count >= N (default: 1).",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path for outcomes and blockers.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = args.path
    if not os.path.exists(path):
        print(f"missing audit file: {path}")
        return 1

    outcomes = Counter()
    blockers = Counter()
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            total += 1
            outcomes[row.get("outcome") or "<unknown>"] += 1
            b = row.get("blockers") or ""
            for item in [x for x in b.split("|") if x]:
                blockers[item] += 1

    print(f"rows: {total} | outcomes: {len(outcomes)} | blockers: {len(blockers)}")
    if not args.summary_only:
        if args.sort == "alpha":
            outcome_items = sorted(outcomes.items(), key=lambda kv: kv[0])
        else:
            outcome_items = outcomes.most_common()
        if args.min > 1:
            outcome_items = [(k, v) for k, v in outcome_items if v >= args.min]
        outcome_items = outcome_items[: max(1, args.top)]
        print("outcomes:")
        for k, v in outcome_items:
            print(f"  {k}: {v}")

        if args.sort == "alpha":
            blocker_items = sorted(blockers.items(), key=lambda kv: kv[0])
        else:
            blocker_items = blockers.most_common()
        if args.min > 1:
            blocker_items = [(k, v) for k, v in blocker_items if v >= args.min]
        blocker_items = blocker_items[: max(1, args.top)]
        print("\nblockers:")
        for k, v in blocker_items:
            print(f"  {k}: {v}")
    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["section", "item", "count"])
                for k, v in outcomes.items():
                    w.writerow(["outcome", k, v])
                for k, v in blockers.items():
                    w.writerow(["blocker", k, v])
        except Exception as e:
            print(f"csv write failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
