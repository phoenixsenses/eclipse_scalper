#!/usr/bin/env python3
"""
Core health dashboard (risk, sizing, signal).

Summarizes telemetry events so you can see exposures, entry confidence,
and risk/exit trends in one place.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path, since_min: int, limit: int = 5000):
    if not path.exists():
        return []
    cutoff = time.time() - since_min * 60
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(out) >= limit:
                break
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


def _summarize(events: list[dict[str, Any]]):
    exposures = defaultdict(float)
    net_orders = Counter()
    confidences = defaultdict(list)
    risk_events = Counter()
    blocked_reasons = Counter()
    exits = Counter()
    exit_codes = Counter()

    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") or {}
        sym = str(data.get("symbol") or ev.get("symbol") or "").upper()

        if name == "order.create" and data:
            amount = float(data.get("amount") or 0.0)
            price = float(data.get("price") or data.get("px") or 0.0)
            notional = abs(amount) * (price or 1.0)
            exposures[sym] += notional
            net_orders[sym] += math.copysign(1, amount) if amount else 0
        elif name == "entry.submitted":
            conf = float(data.get("confidence") or 0.0)
            confidences[sym].append(conf)
        elif name.startswith("risk.") or name.startswith("kill_switch"):
            risk_events[name] += 1
        elif name == "order.blocked":
            reason = str(data.get("reason") or data.get("code") or "unknown")
            blocked_reasons[reason] += 1
        elif name.startswith("exit."):
            exits[name] += 1
            code = str(data.get("code") or "")
            if code:
                exit_codes[code] += 1

    avg_conf = {
        sym: statistics.mean(vals) if vals else 0.0
        for sym, vals in confidences.items()
    }

    return exposures, avg_conf, risk_events, blocked_reasons, exits, exit_codes, net_orders


def _fmt_rows(rows, headers):
    widths = [max(len(str(col)) for col in column) for column in zip(*([headers] + rows))]
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(item).ljust(widths[idx]) for idx, item in enumerate(row)))
    return "\n".join(lines)


def _load_anomaly_state(path: Path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Core health (risk/sizing/signal)")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=60, help="minutes to inspect")
    parser.add_argument("--limit", type=int, default=2000, help="max events to load")
    args = parser.parse_args()

    events = _load_jsonl(Path(args.path), max(1, args.since_min), max(1, args.limit))
    (
        exposures,
        confidences,
        risk_events,
        blocked,
        exits,
        exit_codes,
        net_orders,
    ) = _summarize(events)
    equity = float(os.getenv("CORE_HEALTH_EQUITY", "100000"))
    anomaly_state = _load_anomaly_state(Path("logs/telemetry_anomaly_state.json"))

    print(f"Core health dashboard (last {args.since_min} min, {len(events)} events)")

    if exposures:
        rows = sorted(
            (
                sym,
                f"{exposures[sym]:.2f}",
                f"{confidences.get(sym, 0.0):.2f}",
                net_orders.get(sym, 0),
            )
            for sym in sorted(exposures, key=lambda s: exposures[s], reverse=True)
        )[:8]
        print("\nTop exposures (notional). Format: symbol, notional, avg confidence, net orders")
        print(_fmt_rows(rows, ("Symbol", "Notional", "Avg Conf", "Net Ord")))
        ratio = sum(exposures.values()) / max(1.0, equity)
        print(f"\nTotal exposure / equity: {ratio:.2%} (equity base {equity:,.0f})")
    else:
        print("\nNo exposure events found.")

    if risk_events:
        print("\nRisk events count:")
        for name, count in risk_events.most_common(8):
            print(f"- {name}: {count}")

    if blocked:
        print("\nTop blocked reasons:")
        for reason, count in blocked.most_common(6):
            print(f"- {reason}: {count}")

    if exits:
        print("\nExit events (counts):")
        for name, count in exits.items():
            print(f"- {name}: {count}")
        if exit_codes:
            print("\nExit codes summary:")
            for code, count in exit_codes.most_common(6):
                print(f"- {code}: {count}")
    if anomaly_state:
        pause_until = float(anomaly_state.get("pause_until") or 0.0)
        if pause_until and pause_until > time.time():
            ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(pause_until))
            print(f"\nAnomaly pause in effect until {ts}")
        codes = anomaly_state.get("exit_codes") or []
        print(
            f"\nLast anomaly state: exposures={anomaly_state.get('exposures',0):.0f}, "
            f"avg_conf={anomaly_state.get('avg_confidence',0.0):.2f}, "
            f"risk={anomaly_state.get('risk_total',0)}, codes={','.join(codes) or 'none'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
