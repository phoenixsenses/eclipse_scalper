#!/usr/bin/env python3
"""
Signal-to-exit health report.

Compares exit.* telemetry events against the stored entry signal metadata so you can
spot cases where the bot handed off low-confidence signals straight into exits (or
where telemetry guards forced a close) and route those insights back into the dashboard.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _load_events(path: Path, since_min: int) -> list[dict[str, Any]]:
    cutoff = max(0.0, time.time() - max(1, since_min) * 60)
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ts = _safe_float(ev.get("ts"), 0.0)
            if ts < cutoff:
                continue
            events.append(ev)
    return events


def _fmt_table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> str:
    if not rows:
        return ""
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + rows))]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "-+-".join("-" * w for w in col_widths)
    out = [line, sep]
    for row in rows:
        out.append(" | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)))
    return "\n".join(out)


def _format_ts(ts: float) -> str:
    if ts <= 0:
        return "N/A"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Signal -> exit health report")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=60, help="Minutes to include")
    parser.add_argument("--top", type=int, default=6, help="Top reasons/symbols to show")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Highlight events below")
    args = parser.parse_args(argv)

    events = _load_events(Path(args.path), args.since_min)
    exit_events = [ev for ev in events if isinstance(ev.get("event"), str) and ev["event"].startswith("exit.")]
    if not exit_events:
        print(f"Signal -> exit health (last {args.since_min} min): no exit events found")
        return 0

    reason_stats: dict[str, dict[str, Any]] = {}
    symbol_stats: dict[str, dict[str, Any]] = {}
    low_conf_events: list[dict[str, Any]] = []
    guard_events: list[dict[str, Any]] = []
    timestamps: Counter[str] = Counter()

    for ev in exit_events:
        event_name = ev.get("event", "")
        reason = event_name.split(".", 1)[-1] if "." in event_name else event_name
        data = ev.get("data") or {}
        symbol = str(data.get("symbol") or ev.get("symbol") or "unknown").upper()
        conf = _safe_float(data.get("entry_confidence"), -1.0)
        age = _safe_float(data.get("entry_signal_age_sec"), -1.0)
        ts = _safe_float(ev.get("ts"), 0.0)
        exposures = _safe_float(data.get("exposures"), 0.0)
        guard = str(data.get("guard") or "").lower().strip()

        reason_bucket = reason_stats.setdefault(
            reason,
            {"count": 0, "confidences": [], "ages": [], "guards": 0, "exposures": [], "last_ts": 0.0},
        )
        reason_bucket["count"] += 1
        if conf >= 0:
            reason_bucket["confidences"].append(conf)
        if age >= 0:
            reason_bucket["ages"].append(age)
        if exposures > 0:
            reason_bucket["exposures"].append(exposures)
        if guard:
            reason_bucket["guards"] += 1
        reason_bucket["last_ts"] = max(reason_bucket["last_ts"], ts)

        symbol_bucket = symbol_stats.setdefault(
            symbol,
            {"count": 0, "confidences": [], "hj_guard": 0, "ages": [], "last_ts": 0.0},
        )
        symbol_bucket["count"] += 1
        if conf >= 0:
            symbol_bucket["confidences"].append(conf)
        if age >= 0:
            symbol_bucket["ages"].append(age)
        if guard:
            symbol_bucket["hj_guard"] += 1
        symbol_bucket["last_ts"] = max(symbol_bucket["last_ts"], ts)

        timestamps[event_name] += 1

        if conf >= 0 and conf < args.min_confidence:
            low_conf_events.append(
                {
                    "symbol": symbol,
                    "reason": reason,
                    "confidence": conf,
                    "age": age,
                    "timestamp": ts,
                    "guard": guard,
                }
            )
        elif conf < 0:
            low_conf_events.append(
                {
                    "symbol": symbol,
                    "reason": reason,
                    "confidence": "n/a",
                    "age": age,
                    "timestamp": ts,
                    "guard": guard,
                }
            )

        if guard:
            guard_events.append(
                {
                    "symbol": symbol,
                    "reason": reason,
                    "guard": guard,
                    "confidence": conf if conf >= 0 else None,
                    "timestamp": ts,
                    "exposures": exposures,
                }
            )

    print(f"Signal -> exit health (last {args.since_min} min, {len(exit_events)} exit events)")

    reason_rows = []
    for reason, info in sorted(reason_stats.items(), key=lambda kv: -kv[1]["count"])[: args.top]:
        avg_conf = (
            f"{statistics.mean(info['confidences']):.2f}"
            if info["confidences"]
            else "n/a"
        )
        avg_age = (
            f"{statistics.mean(info['ages']):.1f}s"
            if info["ages"]
            else "n/a"
        )
        avg_exp = (
            f"{statistics.mean(info['exposures']):.0f}"
            if info["exposures"]
            else "n/a"
        )
        reason_rows.append(
            (reason, info["count"], avg_conf, avg_age, avg_exp, info["guards"], _format_ts(info["last_ts"]))
        )
    if reason_rows:
        print("\nExit reason summary:")
        print(
            _fmt_table(
                reason_rows,
                ("Reason", "Count", "Avg Conf", "Avg Age", "Avg Exposure", "Guard Hits", "Last Seen"),
            )
        )

    symbol_rows = []
    for symbol, info in sorted(symbol_stats.items(), key=lambda kv: -kv[1]["count"])[: args.top]:
        avg_conf = (
            f"{statistics.mean(info['confidences']):.2f}"
            if info["confidences"]
            else "n/a"
        )
        avg_age = (
            f"{statistics.mean(info['ages']):.1f}s"
            if info["ages"]
            else "n/a"
        )
        symbol_rows.append(
            (symbol, info["count"], avg_conf, avg_age, info["hj_guard"], _format_ts(info["last_ts"]))
        )
    if symbol_rows:
        print("\nSymbol exit health (highest exit counts):")
        print(
            _fmt_table(
                symbol_rows, ("Symbol", "Count", "Avg Conf", "Avg Age", "Guard Hits", "Last Seen")
            )
        )

    if low_conf_events:
        print(f"\nLow-confidence exits (<{args.min_confidence:.2f} or missing):")
        for row in sorted(low_conf_events, key=lambda x: x["confidence"] if isinstance(x["confidence"], float) else -1)[:10]:
            conf = f"{row['confidence']:.2f}" if isinstance(row["confidence"], float) else row["confidence"]
            print(
                f"- {row['symbol']} {row['reason']} @ {conf} conf age={row['age']:.1f}s guard={row['guard'] or 'none'} ts={_format_ts(row['timestamp'])}"
            )

    if guard_events:
        exposures = [row["exposures"] for row in guard_events if row["exposures"] > 0]
        avg_ex = statistics.mean(exposures) if exposures else 0.0
        print(f"\nTelemetry guard exits triggered: {len(guard_events)} events, avg exposures={avg_ex:.0f}")
        for entry in guard_events[:5]:
            conf = f"{entry['confidence']:.2f}" if isinstance(entry["confidence"], float) else "n/a"
            print(
                f"- {entry['symbol']} {entry['reason']} guard={entry['guard']} conf={conf} exposures={entry['exposures']:.0f} ts={_format_ts(entry['timestamp'])}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
