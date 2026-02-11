#!/usr/bin/env python3
"""
Correlation group guard dashboard.

Summarizes entry.blocked events tagged with corr_group_cap to show which
groups/symbols are hitting position or notional limits.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
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


def _load_events(path: Path, since_min: int, limit: int = 5000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = time.time() - max(1, since_min) * 60
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if len(out) >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            ts = _safe_float(ev.get("ts") or 0.0, 0.0)
            if ts < cutoff:
                continue
            if str(ev.get("event")) != "entry.blocked":
                continue
            data = ev.get("data") or {}
            if str(data.get("reason")) != "corr_group_cap":
                continue
            out.append(ev)
    return out


def _load_scaled_events(path: Path, since_min: int, limit: int = 5000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = time.time() - max(1, since_min) * 60
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if len(out) >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            ts = _safe_float(ev.get("ts") or 0.0, 0.0)
            if ts < cutoff:
                continue
            if str(ev.get("event")) != "entry.notional_scaled":
                continue
            data = ev.get("data") or {}
            reason = str(data.get("reason") or "")
            if "corr_group" not in reason:
                continue
            out.append(ev)
    return out


def _format_table(rows, headers) -> str:
    widths = [max(len(str(col)) for col in column) for column in zip(*([headers] + rows))]
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(item).ljust(widths[idx]) for idx, item in enumerate(row)))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Correlation group guard dashboard")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=1440, help="Minutes to include")
    parser.add_argument("--limit", type=int, default=5000, help="Max events to load")
    parser.add_argument("--output", default="logs/telemetry_corr_group.txt", help="Output path")
    parser.add_argument("--top", type=int, default=8, help="Top rows per table")
    args = parser.parse_args(argv)

    events = _load_events(Path(args.path), args.since_min, limit=max(1, int(args.limit)))
    scaled_events = _load_scaled_events(Path(args.path), args.since_min, limit=max(1, int(args.limit)))
    if not events:
        summary = "Correlation group guard dashboard (no corr_group_cap events found)."
        if scaled_events:
            summary += f"\nCorr-group scaling events: {len(scaled_events)}"
        Path(args.output).write_text(summary + "\n", encoding="utf-8")
        print(summary)
        return 0

    group_counts = Counter()
    symbol_counts = Counter()
    group_notional = defaultdict(list)
    projected_notional = defaultdict(list)
    scaled_by_group = Counter()
    scaled_by_symbol = Counter()

    for ev in events:
        data = ev.get("data") or {}
        group = str(data.get("group") or "unknown").upper()
        symbol = str(ev.get("symbol") or data.get("symbol") or "unknown").upper()
        group_counts[group] += 1
        symbol_counts[symbol] += 1
        group_notional[group].append(_safe_float(data.get("group_notional"), 0.0))
        projected_notional[group].append(_safe_float(data.get("group_projected_notional"), 0.0))

    for ev in scaled_events:
        data = ev.get("data") or {}
        group = str(data.get("group") or "unknown").upper()
        symbol = str(ev.get("symbol") or data.get("symbol") or "unknown").upper()
        scaled_by_group[group] += 1
        scaled_by_symbol[symbol] += 1

    lines = [f"Correlation group guard dashboard (last {len(events)} corr_group_cap events)", ""]

    group_rows = []
    for group, count in group_counts.most_common(args.top):
        avg_notional = sum(group_notional[group]) / max(1, len(group_notional[group]))
        avg_proj = sum(projected_notional[group]) / max(1, len(projected_notional[group]))
        group_rows.append((group, count, f"{avg_notional:.2f}", f"{avg_proj:.2f}"))
    lines.append(_format_table(group_rows, ("Group", "Count", "Avg Notional", "Avg Projected")))

    lines.append("")
    symbol_rows = [(sym, count) for sym, count in symbol_counts.most_common(args.top)]
    lines.append(_format_table(symbol_rows, ("Symbol", "Count")))

    summary = "\n".join(lines)
    if scaled_events:
        lines.append("")
        lines.append(f"Corr-group scaling events: {len(scaled_events)}")
        scaled_rows = [(g, c) for g, c in scaled_by_group.most_common(args.top)]
        lines.append(_format_table(scaled_rows, ("Group", "Scaled Count")))
        lines.append("")
        scaled_sym_rows = [(s, c) for s, c in scaled_by_symbol.most_common(args.top)]
        lines.append(_format_table(scaled_sym_rows, ("Symbol", "Scaled Count")))
        summary = "\n".join(lines)
    Path(args.output).write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
