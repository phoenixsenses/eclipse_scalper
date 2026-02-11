#!/usr/bin/env python3
"""
Exit quality dashboard.

Summarizes `position.closed` telemetry events so you can monitor exit outcomes
per symbol (PnL, duration, exit reasons).
"""

from __future__ import annotations

import argparse
import json
import statistics
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
            if str(ev.get("event")) != "position.closed":
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


def _summarize(events: list[dict[str, Any]], top: int) -> str:
    if not events:
        return "Exit quality dashboard (no position.closed events found)."

    per_symbol = defaultdict(list)
    reasons = Counter()
    for ev in events:
        data = ev.get("data") or {}
        symbol = str(ev.get("symbol") or data.get("symbol") or "unknown").upper()
        per_symbol[symbol].append(data)
        reason = str(data.get("exit_reason") or data.get("reason") or "unknown")
        reasons[reason] += 1

    rows = []
    for sym, samples in per_symbol.items():
        pnls = [_safe_float(s.get("pnl_usdt"), 0.0) for s in samples]
        pct = [_safe_float(s.get("pnl_pct"), 0.0) for s in samples]
        durations = [_safe_float(s.get("duration_sec"), 0.0) for s in samples]
        wins = sum(1 for p in pnls if p > 0)
        total = len(samples)
        avg_pnl = statistics.mean(pnls) if pnls else 0.0
        avg_pct = statistics.mean(pct) if pct else 0.0
        avg_dur = statistics.mean(durations) if durations else 0.0
        win_rate = (wins / total) if total else 0.0
        rows.append(
            (
                sym,
                total,
                f"{avg_pnl:+.2f}",
                f"{avg_pct:+.2%}",
                f"{avg_dur:.0f}s",
                f"{win_rate:.0%}",
            )
        )

    rows.sort(key=lambda r: int(r[1]), reverse=True)
    top_rows = rows[: max(1, top)]

    lines = [
        f"Exit quality dashboard (last {len(events)} position.closed events)",
        "",
        _format_table(top_rows, ("Symbol", "Count", "Avg PnL", "Avg PnL%", "Avg Dur", "Win%")),
    ]
    if reasons:
        lines.append("")
        lines.append("Top exit reasons:")
        for reason, count in reasons.most_common(6):
            lines.append(f"- {reason}: {count}")
    return "\n".join(lines)


def _summarize_window(events: list[dict[str, Any]]) -> dict:
    pnls = []
    pnl_pcts = []
    durations = []
    wins = 0
    for ev in events:
        data = ev.get("data") or {}
        pnl = _safe_float(data.get("pnl_usdt"), 0.0)
        pnl_pct = _safe_float(data.get("pnl_pct"), 0.0)
        duration = _safe_float(data.get("duration_sec"), 0.0)
        pnls.append(pnl)
        pnl_pcts.append(pnl_pct)
        durations.append(duration)
        if pnl > 0:
            wins += 1
    total = len(events)
    return {
        "total": total,
        "win_rate": (wins / total) if total else 0.0,
        "avg_pnl": (sum(pnls) / total) if total else 0.0,
        "avg_pnl_pct": (sum(pnl_pcts) / total) if total else 0.0,
        "avg_duration": (sum(durations) / total) if total else 0.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Exit quality dashboard")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=1440, help="Minutes to include")
    parser.add_argument("--limit", type=int, default=5000, help="Max events to load")
    parser.add_argument("--top", type=int, default=8, help="Top symbols to show")
    parser.add_argument("--output", default="logs/exit_quality.txt", help="Output text path")
    parser.add_argument(
        "--json",
        dest="summary_json",
        default="logs/exit_quality_summary.json",
        help="Structured summary path",
    )
    args = parser.parse_args(argv)

    events = _load_events(Path(args.path), args.since_min, limit=max(1, int(args.limit)))
    summary = _summarize(events, max(1, int(args.top)))
    now = time.time()
    last_24h = [ev for ev in events if _safe_float(ev.get("ts"), 0.0) >= now - 86400]
    last_7d = [ev for ev in events if _safe_float(ev.get("ts"), 0.0) >= now - 7 * 86400]
    if last_24h or last_7d:
        win_24 = _summarize_window(last_24h)
        win_7d = _summarize_window(last_7d)
        summary += (
            "\n\nWindow deltas:\n"
            f"- 24h: n={win_24['total']} win={win_24['win_rate']:.0%} "
            f"avg pnl={win_24['avg_pnl']:+.2f} avg pnl%={win_24['avg_pnl_pct']:+.2%} "
            f"avg dur={win_24['avg_duration']:.0f}s\n"
            f"- 7d:  n={win_7d['total']} win={win_7d['win_rate']:.0%} "
            f"avg pnl={win_7d['avg_pnl']:+.2f} avg pnl%={win_7d['avg_pnl_pct']:+.2%} "
            f"avg dur={win_7d['avg_duration']:.0f}s"
        )
    else:
        win_24 = _summarize_window([])
        win_7d = _summarize_window([])
    Path(args.output).write_text(summary + "\n", encoding="utf-8")
    symbol_counts = Counter(
        str(ev.get("symbol") or (ev.get("data") or {}).get("symbol") or "unknown").upper()
        for ev in events
    )
    summary_payload = {
        "generated_at": now,
        "total_events": len(events),
        "window_24h": win_24,
        "window_7d": win_7d,
        "top_symbols": symbol_counts.most_common(max(1, int(args.top))),
    }
    try:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
