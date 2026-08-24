#!/usr/bin/env python3
"""
Telemetry confidence drift detection.

Compares the early/late halves of the recent entry confidence window per symbol.
If the mean jumps by more than `--zscore` * baseline_std, it flags the symbol/drift.
Optionally appends a `telemetry.confidence_drift` event for downstream alerts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _load_events(path: Path, since_min: int) -> list[Dict[str, Any]]:
    now = time.time()
    cutoff = now - max(1, since_min) * 60
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            ts = _safe_float(ev.get("ts") or ev.get("time"))
            if ts < cutoff:
                continue
            out.append(ev)
    return out


def _stat_window(rows: List[float]) -> Tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    mean = statistics.mean(rows)
    stdev = statistics.pstdev(rows) if len(rows) > 1 else 0.0
    return mean, stdev


def _emit_event(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _format_ts(ts: float) -> str:
    if ts <= 0:
        return "n/a"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _build_summary(drifts: List[Dict[str, Any]], since_min: int, path: Path) -> str:
    lines = [
        f"Telemetry confidence drift detection (last {since_min} min)",
    ]
    if not drifts:
        lines.append("No drift detected.")
        return "\n".join(lines)

    lines.append("Detected drifts:")
    for d in drifts:
        lines.append(
            f"- {d['symbol']} (reason={d['reason']}) mean {d['current_mean']:.3f} vs baseline {d['baseline_mean']:.3f} "
            f"z={d['zscore']:.2f} at {_format_ts(d['last_ts'])}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry confidence drift detection")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=60, help="Minutes to include")
    parser.add_argument("--min-count", type=int, default=6, help="Min samples per half")
    parser.add_argument("--zscore", type=float, default=2.0, help="Z-score threshold")
    parser.add_argument("--summary", default="logs/telemetry_drift_summary.txt", help="Summary path")
    parser.add_argument("--emit-event", action="store_true", help="Append telemetry.confidence_drift event")
    parser.add_argument("--event-path", default="logs/telemetry_drift.jsonl", help="Feedback event path")
    args = parser.parse_args(argv)

    events = _load_events(Path(args.path), args.since_min)
    scores: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for ev in events:
        if str(ev.get("event")) != "entry.submitted":
            continue
        symbol = str(ev.get("symbol") or ev.get("data", {}).get("symbol") or "").upper()
        if not symbol:
            continue
        data = ev.get("data") or {}
        conf = _safe_float(data.get("confidence") or data.get("conf"), -1.0)
        if conf < 0:
            continue
        ts = _safe_float(ev.get("ts") or 0.0)
        scores[symbol].append((ts, conf))

    drifts = []
    for symbol, rows in scores.items():
        if len(rows) < args.min_count * 2:
            continue
        rows.sort(key=lambda x: x[0])
        half = len(rows) // 2
        baseline = [conf for _, conf in rows[:half]]
        current = [conf for _, conf in rows[half:]]
        if len(baseline) < args.min_count or len(current) < args.min_count:
            continue
        baseline_mean, baseline_stdev = _stat_window(baseline)
        current_mean, _ = _stat_window(current)
        if baseline_stdev <= 0:
            continue
        zscore = abs((current_mean - baseline_mean) / baseline_stdev)
        if zscore >= args.zscore:
            drifts.append(
                {
                    "symbol": symbol,
                    "baseline_mean": baseline_mean,
                    "current_mean": current_mean,
                    "baseline_stdev": baseline_stdev,
                    "zscore": zscore,
                    "last_ts": rows[-1][0],
                    "reason": f"z{zscore:.2f}",
                }
            )
            if args.emit_event:
                _emit_event(
                    Path(args.event_path),
                    {
                        "ts": rows[-1][0],
                        "event": "telemetry.confidence_drift",
                        "data": {
                            "symbol": symbol,
                            "baseline_mean": baseline_mean,
                            "current_mean": current_mean,
                            "baseline_stdev": baseline_stdev,
                            "zscore": zscore,
                            "since_min": args.since_min,
                        },
                    },
                )

    summary = _build_summary(drifts, args.since_min, Path(args.summary))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
