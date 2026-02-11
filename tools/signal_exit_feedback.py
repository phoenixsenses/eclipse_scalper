#!/usr/bin/env python3
"""
Signal exit feedback helper.

Summarizes low-confidence exits plus telemetry guard closures and optionally writes a
feedback event (`exit.signal_issue`) back into the telemetry log so the classifier/guards can
adjust sensitivity for the next run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from execution.telemetry_recovery import write_recovery_state  # type: ignore
except Exception:
    write_recovery_state = None


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
    out: list[dict[str, Any]] = []
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
            ts = _safe_float(ev.get("ts"), 0.0)
            if ts < cutoff:
                continue
            out.append(ev)
    return out


def _collect_exit_stats(events: Iterable[dict[str, Any]], min_conf: float) -> dict[str, Any]:
    exit_events = [
        ev for ev in events if isinstance(ev.get("event"), str) and ev["event"].startswith("exit.")
    ]
    total = len(exit_events)
    low_conf = []
    guard_hits = []
    reasons = Counter()
    symbols = Counter()
    confidences = []
    ages = []

    for ev in exit_events:
        data = ev.get("data") or {}
        symbol = str(data.get("symbol") or ev.get("symbol") or "unknown").upper()
        reason = ev.get("event", "exit").split(".", 1)[-1]
        conf = _safe_float(data.get("entry_confidence"), -1.0)
        age = _safe_float(data.get("entry_signal_age_sec"), -1.0)
        guard = str(data.get("guard") or "").strip().lower()

        reasons[reason] += 1
        symbols[symbol] += 1

        if conf >= 0:
            confidences.append(conf)
        if age >= 0:
            ages.append(age)

        if conf >= 0 and conf < min_conf:
            low_conf.append((symbol, reason, conf, age))
        elif conf < 0:
            low_conf.append((symbol, reason, None, age))

        if guard:
            guard_hits.append({"symbol": symbol, "reason": reason, "guard": guard, "exposures": _safe_float(data.get("exposures"))})

    ratio = (len(low_conf) / total) if total > 0 else 0.0
    avg_confidence = statistics.mean(confidences) if confidences else None
    avg_age = statistics.mean(ages) if ages else None

    return {
        "total": total,
        "low_conf": len(low_conf),
        "low_conf_details": low_conf,
        "ratio": ratio,
        "guard_hits": guard_hits,
        "reasons": reasons,
        "symbols": symbols,
        "avg_confidence": avg_confidence,
        "avg_age": avg_age,
    }


def _format_summary(stats: dict[str, Any], min_conf: float, top: int) -> str:
    lines = []
    lines.append(f"Signal exit feedback (low-confidence threshold < {min_conf:.2f})")
    lines.append(f"- total exits: {stats['total']}")
    lines.append(f"- low-confidence exits: {stats['low_conf']} ({stats['ratio']:.1%})")
    lines.append(f"- telemetry guard hits: {len(stats['guard_hits'])}")
    if stats["avg_confidence"] is not None:
        lines.append(f"- avg entry confidence (with data): {stats['avg_confidence']:.2f}")
    if stats["avg_age"] is not None:
        lines.append(f"- avg signal age: {stats['avg_age']:.1f}s")
    if stats["reasons"]:
        rows = []
        for reason, count in stats["reasons"].most_common(top):
            rows.append(f"  - {reason}: {count}")
        lines.append("- top exit reasons:")
        lines.extend(rows)
    if stats["symbols"]:
        rows = []
        for symbol, count in stats["symbols"].most_common(top):
            rows.append(f"  - {symbol}: {count}")
        lines.append("- top symbols with exits:")
        lines.extend(rows)
    if stats["guard_hits"]:
        lines.append("- guard exits (first few):")
        for guard in stats["guard_hits"][:min(top, len(stats["guard_hits"]))]:
            lines.append(
                f"  - {guard['symbol']} {guard['reason']} guard={guard['guard']} exposures={guard['exposures']:.0f}"
            )
    if stats.get("low_conf_details"):
        lines.append("- low-confidence exit samples (first few):")
        for symbol, reason, conf, age in stats["low_conf_details"][:min(top, len(stats["low_conf_details"]))]:
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else "n/a"
            lines.append(f"  - {symbol} {reason} conf={conf_str} age={age:.1f}s")
    return "\n".join(lines)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass


def _emit_feedback_event(
    telemetry_path: Path, stats: dict[str, Any], min_conf: float, severity: str = "info"
) -> None:
    event = {
        "ts": time.time(),
        "event": "exit.signal_issue",
        "data": {
            "total_exits": stats["total"],
            "low_confidence_exits": stats["low_conf"],
            "low_conf_ratio": stats["ratio"],
            "avg_confidence": stats["avg_confidence"],
            "avg_age_sec": stats["avg_age"],
            "min_confidence_threshold": min_conf,
            "guard_hits": len(stats["guard_hits"]),
            "severity": severity,
        },
    }
    try:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with telemetry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Signal exit feedback")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=60, help="Minutes to include")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Low-confidence threshold")
    parser.add_argument("--top", type=int, default=4, help="Top reasons/symbols to print")
    parser.add_argument("--issue-ratio", type=float, default=0.25, help="Ratio of low-confidence exits to trigger feedback")
    parser.add_argument("--issue-count", type=int, default=3, help="Minimum low-confidence exits to trigger feedback")
    parser.add_argument("--summary", default="logs/signal_exit_feedback.json", help="JSON summary output path")
    parser.add_argument("--emit-event", action="store_true", help="Append exit.signal_issue telemetry event when feedback triggers")
    parser.add_argument(
        "--severity",
        choices=("info", "warning", "critical"),
        default="warning",
        help="Severity for the emitted feedback event",
    )
    parser.add_argument(
        "--recovery-min-confidence",
        type=float,
        default=0.0,
        help="Min confidence override when feedback triggers (0 to disable)",
    )
    parser.add_argument(
        "--recovery-duration",
        type=float,
        default=0.0,
        help="Seconds to keep the recovery override active (0 to disable)",
    )
    args = parser.parse_args(argv)

    events = _load_events(Path(args.path), args.since_min)
    stats = _collect_exit_stats(events, args.min_confidence)
    print(_format_summary(stats, args.min_confidence, args.top))
    summary = {
        "generated_at": time.time(),
        "min_confidence": args.min_confidence,
        "total_exits": stats["total"],
        "low_confidence_exits": stats["low_conf"],
        "low_confidence_ratio": stats["ratio"],
        "avg_confidence": stats["avg_confidence"],
        "avg_age_sec": stats["avg_age"],
        "guard_exits": len(stats["guard_hits"]),
        "low_confidence_symbols": [
            {"symbol": symbol, "count": int(count)}
            for symbol, count in stats["symbols"].most_common(3)
        ],
        "low_confidence_samples": [
            {
                "symbol": symbol,
                "reason": reason,
                "confidence": conf if isinstance(conf, float) else None,
                "age_sec": age,
            }
            for symbol, reason, conf, age in stats["low_conf_details"][:3]
        ],
    }
    _write_summary(Path(args.summary), summary)

    if stats["total"] > 0 and stats["low_conf"] >= args.issue_count and stats["ratio"] >= args.issue_ratio:
        if args.emit_event:
            _emit_feedback_event(Path(args.path), stats, args.min_confidence, args.severity)
        print(f"\nFeedback triggered: low-confidence exits ratio {stats['ratio']:.1%} >= {args.issue_ratio:.1%}")
        if (
            write_recovery_state is not None
            and args.recovery_min_confidence > 0
            and args.recovery_duration > 0
        ):
            write_recovery_state(
                args.recovery_min_confidence,
                args.recovery_duration,
                reason=f"low_conf_ratio {stats['ratio']:.1%}",
                severity=args.severity,
                extra={
                    "low_confidence_ratio": stats["ratio"],
                    "low_confidence_count": stats["low_conf"],
                },
            )
            print(f"Recovery override written: min_conf {args.recovery_min_confidence:.2f} for {args.recovery_duration:.0f}s")
    else:
        print("\nFeedback not triggered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
