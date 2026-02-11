#!/usr/bin/env python3
"""
Telemetry dashboard page.

Produces a simple HTML summary combining the core health, signal data health,
and anomaly reports so you can open a single dashboard page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            try:
                return path.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                return ""


def _read_section(path: Path, title: str) -> str:
    if not path.exists():
        return f"<h2>{title}</h2><p><em>Missing {path.name}</em></p>"
    text = _read_text_safe(path)
    if not text:
        return f"<h2>{title}</h2><p><em>Empty {path.name}</em></p>"
    return f"<h2>{title}</h2><pre>{text}</pre>"


def _read_json_section(path: Path, title: str) -> str:
    if not path.exists():
        return f"<h2>{title}</h2><p><em>Missing {path.name}</em></p>"
    raw = _read_text_safe(path)
    if not raw:
        return f"<h2>{title}</h2><p><em>Empty {path.name}</em></p>"
    try:
        payload = json.loads(raw)
        text = json.dumps(payload, indent=2)
    except Exception:
        text = raw
    return f"<h2>{title}</h2><pre>{text}</pre>"


def _read_csv_section(path: Path, title: str, limit: int = 5) -> str:
    if not path.exists():
        return f"<h2>{title}</h2><p><em>Missing {path.name}</em></p>"
    lines = _read_text_safe(path).splitlines()
    if not lines:
        return f"<h2>{title}</h2><p><em>Empty {path.name}</em></p>"
    header = lines[0]
    rows = lines[1:]
    tail = rows[-limit:] if rows else []
    text = "\n".join([header, *tail])
    return f"<h2>{title}</h2><pre>{text}</pre>"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            events.append(json.loads(s))
        except Exception:
            continue
    return events


def _guard_reason_summary(events: list[dict], limit: int = 8) -> str:
    guard_events = {
        "entry.blocked",
        "exit.telemetry_guard",
        "order.create.retry_alert",
        "entry.partial_fill_escalation",
    }
    counts: Counter[str] = Counter()
    for ev in events:
        name = str(ev.get("event") or "")
        if name not in guard_events:
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        reason = (
            str(data.get("reason") or data.get("guard") or data.get("code") or "").strip().lower()
        )
        label = reason if reason else name
        counts[label] += 1
    if not counts:
        return "<h2>Top Guard Reasons</h2><p><em>No guard events found.</em></p>"
    lines = ["Top Guard Reasons", "-" * 18]
    for label, cnt in counts.most_common(limit):
        lines.append(f"{label}: {cnt}")
    return f"<h2>Top Guard Reasons</h2><pre>{'\n'.join(lines)}</pre>"


def _guard_symbol_spark(events: list[dict], limit: int = 8, width: int = 20) -> str:
    guard_events = {
        "entry.blocked",
        "exit.telemetry_guard",
        "order.create.retry_alert",
        "entry.partial_fill_escalation",
    }
    counts: Counter[str] = Counter()
    for ev in events:
        name = str(ev.get("event") or "")
        if name not in guard_events:
            continue
        symbol = str(ev.get("symbol") or "")
        if not symbol:
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            symbol = str(data.get("symbol") or "")
        symbol = symbol.strip().upper() if symbol else "UNKNOWN"
        counts[symbol] += 1
    if not counts:
        return "<h2>Guard Hits By Symbol</h2><p><em>No guard events found.</em></p>"
    max_count = max(counts.values()) if counts else 1
    lines = ["Guard Hits By Symbol", "-" * 20]
    for symbol, cnt in counts.most_common(limit):
        bar_len = max(1, int(round((cnt / max_count) * width))) if cnt > 0 else 0
        bar = "#" * bar_len
        lines.append(f"{symbol:10} | {bar} {cnt}")
    return f"<h2>Guard Hits By Symbol</h2><pre>{'\n'.join(lines)}</pre>"


def _exit_quality_delta(path: Path) -> str:
    if not path.exists():
        return "<h2>Exit Quality Deltas</h2><p><em>Missing exit_quality_summary.json</em></p>"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "<h2>Exit Quality Deltas</h2><p><em>Invalid exit_quality_summary.json</em></p>"
    w24 = payload.get("window_24h", {}) if isinstance(payload, dict) else {}
    w7 = payload.get("window_7d", {}) if isinstance(payload, dict) else {}
    win24 = float(w24.get("win_rate") or 0.0)
    win7 = float(w7.get("win_rate") or 0.0)
    pnl24 = float(w24.get("avg_pnl") or 0.0)
    pnl7 = float(w7.get("avg_pnl") or 0.0)
    win_delta = win24 - win7
    pnl_delta = pnl24 - pnl7
    lines = [
        "Exit Quality Deltas (24h vs 7d)",
        "-" * 34,
        f"win_rate: {win24:.1%} vs {win7:.1%} (delta {win_delta:+.1%})",
        f"avg_pnl: {pnl24:.4f} vs {pnl7:.4f} (delta {pnl_delta:+.4f})",
    ]
    return f"<h2>Exit Quality Deltas</h2><pre>{'\n'.join(lines)}</pre>"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry dashboard HTML page")
    parser.add_argument("--core", default="logs/core_health.txt", help="Core health text")
    parser.add_argument("--signal", default="logs/signal_data_health.txt", help="Signal data health text")
    parser.add_argument("--anomaly", default="logs/telemetry_anomaly.txt", help="Anomaly report text")
    parser.add_argument("--signal-exit", default="logs/signal_exit_notify.txt", help="Signal/exit notify text")
    parser.add_argument("--guard-timeline", default="logs/telemetry_guard_timeline.txt", help="Guard timeline text")
    parser.add_argument("--guard-history", default="logs/telemetry_guard_history.csv", help="Guard history CSV")
    parser.add_argument("--telemetry", default="logs/telemetry.jsonl", help="Telemetry JSONL")
    parser.add_argument(
        "--guard-actions",
        default="logs/telemetry_guard_history_actions.json",
        help="Guard history actions JSON",
    )
    parser.add_argument(
        "--signal-feedback",
        default="logs/signal_exit_feedback.json",
        help="Signal exit feedback JSON",
    )
    parser.add_argument("--exit-quality", default="logs/exit_quality.txt", help="Exit quality dashboard text")
    parser.add_argument("--exit-quality-json", default="logs/exit_quality_summary.json", help="Exit quality JSON")
    parser.add_argument("--corr-group", default="logs/telemetry_corr_group.txt", help="Correlation guard dashboard text")
    parser.add_argument("--output", default="logs/telemetry_dashboard_page.html")
    args = parser.parse_args(argv)

    telemetry_events = _load_jsonl(Path(args.telemetry))

    sections = [
        _read_section(Path(args.core), "Core Health"),
        _read_section(Path(args.signal), "Signal Data Health"),
        _read_section(Path(args.anomaly), "Anomaly Detector"),
        _read_section(Path(args.signal_exit), "Signal/Exit Notify"),
        _read_section(Path(args.guard_timeline), "Guard Timeline"),
        _read_csv_section(Path(args.guard_history), "Guard History (Recent)"),
        _guard_reason_summary(telemetry_events),
        _guard_symbol_spark(telemetry_events),
        _exit_quality_delta(Path(args.exit_quality_json)),
        _read_json_section(Path(args.guard_actions), "Guard History Actions"),
        _read_json_section(Path(args.signal_feedback), "Signal Exit Feedback"),
        _read_section(Path(args.exit_quality), "Exit Quality"),
        _read_section(Path(args.corr_group), "Correlation Guard"),
    ]
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Telemetry Dashboard</title>
    <style>
      body {{ font-family: system-ui, sans-serif; background:#101218; color:#f7f7f7; padding:2rem; }}
      pre {{ background:#181c2a; padding:1rem; border-radius:0.5rem; overflow:auto; }}
      h1 {{ margin-bottom:1rem; }}
    </style>
  </head>
  <body>
    <h1>Telemetry Dashboard Snapshot</h1>
    {"".join(sections)}
  </body>
</html>"""
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Telemetry dashboard page saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
