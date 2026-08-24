#!/usr/bin/env python3
"""
Telemetry guard history.

Appends a combined row to `logs/telemetry_guard_history.csv` by extracting the
latest line items from the drift, signal/exit, and guard timeline artifacts.
Runs alongside the scheduler so you can chart the guard response and drift
signals over time without hand-assembling multiple outputs.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import time
from pathlib import Path
from typing import Any


HISTORY_COLUMNS = [
    "timestamp",
    "drift_count",
    "drift_details",
    "low_conf_ratio",
    "low_conf_total",
    "guard_exits",
    "override_active",
    "override_reason",
    "recent_guard_hit",
    "anomaly_notes",
    "drift_notes",
    "signal_context",
    "partial_retry_hits",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_drift_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"count": 0, "details": [], "note": ""}
    if not path.exists():
        return result
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    capture = False
    for line in lines:
        if "Detected drifts" in line:
            capture = True
            continue
        if capture:
            if line.startswith("-"):
                result["details"].append(line.lstrip("- ").strip())
            else:
                capture = False
        if "No drift detected" in line:
            result["note"] = line
    result["count"] = len(result["details"])
    return result


def _parse_signal_exit(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ratio": 0.0,
        "low_conf": 0,
        "total": 0,
        "guard_exits": 0,
        "context": [],
    }
    if not path.exists():
        return result
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    in_context = False
    for line in lines:
        if line == "=== Signal exit context ===":
            in_context = True
            continue
        if in_context:
            if not line:
                in_context = False
                continue
            if line.startswith("-"):
                text = line.lstrip("- ").strip()
                result["context"].append(text)
                low_conf_match = re.match(
                    r"low-confidence exits: (\d+)/(\d+) \([^)]+ratio ([\d.]+)%\)",
                    text,
                    re.IGNORECASE,
                )
                if low_conf_match:
                    result["low_conf"] = int(low_conf_match.group(1))
                    result["total"] = int(low_conf_match.group(2))
                    result["ratio"] = _safe_float(low_conf_match.group(3))
                elif text.lower().startswith("guard exits"):
                    guard_val = re.search(r"guard exits[: ]+(\d+)", text, re.IGNORECASE)
                    if guard_val:
                        result["guard_exits"] = int(guard_val.group(1))
            else:
                in_context = False
    return result


def _parse_guard_timeline(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "events": "",
        "override_active": "",
        "override_reason": "",
        "recent_hits": [],
        "anomaly_notes": [],
        "drift_notes": [],
    }
    if not path.exists():
        return result
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    reading_recent = False
    reading_anomaly = False
    reading_drift = False
    for line in lines:
        if line.startswith("Events in window:"):
            result["events"] = line.split(":", 1)[1].strip()
            reading_recent = reading_anomaly = reading_drift = False
            continue
        if line.startswith("Recovery override active:"):
            result["override_active"] = line.split(":", 1)[1].strip()
            reading_recent = reading_anomaly = reading_drift = False
            continue
        if line.startswith("Override:"):
            result["override_reason"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Recent guard hits:"):
            reading_recent = True
            reading_anomaly = reading_drift = False
            continue
        if line.startswith("Anomaly action context:"):
            reading_anomaly = True
            reading_recent = reading_drift = False
            continue
        if line.startswith("Drift summary:"):
            reading_drift = True
            reading_recent = reading_anomaly = False
            continue
        if reading_recent and line.startswith("-"):
            result["recent_hits"].append(line.lstrip("- ").strip())
        elif reading_anomaly and line.startswith("-"):
            result["anomaly_notes"].append(line.lstrip("- ").strip())
        elif reading_drift and line.startswith("-"):
            result["drift_notes"].append(line.lstrip("- ").strip())
        else:
            reading_recent = reading_anomaly = reading_drift = False
    return result


def _append_history_row(csv_path: Path, row: dict[str, Any]) -> None:
    exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_html(csv_path: Path, html_path: Path, limit: int = 32) -> None:
    if not csv_path.exists():
        return
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    if not rows:
        return
    rows = rows[-limit:]
    try:
        values = [int(float(row.get("partial_retry_hits", "0") or 0)) for row in rows]
    except Exception:
        values = [0 for _ in rows]
    max_value = max(values) if values else 0
    bar_html = ""
    if values:
        bar_segments = []
        for val in values:
            height = (val / max_value * 48) + 4 if max_value > 0 else 4
            bar_segments.append(
                f"<span class=\"sparkline-bar\" style=\"height:{height:.1f}px\" title=\"{val} hit(s)\"></span>"
            )
        bar_html = (
            "<div class=\"sparkline\"><span class=\"sparkline-label\">Partial/retry hits</span>"
            f"<div class=\"sparkline-bars\">{''.join(bar_segments)}</div></div>"
        )
    body_lines = ["<tr><th>" + "</th><th>".join(html.escape(col) for col in HISTORY_COLUMNS) + "</th></tr>"]
    for row in rows:
        cols = [html.escape(str(row.get(col, ""))) for col in HISTORY_COLUMNS]
        body_lines.append("<tr><td>" + "</td><td>".join(cols) + "</td></tr>")
    html_content = f"""<!doctype html>
<html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Telemetry Guard History</title>
        <style>
          body {{ font-family: system-ui, sans-serif; background:#0f1118; color:#f7f7f7; padding:1rem; }}
          table {{ width:100%; border-collapse:collapse; margin-top:1rem; }}
          th, td {{ border:1px solid #1f2536; padding:0.25rem; text-align:left; }}
          th {{ background:#1b2333; }}
          tr:nth-child(even) td {{ background:#151c2b; }}
          .sparkline {{ display:flex; align-items:flex-end; gap:0.5rem; margin-top:0.5rem; }}
          .sparkline-label {{ font-size:0.9rem; color:#aaaaaa; }}
          .sparkline-bars {{ display:flex; gap:0.2rem; height:54px; align-items:flex-end; }}
          .sparkline-bar {{ width:10px; display:inline-block; background:#58dfb5; border-radius:3px 3px 0 0; }}
        </style>
      </head>
  <body>
    <h1>Telemetry Guard History</h1>
    {bar_html}
    <table>
      {"".join(body_lines)}
    </table>
  </body>
</html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_content, encoding="utf-8")


KEYWORDS = ("partial_fill", "retry_alert")


def _count_partial_retry_hits(guard: dict[str, Any]) -> int:
    hits = 0
    override = str(guard.get("override_reason") or "").lower()
    if any(kw in override for kw in KEYWORDS):
        hits += 1
    for desc in guard.get("recent_hits", []):
        text = str(desc or "").lower()
        if any(kw in text for kw in KEYWORDS):
            hits += 1
    return hits


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry guard + drift history tracker")
    parser.add_argument("--drift-summary", default="logs/telemetry_drift_summary.txt")
    parser.add_argument("--signal-exit", default="logs/signal_exit_notify.txt")
    parser.add_argument("--guard-timeline", default="logs/telemetry_guard_timeline.txt")
    parser.add_argument("--output-csv", default="logs/telemetry_guard_history.csv")
    parser.add_argument("--output-html", default="logs/telemetry_guard_history.html")
    parser.add_argument("--limit", type=int, default=32, help="Rows to show in the HTML")
    args = parser.parse_args(argv)

    drift = _parse_drift_summary(Path(args.drift_summary))
    signal = _parse_signal_exit(Path(args.signal_exit))
    guard = _parse_guard_timeline(Path(args.guard_timeline))

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    row = {
        "timestamp": timestamp,
        "drift_count": drift.get("count", 0),
        "drift_details": "; ".join(drift.get("details", [])[:3]) or drift.get("note", ""),
        "low_conf_ratio": signal.get("ratio", 0.0),
        "low_conf_total": signal.get("total", 0),
        "guard_exits": signal.get("guard_exits", 0),
        "override_active": guard.get("override_active", ""),
        "override_reason": guard.get("override_reason", ""),
        "recent_guard_hit": "; ".join(guard.get("recent_hits", [])[:2]),
        "anomaly_notes": "; ".join(guard.get("anomaly_notes", [])[:2]),
        "drift_notes": "; ".join(guard.get("drift_notes", [])[:2]),
        "signal_context": "; ".join(signal.get("context", [])[:4]),
        "partial_retry_hits": _count_partial_retry_hits(guard),
    }

    _append_history_row(Path(args.output_csv), row)
    _write_html(Path(args.output_csv), Path(args.output_html), limit=max(1, args.limit))
    print(f"Appended guard history row and updated {args.output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
