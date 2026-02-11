#!/usr/bin/env python3
"""
Telemetry recovery dashboard.

Summarizes recovery overrides, signal exit issues, and guard events so cron jobs
can publish a single artifact that shows whether recovery/autopilot mode is active.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
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
    now = time.time()
    cutoff = now - max(1, since_min) * 60
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            ts = _safe_float(ev.get("ts") or ev.get("time"), 0.0)
            if ts < cutoff:
                continue
            out.append(ev)
    return out


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _format_ts(ts: float) -> str:
    if ts <= 0:
        return "n/a"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _build_html(state: dict[str, Any], issues: list[dict[str, Any]], guards: list[dict[str, Any]]) -> str:
    active_until = _format_ts(_safe_float(state.get("expires_at"), 0.0))
    now = time.time()
    active = _safe_float(state.get("expires_at"), 0.0) > now
    guard_section = "".join(
        f"<li>{_format_ts(_safe_float(g.get('ts')))} – {g.get('summary')}</li>" for g in guards
    )
    issue_rows = "".join(
        "<tr><td>{symbol}</td><td>{reason}</td><td>{confidence}</td><td>{guard}</td><td>{exposures}</td><td>{ts}</td></tr>".format(
            symbol=g.get("symbol", "n/a"),
            reason=g.get("reason", "n/a"),
            confidence=g.get("confidence", "n/a"),
            guard=g.get("guard", "none"),
            exposures=f"{_safe_float(g.get('exposures')):.0f}",
            ts=_format_ts(_safe_float(g.get("ts"))),
        )
        for g in issues
    )
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Telemetry Recovery Dashboard</title>
    <style>
      body {{ font-family: system-ui, sans-serif; background:#0f1118; color:#f1f1f1; padding:2rem; }}
      section {{ margin-bottom:2rem; }}
      pre {{ background:#141822; padding:1rem; border-radius:0.5rem; overflow:auto; }}
      table {{ width:100%; border-collapse:collapse; margin-top:0.5rem; }}
      th, td {{ border:1px solid #1d2230; padding:0.4rem; text-align:left; }}
      th {{ background:#1d2230; }}
    </style>
  </head>
  <body>
    <h1>Telemetry Recovery Dashboard</h1>
    <section>
      <h2>Recovery override</h2>
      <p>Active: <strong>{'yes' if active else 'no'}</strong></p>
      <p>Expires: <strong>{active_until}</strong></p>
      <pre>{json.dumps(state, indent=2)}</pre>
    </section>
    <section>
      <h2>Latest exit.signal_issue</h2>
      <table>
        <tr><th>Symbol</th><th>Reason</th><th>Confidence</th><th>Guard</th><th>Exposure</th><th>Timestamp</th></tr>
        {issue_rows or '<tr><td colspan="6">No signal issues in window</td></tr>'}
      </table>
    </section>
    <section>
      <h2>Telemetry guard events (recent)</h2>
      <ul>
        {guard_section or '<li>None</li>'}
      </ul>
    </section>
  </body>
</html>"""
    return html


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry recovery dashboard")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--state", default="logs/telemetry_recovery_state.json", help="Recovery state file")
    parser.add_argument("--since-min", type=int, default=60, help="Minutes to include")
    parser.add_argument("--output-html", default="logs/telemetry_recovery_dashboard.html", help="HTML output path")
    parser.add_argument("--output-txt", default="logs/telemetry_recovery_report.txt", help="Text summary path")
    args = parser.parse_args(argv)

    events = _load_events(Path(args.path), args.since_min)
    state = _load_state(Path(args.state))
    issues = []
    guards = []
    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") or {}
        if name == "exit.signal_issue":
            issues.append(
                {
                    "symbol": str(data.get("symbol") or ev.get("symbol") or "unknown").upper(),
                    "reason": str(data.get("reason") or "signal_issue"),
                    "confidence": data.get("low_confidence_ratio", "n/a"),
                    "guard": str(data.get("severity") or "n/a"),
                    "exposures": _safe_float(data.get("exposures")),
                    "ts": _safe_float(ev.get("ts") or 0.0),
                }
            )
        if name == "exit.telemetry_guard":
            guards.append(
                {
                    "summary": f"{str(ev.get('symbol') or data.get('symbol') or 'unknown')} guard {data.get('reason')}",
                    "ts": _safe_float(ev.get("ts") or 0.0),
                }
            )
    html = _build_html(state, issues[-10:], guards[-10:])
    Path(args.output_html).write_text(html, encoding="utf-8")

    lines = [
        f"Telemetry recovery dashboard ({datetime.now(timezone.utc).isoformat()} UTC)",
        f"Recovery active: {'yes' if _safe_float(state.get('expires_at'), 0.0) > time.time() else 'no'}",
        f"# of signal issues in last {args.since_min} min: {len(issues)}",
        f"# of guard events: {len(guards)}",
        f"State file: {args.state}",
    ]
    Path(args.output_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Dashboard saved to {args.output_html}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
