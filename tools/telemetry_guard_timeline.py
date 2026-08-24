#!/usr/bin/env python3
"""
Telemetry guard timeline.

Builds a simple HTML/text timeline of entry-blocked reasons, recovery overrides,
and exit feedback events so scheduled jobs can publish a narrative of guard activity,
and optionally notifies Telegram with the latest guard context + anomaly/drift metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    cutoff = time.time() - max(1, since_min) * 60
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


def _load_anomaly_actions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _read_text_lines(path: Path, limit: int = 3) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return lines[:limit]
    except Exception:
        return []


def _load_recovery_state(path: Path) -> dict[str, Any]:
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


def _describe_event(ev: dict[str, Any]) -> dict[str, Any]:
    name = str(ev.get("event") or ev.get("type") or "")
    ts = _safe_float(ev.get("ts") or ev.get("time"))
    data = ev.get("data") or {}
    symbol = str(data.get("symbol") or ev.get("symbol") or "unknown")
    desc = ""
    if name == "entry.blocked":
        reason = str(data.get("reason") or "blocked")
        code = str(data.get("code") or "")
        if reason == "partial_fill":
            ratio = _safe_float(data.get("ratio"))
            desc = f"{symbol} partial_fill ratio={ratio:.2f} code={code}"
        else:
            desc = f"{symbol} blocked ({reason}) code={code}"
    elif name == "exit.signal_issue":
        ratio = data.get("low_confidence_ratio")
        desc = f"{symbol} signal_issue ratio={ratio if ratio is not None else 'n/a'}"
    elif name == "exit.telemetry_guard":
        desc = f"{symbol} telemetry_guard {data.get('reason')}"
    elif name == "order.create.retry_alert":
        tries = int(_safe_float(data.get("tries"), 0))
        variants = int(_safe_float(data.get("variants"), 0))
        desc = f"{symbol} retry_alert tries={tries} variants={variants}"
    else:
        desc = f"{symbol} {name}"
    return {
        "ts": ts,
        "name": name,
        "symbol": symbol,
        "desc": desc,
    }


def _build_timeline(events: list[dict[str, Any]], state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    timeline = []
    for ev in events:
        name = str(ev.get("event") or ev.get("type") or "")
        if name not in {"entry.blocked", "exit.signal_issue", "exit.telemetry_guard", "order.create.retry_alert"}:
            continue
        timeline.append(_describe_event(ev))
    timeline.sort(key=lambda x: x["ts"], reverse=True)
    if state:
        start = _safe_float(state.get("generated_at") or state.get("ts"))
        end = _safe_float(state.get("expires_at"))
        reason = str(state.get("reason") or "recovery override")
        timeline.insert(
            0,
            {
                "ts": start,
                "name": "recovery.override",
                "symbol": "system",
                "desc": f"override min_conf {state.get('min_confidence_override')} until {_format_ts(end)} ({reason})",
            },
        )
        timeline.insert(
            1,
            {
                "ts": end,
                "name": "recovery.expire",
                "symbol": "system",
                "desc": "override expires",
            },
        )
    return timeline[:limit]


def _build_html(timeline: list[dict[str, Any]], state: dict[str, Any]) -> str:
    rows = []
    for ev in timeline:
        rows.append(
            "<tr>"
            f"<td>{_format_ts(ev['ts'])}</td>"
            f"<td>{ev['name']}</td>"
            f"<td>{ev['symbol']}</td>"
            f"<td>{ev['desc']}</td>"
            "</tr>"
        )
    state_summary = json.dumps(state, indent=2) if state else "{}"
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Telemetry Guard Timeline</title>
    <style>
      body {{ font-family: system-ui, sans-serif; background:#0f1118; color:#f7f7f7; padding:2rem; }}
      table {{ width:100%; border-collapse:collapse; margin-top:1rem; }}
      th, td {{ border:1px solid #1f2536; padding:0.5rem; }}
      th {{ background:#1b2333; }}
      pre {{ background:#141924; padding:1rem; border-radius:0.5rem; }}
    </style>
  </head>
  <body>
    <h1>Telemetry Guard Timeline</h1>
    <section>
      <h2>Recovery override state</h2>
      <pre>{state_summary}</pre>
    </section>
    <section>
      <h2>Timeline (most recent first)</h2>
      <table>
        <tr><th>Time</th><th>Event</th><th>Symbol</th><th>Description</th></tr>
        {''.join(rows) if rows else '<tr><td colspan="4">No telemetry guard events in window.</td></tr>'}
      </table>
    </section>
  </body>
</html>"""
    return html


def _build_notifier():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return None
    try:
        from eclipse_scalper.notifications.telegram import Notifier

        return Notifier(token=token, chat_id=chat_id)
    except Exception:
        return None


async def _send_alert(notifier, text: str, severity: str) -> None:
    if notifier is None or not text.strip():
        return
    try:
        await notifier.speak(text, severity)
    except Exception:
        pass


def _determine_severity(arg_severity: str, timeline: list[dict[str, Any]], state: dict[str, Any]) -> str:
    if arg_severity and arg_severity.lower() != "auto":
        return arg_severity
    override_active = bool(state and _safe_float(state.get("expires_at")) > time.time())
    if override_active:
        return "critical"
    if timeline:
        return "warning"
    return "info"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry guard timeline")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--state", default="logs/telemetry_recovery_state.json", help="Recovery state JSON")
    parser.add_argument("--since-min", type=int, default=120, help="Lookback window (minutes)")
    parser.add_argument("--limit", type=int, default=40, help="Max timeline rows")
    parser.add_argument("--output-html", default="logs/telemetry_guard_timeline.html", help="HTML output path")
    parser.add_argument("--output-txt", default="logs/telemetry_guard_timeline.txt", help="Text summary path")
    parser.add_argument("--actions-path", default="logs/telemetry_anomaly_actions.json", help="Anomaly actions metadata")
    parser.add_argument("--drift-summary", default="logs/telemetry_drift_summary.txt", help="Drift summary path")
    parser.add_argument("--notify", action="store_true", help="Send timeline summary via Telegram")
    parser.add_argument("--severity", default="auto", help="Telegram severity (info/warning/critical or auto)")
    args = parser.parse_args(argv)

    events = _load_events(Path(args.path), args.since_min)
    state = _load_recovery_state(Path(args.state))
    timeline = _build_timeline(events, state, args.limit)
    html = _build_html(timeline, state)
    Path(args.output_html).write_text(html, encoding="utf-8")
    actions = _load_anomaly_actions(Path(args.actions_path))
    drift_lines = _read_text_lines(Path(args.drift_summary), limit=5)

    summary_lines = [
        f"Telemetry guard timeline ({datetime.now(timezone.utc).isoformat()} UTC)",
        f"Events in window: {len(timeline)}",
        f"Recovery override active: {'yes' if _safe_float(state.get('expires_at')) > time.time() else 'no'}",
    ]
    if state:
        override_reason = str(state.get("reason") or "unspecified")
        summary_lines.append(
            f"Override: min_conf={state.get('min_confidence_override')} reason={override_reason}"
        )
    recent = timeline[:3]
    if recent:
        summary_lines.append("Recent guard hits:")
        for ev in recent:
            summary_lines.append(f"- {ev['symbol']} {ev['name']} ({ev['desc']})")
    if actions:
        summary_lines.append("Anomaly action context:")
        pause_note = str(actions.get("pause_reason") or "none")
        summary_lines.append(f"- pause reason: {pause_note}")
        summary_lines.append(
            f"- classifier exit x{float(actions.get('classifier_exit_multiplier', 1.0)):.2f}, "
            f"confidence x{float(actions.get('classifier_confidence_multiplier', 1.0)):.2f}"
        )
        summary_lines.append(
            f"- exposures {float(actions.get('exposures', 0.0)):.0f}, avg_conf {float(actions.get('avg_confidence', 0.0)):.2f}"
        )
        if actions.get("anomaly_messages"):
            msgs = actions.get("anomaly_messages")[:3]
            summary_lines.append(f"- anomalies: {'; '.join(str(m) for m in msgs)}")
    if drift_lines:
        summary_lines.append("Drift summary:")
        summary_lines.extend(f"- {line}" for line in drift_lines)

    summary_text = "\n".join(summary_lines)
    Path(args.output_txt).write_text(summary_text + "\n", encoding="utf-8")
    print(f"Timeline saved to {args.output_html}")
    print(summary_text)

    if args.notify:
        notifier = _build_notifier()
        severity = _determine_severity(args.severity, timeline, state)
        asyncio.run(_send_alert(notifier, summary_text, severity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
