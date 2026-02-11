#!/usr/bin/env python3
"""Run the signal/exit health helpers and notify Telegram via the shared Notifier."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _python_cmd() -> str:
    return sys.executable or "python"


def _run_helper(cmd: list[str]) -> tuple[str, int]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout.strip()
        if not output and proc.stderr:
            output = proc.stderr.strip()
        return output or "<no output>", proc.returncode
    except Exception as exc:
        return f"failed to run {' '.join(cmd)}: {exc}", -1


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


async def _notify(text: str, severity: str) -> None:
    notifier = _build_notifier()
    if notifier is None or not text.strip():
        return
    try:
        await notifier.speak(text, severity)
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Signal exit telemetry notifier")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--since-min", type=int, default=60, help="Lookback window (min)")
    parser.add_argument("--top", type=int, default=6, help="Top reasons/symbols")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Low-confidence threshold")
    parser.add_argument("--issue-ratio", type=float, default=0.25, help="Ratio to trigger feedback")
    parser.add_argument("--issue-count", type=int, default=3, help="Count to trigger feedback")
    parser.add_argument("--summary", default="logs/signal_exit_notify.txt", help="Summary path")
    parser.add_argument(
        "--feedback-summary-json",
        default="logs/signal_exit_feedback.json",
        help="JSON summary path written by signal_exit_feedback.py",
    )
    parser.add_argument("--recovery-min-confidence", type=float, default=0.7, help="Recovery override min confidence (0=disabled)")
    parser.add_argument("--recovery-duration", type=float, default=600, help="Recovery override duration (sec)")
    parser.add_argument("--emit-event", action="store_true", help="Emit exit.signal_issue event")
    args = parser.parse_args(argv)

    health_cmd = [
        _python_cmd(),
        str(ROOT / "tools" / "signal_exit_health.py"),
        "--path",
        args.path,
        "--since-min",
        str(args.since_min),
        "--top",
        str(args.top),
        "--min-confidence",
        str(args.min_confidence),
    ]
    feedback_cmd = [
        _python_cmd(),
        str(ROOT / "tools" / "signal_exit_feedback.py"),
        "--path",
        args.path,
        "--min-confidence",
        str(args.min_confidence),
        "--issue-ratio",
        str(args.issue_ratio),
        "--issue-count",
        str(args.issue_count),
    ]
    feedback_cmd.extend(
        ["--summary", args.feedback_summary_json]
    )
    if args.emit_event:
        feedback_cmd.append("--emit-event")
    if args.recovery_min_confidence > 0 and args.recovery_duration > 0:
        feedback_cmd.extend(
            [
                "--recovery-min-confidence",
                str(args.recovery_min_confidence),
                "--recovery-duration",
                str(args.recovery_duration),
            ]
        )

    health_out, _ = _run_helper(health_cmd)
    feedback_out, _ = _run_helper(feedback_cmd)

    feedback_summary = _load_json(Path(args.feedback_summary_json))

    summary_lines = [
        "Signal / Exit health notifier",
        "",
        "=== Signal exit health ===",
        health_out,
        "",
        "=== Signal exit feedback ===",
        feedback_out,
    ]

    if feedback_summary:
        ratio = float(feedback_summary.get("low_confidence_ratio") or 0.0)
        low_conf = int(feedback_summary.get("low_confidence_exits") or 0)
        total = int(feedback_summary.get("total_exits") or 0)
        guard_exits = int(feedback_summary.get("guard_exits") or 0)
        context_lines = [
            "",
            "=== Signal exit context ===",
            f"- low-confidence exits: {low_conf}/{total} (ratio {ratio:.1%})",
            f"- guard exits: {guard_exits}",
        ]
        top_symbols = feedback_summary.get("low_confidence_symbols", [])
        if top_symbols:
            symbol_summaries = ", ".join(
                f"{entry.get('symbol')}({entry.get('count')})" for entry in top_symbols
            )
            context_lines.append(f"- top low-confidence symbols: {symbol_summaries}")
        samples = feedback_summary.get("low_confidence_samples", [])
        if samples:
            sample = samples[0]
            conf = sample.get("confidence")
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else "n/a"
            age = sample.get("age_sec")
            age_str = f"{age:.1f}s" if isinstance(age, (int, float)) else "n/a"
            context_lines.append(
                f"- sample exit: {sample.get('symbol')} {sample.get('reason')} conf={conf_str} age={age_str}"
            )
        summary_lines.extend(context_lines)

    summary = "\n".join(summary_lines)

    Path(args.summary).write_text(summary + "\n", encoding="utf-8")
    severity = "info"
    if "Feedback triggered" in feedback_out:
        severity = "critical"
    if "Recovery override written" in feedback_out and severity != "critical":
        severity = "warning"

    asyncio.run(_notify(summary, severity))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
