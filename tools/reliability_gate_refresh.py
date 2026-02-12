#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from tools import reliability_gate as gate  # type: ignore
except Exception:
    from eclipse_scalper.tools import reliability_gate as gate  # type: ignore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh reliability gate file using env-driven defaults.")
    ap.add_argument("--telemetry", default=os.getenv("TELEMETRY_PATH", "logs/telemetry.jsonl"))
    ap.add_argument("--journal", default=os.getenv("EVENT_JOURNAL_PATH", "logs/execution_journal.jsonl"))
    ap.add_argument("--output", default=os.getenv("RELIABILITY_GATE_PATH", "logs/reliability_gate.txt"))
    ap.add_argument(
        "--window-seconds",
        type=float,
        default=float(os.getenv("RELIABILITY_GATE_WINDOW_SECONDS", "14400") or 14400.0),
    )
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "--telemetry",
        str(args.telemetry),
        "--journal",
        str(args.journal),
        "--output",
        str(args.output),
        "--window-seconds",
        str(max(0.0, float(args.window_seconds))),
    ]
    if bool(args.allow_missing):
        cmd.append("--allow-missing")
    return int(gate.main(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
