#!/usr/bin/env python3
"""
Telemetry dashboard page.

Produces a simple HTML summary combining the core health, signal data health,
and anomaly reports so you can open a single dashboard page.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _read_section(path: Path, title: str) -> str:
    if not path.exists():
        return f"<h2>{title}</h2><p><em>Missing {path.name}</em></p>"
    text = path.read_text(encoding="utf-8")
    return f"<h2>{title}</h2><pre>{text}</pre>"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry dashboard HTML page")
    parser.add_argument("--core", default="logs/core_health.txt", help="Core health text")
    parser.add_argument("--signal", default="logs/signal_data_health.txt", help="Signal data health text")
    parser.add_argument("--anomaly", default="logs/telemetry_anomaly.txt", help="Anomaly report text")
    parser.add_argument("--output", default="logs/telemetry_dashboard_page.html")
    args = parser.parse_args(argv)

    sections = [
        _read_section(Path(args.core), "Core Health"),
        _read_section(Path(args.signal), "Signal Data Health"),
        _read_section(Path(args.anomaly), "Anomaly Detector"),
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
