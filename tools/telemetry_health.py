#!/usr/bin/env python3
"""
Telemetry health panel.

Reads the classifier’s CSV history, plots adaptive multiplier / avg confidence /
exit counts over time, and writes an HTML snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import pandas as pd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry health panel (Altair)")
    parser.add_argument("--health-path", default="logs/telemetry_health.csv", help="CSV history path")
    parser.add_argument("--output", default="logs/telemetry_health.html", help="HTML output file")
    parser.add_argument("--max-rows", type=int, default=80, help="Max rows to visualize (tail)")
    args = parser.parse_args(argv)

    health_path = Path(args.health_path)
    if not health_path.exists():
        print(f"Missing health CSV: {health_path}")
        return 1

    df = pd.read_csv(health_path, parse_dates=["ts"])
    if df.empty:
        print("No health data yet.")
        return 1
    if args.max_rows and len(df) > args.max_rows:
        df = df.tail(args.max_rows)

    df["triggered"] = (
        df["triggered"].astype(str).map({"True": True, "False": False}).fillna(False)
    )
    df["new_codes"] = df["new_codes"].fillna("")

    base = alt.Chart(df).encode(x=alt.X("ts:T", title="Timestamp"))

    multiplier_line = base.mark_line(color="steelblue").encode(
        y=alt.Y("multiplier", title="Adaptive multiplier", scale=alt.Scale(domain=(0, 1.1)))
    )
    confidence_line = base.mark_line(color="firebrick").encode(
        y=alt.Y("avg_confidence", title="Average confidence", scale=alt.Scale(domain=(0, 1)))
    )

    multi_chart = (
        alt.layer(multiplier_line, confidence_line)
        .resolve_scale(y="independent")
        .properties(height=250, width=900, title="Multiplier vs. Confidence (tail)")
    )

    exit_chart = (
        base.mark_bar(color="darkorange")
        .encode(y=alt.Y("exit_total", title="Exit events"), tooltip=["exit_total"])
        .properties(height=200, width=900, title="Exit counts (per classifier run)")
    )

    trigger_points = (
        base.mark_circle(color="red", size=80)
        .transform_filter(alt.datum.triggered == True)
        .encode(y="multiplier", tooltip=["ts:T", "new_codes", "exit_total"])
    )

    dashboard = alt.vconcat(
        multi_chart + trigger_points,
        exit_chart,
        spacing=20,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard.save(str(output_path))
    print(f"Health panel saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
