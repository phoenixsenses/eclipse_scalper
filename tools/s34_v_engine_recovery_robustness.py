"""Recovery-confirmation robustness for S34 V Engine.

Sweeps confirmation timing and hold horizon around the frozen winner-extension
idea. Research-only; no live/paper state changes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_winner_extension import (
    DEFAULT_DB,
    SYMBOL,
    annotate,
    cell_str,
    collect_current_events,
    condition_pass,
    current_fill,
    exit_net,
    file_fingerprint,
    finite_float,
    load_mark_index,
    r1,
    utc_now,
)


OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_RECOVERY_ROBUSTNESS.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_RECOVERY_ROBUSTNESS.md"

CONFIRM_MINUTES = (15, 20, 30, 45, 60)
HORIZON_HR = (2.0, 3.0, 4.0, 5.0, 6.0)
CONDITIONS = (
    "all",
    "anchor_reclaimed",
    "btc_not_down_continues",
    "anchor_and_btc",
    "strong_rebound",
    "bull_reclaim",
)


def prefixed(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in row.items()}


def strip_prefix(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    needle = f"{prefix}_"
    return {k.removeprefix(needle): v for k, v in row.items() if k.startswith(needle)}


def build_rows(conn: sqlite3.Connection, *, max_book_staleness_sec: int) -> list[dict[str, Any]]:
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    rows: list[dict[str, Any]] = []
    for event in collect_current_events(conn):
        fill = current_fill(event)
        if not fill:
            continue
        fill_ts, entry_px, fill_leg = fill
        row: dict[str, Any] = {
            "event_id": event.anchor.event_id,
            "bucket": int(event.anchor.bucket),
            "anchor_ts_ms": int(event.anchor.anchor_ts_ms),
            "fill_ts_ms": int(fill_ts),
            "fill_leg": fill_leg,
            "entry_price": float(entry_px),
            "vdepth_bps": r1(event.vdepth_bps),
        }
        for minutes in CONFIRM_MINUTES:
            ann = annotate(event, fill_ts_ms=fill_ts, entry_px=entry_px, eth_marks=eth_marks, btc_marks=btc_marks, minutes=minutes)
            row.update(prefixed(f"m{minutes}", ann))
        for horizon in HORIZON_HR:
            net, exit_ts, source = exit_net(
                conn,
                event,
                fill_ts_ms=fill_ts,
                entry_px=entry_px,
                horizon_hr=horizon,
                max_book_staleness_sec=max_book_staleness_sec,
            )
            row[f"h{horizon:g}_net_bps"] = r1(net)
            row[f"h{horizon:g}_exit_ts_ms"] = int(exit_ts)
            row[f"h{horizon:g}_source"] = source
        rows.append(row)
    rows.sort(key=lambda r: int(r["anchor_ts_ms"]))
    return rows


def eval_cells(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    base_sum = float(baseline.get("sum_bps") or 0.0)
    base_t3r = float(baseline.get("top3_winner_removed_sum_bps") or 0.0)
    for minutes in CONFIRM_MINUTES:
        prefix = f"m{minutes}"
        for condition in CONDITIONS:
            subset = [r for r in rows if condition_pass(strip_prefix(r, prefix), condition)]
            for horizon in HORIZON_HR:
                key = f"h{horizon:g}_net_bps"
                vals = [float(v) for r in subset if (v := finite_float(r.get(key))) is not None]
                summary = summarize(vals)
                cells.append(
                    {
                        "confirm_min": int(minutes),
                        "condition": condition,
                        "horizon_hr": float(horizon),
                        "n": int(summary["n"]),
                        "summary": summary,
                        "delta_sum_bps": r1(float(summary.get("sum_bps") or 0.0) - base_sum),
                        "delta_t3r_bps": r1(float(summary.get("top3_winner_removed_sum_bps") or 0.0) - base_t3r),
                    }
                )
    cells.sort(
        key=lambda r: (
            float(r["delta_t3r_bps"] or -1e18),
            float(r["delta_sum_bps"] or -1e18),
            float(r["summary"].get("median_bps") or -1e18),
        ),
        reverse=True,
    )
    return cells


def build_report(conn: sqlite3.Connection, *, db_path: Path, max_book_staleness_sec: int) -> dict[str, Any]:
    rows = build_rows(conn, max_book_staleness_sec=max_book_staleness_sec)
    baseline = summarize([float(v) for r in rows if (v := finite_float(r.get("h2_net_bps"))) is not None])
    cells = eval_cells(rows, baseline)
    near = [
        r
        for r in cells
        if r["condition"] in {"anchor_and_btc", "btc_not_down_continues", "all"}
        and r["confirm_min"] in {20, 30, 45}
        and r["horizon_hr"] in {3.0, 4.0, 5.0}
        and float(r["delta_t3r_bps"] or 0.0) > 0.0
        and float(r["delta_sum_bps"] or 0.0) > 0.0
    ]
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "config": {
            "entry": "O20_W300_O5_C1",
            "confirm_minutes": list(CONFIRM_MINUTES),
            "horizon_hr": list(HORIZON_HR),
            "conditions": list(CONDITIONS),
        },
        "filled_rows": len(rows),
        "baseline_h2": baseline,
        "positive_neighborhood_cells": len(near),
        "cells": cells,
        "rows": rows,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Recovery Robustness",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. Sweeps confirmation timing and hold horizon; no live/paper state changed.",
        "",
        f"Filled rows: `{report['filled_rows']}`",
        "",
        f"Baseline H2: {cell_str(report['baseline_h2'])}",
        "",
        f"Positive neighborhood cells: `{report['positive_neighborhood_cells']}`",
        "",
        "| Rank | Confirm | Condition | Horizon | Summary | Delta sum | Delta T3R |",
        "| ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for i, row in enumerate(report["cells"][:50], start=1):
        lines.append(
            f"| {i} | {row['confirm_min']}m | `{row['condition']}` | {row['horizon_hr']}h | "
            f"{cell_str(row['summary'])} | {row['delta_sum_bps']} | {row['delta_t3r_bps']} |"
        )
    lines.extend(["", "## Read", ""])
    best = report["cells"][0] if report["cells"] else None
    if best:
        lines.append(
            f"- Best cell: {best['confirm_min']}m `{best['condition']}` {best['horizon_hr']}h -> {cell_str(best['summary'])}."
        )
    all_h4 = next((r for r in report["cells"] if r["condition"] == "all" and r["confirm_min"] == 30 and r["horizon_hr"] == 4.0), None)
    anchor_btc_h4 = next((r for r in report["cells"] if r["condition"] == "anchor_and_btc" and r["confirm_min"] == 30 and r["horizon_hr"] == 4.0), None)
    if all_h4:
        lines.append(f"- Simple all-H4: {cell_str(all_h4['summary'])}; delta T3R `{all_h4['delta_t3r_bps']}`.")
    if anchor_btc_h4:
        lines.append(f"- Frozen winner-extension cell: {cell_str(anchor_btc_h4['summary'])}; delta T3R `{anchor_btc_h4['delta_t3r_bps']}`.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S34 V Engine recovery robustness sweep.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(conn, db_path=args.db, max_book_staleness_sec=int(args.max_book_staleness_sec))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
