"""Robustness map for the S34 Stop-Tighten idea.

Sweeps the neighborhood around the frozen overlay:
delay {3,5,7,10} minutes, stop {60,80,100,120} bps, and related danger states.

Research-only. No live/paper state changes.
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

from tools.s34_v_engine_position_management import (
    CONFIG_ID,
    DEFAULT_CANCEL_REPLACE_JSON,
    DEFAULT_DB,
    condition_map,
    file_fingerprint,
    finite_float,
    load_json,
    load_mark_index,
    r1,
    summarize,
    summarize_rows,
    tighten_stop,
    utc_now,
)


OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_STOP_TIGHTEN_ROBUSTNESS.json"
OUT_MD = OUT_DIR / "S34_STOP_TIGHTEN_ROBUSTNESS.md"
SYMBOL = "ETHUSDT"


def build_report(conn: sqlite3.Connection, *, cancel_replace_path: Path, db_path: Path, max_book_staleness_sec: int) -> dict[str, Any]:
    payload = load_json(cancel_replace_path)
    base_rows = [
        r
        for r in payload.get("rows", [])
        if r.get("config_id") == CONFIG_ID and r.get("status") == "FILLED" and finite_float(r.get("net_bps")) is not None
    ]
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    conds = condition_map()
    baseline = summarize([float(r["net_bps"]) for r in base_rows])
    variants = []
    for delay in (3, 5, 7, 10):
        for condition in ("btc_down_continues", "no_reclaim_btc_down", "failed_v", "weak_first"):
            fn = conds[condition]
            for stop_ref in ("trigger", "entry"):
                for stop_bps in (60.0, 80.0, 100.0, 120.0):
                    rows = [
                        tighten_stop(
                            conn,
                            row,
                            delay_min=delay,
                            stop_bps=stop_bps,
                            condition=condition,
                            fn=fn,
                            stop_reference=stop_ref,
                            eth_marks=eth_marks,
                            btc_marks=btc_marks,
                            max_book_staleness_sec=max_book_staleness_sec,
                        )
                        for row in base_rows
                    ]
                    info = summarize_rows(rows)
                    info.update(
                        {
                            "variant": rows[0]["variant"] if rows else "",
                            "delay_min": delay,
                            "condition": condition,
                            "stop_reference": stop_ref,
                            "stop_bps": stop_bps,
                        }
                    )
                    variants.append(info)
    base_sum = float(baseline.get("sum_bps") or 0.0)
    base_t3r = float(baseline.get("top3_winner_removed_sum_bps") or 0.0)
    for row in variants:
        row["delta_sum_bps"] = r1(float(row["summary"].get("sum_bps") or 0.0) - base_sum)
        row["delta_t3r_bps"] = r1(float(row["summary"].get("top3_winner_removed_sum_bps") or 0.0) - base_t3r)
        row["delta_max_loss_bps"] = r1(float(row["summary"].get("max_loss_bps") or 0.0) - float(baseline.get("max_loss_bps") or 0.0))
    variants.sort(
        key=lambda r: (
            float(r["delta_t3r_bps"] or -1e18),
            float(r["delta_sum_bps"] or -1e18),
            float(r["delta_max_loss_bps"] or -1e18),
        ),
        reverse=True,
    )
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "config_id": CONFIG_ID,
        "baseline": baseline,
        "variants": variants,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']} max_loss={summary['max_loss_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Stop-Tighten Robustness",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Config: `{report['config_id']}`",
        "",
        f"Baseline: {cell(report['baseline'])}",
        "",
        "| Rank | Variant | Trigger | Managed | Delta sum | Delta T3R | Delta max loss |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(report["variants"][:40], start=1):
        lines.append(
            f"| {i} | `{row['variant']}` | {row['trigger_n']}/{row['n']} | {cell(row['summary'])} | "
            f"{row['delta_sum_bps']} | {row['delta_t3r_bps']} | {row['delta_max_loss_bps']} |"
        )
    positive = [r for r in report["variants"] if float(r["delta_t3r_bps"] or 0.0) > 0.0 and float(r["delta_sum_bps"] or 0.0) > 0.0]
    lines.extend(["", "## Read", ""])
    lines.append(f"- Positive neighborhood cells: `{len(positive)}` / `{len(report['variants'])}`.")
    if positive:
        delays = sorted({r["delay_min"] for r in positive})
        stops = sorted({int(r["stop_bps"]) for r in positive})
        conds = sorted({r["condition"] for r in positive})
        lines.append(f"- Positive delays: `{delays}`; stops: `{stops}`; conditions: `{conds}`.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S34 stop-tighten robustness map.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cancel-replace-json", type=Path, default=DEFAULT_CANCEL_REPLACE_JSON)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(conn, cancel_replace_path=args.cancel_replace_json, db_path=args.db, max_book_staleness_sec=int(args.max_book_staleness_sec))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
