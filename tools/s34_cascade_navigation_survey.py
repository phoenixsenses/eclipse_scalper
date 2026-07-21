"""S34 Cascade Navigation Survey.

Batch companion to `s34_cascade_navigation_dashboard.py`. Where the dashboard is a
single point-in-time pilot screen, the survey walks every historical threshold-cross
anchor per lane and tallies the *knowable-at-cross* navigation state: cascade phase,
cluster shape, session, BTC alignment, and executability (book availability).

It answers "what do these cascades actually look like, and how often are they even
executable?" -- descriptive context for the permission layer. It deliberately does
NOT compute forward outcomes; edge lives in the holdout-validated route recheck.
All features come from the AnchorSnapshot (running values at the cross) plus
mark/book lookups at-or-before the anchor, so there is no lookahead.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    book_at,
    iso_ms,
    load_liquidations,
    load_mark_index,
    r1,
    r3,
    reconstruct_anchors,
)
from tools.research_s34_knowable_anchor_route_recheck import anchor_shape_label, day_trend_bps
from tools.s34_cascade_navigation_dashboard import THRESHOLDS_USD, UNIVERSE, Lane, session_label

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_CASCADE_NAVIGATION_SURVEY.json"
OUT_MD = OUT_DIR / "S34_CASCADE_NAVIGATION_SURVEY.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dist(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def survey_lane(
    conn: sqlite3.Connection,
    lane: Lane,
    btc_marks,
    *,
    bucket_sec: int,
    min_gap_sec: int,
    accel_window_sec: int,
    btc_window_sec: int,
    max_book_staleness_sec: int,
    start_ms: int | None,
    end_ms: int | None,
) -> dict[str, Any]:
    liqs = load_liquidations(conn, lane.symbol, lane.liq_side, start_ms, end_ms)
    marks = load_mark_index(conn, lane.symbol)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=bucket_sec,
        min_gap_sec=min_gap_sec,
        thresholds=THRESHOLDS_USD,
        accel_window_sec=accel_window_sec,
    )
    by_threshold: dict[float, list[dict[str, Any]]] = {th: [] for th in THRESHOLDS_USD}
    for anchor in anchors:
        th = float(anchor.threshold_usd)
        if th not in by_threshold:
            continue
        btc_ret = btc_marks.ret_bps(anchor.anchor_ts_ms - int(btc_window_sec) * 1000, anchor.anchor_ts_ms)
        aligned = None
        if btc_ret is not None:
            aligned = (btc_ret > 0) if lane.direction == "LONG" else (btc_ret < 0)
        book = book_at(conn, lane.symbol, anchor.anchor_ts_ms, max_book_staleness_sec)
        by_threshold[th].append(
            {
                "phase": "ACCELERATING" if anchor.acceleration_bucket == "accelerating" else "DECELERATING",
                "shape": anchor_shape_label(anchor),
                "session": session_label(anchor.anchor_ts_ms),
                "btc_aligned": aligned,
                "executable": book is not None,
                "elapsed_since_first_sec": float(anchor.elapsed_since_first_sec),
            }
        )

    thresholds_out: dict[str, Any] = {}
    for th, items in by_threshold.items():
        if not items:
            continue
        n = len(items)
        thresholds_out[f"{int(th/1000)}K"] = {
            "n": n,
            "phase": _dist(Counter(i["phase"] for i in items)),
            "shape": _dist(Counter(i["shape"] for i in items)),
            "session": _dist(Counter(i["session"] for i in items)),
            "btc_aligned_rate": r3(sum(1 for i in items if i["btc_aligned"]) / n),
            "executable_rate": r3(sum(1 for i in items if i["executable"]) / n),
            "median_elapsed_sec": r1(sorted(i["elapsed_since_first_sec"] for i in items)[n // 2]),
        }
    return {
        "lane": {"symbol": lane.symbol, "liq_side": lane.liq_side, "direction": lane.direction, "family": lane.family},
        "total_anchors": len(anchors),
        "thresholds": thresholds_out,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Cascade Navigation Survey",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Knowable-at-cross navigation state across all historical threshold-cross anchors. "
        "No forward outcomes (edge lives in the route recheck).",
        "",
        "| Lane | Thr | N | Accel% | TopShape | TopSession | BTC-Aligned | Executable | MedElapsed |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for lane in report["lanes"]:
        L = lane["lane"]
        for thr, d in lane["thresholds"].items():
            accel = d["phase"].get("ACCELERATING", 0)
            accel_pct = r1(accel / d["n"] * 100.0) if d["n"] else None
            top_shape = next(iter(d["shape"]), "-")
            top_session = next(iter(d["session"]), "-")
            lines.append(
                f"| {L['symbol']} {L['liq_side']} | {thr} | {d['n']} | {accel_pct} | {top_shape} | "
                f"{top_session} | {d['btc_aligned_rate']} | {d['executable_rate']} | {d['median_elapsed_sec']} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch survey of knowable cascade navigation state across historical anchors.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--btc-window-sec", type=int, default=900)
    p.add_argument("--max-book-staleness-sec", type=int, default=5)
    p.add_argument("--start-ms", type=int, default=None)
    p.add_argument("--end-ms", type=int, default=None)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        btc_marks = load_mark_index(conn, "BTCUSDT")
        lanes = [
            survey_lane(
                conn,
                lane,
                btc_marks,
                bucket_sec=int(args.bucket_sec),
                min_gap_sec=int(args.min_gap_sec),
                accel_window_sec=int(args.accel_window_sec),
                btc_window_sec=int(args.btc_window_sec),
                max_book_staleness_sec=int(args.max_book_staleness_sec),
                start_ms=args.start_ms,
                end_ms=args.end_ms,
            )
            for lane in UNIVERSE
        ]
    report = {"generated_at_utc": utc_now(), "lanes": lanes}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
