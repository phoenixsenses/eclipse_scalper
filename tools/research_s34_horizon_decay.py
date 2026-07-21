"""S34 Horizon Decay.

Directly tests the hypothesis "the direction is right, we just exit too early".
For each route's knowable anchors it measures the PURE directional mark-to-mark
return (no TP/SL/BE, no fees) at a ladder of holding horizons from the tradeable
entry (mark at anchor_ts + entry_delay). Continuation direction = the route's
direction.

If the direction were right, gross median would be positive at some horizon and
holding longer would help. If gross is ~zero/negative at every horizon, the cross
carries no exploitable directional move after the (knowable) entry -- and longer
holds cannot rescue it. Net columns subtract the round-trip cost so you can see
where, if anywhere, an edge survives cost.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    signed_return_bps,
)
from tools.research_s34_knowable_anchor_route_recheck import ROUTES, route_filters_pass

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_HORIZON_DECAY.json"
OUT_MD = OUT_DIR / "S34_HORIZON_DECAY.md"

HORIZONS_SEC = (15, 30, 60, 120, 300, 600, 900, 1800, 3600)
DEFAULT_RULES = (
    "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30",
    "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
    "ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40",
    "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
    "BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
)
ROUTE_BY_NAME = {spec.rule_name: spec for spec in ROUTES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summ(vals: list[float], cost_bps: float) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "gross_median": None, "gross_mean": None, "win_rate": None, "net_median": None}
    return {
        "n": len(vals),
        "gross_median": r1(pctile(vals, 0.5)),
        "gross_mean": r1(mean(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "net_median": r1(pctile(vals, 0.5) - cost_bps),
    }


def decay_route(conn, spec, *, bucket_sec, min_gap_sec, accel_window_sec, cost_bps_rt) -> dict[str, Any]:
    liqs = load_liquidations(conn, spec.symbol, spec.liq_side, None, None)
    marks = load_mark_index(conn, spec.symbol)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    anchors = reconstruct_anchors(
        liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
        thresholds=(float(spec.threshold_usd),), accel_window_sec=accel_window_sec,
    )
    filtered = [a for a in anchors if route_filters_pass(spec, a, marks, btc_marks)[0]]
    per_h: dict[int, list[float]] = {h: [] for h in HORIZONS_SEC}
    for anchor in filtered:
        entry_ts = int(anchor.anchor_ts_ms) + int(spec.entry_delay_sec) * 1000
        entry = marks.at_or_after(entry_ts)
        if not entry:
            continue
        for h in HORIZONS_SEC:
            exit_mark = marks.at_or_after(entry_ts + h * 1000)
            if not exit_mark:
                continue
            per_h[h].append(signed_return_bps(spec.direction, float(entry[1]), float(exit_mark[1])))
    return {
        "rule_name": spec.rule_name,
        "family": spec.family,
        "direction": spec.direction,
        "filtered_anchor_n": len(filtered),
        "horizons": {str(h): summ(per_h[h], cost_bps_rt) for h in HORIZONS_SEC},
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Horizon Decay (pure directional, no TP/SL)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  round-trip cost: `{report['config']['cost_bps_rt']}` bps",
        "",
        "Signed mark-to-mark return from the tradeable entry (anchor + entry_delay) at each holding horizon. "
        "`gross` ignores cost; `net` subtracts round-trip cost from the gross median. Tests whether the "
        "direction is right and whether holding longer helps.",
        "",
    ]
    for route in report["routes"]:
        lines.append(f"## `{route['rule_name']}` ({route['direction']}, anchors={route['filtered_anchor_n']})")
        lines.append("")
        lines.append("| Horizon | N | Gross Med | Gross Mean | Win% | Net Med (after cost) |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for h in HORIZONS_SEC:
            s = route["horizons"][str(h)]
            wr = None if s["win_rate"] is None else r1(s["win_rate"] * 100.0)
            lines.append(f"| {h}s | {s['n']} | {s['gross_median']} | {s['gross_mean']} | {wr} | {s['net_median']} |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pure directional mark-return decay across holding horizons for S34 anchors.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--rules", default=",".join(DEFAULT_RULES))
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--cost-bps-rt", type=float, default=6.1)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    names = [n.strip() for n in str(args.rules).split(",") if n.strip()]
    specs = [ROUTE_BY_NAME[n] for n in names if n in ROUTE_BY_NAME]
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        routes = [
            decay_route(conn, spec, bucket_sec=int(args.bucket_sec), min_gap_sec=int(args.min_gap_sec),
                        accel_window_sec=int(args.accel_window_sec), cost_bps_rt=float(args.cost_bps_rt))
            for spec in specs
        ]
    report = {"generated_at_utc": utc_now(), "config": {"rules": names, "cost_bps_rt": float(args.cost_bps_rt)}, "routes": routes}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
